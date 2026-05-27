import base64
from dataclasses import dataclass, field
import pickle
import whisper
import time
import wave
import logging
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from speech_recognition import Recognizer, Microphone, AudioSource, AudioData, WaitTimeoutError
from typing import Callable, Generator, Optional
from bucky.common.gpu_utils import get_free_cuda_device
from bucky.audio.filter import EchoCancellation, SpeechDenoiser
from bucky.audio.source import BufferedAudioSourceWrapper
from pathlib import Path


logger = logging.getLogger(__name__)
cli_grey = "\x1b[38;20m"
cli_red = "\x1b[31;20m"
cli_green = "\x1b[32;20m"
cli_bold_red = "\x1b[31;1m"
cli_bold_green = "\x1b[32;1m"
cli_bold_yellow = "\x1b[33;1m"
cli_bold_blue = "\x1b[34;1m"
cli_color_reset = "\x1b[0m"


@dataclass(frozen=True, eq=False)
class AudioRecord:
    sample_rate: int
    sample_width: int
    chunks: list[AudioData] = field(default_factory=list)
    _cached_audio_data: Optional[AudioData] = field(init=False, default=None, repr=False, compare=False)

    @property
    def audio_data(self) -> AudioData:
        cached = getattr(self, '_cached_audio_data', None)
        if cached is None:
            frame_data = b"".join(chunk.frame_data for chunk in self.chunks)
            cached = AudioData(frame_data=frame_data,
                               sample_rate=self.sample_rate,
                               sample_width=self.sample_width)
            object.__setattr__(self, '_cached_audio_data', cached)
        return cached

    def create_message_content(self) -> list:
        wav_b64 = base64.b64encode(self.audio_data.get_wav_data()).decode("ascii")
        # Ollama currently expects audio to be passed as images.
        return [{"type": "image_url", "image_url": {"url": f"data:audio/wav;base64,{wav_b64}"}}]

    def __add__(self, other: 'AudioRecord') -> 'AudioRecord':
        assert self.sample_rate == other.sample_rate
        assert self.sample_width == other.sample_width
        return AudioRecord(sample_rate=self.sample_rate,
                           sample_width=self.sample_width,
                           chunks=self.chunks + other.chunks)

    def write_debug_files(self, dir_path: Path, debug_message: str):
        if not self.chunks:
            return

        frame_data = b"".join(chunk.frame_data for chunk in self.chunks)

        text = "".join(x for x in debug_message if x.isalnum() or x in [' '])
        filename = str(dir_path / Path(f"{int(time.time())}_{text}"))
        try:
            with wave.open(filename + ".wav", 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(self.sample_width)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframesraw(frame_data)
        except Exception as e:
            logger.error(f"An error occurred: {e}")

        try:
            with open(filename + ".pickle", "wb") as f:
                pickle.dump(self.chunks, f)
        except Exception as e:
            logger.error(f"An error occurred: {e}")


@dataclass(frozen=True, eq=False)
class Transcription:
    phrase: str
    is_noise: bool
    speech_prob: float
    record: AudioRecord

    def __add__(self, other: 'Transcription') -> 'Transcription':
        return Transcription(phrase=f"{self.phrase} {other.phrase}".strip(),
                             is_noise=self.is_noise or other.is_noise,
                             speech_prob=min(self.speech_prob, other.speech_prob),
                             record=self.record + other.record)

    def write_debug_files(self, dir_path: Path):
        self.record.write_debug_files(dir_path, self.phrase)


class Recorder:
    def __init__(
        self,
        wakewords: list[str] = [],
        wakeword_timeout: Optional[float] = 5.0,
        language: str = "english",
        model: str = "base.en",
        audio_source_factory: Callable[[], AudioSource] = Microphone,
        echo_cancellation: Optional[EchoCancellation] = None,
        denoiser:  Optional[SpeechDenoiser] = None,
        wav_output_dir: Optional[Path] = None,
        on_start_listening: Callable = lambda: None,
        on_stop_listening: Callable = lambda: None,
        on_waiting_for_wakeup: Callable = lambda: None,
        on_wakeup: Callable = lambda: None,
        on_unintelligible: Callable[[Transcription], bool] = lambda _: False,
        has_user_attention: Callable[[], bool] = lambda: False,
        transcription_llm: Optional[BaseChatModel] = None,
    ) -> None:
        self._wakewords: list[str] = wakewords
        self._wakeword_timeout: Optional[float] = wakeword_timeout
        self._language: str = language
        self._model: str = model
        self._source_factory: Callable[[], AudioSource] = audio_source_factory
        self._echo_cancellation = echo_cancellation
        self._denoiser = denoiser
        self._wav_output_dir: Optional[Path] = wav_output_dir
        self._on_start_listening: Callable = on_start_listening
        self._on_stop_listening: Callable = on_stop_listening
        self._on_waiting_for_wakeup: Callable = on_waiting_for_wakeup
        self._on_wakeup: Callable = on_wakeup
        self._on_unintelligible: Callable[[Transcription], bool] = on_unintelligible
        self._has_user_attention: Callable[[], bool] = has_user_attention
        self._transcription_llm = transcription_llm

        self._recognizer = Recognizer()
        self._wait_for_wake_word = True
        self._muted = False

        if self._transcription_llm is None:
            # get cuda device with 5GB free memory
            if cuda_device := get_free_cuda_device(5 * (1024**3)):
                logger.info(f"WHISPER: creating GPU instance {cuda_device}")
                torch_device = cuda_device.torch_device
            else:
                logger.info("WHISPER: creating CPU instance")
                torch_device = "cpu"

            # preload the model
            self._recognizer.whisper_model = {self._model: whisper.load_model(self._model,
                                                                              device=torch_device,
                                                                              in_memory=True)}

    def set_muted(self, muted: bool):
        self._muted = muted

    def listen(self) -> Transcription:
        def contains_any_wakeword(phrase: str) -> bool:
            p = phrase.lower().replace(",", "")
            for wakeword in self._wakewords:
                if wakeword in p:
                    return True
            return False

        def is_complex_wakeup_phrase(phrase: str) -> bool:
            return len(phrase.split()) > 3 or phrase.endswith("?")

        last_wakeup: Optional[Transcription] = None

        with BufferedAudioSourceWrapper(audio_source_factory=self._source_factory,
                                        echo_cancellation=self._echo_cancellation,
                                        denoiser=self._denoiser) as source:
            while True:
                if not self._wait_for_wake_word:
                    if last_wakeup and is_complex_wakeup_phrase(last_wakeup.phrase):
                        self._on_stop_listening()
                        return last_wakeup
                    else:
                        source.flush_stream()
                        print(f"{cli_bold_green}Listening...{cli_color_reset}")
                        self._on_start_listening()

                    start = time.time()
                    while True:
                        try:
                            phrase_start_timeout: Optional[float] = self._wakeword_timeout if self._wakewords else None
                            trans: Transcription = self.recognize(source,
                                                                  pause_threshold=1.5,
                                                                  phrase_time_limit=15.0,
                                                                  phrase_start_timeout=phrase_start_timeout)
                            if trans.phrase:
                                if trans.is_noise:
                                    if self._on_unintelligible(trans):
                                        start = time.time()  # reset timeout
                                    source.flush_stream()
                                else:
                                    self._on_stop_listening()
                                    if last_wakeup:
                                        return last_wakeup + trans
                                    return trans
                            if phrase_start_timeout and (time.time() - start) > phrase_start_timeout:
                                raise WaitTimeoutError()
                        except WaitTimeoutError:
                            if not self._has_user_attention():
                                break

                self._wait_for_wake_word = False

                if self._wakewords:
                    self._on_waiting_for_wakeup()

                    print(f"{cli_bold_yellow}Waiting for Wakeword...{cli_color_reset}")
                    source.flush_stream()
                    while True:
                        transcription = self.recognize(source,
                                                       pause_threshold=1.0,
                                                       phrase_time_limit=10.0,
                                                       phrase_start_timeout=None)
                        if not transcription.phrase:
                            continue

                        if contains_any_wakeword(transcription.phrase):
                            last_wakeup = transcription
                            self._on_wakeup()
                            break

    def reset(self):
        self._wait_for_wake_word = True

    def recognize(self,
                  source: AudioSource,
                  pause_threshold: float,
                  phrase_time_limit: Optional[float],
                  phrase_start_timeout: Optional[float]) -> Transcription:
        self._recognizer.pause_threshold = pause_threshold
        self._recognizer.non_speaking_duration = 0.8
        chunks: list[AudioData] = []
        generator = self._recognizer.listen(source, timeout=phrase_start_timeout,
                                            phrase_time_limit=phrase_time_limit, stream=True)
        assert isinstance(generator, Generator)
        for audio_frame in generator:
            if self._muted:
                chunks.clear()
                break
            if not isinstance(audio_frame, AudioData):
                break
            chunks.append(audio_frame)

        record = AudioRecord(sample_rate=source.SAMPLE_RATE,  # type: ignore
                             sample_width=source.SAMPLE_WIDTH,  # type: ignore
                             chunks=chunks)

        if chunks:
            if self._transcription_llm:
                trans = self.recognize_llm(record)
            else:
                trans = self.recognize_whisper(record)

            if trans and self._wav_output_dir:
                trans.write_debug_files(self._wav_output_dir)
        else:
            trans = None

        return trans or Transcription(phrase="", is_noise=True, speech_prob=0.0, record=record)

    def recognize_llm(self, record: AudioRecord) -> Optional[Transcription]:
        system_prompt: str = """Your name is Bucky. You are a professional transcriber."""
        prompt: str = """Transcribe this german speech segment. Only output the transcription."""
        # Ollama currently expects audio to be passed as images.
        content = record.create_message_content()
        content.append({"type": "text", "text": prompt})

        input: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=content)]
        response: BaseMessage = self._transcription_llm.invoke(input)  # type: ignore
        if response.content and isinstance(response.content, str):
            logger.info(f"phrase: {cli_bold_yellow}{response.content}{cli_color_reset}")
            return Transcription(phrase=response.content,
                                 is_noise=False,
                                 speech_prob=1.0,
                                 record=record)
        return None

    def recognize_whisper(self, record: AudioRecord) -> Optional[Transcription]:
        result = self._recognizer.recognize_whisper(
            audio_data=record.audio_data,
            model=self._model,
            show_dict=True,
            load_options=None,
            language=self._language,
            translate=False,
            condition_on_previous_text=False
        )

        phrase: str = result["text"].strip()
        if not phrase:
            return None

        no_speech_prob: float = result['segments'][0]["no_speech_prob"]
        speech_prob: float = 1.0 - max(0.0, min(1.0, (no_speech_prob - 5e-12) / (5e-10 - 5e-12)))
        is_noise: bool = speech_prob < 0.9

        phrase_color: str = cli_bold_yellow if is_noise else cli_bold_green
        prob_color: str = cli_red if is_noise else cli_green
        logger.info(
            f"phrase: {phrase_color}{phrase}{cli_color_reset} {prob_color}({speech_prob=:.2f}){cli_color_reset}")
        return Transcription(phrase=phrase,
                             is_noise=is_noise,
                             speech_prob=speech_prob,
                             record=record)
