import pickle
from speech_recognition import Recognizer, Microphone, AudioSource, AudioData, WaitTimeoutError
from typing import Callable, Generator, NamedTuple, Optional, assert_type, cast
from bucky.common.gpu_utils import get_free_cuda_device
from bucky.audio.filter import SpeechDenoiser
from bucky.audio.source import BufferedAudioSourceWrapper, HttpAudioSource
from pathlib import Path
import bucky.config as cfg
import whisper
import time
import wave
import logging

logger = logging.getLogger(__name__)
cli_grey = "\x1b[38;20m"
cli_red = "\x1b[31;20m"
cli_green = "\x1b[32;20m"
cli_bold_red = "\x1b[31;1m"
cli_bold_green = "\x1b[32;1m"
cli_bold_yellow = "\x1b[33;1m"
cli_bold_blue = "\x1b[34;1m"
cli_color_reset = "\x1b[0m"

Transcription = NamedTuple("Transcription", [("phrase", str), ("is_noise", bool), ("speech_prob", float)])


class Recorder:
    def __init__(
        self,
        wakewords: list[str] = [],
        wakeword_timeout: Optional[float] = 5.0,
        language: str = "english",
        model: str = "base.en",
        audio_source_factory: Callable[[], AudioSource] = Microphone,
        denoiser:  Optional[SpeechDenoiser] = None,
        wav_output_dir: Optional[Path] = None,
        on_start_listening: Callable = lambda: None,
        on_stop_listening: Callable = lambda: None,
        on_waiting_for_wakeup: Callable = lambda: None,
        on_wakeup: Callable = lambda: None,
        on_unintelligible: Callable[[Transcription], bool] = lambda _: False,
        has_user_attention: Callable[[], bool] = lambda: False,
    ) -> None:
        self.wakewords: list[str] = wakewords
        self.wakeword_timeout: Optional[float] = wakeword_timeout
        self.language: str = language
        self.model: str = model
        self.source_factory: Callable[[], AudioSource] = audio_source_factory
        self.wav_output_dir: Optional[Path] = wav_output_dir
        self.on_start_listening: Callable = on_start_listening
        self.on_stop_listening: Callable = on_stop_listening
        self.on_waiting_for_wakeup: Callable = on_waiting_for_wakeup
        self.on_wakeup: Callable = on_wakeup
        self.on_unintelligible: Callable[[Transcription], bool] = on_unintelligible
        self.has_user_attention: Callable[[], bool] = has_user_attention

        self.recognizer = Recognizer()
        self.wait_for_wake_word = True

        # get cuda device with 5GB free memory
        if cuda_device := get_free_cuda_device(5 * (1024**3)):
            logger.info(f"WHISPER: creating GPU instance {cuda_device}")
            torch_device = cuda_device.torch_device
        else:
            logger.info("WHISPER: creating CPU instance")
            torch_device = "cpu"

        # preload the model
        self.recognizer.whisper_model = {self.model: whisper.load_model(
            self.model, device=torch_device, in_memory=True)}

        self.denoiser = denoiser

    def listen(self) -> str:
        def contains_any_wakeword(phrase: str) -> bool:
            p = phrase.lower().replace(",", "")
            for wakeword in self.wakewords:
                if wakeword in p:
                    return True
            return False

        def is_complex_wakeup_phrase(phrase: str) -> bool:
            return len(phrase.split()) > 3 or phrase.endswith("?")

        last_wakeup_phrase: str = ""

        with BufferedAudioSourceWrapper(self.source_factory, self.denoiser) as source:
            while True:
                if not self.wait_for_wake_word:
                    if is_complex_wakeup_phrase(last_wakeup_phrase):
                        self.on_stop_listening()
                        return last_wakeup_phrase
                    else:
                        source.flush_stream()
                        print(f"{cli_bold_green}Listening...{cli_color_reset}")
                        self.on_start_listening()

                    start = time.time()
                    while True:
                        try:
                            phrase_start_timeout: Optional[float] = self.wakeword_timeout if self.wakewords else None
                            trans: Transcription = self.recognize(source,
                                                                  pause_threshold=1.5,
                                                                  phrase_time_limit=15.0,
                                                                  phrase_start_timeout=phrase_start_timeout)
                            if trans.phrase:
                                if trans.is_noise:
                                    if self.on_unintelligible(trans):
                                        start = time.time()  # reset timeout
                                    source.flush_stream()
                                else:
                                    self.on_stop_listening()
                                    return f"{last_wakeup_phrase} {trans.phrase}".strip()
                            if phrase_start_timeout and (time.time() - start) > phrase_start_timeout:
                                raise WaitTimeoutError()
                        except WaitTimeoutError:
                            if not self.has_user_attention():
                                break

                self.wait_for_wake_word = False

                if self.wakewords:
                    self.on_waiting_for_wakeup()

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
                            last_wakeup_phrase = transcription.phrase.strip()
                            self.on_wakeup()
                            break

    def stop_listening(self):
        self.wait_for_wake_word = True

    def recognize(self,
                  source: AudioSource,
                  pause_threshold: float,
                  phrase_time_limit: Optional[float],
                  phrase_start_timeout: Optional[float]) -> Transcription:
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.non_speaking_duration = 0.8
        chunks: list[AudioData] = []
        generator = self.recognizer.listen(source, timeout=phrase_start_timeout,
                                           phrase_time_limit=phrase_time_limit, stream=True)
        assert isinstance(generator, Generator)
        for audio_frame in generator:
            if not isinstance(audio_frame, AudioData):
                break
            chunks.append(audio_frame)

        frame_data = b"".join(chunk.frame_data for chunk in chunks)
        audio: AudioData = AudioData(frame_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)  # type: ignore

        result = self.recognizer.recognize_whisper(
            audio_data=audio, model=self.model, show_dict=True, load_options=None, language=self.language, translate=False,
            condition_on_previous_text=False
        )
        if phrase := result["text"].strip():
            no_speech_prob: float = result['segments'][0]["no_speech_prob"]
            speech_prob: float = 1.0 - max(0.0, min(1.0, (no_speech_prob - 5e-12) / (5e-10 - 5e-12)))
            is_noise: bool = speech_prob < 0.9

            phrase_color: str = cli_bold_yellow if is_noise else cli_bold_green
            speech_prob_color: str = cli_red if is_noise else cli_green
            logger.info(
                f"phrase: {phrase_color}{phrase}{cli_color_reset} {speech_prob_color}({speech_prob=:.2f}){cli_color_reset}")
            if isinstance(audio, AudioData):
                self.write_to_wav_file(phrase, audio, chunks)
            return Transcription(phrase=phrase, is_noise=is_noise, speech_prob=speech_prob)

        return Transcription(phrase="", is_noise=True, speech_prob=0.0)

    def write_to_wav_file(self, transcript: str, data: AudioData, chunks: list[AudioData]):
        if not self.wav_output_dir:
            return

        text = "".join(x for x in transcript if x.isalnum() or x in [' '])
        filename = str(self.wav_output_dir / Path(f"{int(time.time())}_{text}"))
        try:
            with wave.open(filename + ".wav", 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(data.sample_width)
                wav_file.setframerate(data.sample_rate)
                wav_file.writeframesraw(data.frame_data)
        except Exception as e:
            logger.error(f"An error occurred: {e}")

        try:
            with open(filename + ".pickle", "wb") as f:
                pickle.dump(chunks, f)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
