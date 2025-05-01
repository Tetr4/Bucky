from speech_recognition import Recognizer, Microphone, AudioSource, AudioData, WaitTimeoutError
from typing import Callable, NamedTuple, Optional
from bucky.common.gpu_utils import get_free_cuda_device
from bucky.audio_source import BufferedAudioSourceWrapper, HttpAudioSource
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
        wav_output_dir: Optional[Path] = None,
        on_start_listening: Callable = lambda: None,
        on_stop_listening: Callable = lambda: None,
        on_waiting_for_wakeup: Callable = lambda: None,
        on_wakeup: Callable = lambda: None,
        on_unintelligible: Callable[[Transcription], None] = lambda _: None,
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
        self.on_unintelligible: Callable[[Transcription], None] = on_unintelligible
        self.has_user_attention: Callable[[], bool] = has_user_attention

        self.recognizer = Recognizer()
        # self.recognizer.dynamic_energy_threshold = False
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

        with BufferedAudioSourceWrapper(self.source_factory) as source:
            while True:
                if not self.wait_for_wake_word:
                    logger.info(f"{cli_bold_green}Listening...{cli_color_reset}")
                    if is_complex_wakeup_phrase(last_wakeup_phrase):
                        self.on_stop_listening()
                        return last_wakeup_phrase
                    else:
                        self.on_start_listening()

                    start = time.time()
                    source.flush_stream()
                    while True:
                        try:
                            timeout: Optional[float] = self.wakeword_timeout if self.wakewords else None
                            trans: Transcription = self.recognize(source,
                                                                  pause_threshold=2.0,
                                                                  phrase_time_limit=None,
                                                                  timeout=timeout)
                            if trans.phrase:
                                if trans.is_noise:
                                    self.on_unintelligible(trans)
                                    source.flush_stream()
                                else:
                                    self.on_stop_listening()
                                    return f"{last_wakeup_phrase} {trans.phrase}".strip()
                            if timeout and (time.time() - start) > timeout:
                                raise WaitTimeoutError()
                        except WaitTimeoutError:
                            if not self.has_user_attention():
                                break

                self.wait_for_wake_word = False

                if self.wakewords:
                    self.on_waiting_for_wakeup()

                    logger.info(f"{cli_bold_yellow}Waiting for Wakeword...{cli_color_reset}")
                    source.flush_stream()
                    while True:
                        transcription = self.recognize(source,
                                                       pause_threshold=1.0,
                                                       phrase_time_limit=10.0,
                                                       timeout=None)
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
                  timeout: Optional[float]) -> Transcription:
        self.recognizer.pause_threshold = pause_threshold
        # self.recognizer.dynamic_energy_threshold = False
        audio_frames: list[bytes] = []
        for audio_frame in self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit, stream=True):
            audio_frames.append(audio_frame.frame_data)
        frame_data = b"".join(audio_frames)
        audio: AudioData = AudioData(frame_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)

        result = self.recognizer.recognize_whisper(
            audio_data=audio, model=self.model, show_dict=True, load_options=None, language=self.language, translate=False,
            condition_on_previous_text=False
            # no_speech_threshold = None, logprob_threshold = None
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
                self.write_to_wav_file(phrase, audio)
            return Transcription(phrase=phrase, is_noise=is_noise, speech_prob=speech_prob)

        return Transcription(phrase="", is_noise=True, speech_prob=0.0)

    def write_to_wav_file(self, transcript: str, data: AudioData):
        if not self.wav_output_dir:
            return

        text = "".join(x for x in transcript if x.isalnum() or x in [' '])
        filename = self.wav_output_dir / Path(f"{int(time.time())}_{text}.wav")
        try:
            with wave.open(str(filename), 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(data.sample_width)
                wav_file.setframerate(data.sample_rate)
                wav_file.writeframesraw(data.frame_data)
        except Exception as e:
            logger.error(f"An error occurred: {e}")


def robot_mic():
    return HttpAudioSource(f"{cfg.bucky_uri}/mic", chunk_size=1024)


def local_mic():
    return Microphone()


if __name__ == "__main__":
    rec = Recorder(audio_source_factory=robot_mic)
    while True:
        print(rec.listen())
