from speech_recognition import Recognizer, Microphone, AudioSource, AudioData, WaitTimeoutError
from typing import Callable, NamedTuple, Optional
from bucky.common.gpu_utils import get_free_cuda_device
from bucky.audio_source import HttpAudioSource
from pathlib import Path
import bucky.config as cfg
import whisper
import time
import wave
import logging

logger = logging.getLogger(__name__)

Transcription = NamedTuple("Transcription", [('is_noise', bool), ('phrase', str)])


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
        on_wakeup: Callable[[bool], None] = lambda simple_wakeup: None,
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
        self.has_user_attention: Callable[[], bool] = has_user_attention

        self.recognizer = Recognizer()
        # self.recognizer.dynamic_energy_threshold = False
        self.wait_for_wake_word = True

        # try to use the GPU
        torch_device = "cpu"

        # get cuda device with 5GB free memory
        if cuda_device := get_free_cuda_device(5 * (1024**3)):
            logger.info(f"WHISPER: creating GPU instance {cuda_device}")
            torch_device = cuda_device.torch_device
        else:
            logger.info("WHISPER: creating CPU instance")

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

        while True:
            if not self.wait_for_wake_word:
                print("Listening...")
                if is_complex_wakeup_phrase(last_wakeup_phrase):
                    self.on_stop_listening()
                    return last_wakeup_phrase
                else:
                    self.on_start_listening()

                self.recognizer.pause_threshold = 2
                source = self.source_factory()
                with source:
                    start = time.time()
                    while True:
                        try:
                            timeout: Optional[float] = self.wakeword_timeout if self.wakewords else None
                            if phrase := self.recognize(source, ignore_noise=True, timeout=timeout).phrase:
                                self.on_stop_listening()
                                return f"{last_wakeup_phrase} {phrase}".strip()
                            if timeout and (time.time() - start) > timeout:
                                raise WaitTimeoutError()
                        except WaitTimeoutError:
                            if not self.has_user_attention():
                                break

            self.wait_for_wake_word = False

            if self.wakewords:
                self.on_waiting_for_wakeup()

                print("Waiting for Wakeword...")
                self.recognizer.pause_threshold = 1.0
                source = self.source_factory()
                with source:
                    while True:
                        transcription = self.recognize(source, ignore_noise=False)
                        if not transcription.phrase:
                            continue

                        if contains_any_wakeword(transcription.phrase) or (not transcription.is_noise and self.has_user_attention()):
                            last_wakeup_phrase = transcription.phrase.strip()
                            self.on_wakeup(not is_complex_wakeup_phrase(last_wakeup_phrase))
                            break

    def stop_listening(self):
        self.wait_for_wake_word = True

    def recognize(self, source: AudioSource, ignore_noise: bool = False, timeout: Optional[float] = None) -> Transcription:
        audio = self.recognizer.listen(source, timeout=timeout)
        result = self.recognizer.recognize_whisper(
            audio, model=self.model, language=self.language, show_dict=True
        )
        if phrase := result["text"].strip():
            ignored = False
            no_speech_prob = result['segments'][0]["no_speech_prob"]
            no_speech_prob_threshold: float = 5e-11
            is_noise = no_speech_prob > no_speech_prob_threshold
            if ignore_noise:
                ignored = is_noise
            logger.info(f"phrase: {phrase}, ({no_speech_prob=}, {ignored=})")
            if not ignored and isinstance(audio, AudioData):
                self.write_to_wav_file(phrase, audio)
                return Transcription(is_noise, phrase)

        return Transcription(True, "")

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
