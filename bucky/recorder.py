from speech_recognition import Recognizer, Microphone, AudioSource, WaitTimeoutError
from typing import Callable
from bucky.audio_source import HttpAudioSource
import bucky.config as cfg
import whisper
import time


class Recorder:
    def __init__(
        self,
        wakewords: list[str] | None = None,
        language: str = "english",
        model: str = "base.en",
        audio_source_factory: Callable[[], AudioSource] = Microphone,
        on_start_listening: Callable = None,
        on_stop_listening: Callable = None,
        on_wakeword_detected: Callable = None,
    ) -> None:
        self.wakewords = wakewords
        self.language = language
        self.source_factory = audio_source_factory
        self.recognizer = Recognizer()
        self.recognizer.pause_threshold = 2
        self.recognizer.dynamic_energy_threshold = False
        self.model = model
        self.on_start_listening = on_start_listening
        self.on_stop_listening = on_stop_listening
        self.on_wakeword_detected = on_wakeword_detected
        self.first_listen = True

        # preload the model
        self.recognizer.whisper_model = {self.model: whisper.load_model(self.model)}

    def listen(self) -> str:
        def contains_any_wakeword(phrase: str):
            p = phrase.lower().replace(",", "")
            for wakeword in self.wakewords:
                if wakeword in p:
                    return True
            return False

        listen_start = time.time()

        while True:
            if not self.first_listen:
                print("Listening...")

                if self.on_start_listening:
                    self.on_start_listening()

                source = self.source_factory()
                with source:
                    while True:
                        transcription = None
                        try:
                            audio = self.recognizer.listen(source, timeout=5)
                            transcription = self.recognizer.recognize_whisper(
                                audio, model=self.model, language=self.language
                            )
                            transcription = transcription.strip()
                        except WaitTimeoutError:
                            print("WaitTimeoutError")

                        if transcription:
                            return transcription
                        elif (time.time() - listen_start) > 5.0:
                            break

            if self.on_stop_listening:
                self.on_stop_listening()

            self.first_listen = False

            if self.wakewords:
                print("Waiting for Wakeword...")
                source = self.source_factory()
                with source:
                    while True:
                        audio = self.recognizer.listen(source)
                        phrase = self.recognizer.recognize_whisper(
                            audio, model=self.model, language=self.language
                        )
                        if not phrase:
                            continue

                        print("Phrase:", phrase)
                        if contains_any_wakeword(phrase):
                            break

                if self.on_wakeword_detected:
                    self.on_wakeword_detected()


def robot_mic():
    return HttpAudioSource(f"{cfg.bucky_uri}/mic", chunk_size=1024)


def local_mic():
    return Microphone()


if __name__ == "__main__":
    rec = Recorder(audio_source_factory=robot_mic)
    while True:
        print(rec.listen())
