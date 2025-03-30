from speech_recognition import Recognizer, Microphone, AudioSource, AudioData, WaitTimeoutError
from typing import Callable, Optional
from bucky.audio_source import HttpAudioSource
from pathlib import Path
import bucky.config as cfg
import whisper
import time
import wave


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
        on_waiting_for_wakewords: Callable = lambda: None,
        on_wakeword_detected: Callable = lambda: None,
    ) -> None:

        self.wakewords: list[str] = wakewords
        self.wakeword_timeout: Optional[float] = wakeword_timeout
        self.language: str = language
        self.model: str = model
        self.source_factory: Callable[[], AudioSource] = audio_source_factory
        self.wav_output_dir: Optional[Path] = wav_output_dir
        self.on_start_listening: Callable = on_start_listening
        self.on_stop_listening: Callable = on_stop_listening
        self.on_waiting_for_wakewords: Callable = on_waiting_for_wakewords
        self.on_wakeword_detected: Callable = on_wakeword_detected

        self.recognizer = Recognizer()
        # self.recognizer.dynamic_energy_threshold = False
        self.wait_for_wake_word = True

        # preload the model
        self.recognizer.whisper_model = {self.model: whisper.load_model(self.model)}

    def listen(self) -> str:
        def contains_any_wakeword(phrase: str) -> bool:
            p = phrase.lower().replace(",", "")
            for wakeword in self.wakewords:
                if wakeword in p:
                    return True
            return False

        last_wakeword_phrase: str = ""

        while True:
            if not self.wait_for_wake_word:
                print("Listening...")
                if len(last_wakeword_phrase.split()) > 3:
                    self.on_stop_listening()
                    return last_wakeword_phrase
                else:
                    self.on_start_listening()

                self.recognizer.pause_threshold = 2
                source = self.source_factory()
                with source:
                    start = time.time()
                    while True:
                        transcription = None
                        try:
                            timeout: Optional[float] = self.wakeword_timeout if self.wakewords else None
                            if transcription := self.recognize(source, ignore_garbage=True, timeout=timeout):
                                self.on_stop_listening()
                                return f"{last_wakeword_phrase} {transcription}".strip()
                            if timeout and (time.time() - start) > timeout:
                                break
                        except WaitTimeoutError:
                            break

            self.wait_for_wake_word = False

            if self.wakewords:
                self.on_waiting_for_wakewords()

                print("Waiting for Wakeword...")
                self.recognizer.pause_threshold = 1.0
                source = self.source_factory()
                with source:
                    while True:
                        phrase = self.recognize(source)
                        if not phrase:
                            continue
                        if contains_any_wakeword(phrase):
                            last_wakeword_phrase = phrase.strip()
                            break

                self.on_wakeword_detected()

    def stop_listening(self):
        self.wait_for_wake_word = True

    def recognize(self, source: AudioSource, ignore_garbage: bool = False, timeout: Optional[float] = None) -> str:
        audio = self.recognizer.listen(source, timeout=timeout)
        result = self.recognizer.recognize_whisper(
            audio, model=self.model, language=self.language, show_dict=True
        )
        if phrase := result["text"].strip():
            ignored = False
            no_speech_prob = result['segments'][0]["no_speech_prob"]
            if ignore_garbage:
                no_speech_prob_threshold: float = 5e-11
                ignored = no_speech_prob > no_speech_prob_threshold
            print("phrase:", phrase, f"({no_speech_prob=}, {ignored=})")
            if not ignored and isinstance(audio, AudioData):
                self.write_to_wav_file(phrase, audio)
                return phrase

        return ""

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
            print(f"An error occurred: {e}")


def robot_mic():
    return HttpAudioSource(f"{cfg.bucky_uri}/mic", chunk_size=1024)


def local_mic():
    return Microphone()


if __name__ == "__main__":
    rec = Recorder(audio_source_factory=robot_mic)
    while True:
        print(rec.listen())
