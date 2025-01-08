from speech_recognition import Recognizer, Microphone, AudioSource
from typing import Callable
from bucky.audio_source import HttpAudioSource
import bucky.config as cfg

class Recorder:
    def __init__(self, language:str = "english", audio_source_factory: Callable[[], AudioSource] = Microphone) -> None:
       self.language = language
       self.source_factory = audio_source_factory

    def listen(self) -> str:
        print("Listening...")
        recognizer = Recognizer()
        recognizer.pause_threshold = 2
        transcription = None
        while not transcription:
            source = self.source_factory()
            with source:
                audio = recognizer.listen(source)
                transcription = recognizer.recognize_whisper(audio, language=self.language)
                transcription = transcription.strip()
        return transcription

def robot_mic():
    return HttpAudioSource(f"{cfg.bucky_uri}/mic")

def local_mic():
    return Microphone()

if __name__ == "__main__":
    rec = Recorder(audio_source_factory = robot_mic)
    while True:
        print(rec.listen())