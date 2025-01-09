from speech_recognition import Recognizer, Microphone, AudioSource
from typing import Callable
from bucky.audio_source import HttpAudioSource
import bucky.config as cfg
import whisper

class Recorder:
    def __init__(self, language:str = "english", model:str = "base.en", audio_source_factory: Callable[[], AudioSource] = Microphone) -> None:
       self.language = language
       self.source_factory = audio_source_factory
       self.recognizer = Recognizer()
       self.recognizer.pause_threshold = 3
       self.model = model
       # preload the model
       self.recognizer.whisper_model = {self.model : whisper.load_model(self.model)}

    def listen(self) -> str:
        print("Listening...")
        source = self.source_factory()
        with source:
            while True:
                audio = self.recognizer.listen(source)
                transcription = self.recognizer.recognize_whisper(audio, model=self.model, language=self.language)
                transcription = transcription.strip()
                if transcription:
                    return transcription

def robot_mic():
    return HttpAudioSource(f"{cfg.bucky_uri}/mic", chunk_size=1024)

def local_mic():
    return Microphone()

if __name__ == "__main__":
    rec = Recorder(audio_source_factory = robot_mic)
    while True:
        print(rec.listen())