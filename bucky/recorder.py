from speech_recognition import Recognizer, Microphone

class Recorder:
    def __init__(self, language:str = "english") -> None:
       self.language = language

    def listen(self) -> str:
        print("Listening...")
        recognizer = Recognizer()
        recognizer.pause_threshold = 2
        transcription = None
        while not transcription:
            with Microphone() as source:
                audio = recognizer.listen(source)
                transcription = recognizer.recognize_whisper(audio, language=self.language)
                transcription = transcription.strip()
        return transcription