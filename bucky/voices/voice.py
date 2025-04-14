from abc import ABC, abstractmethod

voice_data_dir = "assets/voice-data"


class Voice(ABC):

    @abstractmethod
    def set_filler_phrases_enabled(self, enabled: bool) -> None:
        pass

    @abstractmethod
    def speak(self, message: str) -> None:
        pass
