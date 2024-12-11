from enum import Enum
import os
from bark import SAMPLE_RATE, generate_audio, preload_models
import pyttsx3
import sounddevice

class VoiceMode(Enum):
    TTS_FAST = "tts_fast"
    TTS_REALISTIC = "tts_quality"

class Voice:
    def __init__(self, mode: VoiceMode) -> None:
        self.mode = mode
        pass

    def speak(self, message: str) -> None:
        match self.mode:
            case VoiceMode.TTS_FAST:
                engine = pyttsx3.init()
                engine.say(message)
                engine.runAndWait()
            case VoiceMode.TTS_REALISTIC:
                # TODO Long from generation: https://github.com/suno-ai/bark/blob/main/notebooks/long_form_generation.ipynb
                os.environ["SUNO_USE_SMALL_MODELS"] = "True"
                audio_array = generate_audio(message, history_prompt="v2/en_speaker_6")
                sounddevice.play(audio_array, SAMPLE_RATE, blocking=True)
