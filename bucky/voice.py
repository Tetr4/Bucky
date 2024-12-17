from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import sounddevice
from TTS.api import TTS
from piper.voice import PiperVoice
from piper.download import get_voices, ensure_voice_exists

data_dir = "voice-data"

class Voice(ABC):

    @abstractmethod
    def speak(self, message: str) -> None:
        pass


class VoiceFast(Voice):
    '''
    Generate low latency speech with piper-tts: https://github.com/rhasspy/piper
    Voices: https://rhasspy.github.io/piper-samples/
    '''
    def __init__(self, model: str = "en_US-joe-medium") -> None:
        model_path = Path(data_dir, model + '.onnx')
        if not model_path.exists():
            voices_info = get_voices(data_dir, update_voices=True)
            ensure_voice_exists(model, data_dir, data_dir, voices_info)
        self.voice = PiperVoice.load(model_path)

    def speak(self, message: str) -> None:
        with sounddevice.OutputStream(samplerate=self.voice.config.sample_rate, channels=1, dtype='int16') as stream:
            for audio_bytes in self.voice.synthesize_stream_raw(message):
                int_data = np.frombuffer(audio_bytes, dtype=np.int16)
                stream.write(int_data)


class VoiceQuality(Voice):
    '''
    Generate high quality speech with TTS: https://github.com/coqui-ai/TTS
    Voices: pdm run tts --list_models
    '''
    def __init__(self, model: str = "tts_models/multilingual/multi-dataset/xtts_v2") -> None:
        # Note XTTS is not for commercial use: https://coqui.ai/cpml
       self.tts = TTS(model_name=model, progress_bar=False)

    def speak(self, message: str) -> None:
        voice_template = Path(data_dir, "voice_template.wav")
        wave = self.tts.tts(text=message, language="en", speaker_wav=str(voice_template))
        sounddevice.play(wave, samplerate=22050, blocking=True)


if __name__ == "__main__":
    voice = VoiceFast()
    voice.speak("Howdy!, I am Bucky, the friendly cowboy assistant! Yeehaw!")
