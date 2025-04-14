from pathlib import Path
import numpy as np
import sounddevice
from piper.voice import PiperVoice
from piper.download import get_voices, ensure_voice_exists
from bucky.voices.voice import Voice, voice_data_dir


class VoiceFast(Voice):
    '''
    Generate low latency speech with piper-tts: https://github.com/rhasspy/piper
    Voices: https://rhasspy.github.io/piper-samples/
    '''

    def __init__(self,
                 model: str = "en_US-joe-medium",
                 speaker_id: int | None = None,
                 audio_sink_factory=lambda rate, channels: sounddevice.OutputStream(samplerate=rate, channels=channels, dtype='int16')) -> None:
        model_path = Path(voice_data_dir, model + '.onnx')
        if not model_path.exists():
            voices_info = get_voices(voice_data_dir, update_voices=True)
            ensure_voice_exists(model, voice_data_dir, voice_data_dir, voices_info)
        self.speaker_id = speaker_id
        self.voice = PiperVoice.load(model_path)
        self.audio_sink_factory = audio_sink_factory

    def set_filler_phrases_enabled(self, enabled: bool) -> None:
        pass

    def speak(self, message: str) -> None:
        stream = self.audio_sink_factory(self.voice.config.sample_rate, 1)
        with stream:
            for audio_bytes in self.voice.synthesize_stream_raw(message, speaker_id=self.speaker_id):
                stream.write(np.frombuffer(audio_bytes, dtype=np.int16))
