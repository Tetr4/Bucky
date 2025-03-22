from dataclasses import dataclass
import threading
import numpy as np
import sounddevice
import wave
import os

effects_dir = "assets/sound-effects"

@dataclass
class WaveInfo:
    samples: bytes
    rate: int
    channels: int


class FxPlayer:
    def __init__(self,audio_sink_factory = lambda rate, channels: sounddevice.OutputStream(samplerate=rate, channels=channels, dtype='int16')) -> None:
        self.audio_sink_factory = audio_sink_factory
        self._rising_chime = self._load_wave_file(os.path.join(effects_dir, "chime-rising.wav"))
        self._descending_chime = self._load_wave_file(os.path.join(effects_dir, "chime-descending.wav"))

    def play_rising_chime(self):
        self._play(self._rising_chime)
    
    def play_descending_chime(self):
        self._play(self._descending_chime)


    def _load_wave_file(self, file_path: str) -> WaveInfo:
        with wave.open(file_path, "rb") as wf:
            return WaveInfo(samples=wf.readframes(wf.getnframes()), rate=wf.getframerate(), channels=wf.getnchannels())

    def _play(self, info: WaveInfo) -> None:
        def func():
            stream = self.audio_sink_factory(info.rate, info.channels)
            with stream:
                stream.write(np.frombuffer(info.samples, dtype=np.int16))
        player_thread = threading.Thread(target=func, daemon=True)
        player_thread.start()