from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import sounddevice
import torch
import threading
import queue
import random
import time
from bucky.gpu_utils import get_free_cuda_device
from TTS.api import TTS
from piper.voice import PiperVoice
from piper.download import get_voices, ensure_voice_exists
from bucky.audio_sink import HttpAudioSink
import bucky.config as cfg

data_dir = "assets/voice-data"


class Voice(ABC):

    @abstractmethod
    def set_filler_phrases_enabled(self, enabled: bool) -> None:
        pass

    @abstractmethod
    def speak(self, message: str, cache: bool = False) -> None:
        pass


class VoiceFast(Voice):
    '''
    Generate low latency speech with piper-tts: https://github.com/rhasspy/piper
    Voices: https://rhasspy.github.io/piper-samples/
    '''

    def __init__(self,
                 model: str = "en_US-joe-medium",
                 speaker_id: int | None = None,
                 audio_sink_factory=lambda rate, channels: sounddevice.OutputStream(samplerate=rate, channels=channels, dtype='int16')) -> None:
        model_path = Path(data_dir, model + '.onnx')
        if not model_path.exists():
            voices_info = get_voices(data_dir, update_voices=True)
            ensure_voice_exists(model, data_dir, data_dir, voices_info)
        self.speaker_id = speaker_id
        self.voice = PiperVoice.load(model_path)
        self.audio_sink_factory = audio_sink_factory

    def set_filler_phrases_enabled(self, enabled: bool) -> None:
        pass

    def speak(self, message: str, cache: bool = False) -> None:
        stream = self.audio_sink_factory(self.voice.config.sample_rate, 1)
        with stream:
            for audio_bytes in self.voice.synthesize_stream_raw(message, speaker_id=self.speaker_id):
                stream.write(np.frombuffer(audio_bytes, dtype=np.int16))


class VoiceQuality(Voice):
    '''
    Generate high quality speech with TTS: https://github.com/coqui-ai/TTS
    Voices: pdm run tts --list_models
    '''

    def __init__(self,
                 model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
                 language: str = "en",
                 voice_template: Path = Path(data_dir, "voice_template.wav"),
                 filler_phrases: list[tuple[str, float]] = [("hm", 3), ("jo", 3), ("ähm", 3), ("also", 4)],
                 pre_cached_phrases: list[str] = [],
                 audio_sink_factory=lambda rate, channels: sounddevice.OutputStream(samplerate=rate, channels=channels, dtype='int16')) -> None:
        # Note XTTS is not for commercial use: https://coqui.ai/cpml
        self.tts = TTS(model_name=model, progress_bar=False)

        if cuda_device := get_free_cuda_device():
            print("TTS: switching to CUDA", cuda_device)
            self.tts.to(cuda_device.torch_device, dtype=torch.float, non_blocking=True)

        self.language = language
        self.voice_template = voice_template
        self.audio_sink_factory = audio_sink_factory

        self.cached_sounds = {}

        self.filler_sounds_enabled = False
        self.filler_sounds = []

        self.realtime_factor = 0.9  # faster machine -> smaller value

        # model warm-up
        self.text_to_speech("The quick brown fox jumps over the lazy dog.")

        for _ in range(3):
            for phrase, max_duration in filler_phrases:
                waves = self.text_to_speech(phrase)
                duration = self._get_audio_duration(waves)
                if duration <= max_duration:
                    self.filler_sounds.append(waves)
                else:
                    print(f"skipping: '{phrase}' {duration=}")
        random.shuffle(self.filler_sounds)

        for phrase in pre_cached_phrases:
            shortest_wave: list = []
            for _ in range(3):
                if wave := self.text_to_speech(phrase):
                    if not shortest_wave or len(shortest_wave) > len(wave):
                        shortest_wave = wave
            if shortest_wave:
                self.cached_sounds[phrase] = shortest_wave

        self.wave_queue = queue.Queue()

        def play_sound_proc():
            while True:
                try:
                    wave = self.wave_queue.get(timeout=2.0 if self.filler_sounds else None)
                    try:
                        self._play_audio(wave)
                    finally:
                        self.wave_queue.task_done()
                except queue.Empty:
                    if self.filler_sounds_enabled and self.filler_sounds:
                        self._play_audio(random.choice(self.filler_sounds))

        self.player_thread = threading.Thread(target=play_sound_proc, daemon=True)
        self.player_thread.start()

    def text_to_speech(self, message: str, cache: bool = False) -> list:
        if message in self.cached_sounds:
            return self.cached_sounds[message]

        start_time = time.time()

        waves = self.tts.tts(
            text=message,
            language=self.language,
            speaker_wav=str(self.voice_template),
            split_sentences=False,
        )

        if cache:
            self.cached_sounds[message] = waves

        process_time = time.time() - start_time
        audio_time = self._get_audio_duration(waves)
        self.realtime_factor = 0.5 * self.realtime_factor + 0.5 * process_time / audio_time

        return waves

    def split_into_text_sections(self, message: str) -> list[str]:
        sentences = self.tts.synthesizer.split_into_sentences(message)  # type: ignore
        text_sections = []
        max_character_limit: int = 253
        next_character_limit: int = 0
        while sentences:
            txt = sentences.pop(0)
            if not text_sections:
                # To reduce latency, the first text section should be as short as possible.
                # This can be accomplished by splitting the first sentence at the first comma.
                # If no comma is found, the first sentence will be used as a whole.
                comma_pos = txt.find(",")
                if comma_pos < 16 or len(txt) < 32:
                    text_sections.append(txt)
                else:
                    text_sections.append(txt[:comma_pos])
                    text_sections.append(txt[comma_pos:])
            else:
                # Other sections should be as long as possible to improve the quality of the synthesis.
                # This can be accomplished by joining consecutive sentences into a single text section until it reaches the character limit.
                while sentences and (len(txt) + len(sentences[0])) < next_character_limit:
                    txt += " " + sentences.pop(0)
                text_sections.append(txt)

            # The character limit is gradually increased with each text section until it reaches the character limit of 253.
            if next_character_limit == 0:
                next_character_limit = len(text_sections[-1])
            next_character_limit += round(min(next_character_limit,
                                          len(text_sections[-1])) * (1.0 / self.realtime_factor - 1.0))
            next_character_limit = min(next_character_limit, max_character_limit)

        return text_sections

    def set_filler_phrases_enabled(self, enabled: bool) -> None:
        self.filler_sounds_enabled = enabled

    def speak(self, message: str, cache: bool = False) -> None:
        for text_section in self.split_into_text_sections(message):
            self.wave_queue.put(self.text_to_speech(text_section, cache))
        self.wave_queue.join()

    def _play_audio(self, wave: list) -> None:
        stream = self.audio_sink_factory(22050, 1)
        with stream:
            stream.write((np.array(wave) * 32767).astype(np.int16))

    def _get_audio_duration(self, waves: list) -> float:
        return len(waves) / 22050


def robot_speaker(rate: int, channels: int):
    return HttpAudioSink(f"{cfg.bucky_uri}/speaker/play_sound?rate={rate}&channels={channels}&blocking=false", rate, channels)


def local_speaker(rate: int, channels: int):
    return sounddevice.OutputStream(samplerate=rate, channels=channels, dtype='int16')


if __name__ == "__main__":
    voice = VoiceFast()
    voice.speak("Howdy!, I am Bucky, the friendly cowboy assistant! Yeehaw!")
