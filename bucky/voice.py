from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import sounddevice
import torch
import threading
import queue
import random
import time
from TTS.api import TTS
from piper.voice import PiperVoice
from piper.download import get_voices, ensure_voice_exists
from bucky.audio_sink import HttpAudioSink
import bucky.config as cfg

data_dir = "assets/voice-data"

class Voice(ABC):

    @abstractmethod
    def speek_random_filler_phrase(self) -> None:
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
                 audio_sink_factory = lambda rate, channels: sounddevice.OutputStream(samplerate=rate, channels=channels, dtype='int16')) -> None:
        model_path = Path(data_dir, model + '.onnx')
        if not model_path.exists():
            voices_info = get_voices(data_dir, update_voices=True)
            ensure_voice_exists(model, data_dir, data_dir, voices_info)
        self.speaker_id = speaker_id
        self.voice = PiperVoice.load(model_path)
        self.audio_sink_factory = audio_sink_factory

    def speek_random_filler_phrase(self) -> None:
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
                 filler_phrases: list[str] = ["hm", "jo", "ähm", "also"],
                 pre_cached_phrases: list[str] = [],
                 audio_sink_factory = lambda rate, channels: sounddevice.OutputStream(samplerate=rate, channels=channels, dtype='int16')) -> None:
        # Note XTTS is not for commercial use: https://coqui.ai/cpml
        self.tts = TTS(model_name=model, progress_bar=False, gpu=torch.cuda.is_available())
        self.language = language
        self.voice_template = voice_template
        self.audio_sink_factory = audio_sink_factory
        self.cached_sounds = {}
        self.filler_sounds = []
        self.realtime_factor = 0.9 # faster machine -> smaller value
        for _ in range(3):
            for phrase in filler_phrases:
                self.filler_sounds.append(self.text_to_speach(phrase))
        for phrase in pre_cached_phrases:
            self.text_to_speach(phrase, True)

    def text_to_speach(self, message: str, cache: bool = False) -> list:
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
        audio_time = len(waves) / 22050
        self.realtime_factor = 0.5 * self.realtime_factor + 0.5 * process_time / audio_time

        return waves

    def split_into_text_sections(self, message: str) -> list[str]:
        sentences = self.tts.synthesizer.split_into_sentences(message)   
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
                if comma_pos < 1 or len(txt) < 32:
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
            next_character_limit += round(min(next_character_limit, len(text_sections[-1])) * (1.0 / self.realtime_factor - 1.0))
            next_character_limit = min(next_character_limit, max_character_limit)

        return text_sections

    def speek_random_filler_phrase(self) -> None:
        if self.filler_sounds:
            wave = random.choice(self.filler_sounds)
            stream = self.audio_sink_factory(22050, 1)
            with stream:
                stream.write((np.array(wave) * 32767).astype(np.int16))

    def speak(self, message: str, cache: bool = False) -> None:        
        wave_queue = queue.Queue()
        def play_sound_proc():
            while True:
                wave = None
                try:
                    wave = wave_queue.get(timeout=2.0 if self.filler_sounds else None)
                    if not wave:
                        break
                except queue.Empty:
                    wave = random.choice(self.filler_sounds)

                stream = self.audio_sink_factory(22050, 1)
                with stream:                    
                    stream.write((np.array(wave) * 32767).astype(np.int16))

        player_thread = threading.Thread(target=play_sound_proc, daemon=True)
        player_thread.start()

        for text_section in self.split_into_text_sections(message):
            wave_queue.put(self.text_to_speach(text_section, cache))

        wave_queue.put(None)
        player_thread.join()

def robot_speaker(rate: int, channels: int):
    return HttpAudioSink(f"{cfg.bucky_uri}/speaker/play_sound?rate={rate}&channels={channels}&blocking=false", rate, channels)

def local_speaker(rate: int, channels: int):
    return sounddevice.OutputStream(samplerate=rate, channels=channels, dtype='int16')


if __name__ == "__main__":
    voice = VoiceFast()
    voice.speak("Howdy!, I am Bucky, the friendly cowboy assistant! Yeehaw!")
