from pathlib import Path
import numpy as np
import sounddevice
import torch
import threading
import queue
import random
import time
import hashlib
import pickle
import os
from bucky.gpu_utils import get_cuda_devices
from TTS.api import TTS
from bucky.threading_utils import ThreadWorkerPool
from bucky.voices.voice import Voice, voice_data_dir


class VoiceQuality(Voice):
    '''
    Generate high quality speech with TTS: https://github.com/coqui-ai/TTS
    Voices: pdm run tts --list_models
    '''

    def __init__(self,
                 max_gpus: int = 2,
                 model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
                 language: str = "en",
                 voice_template: Path = Path(voice_data_dir, "voice_template.wav"),
                 filler_phrases: list[str] = ["hm", "jo", "ähm", "ah", "also", "mal überlegen"],
                 pre_cached_phrases: list[str] = [],
                 audio_sink_factory=lambda rate, channels: sounddevice.OutputStream(samplerate=rate, channels=channels, dtype='int16')) -> None:
        # Note XTTS is not for commercial use: https://coqui.ai/cpml
        self.tts_instances: list[TTS] = []
        for cuda_device in get_cuda_devices():
            if max_gpus == len(self.tts_instances):
                break
            print("TTS: creating GPU instance", cuda_device)
            tts = TTS(model_name=model, progress_bar=False)
            tts.to(cuda_device.torch_device, dtype=torch.float, non_blocking=True)
            self.tts_instances.append(tts)

        if not self.tts_instances:
            print("TTS: creating CPU instance")
            self.tts_instances.append(TTS(model_name=model, progress_bar=False))

        self.language = language
        self.voice_template = voice_template
        self.audio_sink_factory = audio_sink_factory

        self._init_phrase_cache()

        self.filler_sounds_enabled = False
        self.filler_sounds = []

        self.realtime_factor = 0.9  # faster machine -> smaller value

        self.lock = threading.RLock()
        self.pool = ThreadWorkerPool(len(self.tts_instances))
        self.pool.start()

        if filler_phrases:
            self.filler_sounds = self.multi_text_to_speech(filler_phrases, cache=True, retries=3)
            random.shuffle(self.filler_sounds)

        if pre_cached_phrases:
            self.multi_text_to_speech(pre_cached_phrases, cache=True, retries=3)

        self.wave_queue = queue.Queue()

        def play_sound_proc():
            next_timeout = 3.0
            while True:
                try:
                    wave = self.wave_queue.get(timeout=next_timeout if self.filler_sounds else None)
                    next_timeout = 3.0
                    try:
                        self._play_audio(wave)
                    finally:
                        self.wave_queue.task_done()
                except queue.Empty:
                    if self.filler_sounds_enabled and self.filler_sounds:
                        next_timeout += 1
                        self._play_audio(random.choice(self.filler_sounds))

        self.player_thread = threading.Thread(target=play_sound_proc, daemon=True)
        self.player_thread.start()

    def multi_text_to_speech(self, messages: list[str], cache: bool = False, retries: int = 1) -> list[list]:
        waves: list[list] = []
        already_cached: set[str] = set(msg for msg in messages if self._is_cached(msg))
        self.pool.wait_for_completion()
        for message in messages * retries:
            self.pool.add_job((message, False), self.text_to_speech_proc, waves.append)
        self.pool.wait_for_completion()

        shortest_waves: list[list] = []
        for i, wave in enumerate(waves):
            idx = i % len(messages)
            if idx >= len(shortest_waves):
                shortest_waves.append(wave)
            elif self._get_audio_duration(shortest_waves[idx]) > self._get_audio_duration(wave):
                shortest_waves[idx] = wave

        if cache:
            with self.lock:
                for i, shortest_wave in enumerate(shortest_waves):
                    if messages[i] not in already_cached:
                        self._add_to_cache(messages[i], shortest_wave)

        return shortest_waves

    def text_to_speech_proc(self, worker_index: int, args: tuple[str, bool]) -> list:
        message, cache = args
        return self.text_to_speech(self.tts_instances[worker_index], message, cache)

    def text_to_speech(self, tts: TTS, message: str, cache: bool = False) -> list:
        with self.lock:
            if cached_waves := self._try_get_from_cache(message):
                return cached_waves

        start_time = time.time()
        waves = tts.tts(
            text=message,
            language=self.language,
            speaker_wav=str(self.voice_template),
            split_sentences=False,
        )
        process_time = time.time() - start_time
        audio_time = self._get_audio_duration(waves)

        with self.lock:
            self.realtime_factor = 0.5 * self.realtime_factor + 0.5 * \
                process_time / audio_time * 1 / len(self.tts_instances)
            if cache:
                self._add_to_cache(message, waves)

        return waves

    def split_into_text_sections(self, tts: TTS, message: str) -> list[str]:
        sentences = tts.synthesizer.split_into_sentences(message)  # type: ignore
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

    def speak(self, message: str) -> None:
        self.pool.wait_for_completion()
        for text_section in self.split_into_text_sections(self.tts_instances[0], message):
            self.pool.add_job((text_section, False), self.text_to_speech_proc, self.wave_queue.put)
        self.pool.wait_for_completion()
        self.wave_queue.join()

    def _play_audio(self, wave: list) -> None:
        stream = self.audio_sink_factory(22050, 1)
        with stream:
            stream.write((np.array(wave) * 32767).astype(np.int16))

    def _get_audio_duration(self, waves: list) -> float:
        return len(waves) / 22050

    def _get_cache_file_path(self, message: str) -> Path:
        msg_hash: str = hashlib.sha256(message.encode('utf-8')).hexdigest()[:8]
        return Path(self.phrase_cache_dir, f"{self.voice_template_hash}_{msg_hash}.bin")

    def _init_phrase_cache(self):
        with open(self.voice_template, "rb") as f:
            self.voice_template_hash = hashlib.sha256(f.read()).hexdigest()[:8]

        self.phrase_cache_dir = Path(voice_data_dir, "cached_phrases")
        self.phrase_cache_dir.mkdir(parents=True, exist_ok=True)

    def _is_cached(self, message: str) -> bool:
        file_path = self._get_cache_file_path(message)
        return os.path.exists(file_path)

    def _try_get_from_cache(self, message: str) -> list:
        file_path = self._get_cache_file_path(message)
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            except Exception as ex:
                print(str(ex))
        return []

    def _add_to_cache(self, message: str, waves: list):
        file_path = self._get_cache_file_path(message)
        print(f"caching audio data of '{message}' to {file_path}")
        try:
            with open(file_path, "wb") as f:
                pickle.dump(waves, f)
        except Exception as ex:
            print(str(ex))
