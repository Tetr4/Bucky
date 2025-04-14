from pathlib import Path
from typing import Iterator
import numpy as np
import sounddevice
import torch
import threading
import queue
import random
import hashlib
import pickle
import os
from bucky.gpu_utils import get_free_cuda_device
from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig
from bucky.voices.voice import Voice, voice_data_dir


class VoiceQualityLowLatency(Voice):
    def __init__(self,
                 language: str = "en",
                 voice_template: Path = Path(voice_data_dir, "voice_template.wav"),
                 filler_phrases: list[str] = ["hm", "jo", "ähm", "ah", "also", "mal überlegen"],
                 pre_cached_phrases: list[str] = [],
                 chunk_size_in_seconds: float = 1.5,
                 audio_sink_factory=lambda rate, channels: sounddevice.OutputStream(samplerate=rate, channels=channels, dtype='int16')) -> None:
        # Note XTTS is not for commercial use: https://coqui.ai/cpml

        self.language = language
        self.audio_sink_factory = audio_sink_factory
        self.chunk_size_in_seconds = chunk_size_in_seconds
        self.filler_phrases = filler_phrases
        self.filler_sounds_enabled = False

        self._init_xtts(model="tts_models/multilingual/multi-dataset/xtts_v2", voice_template=voice_template)
        self._init_phrase_cache(pre_cached_phrases + filler_phrases)

        self.wave_queue: queue.Queue[np.ndarray | torch.Tensor] = queue.Queue()
        self.player_thread = threading.Thread(target=self._playback_proc, daemon=True)
        self.player_thread.start()

    @property
    def synthesizer(self) -> Synthesizer:
        synth = self.tts.synthesizer
        assert isinstance(synth, Synthesizer)
        return synth

    @property
    def tts_model(self) -> Xtts:
        model = self.synthesizer.tts_model
        assert isinstance(model, Xtts)
        return model

    @property
    def tts_config(self) -> XttsConfig:
        cfg = self.synthesizer.tts_config
        assert isinstance(cfg, XttsConfig)
        return cfg

    def set_filler_phrases_enabled(self, enabled: bool) -> None:
        self.filler_sounds_enabled = enabled

    def _init_xtts(self, model: str, voice_template: Path):
        if cuda_device := get_free_cuda_device(2 * (1024**3)):
            print("TTS: creating GPU instance", cuda_device)
            self.tts = TTS(model_name=model, progress_bar=False).to(
                cuda_device.torch_device, dtype=torch.float, non_blocking=True)
        else:
            print("TTS: creating CPU instance")
            self.tts = TTS(model_name=model, progress_bar=False)

        cfg: XttsConfig = self.tts_config
        self.gpt_cond_latent, self.speaker_embedding = self.tts_model.get_conditioning_latents(
            audio_path=voice_template,
            gpt_cond_len=cfg.gpt_cond_len,
            gpt_cond_chunk_len=cfg.gpt_cond_chunk_len,
            max_ref_length=cfg.max_ref_len,
            sound_norm_refs=cfg.sound_norm_refs)

    def _xtts_inference_stream(self, text: str) -> Iterator[torch.Tensor]:
        cfg: XttsConfig = self.tts_config
        chunk: torch.Tensor
        for chunk in self.tts_model.inference_stream(
                text,
                self.language,
                self.gpt_cond_latent,
                self.speaker_embedding,
                stream_chunk_size=int(22050 * self.chunk_size_in_seconds / 1000),
                overlap_wav_len=8,  # 1024,
                temperature=cfg.temperature,
                length_penalty=cfg.length_penalty,
                repetition_penalty=cfg.repetition_penalty,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                do_sample=True):
            yield chunk.cpu().squeeze()

    def _xtts_inference(self, text: str) -> np.ndarray:
        cfg: XttsConfig = self.tts_config
        return self.tts_model.inference(
            text,
            self.language,
            self.gpt_cond_latent,
            self.speaker_embedding,
            temperature=cfg.temperature,
            length_penalty=cfg.length_penalty,
            repetition_penalty=cfg.repetition_penalty,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            do_sample=True)["wav"]

    def _split_into_text_sections(self, text: str, max_section_len: int = 253) -> list[str]:
        sections: list[str] = []
        for sentence in self.tts.synthesizer.split_into_sentences(text):  # type: ignore
            if sections and len(sections[-1]) + len(sentence) < max_section_len:
                sections[-1] = f"{sections[-1]} {sentence}"
            elif len(sentence) <= max_section_len:
                sections.append(sentence)
            else:
                sub_sentence = ""
                for word in sentence.split():
                    if len(sub_sentence) + len(word) < max_section_len:
                        if sub_sentence:
                            sub_sentence += " "
                        sub_sentence += word
                        continue
                    if sub_sentence:
                        sections.append(sub_sentence)
                    sub_sentence = word
                if sub_sentence:
                    sections.append(sub_sentence)
        return sections

    def speak(self, message: str) -> None:
        self.wave_queue.join()

        if not self._enqueu_from_cache(message):
            for text_section in self._split_into_text_sections(message):
                for chunk in self._xtts_inference_stream(text_section):
                    self.wave_queue.put(chunk)

        self.wave_queue.join()

    def _enqueu_from_cache(self, phrase: str) -> bool:
        cached_audio_phrase = self.cached_audio_phrases.get(phrase)
        if not cached_audio_phrase:
            return False
        self.wave_queue.put(random.choice(cached_audio_phrase))
        return True

    def _playback_proc(self):
        stream = self.audio_sink_factory(22050, 1)
        with stream:
            while True:
                try:
                    wave = self.wave_queue.get(timeout=3 if self.filler_phrases else None)
                    try:
                        stream.write((np.array(wave) * 32767).astype(np.int16))
                    finally:
                        self.wave_queue.task_done()
                except queue.Empty:
                    if self.filler_sounds_enabled and self.filler_phrases:
                        self._enqueu_from_cache(random.choice(self.filler_phrases))

    def _init_phrase_cache(self, pre_cached_phrases: list[str], variations: int = 3):
        self.cached_audio_phrases: dict[str, list] = {}
        self.phrase_cache_dir = Path(voice_data_dir, "cached_phrases")
        self.phrase_cache_dir.mkdir(parents=True, exist_ok=True)

        def get_cache_file_path(message: str) -> Path:
            msg_hash: str = hashlib.sha256(message.encode('utf-8')).hexdigest()[:8]
            return Path(self.phrase_cache_dir, f"{msg_hash}.bin")

        def try_get_from_cache(message: str) -> list:
            file_path = get_cache_file_path(message)
            if os.path.exists(file_path):
                try:
                    print(f"TTS: loading audio data of '{message}' from {file_path}")
                    with open(file_path, "rb") as f:
                        return pickle.load(f)
                except Exception as ex:
                    print(str(ex))
            return []

        def add_to_cache(message: str, waves: list):
            file_path = get_cache_file_path(message)
            print(f"TTS: saving audio data of '{message}' to {file_path}")
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(waves, f)
            except Exception as ex:
                print(str(ex))

        additional_variants: int = 2
        for phrase in pre_cached_phrases:
            if waves := try_get_from_cache(phrase):
                self.cached_audio_phrases[phrase] = waves
            else:
                print(f"TTS: preprocessing '{phrase}' ...")
                waves = []
                for _ in range(variations + additional_variants):
                    waves.append(self._xtts_inference(phrase))

                # sort by length
                waves.sort(key=lambda wave: len(wave))

                # remove the longest variants
                waves = waves[:-additional_variants]

                self.cached_audio_phrases[phrase] = waves

                add_to_cache(phrase, waves)
