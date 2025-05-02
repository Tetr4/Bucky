import time
from typing import Callable, Optional
from speech_recognition import AudioSource
from bucky.audio_filter import SpeechDenoiser
import requests
import threading
import queue
import logging

logger = logging.getLogger(__name__)


class HttpAudioSource(AudioSource):
    def __init__(self, url, sample_size=2, sample_rate=44100, chunk_size=8192):
        self.url = url
        self.SAMPLE_WIDTH = sample_size
        self.SAMPLE_RATE = sample_rate
        self.CHUNK = chunk_size

    def __enter__(self):
        self.stream = HttpAudioSource.HttpAudioStream(self.url, self.SAMPLE_WIDTH, self.SAMPLE_RATE)

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.close()

    class HttpAudioStream(object):
        def __init__(self, url, sample_size, sample_rate):
            self.sample_size = sample_size
            self.sample_rate = sample_rate
            self.generator = None
            self.response = requests.get(url, stream=True)

            data = self.response.raw.read(36)
            channels = int.from_bytes(data[22:24], "little")
            sample_rate = int.from_bytes(data[24:28], "little")
            bits_per_sample = int.from_bytes(data[34:36], "little")
            if channels != 1 or sample_rate != self.sample_rate or bits_per_sample / 8 != self.sample_size:
                raise ValueError(
                    f"Invalid audio format: Channels: {channels}, Sample Rate: {sample_rate}, Bits per Sample: {bits_per_sample}")

        def read(self, size):
            while True:
                data = self.response.raw.read(size)
                if not data:
                    continue
                return data

        def close(self):
            self.response.close()


class BufferedAudioSourceWrapper(AudioSource):
    def __init__(self, audio_source_factory: Callable[[], AudioSource], denoiser: Optional[SpeechDenoiser] = None):
        self._source_factory = audio_source_factory
        self._denoiser = denoiser
        self._source: Optional[AudioSource] = None
        self._continue = threading.Event()

    def flush_stream(self):
        self.stream.flush()

    def _read_proc(self):
        while self._continue.is_set():
            samples = self._source.stream.read(self.CHUNK)
            self.stream.put(samples, self._source.SAMPLE_WIDTH, self._source.SAMPLE_RATE)

    def __enter__(self):
        self._source: AudioSource = self._source_factory()
        self._source.__enter__()
        self.SAMPLE_WIDTH = self._source.SAMPLE_WIDTH
        self.SAMPLE_RATE = self._source.SAMPLE_RATE
        self.CHUNK = self._source.CHUNK
        self.stream = BufferedAudioSourceWrapper.OutputStream(self._denoiser)
        self._continue.set()
        self._thread = threading.Thread(target=self._read_proc, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._continue.clear()
        self._thread.join()
        self._source.__exit__(exc_type, exc_value, traceback)

    class OutputStream(object):
        def __init__(self, denoiser: Optional[SpeechDenoiser]):
            self._sample_queue = queue.Queue()
            self._denoiser = denoiser
            self._avg_denoiser_ms = 0.0

        def put(self, samples: bytes, sample_width: int, sampe_rate: int):
            self._sample_queue.put((samples, sample_width, sampe_rate))

        def flush(self):
            self._sample_queue = queue.Queue()

        def read(self, size) -> bytes:
            samples, sample_width, sampe_rate = self._sample_queue.get()
            self._sample_queue.task_done()
            if self._denoiser is None:
                return samples

            denoiser_start: float = time.time()
            denoised_samples = self._denoiser.denoise(samples, sample_width, sampe_rate)
            denoiser_ms: float = (time.time() - denoiser_start) * 1000

            if len(denoised_samples) != len(samples):
                logger.error(
                    f"Denoiser returned {len(denoised_samples)/sample_width} samples from {len(samples)/sample_width} input samples.")

            self._avg_denoiser_ms = 0.99 * self._avg_denoiser_ms + 0.01 * denoiser_ms
            samples_ms: float = (len(samples) / (sample_width * sampe_rate)) * 1000
            if self._avg_denoiser_ms > samples_ms:
                logger.warning(f"Denoiser took {int(self._avg_denoiser_ms)} ms for a {int(samples_ms)} ms sample.")

            return denoised_samples
