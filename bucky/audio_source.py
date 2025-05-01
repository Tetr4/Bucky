from typing import Callable, Optional
from speech_recognition import AudioSource
import requests
import threading
import queue


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
    def __init__(self, audio_source_factory: Callable[[], AudioSource]):
        self._source_factory = audio_source_factory
        self._instance: Optional[AudioSource] = None
        self._continue = threading.Event()
        self.stream = BufferedAudioSourceWrapper.OutputStream()

    def flush_stream(self):
        self.stream.flush()

    def _read_proc(self):
        while self._continue.is_set():
            samples = self._instance.stream.read(self.CHUNK)
            self.stream.put(samples)

    def __enter__(self):
        self._instance: AudioSource = self._source_factory()
        self._instance.__enter__()
        self.SAMPLE_WIDTH = self._instance.SAMPLE_WIDTH
        self.SAMPLE_RATE = self._instance.SAMPLE_RATE
        self.CHUNK = self._instance.CHUNK
        self._continue.set()
        self._thread = threading.Thread(target=self._read_proc, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._continue.clear()
        self._thread.join()
        self._instance.__exit__(exc_type, exc_value, traceback)

    class OutputStream(object):
        def __init__(self):
            self._sample_queue = queue.Queue()

        def put(self, samples):
            self._sample_queue.put(samples)

        def flush(self):
            self._sample_queue = queue.Queue()

        def read(self, size):
            result = self._sample_queue.get()
            self._sample_queue.task_done()
            return result
