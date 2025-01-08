from speech_recognition import AudioSource
import requests

class HttpAudioSource(AudioSource):
    def __init__(self, url, sample_size = 2, sample_rate = 44100, chunk_size = 8192):
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
                raise ValueError(f"Invalid audio format: Channels: {channels}, Sample Rate: {sample_rate}, Bits per Sample: {bits_per_sample}")

        def read(self, size):
            while True:
                data = self.response.raw.read(size)
                if not data:
                    continue
                return data

        def close(self):
            self.response.close()
            