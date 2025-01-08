import requests
import sounddevice

class HttpAudioSink(object):
    def __init__(self, url):
        self.url = url
        self.buffer = bytearray()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        requests.post(self.url, data=self.buffer)

    def write(self, data):
        self.buffer += data.tobytes()

def create_robot_audio_sink(rate: int, channels: int):
    return HttpAudioSink(f"http://bucky.local:5000/speaker/play_sound?channels={channels}&rate={rate}&blocking=true")

def create_soundcard_audio_sink(rate: int, channels: int):
    return sounddevice.OutputStream(samplerate=rate, channels=channels, dtype='int16')