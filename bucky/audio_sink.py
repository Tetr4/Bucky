import requests
import sounddevice
import logging
import time
import bucky.config as cfg

logger = logging.getLogger(__name__)


class HttpAudioSink(object):
    def __init__(self, url: str, rate: int, channels: int):
        self.url = url
        self.bytes_per_seconds = rate * channels * 2
        self.buffer = bytearray()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            requests.post(self.url, data=self.buffer)
            audio_duration = len(self.buffer) / self.bytes_per_seconds
            time.sleep(audio_duration)
        except Exception as ex:
            logger.error(str(ex))

    def write(self, data):
        self.buffer += data.tobytes()


def robot_speaker(rate: int, channels: int):
    return HttpAudioSink(f"{cfg.bucky_uri}/speaker/play_sound?rate={rate}&channels={channels}&blocking=false", rate, channels)


def local_speaker(rate: int, channels: int):
    return sounddevice.OutputStream(samplerate=rate, channels=channels, dtype='int16')
