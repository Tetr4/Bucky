import uuid
import requests
import sounddevice
import logging
import bucky.config as cfg

logger = logging.getLogger(__name__)


class HttpAudioSink(object):
    def __init__(self, url: str, rate: int, channels: int):
        self.url = url.replace("<stream_id>", str(uuid.uuid4()))
        self.bytes_per_seconds = rate * channels * 2

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def write(self, data):
        try:
            requests.post(self.url, data=data.tobytes())
        except Exception as ex:
            logger.error(str(ex))


def robot_speaker(rate: int, channels: int):
    return HttpAudioSink(f"{cfg.bucky_uri}/speaker/play_sound?rate={rate}&channels={channels}&blocking=true&id=<stream_id>", rate, channels)


def local_speaker(rate: int, channels: int):
    return sounddevice.OutputStream(samplerate=rate, channels=channels, dtype='int16')
