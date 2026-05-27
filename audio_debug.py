import logging
from pathlib import Path

from bucky.audio.filter import EchoCancellation, SpeechDenoiserDF
from bucky.audio.source import robot_mic, local_mic
from bucky.audio.sink import robot_speaker, local_speaker
import threading
from bucky.recorder import Recorder
from bucky.voice import Voice
logging.basicConfig(level=logging.INFO)


def main():
    speaker = local_speaker
    mic = local_mic

    aec = EchoCancellation()

    voice = Voice(
        audio_sink_factory=speaker,
        language="de",
        chunk_size_in_seconds=1.5,
        echo_cancellation=aec
    )

    recorder = Recorder(
        audio_source_factory=mic,
        language="german",
        model="turbo",
        echo_cancellation=aec,
        wav_output_dir=Path("C:/temp/wave/")
    )

    def listen_loop():
        while True:
            recorder.listen()

    listener_thread = threading.Thread(target=listen_loop, daemon=True)
    listener_thread.start()

    # text: str = "Wer während der Autofahrt über Handy oder Freisprechanlage telefoniert, fährt wie ein angetrunkener Wagenlenker."
    text: str = " ".join("""
    Wer während der Autofahrt über Handy oder Freisprechanlage telefoniert, fährt wie ein
    angetrunkener Wagenlenker. Zu diesem Schluss kommen die Psychologen Frank Drews,
    David Strayer und der Toxikologe Dennis Crouch von der Universität Utah in ihrer Studie,
    die sie heute in dem Journal Human Factors veröffentlichen. 25 Männer und 15 Frauen im
    Alter zwischen 22 und 34 Jahren nahmen an «A Comparison of the Cell Phone Driver and
    the Drunk Driver» teil. Das Bundesamt für Luftfahrt finanzierte die Untersuchungen mit
    25000 Dollar, um Rückschlüsse auf die Aufmerksamkeit von Piloten ziehen zu können.
    """.split())

    print(text)
    voice.speak(text)

    input()


if __name__ == "__main__":
    main()
