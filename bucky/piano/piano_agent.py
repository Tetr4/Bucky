import logging
import time

from bucky.audio.filter import SpeechDenoiserDF
from bucky.audio.sink import local_speaker
from bucky.audio.source import local_mic
from bucky.fx_player import FxPlayer
from bucky.piano.action_controller import ActionController
from bucky.piano.bottleneck_worker import BottleneckWorker
from bucky.piano.cognition_controller import CognitionController
from bucky.piano.memory_consolidator import MemoryConsolidator
from bucky.piano.shared_agent_state import SharedAgentState
from bucky.piano.transcription_worker import TranscriptionWorker
from bucky.piano.vector_database import VectorDatabase
from bucky.piano.vision_worker import VisionWorker
from bucky.recorder import Recorder, Transcription
from bucky.robot import FakeBot, IRobot
from bucky.voice import Voice

logging.basicConfig(level=logging.INFO)


class PianoAgent:
    def __init__(self):
        print("Booting Object-Oriented PIANO Architecture container...")

        self.db = VectorDatabase()
        self.state = SharedAgentState()

        self.robot: IRobot = FakeBot()
        speaker_factory = local_speaker
        mic_factory = local_mic

        greeting_phrases = ["Howdy Partner!", "Howdy!", "Moin!"]
        question_phrases = ["was?", "wie?", "was hast du gesagt?", "hab dich nicht verstanden"]
        voice = Voice(
            audio_sink_factory=speaker_factory,
            pre_cached_phrases=greeting_phrases + question_phrases,
            language="de",
            chunk_size_in_seconds=1.5
        )

        fx_player = FxPlayer(speaker_factory)

        def on_start_listening():
            fx_player.play_rising_chime().join()

        def on_waiting_for_wakeup():
            fx_player.play_descending_chime().join()

        def on_unintelligible(trans: Transcription) -> bool:
            return trans.speech_prob > 0.1 and trans.phrase not in ["Vielen Dank.", "Untertitelung des ZDF, 2020"]

        recorder = Recorder(
            wakewords=["hey b", "hey p", "hey k", "bucky", "pakki", "kumpel", "howdy"],
            language="german",
            model="turbo",
            # denoiser=SpeechDenoiserDF(),
            audio_source_factory=mic_factory,
            wakeword_timeout=99999,
            on_start_listening=on_start_listening,
            on_waiting_for_wakeup=on_waiting_for_wakeup,
            on_unintelligible=on_unintelligible,
            # has_user_attention=lambda: tracker.max_attention > 0.5,
            # transcription_llm=llm, # use external LLM instead of Whisper
        )

        # mute recorder when robot is speaking to prevent echo transcription
        voice.set_speaking_callback(lambda is_speaking: recorder.set_muted(is_speaking))

        # Worker creation injecting the structural implementations
        self.workers = [
            TranscriptionWorker(self.state, recorder),
            VisionWorker(self.state, self.robot),
            # LidarWorker(self.state, self.robot),
            BottleneckWorker(self.state),
            MemoryConsolidator(self.state, self.db),
            CognitionController(self.state, self.db),
            ActionController(self.state, self.robot, voice)
        ]

    def run(self):
        for worker in self.workers:
            worker.start()

        print(f"Robot execution live! Baseline Goal: '{self.state.get('active_long_term_goal')}'")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nSafe termination sequence initialized.")
            self.robot.drive_forward(distance=0.0, speed=0.0)  # Force stop
            self.robot.release()


if __name__ == "__main__":
    PianoAgent().run()
