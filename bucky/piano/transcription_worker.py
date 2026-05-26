from bucky.piano.base_worker import BaseWorker
from bucky.piano.shared_agent_state import SharedAgentState
from bucky.recorder import Recorder, Transcription


class TranscriptionWorker(BaseWorker):
    def __init__(self, state: SharedAgentState, recorder: Recorder):
        super().__init__(state, "Vision")
        self.recorder = recorder

    def work_loop(self):
        trans: Transcription = self.recorder.listen()
        if trans.phrase:
            self.state.update("latest_speech_input", trans.phrase)
