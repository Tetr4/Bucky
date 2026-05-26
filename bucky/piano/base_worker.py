import threading
import time

from bucky.piano.shared_agent_state import SharedAgentState


class BaseWorker(threading.Thread):
    def __init__(self, state: SharedAgentState, name: str):
        super().__init__(daemon=True, name=name)
        self.state = state

    def run(self):
        print(f"[{self.name}] Thread Started.")
        while True:
            try:
                self.work_loop()
            except Exception as e:
                print(f"[{self.name} Error] {e}")
                time.sleep(1)

    def work_loop(self):
        raise NotImplementedError()
