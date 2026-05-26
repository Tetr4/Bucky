import time

import ollama

from bucky.piano.base_worker import BaseWorker
from bucky.piano.shared_agent_state import SharedAgentState
from bucky.robot import IRobot
import bucky.config as cfg


class VisionWorker(BaseWorker):
    def __init__(self, state: SharedAgentState, hardware: IRobot):
        super().__init__(state, "Vision")
        self.hardware = hardware

    def work_loop(self):
        time.sleep(3)
        if self.state.get("ollama_busy"):
            return

        # Call your strict method signature
        image_base64 = self.hardware.take_image(width=640, height=480)

        self.state.update("ollama_busy", True)
        try:
            response = ollama.generate(
                model=cfg.model,
                prompt='Describe what is directly in front of the camera in one short sentence.',
                images=[image_base64]
            )
            self.state.update("last_seen_description", response['response'].strip())
        finally:
            self.state.update("ollama_busy", False)
