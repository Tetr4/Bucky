import time

from bucky.piano.base_worker import BaseWorker
from bucky.piano.shared_agent_state import SharedAgentState


class BottleneckWorker(BaseWorker):
    def __init__(self, state: SharedAgentState):
        super().__init__(state, "Bottleneck")
        self.last_vision = ""
        self.IDLE_THRESHOLD = 30

    def work_loop(self):
        time.sleep(0.5)
        trigger = False
        facts = []

        if self.state.get("collision_imminent"):
            facts.append("CRITICAL: You just performed an emergency stop due to a frontal obstacle.")
            trigger = True

        speech = self.state.get("latest_speech_input")
        if speech:
            facts.append(f"User just spoke: '{speech}'")
            self.state.update("latest_speech_input", None)
            trigger = True

        vision = self.state.get("last_seen_description")
        if vision != self.last_vision and vision != "Nothing significant.":
            facts.append(f"Visual update: {vision}")
            self.last_vision = vision
            if not self.state.get("ollama_busy"):
                trigger = True

        idle_time = time.time() - self.state.get("last_action_timestamp")
        if not trigger and idle_time > self.IDLE_THRESHOLD:
            goal = self.state.get("active_long_term_goal")
            facts.append(
                f"PROACTIVE CHECK: You have been idle for {self.IDLE_THRESHOLD} seconds. "
                f"Your goal is: '{goal}'. What is your next step?"
            )
            trigger = True
            self.state.update("last_action_timestamp", time.time())

        if trigger:
            self.state.update("filtered_context", " | ".join(facts))
            self.state.salient_event_flag.set()
