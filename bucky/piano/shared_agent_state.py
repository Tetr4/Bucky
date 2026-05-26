from collections import deque
import json
import queue
import threading
import time


class SharedAgentState:
    def __init__(self):
        self.lock = threading.RLock()

        self.last_seen_description = "Nothing significant."
        self.latest_speech_input = None
        self.collision_imminent = False
        self.min_front_distance = float('inf')

        self.salient_event_flag = threading.Event()
        self.filtered_context = ""

        self.active_long_term_goal = self.load_goal()
        self.last_action_timestamp = time.time()
        self.short_term_memory = deque(maxlen=15)

        self.sub_task_queue = queue.Queue()
        self.ollama_busy = False

    def load_goal(self):
        try:
            with open("goals.json", "r") as f:
                return json.load(f).get("current_goal", "Explore the environment safely.")
        except FileNotFoundError:
            return "Explore the environment safely."

    def set_goal(self, new_goal):
        with self.lock:
            self.active_long_term_goal = new_goal
            with open("goals.json", "w") as f:
                json.dump({"current_goal": new_goal}, f)

    def update(self, key, value):
        with self.lock:
            setattr(self, key, value)

    def get(self, key):
        with self.lock:
            return getattr(self, key)

    def add_to_stm(self, event_string):
        with self.lock:
            current_time = time.strftime("%H:%M:%S")
            self.short_term_memory.append(f"[{current_time}] {event_string}")

    def clear_task_queue(self):
        with self.sub_task_queue.mutex:
            self.sub_task_queue.queue.clear()
