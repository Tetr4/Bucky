import queue
import threading
import time

from bucky.piano.base_worker import BaseWorker
from bucky.piano.shared_agent_state import SharedAgentState
from bucky.robot import IRobot
from bucky.voice import Voice


class ActionController(BaseWorker):
    """Production-grade Physical Dispatcher.
    Handles physical tasks asynchronously. Locomotion commands use a non-blocking lock,
    while speech commands are queued sequentially so that no spoken words are ever dropped.
    """

    def __init__(self, state: SharedAgentState, hardware: IRobot, voice: Voice):
        super().__init__(state, "ActionController")
        self.hardware = hardware
        self.voice = voice

        # Hardware locks for physical operations
        self.motor_lock = threading.Lock()
        self.display_lock = threading.Lock()

        # Dedicated thread-safe queue for sequential audio playback
        self.speech_queue = queue.Queue()

        # Spin up the persistent speech processing thread immediately on boot
        threading.Thread(target=self._speech_consumer_loop, daemon=True, name="SpeechConsumer").start()

    # ------------------------------------------------------------------
    # PERSISTENT SPEECH CONSUMER (Sequential, Non-Dropping Audio Thread)
    # ------------------------------------------------------------------
    def _speech_consumer_loop(self):
        """Runs continuously in the background, executing speech events one by one."""
        print("[Speech Consumer] Pipeline initialized.")
        while True:
            try:
                # This blocks indefinitely until a new phrase enters the speech queue
                text = self.speech_queue.get()

                print(f"[Speech Active Execution] -> Processing audio for: \"{text}\"")
                self.voice.speak(text)
                print(f"[Speech Active Execution] -> Finished speaking phrase.")
                self.speech_queue.task_done()

            except Exception as e:
                print(f"[Speech Consumer Error] {e}")
                time.sleep(1)

    # ------------------------------------------------------------------
    # ASYNC MOVEMENT & EMOTION DISPATCHERS
    # ------------------------------------------------------------------
    def _dispatch_emotion(self, emotion_type: str):
        def worker():
            with self.display_lock:
                if emotion_type == "happy":
                    self.hardware.emote_happy()
                elif emotion_type == "angry":
                    self.hardware.emote_angry()
                elif emotion_type == "tired":
                    self.hardware.emote_tired()
                elif emotion_type == "doze":
                    self.hardware.emote_doze()
                elif emotion_type == "idle":
                    self.hardware.emote_idle()
                elif emotion_type == "attention":
                    self.hardware.emote_attention()
        threading.Thread(target=worker, daemon=True).start()

    def _dispatch_movement(self, move_type: str, *args):
        def worker():
            # Non-blocking lock: if the robot is currently driving/turning, drop the new movement
            acquired = self.motor_lock.acquire(blocking=False)
            if not acquired:
                print(f"[Action Dropped] Motors are busy. Ignoring: '{move_type}'")
                return

            try:
                if self.state.get("collision_imminent"):
                    print("[Motor Cancelled] Safety halt active prior to movement initialization.")
                    return

                def hardware_blocker():
                    if move_type == "forward":
                        self.hardware.drive_forward(*args)
                    elif move_type == "backward":
                        self.hardware.drive_backward(*args)
                    elif move_type == "left":
                        self.hardware.turn_left(*args)
                    elif move_type == "right":
                        self.hardware.turn_right(*args)

                hw_thread = threading.Thread(target=hardware_blocker, daemon=True)
                hw_thread.start()

                # Safety Monitor loop checking at 50Hz while the movement execution thread runs
                while hw_thread.is_alive():
                    if self.state.get("collision_imminent"):
                        print("[EMERGENCY OVERRIDE] Threat tripped mid-motion! Applying brakes.")
                        self.hardware.drive_forward(distance=0.0, speed=0.0)
                        break
                    time.sleep(0.02)

            finally:
                self.motor_lock.release()

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # MAIN DISPATCHER WORK LOOP (Microsecond Resolution)
    # ------------------------------------------------------------------
    def work_loop(self):
        task = self.state.sub_task_queue.get()

        # Preemptively catch movement requests if the path is physically blocked
        is_movement = task.startswith("DRIVE:") or task.startswith("TURN:")
        if is_movement and self.state.get("collision_imminent"):
            print(f"[Safety Filter] Dropping movement '{task}' due to active LiDAR threat.")
            self.state.sub_task_queue.task_done()
            return

        # 1. IMMEDIATE PRIORITY REFLEX INTERRUPT
        if task == "REFLEX:STOP":
            print("[CRITICAL] Processing Immediate Interrupt!")
            self.hardware.drive_forward(distance=0.0, speed=0.0)

        # 2. NON-DROPPING SPEECH DISPATCH (Appends directly to audio pipeline)
        elif task.startswith("SPEAK:"):
            text = task.split(":")[1]
            print(f"[Action Controller] Enqueuing speech: \"{text}\"")
            self.speech_queue.put(text)

        # 3. EMOTION DISPATCH (Asynchronous UI thread)
        elif task.startswith("EMOTION:"):
            emotion = task.split(":")[1].lower()
            self._dispatch_emotion(emotion)

        # 4. PHYSICAL MOTION DISPATCH (Non-blocking lock protected thread)
        elif task.startswith("DRIVE:"):
            direction = task.split(":")[1]
            if direction == "forward":
                self._dispatch_movement("forward", 0.5, 0.2)
            elif direction == "backward":
                self._dispatch_movement("backward", 0.5, 0.2)

        elif task.startswith("TURN:"):
            direction = task.split(":")[1]
            if direction == "left":
                self._dispatch_movement("left", 45.0, 0.3)
            elif direction == "right":
                self._dispatch_movement("right", 45.0, 0.3)

        self.state.sub_task_queue.task_done()
