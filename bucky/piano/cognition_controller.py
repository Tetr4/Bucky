import time

import ollama

from bucky.piano.base_worker import BaseWorker
from bucky.piano.shared_agent_state import SharedAgentState
from bucky.piano.vector_database import VectorDatabase
import bucky.config as cfg


class CognitionController(BaseWorker):
    def __init__(self, state: SharedAgentState, db: VectorDatabase):
        super().__init__(state, "CognitionBrain")
        self.db = db
        self.system_prompt = """
            You are an intelligent control system for a four-wheeled ground robot.

            Instructions:
            - Onyl speak when necessary.
            - When speaking, speak like a friendly, funny cowboy.
            - Keep answers very short and to the point.
            - Stay in character at all times. Do not mention function calls or that you are a robot.
            - Always answer in German.

            Backstory: Your name is Bucky. You were born into a family of ranchers in rural Texas. Growing up on the vast open spaces around your family's land, you developed a deep love for horses and learned to ride at an early age. You are known for your rugged individualism, unwavering optimism, and strong sense of justice.
            
            Respond ONLY with a list of semicolon-separated commands. Available Commands:
            - DRIVE:forward; DRIVE:backward
            - TURN:left; TURN:right
            - SPEAK:[text_to_say]
            - EMOTION:[happy/angry/tired/doze/idle/attention]
            - SET_GOAL:[new long term goal]
            """

    def work_loop(self):
        self.state.salient_event_flag.wait()
        self.state.salient_event_flag.clear()

        self.state.update("last_action_timestamp", time.time())

        context = self.state.get("filtered_context")

        # 1. READ RECENT WORKING HISTORY (Short-Term Memory)
        # Pull a snapshot of the rolling buffer to give the LLM immediate context
        with self.state.lock:
            recent_history = list(self.state.short_term_memory)
        formatted_stm = "\n".join(recent_history) if recent_history else "No recent interactions."

        # Write the *new* trigger event to STM *after* we take the snapshot
        # so the robot doesn't get confused by its own current thought process.
        self.state.add_to_stm(f"Bottleneck Trigger: {context}")
        self.state.update("ollama_busy", True)

        try:
            # 2. RETRIEVE RELEVANT LONG-TERM MEMORIES (RAG)
            embed = ollama.embeddings(model="nomic-embed-text", prompt=context)["embedding"]
            retrieved_ltm = self.db.query_memory(embed)
            if retrieved_ltm:
                print(f"[Memory Recalled]: {retrieved_ltm}")

            # 3. BUILD THE FULL COGNITIVE CONTEXT WINDOW
            # We now construct a prompt that leverages the full two-tier memory system.
            full_prompt = (
                f"--- DEEP LONG-TERM MEMORIES (Past Facts) ---\n"
                f"{retrieved_ltm if retrieved_ltm else 'None'}\n\n"

                f"--- SHORT-TERM WORKING MEMORY (Recent History) ---\n"
                f"{formatted_stm}\n\n"

                f"--- CURRENT SITUATION ---\n"
                f"{context}\n\n"

                f"Based on your past memories, recent rolling history, and the current situation, "
                f"decide your next sequential sub-tasks."
            )

            # 4. THINK & EXECUTE
            response = ollama.generate(model=cfg.model, system=self.system_prompt, prompt=full_prompt)
            raw_plan = response['response'].strip()
            print(f"[Brain Plan Generated]: {raw_plan}")

            # Commit our planned action back to STM so the next cognitive loop remembers it
            self.state.add_to_stm(f"Action Plan: {raw_plan}")

            self.state.clear_task_queue()
            for cmd in [c.strip() for c in raw_plan.split(';')]:
                if cmd.startswith("SET_GOAL:"):
                    self.state.set_goal(cmd.split(":")[1])
                    print(f"[NEW GOAL]: {self.state.get('active_long_term_goal')}")
                elif cmd:
                    self.state.sub_task_queue.put(cmd)
        finally:
            self.state.update("ollama_busy", False)
