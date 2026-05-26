import time

import ollama

from bucky.piano.base_worker import BaseWorker
from bucky.piano.shared_agent_state import SharedAgentState
from bucky.piano.vector_database import VectorDatabase
import bucky.config as cfg


class MemoryConsolidator(BaseWorker):
    def __init__(self, state: SharedAgentState, db: VectorDatabase):
        super().__init__(state, "MemoryWorker")
        self.db = db

    def work_loop(self):
        time.sleep(60)
        if self.state.get("ollama_busy") or len(self.state.get("short_term_memory")) < 5:
            return

        with self.state.lock:
            stm_snapshot = list(self.state.short_term_memory)
            self.state.short_term_memory.clear()

        stm_text = "\n".join(stm_snapshot)
        self.state.update("ollama_busy", True)

        try:
            prompt = "Extract permanent facts from this log. If none, output 'NONE'.\nLog:\n" + stm_text
            response = ollama.generate(model=cfg.model, prompt=prompt)
            facts = response['response'].strip()

            if "NONE" not in facts.upper() and len(facts) > 5:
                print(f"[Memory Created]: {facts}")
                embed = ollama.embeddings(model="nomic-embed-text", prompt=facts)["embedding"]
                self.db.add_memory(facts, embed)
        finally:
            self.state.update("ollama_busy", False)
