from typing import Any, Literal
import uuid
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain.storage.file_system import LocalFileStore

class MemoryStore:
    def __init__(self):
        self.store = InMemoryStore() # TODO replace with LocalFileStore
        self.namespace = ("global", "agent_memory")

    def add(self, memory: str):
        key = str(uuid.uuid4())
        value = {"memory": memory}
        self.store.put(self.namespace, key, value)

    def delete(self, key: str):
        self.store.delete(self.namespace, key)

    def update(self, key: str, memory: str):
        self.store.put(self.namespace, key, {"memory": memory})

    def dump(self) -> list[dict[str, str]]:
        return [{entry.key: entry.value['memory']}  for entry in self.store.search(self.namespace)]
