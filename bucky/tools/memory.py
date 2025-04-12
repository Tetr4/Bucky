from typing import Optional, Type
from langchain.tools import BaseTool
from bucky.memory_store import MemoryStore
from pydantic import BaseModel, Field

class MemoryToolInput(BaseModel):
    content: str = Field(
        description="""The main content of the memory. For example: "User expressed interest in learning about French."""
    )
    context: Optional[str] = Field(
        default=None,
        description="""Additonal context of the memory. For example: "User is a student in high school."""
    )
    memory_id: Optional[str] = Field(
        default=None,
        description="""ONLY PROVIDE IF UPDATING AN EXISTING MEMORY. The memory to overwrite."""
    )

class MemoryTool(BaseTool):
    name: str = "upsert_memory"
    description: str = """Use this to remember things for future conversations.
    This tool upserts a memory in the database. If a memory conflicts with an existing one, then just UPDATE the
    existing one by passing in a key - don't create two memories
    that are the same. If the user corrects a memory, UPDATE it."""
    args_schema: Type[BaseModel] = MemoryToolInput
    store: MemoryStore = None # type: ignore

    def __init__(self, store: MemoryStore):
        super().__init__()
        self.store = store

    def _run(self, content: str, context: Optional[str] = None, memory_id: Optional[str] = None) -> str:
        if memory_id is not None:
            self.store.update(memory_id, content)
            return f"Updated memory with ID {memory_id}"
        else:
            self.store.add(content)
            return "Added new memory"
