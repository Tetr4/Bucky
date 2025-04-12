from typing import Optional, Type
from langchain.tools import BaseTool
from bucky.memory_store import MemoryStore
from pydantic import BaseModel, Field


class CreateMemoryToolInput(BaseModel):
    fact: str = Field(description="""A fact or information, e.g. 'User likes Pizza'""")


class CreateMemoryTool(BaseTool):
    name: str = "remember_fact"
    description: str = """Use this tool to remember new facts like people's name, preferences etc."""
    args_schema: Type[BaseModel] = CreateMemoryToolInput  # type: ignore
    store: MemoryStore = None  # type: ignore

    def __init__(self, store: MemoryStore):
        super().__init__()
        self.store = store

    def _run(self, fact: str) -> str:
        self.store.add(fact)
        return "Fact created"


class UpdateMemoryToolInput(BaseModel):
    id: str = Field(description="""ID for the fact""")
    fact: str = Field(description="""A fact or information, e.g. 'User likes Pizza'""")


class UpdateMemoryTool(BaseTool):
    name: str = "update_fact"
    description: str = """Use this tool to update existing facts like people's name, preferences etc."""
    args_schema: Type[BaseModel] = UpdateMemoryToolInput  # type: ignore
    store: MemoryStore = None  # type: ignore

    def __init__(self, store: MemoryStore):
        super().__init__()
        self.store = store

    def _run(self, id: str, fact: str) -> str:
        self.store.update(id, fact)
        return "Fact updated"


class DeleteMemoryToolInput(BaseModel):
    id: str = Field(description="""ID for the fact""")


class DeleteMemoryTool(BaseTool):
    name: str = "delete_fact"
    description: str = """Use this tool if a fact is not important anymore."""
    args_schema: Type[BaseModel] = DeleteMemoryToolInput  # type: ignore
    store: MemoryStore = None  # type: ignore

    def __init__(self, store: MemoryStore):
        super().__init__()
        self.store = store

    def _run(self, id: str) -> str:
        self.store.delete(id)
        return "Fact deleted"
