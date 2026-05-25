from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
import time
from typing import Optional


class PerceptionType(StrEnum):
    UNKNOWN = "unknown"
    USER_INPUT = "user_input"
    VISION = "vision"


@dataclass
class PerceptionInput:
    type: PerceptionType  # Which module sends the data
    content: str          # The actual information
    base_priority: int    # Developer-assigned base priority (1-10)
    timestamp: float      # When the event occurred


class PerceptionModule(ABC):
    @abstractmethod
    def get_input(self) -> Optional[PerceptionInput]:
        ...


class ChatInputModule(PerceptionModule):
    def get_input(self) -> Optional[PerceptionInput]:
        return PerceptionInput(type=PerceptionType.USER_INPUT,
                               content=input("You: "),
                               base_priority=10,
                               timestamp=time.time())


class PerceptionSystem:
    def __init__(self):
        self._modules: list[PerceptionModule] = []

    def register_perception_module(self, module: PerceptionModule):
        if module not in self._modules:
            self._modules.append(module)

    def get_all_inputs(self) -> list[PerceptionInput]:
        inputs: list[PerceptionInput] = []
        for module in self._modules:
            pinput = module.get_input()
            if pinput is not None:
                inputs.append(pinput)
        return inputs
