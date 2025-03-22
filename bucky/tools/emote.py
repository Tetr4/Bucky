from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from bucky.robot import IRobot

class EmoteToolInput(BaseModel):
    emotion: str = Field(description='emotion: The emotion to show. Only these values are allowed: "happy", "angry", "tired". Parameters must be in english.')

class EmoteTool(BaseTool):
    name: str = "emote"
    description: str = "Use this to show an emotion on your face for a few seconds."
    args_schema: Type[BaseModel] = EmoteToolInput
    robot: IRobot = None # type: ignore

    def __init__(self, robot: IRobot):
        super().__init__()
        self.robot = robot

    def _run(self, emotion: str) -> str:
        if emotion == "happy":
            self.robot.emote_happy()
        elif emotion == "angry":
            self.robot.emote_angry()
        elif emotion == "tired":
            self.robot.emote_tired()
        else:
            raise ValueError(f"Invalid emotion. '{emotion}' Only 'happy', 'angry', and 'tired' are allowed")
        return ""
