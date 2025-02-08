from typing import Type
from pydantic import BaseModel, Field
import requests
import time
from datetime import datetime
from pytz import timezone
from langchain_core.tools import tool, BaseTool
import bucky.config as cfg
from bucky.robot import IRobot

@tool(parse_docstring=True)
def get_current_time() -> str:
    """Returns the current time as ISO 8601 string.
    """
    return datetime.now().astimezone(timezone("Europe/Berlin")).isoformat()

@tool(parse_docstring=True)
def get_random_meal() -> str:
    """Returns a random meal as JSON from the meal database."""
    response = requests.get(f"{cfg.meal_db_uri}/api/json/v1/1/random.php")
    return response.json()

@tool(parse_docstring=True)
def search_meal_by_ingredient(ingredient: str) -> str:
    """Returns a meal as JSON from the meal database that contains the given ingredient.

    Args:
        ingredient: An ingredient in the meal. Must be in snake_case.
    """
    response = requests.get(f"{cfg.meal_db_uri}/api/json/v1/1/filter.php?i={ingredient}")
    return response.json()

class EmoteToolInput(BaseModel):
    emotion: str = Field(description='emotion: The emotion to show. Only these values are allowed: "happy", "angry", "tired". Parameters must be in english.')

class EmoteTool(BaseTool):
    name: str = "emote"
    description: str = "Use this to show an emotion on your face for a few seconds."
    args_schema: Type[BaseModel] = EmoteToolInput
    robot: IRobot = None

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

class TakeImageTool(BaseTool):
    name: str = "take_image"
    description: str = "Returns a description of what you currently see with your eyes."
    robot: IRobot | None = None

    def __init__(self, robot: IRobot):
        super().__init__()
        self.robot = robot

    def _run(self) -> str:
        return self.robot.take_image(640, 480) if self.robot else ""