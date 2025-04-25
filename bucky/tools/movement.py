from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from bucky.robot import IRobot


class TurnToolInput(BaseModel):
    direction: str = Field(description="The direction to turn. Possible directions are 'left' or 'right'.")


class TurnTool(BaseTool):
    name: str = "turn"
    description: str = "Use this to turn your body."
    args_schema: Type[BaseModel] = TurnToolInput  # type: ignore
    robot: IRobot = None  # type: ignore

    def __init__(self, robot: IRobot):
        super().__init__()
        self.robot = robot

    def _run(self, direction: str) -> str:
        if direction == "left":
            self.robot.turn_left()
            return "turned left"
        elif direction == "right":
            self.robot.turn_right()
            return "turned right"
        return "not turned"
