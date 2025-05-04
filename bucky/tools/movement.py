from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from bucky.robot import IRobot
from bucky.vision.user_tracking import UserTracker


class TurnToolInput(BaseModel):
    direction: str = Field(
        description='The turn direction. Only Left, Right or User are valid. Must always be in english.')


class TurnTool(BaseTool):
    name: str = "turn"
    description: str = "Use this to turn your robot body left, right or towards the user."
    robot: IRobot
    tracker: UserTracker

    def __init__(self, robot: IRobot, tracker: UserTracker):
        super().__init__(robot=robot, tracker=tracker, args_schema=TurnToolInput)

    def _run(self, direction: str) -> str:
        direction = direction.lower()
        if direction.startswith("l"):
            self.robot.turn_left()
            return "turned left"
        elif direction.startswith("r"):
            self.robot.turn_right()
            return "turned right"
        elif direction.startswith("u"):
            return self._turn_to_user()
        else:
            raise ValueError("invalid direction")

    def _turn_to_user(self) -> str:
        dir: str = self.tracker.user_direction
        if dir == "left":
            self.robot.turn_left()
            return "turned left towards the user"
        elif dir == "right":
            self.robot.turn_right()
            return "turned right towards the user"
        elif dir == "front":
            return "already facing towards the user"
        return "user not found"
