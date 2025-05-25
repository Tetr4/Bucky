from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from bucky.robot import IRobot
from bucky.vision.user_tracking import UserTracker


class TurnToolInput(BaseModel):
    direction: str = Field(
        description='The turn direction. Only Left or Right are valid. Must always be in english.')
    angle: float = Field(description='The turning angle in degrees.')


class TurnTool(BaseTool):
    name: str = "turn"
    description: str = "Use this to turn your robot body left or right."
    robot: IRobot
    tracker: UserTracker

    def __init__(self, robot: IRobot, tracker: UserTracker):
        super().__init__(robot=robot, tracker=tracker, args_schema=TurnToolInput)

    def _run(self, direction: str, angle: float) -> str:
        direction = direction.lower()
        if direction.startswith("l"):
            self.robot.turn_left(abs(angle), 0.8)
            return "turned left"
        elif direction.startswith("r"):
            self.robot.turn_right(abs(angle), 0.8)
            return "turned right"
        elif direction.startswith("u"):
            return self._turn_to_user(angle)
        else:
            raise ValueError("invalid direction")

    def _turn_to_user(self, angle: float) -> str:
        dir: str = self.tracker.user_direction
        angle = max(10, angle)
        if dir == "left":
            self.robot.turn_left(angle, 1.0)
            return "turned left towards the user"
        elif dir == "right":
            self.robot.turn_right(angle, 1.0)
            return "turned right towards the user"
        elif dir == "front":
            return "already facing towards the user"
        return "user not found"


class DriveToolInput(BaseModel):
    direction: str = Field(
        description='The driving direction. Only Forward or Backward are valid. Must always be in english.')
    distance: float = Field(description='The driving distance in meters.')
    speed: float = Field(description='The driving speed between 0.0 and 1.0.')


class DriveTool(BaseTool):
    name: str = "drive"
    description: str = "Use this to move your robot body forward or backward."
    robot: IRobot

    def __init__(self, robot: IRobot):
        super().__init__(robot=robot, args_schema=DriveToolInput)

    def _run(self, direction: str, distance: float, speed: float) -> str:
        direction = direction.lower()
        speed = max(0.5, min(1.0, speed))
        if direction.startswith("f"):
            self.robot.drive_forward(abs(distance), speed)
            return "moved forward"
        elif direction.startswith("b"):
            self.robot.drive_backward(abs(distance), speed)
            return "moved backward"
        else:
            raise ValueError("invalid direction")
