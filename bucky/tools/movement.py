from typing import Optional
from langchain_core.tools import BaseTool
from bucky.robot import IRobot
from bucky.vision.user_tracking import UserTracker


class TurnTowardsUserTool(BaseTool):
    name: str = "turn"
    description: str = "Use this to turn your body towards the user."
    robot: Optional[IRobot] = None
    tracker: Optional[UserTracker] = None

    def __init__(self, robot: IRobot, tracker: UserTracker):
        super().__init__()
        self.robot = robot
        self.tracker = tracker

    def _run(self) -> str:
        assert self.robot
        assert self.tracker
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
