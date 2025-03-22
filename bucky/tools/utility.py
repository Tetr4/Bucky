from typing import Callable
from datetime import datetime
from pytz import timezone
from langchain_core.tools import tool, BaseTool
from bucky.robot import IRobot

@tool(parse_docstring=True)
def get_current_time() -> str:
    """Returns the current time as ISO 8601 string.
    """
    return datetime.now().astimezone(timezone("Europe/Berlin")).isoformat()

class TakeImageTool(BaseTool):
    name: str = "take_image"
    description: str = "Returns a description of what you currently see with your eyes."
    robot: IRobot = None # type: ignore

    def __init__(self, robot: IRobot):
        super().__init__()
        self.robot = robot

    def _run(self) -> str:
        return self.robot.take_image(640, 480)

class EndConversationTool(BaseTool):
    name: str = "end_conversation"
    description: str = "Use this once the conversation has ended, so you can continue with your other tasks. For example if the user says 'bye'."
    on_end_conversation: Callable = None # type: ignore

    def __init__(self, on_end_conversation: Callable):
        super().__init__()
        self.on_end_conversation = on_end_conversation

    def _run(self) -> str:
        self.on_end_conversation()
        return "conversion ended, just say bye to the user"
