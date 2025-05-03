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
    robot: IRobot = None  # type: ignore

    def __init__(self, robot: IRobot):
        super().__init__()
        self.robot = robot

    def _run(self) -> list:
        image_base64 = self.robot.take_image(640, 480)
        return [{"type": "text", "text": "das siehst du gerade vor dir"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]


class EndConversationTool(BaseTool):
    name: str = "end_conversation"
    description: str = "Use this tool when the user does not want to talk to you anymore. For example if the user says 'bye'."
    on_end_conversation: Callable = None  # type: ignore

    def __init__(self, on_end_conversation: Callable):
        super().__init__()
        self.on_end_conversation = on_end_conversation

    def _run(self) -> str:
        self.on_end_conversation()
        return "conversion ended, say bye in just a few words"
