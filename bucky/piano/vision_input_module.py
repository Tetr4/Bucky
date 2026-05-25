import time
from typing import Callable, Optional
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from bucky.piano.perception import PerceptionInput, PerceptionModule, PerceptionType


class VisionInputModule(PerceptionModule):

    def __init__(self, vision_model: BaseChatModel, get_base64_image_func: Callable[[], str]):
        self._vision_model = vision_model
        self._get_base64_image_func = get_base64_image_func

    def get_input(self) -> Optional[PerceptionInput]:
        system_prompt: str = """You are the visual cortex of a person. Provide a clear, concise visual description focused on observable details and emotional cues."""
        image_base64 = self._get_base64_image_func()
        content = [{
            "type": "text",
            "text": (
                    "Describe the image briefly and objectively. Include: visible objects and people, actions, scene/setting, colors, and notable details. "
                    "If people are present, describe observable mood, facial expressions, posture, approximate count, and any apparent relationships or interactions. "
                    "Avoid speculation about unobservable facts (e.g., names, thoughts). Provide confidence level for uncertain observations. Keep the response concise."
            )
        }, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]

        input: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=content)]
        response: BaseMessage = self._vision_model.invoke(input)  # type: ignore
        if response.content and isinstance(response.content, str):
            return PerceptionInput(type=PerceptionType.VISION,
                                   content=response.content,
                                   base_priority=5,
                                   timestamp=time.time())
        return None
