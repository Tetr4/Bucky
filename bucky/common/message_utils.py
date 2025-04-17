from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, ToolCall


def is_message_for_tool_call(message: BaseMessage, tool_call: ToolCall) -> bool:
    return isinstance(message, ToolMessage) and message.tool_call_id == tool_call['id']


def has_image_data(message: BaseMessage) -> bool:
    if not isinstance(message.content, list):
        return False
    return any(content_item.get('type') == 'image_url' for content_item in message.content if isinstance(content_item, dict))


class ContentType(Enum):
    UNKNOWN = auto()
    TEXT = auto()
    IMAGE = auto()
    TOOL_CALL = auto()
    TOOL_STATUS = auto()


@dataclass
class MessageContent:
    type: ContentType
    data: Any


def resolve_content(message: BaseMessage) -> list[MessageContent]:
    result: list[MessageContent] = []
    if isinstance(message.content, str):
        result.append(MessageContent(type=ContentType.TEXT, data=message.content))
    elif isinstance(message.content, list):
        for content_item in message.content:
            if isinstance(content_item, dict):
                if content_item.get('type') == 'image_url':
                    img_url = content_item.get("image_url", {}).get("url")
                    result.append(MessageContent(type=ContentType.IMAGE, data=img_url))
                elif content_item.get('type') == 'text':
                    result.append(MessageContent(type=ContentType.TEXT, data=content_item.get("text")))
            elif isinstance(content_item, str):
                result.append(MessageContent(type=ContentType.TEXT, data=content_item))
            else:
                result.append(MessageContent(type=ContentType.UNKNOWN, data=content_item))

    elif message.content:
        result.append(MessageContent(type=ContentType.UNKNOWN, data=message.content))

    if isinstance(message, AIMessage):
        for tool_call in message.tool_calls:
            result.append(MessageContent(type=ContentType.TOOL_CALL, data=tool_call))

    if isinstance(message, ToolMessage):
        result.append(MessageContent(type=ContentType.TOOL_STATUS, data={"name": message.name,
                                                                         "call_id": message.tool_call_id,
                                                                         "status": message.status}))

    return result


def get_image_data(message: BaseMessage) -> list[str]:
    result = []
    if not isinstance(message.content, list):
        return result
    for content_item in message.content:
        if isinstance(content_item, dict) and content_item.get('type') == 'image_url':
            img_url = content_item.get("image_url", {}).get("url")
            if img_url:
                result.append(img_url)
    return result
