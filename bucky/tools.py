import base64
import httpx
import threading
import requests
import time
from datetime import datetime
from pytz import timezone
from langchain_core.tools import tool
import bucky.config as cfg

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

def emote_happy() -> None:
    requests.get(f"{cfg.bucky_uri}/eyes/set_mood?mood=HAPPY")
    requests.get(f"{cfg.bucky_uri}/eyes/set_colors?main=FFFFFF")
    requests.get(f"{cfg.bucky_uri}/eyes/anim_laugh")
    time.sleep(2)
    requests.get(f"{cfg.bucky_uri}/eyes/set_colors?main=FFFFFF")
    requests.get(f"{cfg.bucky_uri}/eyes/set_mood?mood=NEUTRAL")

def emote_angry() -> None:
    requests.get(f"{cfg.bucky_uri}/eyes/set_mood?mood=ANGRY")
    requests.get(f"{cfg.bucky_uri}/eyes/set_colors?main=FF0000")
    requests.get(f"{cfg.bucky_uri}/eyes/anim_confused")
    time.sleep(2)
    requests.get(f"{cfg.bucky_uri}/eyes/set_colors?main=FFFFFF")
    requests.get(f"{cfg.bucky_uri}/eyes/set_mood?mood=NEUTRAL")

def emote_tired() -> None:
    requests.get(f"{cfg.bucky_uri}/eyes/set_mood?mood=TIRED")
    requests.get(f"{cfg.bucky_uri}/eyes/set_colors?main=0000FF")
    time.sleep(2)
    requests.get(f"{cfg.bucky_uri}/eyes/set_colors?main=FFFFFF")
    requests.get(f"{cfg.bucky_uri}/eyes/set_mood?mood=NEUTRAL")

def emote_idle() -> None:
    requests.get(f"{cfg.bucky_uri}/eyes/set_height?left=120&right=120")
    requests.get(f"{cfg.bucky_uri}/eyes/set_width?left=90&right=90")
    requests.get(f"{cfg.bucky_uri}/eyes/set_idlemode?on=true")

def emote_attention() -> None:
    requests.get(f"{cfg.bucky_uri}/eyes/set_height?left=150&right=150")
    requests.get(f"{cfg.bucky_uri}/eyes/set_widtht?left=95&right=95")
    requests.get(f"{cfg.bucky_uri}/eyes/set_idlemode?on=false")
    requests.get(f"{cfg.bucky_uri}/eyes/set_position?position=CENTER")

@tool(parse_docstring=True)
def emote(emotion: str) -> None:
    """Use this to show an emotion on your face for a few seconds.

    Args:
        emotion: The emotion to show. Only these values are allowed: "happy", "angry", "tired". Parameters must be in english.
    """
    if emotion == "happy":
        threading.Thread(target=emote_happy).start()
    elif emotion == "angry":
        threading.Thread(target=emote_angry).start()
    elif emotion == "tired":
        threading.Thread(target=emote_tired).start()
    else:
        raise ValueError("Invalid emotion. Only 'happy', 'angry', and 'tired' are allowed")

@tool(parse_docstring=True)
def take_image() -> str:
    """Returns a description of what you see."""
    bytes = httpx.get(f"{cfg.bucky_uri}/cam/still?width=640&height=480").content
    return base64.b64encode(bytes).decode("utf-8")
