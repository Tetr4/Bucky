import threading
import requests
import time
from datetime import datetime
from pytz import timezone
from langchain_core.tools import tool

bucky_uri="http://bucky.local:5000"
meal_db_uri="https://www.themealdb.com"

@tool(parse_docstring=True)
def get_current_time(zone: str) -> str:
    """Returns the current time in the given locale as ISO 8601 string.

    Args:
        zone: The timezone, e.g. "US/Eastern"
    """
    return datetime.now().astimezone(timezone(zone)).isoformat()

@tool(parse_docstring=True)
def get_random_meal() -> str:
    """Returns a random meal as JSON from the meal database."""
    response = requests.get(f"{meal_db_uri}/api/json/v1/1/random.php")
    return response.json()

@tool(parse_docstring=True)
def search_meal_by_ingredient(ingredient: str) -> str:
    """Returns a meal as JSON from the meal database that contains the given ingredient.

    Args:
        ingredient: An ingredient in the meal. Must be in snake_case.
    """
    response = requests.get(f"{meal_db_uri}/api/json/v1/1/filter.php?i={ingredient}")
    return response.json()

def emote_happy() -> None:
    requests.get(f"{bucky_uri}/eyes/set_mood?mood=HAPPY")
    requests.get(f"{bucky_uri}/eyes/set_colors?main=FFFFFF")
    requests.get(f"{bucky_uri}/eyes/anim_laugh")
    time.sleep(0.5)
    requests.get(f"{bucky_uri}/eyes/set_colors?main=FFFFFF")
    requests.get(f"{bucky_uri}/eyes/set_mood?mood=NEUTRAL")

def emote_angry() -> None:
    requests.get(f"{bucky_uri}/eyes/set_mood?mood=ANGRY")
    requests.get(f"{bucky_uri}/eyes/set_colors?main=FF0000")
    requests.get(f"{bucky_uri}/eyes/anim_confused")
    time.sleep(0.5)
    requests.get(f"{bucky_uri}/eyes/set_colors?main=FFFFFF")
    requests.get(f"{bucky_uri}/eyes/set_mood?mood=NEUTRAL")

def emote_tired() -> None:
    requests.get(f"{bucky_uri}/eyes/set_mood?mood=TIRED")
    requests.get(f"{bucky_uri}/eyes/set_colors?main=0000FF")
    time.sleep(1)
    requests.get(f"{bucky_uri}/eyes/set_colors?main=FFFFFF")
    requests.get(f"{bucky_uri}/eyes/set_mood?mood=NEUTRAL")

@tool(parse_docstring=True)
def emote(emotion: str) -> None:
    """Use this to show an emotion on your face for a few seconds.

    Args:
        emotion: The emotion to show. Only thes values are allowed: "happy", "angry", "tired". Parameters must be in english.
    """
    if emotion == "happy":
        threading.Thread(target=emote_happy).start()
    elif emotion == "angry":
        threading.Thread(target=emote_angry).start()
    elif emotion == "tired":
        threading.Thread(target=emote_tired).start()
    else:
        raise ValueError("Invalid emotion. Only 'happy', 'angry', and 'tired' are allowed")
