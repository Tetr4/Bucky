import requests
from datetime import datetime
from pytz import timezone
from langchain_core.tools import tool

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
    url = "https://www.themealdb.com/api/json/v1/1/random.php"
    response = requests.get(url)
    return response.json()

@tool(parse_docstring=True)
def search_meal_by_ingredient(ingredient: str) -> str:
    """Returns a meal as JSON from the meal database that contains the given ingredient.

    Args:
        ingredient: An ingredient in the meal.
    """
    url = f"https://www.themealdb.com/api/json/v1/1/filter.php?i={ingredient}"
    response = requests.get(url)
    return response.json()
