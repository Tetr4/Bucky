import requests
from langchain_core.tools import tool
import bucky.config as cfg


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
