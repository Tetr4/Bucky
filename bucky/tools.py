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
