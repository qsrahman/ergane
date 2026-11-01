"""Placeholder tools, to exercise the tool loop until the real tools exist.

Delete this module, and its import in cli.py, when the real tools arrive.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from muagent.tools import tool


@tool
def add(a: int, b: int) -> str:
    """Add two integers and return the sum."""
    return str(a + b)


@tool
def get_time(timezone: str = "UTC") -> str:
    """Return the current date and time in an IANA time zone, such as Europe/Paris."""
    return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")
