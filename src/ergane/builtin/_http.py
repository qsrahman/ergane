"""What the web tools say about themselves to the sites they ask."""

from importlib.metadata import version

# Honest on purpose: DuckDuckGo answers an agent that says what it is, and turns away one
# that wears a browser's headers (see `search.py`).
USER_AGENT = f"ergane/{version('ergane')} (Autonomous)"
