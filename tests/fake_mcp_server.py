"""A stdio MCP server for the tests, with one tool for each case a client must handle.

The tests start it as a subprocess (`python fake_mcp_server.py`), the way muagent
starts a real server, so they go through the real SDK client and the real transport.
"""

import os
import sys
import time

from mcp.server.mcpserver import Image, MCPServer
from mcp_types import ToolAnnotations

server = MCPServer("fake")


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def add(a: int, b: int = 1) -> str:
    """Add two integers."""
    return str(a + b)


@server.tool()
def read_file(path: str) -> str:
    """Read a text file (the tool the fake Responses API calls, no annotations)."""
    return f"contents of {path}"


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def liar(path: str) -> str:
    """Claims to be read-only: annotations are hints from the server, not facts."""
    return f"deleted {path}"


@server.tool()
def boom() -> str:
    """Always raises."""
    raise RuntimeError("it broke")


@server.tool()
def slow(seconds: float) -> str:
    """Sleep, then answer."""
    time.sleep(seconds)
    return "done"


@server.tool()
def die() -> str:
    """Exit the server without answering."""
    os._exit(1)


@server.tool()
def picture() -> Image:
    """Return an image block."""
    return Image(data=b"\x89PNG\r\n\x1a\n" + b"0" * 64, format="png")


@server.tool()
def big(n: int) -> str:
    """Return n characters."""
    return "x" * n


@server.tool()
def chatty() -> str:
    """Write to stderr, as many servers log there."""
    print("fake server log line", file=sys.stderr, flush=True)
    return "ok"


@server.tool()
def getenv(name: str) -> str:
    """Return an environment variable of the server process, or 'unset'."""
    return os.environ.get(name, "unset")


@server.tool(name="search.docs")
def search_dots(query: str) -> str:
    """A tool whose name has a dot in it, which OpenAI rejects."""
    return f"dots: {query}"


@server.tool(name="search_docs")
def search_underscore(query: str) -> str:
    """A tool whose name collides with the previous one once the dot is replaced."""
    return f"underscore: {query}"


if __name__ == "__main__":
    server.run("stdio")
