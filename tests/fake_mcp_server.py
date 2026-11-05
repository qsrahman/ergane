"""A stdio MCP server for the tests, with one tool for each case a client must handle.

The tests start it as a subprocess, the way muagent starts a real server, so they go
through the real SDK client and the real transport: over stdio for `python
fake_mcp_server.py`, or over streamable HTTP on a port for `python fake_mcp_server.py
PORT`.
"""

import os
import sys
import time

from mcp.server.mcpserver import Context, Image, MCPServer
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


@server.tool()
def pid() -> str:
    """Return the process id of the server."""
    return str(os.getpid())


@server.tool(name="inner__tool")
def inner_tool() -> str:
    """With a server named `outer`, this comes to `outer__inner__tool`."""
    return "inner__tool"


@server.tool(name="tool")
def plain_tool() -> str:
    """With a server named `outer__inner`, this comes to `outer__inner__tool` too."""
    return "tool"


@server.tool()
def getcwd() -> str:
    """Return the working directory of the server process."""
    return os.getcwd()


@server.tool()
def header(name: str, ctx: Context) -> str:
    """Return a header of the HTTP request that carried this call."""
    request = ctx.request_context.request
    return request.headers.get(name, "unset") if request is not None else "no request"


@server.tool(name="search.docs")
def search_dots(query: str) -> str:
    """A tool whose name has a dot in it, which OpenAI rejects."""
    return f"dots: {query}"


@server.tool(name="search_docs")
def search_underscore(query: str) -> str:
    """A tool whose name collides with the previous one once the dot is replaced."""
    return f"underscore: {query}"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        server.run("streamable-http", host="127.0.0.1", port=int(sys.argv[1]))
    else:
        server.run("stdio")
