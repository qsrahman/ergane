"""MCP servers as tools: connect to a server over stdio and offer its tools to the agent.

The MCP SDK is async and the rest of muagent is not, so each server runs on a
background thread with its own event loop, and a call blocks until the loop answers.
The `mcp` package is imported only when a server starts, so muagent works without it
as long as no server is configured.
"""

import asyncio
import concurrent.futures
import json
import re
import threading
from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TextIO

from muagent.tools import Tool

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp_types import CallToolResult
    from mcp_types import Tool as ToolDefinition

# OpenAI rejects the whole request, not just the tool, for a name outside this.
_NAME_CHARS = re.compile(r"[a-zA-Z0-9_-]+")
_MAX_NAME_CHARS = 128

# How long a server gets to stop on its own before it is cancelled.
_STOP_SECONDS = 10


class McpServerError(Exception):
    """An MCP server could not be started, or is not running."""


def _tool_name(server: str, tool: str, taken: set[str]) -> str:
    """Name a server's tool for the model, unique among `taken` (which it extends).

    The server's name is a prefix, so servers cannot clash with each other or with
    muagent's own tools; characters the API rejects become underscores, and names
    that collide after that get a number.
    """
    base = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{server}__{tool}")[:_MAX_NAME_CHARS]
    name, number = base, 1
    while name in taken:
        number += 1
        suffix = f"_{number}"
        name = base[: _MAX_NAME_CHARS - len(suffix)] + suffix
    taken.add(name)
    return name


def _text(result: CallToolResult) -> str:
    """The text a tool result carries for the model.

    Blocks that are not text (an image, say) become a short placeholder.
    """
    parts = []
    for block in result.content:
        if block.type == "text":
            parts.append(block.text)
        else:
            mime_type = getattr(block, "mime_type", None)
            parts.append(
                f"[{block.type}: {mime_type}]" if mime_type else f"[{block.type}]"
            )
    text = "\n".join(parts)
    return f"Error: {text}" if result.is_error else text


def _validator(schema: Mapping[str, Any]) -> Callable[[str], str | None]:
    """Check what a call cannot work without: a JSON object with the required keys.

    Types are left to the server. A JSON Schema check would be stricter than the
    server: it rejects `"4"` for an integer, which servers accept.
    """
    required = schema.get("required", [])

    def validate(arguments: str) -> str | None:
        try:
            given = json.loads(arguments)
        except ValueError as exc:
            return f"Invalid arguments: Invalid JSON: {exc}"
        if not isinstance(given, dict):
            return "Invalid arguments: expected a JSON object"
        missing = [key for key in required if key not in given]
        if missing:
            return "Invalid arguments: " + "; ".join(
                f"{key}: Missing required argument" for key in missing
            )
        return None

    return validate


async def _list_tools(session: ClientSession) -> list[ToolDefinition]:
    """Every tool the server lists, across its pages."""
    from mcp_types import PaginatedRequestParams

    found: list[ToolDefinition] = []
    cursor = None
    while True:
        params = PaginatedRequestParams(cursor=cursor) if cursor else None
        page = await session.list_tools(params=params)
        found += page.tools
        cursor = page.next_cursor
        if cursor is None:
            return found


def _leaf(exc: BaseException) -> BaseException:
    """The exception inside the groups the SDK's task groups wrap it in."""
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return exc


class McpServer:
    """A connection to one MCP server over stdio, used from synchronous code."""

    def __init__(
        self,
        name: str,
        command: str,
        args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        *,
        errlog: TextIO,
        read_only: bool = False,
        startup_timeout: float = 20,
        call_timeout: float = 60,
    ) -> None:
        """Describe a server; `start()` connects to it.

        Args:
            name: Prefixes the names of its tools. Letters, digits, `_` and `-`.
            command: The program that runs the server.
            args: Its arguments.
            env: Environment variables for it, added to a short allow-list (PATH,
                HOME and the like): it does not inherit the rest, so it does not
                see this program's API key.
            errlog: Where the server's stderr goes, so that its log lines do not
                land in the terminal.
            read_only: Whether to treat its tools as read-only, so they run without
                asking. The server's own claim is never used: it is unverified.
            startup_timeout: Seconds to connect, and to list the tools.
            call_timeout: Seconds a call may take.

        Raises:
            ValueError: If the name has other characters.
        """
        if not _NAME_CHARS.fullmatch(name):
            raise ValueError(
                f"MCP server name {name!r} must be letters, digits, _ and - only"
            )
        self.name = name
        self._command = command
        self._args = list(args)
        self._env = dict(env) if env else None
        self._errlog = errlog
        self._read_only = read_only
        self._startup_timeout = startup_timeout
        self._call_timeout = call_timeout

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()  # set to let the task end the connection
        self._finished = threading.Event()  # set when the task has ended

    def start(self) -> None:
        """Start the server and connect to it. Call `close()` when done.

        Raises:
            McpServerError: If the server cannot be run, exits before it answers, or
                takes longer than the startup timeout.
        """
        loop = self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=loop.run_forever, name=f"mcp-{self.name}", daemon=True
        )
        self._thread.start()
        ready: concurrent.futures.Future[ClientSession] = concurrent.futures.Future()

        def launch() -> None:
            self._task = loop.create_task(self._serve(ready))

        loop.call_soon_threadsafe(launch)
        try:
            self._session = ready.result(self._startup_timeout)
        except BaseException as exc:
            self._close(cancel=True)
            if not isinstance(exc, Exception):
                raise  # Ctrl-C
            if isinstance(exc, TimeoutError):
                reason = f"did not answer within {self._startup_timeout:g} s"
            else:
                leaf = _leaf(exc)
                reason = f"failed to start: {type(leaf).__name__}: {leaf}"
            raise McpServerError(f"MCP server {self.name!r} {reason}") from None

    async def _serve(self, ready: concurrent.futures.Future[ClientSession]) -> None:
        """Own the connection for as long as it lasts.

        One task enters and leaves all of it: the SDK's task groups must be left by
        the task that entered them.
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self._command, args=self._args, env=self._env
        )
        try:
            async with (
                stdio_client(params, errlog=self._errlog) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                ready.set_result(session)
                await self._stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise
        finally:
            self._finished.set()

    def close(self) -> None:
        """Disconnect and stop the server and its thread. Safe to call twice."""
        self._close(cancel=False)

    def _close(self, cancel: bool) -> None:
        loop, thread, task = self._loop, self._thread, self._task
        if loop is None or thread is None:
            return
        self._loop = self._thread = self._session = self._task = None
        if cancel and task is not None:
            loop.call_soon_threadsafe(task.cancel)
        else:
            loop.call_soon_threadsafe(self._stop.set)
        # Unwinding stops the server process, so wait for it before the loop stops.
        if not self._finished.wait(_STOP_SECONDS) and task is not None:
            loop.call_soon_threadsafe(task.cancel)
            self._finished.wait(_STOP_SECONDS)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(_STOP_SECONDS)

    def tools(self) -> list[Tool]:
        """The server's tools, as tools for the agent.

        Raises:
            McpServerError: If the server is not running.
        """
        definitions = self._submit(_list_tools, self._startup_timeout)
        taken: set[str] = set()
        return [
            self._make_tool(_tool_name(self.name, d.name, taken), d)
            for d in definitions
        ]

    def _make_tool(self, name: str, definition: ToolDefinition) -> Tool:
        def function(**arguments: Any) -> str:
            return _text(
                self._submit(
                    lambda session: session.call_tool(definition.name, arguments),
                    self._call_timeout,
                )
            )

        def runner(arguments: str) -> str:
            return function(**json.loads(arguments))

        return Tool(
            name,
            definition.description or definition.title or "",
            function,
            definition.input_schema,
            _validator(definition.input_schema),
            runner,
            self._read_only,
        )

    def _submit[T](
        self,
        make: Callable[[ClientSession], Coroutine[Any, Any, T]],
        timeout: float,
    ) -> T:
        """Run a coroutine on the server's loop and wait for its result.

        Raises:
            McpServerError: If the server is not running.
            TimeoutError: If it takes longer than `timeout`; the request is cancelled
                and the connection stays usable.
            KeyboardInterrupt: On Ctrl-C, after cancelling the request.
        """
        loop, session = self._loop, self._session
        if loop is None or session is None:
            raise McpServerError(f"MCP server {self.name!r} is not running")
        future = asyncio.run_coroutine_threadsafe(make(session), loop)
        try:
            return future.result(timeout)
        except TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"MCP server {self.name!r} did not answer within {timeout:g} s"
            ) from None
        except KeyboardInterrupt:
            future.cancel()
            raise
