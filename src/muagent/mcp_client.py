"""MCP servers as tools: connect to a server and offer its tools to the agent.

A server is a program muagent runs and talks to over stdio, or a server it connects
to over streamable HTTP.

The MCP SDK is async and the rest of muagent is not, so each server runs on a
background thread with its own event loop, and a call blocks until the loop answers.
The `mcp` package is imported only when a server starts, so muagent works without it
as long as no server is configured.
"""

import asyncio
import concurrent.futures
import importlib.util
import json
import os
import re
import threading
from collections.abc import Callable, Coroutine, Mapping
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, TextIO
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from muagent.tools import Tool, invalid_arguments

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp_types import CallToolResult
    from mcp_types import Tool as ToolDefinition

# OpenAI rejects the whole request, not just the tool, for a name outside this.
_ALLOWED = "a-zA-Z0-9_-"
NAME_PATTERN = rf"^[{_ALLOWED}]+$"  # anchored: pydantic searches, not matches
_NAME = re.compile(NAME_PATTERN)
_DISALLOWED = re.compile(rf"[^{_ALLOWED}]")
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
    base = _DISALLOWED.sub("_", f"{server}__{tool}")[:_MAX_NAME_CHARS]
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


def _required_keys_check(schema: Mapping[str, Any]) -> Callable[[str], str | None]:
    """Check what a call cannot work without: a JSON object with the required keys.

    Types are left to the server. A JSON Schema check would be stricter than the
    server: it rejects `"4"` for an integer, which servers accept.
    """
    required = schema.get("required", [])

    def check(arguments: str) -> str | None:
        try:
            given = json.loads(arguments)
        except ValueError as exc:
            return invalid_arguments([f"Invalid JSON: {exc}"])
        if not isinstance(given, dict):
            return invalid_arguments(["expected a JSON object"])
        missing = [key for key in required if key not in given]
        if missing:
            return invalid_arguments(
                f"{key}: Missing required argument" for key in missing
            )
        return None

    return check


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


# The hosts that may be reached without TLS: the traffic never leaves the machine.
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


def _check_url(url: str) -> None:
    """Refuse a server URL that would send what it carries where it can be read.

    Raises:
        ValueError: If it is not https (http only to this machine), names no host,
            or carries a user name or password, which belong in `headers`.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(
            f"url must be http(s)://host/path, not {parts.scheme or url!r}"
        )
    if parts.scheme == "http" and parts.hostname not in _LOOPBACK_HOSTS:
        raise ValueError(
            "url must be https, except for localhost: headers and tool calls would "
            "cross the network unencrypted"
        )
    if parts.username or parts.password:
        raise ValueError("url must not hold a user name or password: use headers")


class McpServerConfig(BaseModel):
    """One server in config.json: a program to run (`command`) or a `url` to connect to."""

    model_config = ConfigDict(extra="forbid")

    # What kind of server it is. Optional, since command or url says it already, but
    # if given it must agree with them: a mistake is better caught than ignored.
    type: Literal["local", "remote"] | None = None
    enable: bool = True  # False keeps the entry but does not start the server
    command: str | None = Field(default=None, min_length=1)
    args: list[str] = []
    env: dict[str, str] = {}
    url: str | None = None
    headers: dict[str, str] = {}  # e.g. Authorization, for a url
    # Trust the server's tools to run without asking. Off unless the user says so.
    read_only: bool = False
    # Offer only these tools of the server (by the server's own names). Every tool's
    # definition is sent with every request, so a server with many tools costs a
    # good part of the context budget whether or not the model uses them.
    tools: list[str] | None = Field(default=None, min_length=1)
    startup_timeout: float = Field(default=20, gt=0)  # seconds
    call_timeout: float = Field(default=60, gt=0)  # seconds

    @model_validator(mode="after")
    def _one_transport(self) -> Self:
        if (self.command is None) == (self.url is None):
            raise ValueError(
                "give either command (a program to run) or url (a server to connect to)"
            )
        kind = "local" if self.url is None else "remote"
        if self.type not in (None, kind):
            raise ValueError(
                f"type is {self.type!r}, but a server with a "
                f"{'command' if self.url is None else 'url'} is {kind!r}"
            )
        if self.url is None:
            if self.headers:
                raise ValueError("headers go with url")
            return self
        if self.args or self.env:
            raise ValueError("args and env go with command")
        _check_url(self.url)
        return self


class McpServer:
    """A connection to one MCP server, used from synchronous code.

    The server is a program run by `command` and spoken to over stdio, or a `url`
    spoken to over streamable HTTP; exactly one of them.
    """

    def __init__(self, name: str, config: McpServerConfig, log: Path) -> None:
        """Describe a server; `start()` connects to it.

        A McpServer is used once: after `close()` it cannot be started again.

        Args:
            name: Prefixes the names of its tools. Letters, digits, `_` and `-`.
            config: What to run or where to connect, and how far to trust it. Its
                `${NAME}` placeholders must be filled in already (see `_expand`).
            log: For a program, the file for its stderr, so that its log lines do not
                land in the terminal. Opened (and emptied) by `launch()`, closed by
                `close()`. A url has no stderr, and the file is not touched.

        Raises:
            ValueError: If the name has other characters.
        """
        if not _NAME.fullmatch(name):
            raise ValueError(
                f"MCP server name {name!r} must be letters, digits, _ and - only"
            )
        self.name = name
        self._config = config
        self._log_path = log
        self._log: TextIO | None = None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._task: asyncio.Task[None] | None = None
        self._ready: concurrent.futures.Future[ClientSession] | None = None
        self._stop = asyncio.Event()  # set to let the task end the connection
        self._finished = threading.Event()  # set when the task has ended

    def start(self) -> None:
        """Start the server and connect to it. Call `close()` when done.

        Raises:
            McpServerError: See `launch()` and `connect()`.
        """
        self.launch()
        self.connect()

    def launch(self) -> None:
        """Begin starting the server without waiting for it; `connect()` waits.

        Launching several servers before connecting to any lets them start at the
        same time.

        Raises:
            McpServerError: If the `mcp` package is missing or the log cannot be
                opened.
        """
        if importlib.util.find_spec("mcp") is None:
            raise McpServerError(
                f"MCP server {self.name!r} needs the mcp package: "
                "install muagent with its mcp extra"
            )
        log = None
        if self._config.command is not None:
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log = log = self._log_path.open("w")
            except OSError as exc:
                raise McpServerError(
                    f"MCP server {self.name!r}: cannot write its log: {exc}"
                ) from None
        loop = self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=loop.run_forever, name=f"mcp-{self.name}", daemon=True
        )
        self._thread.start()
        ready: concurrent.futures.Future[ClientSession] = concurrent.futures.Future()
        self._ready = ready

        def create_task() -> None:
            self._task = loop.create_task(self._serve(ready, log))
            self._task.add_done_callback(self._task_done)

        loop.call_soon_threadsafe(create_task)

    def connect(self) -> None:
        """Wait until a launched server is up.

        Raises:
            McpServerError: If the server cannot be run, exits before it answers, or
                takes longer than the startup timeout.
        """
        if self._ready is None:
            raise McpServerError(f"MCP server {self.name!r} was not launched")
        try:
            self._session = self._ready.result(self._config.startup_timeout)
        except BaseException as exc:
            self._close(cancel=True)
            if not isinstance(exc, Exception):
                raise  # Ctrl-C
            if isinstance(exc, TimeoutError):
                reason = f"did not answer within {self._config.startup_timeout:g} s"
            else:
                leaf = _leaf(exc)
                reason = f"failed to start: {type(leaf).__name__}: {leaf}"
            raise McpServerError(f"MCP server {self.name!r} {reason}") from None

    def _task_done(self, task: asyncio.Task[None]) -> None:
        """Note that the connection's task has ended, however it ended.

        This is here, not in `_serve`: a task cancelled before its first step never
        runs any of its code.
        """
        if not task.cancelled():
            # A server that dies ends the task with an error. The caller has it,
            # as the error of the call that failed, so asyncio need not log it.
            task.exception()
        self._finished.set()

    async def _serve(
        self, ready: concurrent.futures.Future[ClientSession], log: TextIO | None
    ) -> None:
        """Own the connection for as long as it lasts.

        One task enters and leaves all of it: the SDK's task groups must be left by
        the task that entered them.
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client

        config = self._config
        try:
            async with AsyncExitStack() as stack:
                if config.url is None:
                    assert config.command is not None and log is not None
                    params = StdioServerParameters(
                        command=config.command,
                        args=config.args,
                        env=config.env,
                    )
                    read, write = await stack.enter_async_context(
                        stdio_client(params, errlog=log)
                    )
                else:
                    import httpx2

                    # Redirects are followed only within the url's origin, so
                    # the headers go nowhere else.
                    # The SDK's own timeouts: a server may hold a response open.
                    http = await stack.enter_async_context(
                        httpx2.AsyncClient(
                            headers=config.headers,
                            timeout=httpx2.Timeout(30.0, read=300.0),
                        )
                    )
                    read, write = await stack.enter_async_context(
                        streamable_http_client(config.url, http_client=http)
                    )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                ready.set_result(session)
                await self._stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise

    def close(self) -> None:
        """Disconnect and stop the server and its thread. Safe to call twice."""
        # A server that is still starting cannot be asked to stop; it is cancelled.
        self._close(cancel=self._session is None)

    def _close(self, cancel: bool) -> None:
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        self._loop = self._thread = self._session = None

        def cancel_task() -> None:
            # Run on the loop, after the callback that creates the task.
            if self._task is not None:
                self._task.cancel()

        if cancel:
            loop.call_soon_threadsafe(cancel_task)
        else:
            loop.call_soon_threadsafe(self._stop.set)
        # Unwinding stops the server process, so wait for it before the loop stops.
        if not self._finished.wait(_STOP_SECONDS):
            loop.call_soon_threadsafe(cancel_task)
            self._finished.wait(_STOP_SECONDS)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(_STOP_SECONDS)
        if not thread.is_alive():
            loop.close()  # or its selector and sockets stay open
        self._task = self._ready = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def tools(self) -> list[Tool]:
        """The server's tools, as tools for the agent: all of them, or the ones the
        config lists.

        Raises:
            McpServerError: If the server is not running, or the config lists a
                tool it does not have (a typo would otherwise offer nothing, silently).
        """
        definitions = self._submit(_list_tools, self._config.startup_timeout)
        if self._config.tools is not None:
            have = {d.name for d in definitions}
            unknown = [name for name in self._config.tools if name not in have]
            if unknown:
                raise McpServerError(
                    f"MCP server {self.name!r} has no tool {', '.join(unknown)} "
                    f"(it has: {', '.join(sorted(have))})"
                )
            definitions = [d for d in definitions if d.name in self._config.tools]
        taken: set[str] = set()
        return [
            self._make_tool(_tool_name(self.name, d.name, taken), d)
            for d in definitions
        ]

    def _make_tool(self, name: str, definition: ToolDefinition) -> Tool:
        def runner(arguments: str) -> str:
            return _text(
                self._submit(
                    lambda session: session.call_tool(
                        definition.name, json.loads(arguments)
                    ),
                    self._config.call_timeout,
                )
            )

        return Tool(
            name,
            definition.description or definition.title or "",
            definition.input_schema,
            _required_keys_check(definition.input_schema),
            runner,
            self._config.read_only,
        )

    def _submit[T](
        self,
        make: Callable[[ClientSession], Coroutine[Any, Any, T]],
        timeout: float,
    ) -> T:
        """Run a coroutine on the server's loop and wait for its result.

        Raises:
            McpServerError: If the server is not running, or fails the request (it
                has stopped, say).
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
        except Exception as exc:  # noqa: BLE001
            leaf = _leaf(exc)
            raise McpServerError(
                f"MCP server {self.name!r}: {type(leaf).__name__}: {leaf}"
            ) from None


_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class UnsetVariableError(ValueError):
    """A `${NAME}` placeholder names a variable that is not set."""

    def __init__(self, name: str) -> None:
        super().__init__(f"${{{name}}} is not set")
        self.name = name


def fill_placeholders(text: str, environ: Mapping[str, str]) -> str:
    """Fill in the `${NAME}` placeholders of a text from the environment.

    One pass: the text a variable holds is not looked at again. `$NAME` and a `${`
    without a name are plain text.

    Raises:
        UnsetVariableError: If a variable is not set. It names the variable, never a
            value.
    """

    def value(match: re.Match[str]) -> str:
        variable = match.group(1)
        if variable not in environ:
            raise UnsetVariableError(variable)
        return environ[variable]

    return _PLACEHOLDER.sub(value, text)


def _expand(
    name: str, config: McpServerConfig, environ: Mapping[str, str]
) -> McpServerConfig:
    """Fill in the `${NAME}` placeholders of a server from the environment.

    Only in what a server is given to run with: `args`, `env` and `headers`.
    This keeps keys out of the file, and a server gets only the variables the file
    names. The text a variable holds is not looked at again for placeholders.

    Raises:
        McpServerError: If a variable is not set. It names the variable and where it
            is used, never a value.
    """

    def fill(text: str, where: str) -> str:
        try:
            return fill_placeholders(text, environ)
        except UnsetVariableError as exc:
            raise McpServerError(
                f"MCP server {name!r}: {where} uses ${{{exc.name}}}, which is not set"
            ) from None

    return config.model_copy(
        update={
            "args": [fill(arg, "args") for arg in config.args],
            "env": {k: fill(v, f"env.{k}") for k, v in config.env.items()},
            "headers": {k: fill(v, f"headers.{k}") for k, v in config.headers.items()},
        }
    )


def start_servers(
    configs: Mapping[str, McpServerConfig],
    log_dir: Path,
    on_notice: Callable[[str], None],
    *,
    environ: Mapping[str, str] = os.environ,
) -> tuple[list[McpServer], list[Tool]]:
    """Start the servers, all at once, and collect their tools.

    A server that fails is reported and left out: it must not stop muagent. One
    that is switched off (`enable` false) is left out without a word, and so is
    one that starts: the banner counts its tools.

    Args:
        configs: The servers to start, by name.
        log_dir: Where each server's stderr goes, in a file named after it.
        on_notice: Called with a line about each server that fails.
        environ: Where the `${NAME}` placeholders of the servers are filled in from.

    Returns:
        The servers that are running, which the caller must close, and their tools.
    """
    # Each server, or why it cannot be started, in the order of the file.
    entries: list[tuple[str, McpServer | str]] = []
    try:
        for name, config in configs.items():
            if not config.enable:
                continue
            try:
                server = McpServer(
                    name, _expand(name, config, environ), log_dir / f"{name}.log"
                )
                server.launch()  # they all start at once; connect() waits below
            except McpServerError as exc:
                entries.append((name, str(exc)))
            else:
                entries.append((name, server))

        running: list[McpServer] = []
        tools: list[Tool] = []
        for name, entry in entries:
            if isinstance(entry, str):
                on_notice(entry)
                continue
            try:
                entry.connect()
                found = entry.tools()
            except Exception as exc:  # noqa: BLE001
                entry.close()
                on_notice(
                    str(exc)
                    if isinstance(exc, McpServerError)
                    else f"MCP server {name!r} failed: {type(exc).__name__}: {exc}"
                )
                continue
            running.append(entry)
            tools += found
    except BaseException:
        for _, entry in entries:
            if not isinstance(entry, str):
                entry.close()
        raise
    return running, tools
