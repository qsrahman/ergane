"""MCP servers as tools: connect to a server and offer its tools to the agent.

A server is a program muagent runs and talks to over stdio, or a server it connects
to over streamable HTTP.

The MCP SDK is async and the rest of muagent is not, so each server runs on a
background thread with its own event loop, and a call blocks until the loop answers.
The `mcp` package is imported only when a server starts, so muagent works without it
as long as no server is configured.

Design notes, for whoever changes this:

- One task per server (`_serve`) enters and leaves the whole connection: the SDK's task
  groups must be left by the task that entered them. Its end is noticed by a
  done-callback (`_task_done`), not by code in `_serve`, because a task cancelled before
  its first step never runs any of its code; the callback also reads the task's
  exception, or asyncio logs it to the terminal. `_close` cancels through a callback
  run on the loop after the one that creates the task, for the same reason.
- Starting: `launch()` every server, then `connect()` each. The startup timeout counts
  from `launch()`, so servers that never answer delay the start by one timeout. A
  failed `connect()` leaves the server for `close()` (stopping takes seconds: the SDK
  waits 2 s for a program that ignores its closed input); `start_servers` closes the
  failed ones together, `close_servers` the running ones.
- A server whose connection ended is started again by the next call (`_submit`), up to
  `_MAX_RESTARTS` times; the call that found it dead fails and is not repeated. After
  `_IDLE_SECONDS` of silence a call first pings, because a program that exits while the
  task is only waiting is not noticed otherwise. Only an answer that says the
  connection is closed counts: a busy server is not gone.
- Names are `server__tool` with what the API rejects replaced by `_`, and made unique
  across the servers (`a` with `b__c` and `a__b` with `c` collide). A name the API
  rejects fails the whole request.
- Trust is `read_only` for a whole server, from the user's file. The server's own
  `readOnlyHint` is never used: a server can claim anything.
- A tool's `check` only requires the keys the schema requires: a JSON Schema check is
  stricter than servers, which accept `"4"` for an integer, and the server's own errors
  come back as text for the model.
- A program does not inherit the environment, only the SDK's short allow-list plus its
  `env`, so it never sees `OPENAI_API_KEY`; it starts in muagent's directory (there is
  no `cwd` setting); its stderr goes to a log file, or it would land in the terminal.
  A url must be https unless it is loopback, and may not hold credentials; the SDK
  follows redirects only within the url's origin, so the headers go nowhere else.
- A server that announces new tools (`tools/list_changed`) only sets a flag, from its
  own thread. `McpHub.refresh` lists the flagged servers again and `McpHub.reconnect`
  starts one again, and both return a `ServerUpdate` instead of touching the agent: the
  caller applies it between turns, because the tools of a turn must not change under
  it, and what the server counted for the last request no longer holds. A server that
  started again is flagged too, as it may offer other tools.
- A server's `instructions` (what it says about using its tools) go into the model's
  instructions, framed as notes by the server and not as orders (`SERVER_NOTES`), cut
  to `_MAX_INSTRUCTION_CHARS`, and given only for a server whose config says
  `instructions: true`. They are as much the server's words as its tool descriptions,
  which the model reads already; the frame is for what a server might add.
"""

import asyncio
import concurrent.futures
import functools
import importlib.util
import json
import os
import re
import threading
import time
from collections.abc import Callable, Coroutine, Iterable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from muagent.mcp_config import (
    NAME_CHARS,
    NAME_PATTERN,
    McpServerConfig,
    UnsetVariableError,
    fill_placeholders,
)
from muagent.tools import Tool, invalid_arguments

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp_types import CallToolResult
    from mcp_types import Tool as ToolDefinition

# OpenAI rejects the whole request, not just the tool, for a name outside this.
_NAME = re.compile(NAME_PATTERN)
_DISALLOWED = re.compile(rf"[^{NAME_CHARS}]")
_MAX_NAME_CHARS = 128

# How long a server gets to stop on its own before it is cancelled.
_STOP_SECONDS = 10

# After a connection has been idle this long, a call first asks the server if it is
# still there: the connection's task does not notice a server that left meanwhile.
_IDLE_SECONDS = 10
_PING_SECONDS = 5

# The most of a server's instructions that the model is given: they are in every
# request, and a server may write pages.
_MAX_INSTRUCTION_CHARS = 4000

# How often a server whose connection was lost is started again in one session. A
# server that dies at once would otherwise be started at every call.
_MAX_RESTARTS = 3


@dataclass(frozen=True)
class ServerStatus:
    """What became of one server in config.json when muagent started, for `/servers`.

    A server that failed is reported once, as a notice, and would be forgotten.
    """

    name: str
    state: str  # "running", "failed" or "off"
    detail: str = ""  # why it failed, or why it is off
    tools: tuple[str, ...] = ()  # the names of the tools it offers the model
    hidden: int = 0  # tools it has that its `tools` setting leaves out
    log: Path | None = None  # where its stderr went, for a program


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

    Blocks that are not text (an image, say) become a short placeholder. A text file
    the server embedded is text; a result with only structured content is that,
    as JSON.
    """
    from mcp_types import (
        EmbeddedResource,
        ResourceLink,
        TextContent,
        TextResourceContents,
    )

    parts = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, EmbeddedResource):
            resource = block.resource  # a text file the server embedded is text
            parts.append(
                resource.text
                if isinstance(resource, TextResourceContents)
                else f"[resource: {resource.uri}]"
            )
        elif isinstance(block, ResourceLink):
            parts.append(f"[resource_link: {block.uri}]")
        else:
            mime_type = getattr(block, "mime_type", None)
            parts.append(
                f"[{block.type}: {mime_type}]" if mime_type else f"[{block.type}]"
            )
    if not parts and result.structured_content:
        # The spec asks for text next to it, but a server may send only this.
        parts.append(json.dumps(result.structured_content))
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
        # Each launch has its own: the previous connection's can still fire late.
        self._stop = asyncio.Event()  # set to let the task end the connection
        self._finished = threading.Event()  # set when the task has ended
        self._deadline = 0.0  # when starting must be done, counted from launch()
        self._started = False  # the connection has been up at least once
        self._closed = False  # close() was called: do not start it again
        self._restarts = 0
        self._active = 0.0  # when the server last answered
        self.hidden = 0  # tools it has that the config leaves out, after tools()
        # Set by the server's notification that its tools changed; cleared when they
        # are listed. Set from the server's thread, read from the main one.
        self._tools_changed = threading.Event()
        # What the server says about using its tools, once connected.
        self.instructions: str | None = None

    def start(self) -> None:
        """Start the server and connect to it. Call `close()` when done.

        Raises:
            McpServerError: See `launch()` and `connect()`.
        """
        self.launch()
        try:
            self.connect()
        except BaseException:
            self.close()
            raise

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
                # A restart adds to the log: why the server stopped is in it.
                self._log = log = self._log_path.open("a" if self._restarts else "w")
            except OSError as exc:
                raise McpServerError(
                    f"MCP server {self.name!r}: cannot write its log: {exc}"
                ) from None
        stop = self._stop = asyncio.Event()
        finished = self._finished = threading.Event()
        self._tools_changed.clear()
        # The time to start is counted from here, so that servers launched together
        # are waited for together, not one timeout after another.
        self._deadline = time.monotonic() + self._config.startup_timeout
        loop = self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=loop.run_forever, name=f"mcp-{self.name}", daemon=True
        )
        self._thread.start()
        ready: concurrent.futures.Future[ClientSession] = concurrent.futures.Future()
        self._ready = ready

        def create_task() -> None:
            self._task = loop.create_task(self._serve(ready, log, stop))
            self._task.add_done_callback(functools.partial(self._task_done, finished))

        loop.call_soon_threadsafe(create_task)

    def connect(self) -> None:
        """Wait until a launched server is up, for what is left of its startup timeout.

        If it fails, the server is left for `close()` to clean up: that takes a few
        seconds, and servers that failed together can be closed together.

        Raises:
            McpServerError: If the server cannot be run, exits before it answers, or
                takes longer than the startup timeout.
        """
        if self._ready is None:
            raise McpServerError(f"MCP server {self.name!r} was not launched")
        try:
            remaining = max(0.0, self._deadline - time.monotonic())
            self._session = self._ready.result(remaining)
            self._started = True
            self._active = time.monotonic()
            text = (self._session.instructions or "").strip()
            if len(text) > _MAX_INSTRUCTION_CHARS:
                text = text[:_MAX_INSTRUCTION_CHARS] + "…"
            self.instructions = text or None
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise  # Ctrl-C
            if isinstance(exc, TimeoutError):
                reason = f"did not answer within {self._config.startup_timeout:g} s"
            else:
                leaf = _leaf(exc)
                reason = f"failed to start: {type(leaf).__name__}: {leaf}"
            raise McpServerError(f"MCP server {self.name!r} {reason}") from None

    @staticmethod
    def _task_done(finished: threading.Event, task: asyncio.Task[None]) -> None:
        """Note that the connection's task has ended, however it ended.

        This is here, not in `_serve`: a task cancelled before its first step never
        runs any of its code.
        """
        if not task.cancelled():
            # A server that dies ends the task with an error. The caller has it,
            # as the error of the call that failed, so asyncio need not log it.
            task.exception()
        finished.set()

    async def _serve(
        self,
        ready: concurrent.futures.Future[ClientSession],
        log: TextIO | None,
        stop: asyncio.Event,
    ) -> None:
        """Own the connection for as long as it lasts.

        One task enters and leaves all of it: the SDK's task groups must be left by
        the task that entered them.
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client
        from mcp_types import Implementation, ToolListChangedNotification

        config = self._config
        changed = self._tools_changed

        async def on_message(message: Any) -> None:
            if isinstance(message, ToolListChangedNotification):
                changed.set()

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
                # Say who is calling: a server logs it, and some adapt to the client.
                session = await stack.enter_async_context(
                    ClientSession(
                        read,
                        write,
                        message_handler=on_message,
                        client_info=Implementation(
                            name="muagent", version=version("muagent")
                        ),
                    )
                )
                await session.initialize()
                ready.set_result(session)
                await stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise

    def close(self) -> None:
        """Disconnect and stop the server and its thread. Safe to call twice."""
        # A server that is still starting cannot be asked to stop; it is cancelled.
        self._closed = True
        self._close(cancel=self._session is None)

    def _close(self, cancel: bool) -> None:
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        stop, finished = self._stop, self._finished
        self._loop = self._thread = self._session = None

        def cancel_task() -> None:
            # Run on the loop, after the callback that creates the task.
            if self._task is not None:
                self._task.cancel()

        if cancel:
            loop.call_soon_threadsafe(cancel_task)
        else:
            loop.call_soon_threadsafe(stop.set)
        # Unwinding stops the server process, so wait for it before the loop stops.
        if not finished.wait(_STOP_SECONDS):
            loop.call_soon_threadsafe(cancel_task)
            finished.wait(_STOP_SECONDS)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(_STOP_SECONDS)
        if not thread.is_alive():
            loop.close()  # or its selector and sockets stay open
        self._task = self._ready = None
        if self._log is not None:
            self._log.close()
            self._log = None

    @property
    def tools_changed(self) -> bool:
        """Whether the server said its tools changed since they were last listed."""
        return self._tools_changed.is_set()

    def _lost(self) -> bool:
        """Whether the connection was up, and has ended without `close()`."""
        return (
            self._started
            and not self._closed
            and (self._session is None or self._finished.is_set())
        )

    def _restart(self) -> None:
        """Start the server again after its connection was lost.

        Raises:
            McpServerError: If it was restarted `_MAX_RESTARTS` times already, or
                does not start.
        """
        if self._restarts >= _MAX_RESTARTS:
            raise McpServerError(
                f"MCP server {self.name!r} stopped again after {_MAX_RESTARTS} "
                "restarts, and is not started any more"
            )
        self._restarts += 1
        self._close(cancel=True)  # what is left of the old connection
        try:
            self.launch()
            self.connect()
            self._tools_changed.set()  # a program that started again may offer others
        except McpServerError as exc:
            self._close(cancel=True)
            raise McpServerError(f"{exc} (restarting it after it stopped)") from None

    def tools(self, taken: set[str] | None = None) -> list[Tool]:
        """The server's tools, as tools for the agent: all of them, or the ones the
        config lists.

        Args:
            taken: The names already given to other servers' tools, which these
                must not repeat, and which they are added to. Names are
                `server__tool`, but `a` with `b__c` and `a__b` with `c` both come
                to `a__b__c`.

        Raises:
            McpServerError: If the server is not running, or the config lists a
                tool it does not have (a typo would otherwise offer nothing, silently).
        """
        self._tools_changed.clear()  # a change announced while listing sets it again
        definitions = self._submit(_list_tools, self._config.startup_timeout)
        if self._config.tools is not None:
            have = {d.name for d in definitions}
            unknown = [name for name in self._config.tools if name not in have]
            if unknown:
                raise McpServerError(
                    f"MCP server {self.name!r} has no tool {', '.join(unknown)} "
                    f"(it has: {', '.join(sorted(have))})"
                )
            kept = [d for d in definitions if d.name in self._config.tools]
            self.hidden = len(definitions) - len(kept)
            definitions = kept
        taken = set() if taken is None else taken
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

        A server whose connection was lost (it exited, or a remote one went away) is
        started again first, so the call after the one that failed works. The call that
        failed is not repeated: it may have done part of its work.

        Raises:
            McpServerError: If the server is not running and cannot be started again,
                or fails the request (it has stopped, say).
            TimeoutError: If it takes longer than `timeout`; the request is cancelled
                and the connection stays usable.
            KeyboardInterrupt: On Ctrl-C, after cancelling the request.
        """
        if self._started and not self._closed and (self._lost() or self._stale()):
            self._restart()
        return self._run(make, timeout)

    def _run[T](
        self,
        make: Callable[[ClientSession], Coroutine[Any, Any, T]],
        timeout: float,
    ) -> T:
        """`_submit` without looking at the connection first."""
        loop, session = self._loop, self._session
        if loop is None or session is None:
            raise McpServerError(f"MCP server {self.name!r} is not running")
        future = asyncio.run_coroutine_threadsafe(make(session), loop)
        try:
            result = future.result(timeout)
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
        self._active = time.monotonic()
        return result

    def _stale(self) -> bool:
        """Whether a connection that has been idle for a while is gone.

        A program that exits, or a server that drops the session, between two calls
        is not noticed by the connection's task, which is only waiting; asking is the
        way to know, and it cannot do any harm (a ping changes nothing). A server too
        busy to answer is not gone, so only an answer that says the connection is
        closed counts.
        """
        if time.monotonic() - self._active < _IDLE_SECONDS:
            return False
        try:
            self._run(lambda session: session.send_ping(), _PING_SECONDS)
        except TimeoutError:
            return False
        except McpServerError:
            return True
        return False


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


def _log_file(config: McpServerConfig, log_dir: Path, name: str) -> Path | None:
    """The file a program's stderr went to, if there is one to point the user at."""
    path = log_dir / f"{name}.log"
    return path if config.command and path.exists() else None


def close_servers(servers: Iterable[McpServer]) -> None:
    """Close the servers all at once.

    Closing one can take seconds (a server that does not leave when its input
    ends is waited for, then stopped), and servers closed one after another add up.
    """
    servers = list(servers)
    if servers:
        with concurrent.futures.ThreadPoolExecutor(len(servers)) as pool:
            list(pool.map(McpServer.close, servers))


def start_servers(
    configs: Mapping[str, McpServerConfig],
    log_dir: Path,
    on_notice: Callable[[str], None],
    *,
    environ: Mapping[str, str] = os.environ,
    report: list[ServerStatus] | None = None,
) -> tuple[list[McpServer], list[Tool]]:
    """Start the servers, all at once, and collect their tools.

    A server that fails is reported and left out: it must not stop muagent. One
    that is switched off (`enable` false) is left out without a word, and so is
    one that starts: the banner counts its tools. Every server has its startup
    timeout counted from the moment all of them were launched, so servers that never
    answer delay the start by the longest timeout, not by the sum; the ones that
    failed are closed together at the end.

    Args:
        configs: The servers to start, by name.
        log_dir: Where each server's stderr goes, in a file named after it.
        on_notice: Called with a line about each server that fails.
        environ: Where the `${NAME}` placeholders of the servers are filled in from.
        report: If given, a `ServerStatus` for every server in `configs` is added to
            it, in their order: the notices of the ones that failed are gone once
            they are shown, and `/servers` tells them again.

    Returns:
        The servers that are running, which the caller must close, and their tools.
    """
    statuses: dict[str, ServerStatus] = {}

    def log_of(name: str) -> Path | None:
        return _log_file(configs[name], log_dir, name)

    def failed_with(name: str, message: str) -> None:
        on_notice(message)
        statuses[name] = ServerStatus(name, "failed", message, log=log_of(name))

    # Each server, or why it cannot be started, in the order of the file.
    entries: list[tuple[str, McpServer | str]] = []
    try:
        for name, config in configs.items():
            if not config.enable:
                statuses[name] = ServerStatus(name, "off", "enable is false")
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
        failed: list[McpServer] = []
        tools: list[Tool] = []
        taken: set[str] = set()  # names are unique across the servers too
        for name, entry in entries:
            if isinstance(entry, str):
                failed_with(name, entry)
                continue
            try:
                entry.connect()
                found = entry.tools(taken)
            except Exception as exc:  # noqa: BLE001
                failed.append(entry)
                failed_with(
                    name,
                    str(exc)
                    if isinstance(exc, McpServerError)
                    else f"MCP server {name!r} failed: {type(exc).__name__}: {exc}",
                )
                continue
            running.append(entry)
            tools += found
            statuses[name] = ServerStatus(
                name,
                "running",
                tools=tuple(tool.name for tool in found),
                hidden=entry.hidden,
                log=log_of(name),
            )
        close_servers(failed)
    except BaseException:
        close_servers(entry for _, entry in entries if not isinstance(entry, str))
        raise
    if report is not None:
        report.extend(statuses[name] for name in configs)
    return running, tools


@dataclass(frozen=True)
class ServerUpdate:
    """What a change to one server means for the agent: the tools it offered (to be
    taken away), the tools it offers now, and what it says about using them."""

    name: str
    removed: tuple[str, ...]
    tools: list[Tool]
    instructions: str | None
    message: str  # to tell the user


class McpHub:
    """The servers while muagent runs.

    `start_servers` starts them once. The hub keeps them after that, and so can start
    one again (`reconnect`: a server that failed to start, or hangs, costs a restart
    of muagent otherwise, and the conversation with it) and follow a server that says
    its tools changed (`refresh`). Neither touches the agent: they return what to
    change, and the caller applies it, so that the tools do not change in the middle
    of a turn.
    """

    def __init__(
        self,
        configs: Mapping[str, McpServerConfig],
        log_dir: Path,
        *,
        environ: Mapping[str, str] = os.environ,
    ) -> None:
        self.configs = configs
        self.log_dir = log_dir
        self.environ = environ
        self.statuses: list[
            ServerStatus
        ] = []  # in the order of the file, kept in place
        self.servers: dict[str, McpServer] = {}  # the ones that run

    def start(self, on_notice: Callable[[str], None]) -> list[Tool]:
        """Start the servers (see `start_servers`) and return their tools."""
        running, tools = start_servers(
            self.configs,
            self.log_dir,
            on_notice,
            environ=self.environ,
            report=self.statuses,
        )
        self.servers = {server.name: server for server in running}
        return tools

    def instructions(self) -> dict[str, str]:
        """What the running servers say about using their tools, except the ones whose
        config leaves it out."""
        return {
            name: server.instructions
            for name, server in self.servers.items()
            if server.instructions and self.configs[name].instructions
        }

    def _status(self, name: str) -> ServerStatus:
        return next(status for status in self.statuses if status.name == name)

    def _set_status(self, status: ServerStatus) -> None:
        index = next(i for i, s in enumerate(self.statuses) if s.name == status.name)
        self.statuses[index] = status

    def _taken(self, besides: str) -> set[str]:
        """The names of the tools of the other servers, which a new list must not
        repeat."""
        return {
            tool
            for status in self.statuses
            if status.name != besides
            for tool in status.tools
        }

    def refresh(self) -> list[ServerUpdate]:
        """List again the tools of the servers that said they changed.

        A server whose tools cannot be listed keeps the ones it had, and the update
        only says so.
        """
        updates = []
        for name, server in self.servers.items():
            if not server.tools_changed:
                continue
            status = self._status(name)
            try:
                tools = server.tools(self._taken(name))
            except Exception as exc:  # noqa: BLE001
                reason = (
                    str(exc) if isinstance(exc, McpServerError) else type(exc).__name__
                )
                message = f"MCP server {name!r} changed its tools, which cannot be listed: {reason}"
                updates.append(ServerUpdate(name, (), [], None, message))
                continue
            self._set_status(
                ServerStatus(
                    name,
                    "running",
                    tools=tuple(tool.name for tool in tools),
                    hidden=server.hidden,
                    log=status.log,
                )
            )
            count = f"{len(tools)} tool{'s' if len(tools) != 1 else ''}"
            updates.append(
                ServerUpdate(
                    name,
                    status.tools,
                    tools,
                    self.instructions().get(name),
                    f"MCP server {name!r} changed its tools: now {count}",
                )
            )
        return updates

    def reconnect(self, name: str) -> ServerUpdate:
        """Stop the server `name` if it runs, and start it again.

        Args:
            name: A server in the file.

        Returns:
            What to change in the agent: nothing but a message for a server that is
            switched off, the tools gone for one that fails, the new ones if it starts.
        """
        config = self.configs[name]
        before = self._status(name)
        if before.state == "off":
            return ServerUpdate(name, (), [], None, f"MCP server {name!r} is off")
        previous = self.servers.pop(name, None)
        if previous is not None:
            previous.close()
        server = None
        try:
            server = McpServer(
                name, _expand(name, config, self.environ), self.log_dir / f"{name}.log"
            )
            server.start()
            tools = server.tools(self._taken(name))
        except Exception as exc:  # noqa: BLE001
            if server is not None:
                server.close()
            message = (
                str(exc)
                if isinstance(exc, McpServerError)
                else f"MCP server {name!r} failed: {type(exc).__name__}: {exc}"
            )
            self._set_status(
                ServerStatus(
                    name,
                    "failed",
                    message,
                    log=_log_file(config, self.log_dir, name),
                )
            )
            return ServerUpdate(name, before.tools, [], None, message)
        self.servers[name] = server
        self._set_status(
            ServerStatus(
                name,
                "running",
                tools=tuple(tool.name for tool in tools),
                hidden=server.hidden,
                log=_log_file(config, self.log_dir, name),
            )
        )
        count = f"{len(tools)} tool{'s' if len(tools) != 1 else ''}"
        return ServerUpdate(
            name,
            before.tools,
            tools,
            self.instructions().get(name),
            f"MCP server {name!r} is running: {count}",
        )

    def close(self) -> None:
        """Stop the servers that run."""
        close_servers(self.servers.values())
