"""One MCP server as tools: connect to it and offer its tools to the agent.

A server is a program ergane runs and talks to over stdio, or a server it connects
to over streamable HTTP. Starting several of them, keeping them, and starting one
again is `mcp_hub.py`; what a server's tools and results become for the model is
`mcp_tools.py`.

The MCP SDK is async and the rest of ergane is not, so each server runs on a
background thread with its own event loop (anyio's blocking portal, which every
`openai` install has), and a call blocks until the loop answers. The `mcp` package is
imported only when a server starts, so ergane works without it as long as no server is
configured.

Design notes, for whoever changes this:

- The whole connection is one async context manager (`_connection`), which one task of
  the portal enters and leaves (`wrap_async_context_manager`): the SDK's task groups
  must be left by the task that entered them. `start()` blocks until it is entered;
  leaving it (`close()`) ends the server's process in the usual way. A start that is
  still waiting has no connection to leave, so `close()` cancels it by stopping the
  portal.
- The startup timeout is the outermost cancel scope of that task, so that the scopes of
  the SDK nest inside it, and it is disarmed once the server is up (a scope that ends
  before the ones inside it is an error).
- `start()` blocks, so servers are started at once by starting each on a thread of its
  own, which `mcp_hub.py` does. A `start()` that fails has stopped the server already
  (stopping takes seconds: the SDK waits 2 s for a program that ignores its closed
  input), on its own thread, so servers that failed together stop together.
- A server whose connection ended is started again by the next call (`_submit`), up to
  `_MAX_RESTARTS` times; the call that found it dead fails and is not repeated. After
  `_IDLE_SECONDS` of silence, or after a call that failed, a call first pings, because
  a program that exits while the task is only waiting is not noticed otherwise (the
  connection does not end by itself). Only an answer that says the connection is
  closed counts: a busy server is not gone.
- Trust is `read_only` for a whole server, from the user's file. The server's own
  `readOnlyHint` is never used: a server can claim anything.
- A program does not inherit the environment, only the SDK's short allow-list plus its
  `env`, so it never sees `OPENAI_API_KEY`; it starts in ergane's directory (there is
  no `cwd` setting); its stderr goes to a log file, or it would land in the terminal.
  A url must be https unless it is loopback, and may not hold credentials; the SDK
  follows redirects only within the url's origin, so the headers go nowhere else.
- The tools are listed once; a server's announcement that they changed
  (`tools/list_changed`) is not followed, as few servers send one. Its `instructions`
  are kept, cut to `_MAX_INSTRUCTION_CHARS`.
"""

import asyncio
import concurrent.futures
import importlib.util
import json
import math
import re
import threading
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, suppress
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import anyio
from anyio.from_thread import BlockingPortal, start_blocking_portal

from ergane.config import LocalMcpServer, McpServerConfig, RemoteMcpServer
from ergane.mcp_tools import (
    list_tools,
    required_keys_check,
    result_images,
    result_text,
    tool_name,
)
from ergane.tools import Tool, ToolOutput

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp_types import Tool as ToolDefinition

# The characters a server's name may have (the same rule, for the same reason, as the
# server names of `config.py`: the name is a prefix of the tools' names).
_NAME = re.compile(r"[a-zA-Z0-9_-]+")

# How long a remote server may be silent before its response is given up on (the SDK's
# own value). A call is not waited for longer than this, so a `call_timeout` above it
# raises it too.
_HTTP_READ_SECONDS = 300

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


class McpServerError(Exception):
    """An MCP server could not be started, or is not running."""


def _leaf(exc: BaseException) -> BaseException:
    """The exception inside the groups the SDK's task groups wrap it in."""
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return exc


async def _stdio(stack: AsyncExitStack, config: LocalMcpServer, log: TextIO):
    """Run the program of `config` and give its streams, which `stack` closes."""
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=config.command, args=config.args, env=config.env
    )
    return await stack.enter_async_context(stdio_client(params, errlog=log))


async def _http(stack: AsyncExitStack, config: RemoteMcpServer):
    """Connect to the url of `config` and give its streams, which `stack` closes."""
    import httpx2
    from mcp.client.streamable_http import streamable_http_client

    # Redirects are followed only within the url's origin, so the headers go nowhere
    # else. A server may hold a response open for as long as a call may take.
    http = await stack.enter_async_context(
        httpx2.AsyncClient(
            headers=config.headers,
            timeout=httpx2.Timeout(
                30.0, read=max(_HTTP_READ_SECONDS, config.call_timeout)
            ),
        )
    )
    return await stack.enter_async_context(
        streamable_http_client(config.url, http_client=http)
    )


class McpClient:
    """A connection to one MCP server, used from synchronous code.

    The server is a program run by `command` and spoken to over stdio, or a `url`
    spoken to over streamable HTTP; exactly one of them.
    """

    def __init__(
        self,
        name: str,
        config: McpServerConfig,
        log: Path,
        *,
        append_log: bool = False,
    ) -> None:
        """Describe a server; `start()` connects to it.

        A McpClient is used once: after `close()` it cannot be started again.

        Args:
            name: Prefixes the names of its tools. Letters, digits, `_` and `-`.
            config: What to run or where to connect, and how far to trust it. Its
                `${NAME}` placeholders must be filled in already (see `_expand`).
            log: For a program, the file for its stderr, so that its log lines do not
                land in the terminal. Opened (and emptied) by `start()`, closed by
                `close()`. A url has no stderr, and the file is not touched.
            append_log: Add to the log instead of emptying it, when the server
                replaces one that ran before (see `McpHub.reconnect`): why that one
                stopped is in it.

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
        self._append_log = append_log

        # What `start()` entered, in order: the log, the portal (a thread with an event
        # loop) and the connection. Leaving it in turn stops the server.
        self._stack: ExitStack | None = None
        self._portal: BlockingPortal | None = None
        self._session: ClientSession | None = None
        self._started = False  # the connection has been up at least once
        self._closed = False  # close() was called: do not start it again
        self._restarts = 0
        # `close()` waits for a `start()` that is still unwinding.
        self._lifecycle = threading.Lock()
        # Calls run on several threads at once; checking the connection and starting
        # the server again must happen once, not once for each of them.
        self._connection_lock = threading.Lock()
        # The requests waiting for an answer, so that Ctrl-C can cancel them.
        self._inflight: set[concurrent.futures.Future[Any]] = set()
        self._active = 0.0  # when the server last answered
        self.hidden = 0  # tools it has that the config leaves out, after tools()
        # What the server says about using its tools, once connected.
        self.instructions: str | None = None

    def start(self) -> None:
        """Start the server and connect to it, and wait until it is up. Call `close()`
        when done. A server that fails to start has been stopped.

        Servers are started at once by starting each on a thread of its own.

        Raises:
            McpServerError: If the `mcp` package is missing, the log cannot be
                opened, the server cannot be run, exits before it answers, takes
                longer than the startup timeout, or is closed while it starts.
        """
        if importlib.util.find_spec("mcp") is None:
            raise McpServerError(
                f"MCP server {self.name!r} needs the mcp package: "
                "install ergane with its mcp extra"
            )
        with self._lifecycle:
            stack = self._stack = ExitStack()
            try:
                log = self._open_log()
                if log is not None:
                    stack.enter_context(log)
                portal = stack.enter_context(
                    start_blocking_portal(name=f"mcp-{self.name}")
                )
                self._portal = portal
                if self._closed:  # close() came before there was a portal to stop
                    raise McpServerError(
                        f"MCP server {self.name!r} was closed while it started"
                    )
                self._session = stack.enter_context(
                    portal.wrap_async_context_manager(self._connection(log))
                )
            except BaseException as exc:
                self._shutdown()
                if isinstance(
                    exc, asyncio.CancelledError | concurrent.futures.CancelledError
                ):
                    # close() cancelled the start
                    raise McpServerError(
                        f"MCP server {self.name!r} was closed while it started"
                    ) from None
                if isinstance(exc, McpServerError) or not isinstance(exc, Exception):
                    raise
                if isinstance(exc, TimeoutError):
                    timeout = self._config.startup_timeout
                    reason = f"did not answer within {timeout:g} s"
                else:
                    leaf = _leaf(exc)
                    reason = f"failed to start: {type(leaf).__name__}: {leaf}"
                raise McpServerError(f"MCP server {self.name!r} {reason}") from None
            self._started = True
            self._active = time.monotonic()
            text = (self._session.instructions or "").strip()
            if len(text) > _MAX_INSTRUCTION_CHARS:
                text = text[:_MAX_INSTRUCTION_CHARS] + "…"
            self.instructions = text or None

    def _open_log(self) -> TextIO | None:
        """Open the file for a program's stderr: None for a url, which has none.

        Raises:
            McpServerError: If the file cannot be opened.
        """
        if self._config.type != "local":
            return None
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            # A restart adds to the log: why the server stopped is in it.
            append = self._append_log or self._restarts
            log = self._log_path.open("a" if append else "w")
        except OSError as exc:
            raise McpServerError(
                f"MCP server {self.name!r}: cannot write its log: {exc}"
            ) from None
        # Server logs can hold tool arguments and other sensitive text; keep them to
        # the user who started ergane.
        with suppress(OSError):
            self._log_path.chmod(0o600)
        return log

    @asynccontextmanager
    async def _connection(self, log: TextIO | None) -> AsyncIterator[ClientSession]:
        """The whole connection, which one task of the portal enters and leaves: the
        SDK's task groups must be left by the task that entered them."""
        from mcp import ClientSession
        from mcp_types import Implementation

        config = self._config
        async with AsyncExitStack() as stack:
            # The outermost scope, so that the SDK's scopes nest inside it; it is
            # disarmed once the server is up.
            startup = stack.enter_context(anyio.fail_after(config.startup_timeout))
            if config.type == "local":
                assert log is not None
                read, write = await _stdio(stack, config, log)
            else:
                read, write = await _http(stack, config)
            # Say who is calling: a server logs it, and some adapt to the client.
            session = await stack.enter_async_context(
                ClientSession(
                    read,
                    write,
                    client_info=Implementation(
                        name="ergane", version=version("ergane")
                    ),
                )
            )
            await session.initialize()
            startup.deadline = math.inf
            yield session

    def close(self) -> None:
        """Disconnect and stop the server and its thread. Safe to call twice."""
        self._closed = True
        portal = self._portal
        if portal is not None and self._session is None:
            # Still starting: there is no connection to leave, so cancel the start.
            with suppress(RuntimeError):  # the portal is already stopped
                portal.call(portal.stop, True)
        with self._lifecycle:  # a start that is unwinding ends first
            self._shutdown()

    def _shutdown(self) -> None:
        """Leave what `start()` entered, the connection first. Call it with the
        lifecycle lock held. An error on the way out is the server's, which is
        stopped either way."""
        stack, self._stack = self._stack, None
        self._portal = self._session = None
        if stack is not None:
            with suppress(Exception, BaseExceptionGroup):
                stack.close()

    def _lost(self) -> bool:
        """Whether the connection was up, and is gone without `close()`: a restart
        that failed leaves none."""
        return self._started and not self._closed and self._session is None

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
        with self._lifecycle:
            self._shutdown()  # what is left of the old connection
        try:
            self.start()
        except McpServerError as exc:
            raise McpServerError(f"{exc} (restarting it after it stopped)") from None

    def definitions(self) -> list[ToolDefinition]:
        """What the server lists: all its tools, or the ones the config lists.

        This is the slow part of `tools()` (a round trip, or several), apart so that
        servers can do it at the same time.

        Raises:
            McpServerError: If the server is not running, or the config lists a
                tool it does not have (a typo would otherwise offer nothing, silently).
        """
        definitions = self._submit(list_tools, self._config.startup_timeout)
        if self._config.tools is None:
            return definitions
        have = {d.name for d in definitions}
        unknown = [name for name in self._config.tools if name not in have]
        if unknown:
            raise McpServerError(
                f"MCP server {self.name!r} has no tool {', '.join(unknown)} "
                f"(it has: {', '.join(sorted(have))})"
            )
        kept = [d for d in definitions if d.name in self._config.tools]
        self.hidden = len(definitions) - len(kept)
        return kept

    def tools(
        self,
        taken: set[str] | None = None,
        definitions: list[ToolDefinition] | None = None,
    ) -> list[Tool]:
        """The server's tools, as tools for the agent.

        Args:
            taken: The names already given to other servers' tools, which these
                must not repeat, and which they are added to. Names are
                `server__tool`, but `a` with `b__c` and `a__b` with `c` both come
                to `a__b__c`.
            definitions: What `definitions()` returned, if it was called already.

        Raises:
            McpServerError: See `definitions()`.
        """
        if definitions is None:
            definitions = self.definitions()
        taken = set() if taken is None else taken
        return [
            self._make_tool(tool_name(self.name, d.name, taken), d) for d in definitions
        ]

    def _make_tool(self, name: str, definition: ToolDefinition) -> Tool:
        def runner(arguments: str) -> str | ToolOutput:
            result = self._submit(
                lambda session: session.call_tool(
                    definition.name, json.loads(arguments)
                ),
                self._config.call_timeout,
            )
            images = result_images(result) if self._config.images else ()
            text = result_text(result)
            return ToolOutput(text, images) if images else text

        return Tool(
            name,
            definition.description or definition.title or "",
            definition.input_schema,
            required_keys_check(definition.input_schema),
            runner,
            self._config.read_only,
            self.cancel_inflight,
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
        if self._started and not self._closed:
            with self._connection_lock:
                if self._lost() or self._stale():
                    self._restart()
        return self._run(make, timeout)

    def cancel_inflight(self) -> None:
        """Cancel the requests that are waiting for an answer: each caller gets an
        error. The SDK tells the server, which can stop the work."""
        for future in list(self._inflight):
            future.cancel()

    def _run[T](
        self,
        make: Callable[[ClientSession], Coroutine[Any, Any, T]],
        timeout: float,
    ) -> T:
        """`_submit` without looking at the connection first."""
        portal, session = self._portal, self._session
        if portal is None or session is None:
            raise McpServerError(f"MCP server {self.name!r} is not running")
        try:
            future = portal.start_task_soon(make, session)
        except RuntimeError:  # close() stopped the portal since
            raise McpServerError(f"MCP server {self.name!r} is not running") from None
        self._inflight.add(future)
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
            self._active = 0.0  # the next call checks the connection first
            leaf = _leaf(exc)
            raise McpServerError(
                f"MCP server {self.name!r}: {type(leaf).__name__}: {leaf}"
            ) from None
        finally:
            self._inflight.discard(future)
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
