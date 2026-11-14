"""The MCP servers while muagent runs: start them, keep them, start one again.

One server is `mcp_server.py`. This is what a program does with all of them: start them
(`McpHub.start`, or `begin` and `take` in the background), report what became
of each (`ServerStatus`, for `/servers`), and start one again (`McpHub.reconnect`).

Design notes, for whoever changes this:

- Starting (`_Startup`): `launch()` every server, and then each one connects and lists
  its tools on a thread of its own, so a slow one holds up no other. The startup
  timeout counts from `launch()`, so servers that never answer delay the start by one
  timeout. The tools are named afterwards, in the order of the file, so that the names
  do not depend on who answered first. A server that failed is closed with the other
  failed ones; `close_servers` closes the running ones.
- At a terminal the hub does not wait (`McpHub.begin`): the prompt comes up at once
  and `take` collects the servers when the caller is between inputs, waiting for
  them only before a turn, which needs the tools. `poll` finishes a start that every
  server has answered, for the toolbar. Ctrl-C while waiting leaves the start to be
  waited for again. Notices about servers that failed are kept until `take`: they
  must not be printed over the prompt.
- `McpHub.reconnect` starts a server again and returns a `ServerUpdate` instead of
  touching the agent: the caller applies it between turns, because the tools of a turn
  must not change under it, and what the server counted for the last request no longer
  holds.
- A server's `instructions` (what it says about using its tools) go into the model's
  instructions, framed as notes by the server and not as orders (`SERVER_NOTES`), and
  only for a server whose config says `instructions: true`. They are as much the
  server's words as its tool descriptions, which the model reads already; the frame is
  for what a server might add.
"""

import concurrent.futures
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from muagent.mcp_config import McpServerConfig, UnsetVariableError, fill_placeholders
from muagent.mcp_server import McpServer, McpServerError
from muagent.tools import Tool

if TYPE_CHECKING:
    from mcp_types import Tool as ToolDefinition


@dataclass(frozen=True)
class ServerStatus:
    """What became of one server in config.json, at the start or at the last
    `/reconnect`, for `/servers`.

    A server that failed is reported once, as a notice, and would be forgotten.
    """

    name: str
    state: Literal["starting", "running", "failed", "off"]
    detail: str = ""  # why it failed, or why it is off
    tools: tuple[str, ...] = ()  # the names of the tools it offers the model
    hidden: int = 0  # tools it has that its `tools` setting leaves out
    log: Path | None = None  # where its stderr went, for a program


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


def _failure_text(name: str, exc: Exception) -> str:
    """What to tell the user of a server that could not be started."""
    if isinstance(exc, McpServerError):
        return str(exc)
    return f"MCP server {name!r} failed: {type(exc).__name__}: {exc}"


def _running_status(
    name: str, server: McpServer, tools: list[Tool], log: Path | None
) -> ServerStatus:
    return ServerStatus(
        name,
        "running",
        tools=tuple(tool.name for tool in tools),
        hidden=server.hidden,
        log=log,
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


class _Startup:
    """Servers being started: launched at once, then each connecting and listing its
    tools on a thread of its own, so that a slow one holds up no other.

    `begin()` does not wait. `finish()` does, and then names the tools in the order of
    the file, which is why that is done there and not on the threads: the names must
    not depend on which server answered first.
    """

    def __init__(
        self,
        configs: Mapping[str, McpServerConfig],
        log_dir: Path,
        environ: Mapping[str, str],
    ) -> None:
        self.configs = configs
        self.log_dir = log_dir
        self.environ = environ
        # Each server, or why it cannot be started, in the order of the file.
        self.entries: list[tuple[str, McpServer | str]] = []
        self.off: dict[str, ServerStatus] = {}
        self._futures: dict[str, concurrent.futures.Future[list[ToolDefinition]]] = {}
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None

    def begin(self) -> None:
        """Launch the servers that are on, and start connecting to them."""
        for name, config in self.configs.items():
            if not config.enable:
                self.off[name] = ServerStatus(name, "off", "enable is false")
                continue
            try:
                server = McpServer(
                    name,
                    _expand(name, config, self.environ),
                    self.log_dir / f"{name}.log",
                )
                server.launch()
            except McpServerError as exc:
                self.entries.append((name, str(exc)))
            else:
                self.entries.append((name, server))
        servers = {n: e for n, e in self.entries if not isinstance(e, str)}
        if servers:
            self._pool = concurrent.futures.ThreadPoolExecutor(
                len(servers), thread_name_prefix="mcp-start"
            )
            self._futures = {
                name: self._pool.submit(self._connect, server)
                for name, server in servers.items()
            }

    @staticmethod
    def _connect(server: McpServer) -> list[ToolDefinition]:
        server.connect()
        return server.definitions()

    def done(self) -> bool:
        """Whether every server has answered or failed, so `finish()` does not wait."""
        return all(future.done() for future in self._futures.values())

    def finish(
        self, on_notice: Callable[[str], None]
    ) -> tuple[dict[str, McpServer], dict[str, list[Tool]], dict[str, ServerStatus]]:
        """Wait for the servers, and name their tools.

        A server that failed is reported with `on_notice` and closed, with the other
        failed ones together (stopping takes seconds: the SDK waits 2 s for a program
        that ignores its closed input).

        Returns:
            The servers that run, their tools, and a status for every server in the
            file, each by name and in the order of the file.
        """
        statuses: dict[str, ServerStatus] = dict(self.off)
        running: dict[str, McpServer] = {}
        found: dict[str, list[Tool]] = {}
        failed: list[McpServer] = []
        taken: set[str] = set()  # names are unique across the servers too

        def log_of(name: str) -> Path | None:
            return _log_file(self.configs[name], self.log_dir, name)

        def failed_with(name: str, message: str) -> None:
            on_notice(message)
            statuses[name] = ServerStatus(name, "failed", message, log=log_of(name))

        for name, entry in self.entries:
            if isinstance(entry, str):
                failed_with(name, entry)
                continue
            try:
                tools = entry.tools(taken, self._futures[name].result())
            except Exception as exc:  # noqa: BLE001
                failed.append(entry)
                failed_with(name, _failure_text(name, exc))
                continue
            running[name] = entry
            found[name] = tools
            statuses[name] = _running_status(name, entry, tools, log_of(name))
        if self._pool is not None:
            self._pool.shutdown(wait=False)
        close_servers(failed)
        return running, found, {name: statuses[name] for name in self.configs}

    def abort(self) -> None:
        """Stop what was started, whatever it is doing."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        close_servers(e for _, e in self.entries if not isinstance(e, str))


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

    `start` (or `begin` and `take`) starts them once. The hub keeps them after that,
    and so can start one again (`reconnect`: a server that failed to start, or hangs, costs a restart
    of muagent otherwise, and the conversation with it). That does not touch the
    agent: it returns what to change, and the caller applies it, so that the tools do
    not change in the middle of a turn.
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
        self._startup: _Startup | None = None  # begun, and not finished
        self._started: list[ServerUpdate] = []  # finished, not yet taken
        self._notices: list[str] = []  # about servers that failed, not yet taken

    def start(self, on_notice: Callable[[str], None]) -> list[Tool]:
        """Start the servers, all at once, wait for them, and return their tools.

        A server that fails is reported and left out: it must not stop muagent. One
        that is switched off (`enable` false) is left out without a word, and so is
        one that starts: the banner counts its tools. Servers that never answer delay
        the start by the longest startup timeout, not by the sum (see `_Startup`).
        The servers that run are closed by `close()`; `statuses` has one for every
        server, since a notice is gone once it is shown.

        Args:
            on_notice: Called with a line about each server that failed.

        Raises:
            KeyboardInterrupt: On Ctrl-C, after closing what was started.
        """
        self.begin()
        try:
            updates = self.take(on_notice, wait=True)
        except BaseException:
            self.close()  # Ctrl-C: nobody is left to take them
            raise
        return [tool for update in updates for tool in update.tools]

    def begin(self) -> None:
        """Start the servers without waiting for them: they connect while the caller
        goes on. `take` collects them, and `starting` says whether they are done."""
        self._startup = _Startup(self.configs, self.log_dir, self.environ)
        try:
            self._startup.begin()
        except BaseException:
            self._startup.abort()
            self._startup = None
            raise
        self.statuses[:] = [
            self._startup.off.get(name) or ServerStatus(name, "starting")
            for name in self.configs
        ]

    @property
    def starting(self) -> bool:
        """Whether servers were begun and have not all answered yet."""
        return self._startup is not None and not self._startup.done()

    def poll(self) -> None:
        """Finish a start that every server has answered, so that `statuses` and
        `take` show how it went. Never waits."""
        if self._startup is not None and self._startup.done():
            self._finish()

    def take(
        self, on_notice: Callable[[str], None], *, wait: bool = False
    ) -> list[ServerUpdate]:
        """Collect what the servers that were begun came to.

        Args:
            on_notice: Called with a line about each server that failed to start.
            wait: Whether to wait for the servers that have not answered. If not,
                they are left for the next call.

        Returns:
            One update for each server that now runs, with the tools it offers and
            what it says about using them; nothing to take away, as the agent has
            none of its tools yet. Each is returned once.
        """
        if self._startup is not None and (wait or self._startup.done()):
            self._finish()
        for notice in self._notices:
            on_notice(notice)
        updates, self._started, self._notices = self._started, [], []
        return updates

    def _finish(self) -> None:
        startup = self._startup
        assert startup is not None
        notices: list[str] = []
        try:
            running, found, statuses = startup.finish(notices.append)
        except KeyboardInterrupt:
            raise  # the start goes on, and the next call waits for it again
        except BaseException:
            startup.abort()
            self._startup = None
            raise
        self._startup = None
        self._notices += notices
        self.servers = running
        self.statuses[:] = statuses.values()
        notes = self.instructions()
        self._started = [
            ServerUpdate(name, (), found[name], notes.get(name), "") for name in running
        ]

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
                name,
                _expand(name, config, self.environ),
                self.log_dir / f"{name}.log",
                append_log=True,  # the log of the one that ran before says why it ended
            )
            server.start()
            tools = server.tools(self._taken(name))
        except Exception as exc:  # noqa: BLE001
            if server is not None:
                server.close()
            message = _failure_text(name, exc)
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
            _running_status(name, server, tools, _log_file(config, self.log_dir, name))
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
        """Stop the servers that run, and the ones that are still starting."""
        if self._startup is not None:
            self._startup.abort()
            self._startup = None
        close_servers(self.servers.values())
