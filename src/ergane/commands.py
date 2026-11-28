"""Slash commands: a registry filled by the @command decorator, then the commands."""

import inspect
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from openai.types.shared import ReasoningEffort
from pydantic import TypeAdapter, ValidationError

from ergane.config import EFFORTS
from ergane.context import estimate_tokens
from ergane.errors import TurnError, describe_error
from ergane.sessions import SessionError, SessionInfo, ago

if TYPE_CHECKING:
    from ergane.repl import Repl

# A command handler gets the REPL and the text after the command name (or ""). It
# returns a short status message for the REPL to show, or None.
Handler = Callable[["Repl", str], str | None]
Arguments = tuple[str, ...] | Callable[["Repl"], Iterable[str]]

# Lowercase, because input is lowercased before the lookup; and a leading slash, so
# that a plain prompt can never be mistaken for a command.
_NAME = re.compile(r"/[a-z][a-z0-9-]*")


class Warn(str):
    """What a command returns when it did not do what was asked and the user can fix
    that (a name that is not there): shown as a warning, not as a plain notice."""


class Fail(str):
    """What a command returns when it failed: shown as an error."""


@dataclass(frozen=True)
class Command:
    """A slash command: what it does and the description /help shows."""

    handler: Handler
    description: str
    # What the line editor completes after the name: words, or a function of the REPL
    # for words only it knows (the servers to start again).
    arguments: Arguments = ()


COMMANDS: dict[str, Command] = {}


def command(name: str, arguments: Arguments = ()) -> Callable[[Handler], Handler]:
    """Register a handler as a slash command. Stack it to give a command aliases.

    The first line of the handler's docstring is the description that /help shows.

    Args:
        name: The command as typed: a slash, then lowercase letters, digits and
            hyphens.
        arguments: The words the line editor offers after the name, or a function
            that returns them, given the REPL.

    Returns:
        A decorator that registers the handler and returns it unchanged.

    Raises:
        ValueError: If the name is not valid or is already registered, or the
            handler has no docstring.
    """
    if not _NAME.fullmatch(name):
        raise ValueError(
            "Command names are a slash, then lowercase letters, digits and hyphens: "
            f"{name!r}"
        )

    def register(handler: Handler) -> Handler:
        if name in COMMANDS:
            raise ValueError(f"Command already registered: {name}")
        docstring = inspect.getdoc(handler)
        if not docstring:
            raise ValueError(
                f"Command {name} needs a docstring: its first line is the "
                "description /help shows"
            )
        COMMANDS[name] = Command(handler, docstring.splitlines()[0], arguments)
        return handler

    return register


@command("/help")
def help_(repl: Repl, args: str) -> None:
    """List the available commands."""
    repl.console.columns(
        [(name, cmd.description) for name, cmd in sorted(repl.commands.items())],
        heading="Commands",
    )
    allow = 'At "Allow …?": y allows, a allows the tool for the session, ? shows it all'
    keys = [("y, a, ?", allow)]
    if repl.session:
        keys = [
            (
                "Alt-Enter, Ctrl-J",
                "Add a line to the prompt (a pasted block is one prompt)",
            ),
            ("Tab, Up, Down", "Complete a command, recall earlier prompts"),
            ("Ctrl-C", "Clear the line; on an empty line, press twice to quit"),
            ("Ctrl-R", "Search the earlier prompts"),
            (
                "Ctrl-X Ctrl-E",
                "Edit the prompt in $EDITOR, and send it when that closes",
            ),
            ("Ctrl-D", "Quit; press twice once there is a conversation"),
            *keys,
        ]
    repl.console.columns(sorted(keys, key=lambda row: row[0].lower()), heading="Keys")


@command("/reset")
def reset(repl: Repl, args: str) -> str:
    """Clear the conversation history."""
    if not repl.save_session():
        return Warn("History kept, since it could not be saved")
    repl.agent.reset()
    repl.saved = None  # what follows is a new session; the old one stays saved
    return "History reset"


# How many sessions the menu of /sessions offers: the newest. /sessions ID reaches the rest.
_MENU_SESSIONS = 10


def _described(
    info: SessionInfo, current: SessionInfo | None, short_id: bool = False
) -> str:
    """A session in one line: what it was about, how long, when, and its id (in a menu
    only the end of it, which tells sessions of one title apart)."""
    turns = f"{info.turns} turn{'s' if info.turns != 1 else ''}"
    mark = " (current)" if current is not None and current.id == info.id else ""
    shown = info.id.rsplit("-", 1)[-1] if short_id else info.id
    return f"{info.title}  ·  {turns}  ·  {ago(info)}  ·  {shown}{mark}"


@command("/sessions")
def sessions_(repl: Repl, args: str) -> str | None:
    """Continue a saved session of this directory: choose one, or /sessions ID."""
    store = repl.store
    if store is None:
        return "Sessions are not saved"
    found = store.sessions()
    if not found:
        return "No saved sessions for this directory"
    if args:
        number = int(args) if args.isdigit() else 0
        ref = found[number - 1].id if 1 <= number <= len(found) else args
    elif repl.console.choose is not None:
        options = [
            (i.id, _described(i, repl.saved, short_id=True))
            for i in found[:_MENU_SESSIONS]
        ]
        ref = repl.console.choose(
            "Continue which session?", [*options, ("q", "Cancel")]
        )
        if ref == "q":
            return "Cancelled"
    else:
        repl.console.columns(
            [(str(n), _described(i, repl.saved)) for n, i in enumerate(found, 1)]
        )
        return "Continue one with /sessions NUMBER or /sessions ID"
    # Before the file is read: it may be the one of this conversation.
    if not repl.save_session():
        return Warn("Session not changed, since this one could not be saved")
    try:
        return repl.restore(store.load(ref))
    except SessionError as exc:
        return Warn(str(exc))


@command("/clear")
def clear(repl: Repl, args: str) -> None:
    """Clear the terminal screen; the history is kept."""
    repl.console.clear()


@command("/exit")
@command("/quit")
def stop(repl: Repl, args: str) -> None:
    """Quit ergane."""
    repl.running = False


@command("/undo")
def undo(repl: Repl, args: str) -> str:
    """Remove the last exchange from the history."""
    removed = repl.agent.undo()
    return "Last exchange removed" if removed else "Nothing to undo"


@command("/retry")
def retry(repl: Repl, args: str) -> str | None:
    """Resend the last prompt, replacing its reply.

    Its exchange is removed from the history first, if it was saved.
    """
    prompt = repl.agent.rewind()
    if prompt is None:
        return "Nothing to retry"
    repl.ask(prompt)
    return None


def _model_names(repl: Repl) -> list[str]:
    """What is completed after `/model`: the server's models, or nothing if it cannot
    be asked, which completing must not report."""
    try:
        return repl.agent.models()
    except Exception:  # noqa: BLE001
        return []


@command("/model", _model_names)
def model(repl: Repl, args: str) -> str:
    """List the server's models, or switch with /model NAME."""
    if args:
        repl.agent.config.model = args
        return f"Model: {args}"
    current = f"Model: {repl.agent.config.model}"
    try:
        names = repl.agent.models(refresh=True)
    except Exception as exc:  # noqa: BLE001
        return Warn(f"{current}. Could not list the models: {describe_error(exc)}")
    if not names:
        return f"{current} (the server lists no models)"
    repl.console.columns([("available", ", ".join(names))])
    return current


@command("/effort", (*EFFORTS, "default"))
def effort(repl: Repl, args: str) -> str:
    """Show the reasoning effort, or set it with /effort LEVEL.

    LEVEL is a reasoning effort of the API, or "default" to leave it out of the
    request.
    """
    config = repl.agent.config
    if args.lower() == "default":
        config.reasoning_effort = None
    elif args:
        try:
            config.reasoning_effort = TypeAdapter(ReasoningEffort).validate_python(
                args.lower()
            )
        except ValidationError:
            return Warn(f"Unknown effort: {args}. Use {', '.join(EFFORTS)} or default.")
    return f"Reasoning effort: {config.reasoning_effort or 'model default'}"


@command("/compact")
def compact_(repl: Repl, args: str) -> str | None:
    """Summarize the earlier turns now; FOCUS says what the summary must keep."""
    try:
        done = repl.agent.compact(repl.console.notice, args or None)
    except TurnError as exc:
        return Fail(f"Could not summarize: {exc}")
    except Exception as exc:  # noqa: BLE001
        return Fail(describe_error(exc))
    return None if done else "Nothing to summarize: the history is short"


@command("/summary")
def summary_(repl: Repl, args: str) -> str | None:
    """Show the summary that stands for the earliest turns."""
    summary = repl.agent.summary
    if summary is None:
        return "No summary yet"
    repl.console.write(f"{summary}\n")
    repl.console.blank()
    return None


@command("/todo")
def todo(repl: Repl, args: str) -> str | None:
    """Show the model's todo list."""
    todos = repl.agent.todos
    if not todos.items:
        return "No todos"
    repl.console.todos(todos.report())
    repl.console.blank()
    return None


@command("/context")
def context_(repl: Repl, args: str) -> None:
    """Show how much of the context budget the next request would use."""
    agent = repl.agent
    instructions, tools, history = agent.context_estimate()
    counted = agent.counted_tokens() is not None
    total = agent.request_tokens()
    budget = agent.config.max_context_tokens
    rows = [
        ("instructions", f"~{instructions:,}"),
        (f"tools ({len(agent.offered())})", f"~{tools:,}"),
        ("history", f"~{history:,}"),
        (
            "total",
            f"{'' if counted else '~'}{total:,} of {budget:,} ({total * 100 // budget}%)"
            + (" (counted by the server)" if counted else ""),
        ),
    ]
    if agent.config.auto_compact:
        trigger = int(agent.config.compact_at * budget)
        rows.append(("summarizes at", f"~{trigger:,}"))
    else:
        rows.append(("summarizes at", "off (/compact does it by hand)"))
    if agent.last_input_tokens is not None:
        rows.append(
            ("last request", f"{agent.last_input_tokens:,} (counted by the server)")
        )
    repl.console.columns(rows)


def _servers_to_start(repl: Repl) -> list[str]:
    """The servers `/reconnect` may start again: the ones that are not switched off."""
    return [s.name for s in repl.hub.statuses.values() if s.state != "off"]


@command("/reconnect", _servers_to_start)
def reconnect(repl: Repl, args: str) -> str | None:
    """Start the MCP servers that failed again, or the server NAME if one is given."""
    hub = repl.hub
    if not hub.configs:
        return "No MCP servers to start"
    repl.attach(wait=True)  # the ones still starting are not "failed" yet
    if args:
        if args not in hub.configs:
            return Warn(f"No MCP server {args!r}; /servers lists them")
        names = [args]
    else:
        names = [s.name for s in hub.statuses.values() if s.state == "failed"]
        if not names:
            return "No MCP server failed; /reconnect NAME starts one again"
    for name in names:
        repl.apply(hub.reconnect(name))
    return None


@command("/servers")
def servers_(repl: Repl, args: str) -> str | None:
    """List the MCP servers: which run, which failed, and what their tools cost."""
    if not repl.hub.statuses:
        return "No MCP servers"
    tools = repl.agent.tools
    home = str(Path.home())
    rows = []
    for server in repl.hub.statuses.values():
        if server.state == "running":
            tokens = sum(
                estimate_tokens(tools[name].param())
                for name in server.tools
                if name in tools
            )
            count = len(server.tools)
            text = f"running, {count} tool{'s' if count != 1 else ''}"
            if server.hidden:
                text += f' ({server.hidden} left out by "tools")'
            text += f", ~{tokens:,} tokens"
        elif server.state == "starting":
            text = "starting…"
        else:
            text = f"{server.state}: {server.detail}"
        if server.log:
            log = str(server.log)
            text += f", log {'~' + log[len(home) :] if log.startswith(home) else log}"
        rows.append((server.name, text))
    repl.console.columns(rows)
    return None


@command("/tools")
def tools_(repl: Repl, args: str) -> str | None:
    """List the tools offered to the model."""
    agent = repl.agent
    offered = agent.offered()
    free = [name for name, tool in offered.items() if tool.read_only]
    asking = [name for name, tool in offered.items() if not tool.read_only]
    allowed = [name for name in asking if name in agent.always_allowed]
    asks = [name for name in asking if name not in agent.always_allowed]
    if not (free or allowed or asks):
        return "Tools: none"
    repl.console.columns(
        [
            (label, ", ".join(names))
            for label, names in (
                ("run without asking", free),
                ("allowed this session", allowed),
                ("ask first", asks),
            )
            if names
        ]
    )
    return None
