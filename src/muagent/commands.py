"""Slash commands: a registry filled by the @command decorator, then the commands."""

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from openai.types.shared import ReasoningEffort
from pydantic import TypeAdapter, ValidationError

from muagent.config import EFFORTS
from muagent.context import estimate_tokens
from muagent.errors import TurnError, describe_error

if TYPE_CHECKING:
    from muagent.repl import Repl

# A command handler gets the REPL and the text after the command name (or ""). It
# returns a short status message for the REPL to show, or None.
Handler = Callable[["Repl", str], str | None]

# Lowercase, because input is lowercased before the lookup; and a leading slash, so
# that a plain prompt can never be mistaken for a command.
_NAME = re.compile(r"/[a-z][a-z0-9-]*")


@dataclass(frozen=True)
class Command:
    """A slash command: what it does and the description /help shows."""

    handler: Handler
    description: str
    arguments: tuple[str, ...] = ()  # what the line editor completes after the name


COMMANDS: dict[str, Command] = {}


def command(name: str, arguments: tuple[str, ...] = ()) -> Callable[[Handler], Handler]:
    """Register a handler as a slash command. Stack it to give a command aliases.

    The first line of the handler's docstring is the description that /help shows.

    Args:
        name: The command as typed: a slash, then lowercase letters, digits and
            hyphens.
        arguments: The words the line editor offers after the name.

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
        [(name, cmd.description) for name, cmd in sorted(repl.commands.items())]
    )
    repl.console.blank()
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
            *keys,
        ]
    repl.console.columns(keys)
    repl.console.blank()


@command("/reset")
def reset(repl: Repl, args: str) -> str:
    """Clear the conversation history."""
    repl.agent.context.reset()
    return "History reset"


@command("/clear")
def clear(repl: Repl, args: str) -> None:
    """Clear the terminal screen; the history is kept."""
    repl.console.clear()


@command("/exit")
@command("/quit")
def stop(repl: Repl, args: str) -> None:
    """Quit muagent."""
    repl.running = False


@command("/undo")
def undo(repl: Repl, args: str) -> str:
    """Remove the last exchange from the history."""
    removed = repl.agent.context.undo()
    return "Last exchange removed" if removed else "Nothing to undo"


@command("/retry")
def retry(repl: Repl, args: str) -> str | None:
    """Resend the last prompt, replacing its reply.

    Its exchange is removed from the history first, if it was saved.
    """
    prompt = repl.agent.context.rewind()
    if prompt is None:
        return "Nothing to retry"
    repl.ask(prompt)
    return None


@command("/model")
def model(repl: Repl, args: str) -> str:
    """Show the model, or switch with /model NAME."""
    if args:
        repl.agent.config.model = args
        repl.agent.context.counted = None  # another model may count differently
    return f"Model: {repl.agent.config.model}"


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
            return f"Unknown effort: {args}. Use {', '.join(EFFORTS)} or default."
    return f"Reasoning effort: {config.reasoning_effort or 'model default'}"


@command("/compact")
def compact_(repl: Repl, args: str) -> str | None:
    """Summarize the earlier turns now; FOCUS says what the summary must keep."""
    try:
        done = repl.agent.compact(repl.console.notice, args or None)
    except TurnError as exc:
        return f"Could not summarize: {exc}"
    except Exception as exc:  # noqa: BLE001
        return describe_error(exc)
    return None if done else "Nothing to summarize: the history is short"


@command("/summary")
def summary_(repl: Repl, args: str) -> str | None:
    """Show the summary that stands for the earliest turns."""
    summary = repl.agent.context.summary
    if summary is None:
        return "No summary yet"
    repl.console.write(f"{summary}\n")
    repl.console.blank()
    return None


@command("/context")
def context_(repl: Repl, args: str) -> None:
    """Show how much of the context budget the next request would use."""
    agent = repl.agent
    instructions, tools, history = agent.context_estimate()
    total = instructions + tools + history
    budget = agent.config.max_context_tokens
    rows = [
        ("instructions", f"~{instructions:,}"),
        (f"tools ({len(agent.tools)})", f"~{tools:,}"),
        ("history", f"~{history:,}"),
        ("total", f"~{total:,} of {budget:,} ({total * 100 // budget}%)"),
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
    repl.console.blank()


@command("/servers")
def servers_(repl: Repl, args: str) -> str | None:
    """List the MCP servers: which run, which failed, and what their tools cost."""
    if not repl.servers:
        return "No MCP servers"
    tools = repl.agent.tools
    home = str(Path.home())
    rows = []
    for server in repl.servers:
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
        else:
            text = f"{server.state}: {server.detail}"
        if server.log:
            log = str(server.log)
            text += f", log {'~' + log[len(home) :] if log.startswith(home) else log}"
        rows.append((server.name, text))
    repl.console.columns(rows)
    repl.console.blank()
    return None


@command("/tools")
def tools_(repl: Repl, args: str) -> str | None:
    """List the tools offered to the model."""
    agent = repl.agent
    free = [tool.name for tool in agent.tools.values() if tool.read_only]
    allowed = [
        name
        for name in agent.tools
        if name in agent.always_allowed and name not in free
    ]
    asks = [name for name, tool in agent.tools.items() if not tool.read_only]
    asks = [name for name in asks if name not in allowed]
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
    repl.console.blank()
    return None
