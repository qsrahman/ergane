"""Slash commands: a registry filled by the @command decorator, then the commands."""

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai.types.shared import ReasoningEffort
from pydantic import TypeAdapter, ValidationError

from muagent.config import EFFORTS

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


COMMANDS: dict[str, Command] = {}


def command(name: str) -> Callable[[Handler], Handler]:
    """Register a handler as a slash command. Stack it to give a command aliases.

    The first line of the handler's docstring is the description that /help shows.

    Args:
        name: The command as typed: a slash, then lowercase letters, digits and
            hyphens.

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
        COMMANDS[name] = Command(handler, docstring.splitlines()[0])
        return handler

    return register


@command("/help")
def help_(repl: Repl, args: str) -> None:
    """List the available commands."""
    width = max(len(name) for name in repl.commands)
    for name, cmd in sorted(repl.commands.items()):
        repl.console.write(f"  {name:<{width}}  {cmd.description}\n")
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
    return f"Model: {repl.agent.config.model}"


@command("/effort")
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


@command("/tools")
def tools_(repl: Repl, args: str) -> str:
    """List the tools offered to the model."""
    tools = [
        f"{tool.name}{' (read-only)' if tool.read_only else ''}"
        for tool in repl.agent.tools.values()
    ]
    return f"Tools: {', '.join(tools) or 'none'}"
