"""Slash commands: a registry filled by the @command decorator, then the commands."""

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


@dataclass(frozen=True)
class Command:
    """A slash command: what it does and the description /help shows."""

    handler: Handler
    description: str


COMMANDS: dict[str, Command] = {}


def command(name: str, description: str) -> Callable[[Handler], Handler]:
    """Register a handler as a slash command. Stack it to give a command aliases.

    Args:
        name: The command as typed, lowercase, including the slash.
        description: The one-line description shown by /help.

    Returns:
        A decorator that registers the handler and returns it unchanged.

    Raises:
        ValueError: If the name is already registered.
    """

    def register(handler: Handler) -> Handler:
        if name in COMMANDS:
            raise ValueError(f"Command already registered: {name}")
        COMMANDS[name] = Command(handler, description)
        return handler

    return register


@command("/help", "list commands")
def help_(repl: Repl, args: str) -> None:
    """List the available commands."""
    width = max(len(name) for name in repl.commands)
    for name, cmd in sorted(repl.commands.items()):
        repl.console.write(f"  {name:<{width}}  {cmd.description}\n")
    repl.console.blank()


@command("/reset", "clear the conversation history")
def reset(repl: Repl, args: str) -> str:
    """Clear the conversation history."""
    repl.agent.context.reset()
    return "History reset"


@command("/clear", "clear the screen")
def clear(repl: Repl, args: str) -> None:
    """Clear the terminal screen; the history is kept."""
    repl.console.clear()


@command("/exit", "quit")
@command("/quit", "quit")
def stop(repl: Repl, args: str) -> None:
    """Leave the loop after the current command."""
    repl.running = False


@command("/undo", "remove the last exchange from the history")
def undo(repl: Repl, args: str) -> str:
    """Remove the last user message and its reply from the history."""
    removed = repl.agent.context.undo()
    return "Last exchange removed" if removed else "Nothing to undo"


@command("/retry", "resend the last prompt, replacing its reply")
def retry(repl: Repl, args: str) -> str | None:
    """Resend the last prompt, first removing its exchange if it was saved."""
    prompt = repl.agent.context.rewind()
    if prompt is None:
        return "Nothing to retry"
    repl.ask(prompt)
    return None


@command("/model", "show the model, or switch with /model NAME")
def model(repl: Repl, args: str) -> str:
    """Show the model, or switch to the one named in args."""
    if args:
        repl.agent.config.model = args
    return f"Model: {repl.agent.config.model}"


@command("/effort", "show the reasoning effort, or set it with /effort LEVEL")
def effort(repl: Repl, args: str) -> str:
    """Show the reasoning effort, or set it; "default" leaves it out of the request."""
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
