import sys
from collections.abc import Mapping
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import Output

from muagent.agent import Agent, TurnError, describe_error
from muagent.commands import Command
from muagent.console import Console

# A terminal asked for Shift-Enter (kitty's keyboard protocol, or xterm's
# modifyOtherKeys) sends one of these, which prompt_toolkit does not know. They are
# read as Alt-Enter. A terminal that sends a plain Enter for Shift-Enter has to be set
# to send one of them (or Esc followed by Enter) for the key to tell them apart.
for _sequence in ("\x1b[13;2u", "\x1b[27;2;13~"):
    ANSI_SEQUENCES[_sequence] = (Keys.Escape, Keys.ControlM)

_NEWLINE_KEYS = KeyBindings()


@_NEWLINE_KEYS.add("escape", "enter")  # Alt-Enter, and Shift-Enter where it is sent
@_NEWLINE_KEYS.add("c-j")  # a line feed, which every terminal can send
def _newline(event) -> None:
    event.current_buffer.insert_text("\n")


class _CommandCompleter(Completer):
    """Completes a slash command's name, then the arguments it lists.

    Not prompt_toolkit's NestedCompleter: that one finds the first word with a
    pattern that does not take the slash as part of it.
    """

    def __init__(self, commands: Mapping[str, Command]) -> None:
        self.commands = commands

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        name, space, rest = document.text_before_cursor.lstrip().partition(" ")
        if not space:
            words, typed = list(self.commands), name
        elif name in self.commands:
            words, typed = self.commands[name].arguments, rest
        else:
            return
        for word in words:
            if word.startswith(typed):
                yield Completion(word, -len(typed))


def _history_file(path: Path) -> History:
    """The history kept in `path`, readable by this user only; in memory if the file
    cannot be made."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        path.chmod(0o600)
    except OSError:
        return InMemoryHistory()
    return FileHistory(str(path))


def terminal_session(
    commands: Mapping[str, Command],
    history: Path,
    *,
    input: Input | None = None,
    output: Output | None = None,
) -> PromptSession[str] | None:
    """The line editor for a terminal, or None when input or output is redirected.

    It keeps the history in a file, completes the slash commands and the arguments
    they list, suggests the rest of an earlier prompt, takes a pasted block as one
    prompt (bracketed paste), and lets Alt-Enter or Ctrl-J (and Shift-Enter, where
    the terminal sends it) add a line. Redirected input is read with plain
    `input()`, as prompt_toolkit would only warn about it.

    Args:
        commands: The slash commands to complete.
        history: The file the prompts are kept in; its directory is made.
        input: Where to read from instead of the terminal (for tests).
        output: Where to write to instead of the terminal (for tests).
    """
    if input is None and not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    return PromptSession(
        history=_history_file(history),
        completer=_CommandCompleter(commands),
        auto_suggest=AutoSuggestFromHistory(),
        prompt_continuation=". ",  # lines added by hand or pasted line up under "> "
        key_bindings=_NEWLINE_KEYS,
        input=input,
        output=output,
    )


class Repl:
    """Interactive terminal loop: reads prompts, runs commands and turns."""

    def __init__(
        self,
        agent: Agent,
        commands: Mapping[str, Command],
        session: PromptSession[str] | None = None,
    ) -> None:
        """Create a REPL for an agent.

        Args:
            agent: The agent that answers prompts.
            commands: The slash commands, by name.
            session: The line editor that reads the prompts (see
                `terminal_session`); without one, `input()` does.
        """
        self.agent = agent
        # An answer to "Allow …?" is not a prompt: it has its own editor, so it is
        # not kept in the history or completed as a command.
        ask = (
            PromptSession(input=session.app.input, output=session.app.output).prompt
            if session
            else None
        )
        self.console = Console(ask)
        self.running = False
        self.commands = commands
        self.session = session

    def run(self) -> None:
        """Read and handle prompts until the user quits."""
        tools = len(self.agent.tools)
        count = f" | {tools} tool{'s' if tools != 1 else ''}" if tools else ""
        effort = self.agent.config.reasoning_effort
        shown = f" | effort: {effort}" if effort else ""
        self.console.write(
            f"muagent | model: {self.agent.config.model}{shown}{count}"
            " | /help lists commands\n\n"
        )

        self.running = True
        while self.running:
            try:
                prompt = (
                    self.session.prompt("> ") if self.session else input("> ")
                ).strip()
            except KeyboardInterrupt, EOFError:
                break
            if prompt:
                self.handle(prompt)
        self.console.write("Bye.\n")

    def handle(self, prompt: str) -> None:
        """Run a slash command, or send the prompt to the agent.

        Args:
            prompt: A non-empty line of input.
        """
        # The first word picks the command; the rest is its argument.
        name, _, args = prompt.partition(" ")
        command = self.commands.get(name.lower())
        if command:
            try:
                message = command.handler(self, args.strip())
            except KeyboardInterrupt:
                message = "Cancelled"
            except Exception as exc:  # noqa: BLE001
                message = f"Command error: {type(exc).__name__}: {exc}"
        # A single word starting with "/" is a mistyped command, not a prompt;
        # longer input such as "/usr/bin is missing" still goes to the model.
        elif prompt.startswith("/") and " " not in prompt:
            message = f"Unknown command: {prompt}. Type /help to list commands."
        else:
            self.ask(prompt)
            return
        if message:
            self.console.notice(message)
            self.console.blank()

    def ask(self, prompt: str) -> None:
        """Run one turn, printing the reply as it arrives and reporting
        problems without leaving the loop.

        Args:
            prompt: The user's input prompt.
        """
        console = self.console
        console.blank()
        try:
            self.agent.run_turn(
                prompt,
                on_text=console.write,
                on_notice=console.notice,
                confirm=console.confirm,
                on_call=console.call,
            )
        except TurnError as exc:
            console.notice(str(exc))
        except KeyboardInterrupt:
            console.notice("Cancelled")
        except Exception as exc:  # noqa: BLE001
            console.notice(describe_error(exc))
        console.end_line()
        console.blank()
