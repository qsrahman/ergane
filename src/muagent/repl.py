"""The input loop: reads prompts, dispatches slash commands, runs turns.

The line editor is prompt_toolkit (history, completion, bracketed paste); macOS's
Python has libedit, which has no bracketed paste, so a pasted block would be many
prompts. With input or output redirected there is no session and `input()` reads.
"""

import sys
from collections.abc import Mapping, Sequence
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
from prompt_toolkit.styles import Style

from muagent.agent import Agent
from muagent.commands import Command
from muagent.console import Console
from muagent.errors import TurnError, describe_error
from muagent.mcp_client import McpHub, ServerStatus, ServerUpdate

# A terminal asked for Shift-Enter (kitty's keyboard protocol, or xterm's
# modifyOtherKeys) sends one of these, which prompt_toolkit does not know. They are
# read as Alt-Enter. A terminal that sends a plain Enter for Shift-Enter has to be set
# to send one of them (or Esc followed by Enter) for the key to tell them apart.
for _sequence in ("\x1b[13;2u", "\x1b[27;2;13~"):
    ANSI_SEQUENCES[_sequence] = (Keys.Escape, Keys.ControlM)

# Colour is the terminal's own (the ANSI names), and prompt_toolkit drops it all when
# NO_COLOR is set. The toolbar is not reversed, which would make it a loud bar.
_STYLE = Style.from_dict(
    {"bottom-toolbar": "noreverse ansibrightblack", "placeholder": "dim"}
)

_KEYS = KeyBindings()


@_KEYS.add("escape", "enter")  # Alt-Enter, and Shift-Enter where it is sent
@_KEYS.add("c-j")  # a line feed, which every terminal can send
def _newline(event) -> None:
    event.current_buffer.insert_text("\n")


@_KEYS.add("c-c")
def _interrupt(event) -> None:
    # Typed text is what a stray Ctrl-C should cost, not the session: it clears the
    # line. On an empty line it is the usual interrupt, which `Repl.run` asks twice.
    if event.current_buffer.text:
        event.current_buffer.reset()
    else:
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")


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
            # A command is shown with what it does, so no one has to run /help.
            words = {n: self.commands[n].description for n in sorted(self.commands)}
            typed = name
        elif name in self.commands:
            words = dict.fromkeys(self.commands[name].arguments, "")
            typed = rest
        else:
            return
        for word, meta in words.items():
            if word.startswith(typed):
                yield Completion(word, -len(typed), display_meta=meta)


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
        key_bindings=_KEYS,
        style=_STYLE,
        # A plain string would get no style: only formatted text carries the class.
        placeholder=[
            (
                "class:placeholder",
                "Message, or /help for commands. Alt-Enter adds a line.",
            )
        ],
        # Ctrl-X Ctrl-E edits a long prompt in $EDITOR and sends it when the editor
        # closes; Ctrl-Z suspends muagent (Ctrl-R, searching the history, is built in).
        enable_open_in_editor=True,
        enable_suspend=True,
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
        console: Console | None = None,
        servers: Sequence[ServerStatus] = (),
        hub: McpHub | None = None,
    ) -> None:
        """Create a REPL for an agent.

        Args:
            agent: The agent that answers prompts.
            commands: The slash commands, by name.
            session: The line editor that reads the prompts (see
                `terminal_session`); without one, `input()` does.
            console: What prints; one that already printed (the start-up notices)
                keeps the state that separates its output from the next.
            servers: What became of the MCP servers at start, for `/servers`.
            hub: The running MCP servers, which `/reconnect` starts again; its
                statuses are the servers `/servers` lists.
        """
        self.agent = agent
        self.console = console or Console()
        if session:
            # An answer to "Allow …?" is not a prompt: it has its own editor, so it
            # is not kept in the history or completed as a command.
            self.console.ask = PromptSession(
                input=session.app.input, output=session.app.output
            ).prompt
            session.bottom_toolbar = self._toolbar
        self.hub = hub
        self.servers = hub.statuses if hub is not None else servers
        self.running = False
        self.commands = commands
        self.session = session

    def _toolbar(self) -> str:
        """The line under the prompt: the model, the effort and how full the context
        is, as `/context` counts it."""
        agent = self.agent
        config = agent.config
        used = agent.request_tokens()
        estimated = "" if agent.context.counted is not None else "~"
        effort = (
            f" | effort: {config.reasoning_effort}" if config.reasoning_effort else ""
        )
        return (
            f" {config.model}{effort} | context {estimated}"
            f"{used * 100 // config.max_context_tokens}% of {config.max_context_tokens:,}"
        )

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
        interrupted = False  # the last prompt ended in Ctrl-C or Ctrl-D
        while self.running:
            try:
                prompt = (
                    self.session.prompt("> ") if self.session else input("> ")
                ).strip()
            except EOFError:
                # Redirected input ends here. At a terminal a stray Ctrl-D must not
                # end a conversation that lives only in memory.
                context = self.agent.context
                if (
                    interrupted
                    or not self.session
                    or not (context.items or context.summary)
                ):
                    break
                interrupted = True
                self.console.notice("Press Ctrl-D again, or type /exit, to quit")
                continue
            except KeyboardInterrupt:
                # The conversation lives only in memory: one stray Ctrl-C must not
                # end it.
                if interrupted:
                    break
                interrupted = True
                self.console.notice("Press Ctrl-C again, or type /exit, to quit")
                continue
            interrupted = False
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

    def apply(self, update: ServerUpdate) -> None:
        """Give the agent the tools and the notes of a server started again, and say so.

        Args:
            update: What the hub returned for the server.
        """
        self.agent.update_server(
            update.removed, update.tools, update.name, update.instructions
        )
        self.console.notice(update.message)

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
