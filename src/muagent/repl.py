"""The input loop: reads prompts, dispatches slash commands, runs turns.

The line editor is prompt_toolkit (history, completion, bracketed paste); macOS's
Python has libedit, which has no bracketed paste, so a pasted block would be many
prompts. With input or output redirected there is no session and `input()` reads.
"""

import sys
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application, create_app_session
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import AnyFormattedText, FormattedText
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import Output
from prompt_toolkit.shortcuts.choice_input import ChoiceInput
from prompt_toolkit.styles import Style

from muagent.agent import Agent
from muagent.commands import Command
from muagent.console import Console
from muagent.errors import TurnError, describe_error
from muagent.mcp_hub import McpHub, ServerStatus, ServerUpdate

# A terminal asked for Shift-Enter (kitty's keyboard protocol, or xterm's
# modifyOtherKeys) sends one of these, which prompt_toolkit does not know. They are
# read as Alt-Enter. A terminal that sends a plain Enter for Shift-Enter has to be set
# to send one of them (or Esc followed by Enter) for the key to tell them apart.
for _sequence in ("\x1b[13;2u", "\x1b[27;2;13~"):
    ANSI_SEQUENCES[_sequence] = (Keys.Escape, Keys.ControlM)

# Colour is the terminal's own (the ANSI names), and prompt_toolkit drops it all when
# NO_COLOR is set. The toolbar is not reversed, which would make it a loud bar. The menu
# says `fg` and `bg` both: prompt_toolkit's own menu is black on grey, and a style that
# sets only one of them would leave black text on the terminal's background.
_STYLE = Style.from_dict(
    {
        "prompt": "ansicyan bold",
        "continuation": "ansibrightblack",
        "placeholder": "dim",
        "bottom-toolbar": "noreverse ansibrightblack",
        "toolbar.strong": "fg:default bold",
        "toolbar.meter": "fg:default",
        "toolbar.warn": "ansiyellow",
        "toolbar.bad": "ansired",
        "selected-option": "ansicyan bold",
        "number": "ansibrightblack",
        "completion-menu": "fg:default bg:default",
        "completion-menu.completion": "fg:default bg:default",
        "completion-menu.completion.current": "fg:ansicyan bg:default bold reverse",
        "completion-menu.meta.completion": "fg:ansibrightblack bg:default",
        "completion-menu.meta.completion.current": "fg:ansibrightblack bg:default",
        "scrollbar.background": "bg:default",
        "scrollbar.button": "bg:ansibrightblack",
    }
)

# The prompt at a terminal; redirected input is read with a plain "> ".
_PROMPT = FormattedText([("class:prompt", "❯ ")])

# How many cells the context meter in the toolbar has, and the shares of the budget at
# which it turns yellow and red.
_METER_CELLS = 10
_METER_WARN = 60
_METER_BAD = 85

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


def _continuation(width: int, line_number: int, wrap_count: int) -> FormattedText:
    """The marker in front of a line added by hand or pasted, lined up under the
    prompt; a line that only wraps has none."""
    return FormattedText(
        [("class:continuation", ("· " if not wrap_count else "").rjust(width))]
    )


class _Menu(ChoiceInput[str]):
    """prompt_toolkit's choice menu, gone from the screen once it is answered."""

    def _create_application(self) -> Application[str]:
        # The menu would leave all its options in the scrollback; `Console` prints one
        # line of what was decided instead.
        app = super()._create_application()
        app.erase_when_done = True
        return app


def _choose(
    session: PromptSession[str],
    message: AnyFormattedText,
    options: Sequence[tuple[str, str]],
) -> str:
    """Ask a question on the session's terminal with a menu, and give the key of the
    option chosen. The first option is the default; arrows and Enter, the option's
    number or its key choose. Ctrl-C raises KeyboardInterrupt, as at the prompt.

    Args:
        session: The line editor, whose input and output the menu uses.
        message: The question.
        options: The (key, label) of each option.
    """
    keys = KeyBindings()
    for key, _ in options:
        keys.add(key)(_exit_with(key))
    with create_app_session(input=session.app.input, output=session.app.output):
        return _Menu(
            message=message,
            options=options,
            default=options[0][0],
            style=_STYLE,
            symbol="❯",
            key_bindings=keys,
        ).prompt()


def _exit_with(key: str):
    def handler(event) -> None:
        event.app.exit(result=key, style="class:accepted")

    return handler


class _CommandCompleter(Completer):
    """Completes a slash command's name, then the arguments it lists.

    Not prompt_toolkit's NestedCompleter: that one finds the first word with a
    pattern that does not take the slash as part of it.
    """

    def __init__(self, commands: Mapping[str, Command]) -> None:
        self.commands = commands
        # Set by the Repl that uses this editor: some arguments come from it.
        self.repl: Repl | None = None

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        name, space, rest = document.text_before_cursor.lstrip().partition(" ")
        if not space:
            # A command is shown with what it does, so no one has to run /help.
            words = {n: self.commands[n].description for n in sorted(self.commands)}
            typed = name
        elif name in self.commands:
            arguments = self.commands[name].arguments
            if callable(arguments):
                arguments = arguments(self.repl) if self.repl is not None else ()
            words = dict.fromkeys(arguments, "")
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
        prompt_continuation=_continuation,
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
            # An answer to "Allow …?" is not a prompt: it is a menu of its own, so it
            # is not kept in the history or completed as a command.
            self.console.choose = partial(_choose, session)
            session.bottom_toolbar = self._toolbar
            if isinstance(session.completer, _CommandCompleter):
                session.completer.repl = self
        self.hub = hub
        self.servers = hub.statuses if hub is not None else servers
        self.running = False
        self.commands = commands
        self.session = session

    def _toolbar(self) -> FormattedText:
        """The line under the prompt: the model, the effort, the servers and how full
        the context is, as `/context` counts it, with a meter that turns yellow and
        then red as it fills (the share is written too, for a terminal without colour).
        """
        agent = self.agent
        config = agent.config
        used = agent.request_tokens()
        estimated = "" if agent.counted_tokens() is not None else "~"
        share = used * 100 // config.max_context_tokens
        lit = min(share * _METER_CELLS // 100, _METER_CELLS)
        meter = (
            "class:toolbar.bad"
            if share >= _METER_BAD
            else "class:toolbar.warn"
            if share >= _METER_WARN
            else "class:toolbar.meter"
        )
        bar: list[tuple[str, str]] = [("class:toolbar.strong", " " + config.model)]
        if config.reasoning_effort:
            bar.append(("", f" · {config.reasoning_effort}"))
        if self.hub is not None:
            self.hub.poll()  # servers that have all answered are counted at once
            if self.hub.starting:
                bar.append(("", " · starting MCP servers"))
            else:
                tools = sum(len(status.tools) for status in self.hub.statuses)
                failed = sum(status.state == "failed" for status in self.hub.statuses)
                if tools:
                    bar.append(("", f" · {tools} tools"))
                if failed:
                    bar.append(
                        ("class:toolbar.bad", f" · {failed} MCP failed: /servers")
                    )
        bar += [
            ("", "  "),
            (meter, "▰" * lit),
            ("", "▱" * (_METER_CELLS - lit)),
            ("", f" {estimated}{share}% of {config.max_context_tokens:,}"),
        ]
        return FormattedText(bar)

    def run(self) -> None:
        """Read and handle prompts until the user quits."""
        self.console.write(
            f"muagent · {self.agent.config.model} · /help for commands\n\n"
        )

        self.running = True
        interrupted = False  # the last prompt ended in Ctrl-C or Ctrl-D
        while self.running:
            if self.session and self.hub is not None:
                # The toolbar is redrawn while servers start, to show when they are up.
                self.session.refresh_interval = 0.5 if self.hub.starting else 0
            try:
                prompt = (
                    self.session.prompt(_PROMPT) if self.session else input("> ")
                ).strip()
            except EOFError:
                # Redirected input ends here. At a terminal a stray Ctrl-D must not
                # end a conversation that lives only in memory.
                if interrupted or not self.session or not self.agent.has_conversation:
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
        self.attach(wait=False)
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
        """Give the agent the tools and the notes of a server, and say what the hub
        said about it, if anything.

        Args:
            update: What the hub returned for the server.
        """
        skipped = self.agent.update_server(
            update.removed, update.tools, update.name, update.instructions
        )
        if update.message:
            self.console.notice(update.message)
        for tool in skipped:
            self.console.notice(
                f"Tool {tool} of server {update.name} is not offered: "
                "another tool has that name"
            )

    def attach(self, *, wait: bool) -> None:
        """Give the agent the servers that started in the background, and tell of the
        ones that failed.

        This is between inputs, never in the middle of a turn: the tools of a turn
        must not change under it.

        Args:
            wait: Whether to wait for the servers that have not answered yet. A turn
                waits (it needs the tools); a command does not.
        """
        if self.hub is not None:
            for update in self.hub.take(self.console.notice, wait=wait):
                self.apply(update)

    def ask(self, prompt: str) -> None:
        """Run one turn, printing the reply as it arrives and reporting
        problems without leaving the loop.

        Args:
            prompt: The user's input prompt.
        """
        console = self.console
        console.blank()
        try:
            if self.hub is not None and self.hub.starting:
                console.notice("Waiting for the MCP servers…")
            self.attach(wait=True)  # the tools are needed now
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
