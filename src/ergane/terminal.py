"""The terminal: the line editor, its menus, and how they look.

The editor is prompt_toolkit (history, completion, bracketed paste); macOS's Python
has libedit, which has no bracketed paste, so a pasted block would be many prompts.
The question before a tool runs is a menu of the same library (`choose`), and the line
under the prompt is `toolbar`. `repl.py` is the loop that uses them.
"""

import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

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
from prompt_toolkit.lexers import SimpleLexer
from prompt_toolkit.output import Output
from prompt_toolkit.shortcuts.choice_input import ChoiceInput
from prompt_toolkit.styles import Style

from ergane.agent import Agent
from ergane.commands import Command
from ergane.mcp_hub import McpHub

if TYPE_CHECKING:
    from ergane.repl import Repl

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
        "typed": "bold",
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
PROMPT = FormattedText([("class:prompt", "❯ ")])
PLACEHOLDER = "Ask anything, or /help for commands"

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


def choose(
    session: PromptSession[str],
    message: AnyFormattedText,
    options: Sequence[tuple[str, str]],
) -> str:
    """Ask a question on the session's terminal with a menu, and give the key of the
    option chosen. The first option is the default; arrows and Enter, the option's
    number or its key (when the key is one character) choose. Ctrl-C raises
    KeyboardInterrupt, as at the prompt.

    Args:
        session: The line editor, whose input and output the menu uses.
        message: The question.
        options: The (key, label) of each option.
    """
    keys = KeyBindings()
    for key, _ in options:
        if len(key) == 1:  # a longer key, such as a session's id, is no key to press
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


class CommandCompleter(Completer):
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
            if not name and not complete_event.completion_requested:
                # Nothing typed but blanks: Alt-Enter or a space is no call for the
                # menu of every command. Tab is.
                return
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
        completer=CommandCompleter(commands),
        auto_suggest=AutoSuggestFromHistory(),
        # What is typed stays bold in the scrollback, so a prompt stands out from its reply.
        lexer=SimpleLexer("class:typed"),
        prompt_continuation=_continuation,
        key_bindings=_KEYS,
        style=_STYLE,
        # A plain string would get no style: only formatted text carries the class.
        placeholder=[("class:placeholder", PLACEHOLDER)],
        # Ctrl-X Ctrl-E edits a long prompt in $EDITOR and sends it when the editor
        # closes; Ctrl-Z suspends ergane (Ctrl-R, searching the history, is built in).
        enable_open_in_editor=True,
        enable_suspend=True,
        input=input,
        output=output,
    )


def erase_blank_prompt(session: PromptSession[str]) -> None:
    """Take a prompt that was sent empty off the screen. It still shows its placeholder,
    so pressing Enter on an empty line would leave a column of them in the scrollback.
    The cursor is at the start of the line after the prompt."""
    output = session.app.output
    width = max(output.get_size().columns, 1)
    output.cursor_up(-(-(len("❯ ") + len(PLACEHOLDER)) // width))
    output.erase_down()
    output.flush()


def _tokens(count: int) -> str:
    """A token count short enough for the toolbar: 842, 3.4k, 66k."""
    if count < 1000:
        return str(count)
    return f"{count / 1000:.1f}k" if count < 10_000 else f"{round(count / 1000)}k"


def _branch() -> str | None:
    """The git branch of the working directory (a short commit id on a detached HEAD),
    read from `.git/HEAD` as a prompt redraws the toolbar too often to run `git`; None
    outside a repository or when the file holds anything but a plain name."""
    here = Path.cwd()
    for directory in (here, *here.parents):
        git = directory / ".git"
        try:
            if git.is_file():  # a linked worktree: "gitdir: PATH"
                git = Path(git.read_text().removeprefix("gitdir:").strip())
                git = directory / git
            head = (git / "HEAD").read_text().strip()
        except OSError:
            continue
        name = head.removeprefix("ref: refs/heads/")
        if name == head:
            name = head[:7]  # detached
        # The file is the repository's, which may not be the user's: no control characters.
        return name[:30] if name.isprintable() and name else None
    return None


def toolbar(agent: Agent, hub: McpHub) -> FormattedText:
    """The line under the prompt: the model, the effort, the servers and how full
    the context is, as `/context` counts it (what does not fit the terminal's width is
    left out, the least important first, never the meter), with a meter that turns yellow and
    then red as it fills (the share is written too, for a terminal without colour, and
    the meter's two glyphs differ in weight, not only in colour). They are box-drawing
    characters, which a terminal draws itself, so they fill their cells in any font.
    """
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
    # What goes between the model and the meter, as (how soon it is dropped when the line
    # is too wide for the terminal, style, text): the meter, which is what the toolbar
    # is for, is the last thing to go, and a failed server and the tools allowed
    # without asking are what the user must not miss.
    extras: list[tuple[int, str, str]] = []
    if config.reasoning_effort:
        extras.append((3, "", f" · {config.reasoning_effort}"))
    if branch := _branch():
        extras.append((1, "", f" · git:{branch}"))
    if allowed := sorted(agent.always_allowed):
        shown = ", ".join(allowed) if len(allowed) <= 2 else f"{len(allowed)} tools"
        extras.append((4, "class:toolbar.warn", f" · always: {shown}"))
    hub.poll()  # servers that have all answered are counted at once
    if hub.starting:
        extras.append((2, "", " · starting MCP servers"))
    else:
        statuses = hub.statuses.values()
        tools = sum(len(status.tools) for status in statuses)
        failed = sum(status.state == "failed" for status in statuses)
        if tools:
            extras.append((2, "", f" · {tools} tools"))
        if failed:
            extras.append((5, "class:toolbar.bad", f" · {failed} MCP failed: /servers"))
    tail = [
        ("", "  "),
        (meter, "━" * lit),
        ("", "─" * (_METER_CELLS - lit)),
        (
            "",
            f"  {estimated}{share}% · {_tokens(used)} / {_tokens(config.max_context_tokens)}",
        ),
    ]
    head = " " + config.model
    room = shutil.get_terminal_size().columns - 1 - len(head)
    room -= sum(len(text) for _, text in tail)
    while extras and sum(len(text) for _, _, text in extras) > room:
        extras.remove(min(extras, key=lambda extra: extra[0]))
    bar = [
        ("class:toolbar.strong", head),
        *((style, text) for _, style, text in extras),
        *tail,
    ]
    return FormattedText(bar)
