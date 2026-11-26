"""All terminal output, and the question that asks permission for a tool.

Output is plain text (a streamed reply, a command's output) or a status block (a
notice, a call, the question). Both go through `Console._emit`, the one place that
keeps them apart: a status block starts on a fresh line, and plain text after one
starts after a blank line. A new kind of output goes through `_emit` too. Replies and
notices carry text from the model or a server, so everything is `sanitize`d first.

A status block is a list of styled pieces, as prompt_toolkit has them. On a terminal
with colour they are printed by prompt_toolkit; otherwise only their text is, since
`print_formatted_text` ends every line with `\\r\\n`, which a file or a pipe would keep.
"""

import io
import json
import os
import re
import shutil
import sys
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import AnyFormattedText, FormattedText
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style

from ergane.markdown import Markdown
from ergane.tools import Tool

# The look of what is not a reply: the colours are the terminal's own (ANSI names).
_STYLE = Style.from_dict(
    {
        "notice": "dim",
        "warning": "ansiyellow",
        "failure": "ansired",
        "call.name": "bold",
        "call.args": "dim",
        "allowed": "dim",
        "denied": "ansired",
        "title": "bold",
        "heading": "dim bold",
        "label": "bold",
        # The preview of a change: independent names, as a dotted one inherits.
        "diff-file": "bold",
        "diff-hunk": "dim",
        "diff-add": "ansigreen",
        "diff-del": "ansired",
        "fence": "dim",
        "md-bold": "bold",
        "md-code": "ansicyan",
        "todo-done": "dim",
        "todo-active": "bold",
    }
)

# How serious a notice is: it is told by colour, and the words still say it.
Level = Literal["info", "warning", "error"]
_NOTICE_STYLES = {
    "info": "class:notice",
    "warning": "class:warning",
    "error": "class:failure",
}

# Asks a question with a menu: the question, and the options as (key, label); gives the
# key of the one chosen (the first is the default). `Repl` supplies it on a terminal.
Choose = Callable[[AnyFormattedText, Sequence[tuple[str, str]]], str]

# Control characters that could manipulate the terminal (escape sequences such as
# OSC 52 clipboard writes, cursor movement, BEL, carriage return), Unicode bidi
# marks, overrides and isolates that can make text display misleadingly, and the
# invisible tag characters that can hide text. Newline and tab are kept, and so are
# the zero-width joiners that Persian, Urdu and emoji text need. Removing ESC alone
# defuses an escape sequence, so this works on any chunk of a stream, even one that
# splits a sequence in two.
_UNSAFE = re.compile(
    r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069"
    r"\U000e0000-\U000e007f]"
)


def sanitize(text: str) -> str:
    """Remove characters that could manipulate the terminal or spoof its text.

    Args:
        text: Text from the model or the server.

    Returns:
        The text without control characters (except newline and tab) and without
        bidi overrides and isolates.
    """
    return _UNSAFE.sub("", text)


def _cut_strings(value: Any, cap: int) -> Any:
    """The JSON value with every string longer than `cap` cut in the middle.

    Both ends are kept, because a long command can be harmless at the start and
    not at the end.
    """
    if isinstance(value, str):
        if len(value) <= cap:
            return value
        head = (cap + 1) // 2
        return value[:head] + "…" + value[len(value) - (cap - head) :]
    if isinstance(value, list):
        return [_cut_strings(item, cap) for item in value]
    if isinstance(value, dict):
        return {key: _cut_strings(item, cap) for key, item in value.items()}
    return value


def _short(directory: Path) -> str:
    """A directory for the banner: under the home directory as `~/…`, and only its last
    two parts, after `…/`, when it would be long."""
    try:
        text = "~/" + directory.relative_to(Path.home()).as_posix()
    except ValueError:
        text = directory.as_posix()
    if len(text) > 40:
        text = "…/" + "/".join(directory.parts[-2:])
    return text


def _room(used: int) -> int:
    """How many characters of a terminal line are left after `used` (at least 20): a
    line that fills it exactly would wrap in some terminals."""
    return max(shutil.get_terminal_size().columns - used - 1, 20)


def _pairs(parsed: Any) -> str | None:
    """`key=value` for each argument of a call whose arguments are all plain values,
    which reads lighter than JSON; None for anything else (nested values, a list)."""
    if not isinstance(parsed, dict) or any(
        isinstance(value, dict | list) for value in parsed.values()
    ):
        return None

    def show(value: Any) -> str:
        if isinstance(value, str) and value and not re.search(r'[\s"]', value):
            return value
        return json.dumps(value, ensure_ascii=False)

    return " ".join(f"{key}={show(value)}" for key, value in parsed.items())


def _render(parsed: Any, pretty: bool) -> str:
    """The arguments as one line: `key=value` pairs if `pretty` and they allow it,
    otherwise JSON."""
    if pretty and (line := _pairs(parsed)) is not None:
        return line
    return json.dumps(parsed, ensure_ascii=False)


def _fit(arguments: str, room: int, pretty: bool = False) -> tuple[str, bool]:
    """A tool call's arguments on one line of at most `room` characters.

    The line is the JSON on one line, as it will be run (or, when `pretty`, `key=value`
    pairs, for a line that only reports a call). If that is too long, the long strings
    are cut in the middle, the longest allowance that fits first; failing that, the
    line is cut at its end.

    Args:
        arguments: The model's arguments, as JSON text.
        room: The most characters the line may have.
        pretty: Whether to give plain arguments as `key=value` pairs.

    Returns:
        The line, and whether anything was left out of it.
    """
    try:
        parsed = json.loads(arguments)
    except ValueError:
        line = " ".join(arguments.split())  # not JSON: nothing but its whitespace
    else:
        line = _render(parsed, pretty)
        if len(line) > room:
            for cap in (64, 40, 24, 16, 8):
                line = _render(_cut_strings(parsed, cap), pretty)
                if len(line) <= room:
                    return line, True
    if len(line) <= room:
        return line, False
    return line[: room - 1] + "…", True


class Console:
    """Terminal output that is safe to show and keeps its blocks apart.

    Text is sanitized, since replies and notices carry text from the model or the
    server. There are two kinds of output: plain text (a streamed reply, a command's
    output) and status blocks (notices, calls and questions). Both go through _emit(),
    which keeps them apart: a status block always starts on a fresh line, and plain
    text that follows one starts after a blank line.
    """

    def __init__(
        self,
        choose: Choose | None = None,
        output: Output | None = None,
        *,
        can_ask: bool = True,
    ) -> None:
        """Create a console, with the cursor at the start of a line.

        Args:
            choose: Asks the question of `confirm` with a menu, on a terminal; without
                one the question is typed, and `input()` reads the answer.
            output: Where styled text is written, as prompt_toolkit has it; the
                terminal if omitted. Giving one turns styling on.
            can_ask: Whether there is a user to answer `confirm`. False when input is
                not a terminal: the next line of a file or a pipe is no one's answer.
        """
        self.can_ask = can_ask
        self.choose = choose
        self.output = output
        # Status lines are styled on a terminal, unless NO_COLOR asks for no colour.
        self.color = (output is not None or sys.stdout.isatty()) and not os.environ.get(
            "NO_COLOR"
        )
        self.at_line_start = True
        self.after_status = False  # whether the last output was a status block
        # Whether the last thing printed, since the last status block, was text with
        # something in it (a reply): a status block after it is set apart by a blank line.
        self.after_text = False
        # Newlines at the end of what was printed, at most 2: the cursor starts as if
        # after a blank line, so that nothing is set apart from nothing.
        self.trailing = 2
        self._markdown = Markdown()  # the reply that is being written
        # Text the terminal's encoding cannot show is replaced, not an error that
        # would end the turn halfway through a reply.
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout.reconfigure(errors="replace")

    def _emit(self, pieces: Sequence[tuple[str, str]], *, status: bool) -> None:
        """Print sanitized text, separating status blocks from plain text.

        Args:
            pieces: The (style, text) pieces to print, without a newline added; the
                style is prompt_toolkit's, such as "class:notice".
            status: Whether the text is a status block (a notice, a call or a
                question).
        """
        pieces = [(style, sanitize(text)) for style, text in pieces]
        text = "".join(piece for _, piece in pieces)
        if not text:
            return
        styled = self.color and any(style for style, _ in pieces)  # not a reply
        if status:
            # Never inside a line of reply, and a blank line after one.
            lead = "\n" * max((2 if self.after_text else 1) - self.trailing, 0)
            self._markdown.reset()
        elif self.after_status and text.strip():
            lead = "\n"  # a reply stands apart from the status lines
        else:
            lead = ""
        if lead:
            pieces.insert(0, ("", lead))
            text = lead + text
        if styled:
            print_formatted_text(
                FormattedText(pieces),
                end="",
                flush=True,
                style=_STYLE,
                output=self.output,
                file=None if self.output else sys.stdout,
            )
        else:
            print(text, end="", flush=True)
        self.at_line_start = text.endswith("\n")
        newlines = len(text) - len(text.rstrip("\n"))
        # Text that is only newlines adds to those before it.
        self.trailing = min(
            self.trailing + newlines if not text.strip("\n") else newlines, 2
        )
        self.after_status = status
        if status:
            self.after_text = False
        elif text.strip():
            self.after_text = True

    def write(self, text: str) -> None:
        """Print plain text as it is, without adding a newline.

        Args:
            text: Text to print, such as a piece of a streamed reply.
        """
        pieces = self._markdown.feed(text)
        self._emit(pieces, status=False)

    def blank(self) -> None:
        """Print an empty line."""
        self.write("\n")

    def columns(
        self, rows: Sequence[tuple[str, str]], heading: str | None = None
    ) -> None:
        """Print rows of a label and a text, the texts lined up, and a blank line.

        A text longer than the terminal is wrapped at its spaces, the next lines
        lined up under the first.

        Args:
            rows: The labels and what to say of each. Each line is indented by two
                spaces, and the label is in bold.
            heading: A line above the rows, dim.
        """
        width = max(len(label) for label, _ in rows)
        indent = " " * (width + 4)
        room = _room(len(indent))
        if heading:
            self._emit([("class:heading", f"  {heading}\n")], status=False)
        for label, text in rows:
            lines = textwrap.wrap(
                text, room, break_long_words=False, break_on_hyphens=False
            ) or [""]
            self._emit(
                [
                    ("", "  "),
                    ("class:label", f"{label:<{width}}"),
                    ("", f"  {lines[0]}\n"),
                ],
                status=False,
            )
            for line in lines[1:]:
                self.write(f"{indent}{line}\n")
        self.blank()

    def banner(self, model: str | None, directory: Path | None = None) -> None:
        """Print the line that opens a session: the name in bold, the rest dim. The
        model is named only where there is no toolbar to show it, and the directory is
        the one the file tools work in (shortened, if it is long)."""
        rest = f" · {model}" if model else ""
        if directory is not None:
            rest += f" · {_short(directory)}"
        self._emit(
            [
                ("class:title", "ergane"),
                ("class:notice", f"{rest} · /help for commands\n"),
                ("", "\n"),
            ],
            status=False,
        )

    def todos(self, report: str) -> None:
        """Print the model's checklist after it changed it: a dim heading, then a line
        for each step, the finished ones dim and the one under way bold.

        Args:
            report: `TodoList.report()`: the heading, then one line to a step, each
                starting with its mark (`[x]`, `[>]`, `[ ]` or `[-]`).
        """
        heading, _, steps = report.partition("\n")
        pieces = [("class:heading", heading + "\n")]
        for line in steps.splitlines():
            mark = line[:3]
            style = {
                "[x]": "class:todo-done",
                "[-]": "class:todo-done",
                "[>]": "class:todo-active",
            }.get(mark, "")
            pieces.append((style, line + "\n"))
        self._emit(pieces, status=True)

    def end_line(self) -> None:
        """Finish the current line, if one was started."""
        if not self.at_line_start:
            self.write("\n")

    def notice(self, text: str, level: Level = "info") -> None:
        """Print a status line.

        Args:
            text: The status message; it is flattened to one line, so text from
                the server cannot pass for something else.
            level: How serious it is: dim and plain for what is routine, and in
                brackets, yellow or red, for what the user may want to act on or what
                went wrong (the brackets say so without colour).
        """
        flat = " ".join(text.split())
        lines = textwrap.wrap(
            flat if level == "info" else f"[{flat}]",
            _room(0),
            subsequent_indent=" ",
            break_on_hyphens=False,
        )
        self._emit([(_NOTICE_STYLES[level], "\n".join(lines) + "\n")], status=True)

    def _call_line(self, name: str, arguments: str, outcome: str, style: str) -> None:
        """Print a tool call as one status line: `▸ name  arguments`, then `outcome`.

        The line is as wide as the terminal at most, with the middle of long strings
        in the arguments left out (see `confirm`).
        """
        name = " ".join(name.split())  # one line, whatever the server called it
        used = len(f"▸ {name}  ") + (len(f"  {outcome}") if outcome else 0)
        shown, _ = _fit(arguments, _room(used), pretty=True)
        self._emit(
            [
                ("class:call.name", f"▸ {name}"),
                ("class:call.args", f"  {shown}" if shown else ""),
                (style, f"  {outcome}" if outcome else ""),
                ("", "\n"),
            ],
            status=True,
        )

    def call(self, name: str, arguments: str) -> None:
        """Show a tool call that runs without asking, as one status line.

        Args:
            name: The tool's name.
            arguments: The model's arguments, as JSON text.
        """
        self._call_line(name, arguments, "", "")

    def _ask_typed(self, tool: Tool, arguments: str) -> str:
        """The question as a line the user types the answer to: "y", "a", "?" or "n"
        (anything else, and the end of input, is a no)."""
        before = f"Allow {tool.name} "
        shown, shortened = _fit(arguments, _room(len(before) + len("? [y/N/a/?] ")))
        more = shortened or tool.preview is not None
        prompt = f"{before}{shown}? [y/N/a{'/?' if more else ''}] "
        self.at_line_start = False  # the prompt leaves the cursor after it
        self.trailing = 0
        try:
            answer = input(sanitize(prompt)).strip().lower()
        except EOFError:
            print()  # no Enter ended the line
            answer = ""
        self.at_line_start = True
        self.trailing = 1
        return {"y": "y", "yes": "y", "a": "a", "always": "a", "?": "?"}.get(
            answer, "n"
        )

    def _ask_menu(self, choose: Choose, tool: Tool, arguments: str) -> str:
        """The question as a menu (see `Choose`), whose first option, the default, is
        the no. It is gone when answered: `confirm` prints a line of what was decided."""
        name = " ".join(sanitize(tool.name).split())
        shown, shortened = _fit(arguments, _room(len(f"Allow {name} ") + 4))
        question = FormattedText(
            [
                ("", "Allow "),
                ("class:call.name", name),
                ("class:call.args", f" {sanitize(shown)}"),
                ("", "?"),
            ]
        )
        options = [("n", "Deny"), ("y", "Allow once"), ("a", f"Always allow {name}")]
        if tool.preview is not None:
            options.append(("?", "Show the change"))
        elif shortened:
            options.append(("?", "Show full arguments"))
        return choose(question, options)

    @staticmethod
    def _everything(tool: Tool, arguments: str) -> list[tuple[str, str]]:
        """What the user asked to see of a call, as styled lines: the change, if the tool
        can say what it would be (a diff, whose lines are coloured by what they do),
        otherwise the arguments in full."""
        if tool.preview is None:
            try:
                text = json.dumps(json.loads(arguments), indent=2, ensure_ascii=False)
            except ValueError:
                text = " ".join(arguments.split())
            return [("", text + "\n")]
        try:
            text = tool.preview(arguments)
        except Exception as exc:  # noqa: BLE001
            return [("", f"Could not show the change: {type(exc).__name__}: {exc}\n")]
        pieces = []
        for number, line in enumerate(text.split("\n")):
            if number == 0 and line.startswith("--- "):
                style = "class:diff-file"
            elif line.startswith("@@"):
                style = "class:diff-hunk"
            elif line.startswith("+"):
                style = "class:diff-add"
            elif line.startswith("-"):
                style = "class:diff-del"
            else:
                style = ""
            pieces.append((style, line + "\n"))
        return pieces

    def confirm(self, tool: Tool, arguments: str) -> bool | Literal["always"]:
        """Ask the user whether a tool that is not read-only may run.

        The question is one line, as wide as the terminal at most, so the arguments
        are shown whole if they fit and otherwise with the middle of their long
        strings left out, which the question says (`?` at the prompt, "Show full
        arguments" in the menu). Asking for them prints the arguments in full and asks
        again, so nothing is hidden for good. The line is the JSON that will be run, on
        one line: padding between its tokens cannot push the call off the screen. A tool
        that has a `preview` always offers `?`, "Show the change", which prints what the
        call would change (a diff) instead of the arguments.

        On a terminal the question is a menu with Deny first, which Enter picks; y, a,
        n and ? choose too. When it is answered the menu is gone and a line of the
        call and what was decided is left, as a typed question leaves its answer.

        Args:
            tool: The tool the model wants to call.
            arguments: The model's arguments, as JSON text.

        Returns:
            True if the user allows it (y or yes), "always" for a or always (allow
            this tool for the rest of the session), and False otherwise; end of input
            is a no. With no one to ask (`can_ask`) it is False, and the call's line
            says why.
        """
        if not self.can_ask:
            self._call_line(
                tool.name, arguments, "denied: input is not a terminal", "class:denied"
            )
            return False
        while True:
            self.end_line()
            if self.choose is not None:
                answer = self._ask_menu(self.choose, tool, arguments)
            else:
                answer = self._ask_typed(tool, arguments)
            if answer != "?":
                break
            # Set apart from the call line above it and from the question below it.
            self._emit(
                [("", "\n"), *self._everything(tool, arguments), ("", "\n")],
                status=True,
            )
        # A reply to what was typed stands apart from it, as from a status line.
        self.after_status = True
        if self.choose is not None:
            self.at_line_start = True  # the menu was erased
            self.trailing = 1
            decided = {"y": "allowed", "a": "always allowed"}.get(answer, "denied")
            style = "class:allowed" if answer in ("y", "a") else "class:denied"
            self._call_line(tool.name, arguments, decided, style)
        if answer == "a":
            return "always"
        return answer == "y"

    def clear(self) -> None:
        """Clear the screen and scrollback, like `clear`; the history is kept."""
        # Not when output is redirected, where the escape codes would be garbage.
        if sys.stdout.isatty():
            print("\033[2J\033[3J\033[H", end="", flush=True)
            self.at_line_start = True
            self.after_status = False
            self.after_text = False
            self.trailing = 2
