import io
import json
import os
import re
import shutil
import sys
from collections.abc import Callable
from typing import Any, Literal

from muagent.tools import Tool

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


def _fit(arguments: str, room: int) -> tuple[str, bool]:
    """A tool call's arguments on one line of at most `room` characters.

    The line is the JSON on one line, as it will be run. If that is too long, the
    long strings are cut in the middle, the longest allowance that fits first;
    failing that, the line is cut at its end.

    Args:
        arguments: The model's arguments, as JSON text.
        room: The most characters the line may have.

    Returns:
        The line, and whether anything was left out of it.
    """
    try:
        parsed = json.loads(arguments)
    except ValueError:
        line = " ".join(arguments.split())  # not JSON: nothing but its whitespace
    else:
        line = json.dumps(parsed, ensure_ascii=False)
        if len(line) <= room:
            return line, False
        for cap in (64, 40, 24, 16, 8):
            line = json.dumps(_cut_strings(parsed, cap), ensure_ascii=False)
            if len(line) <= room:
                return line, True
        return line[: room - 1] + "…", True
    if len(line) <= room:
        return line, False
    return line[: room - 1] + "…", True


class Console:
    """Terminal output that is safe to show and keeps its blocks apart.

    Text is sanitized, since replies and notices carry text from the model or the
    server. There are two kinds of output: plain text (a streamed reply, a command's
    output) and status blocks (notices and questions). Both go through _emit(),
    which keeps them apart: a status block always starts on a fresh line, and plain
    text that follows one starts after a blank line.
    """

    def __init__(self, ask: Callable[[str], str] | None = None) -> None:
        """Create a console, with the cursor at the start of a line.

        Args:
            ask: Reads the answer to a question, given the question as its prompt;
                `input()` if omitted.
        """
        self._ask = ask
        # Status lines are dimmed on a terminal, unless NO_COLOR asks for no colour.
        self.color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        self.at_line_start = True
        self.after_status = False  # whether the last output was a status block
        # Text the terminal's encoding cannot show is replaced, not an error that
        # would end the turn halfway through a reply.
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout.reconfigure(errors="replace")

    def _emit(self, text: str, *, status: bool, dim: bool = False) -> None:
        """Print sanitized text, separating status blocks from plain text.

        Args:
            text: The text to print, without a newline added.
            status: Whether the text is a status block (a notice or a question).
            dim: Whether to dim it where there is colour. The codes are added
                after the text is sanitized, which would remove them.
        """
        text = sanitize(text)
        if not text:
            return
        if status and not self.at_line_start:
            text = "\n" + text  # never inside a line of reply
        elif self.after_status and not status and text.strip():
            text = "\n" + text  # a reply stands apart from the status lines before it
        shown = text
        if dim and self.color:
            body = text.strip("\n")
            shown = text.replace(body, f"\033[2m{body}\033[0m", 1)
        print(shown, end="", flush=True)
        self.at_line_start = text.endswith("\n")
        self.after_status = status

    def write(self, text: str) -> None:
        """Print plain text as it is, without adding a newline.

        Args:
            text: Text to print, such as a piece of a streamed reply.
        """
        self._emit(text, status=False)

    def blank(self) -> None:
        """Print an empty line."""
        self.write("\n")

    def end_line(self) -> None:
        """Finish the current line, if one was started."""
        if not self.at_line_start:
            self.write("\n")

    def notice(self, text: str) -> None:
        """Print a bracketed status line.

        Args:
            text: The status message; it is flattened to one line, so text from
                the server cannot break out of the brackets.
        """
        self._emit(f"[{' '.join(text.split())}]\n", status=True, dim=True)

    def call(self, name: str, arguments: str) -> None:
        """Show a tool call that runs without asking, as one status line.

        The line is as wide as the terminal at most, with the middle of long strings
        in the arguments left out (see `confirm`).

        Args:
            name: The tool's name.
            arguments: The model's arguments, as JSON text.
        """
        room = shutil.get_terminal_size().columns - len(name) - len("[()]") - 1
        shown, _ = _fit(arguments, max(room, 20))
        self.notice(f"{name}({shown})")

    def confirm(self, tool: Tool, arguments: str) -> bool | Literal["always"]:
        """Ask the user, on one line, whether a tool that is not read-only may run.

        The line is as wide as the terminal at most, so the arguments are shown
        whole if they fit and otherwise with the middle of their long strings left
        out, which the prompt says (`[y/N/a/?]`). Answering `?` prints the arguments
        in full and asks again, so nothing is hidden for good. The line is the JSON
        that will be run, on one line: padding between its tokens cannot push the
        call off the screen.

        Args:
            tool: The tool the model wants to call.
            arguments: The model's arguments, as JSON text.

        Returns:
            True if the user answers y or yes, "always" for a or always (allow this
            tool for the rest of the session), and False otherwise; end of input is
            a no.
        """
        before = f"Allow {tool.name} "
        room = (
            shutil.get_terminal_size().columns - len(before) - len("? [y/N/a/?] ") - 1
        )
        shown, shortened = _fit(arguments, max(room, 20))
        prompt = f"{before}{shown}? [y/N/a{'/?' if shortened else ''}] "
        while True:
            self.end_line()
            self.at_line_start = False  # the prompt leaves the cursor after it
            try:
                answer = (self._ask or input)(sanitize(prompt)).strip().lower()
            except EOFError:
                print()  # no Enter ended the line
                answer = ""
            self.at_line_start = True
            if answer != "?":
                break
            try:
                everything = json.dumps(
                    json.loads(arguments), indent=2, ensure_ascii=False
                )
            except ValueError:
                everything = " ".join(arguments.split())
            self._emit(everything + "\n", status=True)
        # A reply to what was typed stands apart from it, as from a status line.
        self.after_status = True
        if answer in ("a", "always"):
            return "always"
        return answer in ("y", "yes")

    def clear(self) -> None:
        """Clear the screen and scrollback, like `clear`; the history is kept."""
        # Not when output is redirected, where the escape codes would be garbage.
        if sys.stdout.isatty():
            print("\033[2J\033[3J\033[H", end="", flush=True)
            self.at_line_start = True
            self.after_status = False
