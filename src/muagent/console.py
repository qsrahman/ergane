"""Terminal output: everything the program prints goes through Console."""

import io
import json
import re
import sys

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


class Console:
    """Terminal output that is safe to show and keeps its blocks apart.

    Text is sanitized, since replies and notices carry text from the model or the
    server. There are two kinds of output: plain text (a streamed reply, a command's
    output) and status blocks (notices and questions). Both go through _emit(),
    which keeps them apart: a status block always starts on a fresh line, and plain
    text that follows one starts after a blank line.
    """

    def __init__(self) -> None:
        """Create a console, with the cursor at the start of a line."""
        self.at_line_start = True
        self.after_status = False  # whether the last output was a status block
        # Text the terminal's encoding cannot show is replaced, not an error that
        # would end the turn halfway through a reply.
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout.reconfigure(errors="replace")

    def _emit(self, text: str, *, status: bool) -> None:
        """Print sanitized text, separating status blocks from plain text.

        Args:
            text: The text to print, without a newline added.
            status: Whether the text is a status block (a notice or a question).
        """
        text = sanitize(text)
        if not text:
            return
        if status and not self.at_line_start:
            text = "\n" + text  # never inside a line of reply
        elif self.after_status and not status and text.strip():
            text = "\n" + text  # a reply stands apart from the status lines before it
        print(text, end="", flush=True)
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
        self._emit(f"[{' '.join(sanitize(text).split())}]\n", status=True)

    def confirm(self, tool: Tool, arguments: str) -> bool:
        """Ask the user whether a tool that is not read-only may run.

        The arguments are shown in full, as the JSON that will be run: a shortened
        call could hide what it does, and padding between the JSON tokens could
        scroll it off the screen.

        Args:
            tool: The tool the model wants to call.
            arguments: The model's arguments, as JSON text.

        Returns:
            True only if the user answers y or yes; end of input is a no.
        """
        try:
            shown = json.dumps(json.loads(arguments), indent=2, ensure_ascii=False)
        except ValueError:
            shown = " ".join(arguments.split())
        self._emit(f"Allow {tool.name} with these arguments?\n{shown}\n", status=True)
        try:
            answer = input("[y/N] ")
        except EOFError:
            answer = ""
        return answer.strip().lower() in ("y", "yes")

    def clear(self) -> None:
        """Clear the screen and scrollback, like `clear`; the history is kept."""
        # Not when output is redirected, where the escape codes would be garbage.
        if sys.stdout.isatty():
            print("\033[2J\033[3J\033[H", end="", flush=True)
            self.at_line_start = True
            self.after_status = False
