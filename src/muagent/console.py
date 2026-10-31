"""Terminal output: everything the program prints goes through Console."""

import re
import sys

# Control characters that could manipulate the terminal (escape sequences such as
# OSC 52 clipboard writes, cursor movement, BEL, carriage return) and Unicode bidi
# overrides and isolates that can make text display misleadingly. Newline and tab
# are kept. Removing ESC alone defuses an escape sequence, so this works on any
# chunk of a stream, even one that splits a sequence in two.
_UNSAFE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


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
    """Terminal output that is safe to show and knows where the cursor is.

    Text is sanitized, since replies and notices carry text from the model or the
    server. The console also tracks whether the cursor is at the start of a line,
    so a status line never lands in the middle of a streamed reply.
    """

    def __init__(self) -> None:
        """Create a console, with the cursor at the start of a line."""
        self.at_line_start = True

    def write(self, text: str) -> None:
        """Print text as it is, without adding a newline.

        Args:
            text: Text to print, such as a piece of a streamed reply.
        """
        text = sanitize(text)
        print(text, end="", flush=True)
        if text:
            self.at_line_start = text.endswith("\n")

    def blank(self) -> None:
        """Print an empty line."""
        self.write("\n")

    def end_line(self) -> None:
        """Finish the current line, if one was started."""
        if not self.at_line_start:
            self.write("\n")

    def notice(self, text: str) -> None:
        """Print a bracketed status line, starting a new line if needed.

        Args:
            text: The status message.
        """
        self.end_line()
        self.write(f"[{text}]\n")

    def clear(self) -> None:
        """Clear the screen and scrollback, like `clear`; the history is kept."""
        # Not when output is redirected, where the escape codes would be garbage.
        if sys.stdout.isatty():
            print("\033[2J\033[3J\033[H", end="", flush=True)
