"""Light styling of a streamed reply: what a terminal can show without rendering Markdown.

The reply is printed as the model wrote it, marks included, so text still reads without
colour and nothing is lost when a mark is wrong. A fence line of a code block is dim, a
heading line bold, `**bold**` bold and `code` coloured, the marks inside the styled span.

The reply arrives in pieces of any size, so the state lives across them: whether a code
block is open, and the line so far. Only a line that is whole ends a block or a mark: a
`**` or a backtick left open stops at the end of its line instead of colouring the rest of
the reply. A `**` split between two pieces is not seen (the text is plain, as the model
wrote it). Imports nothing of ergane.
"""

import re

_TOKENS = re.compile(r"(\*\*|`)")
_HEADING = re.compile(r"#{1,6} ")


class Markdown:
    """Splits a streamed reply into (style, text) pieces, as prompt_toolkit has them."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forget the line and the marks that are open (a status block cut in)."""
        self.line = ""  # the line being written, before the piece in hand
        self.fenced = False  # inside a code block
        self.bold = False
        self.code = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        """Style a piece of the reply.

        Args:
            text: The next piece, of any size; it may end inside a line.

        Returns:
            The text in pieces of one style each; the styles are classes of
            `Console`'s style: `fence`, `md-bold` and `md-code`, or "" for none.
        """
        pieces: list[tuple[str, str]] = []
        for part in re.findall(r"[^\n]*\n|[^\n]+", text):
            line = self.line + part
            if line.lstrip().startswith("```"):
                pieces.append(("class:fence", part))
                if part.endswith("\n"):
                    self.fenced = not self.fenced
            elif self.fenced:
                pieces.append(("", part))
            elif _HEADING.match(line):
                pieces.append(("class:md-bold", part))
            else:
                pieces += self._inline(part)
            if part.endswith("\n"):
                self.line = ""
                self.bold = self.code = False
            else:
                self.line = line
        return pieces

    def _inline(self, part: str) -> list[tuple[str, str]]:
        """The pieces of text of one line with its `**` and backtick marks."""
        pieces = []
        for token in _TOKENS.split(part):
            if token == "`":
                self.code = not self.code
                # Both marks are in the span they open and close.
                pieces.append(("class:md-code", token))
            elif token == "**" and not self.code:
                self.bold = not self.bold
                pieces.append(("class:md-bold", token))
            elif token:
                style = (
                    "class:md-code"
                    if self.code
                    else "class:md-bold"
                    if self.bold
                    else ""
                )
                pieces.append((style, token))
        return pieces
