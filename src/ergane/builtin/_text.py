"""Text helpers the built-in tools share: what counts as text, and how long output is cut.

A tool's result is cut from the end at `MAX_OUTPUT_CHARS` (20 KiB, in `tools.py`), which
would lose a footer such as "continue with offset=N". So each tool keeps its own result
under `MAX_TEXT_CHARS`, with room left for that footer, and says where it stopped.
"""

import codecs
from collections.abc import Sequence
from typing import TextIO

# Well under MAX_OUTPUT_CHARS, so a tool's own footer always survives.
MAX_TEXT_CHARS = 16_000
# A line longer than this is cut, so one minified file cannot fill a whole result. Long
# enough for a line of prose or a README's long table row, which a cut at 2,000 broke
# with no way to read the rest.
MAX_LINE_CHARS = 10_000
# How much of a line is read at a time while skipping or measuring it.
_CHUNK = 65_536
# What `head_tail` keeps back for the note that says what was left out.
_NOTE_ROOM = 64


def is_text(data: bytes) -> bool:
    """Whether the start of a file looks like UTF-8 text: no NUL byte, and valid UTF-8.

    `data` may end inside a multi-byte character, as when it is only the start of a
    file: that is not an error.
    """
    if b"\0" in data:
        return False
    try:
        codecs.getincrementaldecoder("utf-8")().decode(data, final=False)
    except UnicodeDecodeError:
        return False
    return True


def take_lines(
    lines: Sequence[str],
    start: int = 0,
    count: int | None = None,
    max_chars: int = MAX_TEXT_CHARS,
) -> tuple[str, int]:
    """Join lines from `start`, whole lines only, until a limit is reached.

    The limits are `count` lines and `max_chars` characters, whichever comes first. At
    least one line is taken.

    Args:
        lines: The lines, without their line endings.
        start: The index of the first line to take.
        count: The most lines to take, or None for as many as fit.
        max_chars: The most characters in the result.

    Returns:
        The text, and the index after the last line taken: the caller names it in its
        footer, and it equals `len(lines)` when nothing is left.
    """
    taken: list[str] = []
    used = 0
    index = start
    while index < len(lines) and (count is None or len(taken) < count):
        line = lines[index]
        if taken and used + len(line) + 1 > max_chars:
            break
        taken.append(line)
        used += len(line) + 1
        index += 1
    return "\n".join(taken), index


def _skip_line(handle: TextIO) -> bool:
    """Move past one line without keeping it; False at the end of the file."""
    chunk = handle.readline(_CHUNK)
    if not chunk:
        return False
    while chunk and not chunk.endswith("\n"):
        chunk = handle.readline(_CHUNK)
    return True


def _next_line(handle: TextIO) -> str | None:
    """The next line without its newline, or None at the end of the file. A line over
    `MAX_LINE_CHARS` is cut and says how long it was; the rest of it is read in chunks
    and dropped, so a file with one enormous line does not fill the memory."""
    line = handle.readline(MAX_LINE_CHARS + 1)
    if not line:
        return None
    if len(line) <= MAX_LINE_CHARS or line.endswith("\n"):
        return line.rstrip("\n")
    length = len(line)
    while True:
        rest = handle.readline(_CHUNK)
        length += len(rest.rstrip("\n"))
        if not rest or rest.endswith("\n"):
            break
    return f"{line[:MAX_LINE_CHARS]}... [line cut: {length:,} characters]"


def read_lines(handle: TextIO, start: int, count: int) -> list[str]:
    """Up to `count` lines of an open text file from the 0-based line `start`, each cut
    at `MAX_LINE_CHARS` (see `_next_line`)."""
    for _ in range(start):
        if not _skip_line(handle):
            return []
    lines: list[str] = []
    while len(lines) < count and (line := _next_line(handle)) is not None:
        lines.append(line)
    return lines


def count_lines(handle: TextIO) -> int:
    """How many lines the rest of an open text file has."""
    count = 0
    while _skip_line(handle):
        count += 1
    return count


def head_tail(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    """The text, or if it is over `limit` its start and its end with a note between.

    For output whose end matters as much as its start, such as a command's: the start
    says what ran, and the failure is usually at the end. The result is never longer
    than `limit`.
    """
    if len(text) <= limit:
        return text
    half = (limit - _NOTE_ROOM) // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n[... {omitted:,} characters omitted ...]\n{text[-half:]}"
