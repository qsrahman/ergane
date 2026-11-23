"""Text helpers the built-in tools share: what counts as text, and how long output is cut.

A tool's result is cut from the end at `MAX_OUTPUT_CHARS` (20 KiB, in `tools.py`), which
would lose a footer such as "continue with offset=N". So each tool keeps its own result
under `MAX_TEXT_CHARS`, with room left for that footer, and says where it stopped.
"""

import codecs
from collections.abc import Sequence

# Well under MAX_OUTPUT_CHARS, so a tool's own footer always survives.
MAX_TEXT_CHARS = 16_000
# A line longer than this is cut, so one minified file cannot fill a whole result.
MAX_LINE_CHARS = 2_000
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
    least one line is taken. A line over `MAX_LINE_CHARS` is cut and says so.

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
        if len(line) > MAX_LINE_CHARS:
            line = f"{line[:MAX_LINE_CHARS]}... [line cut: {len(line):,} characters]"
        if taken and used + len(line) + 1 > max_chars:
            break
        taken.append(line)
        used += len(line) + 1
        index += 1
    return "\n".join(taken), index


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
