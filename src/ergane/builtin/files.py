"""The tool that reads a file: `read_file`.

It only reads, so it runs without asking, on a thread, several at once: it keeps no state.
The path goes through `_paths.resolve` first and the path it returns is the one opened.

The result stays under `MAX_TEXT_CHARS` so that the footer that says where to continue is
never what the generic cut at 20 KiB removes.
"""

import io
from typing import Annotated, BinaryIO

from pydantic import Field

from ergane.builtin._images import picture_type, read_picture
from ergane.builtin._paths import File, not_found, regular_size, resolve
from ergane.builtin._text import (
    MAX_TEXT_CHARS,
    count_lines,
    read_lines,
    take_lines,
)
from ergane.tools import ToolOutput, tool

_SNIFF_BYTES = 8192


def _read_lines(
    raw: BinaryIO, shown: str, head: bytes, offset: int, limit: int | None
) -> str:
    """The lines of a text file from `offset`, cut at the character budget. `raw` is the
    file, open for reading, and `head` the start of it that was already read."""
    # Two different mistakes for the model to hear: it can read a Latin-1 file with
    # `run_command`, and cannot do anything with a binary one.
    if b"\0" in head:
        raise ValueError(f"{shown} looks like a binary file, not text")
    raw.seek(0)
    total = 0  # the lines of the file, counted only when there is nothing to show
    try:
        # utf-8-sig drops a byte order mark, which the model would see as a stray
        # character at the start of the first line.
        with io.TextIOWrapper(raw, encoding="utf-8-sig") as handle:
            lines = read_lines(handle, offset - 1, limit or MAX_TEXT_CHARS + 1)
            if not lines:
                handle.seek(0)
                total = count_lines(handle)
    except UnicodeDecodeError:
        raise ValueError(f"{shown} is not valid UTF-8 text") from None
    if not lines:
        if total == 0:
            return "(empty file)"
        return f"[offset {offset} is past the end of the file, which has {total} lines]"
    text, end = take_lines(lines, limit)
    if end < len(lines) and (limit is None or end < limit):
        return (
            f"{text}\n[... showing lines {offset}-{offset + end - 1}; more lines "
            f"follow. Continue with offset={offset + end}]"
        )
    return text


@tool
def read_file(
    path: File,
    offset: Annotated[
        int, Field(ge=1, description="The line to start from, 1-based.")
    ] = 1,
    limit: Annotated[
        int | None,
        Field(ge=1, description="The most lines to return (default: as many as fit)."),
    ] = None,
) -> str | ToolOutput:
    """Read a text file and return its lines as they are, without line numbers, or look
    at a png, jpeg, webp or gif picture (not a directory or another binary file). Text
    output is capped at about 16,000 characters; a longer file ends with the offset to
    continue from. Use offset and limit for a range of lines. Only works inside the
    working directory and the system temp directory."""
    real = resolve(path)
    if real.is_dir():
        raise IsADirectoryError(
            f"{path} is a directory, not a file; list it with run_command (ls)"
        )
    if not real.exists():
        raise FileNotFoundError(not_found(path, real))
    regular_size(real, path)
    with open(real, "rb") as raw:
        head = raw.read(_SNIFF_BYTES)
        if mime := picture_type(head):
            return read_picture(real, path, mime)
        return _read_lines(raw, path, head, offset, limit)
