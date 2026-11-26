"""The tool that reads a file: `read_file`.

It only reads, so it runs without asking, on a thread, several at once: it keeps no state.
The path goes through `_paths.resolve` first and the path it returns is the one opened.

The result stays under `MAX_TEXT_CHARS` so that the footer that says where to continue is
never what the generic cut at 20 KiB removes.

A picture is told by the first bytes, never by the file's name: a text file called `x.png`
is text, and a png called `x.txt` is a picture. The kinds are those the Responses API takes
(`IMAGE_TYPES` in `tools.py`); an animated PNG is left out, since a model would see one
frame of it and the file would look like a still. A picture over `MAX_IMAGE_CHARS` (as
base64) is refused, because it would sit in every request until dropped; ergane has no
image library to make it smaller, so the model is told to ask for a smaller one.
"""

import base64
import io
import struct
from pathlib import Path
from typing import Annotated, BinaryIO

from pydantic import Field

from ergane.builtin._paths import File, not_found, regular_size, resolve
from ergane.builtin._text import (
    MAX_TEXT_CHARS,
    count_lines,
    read_lines,
    take_lines,
)
from ergane.tools import MAX_IMAGE_CHARS, Image, ToolOutput, tool

_SNIFF_BYTES = 8192
_PNG = b"\x89PNG\r\n\x1a\n"


def _is_still_png(head: bytes) -> bool:
    """A PNG that starts with an IHDR chunk and has no animation chunk (`acTL`) before
    its first image data (`IDAT`)."""
    if not head.startswith(_PNG) or head[12:16] != b"IHDR":
        return False
    offset = len(_PNG)
    while offset + 8 <= len(head):
        length, kind = struct.unpack(">I4s", head[offset : offset + 8])
        if kind == b"acTL":
            return False
        if kind == b"IDAT":
            return True
        offset += 12 + length
    return True  # no animation chunk in the part that was read


def _picture_type(head: bytes) -> str | None:
    """The mime type of a picture whose first bytes are `head`, or None."""
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/png" if _is_still_png(head) else None


def _picture_size(count: int) -> str:
    return (
        f"{count / 1_000_000:.1f} MB"
        if count >= 1_000_000
        else f"{count / 1000:.0f} KB"
    )


def _read_picture(real: Path, shown: str, mime: str) -> ToolOutput:
    """The picture in `real` for the model, with a line that names it.

    Raises:
        ValueError: If the picture is too large to send.
    """
    size = real.stat().st_size
    if -(-size // 3) * 4 > MAX_IMAGE_CHARS:
        limit = MAX_IMAGE_CHARS // 4 * 3
        raise ValueError(
            f"{shown} is a {mime} picture of {_picture_size(size)}, over the {_picture_size(limit)} "
            "limit; ask for a smaller one"
        )
    data = base64.b64encode(real.read_bytes()).decode("ascii")
    return ToolOutput(
        f"Picture: {shown} ({mime}, {_picture_size(size)})", (Image(data, mime),)
    )


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
        if mime := _picture_type(head):
            return _read_picture(real, path, mime)
        return _read_lines(raw, path, head, offset, limit)
