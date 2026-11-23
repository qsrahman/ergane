"""Pictures for `read_file`: which files are one, and how one is handed to the model.

The kind is read from the first bytes, never from the file's name: a text file called
`x.png` is text, and a png called `x.txt` is a picture. The kinds are those the Responses
API takes (`IMAGE_TYPES` in `tools.py`); an animated PNG is left out, since a model would see one frame
of it and the file would look like a still. A picture over `MAX_IMAGE_CHARS` (as base64)
is refused, because it would sit in every request until dropped; ergane has no image
library to make it smaller, so the model is told to ask for a smaller one.
"""

import base64
import struct
from pathlib import Path

from ergane.tools import MAX_IMAGE_CHARS, Image, ToolOutput

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


def picture_type(head: bytes) -> str | None:
    """The mime type of a picture whose first bytes are `head`, or None."""
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/png" if _is_still_png(head) else None


def _size(count: int) -> str:
    return (
        f"{count / 1_000_000:.1f} MB"
        if count >= 1_000_000
        else f"{count / 1000:.0f} KB"
    )


def read_picture(real: Path, shown: str, mime: str) -> ToolOutput:
    """The picture in `real` for the model, with a line that names it.

    Raises:
        ValueError: If the picture is too large to send.
    """
    size = real.stat().st_size
    if -(-size // 3) * 4 > MAX_IMAGE_CHARS:
        limit = MAX_IMAGE_CHARS // 4 * 3
        raise ValueError(
            f"{shown} is a {mime} picture of {_size(size)}, over the {_size(limit)} "
            "limit; ask for a smaller one"
        )
    data = base64.b64encode(real.read_bytes()).decode("ascii")
    return ToolOutput(f"Picture: {shown} ({mime}, {_size(size)})", (Image(data, mime),))
