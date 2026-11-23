"""The tools that change files: `write_file` and `edit_file`.

Neither is read-only, so each asks before it runs, and the question can show the change
(`preview`): a diff, worked out without writing anything. They run one at a time, after
the calls before them, so there is no lock.

Every path goes through `_paths.resolve`, and the path it returns is the one written.
A file is never written through a symbolic link: the path the model wrote must not be one,
a new file is created exclusively, and an existing one must be a regular file, opened
without following a link. A model is told one line of what was done; the diff is for the
user.
"""

import difflib
import os
import stat
from itertools import islice
from pathlib import Path
from typing import Annotated

from pydantic import Field

from ergane.builtin._edit import EditError, edit_text
from ergane.builtin._paths import not_found, resolve
from ergane.tools import tool

# A file bigger than this is not edited: it is not source, and it is read whole.
_MAX_EDIT_BYTES = 10 * 1024 * 1024
# What a preview shows of a long diff or a new file: the user reads it in a terminal.
_PREVIEW_LINES = 120


def _refuse_symlink(path: str) -> None:
    """The path the model wrote must not be a symbolic link: it would be written
    through, to somewhere else than the question named."""
    written = Path(os.path.normpath(Path.cwd() / Path(path).expanduser()))
    if written.is_symlink():
        raise ValueError(
            f"{path} is a symbolic link; write to the file it points to instead"
        )


def _write(real: Path, content: str) -> bool:
    """Write `content` to `real`, making missing directories. Whether it existed."""
    real.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            real, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o666
        )
        existed = False
    except FileExistsError:
        if not stat.S_ISREG(os.lstat(real).st_mode):
            raise ValueError(f"Refusing to write: not a regular file: {real}") from None
        descriptor = os.open(real, os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW)
        existed = True
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    return existed


def _read_text(real: Path, shown: str) -> str:
    """The text of a file that an edit may change: a file, UTF-8, and not huge."""
    if real.is_dir():
        raise IsADirectoryError(f"{shown} is a directory, not a file")
    if not real.exists():
        raise FileNotFoundError(not_found(shown, real, "File"))
    if real.stat().st_size > _MAX_EDIT_BYTES:
        raise ValueError(f"{shown} is too large to edit")
    try:
        return real.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        # Writing it back would make the replacement characters permanent.
        raise ValueError(f"Refusing to edit: {shown} is not valid UTF-8 text") from None


def _diff(before: str, after: str, shown: str) -> str:
    """A unified diff of two texts, cut to what a terminal can show."""
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=shown,
            tofile=shown,
            n=2,
            lineterm="",
        )
    )
    if not lines:
        return "No change: the file already has this content"
    more = len(lines) - _PREVIEW_LINES
    kept = lines[:_PREVIEW_LINES] + ([f"[... {more} more lines]"] if more > 0 else [])
    return "\n".join(kept)


def _preview_write(path: str, content: str) -> str:
    """What `write_file` would do: a diff for a file that exists, else its start."""
    _refuse_symlink(path)
    real = resolve(path)
    if real.is_file():
        try:
            return _diff(real.read_bytes().decode("utf-8"), content, path)
        except UnicodeDecodeError:
            return f"{path} is not text; it would be overwritten ({real.stat().st_size:,} bytes)"
    lines = content.splitlines()
    shown = [f"+{line}" for line in islice(lines, _PREVIEW_LINES)]
    if len(lines) > _PREVIEW_LINES:
        shown.append(f"[... {len(lines) - _PREVIEW_LINES} more lines]")
    return f"New file {path} ({len(lines)} lines):\n" + "\n".join(shown)


def _preview_edit(
    path: str, old_text: str, new_text: str, replace_all: bool = False
) -> str:
    """What `edit_file` would do: a diff, or why it would fail."""
    _refuse_symlink(path)
    real = resolve(path)
    before = _read_text(real, path)
    try:
        after, _ = edit_text(before, old_text, new_text, replace_all, path)
    except EditError as exc:
        return f"This edit would fail: {exc}"
    return _diff(before, after, path)


@tool(read_only=False, preview=_preview_write)
def write_file(
    path: Annotated[
        str,
        Field(
            description="The file: absolute, relative to the working directory, or "
            "starting with ~."
        ),
    ],
    content: Annotated[
        str, Field(description="The complete new contents of the file.")
    ],
) -> str:
    """Create a file or completely overwrite one, creating missing parent directories.
    Overwrites without warning; to change part of a file use edit_file. Only works
    inside the working directory and the system temp directory."""
    _refuse_symlink(path)
    real = resolve(path)
    if real.is_dir():
        raise IsADirectoryError(f"{path} is a directory, not a file")
    existed = _write(real, content)
    lines = len(content.splitlines())
    return f"{'Overwrote' if existed else 'Created'} {path} ({lines} lines)"


@tool(read_only=False, preview=_preview_edit)
def edit_file(
    path: Annotated[
        str,
        Field(
            description="The file: absolute, relative to the working directory, or "
            "starting with ~."
        ),
    ],
    old_text: Annotated[
        str, Field(description="The text to replace, copied exactly from the file.")
    ],
    new_text: Annotated[
        str, Field(description="What to put in its place; empty to delete the text.")
    ],
    replace_all: Annotated[
        bool,
        Field(
            description="Replace every place old_text is found, not only the one "
            "(default false)."
        ),
    ] = False,
) -> str:
    """Replace text in a file. old_text must match exactly one place (add surrounding
    lines to make it unique) unless replace_all is set. When there is no exact match,
    trailing whitespace and typographic quotes, dashes and spaces are ignored, line by
    line. Nothing is written if the edit cannot be made. Only works inside the working
    directory and the system temp directory."""
    _refuse_symlink(path)
    real = resolve(path)
    before = _read_text(real, path)
    after, count = edit_text(before, old_text, new_text, replace_all, path)
    _write(real, after)
    return f"Edited {path} ({count} replacement{'s' if count != 1 else ''})"
