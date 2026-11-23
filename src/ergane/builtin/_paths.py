"""Where the file tools may go: the one place that turns a model's path into a real one.

Every built-in that touches a path calls `resolve` first and uses the path it returns,
so what is checked is what is opened. A path is accepted when, after symlinks are
followed, it is inside one of the `roots` (by default the working directory and the
system temp directory, where the prompt tells the model to put scratch scripts).

A path that does not exist yet is accepted when its nearest existing parent is allowed:
the result is that parent's real path plus the missing names, so that `write_file` can
create them. A `..` is collapsed before anything is looked up, so the path that is
checked is the path that is returned.

ergane's own configuration and state directories are not refused. They hold the API key,
the prompt history and the saved sessions, so a model running in the home directory can
read them with `read_file`; `run_command`, which is not confined and asks first, can
anyway. A deny list would protect only the confined tools.

The messages are for the model, and name the path as it was written.
"""

import difflib
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

from pydantic import Field

# The parameters that name a file or a directory, as the model reads them: the one wording
# that every tool shares.
File = Annotated[
    str,
    Field(
        description="The file: absolute, relative to the working directory, or "
        "starting with ~."
    ),
]
Directory = Annotated[
    str,
    Field(
        description="The directory: absolute, relative to the working directory, or "
        "starting with ~ (default: the working directory)."
    ),
]


class PathError(Exception):
    """A path a tool may not use. The message tells the model why, and what to do."""


def default_roots() -> tuple[Path, ...]:
    """The directories the file tools may use: the working directory and the temp one."""
    return (Path.cwd().resolve(), Path(tempfile.gettempdir()).resolve())


def absolute_path(path: str) -> Path:
    """The path as written, made absolute and with its `..` collapsed, before any
    symlink is looked at: `~` is the home directory, a relative path starts at the
    working directory. It is what `resolve` checks, and what a tool looks at to see
    whether the model named a symbolic link.

    Raises:
        PathError: If the path is empty, has a NUL byte, or needs a home directory that
            does not exist.
    """
    if not path.strip():
        raise PathError("The path is empty")
    if "\0" in path:
        raise PathError(f"Not a valid path: {path!r}")
    try:
        expanded = Path(path).expanduser()
    except RuntimeError:  # no home directory to expand "~" to
        raise PathError(f"Cannot expand {path}: there is no home directory") from None
    # Collapsed first, so that the path checked is the path returned.
    return Path(os.path.normpath(Path.cwd() / expanded))


def resolve(
    path: str,
    roots: Iterable[Path] | None = None,
) -> Path:
    """The real path of what the model named, if a tool may use it.

    Args:
        path: The path as the model wrote it: absolute, relative to the working
            directory, or starting with `~`.
        roots: The directories a path must be in, already resolved; the working
            directory and the temp directory if omitted.

    Returns:
        The path with symlinks followed (the missing tail, if any, added as written).

    Raises:
        PathError: If the path is empty, not a path, outside the roots, a broken symlink,
            or cannot be looked at.
    """
    absolute = absolute_path(path)
    allowed = tuple(default_roots() if roots is None else roots)

    missing: list[str] = []
    existing = absolute
    while True:
        try:
            real = existing.resolve(strict=True)
            break
        except FileNotFoundError, NotADirectoryError:
            if existing.is_symlink():
                raise PathError(
                    f"{path} goes through a symlink whose target does not exist: "
                    f"{existing}"
                ) from None
            if existing.parent == existing:  # nothing of the path exists
                raise PathError(f"Cannot look at {path}") from None
            missing.append(existing.name)
            existing = existing.parent
        except PermissionError:
            raise PathError(f"Permission denied: {path}") from None
        except OSError as exc:  # a symlink loop, say
            raise PathError(f"Cannot look at {path}: {exc.strerror or exc}") from None
    result = real.joinpath(*reversed(missing))

    if not any(result.is_relative_to(root) for root in allowed):
        where = ", ".join(str(root) for root in allowed)
        raise PathError(
            f"Access denied - {path} is outside the directories these tools may use "
            f"({where}). If you need it, ask the user."
        )
    return result


def not_found(shown: str, real: Path, kind: str) -> str:
    """The message for a path that is not there, with the names that are close to it.

    Args:
        shown: The path as the model wrote it.
        real: The path `resolve` gave.
        kind: "File" or "Directory", for the first words of the message.
    """
    message = f"{kind} not found: {shown}"
    if real.parent.is_dir():
        close = difflib.get_close_matches(
            real.name, [entry.name for entry in os.scandir(real.parent)], n=3
        )
        if close:
            near = ", ".join(str(Path(shown).parent / name) for name in close)
            message += f". Did you mean: {near}?"
    return message
