"""The tools that look at files: `read_file`, `list_directory` and `search_files`.

All three only read, so they run without asking, on threads, several at once: they keep no
state. Every path goes through `_paths.resolve` first and the path it returns is the one
opened. `search_files` is `rg` (see `_rg`), which never follows a symbolic link, so a link
inside the working directory cannot lead it out of the directories `resolve` allows, and
which leaves out what `list_directory` leaves out. It needs `rg` installed: without it the
tool is not offered.

Results stay under `MAX_TEXT_CHARS` so that the footer that says where to continue is
never what the generic cut at 20 KiB removes.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import pathspec
from pydantic import Field

from ergane.builtin._images import picture_type, read_picture
from ergane.builtin._paths import Directory, File, not_found, resolve
from ergane.builtin._rg import PROGRAM, SEARCH_SECONDS, find
from ergane.builtin._text import (
    MAX_TEXT_CHARS,
    count_lines,
    is_text,
    read_lines,
    take_lines,
)
from ergane.tools import ToolOutput, tool

_SNIFF_BYTES = 8192
# A matching line is cut here, so one minified file cannot fill the result.
_MAX_MATCH_CHARS = 300
_NEVER_ENTERED = frozenset({".git"})


def _read_lines(real: Path, shown: str, offset: int, limit: int | None) -> str:
    """The lines of a text file from `offset`, cut at the character budget."""
    with open(real, "rb") as raw:
        if not is_text(raw.read(_SNIFF_BYTES)):
            raise ValueError(f"{shown} looks like a binary file, not text")
    total = 0  # the lines of the file, counted only when there is nothing to show
    try:
        # utf-8-sig drops a byte order mark, which the model would see as a stray
        # character at the start of the first line.
        with open(real, encoding="utf-8-sig") as handle:
            # At most one character a line fits in the budget, so no more can be used.
            lines = read_lines(handle, offset - 1, MAX_TEXT_CHARS + 1)
            if not lines:
                handle.seek(0)
                total = count_lines(handle)
    except UnicodeDecodeError:
        raise ValueError(f"{shown} is not valid UTF-8 text") from None
    if not lines:
        if total == 0:
            return "(empty file)"
        return f"[offset {offset} is past the end of the file, which has {total} lines]"
    text, end = take_lines(lines, 0, limit)
    if end < len(lines) and (limit is None or end < limit):
        return (
            f"{text}\n[... showing lines {offset}-{offset + end - 1}; more lines "
            f"follow. Continue with offset={offset + end}]"
        )
    return text


class _Rules:
    """The `.gitignore` rules in force at a place.

    The files are kept shallowest first, so that a deeper one overrides a shallower one
    as it does in git (`!keep.log` in a subdirectory brings back a file that the parent
    ignored). A file's rules apply below its own directory only. Only `.gitignore` files
    are read: not `.git/info/exclude` and not the user's global ignore file.
    """

    def __init__(
        self, specs: tuple[tuple[str, pathspec.GitIgnoreSpec], ...] = ()
    ) -> None:
        self._specs = specs

    def inside(self, directory: Path) -> _Rules:
        """These rules and the ones in `directory`'s own `.gitignore`, if it has one."""
        try:
            lines = (directory / ".gitignore").read_text(encoding="utf-8").splitlines()
            spec = pathspec.GitIgnoreSpec.from_lines(lines)
        except OSError, ValueError:  # no file, not UTF-8, or a pattern git rejects
            return self
        return _Rules((*self._specs, (os.fspath(directory) + os.sep, spec)))

    @classmethod
    def above(cls, directory: Path) -> _Rules:
        """The rules of the `.gitignore` files from the root of the repository that holds
        `directory` down to its parent: none when it is not in a repository."""
        places = [directory, *directory.parents]
        for depth, place in enumerate(places):
            if (place / ".git").exists():
                break
        else:
            return cls()
        rules = cls()
        for ancestor in reversed(places[1 : depth + 1]):
            rules = rules.inside(ancestor)
        return rules

    def ignores(self, path: str, is_dir: bool) -> bool:
        """Whether the entry at `path` (absolute) is ignored. It must be below every rule
        file's directory, which it is: rules are only ever extended on the way down."""
        ignored = False
        for base, spec in self._specs:
            relative = path[len(base) :]
            verdict = spec.check_file(relative + "/" if is_dir else relative).include
            if verdict is not None:  # None: no pattern of this file matched
                ignored = verdict
        return ignored


def _children(directory: Path, rules: _Rules) -> tuple[list[os.DirEntry[str]], int]:
    """What a directory shows: its entries without `.git` and what `.gitignore`
    excludes, directories first and each kind by name, and how many were left out. The
    listing and the search both start from this, so they never disagree about it.

    Raises:
        OSError: If the directory cannot be read.
    """
    entries = list(os.scandir(directory))
    kept = sorted(
        (
            entry
            for entry in entries
            if entry.name not in _NEVER_ENTERED
            and not rules.ignores(entry.path, entry.is_dir(follow_symlinks=False))
        ),
        key=lambda e: (not e.is_dir(follow_symlinks=False), e.name),
    )
    return kept, len(entries) - len(kept)


def _marked(name: str, entry: os.DirEntry[str]) -> str:
    """The name of an entry, with `/` after a directory and `@` after a symbolic link
    (which is never followed)."""
    if entry.is_dir(follow_symlinks=False):
        return f"{name}/"
    return f"{name}@" if entry.is_symlink() else name


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
            f"{path} is a directory, not a file; list it with list_directory"
        )
    if not real.exists():
        raise FileNotFoundError(not_found(path, real, "File"))
    with open(real, "rb") as raw:
        mime = picture_type(raw.read(_SNIFF_BYTES))
    if mime:
        return read_picture(real, path, mime)
    return _read_lines(real, path, offset, limit)


@tool
def list_directory(path: Directory = ".") -> str:
    """List a directory, one entry per line: directories (first) end in / and symbolic
    links in @. Entries that .gitignore excludes, and .git, are left out and counted at
    the end. Output is capped at about 16,000 characters; the result says when it
    stopped, so list a subdirectory for the rest. To find a file by name or content
    anywhere below a directory, use search_files. Only works inside the working
    directory and the system temp directory."""
    real = resolve(path)
    if real.is_file():
        raise NotADirectoryError(
            f"{path} is a file, not a directory; read it with read_file"
        )
    if not real.exists():
        raise FileNotFoundError(not_found(path, real, "Directory"))
    kept, left_out = _children(real, _Rules.above(real).inside(real))
    lines = [_marked(entry.name, entry) for entry in kept]
    note = (
        f"\n[{left_out} {'entry' if left_out == 1 else 'entries'} left out: ignored "
        "by .gitignore, or .git]"
        if left_out
        else ""
    )
    if not lines:
        return ("(no entries to show)" if left_out else "(empty directory)") + note
    text, end = take_lines(lines)
    if end < len(lines):
        text += (
            f"\n[... showing {end} of {len(lines)} entries; list a subdirectory, or "
            "use search_files]"
        )
    return text + note


def _named(name: str) -> Callable[[bytes], bool]:
    """A test of a record of `rg`'s output (a path, or a path then NUL and a match) for a
    path that `name` matches, as a line of a `.gitignore` would: `rg`'s own `--glob`
    would put back the files that `.gitignore` excludes.

    Raises:
        ValueError: If the pattern is not a valid glob.
    """
    try:
        spec = pathspec.GitIgnoreSpec.from_lines([name])
    except ValueError as exc:
        raise ValueError(f"name is not a valid glob: {exc}") from None

    def matches(record: bytes) -> bool:
        path = record.split(b"\0", 1)[0].decode("utf-8", "replace")
        return spec.match_file(path.removeprefix("./"))

    return matches


def _reason(stderr: str) -> str:
    """Why `rg` refused a pattern: its `error:` line, or else its first line."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    errors = [line for line in lines if line.startswith("error: ")]
    return (errors or lines)[0].removeprefix("error: ")


@tool(requires=PROGRAM)
def search_files(
    path: Directory = ".",
    name: Annotated[
        str | None,
        Field(
            min_length=1,
            max_length=500,
            description="A glob for the path of each file relative to path, written "
            "as a line of a .gitignore: '*.py' at any depth, '/*.py' only directly "
            "under path, 'src/**/*.ts' anywhere inside src.",
        ),
    ] = None,
    contains: Annotated[
        str | None,
        Field(
            min_length=1,
            max_length=500,
            description="A regular expression (Rust syntax: no lookaround or "
            "backreferences) to look for in the text of files; start it with (?i) to "
            "ignore case.",
        ),
    ] = None,
    limit: Annotated[
        int, Field(ge=1, le=500, description="The most results to return.")
    ] = 100,
) -> str:
    """Find files by name, by content, or both, under a directory. Give at least one of
    name and contains.

    name is a case-sensitive glob in .gitignore style matched against each file's path
    relative to path: '*.py' matches at any depth, '/*.py' only directly under path.
    contains is a regular expression searched in the text of each file. The result is
    relative paths of files, or path:line: text when contains is given, and says when it
    stopped at limit. Binary files, files over 1 MB, files that .gitignore excludes, .git
    and symbolic links are skipped. Only works inside the working directory and the
    system temp directory. To read a file, use read_file; to see a directory, use
    list_directory.
    """
    if name is None and contains is None:
        raise ValueError("Give name, contains, or both")
    root = resolve(path)
    if not root.is_dir():
        raise NotADirectoryError(
            f"{path} is not a directory; to read a file, use read_file"
        )

    keep = None if name is None else _named(name)
    if contains is None:
        found = find(["--files", "--null", "--"], root, b"\0", limit, keep)
    else:
        found = find(
            [
                "--line-number",
                "--no-heading",
                "--with-filename",
                "--null",
                "--trim",
                f"--max-columns={_MAX_MATCH_CHARS}",
                "--max-columns-preview",
                "-e",
                contains,
                "--",
            ],
            root,
            b"\n",
            limit,
            keep,
        )
    if (
        found.returncode == 2
        and not found.records
        and "regex parse error" in found.stderr
    ):
        raise ValueError(
            f"contains is not a valid regular expression: {_reason(found.stderr)}"
        )

    results = []
    for record in found.records:
        text = record.decode("utf-8", "replace").removeprefix("./")
        if contains is None:
            results.append(text)
        else:
            file, _, rest = text.partition("\0")
            number, _, line = rest.partition(":")
            results.append(f"{file}:{number}: {line}")

    if not results:
        results.append("No matches found")
    notes = []
    if found.cut:
        notes.append(
            f"[... stopped after {limit} results; narrow the search with path, name "
            "or limit]"
        )
    if found.timed_out:
        notes.append(
            f"[... gave up after {SEARCH_SECONDS} seconds; search a subdirectory or "
            "use a more specific name or pattern]"
        )
    elif found.returncode == 2:
        notes.append(
            "[some files or directories could not be read, so results in them are "
            "missing]"
        )
    text, end = take_lines(results, 0, None, MAX_TEXT_CHARS - 500)
    if end < len(results):
        notes.append(f"[... the output was cut after {end} results; narrow the search]")
    return "\n".join([text, *notes])
