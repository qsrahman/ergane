"""The tools that look at files: `read_file`, `list_directory` and `search_files`.

All three only read, so they run without asking, on threads, several at once: they keep no
state. Every path goes through `_paths.resolve` first and the path it returns is the one
opened. A search never follows a symbolic link, so a link inside the working directory
cannot lead it out of the directories `resolve` allows.

Results stay under `MAX_TEXT_CHARS` so that the footer that says where to continue is
never what the generic cut at 20 KiB removes.
"""

import os
import re
from collections import deque
from collections.abc import Iterator
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Annotated

import pathspec
from pydantic import Field

from ergane.builtin._paths import Directory, File, not_found, resolve
from ergane.builtin._text import MAX_TEXT_CHARS, is_text, take_lines
from ergane.tools import tool

_SNIFF_BYTES = 8192
# A file over this is not searched for content: it is a data file, not source.
_MAX_SCAN_BYTES = 1_000_000
# A matching line is cut here, so one minified file cannot fill the result.
_MAX_MATCH_CHARS = 300
# How many entries one search looks at before it gives up.
_MAX_VISITS = 100_000
_NEVER_ENTERED = frozenset({".git"})


def _read_lines(real: Path, shown: str, offset: int, limit: int | None) -> str:
    """The lines of a text file from `offset`, cut at the character budget."""
    with open(real, "rb") as raw:
        if not is_text(raw.read(_SNIFF_BYTES)):
            raise ValueError(f"{shown} looks like a binary file, not text")
    total = 0  # the lines of the file, counted only when there is nothing to show
    try:
        with open(real, encoding="utf-8") as handle:
            # At most one character a line fits in the budget, so no more can be used.
            wanted = islice(handle, offset - 1, offset - 1 + MAX_TEXT_CHARS + 1)
            lines = [line.rstrip("\n") for line in wanted]
            if not lines:
                handle.seek(0)
                total = sum(1 for _ in handle)
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
        int | None, Field(ge=1, description="The most lines to return.")
    ] = None,
) -> str:
    """Read a text file (not a directory or a binary file). Output is capped at about
    16,000 characters; a longer file ends with the offset to continue from. Use offset
    and limit for a range of lines. Only works inside the working directory and the
    system temp directory."""
    real = resolve(path)
    if real.is_dir():
        raise IsADirectoryError(
            f"{path} is a directory, not a file; list it with list_directory"
        )
    if not real.exists():
        raise FileNotFoundError(not_found(path, real, "File"))
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


class _Walk:
    """A breadth-first walk of a directory that leaves out what git ignores.

    The `.gitignore` files of the repository that holds the directory are used, from its
    root down (see `_Rules`); `.git` is never entered, and a symbolic link is listed but
    never followed. Entries are yielded as (path relative to the start, entry).
    `stopped` says that the walk gave up after `_MAX_VISITS` entries, and
    `unreadable` that a directory could not be read.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.stopped = False
        self.unreadable = False

    def entries(self) -> Iterator[tuple[str, os.DirEntry[str]]]:
        start = os.fspath(self.root) + os.sep
        queue = deque([(self.root, _Rules.above(self.root).inside(self.root))])
        visited = 0
        while queue:
            directory, rules = queue.popleft()
            try:
                found, _ = _children(directory, rules)
            except OSError:
                self.unreadable = True
                continue
            for entry in found:
                visited += 1
                if visited > _MAX_VISITS:
                    self.stopped = True
                    return
                yield entry.path[len(start) :], entry
                if entry.is_dir(follow_symlinks=False):
                    child = Path(entry.path)
                    queue.append((child, rules.inside(child)))


def _matching_lines(
    entry: os.DirEntry[str], relative: str, pattern: re.Pattern[str]
) -> Iterator[str]:
    """The lines of a text file that match, as `path:line: text`; none for anything that
    is not a plain file (a directory, a symbolic link, a named pipe, which would block
    for ever when read), or is binary, over `_MAX_SCAN_BYTES` or not UTF-8."""
    try:
        if (
            not entry.is_file(follow_symlinks=False)
            or entry.stat(follow_symlinks=False).st_size > _MAX_SCAN_BYTES
        ):
            return
        data = Path(entry.path).read_bytes()
    except OSError:
        return
    if not is_text(data[:_SNIFF_BYTES]):
        return
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return
    for number, line in enumerate(text.splitlines(), 1):
        if pattern.search(line):
            yield f"{relative}:{number}: {line.strip()[:_MAX_MATCH_CHARS]}"


def _hits(
    walk: _Walk, name: str | None, pattern: re.Pattern[str] | None
) -> Iterator[str]:
    """The results of a search, one at a time: the entries whose path matches the glob
    `name`, or, with `pattern`, the matching lines of the files among them."""
    for relative, entry in walk.entries():
        if name is not None and not PurePosixPath(relative).full_match(name):
            continue
        if pattern is None:
            yield _marked(relative, entry)
        else:
            yield from _matching_lines(entry, relative, pattern)


@tool
def search_files(
    path: Directory = ".",
    name: Annotated[
        str | None,
        Field(
            min_length=1,
            max_length=500,
            description="A glob for the path of each entry relative to path, for "
            "example '**/*.py' or 'src/*.ts'.",
        ),
    ] = None,
    contains: Annotated[
        str | None,
        Field(
            min_length=1,
            max_length=500,
            description="A regular expression to look for in the text of files; "
            "start it with (?i) to ignore case.",
        ),
    ] = None,
    limit: Annotated[
        int, Field(ge=1, le=500, description="The most results to return.")
    ] = 100,
) -> str:
    """Find files by name, by content, or both, under a directory.

    name is a case-sensitive glob matched against each path relative to path: '*.py'
    matches only directly under path, '**/*.py' at any depth. contains is a regular
    expression searched in the text of each file (binary files and files over 1 MB are
    skipped). Give at least one of them. The result is relative paths, directories
    ending in /, or path:line: text when contains is given. Files that .gitignore
    excludes (the repository's, from its root down) and .git are skipped, symbolic links
    are never followed, and at most limit results are returned; the result says when it stopped.
    Paths must be inside the working directory or the system temp directory. To read a
    file, use read_file.
    """
    if name is None and contains is None:
        raise ValueError("Give name, contains, or both")
    pattern = None
    if contains is not None:
        try:
            pattern = re.compile(contains)
        except re.error as exc:
            raise ValueError(
                f"contains is not a valid regular expression: {exc}"
            ) from None
    root = resolve(path)
    if not root.is_dir():
        raise NotADirectoryError(
            f"{path} is not a directory; to read a file, use read_file"
        )

    walk = _Walk(root)
    # One more than the limit, to know whether the search was cut.
    results = list(islice(_hits(walk, name, pattern), limit + 1))
    cut = len(results) > limit
    del results[limit:]

    if not results:
        results.append("No matches found")
    notes = []
    if cut:
        notes.append(
            f"[... stopped after {limit} results; narrow the search with path, name "
            "or limit]"
        )
    if walk.stopped:
        notes.append(
            f"[... gave up after looking at {_MAX_VISITS} entries; search a subdirectory]"
        )
    if walk.unreadable:
        notes.append(
            "[some directories could not be read, so results in them are missing]"
        )
    text, end = take_lines(results, 0, None, MAX_TEXT_CHARS - 500)
    if end < len(results):
        notes.append(f"[... the output was cut after {end} results; narrow the search]")
    return "\n".join([text, *notes])
