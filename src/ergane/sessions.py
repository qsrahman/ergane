"""Sessions saved to disk: one file per conversation, kept for the working directory.

A session is saved in the user's own state directory, as
`sessions/<directory>/<id>.jsonl`, never in the working directory: it is played back to
the model as what the model said and what its tools returned, so a file that a project
shipped could give orders in the model's own voice. The directory is the working
directory's real path, made readable and followed by a short hash of it.

A file is two lines of JSON. The first says enough to list the session (`SessionInfo`),
so listing reads one line of each file and not their histories, which can hold pictures.
The second has the summary and the items, as `Context` keeps them: plain JSON, which
the SDK's items already are. A file is written whole to a temporary name and moved into
place, so a crash leaves the old one; the temporary file of a save that was killed is
removed by a later save. Files are readable by the user only. Only the newest `KEEP`
sessions of a directory are kept.

What is not saved: what the user allowed for the session (a resumed session asks
again), the count of tokens the server made of it (the next reply gives it again), and
the tools and what the servers say, which are the current ones.
"""

import hashlib
import json
import os
import re
import secrets
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ergane.surrogates import without_surrogates

# How many sessions of a directory are kept: the newest, which saving prunes to.
KEEP = 50

# How old a temporary file is before it is taken for one that a killed save left: a
# newer one may be a save that another ergane is making now.
_STALE_SECONDS = 3600

# How much of the first prompt is the title.
_TITLE_CHARS = 60


class SessionError(Exception):
    """A session that cannot be found or read; the message is for the user."""


class SessionInfo(BaseModel):
    """What the first line of a session file says: enough to list it."""

    model_config = ConfigDict(frozen=True)

    version: Literal[1] = 1
    id: str
    cwd: str
    title: str
    created: str  # ISO 8601, UTC
    updated: str
    model: str  # the one that answered last
    turns: int


class _Body(BaseModel):
    """The second line: the conversation."""

    summary: str | None
    items: list[dict[str, Any]]


@dataclass(frozen=True)
class SavedSession:
    """A session as it was read: what it says about itself, and its conversation."""

    info: SessionInfo
    summary: str | None
    items: list[dict[str, Any]]


def now() -> str:
    """The time, as a session file says it."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def ago(info: SessionInfo) -> str:
    """How long ago the session was saved, in words: "5 min ago", "2 h ago"."""
    seconds = (datetime.now(UTC) - datetime.fromisoformat(info.updated)).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    days = int(seconds // 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


def _user_prompts(items: list[dict[str, Any]]) -> list[str]:
    """The text of the user's messages, in order."""
    return [
        item["content"]
        for item in items
        if item.get("role") == "user" and isinstance(item.get("content"), str)
    ]


def _title(items: list[dict[str, Any]]) -> str:
    """The first prompt of the conversation, on one line and cut short."""
    prompts = _user_prompts(items)
    text = " ".join(prompts[0].split()) if prompts else "(no prompt)"
    return text if len(text) <= _TITLE_CHARS else text[: _TITLE_CHARS - 1] + "…"


def _directory_name(cwd: Path) -> str:
    """A name for a working directory: readable, and different for each path."""
    text = str(cwd)
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[-48:]
    digest = hashlib.sha256(text.encode()).hexdigest()[:8]
    return f"{readable}-{digest}" if readable else digest


def _read_info(path: Path) -> SessionInfo | None:
    """The first line of a session file; None if it is not a session of this version
    or does not belong under the name it has (the name is what files are removed by)."""
    try:
        with path.open(encoding="utf-8") as file:
            info = SessionInfo.model_validate_json(file.readline())
    except OSError, ValueError:
        return None
    return info if info.id == path.stem else None


class SessionStore:
    """The sessions saved for one working directory."""

    def __init__(self, root: Path, cwd: Path) -> None:
        """Describe where the sessions of a directory are.

        Args:
            root: The directory all sessions are in (`sessions` in the state directory).
            cwd: The working directory, which is resolved: a link to it and the
                directory itself share their sessions.
        """
        self.cwd = cwd.resolve()
        self.directory = root / _directory_name(self.cwd)

    def sessions(self) -> list[SessionInfo]:
        """The sessions, the most recently saved first. A file that is not a session
        of this version is left out, and left alone."""
        found = [
            info
            for path in self.directory.glob("*.jsonl")
            if (info := _read_info(path)) is not None
        ]
        return sorted(found, key=lambda info: (info.updated, info.id), reverse=True)

    def latest(self) -> SavedSession | None:
        """The session saved most recently, or None if there is none."""
        found = self.sessions()
        return self._read(found[0].id) if found else None

    def load(self, ref: str) -> SavedSession:
        """A session by its id, or by the start of it if that names one.

        Raises:
            SessionError: If none or several match, or the file cannot be read.
        """
        found = self.sessions()
        matches = [info for info in found if info.id == ref] or [
            info for info in found if info.id.startswith(ref)
        ]
        if not matches:
            raise SessionError(f"No saved session {ref!r} for this directory")
        if len(matches) > 1:
            ids = ", ".join(info.id for info in matches)
            raise SessionError(f"{ref!r} starts {len(matches)} sessions: {ids}")
        return self._read(matches[0].id)

    def save(
        self,
        previous: SessionInfo | None,
        summary: str | None,
        items: list[dict[str, Any]],
        model: str,
    ) -> SessionInfo:
        """Write a conversation: as a new session, or over the one it was saved as.

        Args:
            previous: What an earlier save returned for this conversation, or None
                for one that is not saved yet; its id, title and time of creation stay.
            summary: What `Context.summary` is.
            items: What `Context.items` is.
            model: The model of the conversation now.

        Returns:
            What the session says about itself now, to pass to the next save.

        Raises:
            OSError: If the file cannot be written.
        """
        summary, items = without_surrogates((summary, items))
        stamp = now()
        info = SessionInfo(
            id=previous.id if previous else self._new_id(),
            cwd=str(self.cwd),
            title=previous.title if previous else _title(items),
            created=previous.created if previous else stamp,
            updated=stamp,
            model=model,
            turns=len(_user_prompts(items)),
        )
        self._write(info, summary, items)
        for old in self.sessions()[KEEP:]:
            if old.id != info.id:
                (self.directory / f"{old.id}.jsonl").unlink(missing_ok=True)
        self._remove_leftovers()
        return info

    def _remove_leftovers(self) -> None:
        """Remove old temporary files; each can hold a whole conversation."""
        for path in self.directory.glob("*.tmp"):
            with suppress(OSError):  # the session is saved; this can wait
                if time.time() - path.stat().st_mtime > _STALE_SECONDS:
                    path.unlink()

    def _new_id(self) -> str:
        """An id that sorts by time and that no file has."""
        while True:
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            candidate = f"{stamp}-{secrets.token_hex(2)}"
            if not (self.directory / f"{candidate}.jsonl").exists():
                return candidate

    def _write(
        self, info: SessionInfo, summary: str | None, items: list[dict[str, Any]]
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        for directory in (self.directory.parent, self.directory):
            with suppress(OSError):  # as private as the state directory
                directory.chmod(0o700)
        path = self.directory / f"{info.id}.jsonl"
        # A name of its own: two erganes that resume one session must not write the
        # same temporary file, or the later replace would rename the other's write.
        descriptor, temporary = tempfile.mkstemp(
            dir=self.directory, prefix=f".{info.id}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(info.model_dump_json() + "\n")
                file.write(json.dumps({"summary": summary, "items": items}) + "\n")
            os.replace(temporary, path)
        finally:
            with suppress(OSError):
                os.unlink(temporary)

    def _read(self, session_id: str) -> SavedSession:
        path = self.directory / f"{session_id}.jsonl"
        try:
            header, body = path.read_text(encoding="utf-8").split("\n")[:2]
            info = SessionInfo.model_validate_json(header)
            content = _Body.model_validate_json(body)
        except (OSError, ValueError) as exc:
            raise SessionError(
                f"The session {session_id} cannot be read: {exc}"
            ) from None
        return SavedSession(info, content.summary, content.items)
