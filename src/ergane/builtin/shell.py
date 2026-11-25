"""`run_command`: run a shell command and give back what it printed.

The tool is not read-only, so the user is asked before each call, and it runs in the main
thread, so Ctrl-C reaches it. Each call is a fresh `sh -c` in its own session (process
group): stdin is closed, and on a timeout, an overflow of the output or Ctrl-C the whole
group is stopped, so that a test runner's children do not outlive the call.

The command is not confined to any directory: only its working directory (`cwd`) must be
inside the directories `_paths.resolve` allows. It gets the environment of ergane except
`OPENAI_API_KEY`, the one secret ergane holds, so that a command (or a page that talks the
model into running one) cannot print it. Variables the user exported for MCP servers are
part of that environment.

What the model reads back is plain text, not JSON: the output as it is when the command
succeeded and printed only to stdout, and otherwise the exit code and each stream
labelled. Output over about 16,000 characters keeps its start and its end, and the whole
of it is saved to a file in the system temp directory (which `read_file` may read), named
at the end of the result: a build log's cause is often in the middle, and the model would
otherwise have to run the command again to find it. The files are not deleted.
"""

import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from typing import IO, Annotated

from pydantic import Field

from ergane.builtin._paths import resolve
from ergane.builtin._text import MAX_TEXT_CHARS, head_tail
from ergane.tools import tool

DEFAULT_TIMEOUT_S = 120
# More than this much output from a command stops it: it is a runaway, such as `yes`.
_MAX_CAPTURE_BYTES = 10 * 1024 * 1024
# How long the pipes may stay open after the shell has ended: a background process that
# the command started holds them.
_PIPE_GRACE_S = 1.0
# How long a group gets to end after SIGTERM before it is killed.
_TERM_GRACE_S = 1.0
# The room each stream gets when both have something to say; one alone gets nearly all.
_BOTH_STREAMS = (11_000, 4_500)
_ONE_STREAM = MAX_TEXT_CHARS - 1_000

_ESCAPES = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"  # a CSI sequence: colours, cursor movement
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # an OSC sequence: a title, a hyperlink
    r"|\x1b[@-Z\\-_]"  # any other two-character escape
)


class _Capture:
    """Reads a pipe on a thread of its own, keeping the start of it up to a limit.

    Past the limit it goes on reading, so that the command is never blocked on a full
    pipe, and says that it overflowed. It closes the pipe itself when it ends: closing
    it from another thread would wait for a read that a process which left the group
    (and holds the pipe) never finishes."""

    def __init__(self, stream: IO[bytes], limit: int) -> None:
        self.data = bytearray()
        self.overflowed = False
        self._limit = limit
        self.thread = threading.Thread(target=self._read, args=(stream,), daemon=True)
        self.thread.start()

    def _read(self, stream: IO[bytes]) -> None:
        with stream:
            # `read1` returns what there is; `read` would wait for a full buffer.
            while chunk := stream.read1(65536):  # pyright: ignore[reportAttributeAccessIssue]
                room = self._limit - len(self.data)
                if room > 0:
                    self.data += chunk[:room]
                if len(chunk) > room:
                    self.overflowed = True


def _stop(process: subprocess.Popen[bytes]) -> None:
    """End the command's whole process group: politely, then not."""
    for sig, grace in ((signal.SIGTERM, _TERM_GRACE_S), (signal.SIGKILL, None)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:  # nothing is left
            return
        try:
            process.wait(grace)
            return
        except subprocess.TimeoutExpired:
            continue


def _join(captures: tuple[_Capture, ...]) -> None:
    """Wait, for `_PIPE_GRACE_S` in all, for the threads that read the pipes."""
    end = time.monotonic() + _PIPE_GRACE_S
    for capture in captures:
        capture.thread.join(max(0.0, end - time.monotonic()))


def _clean(data: bytes) -> str:
    """Output as the model should read it: decoded, without terminal escape codes, and
    with a progress bar that rewrites its line (a lone carriage return) reduced to
    its last version. Binary output becomes a note."""
    if b"\0" in data:
        return f"[binary output omitted: {len(data):,} bytes]"
    text = _ESCAPES.sub("", data.decode("utf-8", errors="replace"))
    text = text.replace("\r\n", "\n")
    # A line that rewrites itself is what the terminal ends up showing: its last
    # non-empty version (the empty one after a final `\r` is nothing to show).
    lines = (
        next((s for s in reversed(line.split("\r")) if s), "")
        for line in text.split("\n")
    )
    return "\n".join(lines).strip("\n")


def _format(
    returncode: int,
    stdout: str,
    stderr: str,
    notes: list[str],
    out_limit: int,
    err_limit: int,
) -> str:
    if returncode == 0 and not stderr and not notes:
        return head_tail(stdout, out_limit) if stdout else "(no output)"
    parts = []
    if returncode < 0 and not notes:
        parts.append(f"killed by signal {-returncode}")
    elif returncode > 0:
        parts.append(f"exit code {returncode}")
    if stdout:
        parts.append(f"[stdout]\n{head_tail(stdout, out_limit)}")
    if stderr:
        parts.append(f"[stderr]\n{head_tail(stderr, err_limit)}")
    return "\n".join([*parts, *notes])


def _result(returncode: int, out: bytes, err: bytes, notes: list[str]) -> str:
    stdout, stderr = _clean(out), _clean(err)
    out_limit, err_limit = _BOTH_STREAMS if stdout and stderr else (_ONE_STREAM,) * 2
    text = _format(returncode, stdout, stderr, notes, out_limit, err_limit)
    if len(stdout) <= out_limit and len(stderr) <= err_limit:
        return text
    # Saved for `read_file`; a command's result is still given if the disk is full.
    whole = _format(returncode, stdout, stderr, notes, len(stdout), len(stderr))
    try:
        descriptor, path = tempfile.mkstemp(prefix="ergane-output-", suffix=".txt")
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(whole)
    except OSError:
        return text
    return f"{text}\n[full output saved to {path}: read it with read_file]"


@tool(read_only=False)
def run_command(
    command: Annotated[
        str,
        Field(
            min_length=1,
            description="The command to run, for example 'git status' or 'uv run pytest'.",
        ),
    ],
    cwd: Annotated[
        str,
        Field(
            description="The directory to run it in (default: the working directory)."
        ),
    ] = ".",
    timeout_s: Annotated[
        int,
        Field(
            ge=1,
            le=3600,
            description=f"Stop the command after this many seconds (default "
            f"{DEFAULT_TIMEOUT_S}).",
        ),
    ] = DEFAULT_TIMEOUT_S,
) -> str:
    """Run a shell command and return its output; a failed command starts with its exit
    code, and stderr is labelled when there is any. Use it for tests, builds, git and
    other commands; not to read, write or edit files (use read_file, write_file and
    edit_file). Do not run destructive commands (rm -rf, git reset --hard, force
    pushes) unless the user asked.
    stdin is closed and each call is a fresh process, so interactive commands do not
    work and cd does not persist: use cwd. The command and its child processes are
    stopped after timeout_s, so do not start dev servers or watchers. Output past about
    16,000 characters keeps its start and end, and the full output is saved to a file
    named at the end (read it with read_file). Only cwd must be inside the working
    directory or the system temp directory, not what the command touches."""
    directory = resolve(cwd)
    if not directory.exists():
        raise FileNotFoundError(f"{cwd} does not exist")
    if not directory.is_dir():
        raise NotADirectoryError(f"{cwd} is not a directory")
    environment = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=directory,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,  # its own process group, to stop it whole
    )
    assert process.stdout is not None and process.stderr is not None
    out = _Capture(process.stdout, _MAX_CAPTURE_BYTES)
    err = _Capture(process.stderr, _MAX_CAPTURE_BYTES)
    notes: list[str] = []
    runaway = False
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                process.wait(0.1)
                break
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() >= deadline:
                notes.append(f"command timed out after {timeout_s} s and was stopped")
            elif out.overflowed or err.overflowed:
                runaway = True
            if notes or runaway:
                _stop(process)
                break
        _join((out, err))
        if out.thread.is_alive() or err.thread.is_alive():
            # A process the command started in the background holds the pipes.
            _stop(process)
            _join((out, err))
            notes.append(
                "background processes the command started held its output open; "
                "those in its process group were stopped"
            )
    except BaseException:  # Ctrl-C, mostly: leave nothing running
        _stop(process)
        raise
    if out.overflowed or err.overflowed:
        notes.append(
            "the output was over 10 MB and was cut off"
            + ("; the command was stopped" if runaway else "")
        )
    return _result(process.returncode, bytes(out.data), bytes(err.data), notes)
