"""Running ripgrep for `search_files`: the one place that starts it.

`rg` does what a search needs and does it well: it walks a tree, leaves out what
`.gitignore` excludes, skips binary files, and matches with an automaton, so no pattern
can backtrack without end. This module only starts it safely and bounds what it does,
since a search runs on a thread that Ctrl-C cannot stop:

- The flags `FLAGS` make what it reads the same as what `list_directory` reads: dotfiles
  are searched, `.git` is not, and `.gitignore` files from the root of the repository down
  are used, but not `.ignore`, `.rgignore`, `.git/info/exclude`, the user's global ignore
  file or the configuration file `RIPGREP_CONFIG_PATH` names, which a project could not
  be trusted with. A symbolic link is never followed.
- The model's text is only ever the argument of `-e` or `-g`, and the path is after `--`,
  so nothing the model wrote can be taken for a flag.
- The output is read as it comes and `rg` is stopped, with its process group, after the
  records wanted or `SEARCH_SECONDS`.
"""

import os
import shutil
import signal
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

PROGRAM = "rg"
SEARCH_SECONDS = 30
# `--max-filesize`: a file over this is a data file, not source.
_MAX_FILE = "1M"
# How much of what `rg` says on its standard error is kept.
_MAX_STDERR = 4000

# Each is one argument. `--sort path` makes the order the same every time, at the cost of
# searching on one thread.
FLAGS = (
    "--no-config",
    "--hidden",
    "--no-ignore-dot",
    "--no-ignore-global",
    "--no-ignore-exclude",
    "--sort=path",
    f"--max-filesize={_MAX_FILE}",
    "--glob=!.git",
)


@dataclass
class Found:
    """What a run of `rg` gave."""

    records: list[bytes]  # as many as were asked for, at most
    cut: bool  # there were more
    timed_out: bool
    returncode: int | None  # None when it was stopped
    stderr: str  # the start of it


def _stop(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError, PermissionError:  # already gone
        pass


def find(
    arguments: list[str],
    root: Path,
    separator: bytes,
    count: int,
    keep: Callable[[bytes], bool] | None = None,
) -> Found:
    """Run `rg` in `root` and read up to `count` records of its output.

    Args:
        arguments: What goes after `rg` and `FLAGS`, up to and including `--`.
        root: The directory to run in; the paths in the output are relative to it.
        separator: What ends a record: a newline, or NUL for `--files --null`.
        count: How many records to keep; one more is looked for, to know there were
            more, and then `rg` is stopped.
        keep: Which records to count, if not all. The search itself cannot say: an
            `rg` glob puts back the files that `.gitignore` excludes.

    Returns:
        What it gave.

    Raises:
        FileNotFoundError: If `rg` is not installed (it was when ergane started).
    """
    program = shutil.which(PROGRAM)
    if program is None:
        raise FileNotFoundError(f"{PROGRAM} is not installed")
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            [program, *FLAGS, *arguments, "."],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr,
            start_new_session=True,  # its own process group, to stop it whole
        )
        timed_out = threading.Event()

        def deadline() -> None:
            timed_out.set()
            _stop(process)

        timer = threading.Timer(SEARCH_SECONDS, deadline)
        timer.start()
        records: list[bytes] = []
        cut = False
        try:
            assert process.stdout is not None
            descriptor = process.stdout.fileno()
            rest = b""
            while chunk := os.read(descriptor, 65536):
                *whole, rest = (rest + chunk).split(separator)
                records += whole if keep is None else filter(keep, whole)
                if len(records) > count:
                    cut = True
                    break
            else:
                if rest and (keep is None or keep(rest)):
                    records.append(rest)
                    cut = len(records) > count
        finally:
            timer.cancel()
            if process.poll() is None:
                _stop(process)
            process.wait()
            if process.stdout:
                process.stdout.close()
        del records[count:]
        stderr.seek(0)
        text = stderr.read(_MAX_STDERR).decode("utf-8", "replace")
    stopped = cut or timed_out.is_set()
    return Found(
        records, cut, timed_out.is_set(), None if stopped else process.returncode, text
    )
