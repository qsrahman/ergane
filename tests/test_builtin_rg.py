import os
import shutil
import time
from pathlib import Path

import pytest

from ergane.builtin import _rg
from ergane.builtin._rg import find

pytestmark = pytest.mark.skipif(shutil.which("rg") is None, reason="needs ripgrep")


def files(root: Path, count: int = 100, keep=None):
    return find(["--files", "--null", "--"], root, b"\0", count, keep)


def make(root: Path, *names: str) -> None:
    for name in names:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x\n")


def paths(found) -> list[str]:
    return [record.decode().removeprefix("./") for record in found.records]


class TestFind:
    def test_it_gives_the_records_in_path_order(self, tmp_path: Path):
        make(tmp_path, "b.txt", "a.txt", "sub/c.txt")

        found = files(tmp_path)

        assert paths(found) == ["a.txt", "b.txt", "sub/c.txt"]
        assert not found.cut and not found.timed_out and found.returncode == 0

    def test_it_stops_after_the_count_and_says_there_was_more(self, tmp_path: Path):
        make(tmp_path, *[f"f{n}.txt" for n in range(10)])

        found = files(tmp_path, count=3)

        assert paths(found) == ["f0.txt", "f1.txt", "f2.txt"]
        assert found.cut and found.returncode is None

    def test_exactly_the_count_is_not_a_cut(self, tmp_path: Path):
        make(tmp_path, "a.txt", "b.txt")

        found = files(tmp_path, count=2)

        assert len(found.records) == 2 and not found.cut

    def test_only_the_records_kept_are_counted(self, tmp_path: Path):
        make(
            tmp_path, *[f"f{n}.txt" for n in range(5)], *[f"g{n}.py" for n in range(5)]
        )

        found = files(tmp_path, count=2, keep=lambda record: record.endswith(b".py"))

        assert paths(found) == ["g0.py", "g1.py"] and found.cut

    def test_the_last_record_is_kept_when_the_output_does_not_end_with_the_separator(
        self, tmp_path: Path
    ):
        make(tmp_path, "a.txt")
        arguments = ["--files", "--null", "--"]  # NUL ends it; the separator is \n

        assert find(arguments, tmp_path, b"\n", 5).records == [b"./a.txt\0"]
        assert find(arguments, tmp_path, b"\n", 5, lambda r: True).records
        assert not find(arguments, tmp_path, b"\n", 5, lambda r: False).records

    def test_it_says_what_went_wrong_on_standard_error(self, tmp_path: Path):
        found = find(["-e", "(", "--"], tmp_path, b"\n", 5)

        assert found.returncode == 2 and "regex parse error" in found.stderr

    def test_it_is_stopped_at_the_deadline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(_rg, "SEARCH_SECONDS", -1)
        make(tmp_path, "a.txt")

        found = files(tmp_path)

        assert found.timed_out and found.returncode is None

    def test_a_run_that_never_ends_is_killed_with_its_process_group(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # A stand-in for `rg` that prints a record and starts a child that sleeps: the
        # read ends only if the whole group is killed.
        script = tmp_path / "bin" / "rg"
        script.parent.mkdir()
        pidfile = tmp_path / "child.pid"
        script.write_text(f"#!/bin/sh\necho a\nsleep 30 &\necho $! > {pidfile}\nwait\n")
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{script.parent}:/bin:/usr/bin")
        monkeypatch.setattr(_rg, "SEARCH_SECONDS", 3)
        started = time.monotonic()

        found = find(["--"], tmp_path, b"\n", 5)

        assert time.monotonic() - started < 15
        assert found.timed_out and found.records == [b"a"]
        child = int(pidfile.read_text())
        for _ in range(50):  # it is signalled, and reaped a moment later
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(child, 9)
            pytest.fail("the child of the search was left running")

    def test_what_the_model_wrote_is_never_a_flag(self, tmp_path: Path):
        make(tmp_path, "a.txt")
        (tmp_path / "a.txt").write_text("--files\n")

        found = find(["-e", "--files", "--"], tmp_path, b"\n", 5)

        assert found.records == [b"./a.txt:--files"]

    def test_without_ripgrep_it_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("PATH", "")

        with pytest.raises(FileNotFoundError, match="rg is not installed"):
            files(tmp_path)

    def test_ripgrep_does_not_read_the_users_configuration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "rc").write_text("--glob=!*.txt\n")
        monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(tmp_path / "rc"))
        make(tmp_path, "a.txt")

        assert "a.txt" in paths(files(tmp_path))

    def test_it_runs_in_the_directory_it_was_given(self, tmp_path: Path):
        make(tmp_path, "sub/a.txt")

        assert paths(files(tmp_path / "sub")) == ["a.txt"]
        assert os.getcwd() == str(tmp_path)  # the process itself did not move
