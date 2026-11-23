import json
import os
import stat
from pathlib import Path

import pytest

from ergane.builtin._paths import PathError
from ergane.builtin.writes import _preview_edit, _preview_write, edit_file, write_file
from ergane.console import Console
from ergane.tools import Tool


def call(tool: Tool, **arguments: object) -> str:
    return tool.call(json.dumps(arguments))


class TestWriteFile:
    def test_a_new_file_is_created_and_the_model_is_told_in_a_line(
        self, tmp_path: Path
    ):
        result = call(write_file, path="a.txt", content="one\ntwo\n")

        assert result == "Created a.txt (2 lines)"
        assert (tmp_path / "a.txt").read_text() == "one\ntwo\n"

    def test_an_existing_file_is_overwritten(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("old old old")

        result = call(write_file, path="a.txt", content="new")

        assert result == "Overwrote a.txt (1 lines)"
        assert (tmp_path / "a.txt").read_text() == "new"

    def test_missing_directories_are_created(self, tmp_path: Path):
        call(write_file, path="x/y/z.txt", content="hi")

        assert (tmp_path / "x" / "y" / "z.txt").read_text() == "hi"

    def test_the_content_is_written_as_given(self, tmp_path: Path):
        call(write_file, path="w.txt", content="a\r\nb")

        assert (tmp_path / "w.txt").read_bytes() == b"a\r\nb"

    def test_an_empty_file_can_be_made(self, tmp_path: Path):
        assert call(write_file, path="empty", content="") == "Created empty (0 lines)"
        assert (tmp_path / "empty").read_bytes() == b""

    def test_text_in_any_script_is_written_as_utf8(self, tmp_path: Path):
        call(write_file, path="u.txt", content="café 中文 \U0001f600")

        assert (tmp_path / "u.txt").read_bytes().decode() == "café 中文 \U0001f600"

    def test_an_existing_files_permissions_are_kept(self, tmp_path: Path):
        path = tmp_path / "script.sh"
        path.write_text("#!/bin/sh\n")
        path.chmod(0o750)

        call(write_file, path="script.sh", content="#!/bin/sh\necho hi\n")

        assert stat.S_IMODE(path.stat().st_mode) == 0o750

    def test_a_path_outside_the_directories_is_refused_and_nothing_is_made(self):
        result = call(write_file, path="/etc/ergane-test.txt", content="x")

        assert "Access denied" in result and not Path("/etc/ergane-test.txt").exists()

    def test_a_directory_is_not_a_file_to_write(self, tmp_path: Path):
        (tmp_path / "d").mkdir()

        assert "is a directory, not a file" in call(write_file, path="d", content="x")

    def test_a_symbolic_link_is_refused_and_its_target_left_alone(self, tmp_path: Path):
        (tmp_path / "real.txt").write_text("keep")
        (tmp_path / "link.txt").symlink_to(tmp_path / "real.txt")

        result = call(write_file, path="link.txt", content="clobbered")

        assert "is a symbolic link" in result
        assert (tmp_path / "real.txt").read_text() == "keep"

    def test_a_link_to_somewhere_outside_is_refused(self, tmp_path: Path):
        outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
        outside.write_text("keep")
        (tmp_path / "link.txt").symlink_to(outside)

        result = call(write_file, path="link.txt", content="clobbered")

        assert "Error" in result and outside.read_text() == "keep"

    def test_a_file_that_is_not_regular_is_refused(self, tmp_path: Path):
        os.mkfifo(tmp_path / "pipe")

        assert "not a regular file" in call(write_file, path="pipe", content="x")

    def test_it_asks_first(self):
        assert not write_file.read_only and write_file.preview is not None


class TestEditFile:
    @pytest.fixture(autouse=True)
    def a_file(self, tmp_path: Path) -> None:
        (tmp_path / "f.py").write_text("a = 1\nb = 2\nc = 3\n")

    def test_the_text_is_replaced_and_the_model_is_told_in_a_line(self, tmp_path: Path):
        result = call(edit_file, path="f.py", old_text="b = 2", new_text="b = 20")

        assert result == "Edited f.py (1 replacement)"
        assert (tmp_path / "f.py").read_text() == "a = 1\nb = 20\nc = 3\n"

    def test_replace_all_counts_the_places(self, tmp_path: Path):
        (tmp_path / "r.txt").write_text("x\ny\nx\n")

        result = call(
            edit_file, path="r.txt", old_text="x", new_text="z", replace_all=True
        )

        assert result == "Edited r.txt (2 replacements)"
        assert (tmp_path / "r.txt").read_text() == "z\ny\nz\n"

    def test_an_ambiguous_edit_changes_nothing(self, tmp_path: Path):
        (tmp_path / "r.txt").write_text("x\ny\nx\n")

        result = call(edit_file, path="r.txt", old_text="x", new_text="z")

        assert "matches 2 places (lines 1, 3)" in result
        assert (tmp_path / "r.txt").read_text() == "x\ny\nx\n"

    def test_text_that_is_not_found_changes_nothing_and_shows_the_closest(
        self, tmp_path: Path
    ):
        result = call(edit_file, path="f.py", old_text="b = 3", new_text="b = 4")

        assert "Could not find old_text in f.py" in result
        assert "2: b = 2" in result
        assert (tmp_path / "f.py").read_text() == "a = 1\nb = 2\nc = 3\n"

    def test_a_crlf_file_stays_crlf_on_disk(self, tmp_path: Path):
        (tmp_path / "w.txt").write_bytes(b"a\r\nb\r\nc\r\n")

        call(edit_file, path="w.txt", old_text="b", new_text="B")

        assert (tmp_path / "w.txt").read_bytes() == b"a\r\nB\r\nc\r\n"

    def test_a_second_try_edit_leaves_the_other_lines_as_they_were(
        self, tmp_path: Path
    ):
        raw = "say(“hi”)  \nkeep – this\nx = 1\n"
        (tmp_path / "q.py").write_text(raw)

        call(edit_file, path="q.py", old_text='say("hi")', new_text='say("bye")')

        assert (tmp_path / "q.py").read_text() == 'say("bye")\nkeep – this\nx = 1\n'

    def test_a_missing_file_suggests_close_names(self, tmp_path: Path):
        result = call(edit_file, path="g.py", old_text="a", new_text="b")

        assert "File not found: g.py" in result and "Did you mean: f.py" in result

    def test_a_directory_is_not_a_file_to_edit(self, tmp_path: Path):
        (tmp_path / "d").mkdir()

        assert "is a directory, not a file" in call(
            edit_file, path="d", old_text="a", new_text="b"
        )

    def test_a_file_that_is_not_utf8_is_refused_and_untouched(self, tmp_path: Path):
        (tmp_path / "bin").write_bytes(b"caf\xe9 = 1\n")

        result = call(edit_file, path="bin", old_text="1", new_text="2")

        assert "not valid UTF-8" in result
        assert (tmp_path / "bin").read_bytes() == b"caf\xe9 = 1\n"

    def test_a_huge_file_is_refused(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("ergane.builtin.writes._MAX_EDIT_BYTES", 5)

        assert "too large to edit" in call(
            edit_file, path="f.py", old_text="a", new_text="b"
        )

    def test_a_symbolic_link_is_refused(self, tmp_path: Path):
        (tmp_path / "link.py").symlink_to(tmp_path / "f.py")

        result = call(edit_file, path="link.py", old_text="a = 1", new_text="a = 9")

        assert "is a symbolic link" in result
        assert (tmp_path / "f.py").read_text().startswith("a = 1")

    def test_a_path_outside_the_directories_is_refused(self):
        assert "Access denied" in call(
            edit_file, path="/etc/hosts", old_text="a", new_text="b"
        )

    def test_the_files_permissions_are_kept(self, tmp_path: Path):
        (tmp_path / "f.py").chmod(0o750)

        call(edit_file, path="f.py", old_text="a = 1", new_text="a = 2")

        assert stat.S_IMODE((tmp_path / "f.py").stat().st_mode) == 0o750

    def test_an_edit_that_changes_nothing_is_an_error(self):
        assert "No change" in call(
            edit_file, path="f.py", old_text="a = 1", new_text="a = 1"
        )

    def test_it_asks_first(self):
        assert not edit_file.read_only and edit_file.preview is not None


class TestThePreviews:
    def test_a_new_file_shows_how_it_starts(self, tmp_path: Path):
        text = _preview_write("n.txt", "one\ntwo\n")

        assert text == "New file n.txt (2 lines):\n+one\n+two"
        assert not (tmp_path / "n.txt").exists()  # looking is not doing

    def test_a_long_new_file_is_cut(self):
        text = _preview_write("n.txt", "\n".join(str(i) for i in range(500)))

        assert "(500 lines)" in text and "[... 380 more lines]" in text

    def test_an_overwrite_shows_a_diff(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")

        text = _preview_write("a.txt", "one\nTWO\nthree\n")

        assert text.splitlines()[:2] == ["--- a.txt", "+++ a.txt"]
        assert "-two" in text and "+TWO" in text
        assert (tmp_path / "a.txt").read_text() == "one\ntwo\nthree\n"

    def test_a_long_diff_is_cut_and_says_how_much_was_left(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("\n".join(f"old {i}" for i in range(300)))

        text = _preview_write("a.txt", "\n".join(f"new {i}" for i in range(300)))

        assert len(text.splitlines()) == 121  # the limit, and the line that says so
        assert text.splitlines()[-1].startswith("[... ") and "more lines]" in text

    def test_writing_the_same_content_says_so(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("same\n")

        assert _preview_write("a.txt", "same\n").startswith("No change")

    def test_a_binary_file_is_said_to_be_overwritten(self, tmp_path: Path):
        (tmp_path / "b.bin").write_bytes(b"\xff\xfe\x00")

        assert "is not text; it would be overwritten (3 bytes)" in _preview_write(
            "b.bin", "x"
        )

    def test_an_edit_shows_a_diff_and_writes_nothing(self, tmp_path: Path):
        (tmp_path / "f.py").write_text("a = 1\nb = 2\n")

        text = _preview_edit("f.py", "b = 2", "b = 3")

        assert "-b = 2" in text and "+b = 3" in text
        assert (tmp_path / "f.py").read_text() == "a = 1\nb = 2\n"

    def test_an_edit_that_would_fail_says_why(self, tmp_path: Path):
        (tmp_path / "f.py").write_text("a = 1\n")

        text = _preview_edit("f.py", "zzz", "y")

        assert text.startswith("This edit would fail: Could not find old_text")

    def test_a_path_that_is_refused_raises_for_the_console_to_report(self):
        with pytest.raises(PathError):
            _preview_write("/etc/hosts", "x")

    def test_a_link_is_refused_here_too(self, tmp_path: Path):
        (tmp_path / "t.txt").write_text("x")
        (tmp_path / "l.txt").symlink_to(tmp_path / "t.txt")

        with pytest.raises(ValueError, match="symbolic link"):
            _preview_write("l.txt", "y")


class TestTheQuestion:
    def test_the_user_can_see_the_change_before_it_is_allowed(
        self, tmp_path: Path, capsys
    ):
        (tmp_path / "f.py").write_text("a = 1\nb = 2\n")
        answers = iter(["?", "n"])
        console = Console()
        console.choose = lambda question, options: next(answers)
        arguments = json.dumps(
            {"path": "f.py", "old_text": "b = 2", "new_text": "b = 3"}
        )

        decision = console.confirm(edit_file, arguments)

        out = capsys.readouterr().out
        assert decision is False
        assert "--- f.py" in out and "-b = 2" in out and "+b = 3" in out
        assert (tmp_path / "f.py").read_text() == "a = 1\nb = 2\n"

    def test_a_change_to_a_file_that_is_missing_says_so_in_the_question(self, capsys):
        answers = iter(["?", "n"])
        console = Console()
        console.choose = lambda question, options: next(answers)
        arguments = json.dumps({"path": "nope.py", "old_text": "a", "new_text": "b"})

        console.confirm(edit_file, arguments)

        assert "Could not show the change: FileNotFoundError" in capsys.readouterr().out
