import base64
import json
import os
import shutil
import tracemalloc
from pathlib import Path

import pytest

from ergane.builtin.files import list_directory, read_file, search_files
from ergane.tools import MAX_OUTPUT_CHARS, Tool

# A 1x1 png: its chunks are in the right order, which is what the sniffing reads.
PNG_FILE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def call(tool: Tool, **arguments: object) -> str:
    """Call a tool as the model would, with JSON arguments."""
    return tool.call(json.dumps(arguments))


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class TestReadFile:
    def test_a_small_file_comes_back_whole(self, tmp_path: Path):
        write(tmp_path / "a.txt", "one\ntwo\nthree\n")

        assert call(read_file, path="a.txt") == "one\ntwo\nthree"

    def test_an_empty_file_says_so(self, tmp_path: Path):
        write(tmp_path / "empty.txt", "")

        assert call(read_file, path="empty.txt") == "(empty file)"

    def test_crlf_line_endings_read_as_lines(self, tmp_path: Path):
        (tmp_path / "w.txt").write_bytes(b"a\r\nb\r\n")

        assert call(read_file, path="w.txt") == "a\nb"

    def test_offset_and_limit_pick_a_range(self, tmp_path: Path):
        write(tmp_path / "n.txt", "\n".join(f"line {i}" for i in range(1, 11)))

        assert call(read_file, path="n.txt", offset=4, limit=3) == (
            "line 4\nline 5\nline 6"
        )

    def test_a_limit_that_cuts_the_file_needs_no_footer(self, tmp_path: Path):
        write(tmp_path / "n.txt", "\n".join(str(i) for i in range(1, 100)))

        assert "Continue with" not in call(read_file, path="n.txt", limit=5)

    def test_an_offset_past_the_end_says_how_long_the_file_is(self, tmp_path: Path):
        write(tmp_path / "n.txt", "a\nb\nc\n")

        assert call(read_file, path="n.txt", offset=10) == (
            "[offset 10 is past the end of the file, which has 3 lines]"
        )

    def test_a_long_file_is_cut_with_the_offset_to_continue_from(self, tmp_path: Path):
        write(
            tmp_path / "big.txt",
            "\n".join(f"line {i:05d} " + "x" * 90 for i in range(1, 1001)),
        )

        first = call(read_file, path="big.txt")

        assert first.startswith("line 00001")
        assert "more lines follow. Continue with offset=" in first
        assert len(first) < MAX_OUTPUT_CHARS  # the footer survives the generic cut
        next_offset = int(first.rsplit("offset=", 1)[1].rstrip("]"))
        second = call(read_file, path="big.txt", offset=next_offset)
        assert second.startswith(f"line {next_offset:05d}")  # no line lost or repeated

    def test_reading_on_to_the_end_leaves_no_footer(self, tmp_path: Path):
        write(
            tmp_path / "big.txt",
            "\n".join(f"line {i:05d} " + "x" * 90 for i in range(1, 400)),
        )
        offset = 1
        parts = []
        while True:
            part = call(read_file, path="big.txt", offset=offset)
            if "Continue with offset=" not in part:
                parts.append(part)
                break
            body, footer = part.rsplit("\n[...", 1)
            parts.append(body)
            offset = int(footer.rsplit("offset=", 1)[1].rstrip("]"))

        assert "\n".join(parts).count("\n") + 1 == 399

    def test_an_over_long_line_is_cut(self, tmp_path: Path):
        write(tmp_path / "min.js", "x" * 50_000)

        text = call(read_file, path="min.js")

        assert len(text) < 11_000 and "[line cut: 50,000 characters]" in text

    def test_a_picture_comes_back_as_a_picture_with_a_line_that_names_it(
        self, tmp_path: Path
    ):
        (tmp_path / "logo.png").write_bytes(PNG_FILE)

        result = read_file.call_output(json.dumps({"path": "logo.png"}))

        assert (
            result.text
            == f"Picture: logo.png (image/png, {len(PNG_FILE) / 1000:.0f} KB)"
        )
        assert [(i.mime_type, base64.b64decode(i.data)) for i in result.images] == [
            ("image/png", PNG_FILE)
        ]

    def test_the_name_does_not_decide_what_a_file_is(self, tmp_path: Path):
        write(tmp_path / "words.png", "just text\n")
        (tmp_path / "picture.txt").write_bytes(PNG_FILE)

        assert call(read_file, path="words.png") == "just text"
        assert read_file.call_output(json.dumps({"path": "picture.txt"})).images

    def test_the_text_of_a_picture_is_all_that_call_returns(self, tmp_path: Path):
        (tmp_path / "logo.png").write_bytes(PNG_FILE)

        assert call(read_file, path="logo.png").startswith("Picture: logo.png")

    def test_a_picture_over_the_limit_is_refused_and_says_what_to_do(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "big.png").write_bytes(PNG_FILE)
        monkeypatch.setattr("ergane.builtin._images.MAX_IMAGE_CHARS", 8)

        result = call(read_file, path="big.png")

        assert "is a image/png picture" in result and "ask for a smaller one" in result

    def test_the_limit_counts_the_base64_that_is_sent_not_the_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "big.png").write_bytes(PNG_FILE)
        # The file is under this many bytes; its base64 text is not.
        monkeypatch.setattr(
            "ergane.builtin._images.MAX_IMAGE_CHARS", len(PNG_FILE) + 10
        )

        assert "ask for a smaller one" in call(read_file, path="big.png")

    def test_a_picture_at_the_limit_is_sent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "ok.png").write_bytes(PNG_FILE)
        monkeypatch.setattr(
            "ergane.builtin._images.MAX_IMAGE_CHARS", len(PNG_FILE) // 3 * 4 + 4
        )

        assert read_file.call_output(json.dumps({"path": "ok.png"})).images

    def test_an_animated_png_is_a_binary_file_like_any_other(self, tmp_path: Path):
        animated = PNG_FILE.replace(b"IDAT", b"acTL\0\0\0\0\0\0\0\0\0\0\0\0IDAT", 1)
        (tmp_path / "anim.png").write_bytes(animated)

        assert "looks like a binary file" in call(read_file, path="anim.png")

    def test_a_byte_order_mark_is_not_shown(self, tmp_path: Path):
        (tmp_path / "bom.txt").write_bytes(b"\xef\xbb\xbftitle\nbody\n")

        assert call(read_file, path="bom.txt") == "title\nbody"

    def test_a_huge_line_does_not_fill_the_memory(self, tmp_path: Path):
        # 100 MB on one line used to be read whole, and decoded, into memory.
        with (tmp_path / "one.txt").open("wb") as handle:
            for _ in range(100):
                handle.write(b"a" * 1_000_000)
        tracemalloc.start()
        try:
            text = call(read_file, path="one.txt")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert "[line cut: 100,000,000 characters]" in text
        assert peak < 20_000_000

    def test_an_offset_past_the_end_of_a_file_of_long_lines_says_so(
        self, tmp_path: Path
    ):
        write(tmp_path / "long.txt", ("x" * 50_000 + "\n") * 3)

        assert "past the end of the file, which has 3 lines" in call(
            read_file, path="long.txt", offset=10
        )

    def test_a_binary_file_is_refused(self, tmp_path: Path):
        (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02binary")

        assert "looks like a binary file" in call(read_file, path="b.bin")

    def test_text_that_is_not_utf8_late_in_the_file_is_refused(self, tmp_path: Path):
        (tmp_path / "late.txt").write_bytes(b"fine\n" * 5000 + b"\xff\xfe broken\n")

        assert "is not valid UTF-8" in call(read_file, path="late.txt", offset=4999)

    def test_a_missing_file_suggests_the_close_names(self, tmp_path: Path):
        write(tmp_path / "src" / "config.py", "")
        write(tmp_path / "src" / "cli.py", "")

        result = call(read_file, path="src/confg.py")

        assert "File not found: src/confg.py" in result
        assert "Did you mean: src/config.py" in result

    def test_a_missing_file_with_nothing_close_just_says_so(self, tmp_path: Path):
        assert call(read_file, path="zzz.txt").endswith("File not found: zzz.txt")

    def test_a_directory_is_not_a_file_to_read(self, tmp_path: Path):
        (tmp_path / "d").mkdir()

        result = call(read_file, path="d")

        assert "is a directory, not a file" in result and "list_directory" in result

    def test_a_path_outside_the_directories_is_refused_with_what_to_do(self):
        result = call(read_file, path="/etc/hosts")

        assert "Access denied" in result and "ask the user" in result

    def test_it_runs_without_asking(self):
        assert read_file.read_only

    def test_a_bad_argument_is_answered_without_running(self):
        assert call(read_file, path="a", offset=0).startswith("Invalid arguments")


class TestListDirectory:
    def test_entries_come_one_a_line_directories_first(self, tmp_path: Path):
        write(tmp_path / "b.txt", "")
        write(tmp_path / "a.txt", "")
        (tmp_path / "zdir").mkdir()
        (tmp_path / "adir").mkdir()

        assert call(list_directory, path=".") == "adir/\nzdir/\na.txt\nb.txt"

    def test_the_path_is_the_working_directory_when_omitted(self, tmp_path: Path):
        write(tmp_path / "a.txt", "")

        assert call(list_directory) == "a.txt"

    def test_a_subdirectory_is_listed_by_its_path(self, tmp_path: Path):
        write(tmp_path / "src" / "a.py", "")
        write(tmp_path / "top.py", "")

        assert call(list_directory, path="src") == "a.py"

    def test_a_symbolic_link_ends_in_an_at_sign_and_is_not_followed(
        self, tmp_path: Path
    ):
        write(tmp_path / "real.txt", "")
        (tmp_path / "link").symlink_to(tmp_path / "real.txt")

        assert call(list_directory, path=".").splitlines() == ["link@", "real.txt"]

    def test_hidden_entries_are_listed(self, tmp_path: Path):
        write(tmp_path / ".env", "")

        assert ".env" in call(list_directory, path=".")

    def test_an_empty_directory_says_so(self, tmp_path: Path):
        (tmp_path / "e").mkdir()

        assert call(list_directory, path="e") == "(empty directory)"

    def test_a_long_listing_is_cut_and_says_how_to_go_on(self, tmp_path: Path):
        for i in range(800):
            write(tmp_path / f"file-{i:04d}-{'x' * 30}.txt", "")

        result = call(list_directory, path=".")

        assert "list a subdirectory, or use search_files" in result
        assert "showing " in result and " of 800 entries" in result
        assert len(result) < MAX_OUTPUT_CHARS

    def test_a_file_is_not_a_directory_to_list(self, tmp_path: Path):
        write(tmp_path / "a.txt", "")

        result = call(list_directory, path="a.txt")

        assert "is a file, not a directory" in result and "read_file" in result

    def test_a_missing_directory_suggests_the_close_names(self, tmp_path: Path):
        (tmp_path / "source").mkdir()

        result = call(list_directory, path="sorce")

        assert "Directory not found: sorce" in result
        assert "Did you mean: source" in result

    def test_a_path_outside_the_directories_is_refused(self):
        assert "Access denied" in call(list_directory, path="/etc")

    def test_it_runs_without_asking(self):
        assert list_directory.read_only


class TestTheListingLeavesOutWhatGitIgnores:
    def test_ignored_entries_and_the_git_directory_are_left_out_and_counted(
        self, tmp_path: Path
    ):
        write(tmp_path / ".gitignore", "*.log\n__pycache__/\n")
        write(tmp_path / "keep.py", "")
        write(tmp_path / "app.log", "")
        write(tmp_path / "__pycache__" / "x.pyc", "")
        write(tmp_path / ".git" / "HEAD", "")

        assert call(list_directory, path=".").splitlines() == [
            ".gitignore",
            "keep.py",
            "[3 entries left out: ignored by .gitignore, or .git]",
        ]

    def test_one_entry_says_entry(self, tmp_path: Path):
        write(tmp_path / ".gitignore", "a.log\n")
        write(tmp_path / "a.log", "")

        assert call(list_directory, path=".").endswith(
            "[1 entry left out: ignored by .gitignore, or .git]"
        )

    def test_nothing_left_out_says_nothing(self, tmp_path: Path):
        write(tmp_path / "a.txt", "")

        assert call(list_directory, path=".") == "a.txt"

    def test_a_directory_with_only_ignored_entries_says_so(self, tmp_path: Path):
        write(tmp_path / ".gitignore", "*\n")
        write(tmp_path / "a.txt", "")

        assert call(list_directory, path=".").startswith("(no entries to show)")

    def test_a_file_that_is_ignored_can_still_be_read_by_name(self, tmp_path: Path):
        # Ignoring is about what is shown, not what may be opened.
        write(tmp_path / ".gitignore", ".env\n")
        write(tmp_path / ".env", "KEY=1\n")

        assert call(read_file, path=".env") == "KEY=1"

    def test_the_rules_of_the_repository_above_apply(self, tmp_path: Path):
        write(tmp_path / ".git" / "HEAD", "")
        write(tmp_path / ".gitignore", "__pycache__/\n*.pyc\n")
        write(tmp_path / "src" / "pkg" / "mod.py", "")
        write(tmp_path / "src" / "pkg" / "__pycache__" / "mod.pyc", "")
        write(tmp_path / "src" / "pkg" / "cache.pyc", "")

        assert call(list_directory, path="src/pkg").splitlines() == [
            "mod.py",
            "[2 entries left out: ignored by .gitignore, or .git]",
        ]

    def test_without_a_repository_only_the_directorys_own_file_counts(
        self, tmp_path: Path
    ):
        # No .git anywhere above, so a parent's .gitignore is not a git rule.
        write(tmp_path / ".gitignore", "*.py\n")
        write(tmp_path / "sub" / "a.py", "")

        assert call(list_directory, path="sub") == "a.py"

    def test_a_deeper_file_can_bring_back_what_a_shallower_one_ignored(
        self, tmp_path: Path
    ):
        write(tmp_path / ".git" / "HEAD", "")
        write(tmp_path / ".gitignore", "*.log\n")
        write(tmp_path / "sub" / ".gitignore", "!keep.log\n")
        write(tmp_path / "sub" / "keep.log", "")
        write(tmp_path / "sub" / "drop.log", "")

        assert call(list_directory, path="sub").splitlines()[:2] == [
            ".gitignore",
            "keep.log",
        ]

    def test_a_parents_file_does_not_reach_a_sibling_directory(self, tmp_path: Path):
        write(tmp_path / ".git" / "HEAD", "")
        write(tmp_path / "a" / ".gitignore", "*.tmp\n")
        write(tmp_path / "b" / "x.tmp", "")

        assert call(list_directory, path="b") == "x.tmp"

    def test_a_pattern_git_rejects_is_skipped_not_fatal(self, tmp_path: Path):
        write(tmp_path / ".gitignore", "[\n")
        write(tmp_path / "a.txt", "")

        assert "a.txt" in call(list_directory, path=".")


needs_rg = pytest.mark.skipif(shutil.which("rg") is None, reason="needs ripgrep")


def repository(root: Path) -> None:
    """Make `root` a git repository, which is where `.gitignore` files count."""
    write(root / ".git" / "HEAD", "")


@needs_rg
class TestTheListingAndTheSearchAgree:
    """They leave out the same files: what `.gitignore` excludes, and `.git`."""

    def test_the_same_files_are_left_out(self, tmp_path: Path):
        repository(tmp_path)
        write(tmp_path / ".gitignore", "*.log\nbuild/\n!keep.log\n")
        write(tmp_path / "keep.log", "")
        write(tmp_path / "drop.log", "")
        write(tmp_path / "build" / "out.o", "")
        write(tmp_path / "src" / "a.py", "")
        write(tmp_path / "Zeta.txt", "")
        write(tmp_path / "alpha.txt", "")

        listed = call(list_directory, path=".").split("\n[")[0].splitlines()
        files = sorted(entry for entry in listed if not entry.endswith("/"))
        searched = call(search_files, name="*").splitlines()
        top = [path for path in searched if "/" not in path]

        assert files == sorted(top)
        assert "keep.log" in searched  # brought back by the negation
        for left_out in ("drop.log", "build/out.o", ".git/HEAD"):
            assert left_out not in searched
        assert "src/a.py" in searched


@needs_rg
class TestSearchByName:
    @pytest.fixture(autouse=True)
    def tree(self, tmp_path: Path) -> None:
        write(tmp_path / "top.py", "")
        write(tmp_path / "src" / "a.py", "")
        write(tmp_path / "src" / "deep" / "b.py", "")
        write(tmp_path / "src" / "notes.txt", "")
        write(tmp_path / ".hidden.py", "")

    def test_a_star_matches_at_any_depth_as_in_a_gitignore(self):
        assert call(search_files, name="*.py").splitlines() == [
            ".hidden.py",
            "src/a.py",
            "src/deep/b.py",
            "top.py",
        ]

    def test_a_leading_slash_matches_only_directly_under_the_path(self):
        assert call(search_files, name="/*.py").splitlines() == [".hidden.py", "top.py"]

    def test_a_glob_with_a_directory_is_relative_to_the_path(self):
        assert call(search_files, name="src/**/*.py").splitlines() == [
            "src/a.py",
            "src/deep/b.py",
        ]

    def test_paths_are_relative_to_the_search_directory(self):
        assert call(search_files, path="src", name="*.py").splitlines() == [
            "a.py",
            "deep/b.py",
        ]

    def test_a_directory_name_finds_the_files_in_it_as_in_a_gitignore(self):
        # Only files are ever listed: a directory is for `list_directory`.
        assert call(search_files, name="deep") == "src/deep/b.py"

    def test_matching_is_case_sensitive(self):
        assert call(search_files, name="*.PY") == "No matches found"

    def test_no_match_says_so(self):
        assert call(search_files, name="*.rs") == "No matches found"

    def test_the_limit_stops_the_result_and_says_so(self):
        result = call(search_files, name="*.py", limit=2)

        assert result.splitlines()[:2] == [".hidden.py", "src/a.py"]
        assert "stopped after 2 results" in result

    def test_a_symbolic_link_is_never_followed_or_listed(self, tmp_path: Path):
        outside = tmp_path.parent / f"{tmp_path.name}-outside"
        write(outside / "secret.py", "")
        (tmp_path / "escape").symlink_to(outside)
        (tmp_path / "alias.py").symlink_to(tmp_path / "top.py")

        result = call(search_files, name="*.py")

        assert "secret.py" not in result and "alias.py" not in result

    def test_a_named_pipe_is_not_listed(self, tmp_path: Path):
        os.mkfifo(tmp_path / "pipe")

        assert call(search_files, name="pipe") == "No matches found"

    def test_a_bad_glob_is_explained(self):
        assert "name is not a valid glob" in call(search_files, name="\\")

    def test_a_name_that_looks_like_a_flag_is_a_glob(self, tmp_path: Path):
        write(tmp_path / "--files", "")

        assert call(search_files, name="--files") == "--files"


@needs_rg
class TestSearchByContent:
    @pytest.fixture(autouse=True)
    def tree(self, tmp_path: Path) -> None:
        write(tmp_path / "a.py", "import os\n\ndef run():\n    return os.getcwd()\n")
        write(tmp_path / "src" / "b.py", "import sys\nprint('os is not here')\n")
        write(tmp_path / "notes.txt", "os notes\n")
        (tmp_path / "data.bin").write_bytes(b"os\x00\x01 binary")

    def test_matching_lines_come_with_path_and_line_number(self):
        assert call(search_files, contains=r"^import os").splitlines() == [
            "a.py:1: import os"
        ]

    def test_it_is_a_regular_expression(self):
        assert call(search_files, contains=r"def \w+\(") == "a.py:3: def run():"

    def test_the_text_of_a_line_loses_its_indentation(self):
        assert "a.py:4: return os.getcwd()" in call(search_files, contains="getcwd")

    def test_a_colon_in_a_path_or_a_line_is_kept(self, tmp_path: Path):
        write(tmp_path / "a:b.txt", "x: y: needle: z\n")

        assert "a:b.txt:1: x: y: needle: z" in call(search_files, contains="needle")

    def test_a_pattern_that_would_backtrack_without_end_finishes(self, tmp_path: Path):
        # With Python's `re` this never returned, on a thread that Ctrl-C cannot stop.
        write(tmp_path / "trap.txt", "a" * 46 + "!\n")

        assert call(search_files, contains=r"(a|aa)+$", name="trap.txt") == (
            "No matches found"
        )

    def test_a_search_that_runs_out_of_time_says_so(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("ergane.builtin._rg.SEARCH_SECONDS", -1)
        monkeypatch.setattr("ergane.builtin.files.SEARCH_SECONDS", -1)

        assert "gave up after -1 seconds" in call(search_files, contains="os")

    def test_a_case_insensitive_pattern_works_with_an_inline_flag(self):
        assert "a.py:1: import os" in call(search_files, contains=r"(?i)IMPORT OS")

    def test_name_narrows_the_files_that_are_searched(self):
        result = call(search_files, name="*.py", contains="os")

        assert "notes.txt" not in result and "a.py:1: import os" in result

    def test_a_binary_file_is_skipped(self):
        assert "data.bin" not in call(search_files, contains="os")

    def test_the_limit_counts_matching_lines(self):
        result = call(search_files, contains="os", limit=2)

        assert len([l for l in result.splitlines() if ":" in l and "[" not in l]) == 2
        assert "stopped after 2 results" in result

    def test_a_long_matching_line_is_cut(self, tmp_path: Path):
        write(tmp_path / "wide.js", "needle " + "x" * 5_000)

        line = next(
            l
            for l in call(search_files, contains="needle").splitlines()
            if "wide.js" in l
        )

        assert len(line) < 400 and "omitted end of long line" in line

    def test_a_file_over_a_megabyte_is_skipped(self, tmp_path: Path):
        write(tmp_path / "huge.txt", "needle\n" + "x" * 1_100_000)

        assert "huge.txt" not in call(search_files, contains="needle")

    def test_a_symbolic_link_to_a_file_is_not_read(self, tmp_path: Path):
        outside = tmp_path.parent / f"{tmp_path.name}-secret.txt"
        outside.write_text("needle\n")
        (tmp_path / "link.txt").symlink_to(outside)

        assert "link.txt" not in call(search_files, contains="needle")

    @pytest.mark.timeout(10)
    def test_a_named_pipe_is_not_read_for_ever(self, tmp_path: Path):
        # Reading a FIFO with no writer blocks, and a search runs on a thread that
        # Ctrl-C cannot stop.
        os.mkfifo(tmp_path / "pipe")
        write(tmp_path / "a.txt", "needle\n")

        assert call(search_files, contains="needle") == "a.txt:1: needle"

    def test_a_bad_regular_expression_is_explained(self):
        result = call(search_files, contains="(")

        assert "not a valid regular expression: unclosed group" in result

    def test_a_lookaround_is_explained_not_silently_missed(self):
        assert "look-around" in call(search_files, contains="(?<=a)b")

    def test_a_pattern_that_looks_like_a_flag_is_a_pattern(self, tmp_path: Path):
        write(tmp_path / "flags.txt", "use --files here\n")

        assert call(search_files, contains="--files") == "flags.txt:1: use --files here"

    def test_the_output_stays_under_the_generic_cut(self, tmp_path: Path):
        write(
            tmp_path / "lines.txt",
            "\n".join(f"hit {i} " + "y" * 250 for i in range(900)),
        )

        result = call(search_files, contains="hit", limit=500)

        assert len(result) < MAX_OUTPUT_CHARS and "narrow the search" in result

    def test_the_order_is_the_same_every_time(self, tmp_path: Path):
        for number in range(30):
            write(tmp_path / f"f{number:02d}.txt", "needle\n")

        first = call(search_files, contains="needle")

        assert [call(search_files, contains="needle") for _ in range(3)] == [first] * 3
        assert first.splitlines()[:2] == ["f00.txt:1: needle", "f01.txt:1: needle"]


@needs_rg
class TestWhatIsLeftOut:
    def test_gitignore_hides_files_and_directories(self, tmp_path: Path):
        repository(tmp_path)
        write(tmp_path / ".gitignore", "*.log\nbuild/\n/dist\n")
        write(tmp_path / "keep.py", "")
        write(tmp_path / "app.log", "")
        write(tmp_path / "build" / "out.py", "")
        write(tmp_path / "dist" / "x.py", "")
        write(tmp_path / "src" / "dist" / "y.py", "")

        assert call(search_files, name="*").splitlines() == [
            ".gitignore",
            "keep.py",
            "src/dist/y.py",
        ]

    def test_a_negation_brings_a_file_back(self, tmp_path: Path):
        repository(tmp_path)
        write(tmp_path / ".gitignore", "*.log\n!keep.log\n")
        write(tmp_path / "a.log", "")
        write(tmp_path / "keep.log", "")

        assert call(search_files, name="*.log") == "keep.log"

    def test_a_gitignore_below_the_path_applies_to_its_own_directory_only(
        self, tmp_path: Path
    ):
        repository(tmp_path)
        write(tmp_path / "sub" / ".gitignore", "*.tmp\n")
        write(tmp_path / "sub" / "a.tmp", "")
        write(tmp_path / "b.tmp", "")

        assert call(search_files, name="*.tmp") == "b.tmp"

    def test_gitignored_files_are_not_searched_for_content(self, tmp_path: Path):
        repository(tmp_path)
        write(tmp_path / ".gitignore", "secret.txt\n")
        write(tmp_path / "secret.txt", "needle\n")
        write(tmp_path / "open.txt", "needle\n")

        assert call(search_files, contains="needle") == "open.txt:1: needle"

    def test_the_rules_of_the_repository_above_the_searched_directory_apply(
        self, tmp_path: Path
    ):
        repository(tmp_path)
        write(tmp_path / ".gitignore", "__pycache__/\n")
        write(tmp_path / "src" / "a.py", "needle\n")
        write(tmp_path / "src" / "__pycache__" / "a.py", "needle\n")

        assert call(search_files, path="src", contains="needle") == "a.py:1: needle"
        assert call(search_files, path="src", name="__pycache__/*") == (
            "No matches found"
        )

    def test_the_git_directory_is_never_entered(self, tmp_path: Path):
        write(tmp_path / ".git" / "config", "needle\n")
        write(tmp_path / "a.txt", "needle\n")

        assert call(search_files, contains="needle") == "a.txt:1: needle"
        assert call(search_files, name="config") == "No matches found"

    def test_dotfiles_are_searched(self, tmp_path: Path):
        write(tmp_path / ".env.example", "needle\n")

        assert call(search_files, contains="needle") == ".env.example:1: needle"

    @pytest.mark.parametrize(
        ("file", "text"),
        [
            (".ignore", "hidden.txt\n"),
            (".rgignore", "hidden.txt\n"),
            (".git/info/exclude", "hidden.txt\n"),
        ],
    )
    def test_only_gitignore_files_hide_anything(
        self, tmp_path: Path, file: str, text: str
    ):
        # What `list_directory` reads, and no more: a project's own `.ignore` files for
        # other tools are not a way to hide a file from the model.
        repository(tmp_path)
        write(tmp_path / file, text)
        write(tmp_path / "hidden.txt", "needle\n")

        assert call(search_files, contains="needle") == "hidden.txt:1: needle"

    def test_the_users_ripgrep_configuration_is_not_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        write(tmp_path / "rc", "--glob=!*.txt\n")
        monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(tmp_path / "rc"))
        write(tmp_path / "a.txt", "needle\n")

        assert call(search_files, contains="needle") == "a.txt:1: needle"


@needs_rg
class TestSearchArguments:
    def test_neither_name_nor_contains_is_an_error(self):
        assert "Give name, contains, or both" in call(search_files)

    def test_a_file_is_not_a_directory_to_search(self, tmp_path: Path):
        write(tmp_path / "a.txt", "")

        result = call(search_files, path="a.txt", name="*")

        assert "is not a directory" in result and "read_file" in result

    def test_a_path_outside_the_directories_is_refused(self):
        assert "Access denied" in call(search_files, path="/etc", name="*")

    def test_it_runs_without_asking(self):
        assert search_files.read_only

    def test_it_needs_ripgrep_to_be_offered(self):
        assert search_files.requires == "rg"

    def test_an_unreadable_directory_is_reported_not_fatal(self, tmp_path: Path):
        write(tmp_path / "ok" / "a.txt", "")
        locked = tmp_path / "locked"
        locked.mkdir()
        write(locked / "b.txt", "")
        locked.chmod(0)
        try:
            if os.access(locked, os.R_OK):  # running as root: nothing to test
                pytest.skip("directory permissions do not apply")
            result = call(search_files, name="*.txt")
        finally:
            locked.chmod(0o755)

        assert "ok/a.txt" in result and "could not be read" in result

    def test_without_ripgrep_a_call_says_so_instead_of_failing_oddly(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("PATH", "")

        assert "rg is not installed" in call(search_files, name="*")
