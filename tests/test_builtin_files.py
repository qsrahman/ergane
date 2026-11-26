import base64
import json
import os
import struct
import tracemalloc
from pathlib import Path

import pytest

from ergane.builtin.files import _picture_type, read_file
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


def chunk(kind: bytes, body: bytes = b"") -> bytes:
    return struct.pack(">I", len(body)) + kind + body + b"\0\0\0\0"


PNG = b"\x89PNG\r\n\x1a\n"
IHDR = chunk(b"IHDR", b"\0" * 13)


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
        monkeypatch.setattr("ergane.builtin.files.MAX_IMAGE_CHARS", 8)

        result = call(read_file, path="big.png")

        assert "is a image/png picture" in result and "ask for a smaller one" in result

    def test_the_limit_counts_the_base64_that_is_sent_not_the_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "big.png").write_bytes(PNG_FILE)
        # The file is under this many bytes; its base64 text is not.
        monkeypatch.setattr("ergane.builtin.files.MAX_IMAGE_CHARS", len(PNG_FILE) + 10)

        assert "ask for a smaller one" in call(read_file, path="big.png")

    def test_a_picture_at_the_limit_is_sent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "ok.png").write_bytes(PNG_FILE)
        monkeypatch.setattr(
            "ergane.builtin.files.MAX_IMAGE_CHARS", len(PNG_FILE) // 3 * 4 + 4
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

    def test_a_file_that_is_not_utf8_is_not_called_binary(self, tmp_path: Path):
        # The model can read it with `run_command`; it can do nothing with a binary one.
        (tmp_path / "latin1.txt").write_bytes("café au lait\n".encode("latin-1"))

        result = call(read_file, path="latin1.txt")

        assert "is not valid UTF-8 text" in result and "binary" not in result

    @pytest.mark.timeout(10)
    def test_a_named_pipe_is_refused_not_waited_for(self, tmp_path: Path):
        # Opening a pipe nobody writes to blocks for ever, on a thread Ctrl-C cannot stop.
        os.mkfifo(tmp_path / "pipe")

        assert "not a regular file" in call(read_file, path="pipe")

    def test_a_limit_does_not_read_the_rest_of_a_file_of_long_lines(
        self, tmp_path: Path
    ):
        # 16,001 lines of 5,000 characters were read to show one of them: 80 MB.
        with (tmp_path / "wide.txt").open("w") as handle:
            for _ in range(20_000):
                handle.write("x" * 5_000 + "\n")
        tracemalloc.start()
        try:
            text = call(read_file, path="wide.txt", limit=1)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert text == "x" * 5_000
        assert peak < 5_000_000

    def test_a_file_of_long_lines_still_says_where_to_continue(self, tmp_path: Path):
        write(tmp_path / "wide.txt", ("x" * 5_000 + "\n") * 10)

        result = call(read_file, path="wide.txt")

        assert result.endswith("Continue with offset=4]")
        assert result.count("x" * 5_000) == 3

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

    def test_a_missing_file_in_a_directory_that_cannot_be_listed_just_says_so(
        self, tmp_path: Path
    ):
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o311)  # may be entered, not read
        try:
            if os.access(locked, os.R_OK):  # running as root: nothing to test
                pytest.skip("directory permissions do not apply")
            result = call(read_file, path="locked/zzz.txt")
        finally:
            locked.chmod(0o755)

        assert result == "Error: FileNotFoundError: File not found: locked/zzz.txt"

    def test_a_missing_file_with_nothing_close_just_says_so(self, tmp_path: Path):
        assert call(read_file, path="zzz.txt").endswith("File not found: zzz.txt")

    def test_a_directory_is_not_a_file_to_read(self, tmp_path: Path):
        (tmp_path / "d").mkdir()

        result = call(read_file, path="d")

        assert "is a directory, not a file" in result and "run_command" in result

    def test_a_path_outside_the_directories_is_refused_with_what_to_do(self):
        result = call(read_file, path="/etc/hosts")

        assert "Access denied" in result and "ask the user" in result

    def test_it_runs_without_asking(self):
        assert read_file.read_only

    def test_a_bad_argument_is_answered_without_running(self):
        assert call(read_file, path="a", offset=0).startswith("Invalid arguments")


class TestWhichFilesArePictures:
    @pytest.mark.parametrize(
        ("head", "mime"),
        [
            (PNG + IHDR + chunk(b"IDAT", b"x") + chunk(b"IEND"), "image/png"),
            (b"\xff\xd8\xff\xe0\x00\x10JFIF", "image/jpeg"),
            (b"GIF89a\x01\x00\x01\x00", "image/gif"),
            (b"GIF87a\x01\x00\x01\x00", "image/gif"),
            (b"RIFF\x24\x00\x00\x00WEBPVP8 ", "image/webp"),
        ],
    )
    def test_the_first_bytes_say_what_it_is(self, head, mime):
        assert _picture_type(head) == mime

    def test_a_png_with_no_image_data_in_what_was_read_is_still_a_png(self):
        assert _picture_type(PNG + IHDR + chunk(b"tEXt", b"k\0v")) == "image/png"

    def test_an_animated_png_is_not_a_picture(self):
        animated = PNG + IHDR + chunk(b"acTL", b"\0" * 8) + chunk(b"IDAT", b"x")

        assert _picture_type(animated) is None

    def test_an_animation_chunk_after_the_image_data_does_not_count(self):
        # The data comes first, so this is a still (and acTL there is not a thing).
        assert (
            _picture_type(PNG + IHDR + chunk(b"IDAT") + chunk(b"acTL")) == "image/png"
        )

    @pytest.mark.parametrize(
        "head",
        [
            PNG,  # a signature and nothing else
            PNG + chunk(b"tEXt", b"x"),  # no IHDR first
            b"RIFF\x24\x00\x00\x00WAVEfmt ",  # a RIFF file that is not a picture
            b"BM" + b"\0" * 30,  # a bitmap: the API does not take it
            b"GIF90a",
            b"hello, world\n",
            b"",
        ],
    )
    def test_anything_else_is_not(self, head):
        assert _picture_type(head) is None
