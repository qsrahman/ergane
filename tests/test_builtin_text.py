import pytest

from ergane.builtin._text import (
    MAX_LINE_CHARS,
    MAX_TEXT_CHARS,
    head_tail,
    is_text,
    take_lines,
)
from ergane.tools import MAX_OUTPUT_CHARS


def test_a_footer_fits_under_the_generic_cut():
    # The tool's own footer must never be what the 20 KiB cut removes.
    assert MAX_TEXT_CHARS + 1_000 < MAX_OUTPUT_CHARS


class TestIsText:
    @pytest.mark.parametrize(
        "data",
        [b"", b"plain", "café 中文".encode(), b"line\r\nline\n"],
    )
    def test_text(self, data: bytes):
        assert is_text(data)

    def test_a_nul_byte_is_binary(self):
        assert not is_text(b"abc\0def")

    def test_bytes_that_are_not_utf8_are_binary(self):
        assert not is_text(b"\xff\xfe\x00\x01")
        assert not is_text(b"caf\xe9 au lait")

    def test_a_character_cut_in_two_at_the_end_is_not_an_error(self):
        # The start of a file may end inside a multi-byte character.
        assert is_text("café".encode()[:-1])


LINES = [f"line {i}" for i in range(1, 11)]


class TestTakeLines:
    def test_all_lines_when_they_fit(self):
        text, end = take_lines(LINES)

        assert text == "\n".join(LINES)
        assert end == len(LINES)

    def test_from_a_start(self):
        text, end = take_lines(LINES, start=7)

        assert text == "line 8\nline 9\nline 10"
        assert end == 10

    def test_at_most_count_lines(self):
        text, end = take_lines(LINES, start=2, count=3)

        assert text == "line 3\nline 4\nline 5"
        assert end == 5

    def test_whole_lines_only_when_the_budget_runs_out(self):
        text, end = take_lines(["aaaa"] * 10, max_chars=12)

        # Each line costs its length and a newline: 5, 10; a third would be 15.
        assert text == "aaaa\naaaa"
        assert end == 2

    def test_at_least_one_line_even_over_the_budget(self):
        text, end = take_lines(["a" * 50, "b"], max_chars=10)

        assert text == "a" * 50
        assert end == 1

    def test_a_start_past_the_end_takes_nothing(self):
        assert take_lines(LINES, start=99) == ("", 99)

    def test_a_very_long_line_is_cut_and_says_so(self):
        text, end = take_lines(["x" * (MAX_LINE_CHARS + 500)])

        assert text.startswith("x" * MAX_LINE_CHARS + "...")
        assert f"[line cut: {MAX_LINE_CHARS + 500:,} characters]" in text
        assert end == 1

    def test_the_default_budget_leaves_room_for_a_footer(self):
        text, _ = take_lines(["y" * 100] * 1_000)

        assert len(text) <= MAX_TEXT_CHARS


class TestHeadTail:
    def test_short_text_is_unchanged(self):
        assert head_tail("hello", limit=100) == "hello"

    def test_text_exactly_at_the_limit_is_unchanged(self):
        assert head_tail("x" * 100, limit=100) == "x" * 100

    def test_long_text_keeps_its_start_and_end_with_a_note(self):
        text = "START" + "m" * 50_000 + "END"

        result = head_tail(text, limit=1_000)

        assert result.startswith("START")
        assert result.endswith("END")
        assert "characters omitted" in result
        assert len(result) <= 1_000

    def test_the_note_counts_what_was_left_out(self):
        result = head_tail("z" * 5_000, limit=1_000)

        note = result.replace("z", "")  # what is left is the note
        omitted = int(note.split("[... ")[1].split(" characters")[0].replace(",", ""))
        assert omitted == 5_000 - result.count("z")

    def test_the_default_limit_is_the_tool_budget(self):
        assert len(head_tail("z" * 200_000)) <= MAX_TEXT_CHARS
