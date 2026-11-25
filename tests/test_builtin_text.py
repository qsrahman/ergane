import io

import pytest

from ergane.builtin._text import (
    MAX_LINE_CHARS,
    MAX_TEXT_CHARS,
    count_lines,
    head_tail,
    read_lines,
    take_lines,
)
from ergane.tools import MAX_OUTPUT_CHARS


def test_a_footer_fits_under_the_generic_cut():
    # The tool's own footer must never be what the 20 KiB cut removes.
    assert MAX_TEXT_CHARS + 1_000 < MAX_OUTPUT_CHARS


LINES = [f"line {i}" for i in range(1, 11)]


class TestTakeLines:
    def test_all_lines_when_they_fit(self):
        text, end = take_lines(LINES)

        assert text == "\n".join(LINES)
        assert end == len(LINES)

    def test_at_most_count_lines(self):
        text, end = take_lines(LINES, count=3)

        assert text == "line 1\nline 2\nline 3"
        assert end == 3

    def test_whole_lines_only_when_the_budget_runs_out(self):
        # Each line costs its length and a newline.
        line = "a" * (MAX_TEXT_CHARS // 2 - 1)
        text, end = take_lines([line] * 4)

        assert text == f"{line}\n{line}"
        assert end == 2

    def test_at_least_one_line_even_over_the_budget(self):
        text, end = take_lines(["a" * (MAX_TEXT_CHARS + 50), "b"])

        assert text == "a" * (MAX_TEXT_CHARS + 50)
        assert end == 1

    def test_no_lines_take_nothing(self):
        assert take_lines([]) == ("", 0)

    def test_the_default_budget_leaves_room_for_a_footer(self):
        text, _ = take_lines(["y" * 100] * 1_000)

        assert len(text) <= MAX_TEXT_CHARS


class Spy(io.StringIO):
    """A text file that records how much each read asked for."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.asked: list[int] = []

    def readline(self, size: int = -1, /) -> str:
        self.asked.append(size)
        return super().readline(size)


class TestReadLines:
    def test_lines_come_back_without_their_newlines(self):
        assert read_lines(io.StringIO("a\nb\nc\n"), 0, 10) == ["a", "b", "c"]

    def test_start_and_count_pick_the_range(self):
        assert read_lines(io.StringIO("a\nb\nc\nd\n"), 1, 2) == ["b", "c"]

    def test_reading_stops_once_the_lines_are_more_than_can_be_shown(self):
        handle = io.StringIO("".join("x" * 100 + "\n" for _ in range(1_000)))

        lines = read_lines(handle, 0, 1_000, max_chars=1_000)

        # The line that went over is included, so that `take_lines` can tell there is
        # more; none after it is read.
        assert len(lines) == 10  # 101 characters each: the tenth is the first over
        assert handle.readline() == "x" * 100 + "\n"

    def test_a_last_line_without_a_newline_is_kept(self):
        assert read_lines(io.StringIO("a\nb"), 0, 10) == ["a", "b"]

    def test_a_start_past_the_end_gives_nothing(self):
        assert read_lines(io.StringIO("a\nb\n"), 5, 10) == []

    @pytest.mark.parametrize("extra", [500, 200_000])
    def test_a_very_long_line_is_cut_and_says_how_long_it_was(self, extra):
        length = MAX_LINE_CHARS + extra
        lines = read_lines(io.StringIO("x" * length + "\nnext\n"), 0, 10)

        assert lines[0].startswith("x" * MAX_LINE_CHARS + "...")
        assert f"[line cut: {length:,} characters]" in lines[0]
        assert lines[1:] == ["next"]  # the rest of the long line was dropped

    def test_a_line_of_exactly_the_limit_is_not_cut(self):
        assert read_lines(io.StringIO("y" * MAX_LINE_CHARS + "\n"), 0, 1) == [
            "y" * MAX_LINE_CHARS
        ]

    def test_one_character_over_the_limit_is_cut(self):
        line = read_lines(io.StringIO("y" * (MAX_LINE_CHARS + 1) + "\n"), 0, 1)[0]

        assert f"[line cut: {MAX_LINE_CHARS + 1:,} characters]" in line

    def test_skipped_lines_may_be_enormous(self):
        assert read_lines(io.StringIO("z" * 1_000_000 + "\nafter\n"), 1, 1) == ["after"]

    def test_no_read_asks_for_more_than_a_chunk(self):
        text = "w" * 1_000_000 + "\nb\n" + "v" * 1_000_000
        handle = Spy(text)
        read_lines(handle, 0, 10)
        skipping = Spy(text)
        read_lines(skipping, 1, 10)
        counting = Spy(text)
        count_lines(counting)

        for spy in (handle, skipping, counting):
            # -1 is "the whole line": the read that must never be made.
            assert all(0 < size <= 65_536 for size in spy.asked), spy.asked[:3]

    def test_count_lines_counts_what_is_left(self):
        handle = io.StringIO("a\nb\nc")
        handle.readline()

        assert count_lines(handle) == 2


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
