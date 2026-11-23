import pytest

from ergane.builtin._edit import EditError, edit_text


def edit(raw: str, old: str, new: str, replace_all: bool = False) -> tuple[str, int]:
    return edit_text(raw, old, new, replace_all, "f.py")


class TestExactMatches:
    def test_the_text_is_replaced_where_it_is_found_once(self):
        assert edit("a = 1\nb = 2\n", "b = 2", "b = 3") == ("a = 1\nb = 3\n", 1)

    def test_a_deletion_removes_the_text(self):
        assert edit("a\nb\nc\n", "b\n", "") == ("a\nc\n", 1)

    def test_several_places_are_an_error_that_names_the_lines(self):
        with pytest.raises(EditError) as raised:
            edit("x\ny\nx\n", "x", "z")

        message = str(raised.value)
        assert "matches 2 places (lines 1, 3)" in message
        assert "replace_all" in message and "Nothing was changed" in message

    def test_only_the_first_ten_lines_are_named(self):
        with pytest.raises(
            EditError,
            match=r"matches 12 places \(lines 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, \.\.\.\)",
        ):
            edit("x\n" * 12, "x", "y")

    def test_replace_all_replaces_every_place_and_counts_them(self):
        assert edit("x\ny\nx\n", "x", "z", replace_all=True) == ("z\ny\nz\n", 2)

    def test_a_match_that_is_one_place_needs_no_replace_all(self):
        assert edit("a\nb\n", "a", "c", replace_all=True) == ("c\nb\n", 1)

    def test_matches_do_not_overlap(self):
        assert edit("aaaa", "aa", "b", replace_all=True) == ("bb", 2)

    def test_the_replacement_is_taken_literally(self):
        assert edit("x = 1", "1", r"\1 \g<0> $1")[0] == r"x = \1 \g<0> $1"


class TestNotFound:
    def test_the_error_says_where_and_how_to_fix_it(self):
        with pytest.raises(EditError) as raised:
            edit("one\ntwo\n", "three", "four")

        assert "Could not find old_text in f.py" in str(raised.value)
        assert "including whitespace and indentation" in str(raised.value)

    def test_the_closest_lines_of_the_file_are_shown_with_their_numbers(self):
        raw = "import os\n\ndef run_the_thing(a, b):\n    return a + b\n"

        with pytest.raises(EditError) as raised:
            edit(raw, "def run_the_thng(a, b):\n    return a - b", "x")

        message = str(raised.value)
        assert "The closest lines in the file:" in message
        assert "3: def run_the_thing(a, b):" in message
        assert "4:     return a + b" in message

    def test_of_equally_close_lines_the_first_is_shown(self):
        # "b = 3" is as far from "b = 2" as from "c = 3"; the earlier line is shown.
        with pytest.raises(EditError) as raised:
            edit("a = 1\nb = 2\nc = 3\n", "b = 3", "x")

        assert "closest lines in the file:\n2: b = 2" in str(raised.value)

    def test_the_empty_piece_after_the_last_newline_is_not_shown_as_a_line(self):
        with pytest.raises(EditError) as raised:
            edit("one\nlast line\n", "last lines", "x")

        assert str(raised.value).endswith("2: last line")

    def test_nothing_is_shown_when_nothing_is_close(self):
        with pytest.raises(EditError) as raised:
            edit("alpha\nbeta\n", "zzzzzzzzzz", "x")

        assert "closest lines" not in str(raised.value)

    def test_a_different_indentation_is_not_a_match(self):
        with pytest.raises(EditError, match="Could not find"):
            edit("def f():\n    return 1\n", "def f():\nreturn 1", "x")


class TestNothingToDo:
    def test_an_empty_old_text_is_refused(self):
        with pytest.raises(EditError, match="must not be empty.*write_file"):
            edit("abc", "", "x")

    def test_the_same_text_twice_is_refused(self):
        with pytest.raises(EditError, match="No change"):
            edit("abc", "b", "b")

    def test_an_edit_that_leaves_the_file_as_it_was_is_refused(self):
        # Found only once the lines are made plain, and replaced by the same line.
        with pytest.raises(EditError, match="leaves f.py as it was"):
            edit("x  \ny\n", "x\n", "x")


class TestLineEndings:
    def test_a_crlf_file_stays_crlf(self):
        assert edit("a\r\nb\r\nc\r\n", "b", "B") == ("a\r\nB\r\nc\r\n", 1)

    def test_old_text_the_model_wrote_with_crlf_still_matches(self):
        assert edit("a\nb\nc\n", "a\r\nb", "x\r\ny") == ("x\ny\nc\n", 1)

    def test_a_new_line_in_a_crlf_file_gets_crlf(self):
        assert edit("a\r\nb\r\n", "b", "b\nc")[0] == "a\r\nb\r\nc\r\n"

    def test_a_file_with_mostly_lf_stays_lf(self):
        assert edit("a\nb\nc\r\n", "b", "B")[0] == "a\nB\nc\n"

    def test_a_file_without_a_final_newline_gets_none(self):
        assert edit("a\nb", "b", "c")[0] == "a\nc"


class TestSecondTryAfterTheLinesAreMadePlain:
    def test_trailing_whitespace_in_the_file_is_ignored(self):
        # The unchanged first line is the file's own, trailing spaces and all.
        result = edit(
            "def f():  \n    pass\t\n", "def f():\n    pass", "def f():\n    return 1"
        )

        assert result == ("def f():  \n    return 1\n", 1)

    def test_typographic_quotes_in_the_file_match_straight_ones(self):
        raw = "say(“hello”)\nsay(‘x’)\n"

        assert edit(raw, 'say("hello")', 'say("bye")')[0] == 'say("bye")\nsay(‘x’)\n'

    def test_unicode_dashes_and_spaces_match_ascii_ones(self):
        raw = "range – 1 to 3\nnext\n"

        assert (
            edit(raw, "range - 1 to 3", "range - 1 to 4")[0] == "range - 1 to 4\nnext\n"
        )

    def test_unchanged_lines_at_the_ends_are_the_files_own(self):
        raw = "x = “a”\ny = 1\nz = “b”\n"
        old = 'x = "a"\ny = 1\nz = "b"'
        new = 'x = "a"\ny = 2\nz = "b"'

        assert edit(raw, old, new)[0] == "x = “a”\ny = 2\nz = “b”\n"

    def test_a_deletion_removes_the_lines_and_leaves_no_blank_one(self):
        assert edit("a  \nb  \nc\n", "a\nb", "") == ("c\n", 1)

    def test_several_places_are_still_an_error_without_replace_all(self):
        with pytest.raises(EditError, match="matches 2 places \\(lines 1, 3\\)"):
            edit("a \nx\na \nx\n", "a\nx", "z")

    def test_replace_all_works_on_this_try_too(self):
        result = edit("a \nx\nb\na \nx\n", "a\nx", "z", replace_all=True)

        assert result == ("z\nb\nz\n", 2)

    def test_the_exact_try_comes_first(self):
        # "x" is in the file exactly once, in the second line; the first line only
        # matches once its trailing space is ignored.
        assert edit("x \nx\n", "x\n", "z\n") == ("x \nz\n", 1)
