import io
import sys

import pytest

from helpers import make_config, run_repl
from muagent.agent import Agent, TurnError
from muagent.console import Console, sanitize
from muagent.tools import Tool


class TestSanitize:
    def test_removes_escape_sequences_and_bidi_controls(self):
        # An OSC 52 clipboard write, a carriage return and a right-to-left override.
        text = "a\x1b]52;c;ZXZpbA==\x07b\rc\u202ed"

        assert sanitize(text) == "a]52;c;ZXZpbA==bcd"

    @pytest.mark.parametrize(
        "hidden",
        ["\u200e", "\u200f", "\u061c", "\U000e0041", "\U000e007f"],
        ids=["lrm", "rlm", "alm", "tag-a", "tag-cancel"],
    )
    def test_removes_bidi_marks_and_invisible_tag_characters(self, hidden):
        assert sanitize(f"a{hidden}b") == "ab"

    def test_keeps_the_joiners_that_persian_urdu_and_emoji_need(self):
        text = "می\u200cخواهم 👨\u200d👩\u200d👧"

        assert sanitize(text) == text

    def test_keeps_newlines_tabs_and_ordinary_unicode(self):
        text = "line 1\n\tπ ≈ 3.14 字\n"

        assert sanitize(text) == text

    def test_notices_are_sanitized(self, monkeypatch, capsys):
        agent = Agent(make_config())

        def fail(prompt: str, on_text=None, on_notice=None, confirm=None) -> None:
            raise TurnError("Response failed: \x1b[2Jboom")

        monkeypatch.setattr(agent, "run_turn", fail)
        run_repl(monkeypatch, agent, ["hi", "/exit"])

        out = capsys.readouterr().out
        assert "[Response failed: [2Jboom]" in out
        assert "\x1b" not in out


class TestConsole:
    def test_write_prints_sanitized_text_without_a_newline(self, capsys):
        Console().write("a\x1b[2Jb")

        assert capsys.readouterr().out == "a[2Jb"

    def test_text_the_terminal_cannot_encode_is_replaced_not_an_error(
        self, monkeypatch
    ):
        raw = io.BytesIO()
        monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="ascii"))

        Console().notice("→ ok")
        sys.stdout.flush()

        assert raw.getvalue() == b"[? ok]\n"

    def test_a_reply_after_a_status_line_starts_after_a_blank_line(self, capsys):
        console = Console()
        console.notice("read_file({})")

        console.write("Hello")
        console.write(" world")

        assert capsys.readouterr().out == "[read_file({})]\n\nHello world"

    def test_a_reply_after_a_blank_line_gets_no_second_one(self, capsys):
        console = Console()
        console.notice("read_file({})")
        console.blank()

        console.write("Hello")

        assert capsys.readouterr().out == "[read_file({})]\n\nHello"

    def test_a_reply_that_starts_with_its_own_newlines_gets_no_extra_blank(
        self, capsys
    ):
        console = Console()
        console.notice("read_file({})")

        console.write("\n")
        console.write("Hello")

        assert capsys.readouterr().out == "[read_file({})]\n\nHello"

    def test_text_before_a_status_line_and_the_reply_after_it_are_separate(
        self, capsys
    ):
        console = Console()
        console.write("Checking.")
        console.notice("read_file({})")

        console.write("Done.")

        assert capsys.readouterr().out == "Checking.\n[read_file({})]\n\nDone."

    def test_an_empty_piece_does_not_use_up_the_blank_line(self, capsys):
        console = Console()
        console.notice("read_file({})")

        console.write("")
        console.write("Hello")

        assert capsys.readouterr().out == "[read_file({})]\n\nHello"

    def test_a_reply_after_a_question_starts_after_a_blank_line(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")
        console = Console()
        console.confirm(SAVE, "{}")
        capsys.readouterr()

        console.write("Saved.")

        assert capsys.readouterr().out == "\nSaved."

    def test_tracks_whether_the_cursor_is_at_a_line_start(self):
        console = Console()
        assert console.at_line_start

        console.write("partial")
        assert not console.at_line_start

        console.write("done\n")
        assert console.at_line_start

    def test_empty_text_leaves_the_line_state_alone(self):
        console = Console()
        console.write("partial")

        console.write("")

        assert not console.at_line_start

    def test_notice_starts_a_new_line_only_when_needed(self, capsys):
        console = Console()

        console.notice("one")
        console.write("reply")
        console.notice("two")

        assert capsys.readouterr().out == "[one]\n\nreply\n[two]\n"

    def test_notice_text_is_sanitized(self, capsys):
        Console().notice("failed: \x1b[2Jboom")

        assert capsys.readouterr().out == "[failed: [2Jboom]\n"

    def test_notice_text_is_flattened_to_one_line(self, capsys):
        Console().notice("line one\nline two\n\tindented")

        assert capsys.readouterr().out == "[line one line two indented]\n"

    def test_end_line_and_blank(self, capsys):
        console = Console()
        console.end_line()  # nothing to finish
        console.write("partial")
        console.end_line()
        console.blank()

        assert capsys.readouterr().out == "partial\n\n"

    @pytest.mark.parametrize("is_terminal", [True, False])
    def test_clear_prints_the_codes_only_on_a_terminal(
        self, capsys, monkeypatch, is_terminal
    ):
        # capsys replaces stdout with a capture object, which is never a terminal.
        monkeypatch.setattr(sys.stdout, "isatty", lambda: is_terminal)

        Console().clear()

        printed_codes = "\033[2J\033[3J\033[H" in capsys.readouterr().out
        assert printed_codes is is_terminal

    def test_clear_puts_the_cursor_at_a_line_start(self, capsys, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        console = Console()
        console.write("partial")

        console.clear()

        assert console.at_line_start


def save(path: str) -> str:
    """Save a file."""
    return path


SAVE = Tool.from_function(save, read_only=False)


class TestConfirm:
    @pytest.fixture
    def answer(self, monkeypatch: pytest.MonkeyPatch):
        """Ask Console().confirm with scripted input; returns its decision."""

        def ask(reply, arguments: str = '{"path": "notes.txt"}') -> bool:
            def fake_input(_prompt: str) -> str:
                if isinstance(reply, BaseException):
                    raise reply
                return reply

            monkeypatch.setattr("builtins.input", fake_input)
            return Console().confirm(SAVE, arguments)

        return ask

    @pytest.mark.parametrize("reply", ["y", " YES "])
    def test_yes_allows_it(self, answer, reply):
        assert answer(reply) is True

    @pytest.mark.parametrize("reply", ["", "n", "no", "yep", "ok"])
    def test_anything_else_denies_it(self, answer, reply):
        assert answer(reply) is False

    def test_end_of_input_denies_it(self, answer):
        assert answer(EOFError()) is False

    def test_ctrl_c_is_not_swallowed(self, answer):
        with pytest.raises(KeyboardInterrupt):
            answer(KeyboardInterrupt())

    def test_the_question_shows_the_full_arguments(self, answer, capsys):
        answer("n")

        out = capsys.readouterr().out
        assert 'Allow save with these arguments?\n{\n  "path": "notes.txt"\n}\n' in out

    def test_padding_between_tokens_cannot_push_the_call_off_the_screen(
        self, answer, capsys
    ):
        answer("n", '{"path": "rm -rf ~"' + "\n" * 200 + "}")

        out = capsys.readouterr().out
        assert out == 'Allow save with these arguments?\n{\n  "path": "rm -rf ~"\n}\n'

    def test_arguments_that_are_not_json_are_shown_on_one_line(self, answer, capsys):
        answer("n", "not json\n\n\n  at all")

        assert "arguments?\nnot json at all\n" in capsys.readouterr().out

    def test_long_arguments_are_not_shortened_and_are_sanitized(self, answer, capsys):
        answer("n", '{"path": "' + "a" * 300 + '\x1b]52;c;x"}')

        out = capsys.readouterr().out
        assert "a" * 300 in out
        assert "\x1b" not in out

    def test_it_starts_on_a_new_line_after_a_cut_off_reply(self, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")
        console = Console()
        console.write("partial")

        console.confirm(SAVE, "{}")

        assert capsys.readouterr().out.startswith("partial\nAllow save")
