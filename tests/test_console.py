import sys

import pytest

from helpers import make_config, run_repl
from muagent.agent import Agent, TurnError
from muagent.console import Console, sanitize


class TestSanitize:
    def test_removes_escape_sequences_and_bidi_controls(self):
        # An OSC 52 clipboard write, a carriage return and a right-to-left override.
        text = "a\x1b]52;c;ZXZpbA==\x07b\rc\u202ed"

        assert sanitize(text) == "a]52;c;ZXZpbA==bcd"

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

        assert capsys.readouterr().out == "[one]\nreply\n[two]\n"

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
