import io
import json
import sys

import pytest

from helpers import make_config, run_repl
from muagent.agent import Agent
from muagent.console import Console, sanitize
from muagent.errors import TurnError
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

        def fail(
            prompt: str, on_text=None, on_notice=None, confirm=None, on_call=None
        ) -> None:
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
    def asked(self, monkeypatch: pytest.MonkeyPatch):
        """Ask Console().confirm with scripted input.

        Returns a function that gives the decision, and fills `prompts` with what
        `input()` was asked: the question is the prompt, which is all of what the
        user sees of it.
        """
        prompts: list[str] = []
        monkeypatch.setenv("COLUMNS", "100")

        def ask(reply, arguments: str = '{"path": "notes.txt"}') -> bool | str:
            def fake_input(prompt: str) -> str:
                prompts.append(prompt)
                if isinstance(reply, BaseException):
                    raise reply
                return reply

            monkeypatch.setattr("builtins.input", fake_input)
            return Console().confirm(SAVE, arguments)

        ask.prompts = prompts  # pyright: ignore[reportFunctionMemberAccess]
        return ask

    @pytest.mark.parametrize("reply", ["y", " YES "])
    def test_yes_allows_it(self, asked, reply):
        assert asked(reply) is True

    @pytest.mark.parametrize("reply", ["a", " Always "])
    def test_a_allows_this_tool_for_the_session(self, asked, reply):
        assert asked(reply) == "always"

    @pytest.mark.parametrize("reply", ["", "n", "no", "yep", "ok"])
    def test_anything_else_denies_it(self, asked, reply):
        assert asked(reply) is False

    def test_end_of_input_denies_it(self, asked):
        assert asked(EOFError()) is False

    def test_ctrl_c_is_not_swallowed(self, asked):
        with pytest.raises(KeyboardInterrupt):
            asked(KeyboardInterrupt())

    def test_the_question_is_one_line_with_the_tool_and_the_full_arguments(self, asked):
        asked("n")

        assert asked.prompts == ['Allow save {"path": "notes.txt"}? [y/N/a] ']

    def test_nothing_else_is_printed(self, asked, capsys):
        asked("n")

        assert capsys.readouterr().out == ""

    def test_padding_between_tokens_cannot_push_the_call_off_the_screen(self, asked):
        asked("n", '{"path": "rm -rf ~"' + "\n" * 200 + "}")

        assert asked.prompts == ['Allow save {"path": "rm -rf ~"}? [y/N/a] ']

    def test_arguments_that_are_not_json_are_shown_on_the_same_line(self, asked):
        asked("n", "not json\n\n\n  at all")

        assert asked.prompts == ["Allow save not json at all? [y/N/a] "]

    def test_a_newline_inside_a_value_stays_on_the_line(self, asked):
        asked("n", '{"text": "a\\nb"}')

        assert asked.prompts == ['Allow save {"text": "a\\nb"}? [y/N/a] ']

    def test_the_question_is_sanitized(self, asked):
        asked("n", '{"path": "a\x1b]52;c;x"}')

        (prompt,) = asked.prompts
        assert "\x1b" not in prompt
        assert "\n" not in prompt

    def test_arguments_that_fit_are_shown_whole_and_the_prompt_does_not_offer_more(
        self, asked
    ):
        asked("n", json.dumps({"path": "a" * 40}))

        (prompt,) = asked.prompts
        assert "a" * 40 in prompt
        assert prompt.endswith("? [y/N/a] ")

    def test_a_long_value_is_cut_in_the_middle_to_fit_the_terminal(self, asked):
        asked("n", json.dumps({"path": "a.py", "content": "x" * 5000}))

        (prompt,) = asked.prompts
        assert len(prompt) <= 100
        assert '"path": "a.py"' in prompt  # a short value is left alone
        assert "…" in prompt
        assert prompt.endswith("? [y/N/a/?] ")  # and the prompt says there is more

    def test_both_ends_of_a_long_command_are_kept(self, asked):
        command = "cd /tmp && " + "a" * 500 + "; rm -rf ~"

        asked("n", json.dumps({"command": command}))

        (prompt,) = asked.prompts
        assert "cd /tmp" in prompt
        assert "rm -rf ~" in prompt
        assert len(prompt) <= 100

    def test_a_line_that_is_too_long_even_so_is_cut_at_its_end(self, asked):
        asked("n", json.dumps({f"key{i}": i for i in range(60)}))

        (prompt,) = asked.prompts
        assert len(prompt) <= 100
        assert "…? [y/N/a/?] " in prompt

    def test_arguments_that_are_not_json_are_cut_at_their_end(self, asked):
        asked("n", "word " * 100)

        (prompt,) = asked.prompts
        assert len(prompt) <= 100
        assert "…? [y/N/a/?] " in prompt

    def test_a_narrow_terminal_still_gets_a_line_with_some_arguments(
        self, asked, monkeypatch
    ):
        monkeypatch.setenv("COLUMNS", "30")

        asked("n", json.dumps({"path": "a" * 100}))

        (prompt,) = asked.prompts
        assert len(prompt) < 60  # not the whole value, and not a line of nothing
        assert '{"path"' in prompt

    def test_a_question_mark_shows_the_arguments_in_full_and_asks_again(
        self, monkeypatch, capsys
    ):
        monkeypatch.setenv("COLUMNS", "100")
        answers = iter(["?", "y"])
        prompts: list[str] = []

        def fake_input(prompt: str) -> str:
            prompts.append(prompt)
            return next(answers)

        monkeypatch.setattr("builtins.input", fake_input)
        arguments = json.dumps({"content": "x" * 300 + "\x1b[2J"})

        assert Console().confirm(SAVE, arguments) is True

        assert len(prompts) == 2 and prompts[0] == prompts[1]
        assert prompts[0].endswith("? [y/N/a/?] ")
        out = capsys.readouterr().out
        assert "x" * 300 in out  # all of it
        assert '{\n  "content"' in out
        assert "\x1b" not in out  # sanitized like everything else

    def test_a_question_mark_works_whatever_was_shown(self, monkeypatch, capsys):
        answers = iter(["?", "n"])
        monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

        assert Console().confirm(SAVE, '{"path": "a"}') is False

        assert '"path": "a"' in capsys.readouterr().out

    def test_it_starts_on_a_new_line_after_a_cut_off_reply(self, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")
        console = Console()
        console.write("partial")

        console.confirm(SAVE, "{}")

        assert capsys.readouterr().out == "partial\n"

    def test_a_reply_after_the_answer_starts_after_a_blank_line(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")
        console = Console()
        console.confirm(SAVE, "{}")

        console.write("Saved.")

        assert capsys.readouterr().out == "\nSaved."

    def test_ctrl_c_leaves_the_cursor_after_the_question(self, monkeypatch, capsys):
        def interrupted(_prompt: str) -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", interrupted)
        console = Console()

        with pytest.raises(KeyboardInterrupt):
            console.confirm(SAVE, "{}")
        console.notice("Cancelled")

        # The notice starts a new line: the cursor was mid-line, after "^C".
        assert capsys.readouterr().out == "\n[Cancelled]\n"

    def test_end_of_input_ends_the_line_the_question_left_open(
        self, monkeypatch, capsys
    ):
        def closed(_prompt: str) -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", closed)
        console = Console()

        console.confirm(SAVE, "{}")
        console.notice("Closed")

        assert capsys.readouterr().out == "\n[Closed]\n"


class TestCall:
    def test_a_call_is_a_status_line_with_the_name_and_the_arguments(self, capsys):
        Console().call("read_file", '{"path":"notes.txt"}')

        assert capsys.readouterr().out == '[read_file({"path": "notes.txt"})]\n'

    def test_the_line_follows_the_width_of_the_terminal(self, monkeypatch, capsys):
        arguments = json.dumps({"text": "x" * 300})
        monkeypatch.setenv("COLUMNS", "60")
        Console().call("read_file", arguments)
        narrow = capsys.readouterr().out
        monkeypatch.setenv("COLUMNS", "200")
        Console().call("read_file", arguments)
        wide = capsys.readouterr().out

        assert len(narrow.rstrip("\n")) < 60 and "…" in narrow
        assert 60 < len(wide.rstrip("\n")) < 200 and "…" in wide

    def test_a_call_that_fits_is_shown_whole(self, monkeypatch, capsys):
        monkeypatch.setenv("COLUMNS", "200")
        arguments = json.dumps({"path": "a" * 100})

        Console().call("read_file", arguments)

        assert "…" not in capsys.readouterr().out


class TestDim:
    @pytest.fixture
    def console(self):
        console = Console()
        console.color = True
        return console

    def test_a_notice_is_dimmed_where_there_is_colour(self, console, capsys):
        console.notice("Cancelled")

        assert capsys.readouterr().out == "\033[2m[Cancelled]\033[0m\n"

    def test_the_fresh_line_stays_outside_the_dimmed_part(self, console, capsys):
        console.write("partial")

        console.notice("Cancelled")

        assert capsys.readouterr().out == "partial\n\033[2m[Cancelled]\033[0m\n"

    def test_a_reply_is_not_dimmed(self, console, capsys):
        console.write("hello\n")

        assert capsys.readouterr().out == "hello\n"

    def test_text_from_the_model_cannot_add_codes_of_its_own(self, console, capsys):
        console.notice("a\033[31mb")

        assert capsys.readouterr().out.count("\033") == 2  # ours: dim and reset

    def test_without_a_terminal_there_are_no_codes(self, capsys):
        # Under pytest stdout is not a terminal.
        Console().notice("Cancelled")

        assert capsys.readouterr().out == "[Cancelled]\n"

    def test_no_color_turns_it_off(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        assert Console().color is True
        monkeypatch.setenv("NO_COLOR", "1")

        assert Console().color is False
