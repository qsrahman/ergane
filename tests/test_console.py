import io
import json
import sys

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text
from prompt_toolkit.output.vt100 import Vt100_Output

from helpers import make_config, run_repl
from muagent.agent import Agent
from muagent.console import Console, _room, sanitize  # pyright: ignore[reportPrivateUsage]
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


class TestConfirmMenu:
    """At a terminal the question is a menu: `Repl` supplies `choose`, which gives the
    key of the option chosen (the real one is tested with the REPL)."""

    @staticmethod
    def ask(answers, arguments: str = '{"path": "notes.txt"}', console=None):
        """Ask `confirm` with a menu that answers from `answers` (a key, or an
        exception to raise); gives the decision and what the menu was shown."""
        replies = iter(
            [answers] if isinstance(answers, (str, BaseException)) else answers
        )
        shown: list[tuple[str, list[tuple[str, str]]]] = []

        def choose(question, options):
            shown.append(
                (fragment_list_to_text(to_formatted_text(question)), list(options))
            )
            reply = next(replies)
            if isinstance(reply, BaseException):
                raise reply
            return reply

        console = console or Console()
        console.choose = choose
        return console.confirm(SAVE, arguments), shown

    @pytest.mark.parametrize(
        ("key", "decision"), [("y", True), ("a", "always"), ("n", False)]
    )
    def test_the_key_chosen_is_the_decision(self, key, decision):
        assert self.ask(key)[0] == decision

    def test_deny_is_first_so_that_it_is_the_default(self):
        _, [(question, options)] = self.ask("n")

        assert question == 'Allow save {"path": "notes.txt"}?'
        assert options == [
            ("n", "Deny"),
            ("y", "Allow once"),
            ("a", "Always allow save"),
        ]

    def test_arguments_that_are_cut_can_be_shown_in_full(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "100")
        arguments = json.dumps({"content": "x" * 300})

        _, [(question, options)] = self.ask("n", arguments)

        assert "…" in question and len(question) < 100
        assert options[-1] == ("?", "Show full arguments")

    def test_arguments_that_fit_offer_nothing_more(self):
        _, [(_, options)] = self.ask("n")

        assert [key for key, _ in options] == ["n", "y", "a"]

    def test_a_question_mark_prints_the_arguments_in_full_and_asks_again(self, capsys):
        arguments = json.dumps({"content": "x" * 300 + "\x1b[2J"})

        decision, shown = self.ask(["?", "y"], arguments)

        assert decision is True and len(shown) == 2 and shown[0] == shown[1]
        out = capsys.readouterr().out
        assert "x" * 300 in out and '{\n  "content"' in out
        assert "\x1b" not in out

    @pytest.mark.parametrize(
        ("key", "line"),
        [
            ("y", "allowed"),
            ("a", "always allowed"),
            ("n", "denied"),
        ],
    )
    def test_a_line_of_what_was_decided_is_left_in_place_of_the_menu(
        self, capsys, key, line
    ):
        self.ask(key)

        assert capsys.readouterr().out == f'▸ save  {{"path": "notes.txt"}}  {line}\n'

    def test_the_decision_is_in_colour_where_there_is_colour(self):
        written = io.StringIO()
        console = Console(output=Vt100_Output(written, lambda: Size(24, 80)))

        self.ask("n", console=console)

        assert f"{BOLD}▸ save" in written.getvalue()
        assert "\x1b[0;31m  denied" in written.getvalue()  # red, and not dim

    @pytest.mark.parametrize(
        "arguments", ['{"path": "a\x1b]52;c;x"}', "not json\x1b[2J"]
    )
    def test_the_question_is_sanitized(self, arguments):
        _, [(question, _)] = self.ask("n", arguments)

        assert "\x1b" not in question and "\n" not in question

    def test_it_starts_on_a_new_line_after_a_cut_off_reply(self, capsys):
        console = Console()
        console.write("partial")

        self.ask("y", console=console)

        assert capsys.readouterr().out.startswith("partial\n▸ save")

    def test_a_reply_after_the_decision_starts_after_a_blank_line(self, capsys):
        console = Console()
        self.ask("y", console=console)
        capsys.readouterr()

        console.write("Saved.")

        assert capsys.readouterr().out == "\nSaved."

    def test_ctrl_c_is_not_swallowed_and_nothing_is_printed(self, capsys):
        with pytest.raises(KeyboardInterrupt):
            self.ask(KeyboardInterrupt())

        assert capsys.readouterr().out == ""


class TestCall:
    def test_a_call_is_a_status_line_with_the_name_and_the_arguments(self, capsys):
        Console().call("read_file", '{"path":"notes.txt"}')

        assert capsys.readouterr().out == '▸ read_file  {"path": "notes.txt"}\n'

    def test_a_name_with_a_line_break_stays_on_the_line(self, capsys):
        Console().call("read\nfile\x1b[2J", "{}")

        assert capsys.readouterr().out == "▸ read file[2J  {}\n"

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


class TestRoom:
    def test_a_line_never_fills_the_terminal_and_has_room_for_twenty(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "80")

        assert _room(10) == 69
        assert _room(200) == 20


DIM = "\x1b[0;2m"  # what prompt_toolkit writes for dim text
BOLD = "\x1b[0;1m"


class TestStyle:
    """Where there is colour, status lines are written by prompt_toolkit to `output`
    (a terminal's would be a `Vt100_Output`); replies and everything else are printed."""

    @pytest.fixture
    def written(self):
        return io.StringIO()

    @pytest.fixture
    def console(self, written):
        output = Vt100_Output(written, lambda: Size(24, 80), term="xterm-256color")
        return Console(output=output)

    def test_a_notice_is_dimmed_where_there_is_colour(self, console, written, capsys):
        console.notice("Cancelled")

        assert f"{DIM}[Cancelled]" in written.getvalue()
        assert capsys.readouterr().out == ""

    def test_a_call_has_its_name_in_bold_and_its_arguments_dim(self, console, written):
        console.call("read_file", '{"path": "notes.txt"}')

        assert f'{BOLD}▸ read_file{DIM}  {{"path": "notes.txt"}}' in written.getvalue()

    def test_the_fresh_line_comes_before_the_styled_part(
        self, console, written, capsys
    ):
        console.write("partial")

        console.notice("Cancelled")

        assert capsys.readouterr().out == "partial"
        styled = written.getvalue()
        assert styled.index("\n") < styled.index(f"{DIM}[Cancelled]")

    def test_a_reply_is_not_styled(self, console, written, capsys):
        console.write("hello\n")

        assert capsys.readouterr().out == "hello\n"
        assert written.getvalue() == ""

    def test_text_from_the_model_cannot_add_codes_of_its_own(self, console, written):
        console.notice("a\033[31mb")

        assert "\x1b[31m" not in written.getvalue()
        assert "a[31mb" in written.getvalue()

    def test_without_a_terminal_there_are_no_codes_and_no_carriage_returns(
        self, capsys
    ):
        # Under pytest stdout is not a terminal.
        Console().notice("Cancelled")

        assert capsys.readouterr().out == "[Cancelled]\n"

    def test_no_color_turns_it_off(self, monkeypatch, written):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        assert Console().color is True
        monkeypatch.setenv("NO_COLOR", "1")

        assert Console().color is False
        assert (
            Console(output=Vt100_Output(written, lambda: Size(24, 80))).color is False
        )


class TestColumns:
    def test_the_texts_are_lined_up_under_the_longest_label(self, capsys):
        Console().columns([("a", "one"), ("long", "two")])

        assert capsys.readouterr().out == "  a     one\n  long  two\n"

    def test_it_is_plain_text_that_the_next_notice_stands_apart_from(self, capsys):
        console = Console()

        console.columns([("a", "one")])
        console.notice("done")

        assert capsys.readouterr().out == "  a  one\n[done]\n"

    def test_a_long_text_wraps_under_its_own_column(self, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "40")
        names = ", ".join(f"server__tool_{n}" for n in range(6))

        Console().columns([("ask first", names), ("b", "short")])

        out = capsys.readouterr().out.splitlines()
        assert len(out) > 3 and all(len(line) < 40 for line in out)
        assert out[0].startswith("  ask first  server__tool_0,")
        assert out[1].startswith(" " * 13 + "server__tool_")
        assert out[-1] == "  b          short"

    def test_a_name_with_a_hyphen_is_not_broken(self, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "40")

        Console().columns([("x", "aaaa, context7__resolve-library-id, bbbb")])

        assert "context7__resolve-library-id," in capsys.readouterr().out
