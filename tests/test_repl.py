import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from fake_api import REFUSAL, STREAM_ERROR, TEXT
from helpers import make_config, make_context, run_repl
from muagent.agent import Agent
from muagent.commands import COMMANDS
from muagent.repl import Repl, terminal_session
from muagent.tools import Tool


class TestRepl:
    @pytest.fixture
    def repl(self, monkeypatch: pytest.MonkeyPatch):
        """Run a Repl with scripted input; returns the prompts sent to the agent.

        A callable passed as `turn` is called with each prompt and the agent's
        context in place of Agent.run_turn.
        """

        def run(inputs: list, turn=None) -> list[str]:
            agent = Agent(make_config())
            prompts: list[str] = []

            def fake_run_turn(
                prompt: str, on_text=None, on_notice=None, confirm=None, on_call=None
            ) -> None:
                prompts.append(prompt)
                if turn:
                    turn(prompt, agent.context)

            monkeypatch.setattr(agent, "run_turn", fake_run_turn)
            run_repl(monkeypatch, agent, inputs)
            return prompts

        return run

    @pytest.mark.parametrize(
        "stop",
        [
            "/exit",
            "/quit",
            "  /EXIT  ",
            EOFError(),
            [KeyboardInterrupt(), KeyboardInterrupt()],
        ],
    )
    def test_stops_on_exit_command_or_eof(self, repl, capsys, stop):
        assert repl(stop if isinstance(stop, list) else [stop]) == []
        assert capsys.readouterr().out.endswith("Bye.\n")

    def test_end_of_input_quits_at_once_when_it_is_not_a_terminal(self, repl):
        def talk(prompt, context):
            context.items.append({"role": "user", "content": prompt})

        assert repl(["hi", EOFError()], turn=talk) == ["hi"]

    def test_one_ctrl_c_at_the_prompt_does_not_quit(self, repl, capsys):
        assert repl([KeyboardInterrupt(), "hi", "/exit"]) == ["hi"]
        assert "[Press Ctrl-C again, or type /exit, to quit]" in capsys.readouterr().out

    def test_typing_between_two_ctrl_c_starts_the_count_again(self, repl, capsys):
        inputs = [KeyboardInterrupt(), "hi", KeyboardInterrupt(), "there", "/exit"]

        assert repl(inputs) == ["hi", "there"]
        assert capsys.readouterr().out.count("Press Ctrl-C again") == 2

    def test_strips_input_and_skips_blank_lines(self, repl):
        assert repl(["", "   ", "  hi  ", "/exit"]) == ["hi"]

    def test_ctrl_c_cancels_only_the_current_turn(self, repl, capsys):
        def turn(prompt: str, _context) -> None:
            if prompt == "slow":
                raise KeyboardInterrupt

        assert repl(["slow", "next", "/exit"], turn) == ["slow", "next"]
        assert "[Cancelled]" in capsys.readouterr().out

    def test_reply_ending_in_a_newline_gets_no_extra_blank_line(
        self, monkeypatch, capsys
    ):
        agent = Agent(make_config())

        def turn(prompt: str, on_text, on_notice, confirm, on_call) -> None:
            on_text("line\n")

        monkeypatch.setattr(agent, "run_turn", turn)
        run_repl(monkeypatch, agent, ["hi", "/exit"])

        assert "\nline\n\nBye." in capsys.readouterr().out

    def test_ctrl_c_mid_reply_puts_the_notice_on_its_own_line(
        self, monkeypatch, capsys
    ):
        agent = Agent(make_config())

        def turn(prompt: str, on_text, on_notice, confirm, on_call) -> None:
            on_text("partial")
            raise KeyboardInterrupt

        monkeypatch.setattr(agent, "run_turn", turn)
        run_repl(monkeypatch, agent, ["hi", "/exit"])

        assert "partial\n[Cancelled]\n" in capsys.readouterr().out

    def test_turn_errors_are_reported_and_loop_continues(self, repl, capsys):
        def turn(prompt: str, _context) -> None:
            if prompt == "bad":
                raise ValueError("boom")

        assert repl(["bad", "next", "/exit"], turn) == ["bad", "next"]
        assert "Agent error: ValueError: boom" in capsys.readouterr().out

    def test_unknown_command_is_not_sent_to_the_agent(self, repl, capsys):
        assert repl(["/rest", "/exit"]) == []
        assert "[Unknown command: /rest. Type /help" in capsys.readouterr().out

    def test_slash_prompt_with_spaces_goes_to_the_agent(self, repl):
        prompt = "/usr/bin is missing"

        assert repl([prompt, "/exit"]) == [prompt]


def read_file(path: str) -> str:
    """Read a text file (the tool the fake API calls)."""
    return "contents"


@pytest.mark.parametrize("streaming", [True, False])
class TestReplRendering:
    """The Repl against a real Agent and the fake server."""

    def test_prints_the_reply(self, monkeypatch, make_agent, capsys, streaming):
        run_repl(monkeypatch, make_agent(STREAMING=streaming), ["hi", "/exit"])

        assert f"\n{TEXT}\n\nBye." in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("prompt", "notice"),
        [
            ("incomplete", "[Response incomplete: max_output_tokens]"),
            ("failed", "[Response failed: server_error: boom]"),
            ("empty", "[No text in response]"),
            ("refusal", f"[Refused: {REFUSAL}]"),
        ],
    )
    def test_reports_unsuccessful_responses_on_their_own_line(
        self, monkeypatch, make_agent, capsys, streaming, prompt, notice
    ):
        run_repl(monkeypatch, make_agent(STREAMING=streaming), [prompt, "/exit"])

        assert f"\n{notice}\n" in capsys.readouterr().out

    @staticmethod
    def terminal(monkeypatch, answers):
        """Stand in for input() the way a terminal looks: the prompt is shown, then
        what was typed and the Enter that ends the line."""
        typed = iter(answers)

        def fake_input(prompt: str) -> str:
            line = next(typed)
            print(f"{prompt}{line}")
            return line

        monkeypatch.setattr("builtins.input", fake_input)

    def run(self, agent):
        Repl(agent, COMMANDS).run()

    def test_a_call_and_the_reply_are_each_set_apart_by_a_blank_line(
        self, monkeypatch, make_agent, capsys, streaming
    ):
        agent = make_agent(tools=[Tool.from_function(read_file)], STREAMING=streaming)
        self.terminal(monkeypatch, ["tool_call", "/exit"])

        self.run(agent)

        out = capsys.readouterr().out
        assert out.endswith(
            f'> tool_call\n\n[read_file({{"path": "notes.txt"}})]\n\n{TEXT}\n\n> /exit\nBye.\n'
        )

    def test_a_call_that_asks_is_one_line_and_nothing_else(
        self, monkeypatch, make_agent, capsys, streaming
    ):
        tool = Tool.from_function(read_file, read_only=False)
        agent = make_agent(tools=[tool], STREAMING=streaming)
        self.terminal(monkeypatch, ["tool_call", "y", "/exit"])

        self.run(agent)

        out = capsys.readouterr().out
        assert out.endswith(
            '> tool_call\n\nAllow read_file {"path": "notes.txt"}? [y/N/a] y\n\n'
            f"{TEXT}\n\n> /exit\nBye.\n"
        )

    @pytest.mark.parametrize(
        ("tools", "banner"),
        [
            ([], "muagent | model: gpt-4o | /help lists commands"),
            (["one"], "muagent | model: gpt-4o | 1 tool | /help lists commands"),
            (
                ["one", "two"],
                "muagent | model: gpt-4o | 2 tools | /help lists commands",
            ),
        ],
    )
    def test_the_banner_is_one_line(
        self, monkeypatch, make_agent, capsys, streaming, tools, banner
    ):
        def tool_named(name: str) -> Tool:
            def function() -> str:
                """Do nothing."""
                return ""

            function.__name__ = name
            return Tool.from_function(function)

        agent = make_agent(tools=[tool_named(n) for n in tools], STREAMING=streaming)
        self.terminal(monkeypatch, ["/exit"])

        self.run(agent)

        assert capsys.readouterr().out.startswith(f"{banner}\n\n> /exit\n")

    def test_the_banner_names_the_effort_when_it_is_set(
        self, monkeypatch, make_agent, capsys, streaming
    ):
        agent = make_agent(STREAMING=streaming, REASONING_EFFORT="high")
        self.terminal(monkeypatch, ["/exit"])

        self.run(agent)

        assert capsys.readouterr().out.startswith(
            "muagent | model: gpt-4o | effort: high | /help lists commands\n\n"
        )

    def test_an_api_error_is_one_line_with_what_to_check(
        self, monkeypatch, make_agent, capsys, streaming
    ):
        agent = make_agent(STREAMING=streaming)
        self.terminal(monkeypatch, ["status_429", "/exit"])

        self.run(agent)

        assert (
            "[API error 429 (code_429): message 429 The account is out of quota, "
            "or rate limited.]"
        ) in capsys.readouterr().out

    def test_a_tool_call_is_shown_as_a_notice_before_the_reply(
        self, monkeypatch, make_agent, capsys, streaming
    ):
        def read_file(path: str) -> str:
            """Read a text file."""
            return "contents"

        agent = make_agent(tools=[Tool.from_function(read_file)], STREAMING=streaming)

        run_repl(monkeypatch, agent, ["tool_call", "/exit"])

        out = capsys.readouterr().out
        shown = '[read_file({"path": "notes.txt"})]'
        assert f"\n{shown}\n\n{TEXT}\n\nBye." in out

    def test_reports_trimmed_context(self, monkeypatch, make_agent, capsys, streaming):
        agent = make_agent(STREAMING=streaming, MAX_CONTEXT_TOKENS=1)
        agent.context.items = make_context(2)

        run_repl(monkeypatch, agent, ["hi", "/exit"])

        out = capsys.readouterr().out
        assert f"[Context trimmed: oldest turns dropped]\n\n{TEXT}\n" in out


def test_stream_error_notice_follows_the_partial_reply(monkeypatch, make_agent, capsys):
    run_repl(monkeypatch, make_agent(STREAMING=True), ["error_event", "/exit"])

    out = capsys.readouterr().out
    assert f"Hello\n[Stream error: server_error: {STREAM_ERROR}]\n" in out


class TestReplConfirm:
    """The Repl hands the console's question to the agent; see test_console."""

    @pytest.fixture
    def run(self, monkeypatch, make_agent):
        """Run "tool_call" against a tool that is not read-only, answering its
        question with `answer`; returns the paths the tool was run with."""

        def run(answer) -> list[str]:
            ran: list[str] = []

            def read_file(path: str) -> str:
                """Read a text file."""
                ran.append(path)
                return "contents"

            agent = make_agent(tools=[Tool.from_function(read_file, read_only=False)])
            run_repl(monkeypatch, agent, ["tool_call", answer, "/exit"])
            return ran

        return run

    @pytest.mark.parametrize("answer", ["y", " YES "])
    def test_yes_lets_the_tool_run(self, run, answer):
        assert run(answer) == ["notes.txt"]

    @pytest.mark.parametrize("answer", ["", "n", "no", "yep", "ok"])
    def test_anything_else_denies_it(self, run, answer):
        assert run(answer) == []

    def test_end_of_input_denies_it(self, run):
        assert run(EOFError()) == []

    def test_ctrl_c_at_the_question_cancels_the_turn(self, run, capsys):
        assert run(KeyboardInterrupt()) == []
        assert "[Cancelled]" in capsys.readouterr().out


class TestTerminalSession:
    """The line editor, driven through a pipe the way prompt_toolkit's docs test it."""

    @staticmethod
    def read(tmp_path, typed: str, conversation: bool = False) -> tuple[list[str], str]:
        """Run a Repl whose session reads `typed`; return the prompts and the history."""
        agent = Agent(make_config())
        if conversation:
            agent.context.items = make_context(1)
        prompts: list[str] = []
        agent.run_turn = lambda prompt, **_: prompts.append(prompt)  # pyright: ignore[reportAttributeAccessIssue]
        history = tmp_path / "state" / "history"
        with create_pipe_input() as pipe:
            pipe.send_text(typed)
            session = terminal_session(
                COMMANDS, history, input=pipe, output=DummyOutput()
            )
            Repl(agent, COMMANDS, session).run()
        return prompts, history.read_text() if history.exists() else ""

    def test_ctrl_c_clears_what_was_typed_and_does_not_quit(self, tmp_path, capsys):
        prompts, _ = self.read(tmp_path, "abc\x03def\r/exit\r")

        assert prompts == ["def"]
        assert "Press Ctrl-C again" not in capsys.readouterr().out

    def test_ctrl_d_quits_at_once_when_there_is_no_conversation(self, tmp_path, capsys):
        self.read(tmp_path, "\x04")

        out = capsys.readouterr().out
        assert "Press Ctrl-D again" not in out and out.endswith("Bye.\n")

    def test_ctrl_d_in_a_conversation_asks_for_a_second_press(self, tmp_path, capsys):
        prompts, _ = self.read(tmp_path, "\x04x\r/exit\r", conversation=True)

        assert prompts == ["x"]
        assert "[Press Ctrl-D again, or type /exit, to quit]" in capsys.readouterr().out

    def test_two_ctrl_d_in_a_row_quit(self, tmp_path, capsys):
        self.read(tmp_path, "\x04\x04", conversation=True)

        assert capsys.readouterr().out.endswith("Bye.\n")

    def test_ctrl_c_on_an_empty_line_asks_for_a_second_press(self, tmp_path, capsys):
        prompts, _ = self.read(tmp_path, "\x03x\r/exit\r")

        assert prompts == ["x"]
        assert "Press Ctrl-C again" in capsys.readouterr().out

    def test_two_ctrl_c_in_a_row_on_an_empty_line_quit(self, tmp_path, capsys):
        prompts, _ = self.read(tmp_path, "\x03\x03")

        assert prompts == [] and capsys.readouterr().out.endswith("Bye.\n")

    def test_help_lists_the_keys_of_the_editor(self, tmp_path, capsys):
        self.read(tmp_path, "/help\r/exit\r")

        out = capsys.readouterr().out
        assert "  Alt-Enter, Ctrl-J  Add a line to the prompt" in out
        assert "Ctrl-C             Clear the line; on an empty line, press twice" in out
        assert '"Allow …?": y allows, a allows the tool for the session' in out

    def test_a_line_added_by_hand_has_a_marker_under_the_prompt(self, tmp_path):
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )

            assert session is not None and session.prompt_continuation == ". "

    def test_a_pasted_block_is_one_prompt(self, tmp_path):
        paste = "\x1b[200~line one\nline two\x1b[201~\r"

        prompts, _ = self.read(tmp_path, paste + "/exit\r")

        assert prompts == ["line one\nline two"]

    def test_prompts_are_kept_in_the_history_file(self, tmp_path):
        _, history = self.read(tmp_path, "first\rsecond\r/exit\r")

        assert "+first" in history and "+second" in history

    def test_the_slash_commands_are_completed(self, tmp_path):
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )
            assert session is not None and session.completer is not None
            found = session.completer.get_completions(Document("/re"), None)  # pyright: ignore[reportArgumentType]

            assert sorted(c.text for c in found) == ["/reset", "/retry"]

    def test_redirected_input_is_read_with_plain_input(self, tmp_path):
        # Under pytest stdin is not a terminal.
        assert terminal_session(COMMANDS, tmp_path / "state" / "history") is None
        assert not (tmp_path / "state").exists()

    @pytest.mark.parametrize(
        ("typed", "prompt"),
        [
            ("one\x1b\rtwo\r", "one\ntwo"),  # Alt-Enter: Esc, then Enter
            ("one\ntwo\r", "one\ntwo"),  # Ctrl-J, a line feed
            ("one\x1b[13;2utwo\r", "one\ntwo"),  # Shift-Enter, kitty protocol
            ("one\x1b[27;2;13~two\r", "one\ntwo"),  # Shift-Enter, modifyOtherKeys
        ],
        ids=["alt-enter", "ctrl-j", "shift-enter-csi-u", "shift-enter-xterm"],
    )
    def test_a_line_can_be_added_by_hand_and_enter_sends(self, tmp_path, typed, prompt):
        prompts, _ = self.read(tmp_path, typed + "/exit\r")

        assert prompts == [prompt]

    def complete(self, tmp_path, text: str) -> list[str]:
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )
            assert session is not None and session.completer is not None
            found = session.completer.get_completions(Document(text), None)  # pyright: ignore[reportArgumentType]
            return sorted(c.text for c in found)

    def test_the_levels_are_completed_after_effort(self, tmp_path):
        assert self.complete(tmp_path, "/effort h") == ["high"]
        assert "default" in self.complete(tmp_path, "/effort ")

    def test_a_command_that_lists_no_arguments_completes_none(self, tmp_path):
        assert self.complete(tmp_path, "/model ") == []
        assert self.complete(tmp_path, "hello") == []

    def test_an_earlier_prompt_is_suggested(self, tmp_path):
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )
            assert session is not None and session.auto_suggest is not None
            session.default_buffer.history.append_string("hello there")

            suggestion = session.auto_suggest.get_suggestion(
                session.default_buffer, Document("hel")
            )

            assert suggestion is not None and suggestion.text == "lo there"

    def test_the_history_is_for_this_user_only(self, tmp_path):
        history = tmp_path / "history"
        with create_pipe_input() as pipe:
            terminal_session(COMMANDS, history, input=pipe, output=DummyOutput())

        assert history.stat().st_mode & 0o777 == 0o600

    def test_a_history_file_that_cannot_be_made_is_not_an_error(self, tmp_path):
        blocked = tmp_path / "a-file"
        blocked.write_text("")
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, blocked / "history", input=pipe, output=DummyOutput()
            )

        assert session is not None  # the prompts are kept in memory instead

    def test_an_answer_to_a_question_is_not_a_prompt(self, tmp_path):
        history = tmp_path / "history"
        agent = Agent(make_config())
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, history, input=pipe, output=DummyOutput()
            )
            repl = Repl(agent, COMMANDS, session)
            pipe.send_text("y\r")

            def danger(path: str) -> str:
                """Do something."""
                return ""

            answer = repl.console.confirm(
                Tool.from_function(danger, read_only=False), "{}"
            )

        assert answer is True
        assert history.read_text() == ""  # "y" is not a prompt to recall
