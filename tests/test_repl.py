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
                prompt: str, on_text=None, on_notice=None, confirm=None
            ) -> None:
                prompts.append(prompt)
                if turn:
                    turn(prompt, agent.context)

            monkeypatch.setattr(agent, "run_turn", fake_run_turn)
            run_repl(monkeypatch, agent, inputs)
            return prompts

        return run

    @pytest.mark.parametrize(
        "stop", ["/exit", "/quit", "  /EXIT  ", EOFError(), KeyboardInterrupt()]
    )
    def test_stops_on_exit_command_or_eof(self, repl, capsys, stop):
        assert repl([stop]) == []
        assert capsys.readouterr().out.endswith("Bye.\n")

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

        def turn(prompt: str, on_text, on_notice, confirm) -> None:
            on_text("line\n")

        monkeypatch.setattr(agent, "run_turn", turn)
        run_repl(monkeypatch, agent, ["hi", "/exit"])

        assert "\nline\n\nBye." in capsys.readouterr().out

    def test_ctrl_c_mid_reply_puts_the_notice_on_its_own_line(
        self, monkeypatch, capsys
    ):
        agent = Agent(make_config())

        def turn(prompt: str, on_text, on_notice, confirm) -> None:
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
            '> tool_call\n\nAllow read_file {"path": "notes.txt"}? [y/N] y\n\n'
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
    def read(tmp_path, typed: str) -> tuple[list[str], str]:
        """Run a Repl whose session reads `typed`; return the prompts and the history."""
        agent = Agent(make_config())
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
