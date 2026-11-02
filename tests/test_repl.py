import pytest

from fake_api import REFUSAL, STREAM_ERROR, TEXT
from helpers import make_config, make_context, run_repl
from muagent.agent import Agent
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
