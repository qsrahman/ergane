import sys

import pytest

from fake_api import REFUSAL, TEXT
from helpers import make_config, make_context, run_repl
from muagent.agent import Agent
from muagent.context import Context


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

            def fake_run_turn(prompt: str, on_text=None, on_notice=None) -> None:
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

        def turn(prompt: str, on_text, on_notice) -> None:
            on_text("line\n")

        monkeypatch.setattr(agent, "run_turn", turn)
        run_repl(monkeypatch, agent, ["hi", "/exit"])

        assert "\nline\n\nBye." in capsys.readouterr().out

    def test_ctrl_c_mid_reply_puts_the_notice_on_its_own_line(
        self, monkeypatch, capsys
    ):
        agent = Agent(make_config())

        def turn(prompt: str, on_text, on_notice) -> None:
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

    @staticmethod
    def committing_turn(history_sizes: list[int]):
        """A fake turn that records the history size, then commits the prompt."""

        def turn(prompt: str, context: Context) -> None:
            history_sizes.append(len(context.items))
            draft, _ = context.draft(prompt)
            context.commit(draft, "reply")

        return turn

    def test_reset_clears_history(self, repl, capsys):
        sizes: list[int] = []

        prompts = repl(
            ["a", "b", " /Reset ", "c", "/exit"], self.committing_turn(sizes)
        )

        assert prompts == ["a", "b", "c"]
        assert sizes == [0, 2, 0]
        assert "[History reset]" in capsys.readouterr().out

    @pytest.mark.parametrize("is_terminal", [True, False])
    def test_clear_clears_screen_only_on_a_terminal_and_keeps_history(
        self, repl, capsys, monkeypatch, is_terminal
    ):
        # capsys replaces stdout with a capture object, which is never a terminal.
        monkeypatch.setattr(sys.stdout, "isatty", lambda: is_terminal)
        sizes: list[int] = []

        prompts = repl(["a", "/clear", "b", "/exit"], self.committing_turn(sizes))

        assert prompts == ["a", "b"]
        assert sizes == [0, 2]
        printed_codes = "\033[2J\033[3J\033[H" in capsys.readouterr().out
        assert printed_codes is is_terminal

    def test_help_lists_every_command(self, repl, capsys):
        repl(["/help", "/exit"])

        out = capsys.readouterr().out
        for name in ("/help", "/reset", "/clear", "/exit", "/quit"):
            assert name in out

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

    def test_reports_trimmed_context(self, monkeypatch, make_agent, capsys, streaming):
        agent = make_agent(STREAMING=streaming, MAX_CONTEXT_TOKENS=1)
        agent.context.items = make_context(2)

        run_repl(monkeypatch, agent, ["hi", "/exit"])

        out = capsys.readouterr().out
        assert f"[Context trimmed: oldest turns dropped]\n{TEXT}\n" in out
