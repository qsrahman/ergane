import re
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from openai import OpenAI
from openai.types.responses import ResponseInputParam
from pydantic import ValidationError

from fake_api import REFUSAL, TEXT
from muagent import main


def make_context(turns: int) -> ResponseInputParam:
    context: ResponseInputParam = []
    for i in range(turns):
        context.append({"role": "user", "content": f"question {i}"})
        context.append({"role": "assistant", "content": f"answer {i}"})
    return context


def make_config(**values) -> main.Config:
    return main.Config.model_validate(
        {"OPENAI_API_KEY": "test", "OPENAI_MODEL": "gpt-4o", **values}
    )


@pytest.fixture
def client(fake_server: ThreadingHTTPServer) -> OpenAI:
    host, port = fake_server.server_address[:2]
    return OpenAI(api_key="test", base_url=f"http://{host}:{port}/v1", max_retries=0)


@pytest.fixture
def make_agent(client: OpenAI):
    """Build an Agent wired to the fake server, with config from the given values."""

    def make(**values) -> main.Agent:
        return main.Agent(make_config(**values), client)

    return make


class TestContext:
    def test_draft_appends_prompt_without_changing_history(self):
        context = main.Context(max_tokens=10_000)
        context.items = make_context(2)
        original = list(context.items)

        draft, trimmed = context.draft("new")

        assert not trimmed
        assert draft == [*original, {"role": "user", "content": "new"}]
        assert context.items == original

    @pytest.mark.parametrize("kept_turns", [1, 2])
    def test_draft_drops_only_as_many_oldest_turns_as_needed(self, kept_turns):
        # Each turn is about 2,000 tokens, so a budget of kept_turns and a half
        # keeps exactly kept_turns, whatever the estimate's per-item overhead.
        big = "x" * 4000
        history: ResponseInputParam = []
        for i in range(3):
            history.append({"role": "user", "content": f"{i}{big}"})
            history.append({"role": "assistant", "content": big})
        context = main.Context(max_tokens=kept_turns * 2000 + 1000)
        context.items = history

        draft, trimmed = context.draft("new")

        assert trimmed
        assert draft == [
            *history[-2 * kept_turns :],
            {"role": "user", "content": "new"},
        ]

    def test_draft_keeps_last_user_turn_even_if_over_budget(self):
        context = main.Context(max_tokens=1)
        prompt = "word " * 100

        draft, trimmed = context.draft(prompt)

        assert not trimmed
        assert draft == [{"role": "user", "content": prompt}]

    def test_commit_appends_reply_to_draft(self):
        context = main.Context(max_tokens=10_000)
        draft, _ = context.draft("hi")

        context.commit(draft, "hello")

        assert context.items == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_reset_forgets_history(self):
        context = main.Context(max_tokens=10_000)
        context.items = make_context(2)

        context.reset()

        assert context.items == []

    def test_draft_counts_utf8_bytes_so_non_latin_text_is_not_undercounted(self):
        # The same number of characters is about 3x the bytes in CJK as in ASCII,
        # so only the CJK history goes over the budget.
        def draft_trimmed(text: str) -> bool:
            context = main.Context(max_tokens=200)
            context.items = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "ok"},
            ]
            _, trimmed = context.draft("new")
            return trimmed

        assert not draft_trimmed("a" * 400)
        assert draft_trimmed("字" * 400)


@pytest.mark.parametrize("streaming", [True, False])
class TestAgentRunTurn:
    def test_completed_response_is_committed(self, make_agent, streaming):
        agent = make_agent(STREAMING=streaming)

        notices: list[str] = []
        agent.run_turn("hi", on_notice=notices.append)

        assert notices == []
        assert agent.context.items == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": TEXT},
        ]

    def test_passes_reply_text_to_on_text(self, make_agent, streaming):
        agent = make_agent(STREAMING=streaming)
        pieces: list[str] = []

        agent.run_turn("hi", on_text=pieces.append)

        # Streamed deltas arrive one by one; otherwise the whole reply at once.
        assert pieces == (["Hello", " world"] if streaming else [TEXT])

    def test_does_no_terminal_output(self, make_agent, capsys, streaming):
        agent = make_agent(STREAMING=streaming)

        agent.run_turn("hi")
        with pytest.raises(main.TurnError):
            agent.run_turn("failed")

        assert capsys.readouterr() == ("", "")

    @pytest.mark.parametrize(
        ("prompt", "message"),
        [
            ("failed", "Response failed: server_error: boom"),
            ("empty", "No text in response"),
            ("refusal", f"Refused: {REFUSAL}"),
        ],
    )
    def test_unsuccessful_response_raises_and_leaves_history_untouched(
        self, make_agent, streaming, prompt, message
    ):
        agent = make_agent(STREAMING=streaming)
        agent.context.items = make_context(1)
        original = list(agent.context.items)

        with pytest.raises(main.TurnError, match=f"^{re.escape(message)}$"):
            agent.run_turn(prompt)

        assert agent.context.items == original

    def test_incomplete_response_raises_but_keeps_the_partial_reply(
        self, make_agent, streaming
    ):
        agent = make_agent(STREAMING=streaming)
        pieces: list[str] = []

        with pytest.raises(main.TurnError, match="^Response incomplete: "):
            agent.run_turn("incomplete", on_text=pieces.append)

        assert "".join(pieces) == TEXT  # shown to the user, streaming or not
        assert agent.context.items == [
            {"role": "user", "content": "incomplete"},
            {"role": "assistant", "content": TEXT},
        ]

    def test_response_without_a_status_counts_as_completed(self, make_agent, streaming):
        agent = make_agent(STREAMING=streaming)

        agent.run_turn("nostatus")

        assert agent.context.items[-1] == {"role": "assistant", "content": TEXT}

    def test_reasoning_effort_is_sent_only_when_configured(
        self, make_agent, fake_server, streaming
    ):
        make_agent(STREAMING=streaming).run_turn("hi")
        make_agent(STREAMING=streaming, REASONING_EFFORT="low").run_turn("hi")

        unset, low = fake_server.requests[-2:]
        assert "reasoning" not in unset
        assert low["reasoning"] == {"effort": "low"}

    def test_sends_full_history_statelessly(self, make_agent, fake_server, streaming):
        agent = make_agent(STREAMING=streaming)

        agent.run_turn("first")
        agent.run_turn("second")

        request = fake_server.requests[-1]
        assert request["store"] is False
        assert request.get("stream", False) is streaming
        assert [item["content"] for item in request["input"]] == [
            "first",
            TEXT,
            "second",
        ]

    def test_reports_trimmed_context(self, make_agent, fake_server, streaming):
        agent = make_agent(STREAMING=streaming, MAX_CONTEXT_TOKENS=1)
        agent.context.items = make_context(2)

        notices: list[str] = []
        agent.run_turn("hi", on_notice=notices.append)

        assert notices == ["Context trimmed: oldest turns dropped"]
        assert fake_server.requests[-1]["input"] == [{"role": "user", "content": "hi"}]


def test_stream_without_final_response_raises(make_agent):
    agent = make_agent(STREAMING=True)

    with pytest.raises(main.TurnError, match="^Stream ended without a final response$"):
        agent.run_turn("truncated")

    assert agent.context.items == []


def run_repl(monkeypatch: pytest.MonkeyPatch, agent: main.Agent, inputs: list) -> None:
    """Run a Repl over scripted input.

    Each entry is either a line of input or an exception that input() raises.
    """
    lines = iter(inputs)

    def fake_input(_prompt: str) -> str:
        line = next(lines)
        if isinstance(line, BaseException):
            raise line
        return line

    monkeypatch.setattr("builtins.input", fake_input)
    main.Repl(agent).run()


class TestRepl:
    @pytest.fixture
    def repl(self, monkeypatch: pytest.MonkeyPatch):
        """Run a Repl with scripted input; returns the prompts sent to the agent.

        A callable passed as `turn` is called with each prompt and the agent's
        context in place of Agent.run_turn.
        """

        def run(inputs: list, turn=None) -> list[str]:
            agent = main.Agent(make_config())
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
        agent = main.Agent(make_config())

        def turn(prompt: str, on_text, on_notice) -> None:
            on_text("line\n")

        monkeypatch.setattr(agent, "run_turn", turn)
        run_repl(monkeypatch, agent, ["hi", "/exit"])

        assert "\nline\n\nBye." in capsys.readouterr().out

    def test_ctrl_c_mid_reply_puts_the_notice_on_its_own_line(
        self, monkeypatch, capsys
    ):
        agent = main.Agent(make_config())

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

        def turn(prompt: str, context: main.Context) -> None:
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
        monkeypatch.setattr(main.sys.stdout, "isatty", lambda: is_terminal)
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


def test_main_wires_config_agent_and_repl(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_MODEL", "wired-model")
    monkeypatch.setattr("builtins.input", lambda _prompt: "/exit")

    main.main()

    out = capsys.readouterr().out
    assert "model: wired-model" in out
    assert out.endswith("Bye.\n")


def test_main_exits_on_invalid_config(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_invalid_reasoning_effort_is_a_config_error():
    with pytest.raises(ValidationError, match="REASONING_EFFORT"):
        make_config(REASONING_EFFORT="extreme")


class TestLoadConfig:
    """isolate_config makes tmp_path both the working directory and XDG_CONFIG_HOME."""

    @staticmethod
    def write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_reads_the_user_config_file(self, tmp_path: Path):
        self.write(
            tmp_path / "muagent" / ".env",
            "OPENAI_API_KEY=user\nOPENAI_MODEL=user-model\n",
        )

        assert main.load_config().model == "user-model"

    def test_project_env_overrides_the_user_file(self, tmp_path: Path):
        self.write(
            tmp_path / "muagent" / ".env",
            "OPENAI_API_KEY=user\nOPENAI_MODEL=user-model\n",
        )
        self.write(tmp_path / ".env", "OPENAI_MODEL=project-model\n")

        config = main.load_config()

        assert config.model == "project-model"
        assert config.api_key == "user"

    def test_environment_overrides_both_files(self, tmp_path: Path, monkeypatch):
        self.write(tmp_path / "muagent" / ".env", "OPENAI_API_KEY=user\n")
        self.write(tmp_path / ".env", "OPENAI_MODEL=project-model\n")
        monkeypatch.setenv("OPENAI_MODEL", "env-model")

        assert main.load_config().model == "env-model"

    def test_ignores_keys_that_are_not_settings(self, tmp_path: Path):
        self.write(tmp_path / ".env", "OPENAI_API_KEY=x\nSOME_OTHER_TOOL=1\n")

        assert main.load_config().api_key == "x"


class TestSanitize:
    def test_removes_escape_sequences_and_bidi_controls(self):
        # An OSC 52 clipboard write, a carriage return and a right-to-left override.
        text = "a\x1b]52;c;ZXZpbA==\x07b\rc\u202ed"

        assert main.sanitize(text) == "a]52;c;ZXZpbA==bcd"

    def test_keeps_newlines_tabs_and_ordinary_unicode(self):
        text = "line 1\n\tπ ≈ 3.14 字\n"

        assert main.sanitize(text) == text

    def test_notices_are_sanitized(self, monkeypatch, capsys):
        agent = main.Agent(make_config())

        def fail(prompt: str, on_text=None, on_notice=None) -> None:
            raise main.TurnError("Response failed: \x1b[2Jboom")

        monkeypatch.setattr(agent, "run_turn", fail)
        run_repl(monkeypatch, agent, ["hi", "/exit"])

        out = capsys.readouterr().out
        assert "[Response failed: [2Jboom]" in out
        assert "\x1b" not in out
