from http.server import ThreadingHTTPServer

import pytest
from openai import OpenAI
from openai.types.responses import ResponseInputParam

import main
from conftest import TEXT


def make_context(turns: int) -> ResponseInputParam:
    context: ResponseInputParam = []
    for i in range(turns):
        context.append({"role": "user", "content": f"question {i}"})
        context.append({"role": "assistant", "content": f"answer {i}"})
    return context


@pytest.fixture
def client(fake_server: ThreadingHTTPServer) -> OpenAI:
    host, port = fake_server.server_address[:2]
    return OpenAI(api_key="test", base_url=f"http://{host}:{port}/v1", max_retries=0)


@pytest.fixture
def use_config(monkeypatch: pytest.MonkeyPatch):
    """Make run_turn use a config built from the given values, ignoring .env."""

    def apply(**values) -> None:
        config = main.Config.model_validate(
            {"OPENAI_API_KEY": "test", "OPENAI_MODEL": "gpt-4o", **values}
        )
        monkeypatch.setattr(main, "get_config", lambda: config)

    return apply


class TestTrimContext:
    def test_leaves_context_within_budget(self):
        context = make_context(2)
        original = list(context)

        assert not main.trim_context(context, "gpt-4o", max_tokens=10_000)
        assert context == original

    @pytest.mark.parametrize("kept_turns", [1, 2])
    def test_drops_only_as_many_oldest_turns_as_needed(self, kept_turns):
        context = make_context(3)
        newest = context[-2 * kept_turns :]
        budget = sum(main.count_item_tokens(item, "gpt-4o") for item in newest)

        assert main.trim_context(context, "gpt-4o", max_tokens=budget)
        assert context == newest

    def test_keeps_last_user_turn_even_if_over_budget(self):
        context: ResponseInputParam = [{"role": "user", "content": "word " * 100}]
        original = list(context)

        assert not main.trim_context(context, "gpt-4o", max_tokens=1)
        assert context == original


def test_get_encoding_falls_back_for_unknown_model():
    assert main.get_encoding("gemma4:cloud").name == "o200k_base"


def test_get_config_exits_on_missing_api_key(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc:
        main.get_config()

    assert exc.value.code == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("streaming", [True, False])
class TestRunTurn:
    def test_completed_response_is_committed_to_history(
        self, client, use_config, capsys, streaming
    ):
        use_config(STREAMING=streaming)
        context: ResponseInputParam = []

        main.run_turn(client, context, "hi")

        assert context == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": TEXT},
        ]
        assert TEXT in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("prompt", "message"),
        [
            ("incomplete", "[Response incomplete: max_output_tokens]"),
            ("failed", "[Response failed: server_error: boom]"),
            ("empty", "[No text in response]"),
        ],
    )
    def test_unsuccessful_response_leaves_history_untouched(
        self, client, use_config, capsys, streaming, prompt, message
    ):
        use_config(STREAMING=streaming)
        context = make_context(1)
        original = list(context)

        main.run_turn(client, context, prompt)

        assert context == original
        assert message in capsys.readouterr().out

    def test_sends_full_history_statelessly(
        self, client, use_config, fake_server, streaming
    ):
        use_config(STREAMING=streaming)
        context: ResponseInputParam = []

        main.run_turn(client, context, "first")
        main.run_turn(client, context, "second")

        request = fake_server.requests[-1]
        assert request["store"] is False
        assert request.get("stream", False) is streaming
        assert [item["content"] for item in request["input"]] == [
            "first",
            TEXT,
            "second",
        ]

    def test_reports_trimmed_context(
        self, client, use_config, fake_server, capsys, streaming
    ):
        use_config(STREAMING=streaming, MAX_CONTEXT_TOKENS=1)
        context = make_context(2)

        main.run_turn(client, context, "hi")

        assert "[Context trimmed: oldest turns dropped]" in capsys.readouterr().out
        assert fake_server.requests[-1]["input"] == [{"role": "user", "content": "hi"}]


class TestRepl:
    @pytest.fixture
    def repl(self, monkeypatch: pytest.MonkeyPatch, use_config):
        """Run main() with scripted input; returns the prompts passed to run_turn.

        Each scripted entry is either a line of input or an exception that
        input() raises. A callable passed as `turn` replaces run_turn.
        """
        use_config()

        def run(inputs: list, turn=None) -> list[str]:
            lines = iter(inputs)
            prompts: list[str] = []

            def fake_input(_prompt: str) -> str:
                line = next(lines)
                if isinstance(line, BaseException):
                    raise line
                return line

            def fake_run_turn(_client, _context, prompt: str) -> None:
                prompts.append(prompt)
                if turn:
                    turn(prompt)

            monkeypatch.setattr("builtins.input", fake_input)
            monkeypatch.setattr(main, "run_turn", fake_run_turn)
            main.main()
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

    def test_ctrl_c_cancels_only_the_current_turn(self, repl):
        def turn(prompt: str) -> None:
            if prompt == "slow":
                raise KeyboardInterrupt

        assert repl(["slow", "next", "/exit"], turn) == ["slow", "next"]

    def test_turn_errors_are_reported_and_loop_continues(self, repl, capsys):
        def turn(prompt: str) -> None:
            if prompt == "bad":
                raise ValueError("boom")

        assert repl(["bad", "next", "/exit"], turn) == ["bad", "next"]
        assert "Agent error: ValueError: boom" in capsys.readouterr().out
