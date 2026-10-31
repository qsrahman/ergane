import re

import pytest

from fake_api import REFUSAL, STREAM_ERROR, TEXT
from helpers import make_context
from muagent.agent import TurnError


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
        with pytest.raises(TurnError):
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

        with pytest.raises(TurnError, match=f"^{re.escape(message)}$"):
            agent.run_turn(prompt)

        assert agent.context.items == original

    def test_incomplete_response_raises_but_keeps_the_partial_reply(
        self, make_agent, streaming
    ):
        agent = make_agent(STREAMING=streaming)
        pieces: list[str] = []

        with pytest.raises(TurnError, match="^Response incomplete: "):
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

    def test_a_budget_changed_after_startup_takes_effect(self, make_agent, streaming):
        # A setting changed at runtime (as a command would) must be used right away.
        agent = make_agent(STREAMING=streaming)
        agent.context.items = make_context(2)
        notices: list[str] = []

        agent.run_turn("one", on_notice=notices.append)
        assert notices == []  # the default budget leaves plenty of room

        agent.config.max_context_tokens = 1
        agent.run_turn("two", on_notice=notices.append)

        assert notices == ["Context trimmed: oldest turns dropped"]


def test_stream_without_final_response_raises(make_agent):
    agent = make_agent(STREAMING=True)

    with pytest.raises(TurnError, match="^Stream ended without a final response$"):
        agent.run_turn("truncated")

    assert agent.context.items == []


@pytest.mark.parametrize(
    ("prompt", "message"),
    [
        ("error_event", f"Stream error: server_error: {STREAM_ERROR}"),
        ("error_event_nocode", f"Stream error: {STREAM_ERROR}"),
    ],
)
def test_stream_error_event_reports_the_servers_reason(make_agent, prompt, message):
    agent = make_agent(STREAMING=True)
    pieces: list[str] = []

    with pytest.raises(TurnError, match=f"^{re.escape(message)}$"):
        agent.run_turn(prompt, on_text=pieces.append)

    assert pieces == ["Hello"]  # the partial reply was shown...
    assert agent.context.items == []  # ...but a broken stream is not saved
