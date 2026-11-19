"""The Responder on its own: a request to the fake API, no agent, no conversation."""

import pytest

from ergane.errors import StreamCut, TurnError
from ergane.responses import STREAM_RETRIES, Responder, response_problem
from fake_api import TEXT
from helpers import make_config

DRAFT = [{"role": "user", "content": "hi"}]
TOOL = {
    "type": "function",
    "name": "look",
    "description": "d",
    "parameters": {"type": "object", "properties": {}},
    "strict": False,
}


@pytest.fixture
def responder(client):
    def make(**settings) -> Responder:
        return Responder(client, make_config(**settings))

    return make


def send(responder: Responder, prompt: str = "hi", **kwargs):
    pieces: list[str] = []
    notices: list[str] = []
    response = responder.send(
        [{"role": "user", "content": prompt}],
        "be brief",
        pieces.append,
        notices.append,
        **kwargs,
    )
    return response, pieces, notices


@pytest.mark.parametrize("streaming", [True, False])
class TestSend:
    def test_the_text_is_passed_on_and_the_final_response_returned(
        self, responder, streaming
    ):
        response, pieces, _ = send(responder(streaming=streaming))

        assert "".join(pieces) == TEXT == response.output_text

    def test_the_request_is_not_stored_and_asks_for_the_reasoning_back(
        self, responder, fake_server, streaming
    ):
        send(responder(streaming=streaming, model="m1"))

        body = fake_server.requests[0]
        assert body["store"] is False and body["model"] == "m1"
        assert body["instructions"] == "be brief"
        assert body["include"] == ["reasoning.encrypted_content"]

    def test_tools_are_offered_only_when_there_are_some(
        self, responder, fake_server, streaming
    ):
        send(responder(streaming=streaming))
        send(responder(streaming=streaming), tools=[TOOL])

        assert "tools" not in fake_server.requests[0]
        assert [t["name"] for t in fake_server.requests[1]["tools"]] == ["look"]

    def test_the_answer_can_be_capped(self, responder, fake_server, streaming):
        send(responder(streaming=streaming), max_output_tokens=50)
        send(responder(streaming=streaming))

        assert fake_server.requests[0]["max_output_tokens"] == 50
        assert "max_output_tokens" not in fake_server.requests[1]

    def test_no_reasoning_is_asked_for_unless_the_user_set_one(
        self, responder, fake_server, streaming
    ):
        send(responder(streaming=streaming))
        send(responder(streaming=streaming, reasoning_effort="high"))

        assert "reasoning" not in fake_server.requests[0]
        assert fake_server.requests[1]["reasoning"] == {"effort": "high"}

    @pytest.mark.parametrize(
        ("set_to", "sent"),
        [("high", "low"), ("medium", "low"), ("none", "none"), ("low", "low")],
    )
    def test_low_effort_thinks_as_little_as_it_can_but_never_more_than_set(
        self, responder, fake_server, streaming, set_to, sent
    ):
        send(responder(streaming=streaming, reasoning_effort=set_to), low_effort=True)

        assert fake_server.requests[0]["reasoning"] == {"effort": sent}


class TestStreamRetries:
    def test_a_stream_cut_before_any_text_is_sent_again(self, responder, fake_server):
        _response, pieces, notices = send(responder(streaming=True), "flaky")

        assert len(fake_server.requests) == 2
        assert len(notices) == 1 and notices[0].endswith("trying again (1 of 2)")
        assert "".join(pieces) == TEXT  # shown once

    def test_it_gives_up_after_the_retries(self, responder, fake_server):
        with pytest.raises(StreamCut):
            send(responder(streaming=True), "truncated")

        assert len(fake_server.requests) == 1 + STREAM_RETRIES

    def test_a_stream_cut_after_text_is_not_sent_again(self, responder, fake_server):
        with pytest.raises(StreamCut):
            send(responder(streaming=True), "cut_after_text")

        assert len(fake_server.requests) == 1

    def test_the_server_saying_why_ends_it_at_once(self, responder, fake_server):
        with pytest.raises(TurnError):
            send(responder(streaming=True), "error_event")

        assert len(fake_server.requests) == 1

    def test_without_streaming_nothing_is_retried_by_it(self, responder, fake_server):
        _, _, notices = send(responder(streaming=False), "flaky")

        assert notices == [] and len(fake_server.requests) == 1


def test_it_needs_nothing_of_the_agent_or_the_history():
    # A fresh interpreter: this process has imported everything already.
    import subprocess
    import sys

    code = (
        "import sys, ergane.responses; "
        "print([m for m in ('ergane.agent', 'ergane.context') "
        "if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"


def test_a_response_with_nothing_is_a_problem():
    from openai.types.responses import Response

    from fake_api import make_response

    empty = Response.model_validate(make_response("completed", []))

    assert response_problem(empty) == "No text in response"
