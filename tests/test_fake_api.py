"""The fake API's tool scenarios, driven through the real SDK in both modes.

They are the ground the tool-calling tests will stand on, so they check the whole
exchange the way the OpenAI docs describe it.
"""

from typing import Any

import pytest
from openai import OpenAI

from fake_api import FUNCTION_CALL, REASONING_ITEM, TEXT


def respond(client: OpenAI, items: Any, streaming: bool):
    """The final response for an input, whichever way it is fetched."""
    if streaming:
        with client.responses.stream(model="m", input=items) as stream:
            for _ in stream:
                pass
            return stream.get_final_response()
    return client.responses.create(model="m", input=items)


def continue_after_tool(prompt: str, response, output: str = "file contents") -> list:
    """The input that goes on after a tool call, as the OpenAI docs build it.

    That is the user's message, every item the model produced (reasoning
    included), then the tool's output.
    """
    return [
        {"role": "user", "content": prompt},
        *(item.model_dump(exclude_none=True) for item in response.output),
        *(
            {"type": "function_call_output", "call_id": call.call_id, "output": output}
            for call in response.output
            if call.type == "function_call"
        ),
    ]


@pytest.mark.parametrize("streaming", [True, False])
class TestToolScenarios:
    def test_a_tool_call_then_text_once_the_output_is_sent(self, client, streaming):
        first = respond(client, [{"role": "user", "content": "tool_call"}], streaming)

        assert [item.type for item in first.output] == ["function_call"]
        call = first.output[0]
        assert call.type == "function_call"
        assert call.name == FUNCTION_CALL["name"]
        assert call.call_id == FUNCTION_CALL["call_id"]
        assert call.arguments == FUNCTION_CALL["arguments"]
        assert first.output_text == ""

        second = respond(client, continue_after_tool("tool_call", first), streaming)

        assert second.output_text == TEXT

    def test_a_reasoning_item_comes_with_the_call_and_goes_back_with_the_output(
        self, client, fake_server, streaming
    ):
        prompt = "tool_call_reasoning"
        first = respond(client, [{"role": "user", "content": prompt}], streaming)

        assert [item.type for item in first.output] == ["reasoning", "function_call"]
        reasoning = first.output[0]
        assert reasoning.type == "reasoning"
        assert reasoning.encrypted_content == REASONING_ITEM["encrypted_content"]

        second = respond(client, continue_after_tool(prompt, first), streaming)

        assert second.output_text == TEXT
        sent = fake_server.requests[-1]["input"]
        assert [item.get("type", item.get("role")) for item in sent] == [
            "user",
            "reasoning",
            "function_call",
            "function_call_output",
        ]

    def test_two_calls_in_one_response_are_both_answered_in_the_next_request(
        self, client, fake_server, streaming
    ):
        prompt = "tool_call_parallel"
        first = respond(client, [{"role": "user", "content": prompt}], streaming)

        assert [item.type for item in first.output] == ["function_call"] * 2
        assert [
            item.call_id for item in first.output if item.type == "function_call"
        ] == [
            "call_1",
            "call_2",
        ]

        second = respond(client, continue_after_tool(prompt, first), streaming)

        assert second.output_text == TEXT
        sent = fake_server.requests[-1]["input"]
        assert [
            item.get("call_id")
            for item in sent
            if item.get("type") == "function_call_output"
        ] == [
            "call_1",
            "call_2",
        ]

    def test_a_tool_loop_calls_again_after_every_output(self, client, streaming):
        first = respond(client, [{"role": "user", "content": "tool_loop"}], streaming)

        again = respond(client, continue_after_tool("tool_loop", first), streaming)

        assert [item.type for item in again.output] == ["function_call"]
