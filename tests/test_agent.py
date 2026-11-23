import os
import re
import signal
import socket
import threading
import time
from datetime import date

import pytest
from openai import APIConnectionError, BadRequestError, OpenAI
from openai.types.responses import Response, ResponseInputParam

from ergane.agent import Agent
from ergane.context import estimate_tokens
from ergane.errors import TurnError, describe_error
from ergane.loops import MAX_REPEATS, WARN_REPEATS
from ergane.outputs import DROPPED_OUTPUT, IMAGE_REFUSED
from ergane.prompts import (
    SERVER_NOTES,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_NO_TOOLS,
)
from ergane.responses import STREAM_RETRIES, response_problem
from ergane.tools import MAX_OUTPUT_CHARS, Image, Tool, ToolOutput
from fake_api import (
    FUNCTION_CALL,
    REASONING_ITEM,
    REFUSAL,
    SECOND_FUNCTION_CALL,
    STREAM_ERROR,
    TEXT,
    USAGE_INPUT_TOKENS,
    make_response,
)
from helpers import make_config, make_context, tool_with


@pytest.mark.parametrize("streaming", [True, False])
class TestAgentRunTurn:
    @pytest.mark.parametrize("with_tools", [False, True])
    def test_the_instructions_are_the_prompt_then_the_environment(
        self, make_agent, fake_server, streaming, with_tools
    ):
        tools = [READ_FILE] if with_tools else []
        make_agent(tools=tools, STREAMING=streaming).run_turn("hi")

        instructions = fake_server.requests[0]["instructions"]
        prompt = SYSTEM_PROMPT if with_tools else SYSTEM_PROMPT_NO_TOOLS
        # The prompt first, so the part that never changes can be cached.
        assert instructions.startswith(prompt)
        assert "- Operating system: " in instructions
        assert instructions.endswith(f"- Current date: {date.today().isoformat()}\n")

    def test_a_picture_from_a_tool_goes_to_the_model_and_stays_in_the_history(
        self, make_agent, fake_server, streaming
    ):
        picture = Image("aGk=", "image/png")
        shot = Tool(
            "read_file",
            "Take a screenshot.",
            {"type": "object", "properties": {}},
            lambda arguments: None,
            lambda arguments: ToolOutput("the page", (picture,)),
            True,
        )
        agent = make_agent(tools=[shot], STREAMING=streaming)

        agent.run_turn("tool_call")

        output = [
            {"type": "input_text", "text": "the page"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,aGk=",
                "detail": "auto",
            },
        ]
        sent = fake_server.requests[1]["input"][-1]
        assert sent == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": output,
        }
        assert agent.context.items[2]["output"] == output  # type: ignore[index]

    @staticmethod
    def shot() -> Tool:
        return Tool(
            "read_file",
            "Take a screenshot.",
            {"type": "object", "properties": {}},
            lambda arguments: None,
            lambda arguments: ToolOutput("the page", (Image("aGk=", "image/png"),)),
            True,
        )

    def test_a_model_that_takes_no_pictures_gets_the_turn_again_without_them(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[self.shot()], STREAMING=streaming)
        notices: list[str] = []

        agent.run_turn("novision_tool_call", on_notice=notices.append)

        assert len(fake_server.requests) == 3  # the call, the refusal, the retry
        assert len(notices) == 1 and "does not take pictures" in notices[0]
        assert "input_image" not in str(fake_server.requests[2]["input"])
        # The history has the result as text, and a reply: the turn was saved whole.
        assert agent.context.items[2]["output"] == f"the page\n{IMAGE_REFUSED}"  # type: ignore[index]
        assert agent.context.items[-1]["role"] == "assistant"  # type: ignore[index]

    def test_the_conversation_goes_on_after_that(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[self.shot()], STREAMING=streaming)
        agent.run_turn("novision_tool_call")

        agent.run_turn("hello")

        assert "input_image" not in str(fake_server.requests[-1]["input"])
        assert agent.context.items[-1]["role"] == "assistant"  # type: ignore[index]

    def test_a_refusal_with_no_picture_to_take_out_ends_the_turn(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(STREAMING=streaming)

        with pytest.raises(BadRequestError):
            agent.run_turn("picture_error")

        assert len(fake_server.requests) == 1  # nothing was sent again

    def test_after_a_refusal_later_pictures_are_not_sent_at_all(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[self.shot()], STREAMING=streaming)
        notices: list[str] = []

        with pytest.raises(TurnError, match="^Stopped: the model keeps repeating"):
            agent.run_turn("novision_tool_loop", on_notice=notices.append)

        # Every round got a picture back, but only the first was sent (and refused);
        # the others were replaced by a note, so no request was spent on them.
        refusals = [n for n in notices if "does not take pictures" in n]
        assert len(refusals) == 1
        sent = [r for r in fake_server.requests if "input_image" in str(r["input"])]
        assert len(sent) == 1
        assert IMAGE_REFUSED in str(fake_server.requests[-1]["input"])

    def test_a_later_picture_is_answered_with_a_note_that_tells_the_model_why(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[self.shot()], STREAMING=streaming)
        agent.run_turn("novision_tool_call")
        before = len(fake_server.requests)

        agent.run_turn("novision_tool_call")

        outputs = [
            item["output"]  # type: ignore[index]
            for item in agent.context.items
            if item.get("type") == "function_call_output"
        ]
        assert outputs[-1] == f"the page\n{IMAGE_REFUSED}"
        # Two requests for the turn, neither refused: no retry was needed.
        assert len(fake_server.requests) - before == 2

    def test_the_memory_is_of_the_model_so_another_one_gets_its_pictures(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[self.shot()], STREAMING=streaming)
        agent.run_turn("novision_tool_call")
        agent.config.model = "another-model"

        agent.run_turn("novision_tool_call")

        sent = [r for r in fake_server.requests if "input_image" in str(r["input"])]
        assert len(sent) == 2  # one refused request for each model

    def test_an_error_that_is_not_about_pictures_leaves_them_alone(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[self.shot()], STREAMING=streaming)

        with pytest.raises(BadRequestError):
            agent.run_turn("tool_then_http_error")

        output = agent.context.items[2]["output"]  # type: ignore[index]
        assert isinstance(output, list)  # the picture is still there
        assert len(fake_server.requests) == 2  # and nothing was sent again

    def test_what_servers_say_follows_the_prompt_and_comes_before_the_environment(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(
            tools=[READ_FILE],
            server_notes={"fake": "Dates are YYYY-MM-DD."},
            STREAMING=streaming,
        )

        agent.run_turn("hi")

        instructions = fake_server.requests[0]["instructions"]
        assert instructions.startswith(SYSTEM_PROMPT + SERVER_NOTES)
        assert "## fake\nDates are YYYY-MM-DD.\n" in instructions
        assert instructions.index("## fake") < instructions.index("Environment:")

    def test_what_servers_say_is_left_out_when_there_are_no_tools(
        self, make_agent, fake_server, streaming
    ):
        make_agent(server_notes={"fake": "Dates."}, STREAMING=streaming).run_turn("hi")

        assert "Dates." not in fake_server.requests[0]["instructions"]

    def test_what_every_request_carries_counts_the_instructions_and_the_tools(
        self, make_agent, streaming
    ):
        one = make_agent(tools=[READ_FILE], STREAMING=streaming)
        two = make_agent(
            tools=[READ_FILE, Tool.from_function(other)], STREAMING=streaming
        )

        assert make_agent(STREAMING=streaming)._fixed_tokens() >= estimate_tokens(
            {"instructions": SYSTEM_PROMPT_NO_TOOLS}
        )
        assert (
            two._fixed_tokens()
            > one._fixed_tokens()
            > estimate_tokens({"instructions": SYSTEM_PROMPT})
        )

    def test_the_history_gets_what_the_instructions_leave_of_the_budget(
        self, make_agent, streaming
    ):
        history = make_context(4)
        prompt = {"role": "user", "content": "hi"}
        size = sum(estimate_tokens(item) for item in [*history, prompt])

        def trimmed_with(budget: int) -> bool:
            agent = make_agent(STREAMING=streaming, MAX_CONTEXT_TOKENS=budget)
            agent.context.items = list(history)
            notices: list[str] = []
            agent.run_turn("hi", on_notice=notices.append)
            return "Context trimmed: oldest turns dropped" in notices

        fixed = make_agent(STREAMING=streaming)._fixed_tokens()
        # The history and the prompt fit the budget alone, but the instructions
        # have to fit in it as well.
        assert trimmed_with(size + fixed - 1)
        assert not trimmed_with(size + fixed)

    def test_completed_response_is_saved(self, make_agent, streaming):
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

    def test_an_empty_reply_is_asked_for_again_once(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(STREAMING=streaming)
        notices: list[str] = []

        agent.run_turn("empty_once", on_notice=notices.append)

        assert notices == ["The model sent an empty reply; trying again"]
        assert len(fake_server.requests) == 2
        assert agent.context.items[-1]["content"] == "Hello world"  # type: ignore[index]

    def test_a_second_empty_reply_is_the_error(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(STREAMING=streaming)
        notices: list[str] = []

        with pytest.raises(TurnError, match="No text in response"):
            agent.run_turn("empty", on_notice=notices.append)

        assert len(notices) == 1
        assert len(fake_server.requests) == 2
        assert agent.context.items == []

    @pytest.mark.parametrize(
        ("prompt", "message"),
        [
            ("failed", "Response failed: server_error: boom"),
            ("empty", "No text in response"),
            ("null_output", "No text in response"),
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

    def test_a_response_that_is_not_completed_is_not_saved_even_with_text(
        self, make_agent, streaming
    ):
        # Only "incomplete" keeps its partial text; any other status saves nothing.
        agent = make_agent(STREAMING=streaming)

        with pytest.raises(TurnError, match="^Response not completed: cancelled$"):
            agent.run_turn("cancelled")

        assert agent.context.items == []

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
        assert request["include"] == ["reasoning.encrypted_content"]
        assert request.get("stream", False) is streaming
        assert [item["content"] for item in request["input"]] == [
            "first",
            TEXT,
            "second",
        ]

    def test_reports_trimmed_context(self, make_agent, fake_server, streaming):
        # Summaries off: this is the trimming that is left when they are not made.
        agent = make_agent(
            STREAMING=streaming, MAX_CONTEXT_TOKENS=1, AUTO_COMPACT=False
        )
        agent.context.items = make_context(2)

        notices: list[str] = []
        agent.run_turn("hi", on_notice=notices.append)

        assert notices == ["Context trimmed: oldest turns dropped"]
        assert fake_server.requests[-1]["input"] == [{"role": "user", "content": "hi"}]

    def test_a_budget_changed_after_startup_takes_effect(self, make_agent, streaming):
        # A setting changed at runtime (as a command would) must be used right away.
        agent = make_agent(STREAMING=streaming, AUTO_COMPACT=False)
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


def test_a_response_that_only_calls_a_tool_is_not_a_problem():
    # Having no text is a problem only for the response that ends the turn.
    calls = Response.model_validate(make_response("completed", [FUNCTION_CALL]))
    empty = Response.model_validate(make_response("completed", []))

    assert response_problem(calls) is None
    assert response_problem(empty) == "No text in response"


USAGE = {
    "input_tokens": 1000,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 50,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 1050,
}


def test_the_size_of_the_next_request_is_the_servers_count_plus_what_was_added():
    response = Response.model_validate(make_response("completed", usage=USAGE))
    items: ResponseInputParam = [
        {"role": "user", "content": "x" * 4000},  # sent, so already in the 1,000
        {"type": "function_call_output", "call_id": "c", "output": "y" * 400},
    ]

    assert Agent._next_request_tokens(response, items, 1) == 1000 + estimate_tokens(
        items[1]
    )


def test_without_usage_the_size_is_estimated_from_all_the_items():
    response = Response.model_validate(make_response("completed"))
    items: ResponseInputParam = [
        {"role": "user", "content": "x" * 4000},
        {"type": "function_call_output", "call_id": "c", "output": "y" * 400},
    ]

    assert Agent._next_request_tokens(response, items, 1) == sum(
        estimate_tokens(i) for i in items
    )


def read_file(path: str) -> str:
    """Read a text file."""
    return f"contents of {path}"


def other(path: str) -> str:
    """Another tool, with a different name from the one the fake API calls."""
    return path


READ_FILE = Tool.from_function(read_file)
# The rounds of a model that makes the same step every time, until the guard stops it.
LOOP_ROUNDS = MAX_REPEATS + 1
USER_TURN = {"role": "user", "content": "tool_call"}
FIRST_OUTPUT = {
    "type": "function_call_output",
    "call_id": "call_1",
    "output": "contents of notes.txt",
}


@pytest.mark.parametrize("streaming", [True, False])
class TestAgentToolLoop:
    def test_a_call_is_run_and_its_output_goes_back_to_the_model(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)

        agent.run_turn("tool_call")

        assert len(fake_server.requests) == 2
        assert fake_server.requests[1]["input"] == [
            USER_TURN,
            FUNCTION_CALL,
            FIRST_OUTPUT,
        ]
        # The whole turn is saved: the call and its output, then the reply.
        assert agent.context.items == [
            USER_TURN,
            FUNCTION_CALL,
            FIRST_OUTPUT,
            {"role": "assistant", "content": TEXT},
        ]

    def test_the_tool_work_of_a_finished_turn_is_resent_once_and_then_dropped(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)
        agent.run_turn("tool_call")

        agent.run_turn("and then?")
        agent.run_turn("and then?")

        def types(request):
            return [item.get("type") for item in request["input"]]

        # The follow-up still sees the calls of the turn before it...
        assert types(fake_server.requests[2]) == [
            None,
            "function_call",
            "function_call_output",
            None,
            None,
        ]
        # ...and the next one only its messages.
        assert types(fake_server.requests[3]) == [None] * 5

    def test_two_tools_with_the_same_name_are_an_error(self, make_agent, streaming):
        with pytest.raises(ValueError, match="unique: read_file"):
            make_agent(tools=[READ_FILE, READ_FILE], STREAMING=streaming)

    def test_add_tools_keeps_the_first_of_a_name_and_names_the_others(
        self, make_agent, streaming
    ):
        agent = make_agent(STREAMING=streaming)
        clash = Tool.from_function(read_file)

        skipped = agent.add_tools([READ_FILE, Tool.from_function(other), clash])

        assert skipped == ["read_file"]
        assert agent.tools["read_file"] is READ_FILE
        assert list(agent.tools) == ["read_file", "other"]

    def test_the_tools_are_offered_in_every_request(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)

        agent.run_turn("tool_call")

        # The todo tool goes with any other tool (see test_todos.py).
        assert [request["tools"] for request in fake_server.requests] == [
            [agent._todo_tool.param(), READ_FILE.param()]
        ] * 2

    def test_no_tools_are_sent_when_there_are_none(
        self, make_agent, fake_server, streaming
    ):
        make_agent(STREAMING=streaming).run_turn("hi")

        assert "tools" not in fake_server.requests[0]

    def test_a_reasoning_item_is_sent_back_and_saved_with_its_call(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)

        agent.run_turn("tool_call_reasoning")

        sent = fake_server.requests[1]["input"]
        assert [item.get("type", item.get("role")) for item in sent] == [
            "user",
            "reasoning",
            "function_call",
            "function_call_output",
        ]
        assert sent[1]["encrypted_content"] == REASONING_ITEM["encrypted_content"]
        assert REASONING_ITEM["id"] in [item.get("id") for item in agent.context.items]

    def test_several_calls_in_one_response_are_all_answered_in_one_round(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)

        agent.run_turn("tool_call_parallel")

        assert len(fake_server.requests) == 2
        outputs = [
            (item["call_id"], item["output"])
            for item in fake_server.requests[1]["input"]
            if item.get("type") == "function_call_output"
        ]
        assert outputs == [
            ("call_1", "contents of notes.txt"),
            ("call_2", "contents of todo.txt"),
        ]
        assert agent.context.items[1:3] == [FUNCTION_CALL, SECOND_FUNCTION_CALL]

    def test_a_tool_that_fails_is_reported_to_the_model_and_the_turn_goes_on(
        self, make_agent, fake_server, streaming
    ):
        def failing(path: str) -> str:
            """Read a text file."""
            raise FileNotFoundError(f"no such file: {path}")

        agent = make_agent(tools=[tool_with(failing)], STREAMING=streaming)

        agent.run_turn("tool_call")

        output = fake_server.requests[1]["input"][-1]
        assert output["output"] == "Error: FileNotFoundError: no such file: notes.txt"
        assert agent.context.items[-1] == {"role": "assistant", "content": TEXT}

    def test_invalid_arguments_are_reported_to_the_model(
        self, make_agent, fake_server, streaming
    ):
        def needs_more(path: str, lines: int) -> str:
            """Read some lines of a text file."""
            return path

        agent = make_agent(tools=[tool_with(needs_more)], STREAMING=streaming)

        agent.run_turn("tool_call")

        output = fake_server.requests[1]["input"][-1]["output"]
        assert output == "Invalid arguments: lines: Missing required argument"

    @pytest.mark.parametrize(
        ("tools", "available"),
        [([], "none"), ([Tool.from_function(other)], "todo, other")],
    )
    def test_an_unknown_tool_is_reported_to_the_model(
        self, make_agent, fake_server, streaming, tools, available
    ):
        agent = make_agent(tools=tools, STREAMING=streaming)

        agent.run_turn("tool_call")

        output = fake_server.requests[1]["input"][-1]["output"]
        assert output == f"Unknown tool: read_file. Available tools: {available}."

    def test_a_huge_output_is_cut_before_it_goes_back(
        self, make_agent, fake_server, streaming
    ):
        def huge(path: str) -> str:
            """Read a text file."""
            return "x" * (MAX_OUTPUT_CHARS + 10)

        agent = make_agent(tools=[tool_with(huge)], STREAMING=streaming)

        agent.run_turn("tool_call")

        output = fake_server.requests[1]["input"][-1]["output"]
        assert output.endswith("\n[output truncated: 10 more characters]")
        assert len(output) < MAX_OUTPUT_CHARS + 100

    def test_each_call_is_reported_as_it_is_run(self, make_agent, streaming):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)
        calls: list[tuple[str, str]] = []

        agent.run_turn("tool_call_parallel", on_call=lambda *call: calls.append(call))

        # Only the calls: their results go to the model, not to the screen.
        assert calls == [
            ("read_file", '{"path": "notes.txt"}'),
            ("read_file", '{"path": "todo.txt"}'),
        ]

    def test_a_call_that_is_asked_about_gets_no_line_of_its_own(
        self, make_agent, streaming
    ):
        # `confirm` shows the user the call it asks about; a second line would repeat it.
        agent = make_agent(
            tools=[Tool.from_function(read_file, read_only=False)], STREAMING=streaming
        )
        calls: list[tuple[str, str]] = []
        asked: list[str] = []

        def confirm(tool: Tool, arguments: str) -> bool:
            asked.append(arguments)
            return False

        agent.run_turn(
            "tool_call", on_call=lambda *call: calls.append(call), confirm=confirm
        )

        assert calls == []
        assert asked == [FUNCTION_CALL["arguments"]]

    def test_a_call_that_cannot_run_is_shown_even_for_a_tool_that_asks(
        self, make_agent, streaming
    ):
        # It is never asked about, so nothing else would show it.
        def read_file(path: str, mode: str) -> str:
            """Read a text file."""
            return "contents"

        agent = make_agent(
            tools=[Tool.from_function(read_file, read_only=False)], STREAMING=streaming
        )
        calls: list[tuple[str, str]] = []

        agent.run_turn("tool_call", on_call=lambda *call: calls.append(call))

        assert calls == [("read_file", '{"path": "notes.txt"}')]

    def test_a_call_to_an_unknown_tool_is_shown(self, make_agent, streaming):
        agent = make_agent(STREAMING=streaming)
        calls: list[tuple[str, str]] = []

        agent.run_turn("tool_call", on_call=lambda *call: calls.append(call))

        assert calls == [("read_file", '{"path": "notes.txt"}')]

    def test_a_tool_that_is_not_read_only_is_denied_unless_confirmed(
        self, make_agent, fake_server, streaming
    ):
        ran: list[str] = []

        def read_file(path: str) -> str:
            """Read a text file."""
            ran.append(path)
            return "contents"

        agent = make_agent(
            tools=[Tool.from_function(read_file, read_only=False)], STREAMING=streaming
        )
        notices: list[str] = []

        agent.run_turn("tool_call", on_notice=notices.append)

        # Nothing ran; the model was told, and the turn went on to its reply.
        assert ran == []
        assert fake_server.requests[1]["input"][-1]["output"] == (
            "The user denied this call."
        )
        assert agent.context.items[-1] == {"role": "assistant", "content": TEXT}

    def test_confirm_is_given_the_tool_and_the_full_arguments(
        self, make_agent, streaming
    ):
        agent = make_agent(
            tools=[Tool.from_function(read_file, read_only=False)], STREAMING=streaming
        )
        asked: list[tuple[str, str]] = []

        def confirm(tool: Tool, arguments: str) -> bool:
            asked.append((tool.name, arguments))
            return True

        agent.run_turn("tool_call", confirm=confirm)

        assert asked == [("read_file", FUNCTION_CALL["arguments"])]
        assert agent.context.items[2] == FIRST_OUTPUT

    def test_a_call_with_invalid_arguments_is_not_asked_about(
        self, make_agent, fake_server, streaming
    ):
        ran: list[str] = []

        def read_file(path: str, mode: str) -> str:
            """Read a text file."""
            ran.append(path)
            return "contents"

        agent = make_agent(
            tools=[Tool.from_function(read_file, read_only=False)], STREAMING=streaming
        )

        def confirm(tool: Tool, arguments: str) -> bool:
            raise AssertionError("asked about a call that would be rejected")

        agent.run_turn("tool_call", confirm=confirm)

        # The model reads why the call failed, as it would for a read-only tool.
        assert fake_server.requests[1]["input"][-1]["output"].startswith(
            "Invalid arguments: mode: Missing required argument"
        )
        assert ran == []

    def test_each_call_is_confirmed_on_its_own(self, make_agent, streaming):
        agent = make_agent(
            tools=[Tool.from_function(read_file, read_only=False)], STREAMING=streaming
        )
        answers = iter([True, False])

        agent.run_turn(
            "tool_call_parallel", confirm=lambda tool, arguments: next(answers)
        )

        outputs = [
            item["output"]
            for item in agent.context.items
            if item.get("type") == "function_call_output"
        ]
        assert outputs == ["contents of notes.txt", "The user denied this call."]

    def test_a_read_only_tool_is_never_confirmed(self, make_agent, streaming):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)

        def confirm(tool: Tool, arguments: str) -> bool:
            raise AssertionError("asked about a read-only tool")

        agent.run_turn("tool_call", confirm=confirm)

        assert agent.context.items[2] == FIRST_OUTPUT

    def test_older_tool_results_are_dropped_to_make_room_and_the_turn_goes_on(
        self, make_agent, fake_server, streaming
    ):
        def huge(path: str) -> str:
            """Read a text file."""
            return "x" * MAX_OUTPUT_CHARS  # about 5,000 estimated tokens

        agent = make_agent(
            tools=[tool_with(huge)],
            STREAMING=streaming,
            MAX_CONTEXT_TOKENS=6000,
        )
        notices: list[str] = []

        with pytest.raises(TurnError, match="^Stopped: the model keeps repeating"):
            agent.run_turn("tool_loop", on_notice=notices.append)

        def outputs(request: dict) -> list[str]:
            return [
                item["output"]
                for item in request["input"]
                if item.get("type") == "function_call_output"
            ]

        # Each request has the latest result in full and the earlier ones dropped.
        assert len(fake_server.requests) == LOOP_ROUNDS
        last = outputs(fake_server.requests[-1])
        assert last[:-1] == [DROPPED_OUTPUT] * (LOOP_ROUNDS - 2)
        assert last[-1] == "x" * MAX_OUTPUT_CHARS
        # Not after the last round: the guard stops the turn before the budget check.
        assert notices.count("Context full: older tool results dropped") == (
            LOOP_ROUNDS - 2
        )
        # Every call still has an output, so the kept turn is valid history.
        kept = [i for i in agent.context.items if i.get("type") == "function_call"]
        assert len(kept) == len(outputs({"input": agent.context.items})) == LOOP_ROUNDS

    def test_a_result_that_alone_outgrows_the_budget_is_shortened_and_the_turn_goes_on(
        self, make_agent, fake_server, streaming
    ):
        def huge(path: str) -> str:
            """Read a text file."""
            # Exactly the most a tool may return, so that nothing else cuts it.
            return "head" + "x" * (MAX_OUTPUT_CHARS - 8) + "tail"

        agent = make_agent(
            tools=[tool_with(huge)],
            STREAMING=streaming,
            MAX_CONTEXT_TOKENS=3000,
            AUTO_COMPACT=False,
        )
        notices: list[str] = []

        with pytest.raises(TurnError, match="^Stopped: the model keeps repeating"):
            agent.run_turn("tool_loop", on_notice=notices.append)

        sent = fake_server.requests[1]["input"][-1]["output"]
        assert sent.startswith("head") and sent.endswith("tail") and "hidden" in sent
        assert "Context full: this round's tool results shortened" in notices

    def test_a_round_that_cannot_fit_even_shortened_keeps_the_turn_without_its_results(
        self, make_agent, fake_server, streaming
    ):
        ran: list[str] = []

        def write_file(path: str) -> str:
            """Pretend to write a file, and return a lot of text."""
            ran.append(path)
            return "x" * MAX_OUTPUT_CHARS

        agent = make_agent(
            tools=[tool_with(write_file)],
            STREAMING=streaming,
            MAX_CONTEXT_TOKENS=500,
            AUTO_COMPACT=False,
        )

        with pytest.raises(TurnError, match="grew past the context budget"):
            agent.run_turn("tool_loop")

        # The tool ran; the model must know, and the history must stay small.
        assert ran and len(fake_server.requests) == 1
        items = agent.context.items
        assert [i.get("type") or i["role"] for i in items] == [
            "user",
            "function_call",
            "function_call_output",
            "assistant",
        ]
        assert items[2]["output"] == DROPPED_OUTPUT
        assert items[3]["content"] == (
            "[Stopped after 1 model calls: its results passed the context budget]"
        )

    def test_a_model_that_repeats_itself_is_stopped_and_the_turn_kept(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)

        with pytest.raises(TurnError, match="^Stopped: the model keeps repeating"):
            agent.run_turn("tool_loop")

        assert len(fake_server.requests) == LOOP_ROUNDS
        items = agent.context.items
        # The question, the calls with their outputs, and a closing note: the model
        # still knows what was asked and what was done when the user says to go on.
        assert len(items) == 2 + 2 * LOOP_ROUNDS
        assert items[0] == {"role": "user", "content": "tool_loop"}
        assert [i.get("type") for i in items[1:-1]] == [
            "function_call",
            "function_call_output",
        ] * LOOP_ROUNDS
        assert items[-1] == {
            "role": "assistant",
            "content": f"[Stopped after {LOOP_ROUNDS} model calls: "
            "it kept repeating the same tool calls]",
        }

    def test_it_is_told_in_the_instructions_before_it_is_stopped(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)
        notices: list[str] = []

        with pytest.raises(TurnError):
            agent.run_turn("tool_loop", on_notice=notices.append)

        for i, request in enumerate(fake_server.requests):
            # The request after the third identical step is the first with the notice.
            told = "same tool call with the same result" in request["instructions"]
            assert told == (i >= WARN_REPEATS + 1), i
        assert notices == ["The model is repeating itself; told to try something else"]

    def test_a_model_that_changes_course_is_told_no_more(
        self, make_agent, fake_server, streaming
    ):
        calls: list[str] = []

        def read_file(path: str) -> str:
            """Read a text file."""
            calls.append(path)
            return "stuck" if len(calls) <= 4 else f"new {len(calls)}"

        agent = make_agent(tools=[tool_with(read_file)], STREAMING=streaming)

        agent.run_turn("tool_loop_8")

        told = [
            "same tool call with the same result" in request["instructions"]
            for request in fake_server.requests
        ]
        # After the third and the fourth identical step, then the steps differ.
        assert told == [False] * 3 + [True] * 2 + [False] * 4
        assert agent.context.items[-1]["content"] == "Hello world"  # type: ignore[index]

    def test_the_notice_does_not_outlive_the_turn(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)
        with pytest.raises(TurnError):
            agent.run_turn("tool_loop")

        agent.run_turn("hi")

        assert "same tool call" not in fake_server.requests[-1]["instructions"]

    def test_a_turn_of_many_different_steps_is_not_cut_off(
        self, make_agent, fake_server, streaming
    ):
        calls: list[str] = []

        def read_file(path: str) -> str:
            """Read a text file."""
            calls.append(path)
            return f"result {len(calls)}"

        agent = make_agent(tools=[tool_with(read_file)], STREAMING=streaming)

        agent.run_turn("tool_loop_60")

        assert len(calls) == 60
        assert agent.context.items[-1]["content"] == "Hello world"  # type: ignore[index]

    def test_a_turn_stopped_by_the_guard_can_be_continued_or_retried(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)
        with pytest.raises(TurnError):
            agent.run_turn("tool_loop")

        agent.run_turn("go on")
        assert fake_server.requests[-1]["input"][0] == {
            "role": "user",
            "content": "tool_loop",
        }

        # /retry drops only the last turn; the stopped one stays.
        assert agent.context.rewind() == "go on"
        assert agent.context.items[-1] == {
            "role": "assistant",
            "content": f"[Stopped after {LOOP_ROUNDS} model calls: "
            "it kept repeating the same tool calls]",
        }

    @pytest.mark.parametrize(
        "scenario, reason",
        [
            ("failed", "Response failed: server_error: boom"),
            ("empty", "No text in response"),
            ("null_output", "No text in response"),
            ("refusal", f"Refused: {REFUSAL}"),
            ("cancelled", "Response not completed: cancelled"),
        ],
    )
    def test_a_response_with_a_problem_after_a_round_keeps_the_work(
        self, make_agent, fake_server, streaming, scenario, reason
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)

        with pytest.raises(TurnError, match=re.escape(reason)):
            agent.run_turn(f"tool_then_{scenario}")

        assert agent.context.items == [
            {"role": "user", "content": f"tool_then_{scenario}"},
            FUNCTION_CALL,
            FIRST_OUTPUT,
            {
                "role": "assistant",
                "content": f"[Stopped after 1 model calls: {reason}]",
            },
        ]

    def test_an_api_error_after_a_round_keeps_the_work(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)

        with pytest.raises(BadRequestError):
            agent.run_turn("tool_then_http_error")

        assert agent.context.items[-1] == {
            "role": "assistant",
            "content": "[Stopped after 1 model calls: BadRequestError]",
        }
        assert agent.context.items[1:3] == [FUNCTION_CALL, FIRST_OUTPUT]

    def test_an_api_error_in_the_first_round_saves_nothing(self, make_agent, streaming):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)

        with pytest.raises(BadRequestError):
            agent.run_turn("http_error")

        assert agent.context.items == []

    def test_a_turn_kept_after_a_failure_can_be_continued_or_retried(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)
        with pytest.raises(TurnError):
            agent.run_turn("tool_then_failed")

        agent.run_turn("go on")
        sent = fake_server.requests[-1]["input"]
        assert sent[:3] == [
            {"role": "user", "content": "tool_then_failed"},
            FUNCTION_CALL,
            FIRST_OUTPUT,
        ]
        assert sent[3]["content"].startswith("[Stopped after 1 model calls: ")

        # /retry drops only the last turn; the one kept after the failure stays.
        assert agent.context.rewind() == "go on"
        assert agent.context.items[-1]["content"].startswith("[Stopped after 1")

    def test_ctrl_c_in_the_first_model_call_saves_nothing(self, make_agent, streaming):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)

        def interrupt(text: str) -> None:
            if text:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            agent.run_turn("hello", on_text=interrupt)

        assert agent.context.items == []

    def test_ctrl_c_in_a_later_round_keeps_the_work_already_done(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)

        def interrupt(text: str) -> None:
            if text:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            agent.run_turn("tool_call", on_text=interrupt)  # the reply is round 2

        assert agent.context.items == [
            USER_TURN,
            FUNCTION_CALL,
            FIRST_OUTPUT,
            {
                "role": "assistant",
                "content": "[Stopped after 1 model calls: interrupted by the user]",
            },
        ]
        agent.run_turn("go on")  # the history is valid: the API takes it
        assert fake_server.requests[-1]["input"][2] == FIRST_OUTPUT

    def test_read_only_calls_of_one_response_run_at_the_same_time(
        self, make_agent, streaming
    ):
        spans: list[tuple[float, float]] = []

        def slow_read(path: str) -> str:
            """Read a text file."""
            begun = time.monotonic()
            time.sleep(0.4)
            spans.append((begun, time.monotonic()))
            return path

        agent = make_agent(tools=[tool_with(slow_read)], STREAMING=streaming)

        agent.run_turn("tool_call_parallel")

        # Each began before either ended: one after the other would not do that.
        assert len(spans) == 2
        assert max(begun for begun, _ in spans) < min(end for _, end in spans)

    def test_the_outputs_are_in_the_order_of_the_calls_whoever_ends_first(
        self, make_agent, fake_server, streaming
    ):
        def first_is_slower(path: str) -> str:
            """Read a text file."""
            if path == "notes.txt":
                time.sleep(0.4)
            return f"contents of {path}"

        agent = make_agent(tools=[tool_with(first_is_slower)], STREAMING=streaming)

        agent.run_turn("tool_call_parallel")

        outputs = [
            (i["call_id"], i["output"])
            for i in fake_server.requests[1]["input"]
            if i.get("type") == "function_call_output"
        ]
        assert outputs == [
            ("call_1", "contents of notes.txt"),
            ("call_2", "contents of todo.txt"),
        ]

    def test_each_call_is_shown_before_any_of_them_ends(self, make_agent, streaming):
        log: list[str] = []

        def read(path: str) -> str:
            """Read a text file."""
            time.sleep(0.2)
            log.append(f"ran {path}")
            return path

        agent = make_agent(tools=[tool_with(read)], STREAMING=streaming)

        agent.run_turn(
            "tool_call_parallel", on_call=lambda name, arguments: log.append("shown")
        )

        assert log[:2] == ["shown", "shown"]

    def test_calls_that_are_not_read_only_run_one_at_a_time_and_ask_each(
        self, make_agent, streaming
    ):
        log: list[str] = []

        def write(path: str) -> str:
            """Write a text file."""
            log.append(f"start {path}")
            time.sleep(0.2)
            log.append(f"end {path}")
            return path

        write.__name__ = "read_file"
        agent = make_agent(
            tools=[Tool.from_function(write, read_only=False)], STREAMING=streaming
        )
        asked: list[str] = []

        def confirm(tool, arguments):
            asked.append(arguments)
            return True

        agent.run_turn("tool_call_parallel", confirm=confirm)

        assert log == [
            "start notes.txt",
            "end notes.txt",
            "start todo.txt",
            "end todo.txt",
        ]
        assert len(asked) == 2

    def test_a_call_that_writes_waits_for_the_reads_before_it_and_the_reads_after_wait_for_it(
        self, make_agent, streaming
    ):
        log: list[str] = []

        def read_file(path: str) -> str:
            """Read a text file."""
            log.append(f"read {path} start")
            time.sleep(0.2)
            log.append(f"read {path} end")
            return path

        def write_file(path: str) -> str:
            """Write a text file."""
            log.append("write")
            return path

        agent = make_agent(
            tools=[
                Tool.from_function(read_file),
                Tool.from_function(write_file, read_only=False),
            ],
            STREAMING=streaming,
        )

        agent.run_turn("tool_call_mixed", confirm=lambda tool, arguments: True)

        assert log == [
            "read notes.txt start",
            "read notes.txt end",
            "write",
            "read todo.txt start",
            "read todo.txt end",
        ]

    @staticmethod
    def press_ctrl_c(after: float = 0.3) -> threading.Timer:
        """Send SIGINT to this process after a while, as the terminal does."""
        timer = threading.Timer(after, lambda: os.kill(os.getpid(), signal.SIGINT))
        timer.start()
        return timer

    def test_ctrl_c_while_a_tool_runs_keeps_the_call_with_an_output_that_says_so(
        self, make_agent, fake_server, streaming
    ):
        release = threading.Event()

        def hangs(path: str) -> str:
            """Read a text file."""
            release.wait(10)
            return "late"

        agent = make_agent(tools=[tool_with(hangs)], STREAMING=streaming)
        timer = self.press_ctrl_c()
        try:
            with pytest.raises(KeyboardInterrupt):
                agent.run_turn("tool_call")
        finally:
            timer.cancel()
            release.set()

        items = agent.context.items
        assert [i.get("type") or i["role"] for i in items] == [
            "user",
            "function_call",
            "function_call_output",
            "assistant",
        ]
        assert items[2]["output"].startswith("The user interrupted this call")
        assert items[3]["content"].endswith("interrupted by the user]")
        agent.run_turn("go on")
        assert fake_server.requests[-1]["input"][-1]["content"] == "go on"

    def test_ctrl_c_in_one_of_two_calls_keeps_the_finished_output(
        self, make_agent, streaming
    ):
        release = threading.Event()

        def second_hangs(path: str) -> str:
            """Read a text file."""
            if path == "todo.txt":
                release.wait(10)
            return f"contents of {path}"

        agent = make_agent(tools=[tool_with(second_hangs)], STREAMING=streaming)
        timer = self.press_ctrl_c()
        try:
            with pytest.raises(KeyboardInterrupt):
                agent.run_turn("tool_call_parallel")
        finally:
            timer.cancel()
            release.set()

        outputs = [
            i["output"]
            for i in agent.context.items
            if i.get("type") == "function_call_output"
        ]
        assert outputs[0] == "contents of notes.txt"
        assert outputs[1].startswith("The user interrupted this call")

    def test_a_finished_tool_turn_is_sent_again_in_the_next_turn(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)
        agent.run_turn("tool_call")

        agent.run_turn("thanks")

        assert fake_server.requests[-1]["input"] == [
            USER_TURN,
            FUNCTION_CALL,
            FIRST_OUTPUT,
            {"role": "assistant", "content": TEXT},
            {"role": "user", "content": "thanks"},
        ]

    def test_undo_removes_a_whole_tool_turn(self, make_agent, streaming):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming)
        agent.run_turn("tool_call")

        assert agent.context.undo()
        assert agent.context.items == []


@pytest.mark.parametrize("prompt", [SYSTEM_PROMPT, SYSTEM_PROMPT_NO_TOOLS])
def test_the_prompt_keeps_what_the_terminal_depends_on(prompt):
    # The terminal shows LaTeX and Markdown as they are.
    assert "LaTeX" in prompt
    assert "Markdown" in prompt


def test_the_agent_needs_nothing_of_mcp():
    # A fresh interpreter: this process has imported everything already.
    import subprocess
    import sys

    code = (
        "import sys, ergane.agent; "
        "loaded = [m for m in ('ergane.mcp_client', 'ergane.mcp_hub') "
        "if m in sys.modules]; "
        "print(loaded)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"


class TestUpdateServer:
    def test_it_swaps_the_tools_and_the_notes_of_a_server(self, make_agent):
        agent = make_agent(tools=[READ_FILE], server_notes={"a": "old"})
        new = Tool.from_function(other)

        agent.update_server(["read_file"], [new], "a", "new")

        assert list(agent.tools) == [new.name]
        assert agent.server_notes == {"a": "new"}

    def test_a_server_without_notes_loses_them(self, make_agent):
        agent = make_agent(server_notes={"a": "old", "b": "kept"})

        agent.update_server([], [], "a", None)

        assert agent.server_notes == {"b": "kept"}

    def test_a_tool_of_a_name_that_is_taken_is_not_added_and_is_reported(
        self, make_agent
    ):
        agent = make_agent(tools=[READ_FILE])
        clash = Tool.from_function(read_file)
        new = Tool.from_function(other)

        skipped = agent.update_server([], [clash, new], "a", None)

        assert skipped == ["read_file"]
        assert agent.tools["read_file"] is READ_FILE
        assert list(agent.tools) == ["read_file", "other"]

    def test_a_server_that_comes_back_with_its_own_tool_is_not_skipped(
        self, make_agent
    ):
        agent = make_agent(tools=[READ_FILE])

        skipped = agent.update_server(["read_file"], [READ_FILE], "a", None)

        assert skipped == [] and list(agent.tools) == ["read_file"]

    def test_what_the_server_counted_does_not_hold_any_more(self, make_agent):
        agent = make_agent(tools=[READ_FILE])
        agent.context.counted = 1000

        agent.update_server([], [Tool.from_function(other)], "a", None)

        assert agent.context.counted is None


def test_the_prompt_with_tools_says_what_the_program_does_with_them():
    # The user is not shown what a tool returns, and a denied call comes back as text.
    assert "cannot see tool results" in SYSTEM_PROMPT
    assert "denied" in SYSTEM_PROMPT


def test_the_prompt_tells_the_model_to_show_what_it_is_asked_to_show():
    # Models took "do not paste long output" as a reason to give 6 lines of a 199-line
    # file when asked to see it; the rule that says to show it is with the tools, since
    # the user cannot see a result, and the one against pasting makes an exception.
    assert "asks to see a file" in SYSTEM_PROMPT
    assert "in full, after reading all of it" in SYSTEM_PROMPT
    for prompt in (SYSTEM_PROMPT, SYSTEM_PROMPT_NO_TOOLS):
        assert "unless they ask to see it" in prompt


def test_the_prompt_with_tools_asks_for_scratch_scripts_to_be_cleaned_up():
    # "A new temporary directory" alone made gemma4 create one in the project, so
    # it names the system's temporary location; and it says the project is no place.
    assert "scratch script" in SYSTEM_PROMPT
    assert "system's temporary location" in SYSTEM_PROMPT
    assert "never in the user's project" in SYSTEM_PROMPT
    assert "delete that directory when you are done, also when the script fails" in (
        SYSTEM_PROMPT
    )


def test_the_prompt_does_not_forbid_the_cleanup_it_asks_for():
    # Deleting the scratch directory is `rm -rf`, so that cannot be named as
    # forbidden; what is forbidden is deleting what the model did not create.
    for prompt in (SYSTEM_PROMPT, SYSTEM_PROMPT_NO_TOOLS):
        assert "rm -rf" not in prompt
        assert "do not delete anything you did not create" in prompt


def test_the_prompt_without_tools_says_there_are_none_and_does_not_talk_about_calls():
    # Models trained to call tools try to when told about them, with none offered:
    # gpt-5.5 types a call as text, gemma4 returns nothing. So it only says what is true.
    assert "You have no tools" in SYSTEM_PROMPT_NO_TOOLS
    for priming in (
        "approve",
        "denied",
        "make independent calls",
        "tool call",
        "scratch script",
        "mktemp",
    ):
        assert priming not in SYSTEM_PROMPT_NO_TOOLS


class TestStreamRetry:
    """A stream that breaks before any text is shown is sent again."""

    EARLY = "Stream ended without a final response"
    CUT = "Stream cut off: Connection error"

    def test_a_stream_that_ends_early_is_sent_again(self, make_agent, fake_server):
        agent = make_agent(STREAMING=True)
        pieces: list[str] = []
        notices: list[str] = []

        agent.run_turn("flaky", on_text=pieces.append, on_notice=notices.append)

        assert len(fake_server.requests) == 2
        assert notices == [f"{self.EARLY}; trying again (1 of 2)"]
        # The reply is there once: nothing of the first try had been shown.
        assert pieces == ["Hello", " world"]
        assert agent.context.items[-1] == {"role": "assistant", "content": TEXT}

    def test_a_reset_connection_is_sent_again(self, make_agent, fake_server):
        agent = make_agent(STREAMING=True)
        notices: list[str] = []

        agent.run_turn("flaky_reset", on_notice=notices.append)

        assert len(fake_server.requests) == 2
        assert notices == [f"{self.CUT}; trying again (1 of 2)"]
        assert agent.context.items[-1] == {"role": "assistant", "content": TEXT}

    def test_it_gives_up_after_the_retries(self, make_agent, fake_server):
        agent = make_agent(STREAMING=True)
        notices: list[str] = []

        with pytest.raises(TurnError, match=f"^{self.EARLY}$"):
            agent.run_turn("truncated", on_notice=notices.append)

        assert len(fake_server.requests) == 1 + STREAM_RETRIES
        assert notices == [
            f"{self.EARLY}; trying again (1 of 2)",
            f"{self.EARLY}; trying again (2 of 2)",
        ]
        assert agent.context.items == []  # a turn that fails is not kept

    def test_a_connection_that_keeps_resetting_ends_the_turn_as_a_cut_stream(
        self, make_agent, fake_server
    ):
        agent = make_agent(STREAMING=True)

        with pytest.raises(TurnError, match=f"^{self.CUT}$"):
            agent.run_turn("reset")

        assert len(fake_server.requests) == 1 + STREAM_RETRIES

    def test_a_stream_that_breaks_after_some_text_is_not_sent_again(
        self, make_agent, fake_server
    ):
        # The reply would appear twice, the second time perhaps different.
        agent = make_agent(STREAMING=True)
        pieces: list[str] = []
        notices: list[str] = []

        with pytest.raises(TurnError, match=f"^{self.EARLY}$"):
            agent.run_turn(
                "cut_after_text", on_text=pieces.append, on_notice=notices.append
            )

        assert len(fake_server.requests) == 1
        assert pieces == ["Hello"]
        assert notices == []

    def test_a_stream_the_server_ended_with_an_error_is_not_sent_again(
        self, make_agent, fake_server
    ):
        # The server said why; the same request would get the same answer.
        agent = make_agent(STREAMING=True)

        with pytest.raises(TurnError, match="^Stream error: server_error: "):
            agent.run_turn("error_event")

        assert len(fake_server.requests) == 1

    def test_a_round_after_tool_calls_is_sent_again_and_the_tool_runs_once(
        self, make_agent, fake_server
    ):
        # The case that lost a whole turn of reading: the request after the tool's
        # output broke. The tool had run, so only the request may be repeated.
        ran: list[str] = []

        def read_file(path: str) -> str:
            """Read a text file."""
            ran.append(path)
            return "contents"

        agent = make_agent(tools=[Tool.from_function(read_file)], STREAMING=True)
        notices: list[str] = []

        agent.run_turn("tool_call_flaky", on_notice=notices.append)

        assert ran == ["notes.txt"]
        assert len(fake_server.requests) == 3  # the call, the broken try, the retry
        assert notices == [f"{self.EARLY}; trying again (1 of 2)"]
        kinds = [item.get("type") or item.get("role") for item in agent.context.items]
        assert kinds == ["user", "function_call", "function_call_output", "assistant"]

    @pytest.mark.parametrize(
        "scenario, reason",
        [
            ("truncated", EARLY),
            ("reset", CUT),
            ("error_event", f"Stream error: server_error: {STREAM_ERROR}"),
        ],
    )
    def test_a_stream_that_keeps_breaking_after_a_round_keeps_the_work(
        self, make_agent, fake_server, scenario, reason
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=True)

        with pytest.raises(TurnError):
            agent.run_turn(f"tool_then_{scenario}")

        assert agent.context.items[1:] == [
            FUNCTION_CALL,
            FIRST_OUTPUT,
            {
                "role": "assistant",
                "content": f"[Stopped after 1 model calls: {reason}]",
            },
        ]

    def test_a_connection_that_never_started_is_left_to_the_sdk(self, make_agent):
        # The SDK already retried it; trying again here would multiply the waiting.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            closed = probe.getsockname()[1]
        client = OpenAI(
            api_key="test", base_url=f"http://127.0.0.1:{closed}/v1", max_retries=0
        )
        agent = Agent(make_config(STREAMING=True), client)
        notices: list[str] = []

        with pytest.raises(APIConnectionError):
            agent.run_turn("hi", on_notice=notices.append)

        assert notices == []

    def test_without_streaming_the_sdk_does_the_retrying(self, make_agent, fake_server):
        agent = make_agent(STREAMING=False)

        agent.run_turn("truncated")  # a plain response: nothing to break

        assert len(fake_server.requests) == 1


class TestDescribeError:
    @pytest.mark.parametrize(
        ("status", "text"),
        [
            (429, "API error 429 (code_429): message 429 The account is out of quota"),
            (
                401,
                "API error 401 (code_401): message 401 Check the API key (OPENAI_API_KEY",
            ),
            (
                404,
                "API error 404 (code_404): message 404 Check the model and base_url in config.json.",
            ),
            (500, "API error 500 (code_500): message 500"),
        ],
    )
    def test_an_api_error_gives_its_status_code_and_message_and_a_hint(
        self, make_agent, status, text
    ):
        agent = make_agent(STREAMING=False)

        with pytest.raises(Exception) as caught:
            agent.run_turn(f"status_{status}")

        assert describe_error(caught.value).startswith(text)
        assert "{" not in describe_error(caught.value)  # not the SDK's dict

    def test_a_connection_error_names_the_address_to_check(self):
        import httpx2

        exc = APIConnectionError(request=httpx2.Request("POST", "http://x"))

        assert (
            describe_error(exc)
            == "Connection error. Check base_url in config.json and the network."
        )

    def test_anything_else_gives_its_type_and_message(self):
        assert describe_error(ValueError("boom")) == "Agent error: ValueError: boom"


@pytest.mark.parametrize("streaming", [True, False])
class TestAlwaysAllow:
    def agent(self, make_agent, streaming, ran, **config):
        def read_file(path: str) -> str:
            """Read a text file."""
            ran.append(path)
            return "contents"

        return make_agent(
            tools=[Tool.from_function(read_file, read_only=False)],
            STREAMING=streaming,
            **config,
        )

    def test_always_allows_the_call_and_the_later_ones_of_that_tool(
        self, make_agent, streaming
    ):
        ran: list[str] = []
        agent = self.agent(make_agent, streaming, ran)
        asked: list[str] = []
        calls: list[tuple[str, str]] = []

        def confirm(tool: Tool, arguments: str):
            asked.append(tool.name)
            return "always"

        with pytest.raises(TurnError):
            agent.run_turn(
                "tool_loop", on_call=lambda *call: calls.append(call), confirm=confirm
            )

        assert asked == ["read_file"] and len(ran) == LOOP_ROUNDS
        assert agent.always_allowed == {"read_file"}
        # A call that runs without asking is shown as a line, like a read-only one.
        assert calls == [("read_file", '{"path": "notes.txt"}')] * (LOOP_ROUNDS - 1)

    def test_it_lasts_for_the_session_not_just_the_turn(self, make_agent, streaming):
        ran: list[str] = []
        agent = self.agent(make_agent, streaming, ran)
        agent.run_turn("tool_call", confirm=lambda tool, arguments: "always")

        agent.run_turn("tool_call")  # the default confirm denies; it is not asked

        assert len(ran) == 2

    def test_yes_allows_this_call_only(self, make_agent, streaming):
        ran: list[str] = []
        agent = self.agent(make_agent, streaming, ran)
        asked: list[str] = []

        def confirm(tool: Tool, arguments: str) -> bool:
            asked.append(tool.name)
            return True

        agent.run_turn("tool_call", confirm=confirm)
        agent.run_turn("tool_call", confirm=confirm)

        assert asked == ["read_file", "read_file"] and agent.always_allowed == set()

    def test_a_call_that_cannot_run_does_not_make_the_tool_allowed(
        self, make_agent, streaming
    ):
        ran: list[str] = []
        agent = self.agent(make_agent, streaming, ran)
        # The arguments are invalid (the fake sends "path"), so nothing is asked.
        agent.tools["read_file"] = Tool(
            "read_file",
            "d",
            {},
            lambda arguments: "Invalid arguments: x",
            lambda arguments: "",
            False,
        )

        agent.run_turn("tool_call", confirm=lambda tool, arguments: "always")

        assert agent.always_allowed == set()


@pytest.mark.parametrize("streaming", [True, False])
class TestContextEstimate:
    def test_the_count_the_server_reports_is_kept(self, make_agent, streaming):
        agent = make_agent(STREAMING=streaming)
        assert agent.last_input_tokens is None

        agent.run_turn("usage")

        assert agent.last_input_tokens == USAGE_INPUT_TOKENS

    def test_a_reply_without_a_count_leaves_the_last_one(self, make_agent, streaming):
        agent = make_agent(STREAMING=streaming)
        agent.run_turn("usage")

        agent.run_turn("hello")  # the fake sends no usage for this one

        assert agent.last_input_tokens == USAGE_INPUT_TOKENS

    def test_the_count_and_the_reply_are_kept_for_the_next_request(
        self, make_agent, streaming
    ):
        agent = make_agent(STREAMING=streaming)

        agent.run_turn("usage")

        reply = {"role": "assistant", "content": TEXT}
        assert agent.context.counted == USAGE_INPUT_TOKENS + estimate_tokens(reply)

    def test_the_count_holds_for_the_model_that_made_it(self, make_agent, streaming):
        agent = make_agent(STREAMING=streaming)
        agent.run_turn("usage")
        counted = agent.counted_tokens()
        assert counted is not None

        agent.config.model = "another-model"
        assert agent.counted_tokens() is None
        assert agent.request_tokens() != counted  # estimated again

        agent.config.model = "gpt-4o"
        assert agent.counted_tokens() == counted

    def test_a_reply_without_a_count_leaves_no_count(self, make_agent, streaming):
        agent = make_agent(STREAMING=streaming)
        agent.run_turn("usage")

        agent.run_turn("hello")

        assert agent.context.counted is None

    def test_instructions_tools_and_history_are_estimated_apart(
        self, make_agent, streaming
    ):
        bare = make_agent(STREAMING=streaming)
        with_tools = make_agent(tools=[READ_FILE], STREAMING=streaming)

        before = with_tools.context_estimate()
        with_tools.run_turn("tool_call")
        after = with_tools.context_estimate()

        assert bare.context_estimate()[1] == 0
        assert before[1] > bare.context_estimate()[1]  # a tool is a definition
        assert before[2] == 0 and after[2] > 0  # the turn is history now
        assert after[:2] == before[:2]


def test_a_restored_conversation_is_sent_with_the_next_prompt(make_agent, fake_server):
    agent = make_agent(STREAMING=False)
    agent.last_input_tokens = 99

    agent.restore(make_context(2), "NOTES")
    assert agent.last_input_tokens is None  # what was counted was another request
    agent.run_turn("next")

    sent = fake_server.requests[-1]["input"]
    assert "NOTES" in str(sent[0])
    assert [m["content"] for m in sent[1:]] == [
        "question 0",
        "answer 0",
        "question 1",
        "answer 1",
        "next",
    ]
