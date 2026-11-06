import re
import socket
from datetime import date

import pytest
from openai import APIConnectionError, BadRequestError, OpenAI
from openai.types.responses import Response, ResponseInputParam

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
from helpers import make_config, make_context
from muagent.agent import (
    COMPACT_FAILURES,
    COMPACT_PROMPT,
    DROPPED_OUTPUT,
    PRUNE_CHARS,
    STREAM_RETRIES,
    SUMMARY_MAX_TOKENS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_NO_TOOLS,
    Agent,
    TurnError,
    describe_error,
    is_context_overflow,
)
from muagent.context import estimate_tokens
from muagent.tools import MAX_OUTPUT_CHARS, Tool


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

    assert Agent._response_status(calls) is None
    assert Agent._response_status(empty) == "No text in response"


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
USER_TURN = {"role": "user", "content": "tool_call"}
FIRST_OUTPUT = {
    "type": "function_call_output",
    "call_id": "call_1",
    "output": "contents of notes.txt",
}


def tool_with(function) -> Tool:
    """A tool named read_file (the one the fake API calls) that runs `function`."""
    function.__name__ = "read_file"
    return Tool.from_function(function)


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
        with pytest.raises(ValueError, match="unique"):
            make_agent(tools=[READ_FILE, READ_FILE], STREAMING=streaming)

    def test_the_tools_are_offered_in_every_request(
        self, make_agent, fake_server, streaming
    ):
        make_agent(tools=[READ_FILE], STREAMING=streaming).run_turn("tool_call")

        assert [request["tools"] for request in fake_server.requests] == [
            [READ_FILE.param()]
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
        ("tools", "available"), [([], "none"), ([Tool.from_function(other)], "other")]
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
            MAX_TURNS=4,
        )
        notices: list[str] = []

        with pytest.raises(TurnError, match="^Stopped after 4 model calls"):
            agent.run_turn("tool_loop", on_notice=notices.append)

        def outputs(request: dict) -> list[str]:
            return [
                item["output"]
                for item in request["input"]
                if item.get("type") == "function_call_output"
            ]

        # Each request has the latest result in full and the earlier ones dropped.
        assert len(fake_server.requests) == 4
        last = outputs(fake_server.requests[3])
        assert last[:2] == [DROPPED_OUTPUT, DROPPED_OUTPUT]
        assert last[2] == "x" * MAX_OUTPUT_CHARS
        assert notices.count("Context full: older tool results dropped") == 3
        # Every call still has an output, so the kept turn is valid history.
        kept = [i for i in agent.context.items if i.get("type") == "function_call"]
        assert len(kept) == len(outputs({"input": agent.context.items})) == 4

    def test_a_result_that_alone_outgrows_the_context_budget_stops_the_turn(
        self, make_agent, fake_server, streaming
    ):
        def huge(path: str) -> str:
            """Read a text file."""
            return "x" * MAX_OUTPUT_CHARS  # about 5,000 estimated tokens

        agent = make_agent(
            tools=[tool_with(huge)], STREAMING=streaming, MAX_CONTEXT_TOKENS=3000
        )

        with pytest.raises(TurnError, match="grew past the context budget"):
            agent.run_turn("tool_loop")

        # Nothing older to drop, and the latest result is what the model must read.
        assert len(fake_server.requests) == 1
        assert agent.context.items == []

    def test_a_model_that_never_stops_calling_tools_is_stopped_and_the_turn_kept(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming, MAX_TURNS=3)

        with pytest.raises(TurnError, match="^Stopped after 3 model calls"):
            agent.run_turn("tool_loop")

        assert len(fake_server.requests) == 3
        items = agent.context.items
        # The question, three calls with their outputs, and a closing note: the model
        # still knows what was asked and what was done when the user says to go on.
        assert len(items) == 8
        assert items[0] == {"role": "user", "content": "tool_loop"}
        assert [i.get("type") for i in items[1:7]] == [
            "function_call",
            "function_call_output",
        ] * 3
        assert items[7] == {
            "role": "assistant",
            "content": "[Stopped after 3 model calls without a final answer]",
        }

    def test_a_turn_stopped_by_the_cap_can_be_continued_or_retried(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(tools=[READ_FILE], STREAMING=streaming, MAX_TURNS=2)
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
            "content": "[Stopped after 2 model calls without a final answer]",
        }

    @pytest.mark.parametrize(
        "scenario, reason",
        [
            ("failed", "Response failed: server_error: boom"),
            ("empty", "No text in response"),
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

    def test_ctrl_c_while_a_tool_runs_keeps_the_call_with_an_output_that_says_so(
        self, make_agent, fake_server, streaming
    ):
        def interrupted(path: str) -> str:
            """Read a text file."""
            raise KeyboardInterrupt

        agent = make_agent(tools=[tool_with(interrupted)], STREAMING=streaming)

        with pytest.raises(KeyboardInterrupt):
            agent.run_turn("tool_call")

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
        calls = []

        def second_is_interrupted(path: str) -> str:
            """Read a text file."""
            calls.append(path)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return f"contents of {path}"

        agent = make_agent(
            tools=[tool_with(second_is_interrupted)], STREAMING=streaming
        )

        with pytest.raises(KeyboardInterrupt):
            agent.run_turn("tool_call_parallel")

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


def test_the_prompt_with_tools_says_what_the_program_does_with_them():
    # The user is not shown what a tool returns, and a denied call comes back as text.
    assert "cannot see tool results" in SYSTEM_PROMPT
    assert "denied" in SYSTEM_PROMPT


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
            (401, "API error 401 (code_401): message 401 Check OPENAI_API_KEY."),
            (
                404,
                "API error 404 (code_404): message 404 Check OPENAI_MODEL and OPENAI_BASE_URL.",
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
            == "Connection error. Check OPENAI_BASE_URL and the network."
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
        agent = self.agent(make_agent, streaming, ran, MAX_TURNS=3)
        asked: list[str] = []
        calls: list[tuple[str, str]] = []

        def confirm(tool: Tool, arguments: str):
            asked.append(tool.name)
            return "always"

        with pytest.raises(TurnError):
            agent.run_turn(
                "tool_loop", on_call=lambda *call: calls.append(call), confirm=confirm
            )

        assert asked == ["read_file"] and len(ran) == 3
        assert agent.always_allowed == {"read_file"}
        # A call that runs without asking is shown as a line, like a read-only one.
        assert calls == [("read_file", '{"path": "notes.txt"}')] * 2

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


def old_turns(count: int = 10, marker: str = "") -> list:
    """`count` turns of about 200 tokens each; `marker` goes into the first prompt."""
    items = []
    for i in range(count):
        items.append(
            {"role": "user", "content": f"q{i} {marker if i == 0 else ''}" + "x" * 400}
        )
        items.append({"role": "assistant", "content": f"a{i} " + "y" * 400})
    return items


def summarizing(request: dict) -> bool:
    return request["instructions"].startswith(COMPACT_PROMPT[:20])


@pytest.mark.parametrize("streaming", [True, False])
class TestCompaction:
    """About 700 tokens of instructions, 4,000 of budget, summaries from 3,000."""

    def agent(self, make_agent, streaming, history=None, **config):
        config = {"MAX_CONTEXT_TOKENS": 3600, "KEEP_RECENT_TOKENS": 500, **config}
        agent = make_agent(STREAMING=streaming, **config)
        agent.context.items = old_turns() if history is None else history
        return agent

    def test_a_long_history_is_summarized_before_the_turn(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming)
        notices: list[str] = []

        agent.run_turn("go", on_notice=notices.append)

        assert notices[0] == "Summarizing earlier messages…"
        assert notices[1].startswith("Summarized 8 earlier turns")
        summary_request, turn_request = fake_server.requests
        assert summarizing(summary_request) and not summarizing(turn_request)
        assert "tools" not in summary_request
        assert summary_request["max_output_tokens"] == SUMMARY_MAX_TOKENS
        # The turn is sent with the summary first, then the newest turns kept as they were.
        first = turn_request["input"][0]
        assert first["role"] == "user" and "NOTES 8" in first["content"]
        assert turn_request["input"][1]["content"].startswith("q8")
        assert agent.context.summary == "## Goal\nNOTES 8"
        assert agent.context.items[0]["content"].startswith("q8")

    def test_the_summary_is_asked_for_as_labelled_text_with_tool_results_cut(
        self, make_agent, fake_server, streaming
    ):
        history = old_turns(10)
        history[1:1] = [  # the first turn used a tool
            {
                "type": "function_call",
                "call_id": "c",
                "name": "look",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "c", "output": "o" * 5000},
        ]
        agent = self.agent(make_agent, streaming, history)

        agent.run_turn("go")

        text = fake_server.requests[0]["input"][0]["content"]
        assert text.startswith("Conversation:\n[User]: q0")
        assert "[Tool call]: look({})" in text
        assert "[Tool result]: " + "o" * 2000 + " [... 3,000 more characters]" in text
        assert "[Assistant]: a0" in text

    def test_a_history_under_the_trigger_is_left_alone(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming, old_turns(3))
        notices: list[str] = []

        agent.run_turn("go", on_notice=notices.append)

        assert notices == [] and len(fake_server.requests) == 1
        assert agent.context.summary is None

    def test_with_summaries_off_the_history_is_sent_as_it_is_while_it_fits(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming, AUTO_COMPACT=False)
        notices: list[str] = []

        agent.run_turn("go", on_notice=notices.append)

        assert notices == [] and len(fake_server.requests) == 1
        assert len(fake_server.requests[0]["input"]) == 21
        assert agent.context.summary is None

    def test_a_summary_that_fails_does_not_stop_the_turn(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming, old_turns(10, "SUMMARY_FAIL"))
        notices: list[str] = []

        agent.run_turn("go", on_notice=notices.append)

        assert notices[0] == "Summarizing earlier messages…"
        assert notices[1].startswith(
            "Could not summarize earlier messages: API error 400"
        )
        assert agent.context.summary is None and agent.compact_failures == 1
        assert agent.context.items[-1] == {"role": "assistant", "content": TEXT}
        assert len(agent.context.items) == 22  # the history is whole

    def test_what_does_not_fit_is_still_dropped_when_the_summary_fails(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(
            make_agent,
            streaming,
            old_turns(10, "SUMMARY_FAIL"),
            MAX_CONTEXT_TOKENS=1800,
        )
        notices: list[str] = []

        agent.run_turn("go", on_notice=notices.append)

        assert notices[-1] == "Context trimmed: oldest turns dropped"
        assert len(fake_server.requests[-1]["input"]) < 21

    @pytest.mark.parametrize(
        ("marker", "reason"),
        [
            ("SUMMARY_EMPTY", "the summary is empty"),
            ("SUMMARY_NOHEADINGS", "it has no headings"),
            ("SUMMARY_LONG", "longer than the text it replaces"),
            ("SUMMARY_INCOMPLETE", "Response incomplete: max_output_tokens"),
        ],
    )
    def test_a_summary_that_is_no_good_is_not_kept(
        self, make_agent, streaming, marker, reason
    ):
        agent = self.agent(make_agent, streaming, old_turns(10, marker))
        before = list(agent.context.items)

        with pytest.raises(TurnError, match=reason):
            agent.compact(lambda text: None)

        assert agent.context.items == before and agent.context.summary is None

    @pytest.mark.parametrize(
        ("marker", "summary"),
        [
            ("SUMMARY_PLAIN", "## Goal\nNOTES plain"),
            ("SUMMARY_UNCLOSED", "## Goal\nNOTES unclosed"),
        ],
    )
    def test_a_good_summary_without_the_tags_is_kept(
        self, make_agent, streaming, marker, summary
    ):
        # gemma4 writes the headings but not the <summary> tags.
        agent = self.agent(make_agent, streaming, old_turns(10, marker))

        assert agent.compact(lambda text: None)

        assert agent.context.summary == summary

    def test_summaries_pause_after_repeated_failures_and_a_manual_one_resumes_them(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming, old_turns(10, "SUMMARY_FAIL"))
        for _ in range(COMPACT_FAILURES + 2):
            agent.context.items = old_turns(10, "SUMMARY_FAIL")
            agent.run_turn("go")

        tried = [r for r in fake_server.requests if summarizing(r)]
        assert len(tried) == COMPACT_FAILURES  # then none by itself

        agent.context.items = old_turns(10)
        assert agent.compact(lambda text: None)
        assert agent.compact_failures == 0

    def test_a_second_summary_updates_the_first(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming)
        agent.compact(lambda text: None)
        agent.context.items = [*agent.context.items, *old_turns(8)]

        agent.compact(lambda text: None)

        second = fake_server.requests[1]["input"][0]["content"]
        assert second.startswith("Previous summary:\n## Goal\nNOTES 8\n\nConversation:")
        # The new one replaces the old, which was an input to it: one summary, not two.
        assert (agent.context.summary or "").startswith("## Goal\nNOTES ")
        assert agent.context.summary.count("## Goal") == 1  # type: ignore[union-attr]

    def test_a_focus_is_passed_on(self, make_agent, fake_server, streaming):
        agent = self.agent(make_agent, streaming)

        agent.compact(lambda text: None, focus="the schema")

        assert fake_server.requests[0]["input"][0]["content"].endswith(
            "Focus: the schema"
        )

    def test_a_reasoning_model_is_asked_to_think_little_for_the_summary(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming, REASONING_EFFORT="high")

        agent.run_turn("go")

        summary_request, turn_request = fake_server.requests
        assert summary_request["reasoning"] == {"effort": "low"}
        assert turn_request["reasoning"] == {"effort": "high"}

    def test_the_summary_is_not_part_of_the_instructions_of_the_chat(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming)

        agent.run_turn("go")

        assert "NOTES" not in fake_server.requests[1]["instructions"]

    def test_ctrl_c_after_the_summary_keeps_it_and_not_the_turn(
        self, make_agent, streaming
    ):
        agent = self.agent(make_agent, streaming)
        before = list(agent.context.items)

        def interrupt(text: str) -> None:
            if text.startswith("Summarized"):
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            agent.run_turn("go", on_notice=interrupt)

        # The interrupt came after the summary was made, so it was kept; the turn was not.
        assert agent.context.summary == "## Goal\nNOTES 8"
        assert all(item["content"] != "go" for item in agent.context.items)
        assert agent.context.items != before


@pytest.mark.parametrize("streaming", [True, False])
class TestTooLong:
    def test_a_request_the_server_refuses_as_too_long_is_summarized_and_sent_again(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(
            STREAMING=streaming, MAX_CONTEXT_TOKENS=4000, AUTO_COMPACT=True,
            COMPACT_AT=1, KEEP_RECENT_TOKENS=500,
        )  # fmt: skip
        agent.context.items = old_turns(10)[:12]  # under the trigger: nothing yet
        notices: list[str] = []

        agent.run_turn("overflow_once", on_notice=notices.append)

        kinds = ["summary" if summarizing(r) else "chat" for r in fake_server.requests]
        assert kinds == ["chat", "summary", "chat"]  # refused, summarized, sent again
        assert (
            "The request was too long for the model; trying again, shorter" in notices
        )
        assert agent.context.items[-1] == {"role": "assistant", "content": TEXT}

    def test_after_a_tool_the_results_of_earlier_rounds_are_shortened(
        self, make_agent, fake_server, streaming
    ):
        def huge(path: str) -> str:
            """Read a text file."""
            return "head" + "x" * 20000 + "tail"

        agent = make_agent(tools=[tool_with(huge)], STREAMING=streaming)
        notices: list[str] = []

        # Round 1 calls the tool, round 2 is refused, round 3 is the same request,
        # but the result is shortened.
        agent.run_turn("tool_then_overflow_once", on_notice=notices.append)

        assert len(fake_server.requests) == 3
        sent = fake_server.requests[2]["input"][-1]["output"]
        assert sent.startswith("head") and sent.endswith("tail")
        assert len(sent) < PRUNE_CHARS

    def test_nothing_to_shorten_is_the_servers_error(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(STREAMING=streaming)

        with pytest.raises(Exception) as caught:
            agent.run_turn("overflow_once")

        assert is_context_overflow(caught.value)
        assert "Lower max_context_tokens, or use /compact." in describe_error(
            caught.value
        )
        assert len(fake_server.requests) == 1

    def test_it_is_tried_once_in_a_turn(self, make_agent, fake_server, streaming):
        agent = make_agent(
            STREAMING=streaming, MAX_CONTEXT_TOKENS=4000, COMPACT_AT=1,
            KEEP_RECENT_TOKENS=500,
        )  # fmt: skip
        agent.context.items = old_turns(10)[:12]

        with pytest.raises(Exception) as caught:
            agent.run_turn("overflow")

        assert is_context_overflow(caught.value)
        kinds = ["summary" if summarizing(r) else "chat" for r in fake_server.requests]
        assert kinds == ["chat", "summary", "chat"]  # and no more


class TestOverflowDetection:
    def error(self, status: int, message: str, code: str | None = None):
        import httpx2
        from openai import APIStatusError

        response = httpx2.Response(status, request=httpx2.Request("POST", "http://x"))
        return APIStatusError(
            message,
            response=response,
            body={"message": message, "code": code},
        )

    @pytest.mark.parametrize(
        ("status", "message", "code"),
        [
            (400, "anything", "context_length_exceeded"),
            (400, "This model's maximum context length is 8192 tokens", None),
            (400, "prompt is too long: 9000 tokens", None),
            (413, "request too large: token limit exceeded", None),
            (400, "input exceeds the context window", None),
        ],
    )
    def test_the_ways_servers_say_it(self, status, message, code):
        assert is_context_overflow(self.error(status, message, code))

    @pytest.mark.parametrize(
        ("status", "message"),
        [(400, "invalid tool schema"), (429, "context length"), (500, "too long")],
    )
    def test_other_errors_are_not_it(self, status, message):
        assert not is_context_overflow(self.error(status, message))


@pytest.mark.parametrize("streaming", [True, False])
class TestShorteningResults:
    def test_a_long_result_keeps_its_start_and_end_and_says_what_is_hidden(
        self, make_agent, fake_server, streaming
    ):
        def huge(path: str) -> str:
            """Read a text file."""
            return "head" + "x" * 20000 + "tail"

        agent = make_agent(
            tools=[tool_with(huge)], STREAMING=streaming,
            MAX_CONTEXT_TOKENS=9000, MAX_TURNS=3, AUTO_COMPACT=False,
        )  # fmt: skip

        with pytest.raises(TurnError, match="^Stopped after 3"):
            agent.run_turn("tool_loop")

        outputs = [
            item["output"]
            for item in fake_server.requests[2]["input"]
            if item.get("type") == "function_call_output"
        ]
        assert outputs[0].startswith("head") and outputs[0].endswith("tail")
        assert "characters hidden; call the tool again" in outputs[0]
        assert len(outputs[0]) < PRUNE_CHARS
        assert outputs[-1].startswith("head") and "hidden" not in outputs[-1]
