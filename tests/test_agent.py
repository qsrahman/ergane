import re

import pytest
from openai.types.responses import Response

from fake_api import (
    FUNCTION_CALL,
    REASONING_ITEM,
    REFUSAL,
    SECOND_FUNCTION_CALL,
    STREAM_ERROR,
    TEXT,
    make_response,
)
from helpers import make_context
from muagent.agent import NOTICE_CHARS, Agent, TurnError
from muagent.tools import MAX_OUTPUT_CHARS, Tool


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


def test_a_response_that_only_calls_a_tool_has_no_status_problem():
    # The two checks are separate so a tool loop can tell an error from a tool call:
    # having no text is a problem only for the response that ends the turn.
    response = Response.model_validate(make_response("completed", [FUNCTION_CALL]))

    assert Agent._status_problem(response) is None
    assert Agent._text_problem(response) == "No text in response"


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
        assert output == "Invalid arguments: lines: Field required"

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
        notices: list[str] = []

        agent.run_turn("tool_call_parallel", on_notice=notices.append)

        assert notices == [
            'read_file({"path": "notes.txt"})',
            "→ contents of notes.txt",
            'read_file({"path": "todo.txt"})',
            "→ contents of todo.txt",
        ]

    def test_a_failed_call_shows_its_error_in_the_notice(self, make_agent, streaming):
        def failing(path: str) -> str:
            """Read a text file."""
            raise FileNotFoundError(f"no such file: {path}")

        agent = make_agent(tools=[tool_with(failing)], STREAMING=streaming)
        notices: list[str] = []

        agent.run_turn("tool_call", on_notice=notices.append)

        assert notices[-1] == "→ Error: FileNotFoundError: no such file: notes.txt"

    def test_a_long_result_is_shortened_to_one_line_in_its_notice(
        self, make_agent, streaming
    ):
        def long(path: str) -> str:
            """Read a text file."""
            return "line one\n" + "word " * 100

        agent = make_agent(tools=[tool_with(long)], STREAMING=streaming)
        notices: list[str] = []

        agent.run_turn("tool_call", on_notice=notices.append)

        result = notices[-1]
        assert result.startswith("→ line one word word")
        assert result.endswith("…")
        assert "\n" not in result
        assert len(result) == len("→ ") + NOTICE_CHARS

    def test_a_turn_that_outgrows_the_context_budget_is_stopped_and_not_saved(
        self, make_agent, fake_server, streaming
    ):
        def huge(path: str) -> str:
            """Read a text file."""
            return "x" * MAX_OUTPUT_CHARS  # about 5,000 estimated tokens

        agent = make_agent(
            tools=[tool_with(huge)], STREAMING=streaming, MAX_CONTEXT_TOKENS=6000
        )

        with pytest.raises(TurnError, match="grew past the context budget"):
            agent.run_turn("tool_loop")

        # The first result fits; the second pushes the turn over, so no third request.
        assert len(fake_server.requests) == 2
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

    def test_ctrl_c_while_a_tool_runs_saves_nothing(self, make_agent, streaming):
        def interrupted(path: str) -> str:
            """Read a text file."""
            raise KeyboardInterrupt

        agent = make_agent(tools=[tool_with(interrupted)], STREAMING=streaming)

        with pytest.raises(KeyboardInterrupt):
            agent.run_turn("tool_call")

        assert agent.context.items == []

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
