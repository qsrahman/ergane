import re
from datetime import date

import pytest
from openai.types.responses import Response, ResponseInputParam

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
from muagent.agent import (
    NOTICE_CHARS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_NO_TOOLS,
    Agent,
    TurnError,
    _one_line,
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
        notices: list[str] = []

        agent.run_turn("tool_call_parallel", on_notice=notices.append)

        # Only the calls: their results go to the model, not to the screen.
        assert notices == [
            'read_file({"path": "notes.txt"})',
            'read_file({"path": "todo.txt"})',
        ]

    def test_a_call_that_is_asked_about_gets_no_line_of_its_own(
        self, make_agent, streaming
    ):
        # `confirm` shows the user the call it asks about; a second line would repeat it.
        agent = make_agent(
            tools=[Tool.from_function(read_file, read_only=False)], STREAMING=streaming
        )
        notices: list[str] = []
        asked: list[str] = []

        def confirm(tool: Tool, arguments: str) -> bool:
            asked.append(arguments)
            return False

        agent.run_turn("tool_call", on_notice=notices.append, confirm=confirm)

        assert notices == []
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
        notices: list[str] = []

        agent.run_turn("tool_call", on_notice=notices.append)

        assert notices == ['read_file({"path": "notes.txt"})']

    def test_a_call_to_an_unknown_tool_is_shown(self, make_agent, streaming):
        agent = make_agent(STREAMING=streaming)
        notices: list[str] = []

        agent.run_turn("tool_call", on_notice=notices.append)

        assert notices == ['read_file({"path": "notes.txt"})']

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


@pytest.mark.parametrize("prompt", [SYSTEM_PROMPT, SYSTEM_PROMPT_NO_TOOLS])
def test_the_prompt_keeps_what_the_terminal_depends_on(prompt):
    # The terminal shows LaTeX and Markdown as they are.
    assert "LaTeX" in prompt
    assert "Markdown" in prompt


def test_the_prompt_with_tools_says_what_the_program_does_with_them():
    # The user is not shown what a tool returns, and a denied call comes back as text.
    assert "cannot see tool results" in SYSTEM_PROMPT
    assert "denied" in SYSTEM_PROMPT


def test_the_prompt_without_tools_says_there_are_none_and_does_not_talk_about_calls():
    # Models trained to call tools try to when told about them, with none offered:
    # gpt-5.5 types a call as text, gemma4 returns nothing. So it only says what is true.
    assert "You have no tools" in SYSTEM_PROMPT_NO_TOOLS
    for priming in ("approve", "denied", "make independent calls", "tool call"):
        assert priming not in SYSTEM_PROMPT_NO_TOOLS


def test_a_long_call_is_flattened_and_cut_to_one_line_for_its_notice():
    line = _one_line('read_file({"path": "' + "a b\n" * 100 + '"})')

    assert "\n" not in line
    assert line.endswith("…")
    assert len(line) == NOTICE_CHARS
