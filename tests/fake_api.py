"""A minimal fake of the OpenAI Responses API for tests."""

import json
import re
import socket
import struct
import time
from http.server import BaseHTTPRequestHandler

TEXT = "Hello world"
MESSAGE = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": TEXT, "annotations": []}],
}
REFUSAL = "I can't help with that."
STREAM_ERROR = "the model crashed"
REFUSAL_MESSAGE = {
    **MESSAGE,
    "content": [{"type": "refusal", "refusal": REFUSAL}],
}
# What a model emits when it calls a tool; a reasoning model adds a reasoning item,
# which must be sent back along with the tool's output.
REASONING_ITEM = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [],
    "encrypted_content": "ENCRYPTED",
}
FUNCTION_CALL = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "read_file",
    "arguments": '{"path": "notes.txt"}',
    "status": "completed",
}
SECOND_FUNCTION_CALL = {
    **FUNCTION_CALL,
    "id": "fc_2",
    "call_id": "call_2",
    "arguments": '{"path": "todo.txt"}',
}


# What the "usage" scenario says the server counted in the request.
USAGE_INPUT_TOKENS = 1234


def make_response(status: str | None, output: list | None = None, **extra) -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-4o",
        "status": status,
        "output": output or [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "incomplete_details": None,
        "error": None,
        **extra,
    }


def todo_call(*steps: tuple[str, str]) -> dict:
    """A call of the todo tool, the one ergane offers itself, with these steps."""
    todos = [{"content": c, "status": s} for c, s in steps]
    return {
        **FUNCTION_CALL,
        "id": "fc_todo",
        "call_id": "call_todo",
        "name": "todo",
        "arguments": json.dumps({"todos": todos}),
    }


# What each tool scenario's first response contains (tool_loop uses the default).
TOOL_CALLS = {
    "tool_call": [FUNCTION_CALL],
    "tool_call_reasoning": [REASONING_ITEM, FUNCTION_CALL],
    "tool_call_parallel": [FUNCTION_CALL, SECOND_FUNCTION_CALL],
    # A read, a write, a read: the write is a barrier between the two reads.
    "tool_call_mixed": [
        FUNCTION_CALL,
        {**SECOND_FUNCTION_CALL, "name": "write_file"},
        {**SECOND_FUNCTION_CALL, "id": "fc_3", "call_id": "call_3"},
    ],
    "tool_call_flaky": [FUNCTION_CALL],
    # The todo tool: a list with a step under way, one that the tool refuses (two
    # steps in progress), and one the arguments check refuses (a status that is not).
    "tool_todo": [todo_call(("write it", "in_progress"), ("test it", "pending"))],
    "tool_todo_two_active": [todo_call(("a", "in_progress"), ("b", "in_progress"))],
    "tool_todo_bad_status": [todo_call(("a", "done"))],
}


SUMMARY_PROMPT_START = "You are summarizing a conversation"


def summary_response(body: dict) -> tuple[str, dict]:
    """The answer to the request that summarizes old turns.

    The text of the conversation picks the outcome: SUMMARY_EMPTY, SUMMARY_NOHEADINGS
    (not a summary), SUMMARY_LONG (longer than what it replaces) and
    SUMMARY_INCOMPLETE (stopped by the output cap) are the ways it goes wrong;
    SUMMARY_PLAIN and SUMMARY_UNCLOSED are good summaries without the tags, or
    without the closing one; otherwise it is a good summary that says how many user
    lines it was given.
    """
    text = body["input"][0]["content"]
    if "SUMMARY_EMPTY" in text:
        answer = "<analysis>a</analysis><summary></summary>"
    elif "SUMMARY_NOHEADINGS" in text:
        answer = "<summary>I am not able to do that.</summary>"
    elif "SUMMARY_LONG" in text:
        answer = f"<summary>## Goal\n{'x' * (len(text) + 10)}</summary>"
    elif "SUMMARY_UNCLOSED" in text:
        answer = "<summary>## Goal\nNOTES unclosed"
    elif "SUMMARY_PLAIN" in text:
        answer = "## Goal\nNOTES plain"
    else:
        count = text.count("[User]:")
        answer = (
            f"<analysis>thinking</analysis><summary>## Goal\nNOTES {count}</summary>"
        )
    message = {
        **MESSAGE,
        "content": [{"type": "output_text", "text": answer, "annotations": []}],
    }
    if "SUMMARY_INCOMPLETE" in text:
        return "response.incomplete", make_response(
            "incomplete", [message], incomplete_details={"reason": "max_output_tokens"}
        )
    return "response.completed", make_response("completed", [message])


def last_user_prompt(items: list[dict]) -> str:
    """The most recent user message in a request's input.

    Not simply the last item: after a tool call the input ends with a
    function_call_output, which has no content.
    """
    for item in reversed(items):
        if item.get("role") == "user":
            return item["content"]
    return ""


def has_tool_output(items: list[dict]) -> bool:
    """Whether a tool's output follows the most recent user message."""
    for item in reversed(items):
        if item.get("type") == "function_call_output":
            return True
        if item.get("role") == "user":
            return False
    return False


def how_it_dies(scenario: str, attempt: int, tool_output_sent: bool) -> str | None:
    """How a streamed response ends badly, if it does: "truncated" (the stream ends
    after its first event with no terminal event), "reset" (the connection is reset
    then) or "after_text" (the stream ends after its first piece of text).

    `attempt` counts how often this exact request was sent before, so that a
    scenario can break once and work when it is sent again.
    """
    if scenario in ("truncated", "reset"):
        return scenario
    if scenario == "cut_after_text":
        return "after_text"
    if scenario == "flaky" and attempt == 0:
        return "truncated"
    if scenario == "flaky_reset" and attempt == 0:
        return "reset"
    if scenario == "tool_call_flaky" and tool_output_sent and attempt == 0:
        return "truncated"
    return None


# The errors a stream can end with before any of the reply, as (code, message).
STREAM_FAILURES = {
    "busy": ("server_error", "The server is overloaded"),
    "overflow": ("context_length_exceeded", "Your input exceeds the context window"),
    "bad": ("invalid_prompt", "boom"),
}
STREAM_FAILURE = re.compile(r"(event|failed|sdk)_(busy|overflow|bad)(_once)?")


def final_event(scenario: str, tool_output_sent: bool = False) -> tuple[str, dict]:
    """Return the terminal event type and response for a scenario."""
    if scenario == "tool_loop" or (scenario in TOOL_CALLS and not tool_output_sent):
        calls = TOOL_CALLS.get(scenario, [FUNCTION_CALL])
        return "response.completed", make_response("completed", calls)
    if scenario == "incomplete":
        return "response.incomplete", make_response(
            "incomplete",
            [MESSAGE],
            incomplete_details={"reason": "max_output_tokens"},
        )
    if scenario == "refusal":
        return "response.completed", make_response("completed", [REFUSAL_MESSAGE])
    if scenario == "cancelled":
        return "response.completed", make_response("cancelled", [MESSAGE])
    if scenario == "nostatus":
        return "response.completed", make_response(None, [MESSAGE])
    if scenario == "usage":
        usage = {
            "input_tokens": USAGE_INPUT_TOKENS,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": USAGE_INPUT_TOKENS + 5,
        }
        return "response.completed", make_response("completed", [MESSAGE], usage=usage)
    if scenario == "empty":
        return "response.completed", make_response("completed")
    if scenario == "null_output":  # some compatible servers send `"output": null`
        return "response.completed", {**make_response("completed"), "output": None}
    if scenario == "failed":
        return "response.failed", make_response(
            "failed",
            error={"code": "invalid_prompt", "message": "boom"},
        )
    return "response.completed", make_response("completed", [MESSAGE])


class FakeResponsesHandler(BaseHTTPRequestHandler):
    """Minimal fake of POST /v1/responses.

    The most recent user message selects the scenario: "incomplete", "failed",
    "empty" (completed with no output), "null_output" (the same with `"output": null`), "empty_once" (the same, only for the first time a request comes), "refusal", "cancelled" (text, but a
    status other than completed or incomplete), "nostatus" (a reply without a
    status, as some compatible servers send), "truncated" (a stream that ends
    without a terminal event), "reset" (the connection is reset instead),
    "flaky" and "flaky_reset" (the first stream breaks that way, the same request
    sent again works), "cut_after_text" (the stream ends after some text),
    "tool_call_flaky" (like "tool_call", but the stream after the tool's output
    breaks once), "error_event" or "error_event_nocode" (a stream
    `error` event, with or without a code, after the first delta, then the stream
    ends), "HOW_WHAT" or "HOW_WHAT_once" (a stream that ends in an error before any
    text: HOW is "event" for an `error` event, "failed" for `response.failed` or "sdk"
    for the form the SDK raises itself; WHAT is "busy", "overflow" or "bad"; with
    "_once" only the first time), "tool_call" (calls a tool until its output is in the input, then
    replies with text), "tool_call_reasoning" (the same, with a reasoning item
    before each call), "tool_call_parallel" (two calls in one response), "tool_call_mixed" (a call to
    read_file, one to write_file, another to read_file),
    "tool_loop" (calls a tool every time), "tool_loop_N" (calls a tool until N outputs are in the input, then replies with text), "http_error" (a 400 response), "usage" (a reply that carries a token count), "overflow_once" and "overflow" (a request refused as too long, the first time or
    every time), "status_N" (an error response with
    status N and an OpenAI-style body),
    "picture_error" (a refusal for pictures, whatever the request holds), "novision_X" (scenario X, but a request that carries a picture is refused, as
    by a model without vision), "tool_todo", "tool_todo_two_active" and "tool_todo_bad_status" (a call of the todo tool, then text), "tool_then_X" (a tool call, then the scenario X once the tool's output is
    in the input, such as "tool_then_failed"), or anything else for a completed
    response.
    """

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        """The list of models: `server.models`, a list of names; unset, two names;
        None, a server that has no list (404); "broken", one that fails (500). Each
        request is counted in `server.listed`."""
        self.server.listed = getattr(self.server, "listed", 0) + 1  # pyright: ignore[reportAttributeAccessIssue]
        names = getattr(self.server, "models", ["gpt-4o", "gpt-4o-mini"])
        if names == "broken":
            status, body = 500, {"error": {"message": "broken"}}
        elif self.path != "/v1/models" or names is None:
            status, body = 404, {"error": {"message": "no such page"}}
        else:
            data = [
                {"id": name, "object": "model", "created": 0, "owned_by": "x"}
                for name in names
            ]
            status, body = 200, {"object": "list", "data": data}
        reply = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)  # pyright: ignore[reportAttributeAccessIssue]
        scenario = last_user_prompt(body["input"])
        tool_output_sent = has_tool_output(body["input"])
        if scenario.startswith("tool_then_"):
            # A tool call first, then whatever follows the prefix.
            scenario = (
                scenario.removeprefix("tool_then_") if tool_output_sent else "tool_call"
            )
        if match := re.fullmatch(r"tool_loop_(\d+)", scenario):
            # A tool call until the input holds that many outputs, then a reply.
            done = sum(
                item.get("type") == "function_call_output" for item in body["input"]
            )
            scenario = "tool_loop" if done < int(match[1]) else "hello"
        if scenario == "empty_once":
            seen = any(
                earlier["input"] == body["input"]
                for earlier in self.server.requests[:-1]  # pyright: ignore[reportAttributeAccessIssue]
            )
            scenario = "hello" if seen else "empty"  # answered the second time
        if scenario.startswith("novision_"):
            # A model that takes no pictures: a request that carries one is refused.
            scenario = scenario.removeprefix("novision_")
            if "input_image" in json.dumps(body["input"]):
                data = json.dumps(
                    {
                        "error": {
                            "message": "Multimodal data provided, but model does "
                            "not support multimodal requests"
                        }
                    }
                ).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        if scenario == "picture_error":  # refused for pictures, though there are none
            data = json.dumps(
                {"error": {"message": "this model does not support image input"}}
            ).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if scenario.startswith("status_"):
            code = int(scenario.removeprefix("status_"))
            data = json.dumps(
                {"error": {"message": f"message {code}", "code": f"code_{code}"}}
            ).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        summarizing = body.get("instructions", "").startswith(SUMMARY_PROMPT_START)
        if scenario == "overflow" or (
            scenario == "overflow_once" and not getattr(self.server, "refused", 0)
        ):
            self.server.refused = 1  # pyright: ignore[reportAttributeAccessIssue]
            data = json.dumps(
                {
                    "error": {
                        "message": "This model's maximum context length is 8192 tokens.",
                        "code": "context_length_exceeded",
                    }
                }
            ).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if scenario == "http_error" or (
            summarizing and "SUMMARY_FAIL" in body["input"][0]["content"]
        ):
            data = json.dumps({"error": {"message": "bad request"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if summarizing:
            event_type, response = summary_response(body)
        else:
            event_type, response = final_event(scenario, tool_output_sent)
        # How often this exact request came before: a retry sends the same input.
        attempt = sum(
            1
            for earlier in self.server.requests[:-1]  # pyright: ignore[reportAttributeAccessIssue]
            if earlier["input"] == body["input"]
        )

        if not body.get("stream"):
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if match := STREAM_FAILURE.fullmatch(scenario):
            failed = self.server.__dict__.setdefault("failed_once", set())
            if not (match[3] and scenario in failed):
                failed.add(scenario)
                self.fail_stream(match[1], *STREAM_FAILURES[match[2]])
                return
        self.stream(
            scenario,
            event_type,
            response,
            how_it_dies(scenario, attempt, tool_output_sent),
        )

    def reset(self) -> None:
        """Reset the connection (a TCP RST) instead of closing it."""
        time.sleep(0.2)  # let the client read what was sent before the reset
        self.connection.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
        )
        self.connection.close()

    def fail_stream(self, how: str, code: str, message: str) -> None:
        """A stream that starts and then ends in an error, with no text."""
        error = {"code": code, "message": message}
        events: list[tuple[str, dict]] = [
            ("response.created", {"response": make_response("in_progress")})
        ]
        if how == "event":
            events.append(("error", error))
        elif how == "failed":
            response = make_response("failed", error=error)
            events.append(("response.failed", {"response": response}))
        for sequence, (type_, fields) in enumerate(events):
            event = {"type": type_, "sequence_number": sequence, **fields}
            self.wfile.write(f"event: {type_}\ndata: {json.dumps(event)}\n\n".encode())
        if how == "sdk":
            data = json.dumps({"error": error})
            self.wfile.write(f"event: error\ndata: {data}\n\n".encode())

    def stream(
        self, scenario: str, event_type: str, response: dict, dies: str | None
    ) -> None:
        """Write a response as server-sent events, item by item."""
        sequence = 0

        def send(type_: str, **fields):
            nonlocal sequence
            event = {"type": type_, "sequence_number": sequence, **fields}
            sequence += 1
            self.wfile.write(f"event: {type_}\ndata: {json.dumps(event)}\n\n".encode())

        send("response.created", response=make_response("in_progress"))
        if dies == "truncated":
            return
        if dies == "reset":
            self.reset()
            return
        for index, item in enumerate(response["output"] or []):
            ids = {"item_id": item["id"], "output_index": index}
            if item["type"] == "function_call":
                started = {**item, "arguments": "", "status": "in_progress"}
                send("response.output_item.added", output_index=index, item=started)
                send(
                    "response.function_call_arguments.delta",
                    **ids,
                    delta=item["arguments"],
                )
                send(
                    "response.function_call_arguments.done",
                    **ids,
                    arguments=item["arguments"],
                )
            elif item["type"] == "reasoning":
                started = {**item, "status": "in_progress"}
                send("response.output_item.added", output_index=index, item=started)
            else:
                part = item["content"][0]
                ids["content_index"] = 0
                started = {**item, "status": "in_progress", "content": []}
                send("response.output_item.added", output_index=index, item=started)
                if part["type"] == "refusal":
                    send(
                        "response.content_part.added",
                        **ids,
                        part={**part, "refusal": ""},
                    )
                    send("response.refusal.delta", **ids, delta=REFUSAL)
                    send("response.refusal.done", **ids, refusal=REFUSAL)
                else:
                    send(
                        "response.content_part.added", **ids, part={**part, "text": ""}
                    )
                    half = len(part["text"]) // 2
                    for delta in (part["text"][:half], part["text"][half:]):
                        send(
                            "response.output_text.delta",
                            **ids,
                            delta=delta,
                            logprobs=[],
                        )
                        if dies == "after_text":
                            return
                        if scenario.startswith("error_event"):
                            code = (
                                None
                                if scenario == "error_event_nocode"
                                else "server_error"
                            )
                            send("error", code=code, message=STREAM_ERROR)
                            return
                    send(
                        "response.output_text.done",
                        **ids,
                        text=part["text"],
                        logprobs=[],
                    )
                send("response.content_part.done", **ids, part=part)
            send("response.output_item.done", output_index=index, item=item)
        send(event_type, response=response)
