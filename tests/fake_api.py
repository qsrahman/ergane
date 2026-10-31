"""A minimal fake of the OpenAI Responses API for tests."""

import json
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


def final_event(scenario: str, tool_output_sent: bool = False) -> tuple[str, dict]:
    """Return the terminal event type and response for a scenario."""
    if scenario == "tool_loop" or (
        scenario in ("tool_call", "tool_call_reasoning") and not tool_output_sent
    ):
        calls = [FUNCTION_CALL]
        if scenario == "tool_call_reasoning":
            calls = [REASONING_ITEM, FUNCTION_CALL]
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
    if scenario == "empty":
        return "response.completed", make_response("completed")
    if scenario == "failed":
        return "response.failed", make_response(
            "failed",
            error={"code": "server_error", "message": "boom"},
        )
    return "response.completed", make_response("completed", [MESSAGE])


class FakeResponsesHandler(BaseHTTPRequestHandler):
    """Minimal fake of POST /v1/responses.

    The most recent user message selects the scenario: "incomplete", "failed",
    "empty" (completed with no output), "refusal", "cancelled" (text, but a
    status other than completed or incomplete), "nostatus" (a reply without a
    status, as some compatible servers send), "truncated" (a stream that ends
    without a terminal event), "error_event" or "error_event_nocode" (a stream
    `error` event, with or without a code, after the first delta, then the stream
    ends), "tool_call" (calls a tool until its output is in the input, then
    replies with text), "tool_call_reasoning" (the same, with a reasoning item
    before each call), "tool_loop" (calls a tool every time), or anything else for
    a completed response.
    """

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)  # pyright: ignore[reportAttributeAccessIssue]
        scenario = last_user_prompt(body["input"])
        event_type, response = final_event(scenario, has_tool_output(body["input"]))

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
        self.stream(scenario, event_type, response)

    def stream(self, scenario: str, event_type: str, response: dict) -> None:
        """Write a response as server-sent events, item by item."""
        sequence = 0

        def send(type_: str, **fields):
            nonlocal sequence
            event = {"type": type_, "sequence_number": sequence, **fields}
            sequence += 1
            self.wfile.write(f"event: {type_}\ndata: {json.dumps(event)}\n\n".encode())

        send("response.created", response=make_response("in_progress"))
        if scenario == "truncated":
            return
        for index, item in enumerate(response["output"]):
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
                    for delta in ("Hello", " world"):
                        send(
                            "response.output_text.delta",
                            **ids,
                            delta=delta,
                            logprobs=[],
                        )
                        if scenario.startswith("error_event"):
                            code = (
                                None
                                if scenario == "error_event_nocode"
                                else "server_error"
                            )
                            send("error", code=code, message=STREAM_ERROR)
                            return
                    send("response.output_text.done", **ids, text=TEXT, logprobs=[])
                send("response.content_part.done", **ids, part=part)
            send("response.output_item.done", output_index=index, item=item)
        send(event_type, response=response)
