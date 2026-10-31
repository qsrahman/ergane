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


def final_event(scenario: str) -> tuple[str, dict]:
    """Return the terminal event type and response for a scenario."""
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

    The last input item's content selects the scenario: "incomplete",
    "failed", "empty" (completed with no output), "refusal", "cancelled" (a
    response with text but a status other than completed or incomplete), "nostatus" (a
    reply without a status, as some compatible servers send), "truncated" (a
    stream that ends without a terminal event), "error_event" or
    "error_event_nocode" (a stream `error` event, with or without a code, after
    the first delta, then the stream ends), or anything else for a completed
    response.
    """

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)  # pyright: ignore[reportAttributeAccessIssue]
        scenario = body["input"][-1]["content"]
        event_type, response = final_event(scenario)

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
        sequence = 0

        def send(type_: str, **fields):
            nonlocal sequence
            event = {"type": type_, "sequence_number": sequence, **fields}
            sequence += 1
            self.wfile.write(f"event: {type_}\ndata: {json.dumps(event)}\n\n".encode())

        send("response.created", response=make_response("in_progress"))
        if scenario == "truncated":
            return
        if response["output"]:
            message = response["output"][0]
            part = message["content"][0]
            ids = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
            send(
                "response.output_item.added",
                output_index=0,
                item={**message, "status": "in_progress", "content": []},
            )
            if part["type"] == "refusal":
                send("response.content_part.added", **ids, part={**part, "refusal": ""})
                send("response.refusal.delta", **ids, delta=REFUSAL)
                send("response.refusal.done", **ids, refusal=REFUSAL)
            else:
                send("response.content_part.added", **ids, part={**part, "text": ""})
                for delta in ("Hello", " world"):
                    send("response.output_text.delta", **ids, delta=delta, logprobs=[])
                    if scenario.startswith("error_event"):
                        code = (
                            None if scenario == "error_event_nocode" else "server_error"
                        )
                        send("error", code=code, message=STREAM_ERROR)
                        return
                send("response.output_text.done", **ids, text=TEXT, logprobs=[])
            send("response.content_part.done", **ids, part=part)
            send("response.output_item.done", output_index=0, item=message)
        send(event_type, response=response)
