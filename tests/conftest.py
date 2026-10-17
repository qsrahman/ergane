import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import main

TEXT = "Hello world"
MESSAGE = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": TEXT, "annotations": []}],
}


def make_response(status: str, output: list | None = None, **extra) -> dict:
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
    "failed", "empty" (completed with no output), or anything else for a
    completed response.
    """

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)  # pyright: ignore[reportAttributeAccessIssue]
        event_type, response = final_event(body["input"][-1]["content"])

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
        if response["output"]:
            ids = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
            part = MESSAGE["content"][0]
            send(
                "response.output_item.added",
                output_index=0,
                item={**MESSAGE, "status": "in_progress", "content": []},
            )
            send("response.content_part.added", **ids, part={**part, "text": ""})
            for delta in ("Hello", " world"):
                send("response.output_text.delta", **ids, delta=delta, logprobs=[])
            send("response.output_text.done", **ids, text=TEXT, logprobs=[])
            send("response.content_part.done", **ids, part=part)
            send("response.output_item.done", output_index=0, item=MESSAGE)
        send(event_type, response=response)


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's .env and environment out of Config."""
    monkeypatch.setitem(main.Config.model_config, "env_file", None)
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "STREAMING",
        "MAX_CONTEXT_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_server() -> Iterator[ThreadingHTTPServer]:
    """Run the fake Responses API on a free local port. Received request bodies
    are collected in `server.requests`."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeResponsesHandler)
    server.requests = []  # pyright: ignore[reportAttributeAccessIssue]
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
