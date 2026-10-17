import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer

import pytest

import main
from fake_api import FakeResponsesHandler


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
