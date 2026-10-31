import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from openai import OpenAI

from fake_api import FakeResponsesHandler
from helpers import make_config
from muagent.agent import Agent
from muagent.config import Config


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the developer's config files and environment out of load_config().

    Each test runs in an empty directory that is also XDG_CONFIG_HOME, so
    neither the project's .env nor the user-level config file is found.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # Taken from Config itself, so a new setting needs no change here.
    for field in Config.model_fields.values():
        monkeypatch.delenv(str(field.validation_alias), raising=False)


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


@pytest.fixture
def client(fake_server: ThreadingHTTPServer) -> OpenAI:
    host, port = fake_server.server_address[:2]
    return OpenAI(api_key="test", base_url=f"http://{host}:{port}/v1", max_retries=0)


@pytest.fixture
def make_agent(client: OpenAI):
    """Build an Agent wired to the fake server, with config from the given values."""

    def make(**values) -> Agent:
        return Agent(make_config(**values), client)

    return make
