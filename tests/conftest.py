import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from openai import OpenAI
from pydantic import TypeAdapter

from fake_api import FakeResponsesHandler
from helpers import FAKE, make_config, serve_http
from muagent.agent import Agent
from muagent.config import LocalMcpServer, McpServerConfig
from muagent.mcp_client import McpClient
from muagent.mcp_hub import McpHub
from muagent.tools import Tool


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the developer's config files and environment out of load_config().

    Each test runs in an empty directory that is also XDG_CONFIG_HOME (and
    XDG_STATE_HOME, so no test writes a log into the developer's home), so the
    developer's own user-level config file is not found.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    # The one variable muagent reads (see config._Environment).
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


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

    def make(
        *,
        tools: Sequence[Tool] = (),
        server_notes: Mapping[str, str] | None = None,
        **values,
    ) -> Agent:
        return Agent(make_config(**values), client, tools, server_notes)

    return make


@pytest.fixture(scope="module")
def tools(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Tool]]:
    """The fake server's tools, from one server shared by the tests that only call it."""
    log = tmp_path_factory.mktemp("mcp") / "log"
    server = McpClient("fake", LocalMcpServer(command=sys.executable, args=[FAKE]), log)
    server.start()
    yield {tool.name: tool for tool in server.tools()}
    server.close()


@pytest.fixture(scope="module")
def http_url() -> Iterator[str]:
    with serve_http() as url:
        yield url


@pytest.fixture
def start_hub(tmp_path: Path) -> Iterator[Callable[..., tuple[McpHub, list[Tool]]]]:
    """Start servers through a hub and wait for them; they are closed after the test.

    The logs go to `tmp_path`. Returns the hub and the tools of the servers that run.
    """
    hubs: list[McpHub] = []

    def start(
        configs: Mapping[str, McpServerConfig],
        on_notice: Callable[[str], None] = lambda notice: None,
        **values,
    ) -> tuple[McpHub, list[Tool]]:
        hub = McpHub(configs, tmp_path, **values)
        hubs.append(hub)
        return hub, hub.start(on_notice)

    yield start
    for hub in hubs:
        hub.close()


@pytest.fixture
def make_server(tmp_path: Path) -> Iterator[Callable[..., McpClient]]:
    """Start servers for a test (the fake one unless told otherwise); stop them after.

    The server's stderr goes to `tmp_path / "mcp.log"`.
    """
    started: list[McpClient] = []

    def make(
        *, start: bool = True, name: str = "fake", log: Path | None = None, **values
    ) -> McpClient:
        if "url" not in values:
            values = {"command": sys.executable, "args": [FAKE], **values}
        config = TypeAdapter(McpServerConfig).validate_python(values)
        server = McpClient(name, config, log or tmp_path / "mcp.log")
        started.append(server)
        if start:
            server.start()
        return server

    yield make
    for server in started:
        server.close()
