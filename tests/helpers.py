"""Plain helper functions shared by the tests (fixtures live in conftest.py)."""

import contextlib
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from openai.types.responses import ResponseInputParam

from muagent.agent import Agent
from muagent.commands import COMMANDS
from muagent.config import Config
from muagent.mcp_hub import McpHub, ServerStatus
from muagent.repl import Repl
from muagent.tools import Tool


def make_context(turns: int) -> ResponseInputParam:
    context: ResponseInputParam = []
    for i in range(turns):
        context.append({"role": "user", "content": f"question {i}"})
        context.append({"role": "assistant", "content": f"answer {i}"})
    return context


def make_config(**values) -> Config:
    # Tests spell the settings in capitals (STREAMING=False); the fields are lowercase.
    names = {name.lower(): value for name, value in values.items()}
    return Config.model_validate({"api_key": "test", "model": "gpt-4o", **names})


def run_repl(
    monkeypatch: pytest.MonkeyPatch,
    agent: Agent,
    inputs: list,
    servers: Sequence[ServerStatus] = (),
    hub: McpHub | None = None,
) -> None:
    """Run a Repl over scripted input.

    Each entry is either a line of input or an exception that input() raises.
    """
    lines = iter(inputs)

    def fake_input(_prompt: str) -> str:
        line = next(lines)
        if isinstance(line, BaseException):
            raise line
        return line

    monkeypatch.setattr("builtins.input", fake_input)
    Repl(agent, COMMANDS, servers=servers, hub=hub).run()


def anchor(agent: Agent, tokens: int) -> None:
    """Make the agent believe the server counted `tokens` for its context, as its model."""
    agent.context.counted = tokens
    agent.context.counted_model = agent.config.model


def tool_with(function) -> Tool:
    """A tool named read_file (the one the fake API calls) that runs `function`."""
    function.__name__ = "read_file"
    return Tool.from_function(function)


FAKE = str(Path(__file__).parent / "fake_mcp_server.py")


@contextlib.contextmanager
def serve_http() -> Iterator[str]:
    """Run the fake server over streamable HTTP on a free port; yield its url."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, FAKE, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(200):
            try:
                socket.create_connection(("127.0.0.1", port), 0.1).close()
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("the fake HTTP server did not start")
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        process.terminate()
        process.wait(10)
