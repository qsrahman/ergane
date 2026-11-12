"""Plain helper functions shared by the tests (fixtures live in conftest.py)."""

import pytest
from openai.types.responses import ResponseInputParam

from muagent.agent import Agent
from muagent.commands import COMMANDS
from muagent.config import Config
from muagent.repl import Repl
from muagent.tools import Tool


def make_context(turns: int) -> ResponseInputParam:
    context: ResponseInputParam = []
    for i in range(turns):
        context.append({"role": "user", "content": f"question {i}"})
        context.append({"role": "assistant", "content": f"answer {i}"})
    return context


def make_config(**values) -> Config:
    # Not model_validate(): only the constructor reads names in any case.
    return Config(**{"api_key": "test", "model": "gpt-4o", **values})  # pyright: ignore[reportCallIssue]


def run_repl(monkeypatch: pytest.MonkeyPatch, agent: Agent, inputs: list) -> None:
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
    Repl(agent, COMMANDS).run()


def tool_with(function) -> Tool:
    """A tool named read_file (the one the fake API calls) that runs `function`."""
    function.__name__ = "read_file"
    return Tool.from_function(function)
