"""Plain helper functions shared by the tests (fixtures live in conftest.py)."""

import pytest
from openai.types.responses import ResponseInputParam

from muagent.agent import Agent
from muagent.config import Config
from muagent.repl import Repl


def make_context(turns: int) -> ResponseInputParam:
    context: ResponseInputParam = []
    for i in range(turns):
        context.append({"role": "user", "content": f"question {i}"})
        context.append({"role": "assistant", "content": f"answer {i}"})
    return context


def make_config(**values) -> Config:
    return Config.model_validate(
        {"OPENAI_API_KEY": "test", "OPENAI_MODEL": "gpt-4o", **values}
    )


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
    Repl(agent).run()
