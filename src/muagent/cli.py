import sys

from pydantic import ValidationError

from muagent.agent import Agent
from muagent.commands import COMMANDS
from muagent.config import load_config

# Remove demo_tools when the real tools arrive.
from muagent.demo_tools import add, get_time
from muagent.repl import Repl


def main() -> None:
    """Entry point for the AI coding agent CLI.

    Raises:
        SystemExit: If the configuration is invalid.
    """
    try:
        config = load_config()
    except ValidationError as exc:
        print("Configuration error:", file=sys.stderr)
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"]).upper()
            print(f"  - {field}: {error['msg']}", file=sys.stderr)
        raise SystemExit(1) from None

    Repl(Agent(config, tools=[add, get_time]), COMMANDS).run()
