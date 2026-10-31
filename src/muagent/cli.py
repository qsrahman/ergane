"""Command-line entry point: load the settings and start the REPL."""

import sys

from pydantic import ValidationError

from muagent.agent import Agent
from muagent.config import load_config
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
            field = ".".join(str(part) for part in error["loc"])
            print(f"  - {field}: {error['msg']}", file=sys.stderr)
        raise SystemExit(1) from None

    Repl(Agent(config)).run()
