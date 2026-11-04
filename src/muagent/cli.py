import sys
from typing import NoReturn

from pydantic import ValidationError

from muagent.agent import Agent
from muagent.commands import COMMANDS
from muagent.config import load_config, user_config_dir, user_state_dir
from muagent.console import Console
from muagent.mcp_client import load_servers, start_servers
from muagent.repl import Repl, terminal_session


def _exit_with(title: str, exc: ValidationError, *, upper: bool = False) -> NoReturn:
    """Print what is wrong with a configuration, and exit.

    Args:
        title: The heading of the report.
        exc: The validation errors.
        upper: Whether to show the names in upper case, as environment variables.

    Raises:
        SystemExit: Always, with status 1.
    """
    print(title, file=sys.stderr)
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"])
        field = field.upper() if upper else field
        print(f"  - {field + ': ' if field else ''}{error['msg']}", file=sys.stderr)
    raise SystemExit(1) from None


def main() -> None:
    """Entry point for the AI coding agent CLI.

    Raises:
        SystemExit: If the configuration is invalid, or Ctrl-C ends the start.
    """
    try:
        config = load_config()
    except ValidationError as exc:
        _exit_with("Configuration error:", exc, upper=True)

    mcp_file = user_config_dir() / "mcp.json"
    try:
        mcp_configs = load_servers(mcp_file)
    except ValidationError as exc:
        _exit_with(f"Configuration error in {mcp_file}:", exc)

    console = Console()
    try:
        servers, mcp_tools = start_servers(
            mcp_configs, user_state_dir(), console.notice
        )
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    if console.after_status:  # a server failed, and said so
        console.blank()

    try:
        session = terminal_session(COMMANDS, user_state_dir() / "history")
        Repl(Agent(config, tools=mcp_tools), COMMANDS, session).run()
    finally:
        for server in servers:
            server.close()
