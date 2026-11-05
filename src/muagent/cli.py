import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

from pydantic import ValidationError

from muagent.agent import Agent
from muagent.commands import COMMANDS
from muagent.config import EFFORTS, load_config, private_state_dir
from muagent.console import Console
from muagent.mcp_client import start_servers
from muagent.repl import Repl, terminal_session


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command line. A flag that is not given is not in the result, so that
    the environment and the file can supply the setting.

    Args:
        argv: The arguments after the program name; `sys.argv[1:]` if omitted.
    """
    parser = argparse.ArgumentParser(
        prog="muagent",
        argument_default=argparse.SUPPRESS,  # a flag not given is not in the result
        description="A minimal terminal chat agent on the OpenAI Responses API.",
        epilog="Each setting comes from the first of: its flag, its environment "
        "variable, config.json, its default. Set the key in config.json "
        '("api_key": "${OPENAI_API_KEY}") or in OPENAI_API_KEY; there is no flag '
        "for it, since a flag shows in the process list.",
    )
    parser.add_argument(
        "--config", type=Path, metavar="FILE", help="the config file to use"
    )
    parser.add_argument("--model", metavar="NAME", help="OPENAI_MODEL")
    parser.add_argument("--base-url", metavar="URL", help="OPENAI_BASE_URL")
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        help="STREAMING: stream the reply as it arrives",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        metavar="N",
        help="MAX_CONTEXT_TOKENS: budget for a whole request",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        metavar="N",
        help="MAX_TURNS: most model calls for one prompt",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=[*EFFORTS, "default"],
        metavar="LEVEL",
        help=f"REASONING_EFFORT: {', '.join(EFFORTS)}, or default to send none",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        default=False,
        help="do not start the MCP servers of config.json",
    )
    return parser.parse_args(argv)


def _exit_with(title: str, exc: ValidationError) -> NoReturn:
    """Print what is wrong with a configuration, and exit.

    Args:
        title: The heading of the report.
        exc: The validation errors.

    Raises:
        SystemExit: Always, with status 1.
    """
    print(title, file=sys.stderr)
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"])
        print(f"  - {field + ': ' if field else ''}{error['msg']}", file=sys.stderr)
    raise SystemExit(1) from None


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the AI coding agent CLI.

    Args:
        argv: The arguments after the program name; `sys.argv[1:]` if omitted.

    Raises:
        SystemExit: If the arguments or the configuration are invalid, or Ctrl-C ends
            the start.
    """
    flags: dict[str, Any] = vars(parse_args(argv))
    path = flags.pop("config", None)
    no_mcp = flags.pop("no_mcp")
    if flags.get("reasoning_effort") == "default":
        flags["reasoning_effort"] = None
    try:
        config = load_config(path, **flags)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None
    except ValidationError as exc:
        _exit_with("Configuration error:", exc)

    console = Console()
    state = private_state_dir()
    try:
        servers, mcp_tools = start_servers(
            {} if no_mcp else config.mcp_servers, state, console.notice
        )
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    if console.after_status:  # a server failed, and said so
        console.blank()

    try:
        session = terminal_session(COMMANDS, state / "history")
        Repl(Agent(config, tools=mcp_tools), COMMANDS, session).run()
    finally:
        for server in servers:
            server.close()
