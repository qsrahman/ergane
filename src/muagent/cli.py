"""The command line: parses the flags, loads the config, starts the MCP servers and
the REPL, and closes the servers whatever happens."""

import argparse
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, NoReturn

from pydantic import ValidationError

from muagent.agent import Agent
from muagent.commands import COMMANDS
from muagent.config import EFFORTS, config_file, load_config, private_state_dir
from muagent.console import Console
from muagent.mcp_hub import McpHub, ServerStatus
from muagent.repl import Repl, terminal_session
from muagent.tools import Tool


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
        epilog="Each setting comes from the first of: its flag, config.json, its "
        "default. The key is the exception: the environment variable "
        'OPENAI_API_KEY beats "api_key" in config.json (which may say '
        '"${NAME}"). There is no flag for it, since a flag shows in the process '
        "list.",
    )
    parser.add_argument(
        "--version", action="version", version=f"muagent {version('muagent')}"
    )
    parser.add_argument(
        "--config", type=Path, metavar="FILE", help="the config file to use"
    )
    parser.add_argument("--model", metavar="NAME", help="the model")
    parser.add_argument(
        "--base-url", metavar="URL", help="the base URL of an OpenAI-compatible API"
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        help="stream the reply as it arrives",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        metavar="N",
        help="budget for a whole request",
    )
    parser.add_argument(
        "--auto-compact",
        action=argparse.BooleanOptionalAction,
        help="summarize the oldest turns when the history gets long",
    )
    parser.add_argument(
        "--compact-at",
        type=float,
        metavar="SHARE",
        help="share of the context budget at which to summarize",
    )
    parser.add_argument(
        "--keep-recent-tokens",
        type=int,
        metavar="N",
        help="tokens of the newest turns kept verbatim",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        metavar="N",
        help="most model calls for one prompt",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=[*EFFORTS, "default"],
        metavar="LEVEL",
        help=f"{', '.join(EFFORTS)}, or default to send none",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        default=False,
        help="do not start the MCP servers of config.json",
    )
    return parser.parse_args(argv)


def _exit_with(title: str, exc: ValidationError, file: Path) -> NoReturn:
    """Print what is wrong with a configuration, and exit.

    Args:
        title: The heading of the report.
        exc: The validation errors.
        file: The config file in use, which the help for a missing key points to.

    Raises:
        SystemExit: Always, with status 1.
    """
    print(title, file=sys.stderr)
    for error in exc.errors():
        # An error about the file as a whole has no field: it is titled with the path.
        field = ".".join(str(part) for part in error["loc"]) or exc.title
        print(f"  - {field + ': ' if field else ''}{error['msg']}", file=sys.stderr)
    if any(e["type"] == "missing" and e["loc"] == ("api_key",) for e in exc.errors()):
        print(
            f'No API key. Put "api_key" in {file} (the key itself, or '
            '"${OPENAI_API_KEY}"), or set the OPENAI_API_KEY environment variable.',
            file=sys.stderr,
        )
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
    except OSError as exc:  # the file the user named is missing, or a directory
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None
    except ValidationError as exc:
        _exit_with("Configuration error:", exc, config_file(path))

    console = Console()
    state = private_state_dir()
    session = terminal_session(COMMANDS, state / "history")
    hub = None if no_mcp else McpHub(config.mcp_servers, state)
    statuses = [ServerStatus(n, "off", "--no-mcp") for n in config.mcp_servers]
    mcp_tools: list[Tool] = []
    try:
        if hub is not None and session is not None:
            # At a terminal the prompt comes up at once and the servers connect while
            # the user types; the first turn waits for them if they are not up.
            hub.begin()
        elif hub is not None:
            mcp_tools = hub.start(console.notice)
    except KeyboardInterrupt:
        raise SystemExit(130) from None

    try:
        # The banner is separated from the notices of servers that failed by the
        # console itself, which is why it is the same one.
        agent = Agent(
            config, tools=mcp_tools, server_notes=hub.instructions() if hub else None
        )
        Repl(agent, COMMANDS, session, console, statuses, hub).run()
    finally:
        if hub:
            hub.close()
