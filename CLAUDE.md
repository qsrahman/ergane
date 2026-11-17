# CLAUDE.md

Guidance for working in this repository. The mechanics are in the modules' docstrings
(start with the one at the top of a file) and in [TURN.md](TURN.md), which walks one
turn through the code; this file keeps what a change must not break.

## Overview

`muagent` is a minimal interactive terminal chat agent on the OpenAI **Responses API**
(`src/muagent/`, `src` layout). Its tools come from MCP servers listed in
`config.json` and from the Python functions in `src/muagent/builtin/` (none yet); with
no tool at all it is a plain chat.

## Commands

Uses `uv` with Python 3.14 (`.python-version`, `requires-python >= 3.14`).

```bash
uv sync              # install dependencies (including pytest and mcp) into .venv
uv sync --extra mcp  # what a user needs for MCP servers; `mcp` is an optional extra
uv run muagent       # start the REPL; same as `uv run python -m muagent`
uv run pytest        # run the tests
uv run pytest tests/test_repl.py::TestRepl::test_strips_input_and_skips_blank_lines  # one test
uv run ruff check . && uv run ruff format --check .   # lint and format check
uv run pyright src tests   # type check
```

The package is built with `uv_build` (needs `src/muagent/__init__.py`); `uv sync` installs
it editable; the script is `muagent.cli:main`. Tests import modules directly
(`from muagent.agent import Agent`).

## Tests

- `tests/fake_api.py` is a fake Responses API run on a local port by the `fake_server`
  fixture, through the real SDK client and streaming parser. The most recent user
  message picks the outcome; the scenarios are listed in `FakeResponsesHandler`'s
  docstring (failures, broken streams, tool calls, `status_N`, `overflow`, `usage`, and
  a summary that goes wrong in each way). Request bodies are in `fake_server.requests`.
- `tests/fake_mcp_server.py` is a stdio (or, with a port, HTTP) MCP server on the SDK's
  own `MCPServer`, with a tool for each case a client must handle; `test_mcp_client.py` and
  `test_mcp_hub.py` start it as a subprocess (`FAKE`, `serve_http` in `helpers.py`; the
  `make_server`, `http_url` and `tools` fixtures in `conftest.py`).
- The autouse `isolate_config` fixture runs each test in an empty `tmp_path` that is
  also `XDG_CONFIG_HOME`, and clears `OPENAI_API_KEY`, the one variable muagent reads.
- Test files mirror the modules (`test_mcp_config.py` tests the server models of `config.py`), plus `test_compaction.py` (summaries, shortened
  results, the too-long retry) and `test_fake_api.py`. Helpers (`make_config`,
  `make_context`, `run_repl`, `tool_with`) are in `tests/helpers.py`; the `client` and
  `make_agent` fixtures are in `conftest.py`.
- Build objects directly instead of monkeypatching: `make_agent(tools=[...],
  STREAMING=False)` wires an `Agent` to the fake server; tools a test defines come from
  `Tool.from_function`. `Agent` tests check return values and `TurnError`, not printed
  output. `TestRepl` replaces `agent.run_turn` on one instance; `TestReplRendering`
  drives a real `Agent` through `run_repl()` and pins the layout of an exchange.
- A behaviour change gets a mutation check: break the code, see the test fail.

## Configuration

Highest first: the command-line flags, the user's
`$XDG_CONFIG_HOME/muagent/config.json` (default `~/.config/muagent/`), the defaults.
The environment has one variable, `OPENAI_API_KEY`, which beats `api_key` in the file;
there is none for any other setting (a variable for `base_url` would let a project's
environment send the key and every prompt elsewhere, one for the servers would start a
program of its own, and `MODEL`, `BASE_URL` are exported by other programs). Only
`--config` names another file. `load_config(path=None, **flags)` reads the file, puts
the key and the flags over it and validates the result as one `Config` (a plain pydantic
model; there is no pydantic-settings); `main()` prints the per-key errors and exits with
status 1. Read the docstring of `config.py` for how, and why.

| Key | Default | Notes |
|---|---|---|
| `api_key` | — | required; a `SecretStr`; may say `${NAME}`; no flag |
| `base_url` | `None` | for OpenAI-compatible endpoints |
| `model` | `gpt-5.4-mini` | |
| `streaming` | `true` | stream deltas vs. one `responses.create` |
| `max_context_tokens` | `65536` | budget for the whole request input: instructions, tool definitions and history (the output excluded) |
| `auto_compact` | `true` | summarize the oldest turns past `compact_at` |
| `compact_at` | `0.75` | share of the budget at which to summarize (0 < x ≤ 1) |
| `keep_recent_tokens` | `15000` | newest turns a summary leaves as they are; at most half the trigger |
| `reasoning_effort` | unset | unset sends no `reasoning`, since non-reasoning models reject it |
| `max_turns` | `25` | most model calls per prompt |

MCP servers are `mcpServers` in the same file (`Config.mcp_servers`), and only there: a
server is a program, so neither the environment nor a project's files may define one.
Each server's stderr goes to `$XDG_STATE_HOME/muagent/NAME.log`; the state directory holds
the prompt history too and is made `0700`.

## Architecture

The modules, in dependency order (each imports only from those above it):

- `tools.py`: the `@tool` decorator and `Tool` (`name`, `description`, `parameters`,
  `check`, `runner`, `read_only`). `Tool.call` never raises (except Ctrl-C): a bad call
  or an exception comes back as text for the model, and output is cut at
  `MAX_OUTPUT_CHARS`. `tools_in(package)` finds the tools of a package: a `@tool`
  function in a module of `builtin/` is offered with nothing else to edit; [TOOLS.md](TOOLS.md)
  is the guide to writing one.
- `config.py`: `Config`, `load_config()`, `EFFORTS`, the XDG directories, the model of a
  server in `config.json`: `LocalMcpServer` (a `command`) or `RemoteMcpServer` (a
  `url`), told apart by `type` or by having a `url`, as the union `McpServerConfig` (a
  field of the other kind is an error of its own, so there is no check of
  combinations), and what the MCP code shares with it: the rules for a server's name
  and the `${NAME}` placeholders (`fill_placeholders`).
- `mcp_client.py`: `McpClient`, one connection used from synchronous code; its tools
  become `Tool`s. Imports `mcp` only when a server starts. The design notes are in its
  docstring.
- `mcp_hub.py`: the servers together: `close_servers`, `McpHub` (starts
  them, at a terminal in the background, and starts one again for `/reconnect`), and a
  `ServerStatus` per server, which records what became of it for `/servers`.
- `errors.py`, `prompts.py`: `TurnError`, `StreamCut`, `describe_error`,
  `is_context_overflow`, `is_images_rejected`; the three prompts as text. They import nothing of muagent.
- `context.py`: `Context` (the history and the summary), `estimate_tokens()`, and the
  functions on a turn's items (`drop_outputs`, `answer_open_calls`, `transcript`,
  `extract_summary`).
- `compaction.py`: `Compactor`: when to summarize the oldest turns, what to keep, and the
  summary itself, through a function that asks the model. It needs nothing of the agent.
- `responses.py`: `Responder` (one request to the API, streamed, sent again when the stream
  breaks before any text) and `response_problem`. It needs nothing of the agent.
- `agent.py`: `Agent`: `run_turn` (a loop of `_ask` and `_run_round` around a `_Turn`),
  `compact` (which is `Compactor`'s), and the tools. It imports nothing of
  `config.py` or MCP: the settings it reads are `AgentSettings`, which `Config` satisfies
  (a test pins this).
- `console.py`: all terminal output, `sanitize()`, and the permission question.
- `commands.py`: the `@command` registry and the slash commands (imports `Repl` for type
  checking only).
- `repl.py`: `Repl`, the input loop, and `terminal_session` (prompt_toolkit).
- `cli.py`: `main()`.

Dependencies are passed in through constructors (`Agent(config, client=None, tools=())`,
`Repl(agent, commands, session=None, console=None)`); there is no global config lookup.
Every setting is read where it is used, so `/model` and `/effort` take effect at once.

What must stay true:

- **The history changes in four places only:** `Context.save_turn`, `undo` (which
  `rewind` uses), `reset` and `compact`. A turn is saved with its reply or not at all,
  apart from the exceptions in `agent.py`'s docstring. User and assistant messages
  alternate; the summary is kept apart (`Context.summary`), put first in each request as
  a `user` message framed as notes, never as instructions, since it is written from text
  that may hold a web page's orders.
- **Every function call has an output** before anything is sent or saved; calls and
  outputs are dropped together; only the newest turn keeps its tool work.
- **A tool that is not `read_only` runs only if `confirm(tool, arguments)` returns true or
  `"always"`** (which allows that tool for the session). The default denies. A call whose
  arguments `check` rejects is answered without asking.
- **Only read-only calls run together** (on threads, `Agent._run_tools`); any other call
  runs alone, after the calls before it. The outputs are added in the order of the calls.
  `Console` is used from the main thread only, so a call that asks never asks at the same
  time as another.
- **Trimming never drops the summary or the newest user message.** The budget is for the
  whole request; the in-turn check uses the server's `usage.input_tokens` plus an
  estimate of what was added, shortens old tool results before dropping them, then the
  latest round's, and fails only when even that does not fit; the turn is then kept with
  a note and its results dropped, like any turn after a completed round.
- **The agent prints nothing.** What the user should see goes through `on_text`,
  `on_notice`, `on_call` or `confirm`, and the console prints it through `_emit`.
- **Credentials and programs come only from the user's own files:** no config, `.env` or
  server list is read from the working directory.

## Conventions

- Python 3.14 syntax, such as `except KeyboardInterrupt, EOFError:` (PEP 758). ruff takes
  its target from `requires-python`; `[tool.ruff]` sets `src = ["src", "tests"]` and
  turns on `BLE001` (a broad `except` says it is meant: `# noqa: BLE001`), `B905` and
  `RUF100` (a noqa that silences nothing goes).
- Google-style docstrings (`Args:` / `Returns:` / `Raises:`) on functions.
- Show the user only what they need, to decide or to know that something went wrong. A
  call that asks is one line, the question; a call that runs freely is one line; a result
  is not shown (the reply says what it found); a server that starts is not announced; the
  banner is one line. Before adding output, ask whether anyone would miss it.
- Output is a plain terminal: the system prompt forbids LaTeX in favor of Unicode math
  and says Markdown is not rendered. Keep those if editing the prompt.
- The system prompt follows OpenAI's guidance for agentic coding models and depends on how
  the program behaves: the user sees each tool call but not its result, a denied call
  comes back as `The user denied this call.`, tool results are untrusted text, and a
  scratch script goes in the system's temporary location and is deleted afterwards
  (a request, not a guarantee: muagent tracks no files). If one of these changes in the
  code, change the prompt; `test_the_prompt_*` points at the words. The with-tools
  section is absent from `SYSTEM_PROMPT_NO_TOOLS`, which never mentions calls.
- **Test a prompt change against the models, not by reading it.** A harmless-looking
  change can break one: "never write a tool call out as text" made gpt-5.5 do it (right
  1 time in 6, 6 in 6 once the sentence was gone); tool language in the no-tools prompt
  made gemma4 return an empty reply; "a new temporary directory" made it create
  `temp_x/` in the project until the prompt named the system's temporary location. Compare
  old and new on scenarios with a pass/fail check each, several runs, on the models you
  use. Test a rule that touches files with a simulated shell that records commands and
  gives realistic answers (one that answered with only an exit code made gemma4 repeat
  `git status` 25 times, which looked like a regression and was the harness). The same
  goes for the summary prompt: a recall check (plant five facts, summarize twice, ask)
  gave 5 of 5 on gpt-5.5 and gemma4, and showed that gemma4 does not write the
  `<summary>` tags, which a strict parser would have refused.
