# CLAUDE.md

Guidance for working in this repository. The mechanics are in the modules' docstrings
(start with the one at the top of a file) and in [TURN.md](TURN.md), which walks one
turn through the code; this file keeps what a change must not break.

## Overview

`muagent` is a minimal interactive terminal chat agent on the OpenAI **Responses API**
(`src/muagent/`, `src` layout). Its tools all come from MCP servers listed in
`config.json`; with none it is a plain chat.

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
  own `MCPServer`, with a tool for each case a client must handle; `test_mcp_client.py`
  starts it as a subprocess.
- The autouse `isolate_config` fixture runs each test in an empty `tmp_path` that is
  also `XDG_CONFIG_HOME`, and clears the config environment variables (taken from
  `Config`'s fields, so a new setting needs no change there).
- Test files mirror the modules, plus `test_compaction.py` (summaries, shortened
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

Highest first: the command-line flags, environment variables, the user's
`$XDG_CONFIG_HOME/muagent/config.json` (default `~/.config/muagent/`), the defaults.
`load_config(path=None, **flags)` builds `Config`; `main()` prints the per-key errors and
exits with status 1. Read the docstring of `config.py` for how, and why.

| Variable (file key) | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` (`api_key`) | — | required; a `SecretStr`; the file may say `${NAME}`; no flag |
| `OPENAI_BASE_URL` (`base_url`) | `None` | for OpenAI-compatible endpoints |
| `OPENAI_MODEL` (`model`) | `gpt-5.4-mini` | |
| `STREAMING` (`streaming`) | `true` | stream deltas vs. one `responses.create` |
| `MAX_CONTEXT_TOKENS` (`max_context_tokens`) | `65536` | budget for the whole request input: instructions, tool definitions and history (the output excluded) |
| `AUTO_COMPACT` (`auto_compact`) | `true` | summarize the oldest turns past `compact_at` |
| `COMPACT_AT` (`compact_at`) | `0.75` | share of the budget at which to summarize (0 < x ≤ 1) |
| `KEEP_RECENT_TOKENS` (`keep_recent_tokens`) | `15000` | newest turns a summary leaves as they are; at most half the trigger |
| `REASONING_EFFORT` (`reasoning_effort`) | unset | unset sends no `reasoning`, since non-reasoning models reject it |
| `MAX_TURNS` (`max_turns`) | `25` | most model calls per prompt |

The environment is read by the SDK's names (`OPENAI_*`) and the setting's own name for the
rest; the bare `MODEL`, `API_KEY` and `BASE_URL` that other programs export are ignored.
MCP servers are `mcpServers` in the same file (`Config.mcp_servers`), and only there: a
server is a program, so neither the environment nor a project's files may define one.
Each server's stderr goes to `$XDG_STATE_HOME/muagent/NAME.log`; the state directory holds
the prompt history too and is made `0700`.

## Architecture

The modules, in dependency order (each imports only from those above it):

- `tools.py`: the `@tool` decorator and `Tool` (`name`, `description`, `parameters`,
  `check`, `runner`, `read_only`). `Tool.call` never raises (except Ctrl-C): a bad call
  or an exception comes back as text for the model, and output is cut at
  `MAX_OUTPUT_CHARS`.
- `mcp_client.py`: `McpServer`, `start_servers`, `close_servers`; a server's tools become
  `Tool`s. Imports `mcp` only when a server starts. The design notes are in its docstring.
- `errors.py`, `prompts.py`: `TurnError`, `StreamCut`, `describe_error`,
  `is_context_overflow`; the three prompts as text. They import nothing of muagent.
- `config.py`: `Config`, `load_config()`, `EFFORTS`, the XDG directories.
- `context.py`: `Context` (the history and the summary), `estimate_tokens()`, and the
  functions on a turn's items (`drop_outputs`, `answer_open_calls`, `transcript`,
  `extract_summary`).
- `agent.py`: `Agent`: `run_turn` (a loop of `_ask` and `_run_round` around a `_Turn`),
  `compact`, and the API calls.
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
  its target from `requires-python`; `[tool.ruff]` sets `src = ["src", "tests"]`.
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
