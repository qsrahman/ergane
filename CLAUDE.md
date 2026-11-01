# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`muagent` is a minimal interactive terminal chat agent built on the OpenAI **Responses API**. It's a `src`-layout package (`src/muagent/`); see Architecture for the modules.

## Commands

Uses `uv` with Python 3.14 (`.python-version`, `requires-python >= 3.14`).

```bash
uv sync              # install dependencies (including pytest) into .venv
uv run muagent      # start the REPL; same as `uv run python -m muagent`
uv run pytest        # run the tests
uv run pytest tests/test_repl.py::TestRepl::test_strips_input_and_skips_blank_lines  # single test
uv run ruff check . && uv run ruff format --check .   # lint and format check
uv run pyright src tests   # type check (pyright is a dev dependency)
```

## Tests

- `tests/fake_api.py` is a fake Responses API, and the `fake_server` fixture in `tests/conftest.py` runs it on a local port, using the real SDK client and streaming parser. The most recent user message picks the outcome: `"incomplete"`, `"failed"`, `"empty"` (completed with no output), `"refusal"`, `"cancelled"` (text, but a status other than completed or incomplete), `"truncated"` (a stream with no terminal event), `"tool_call"` (calls a tool until its output is in the input, then replies with text), `"tool_call_reasoning"` (the same, with a reasoning item before each call, as a reasoning model sends), `"tool_call_parallel"` (two calls in one response), `"tool_loop"` (calls a tool every time), `"error_event"` and `"error_event_nocode"` (a stream `error` event, with or without a code, after the first delta, then the stream ends), or anything else for a completed `"Hello world"` reply. Request bodies are recorded in `fake_server.requests`.
- The autouse `isolate_config` fixture runs each test in an empty `tmp_path` that is also `XDG_CONFIG_HOME`, and clears the config environment variables (taken from `Config`'s fields, so a new setting needs no change here), so neither the project's `.env` nor the user config file leaks in. `TestLoadConfig` writes its config files into that `tmp_path`, and the default-location tests point `HOME` there too.
- The test files mirror the modules: `test_config.py`, `test_context.py`, `test_agent.py`, `test_commands.py`, `test_repl.py`, `test_console.py`, `test_cli.py`, `test_tools.py`, `test_demo_tools.py`, plus `test_fake_api.py`, which checks the fake's tool scenarios through the real SDK in both modes. Plain helpers (`make_config`, `make_context`, `run_repl`) live in `tests/helpers.py`; the `client` and `make_agent` fixtures are in `conftest.py`.
- Tests build objects directly instead of monkeypatching. `make_config(...)` builds a `Config`, and the `make_agent` fixture builds an `Agent` wired to the fake server (`make_agent(tools=[...], STREAMING=False)`; tools a test defines are made with `Tool.from_function`, which does not register them). `Agent` tests check return values and `TurnError`, not printed output. `TestRepl` replaces `agent.run_turn` on a single instance, and `TestReplRendering` drives a real `Agent` through `run_repl()`.

The package is built with `uv_build`, which expects `src/muagent/__init__.py`. The `muagent` script is `muagent.cli:main` in `[project.scripts]` (`python -m muagent` runs `__main__.py`), and `uv sync` installs the package in editable mode. Tests import the modules directly, for example `from muagent.agent import Agent`.

## Configuration

`load_config()` builds `Config` (pydantic-settings) from, highest priority first: environment variables, a `.env` in the current working directory, and the user file `$XDG_CONFIG_HOME/muagent/.env` (default `~/.config/muagent/.env`). Missing files are skipped, and keys that aren't settings are ignored (`extra="ignore"`). Only the `OPENAI_*` settings have an alias; the others are read from the variable named like the field (`max_turns` from `MAX_TURNS`, in any case), and `main()` shows a bad one in upper case.

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | required; kept as a `SecretStr`, so printing the config doesn't reveal it |
| `OPENAI_BASE_URL` | `None` | for OpenAI-compatible endpoints |
| `OPENAI_MODEL` | `gpt-5.4-mini` | |
| `STREAMING` | `true` | stream deltas vs. single `responses.create` |
| `MAX_CONTEXT_TOKENS` | `65536` | client-side history budget; excludes instructions and output |
| `REASONING_EFFORT` | unset | SDK `ReasoningEffort`; unset sends no `reasoning`, since non-reasoning models reject it |
| `MAX_TURNS` | `25` | most model calls per prompt: the cap on the tool loop |

`Config` itself only validates and raises `ValidationError`. `main()` prints the per-field errors and exits with status 1.

## Architecture

The modules, in dependency order (each imports only from those above it):

- `tools.py`: the `@tool` decorator, `Tool`, and the `TOOLS` registry.
- `demo_tools.py`: placeholder tools (`add`, `get_time`) that exercise the loop. **Delete it, and its import in `cli.py`, when the real tools arrive.**
- `config.py`: `Config` (the settings model), `load_config()` and `EFFORTS`.
- `context.py`: `Context`, the conversation history and how it is trimmed, and `estimate_tokens()`, which the agent's tool loop uses too.
- `agent.py`: `Agent`, `TurnError` and `SYSTEM_PROMPT`.
- `console.py`: `Console`, which owns all terminal output, and `sanitize()`.
- `commands.py`: the `@command` registry and the slash commands. It imports `Repl` for type checking only.
- `repl.py`: `Repl`, the input loop and dispatch.
- `cli.py`: `main()`, which `__main__.py` calls.

`main()` only loads the config, reporting errors, and wires the pieces together: `Repl(Agent(config, tools=list(TOOLS.values())), COMMANDS).run()`. Dependencies are passed in through constructors (`Agent(config, client=None, tools=())`, `Repl(agent, commands)`); there is no global config lookup.

- **`Context`** holds the history (`items`). Every request uses `store=False`, so the full history is resent each turn.
  - `draft(prompt, max_tokens)` returns a copy of the history plus the new user message, trimmed to the budget, without changing `items`. The budget is an argument, not state: `Agent` passes `config.max_context_tokens` on every call, so every setting is read where it is used and none is copied at startup (except `api_key` and `base_url`, which build the client).
  - `commit(items)` is the only way a turn becomes history. It takes the whole history after the turn (the drafted input plus everything the model produced), so a turn can hold more than a prompt and a text reply, such as reasoning, function-call and function-call-output items. They stay together: `undo()`, `rewind()` and trimming all work on whole turns, split at `user` messages. `Agent` calls it after a `completed` (or status-less, as some compatible servers send) response with non-empty `output_text`, and also for an `incomplete` response with text, so the partial reply the user saw is kept. Failed or empty responses and exceptions leave history untouched, so user and assistant messages always alternate. Preserve this invariant when changing the turn flow.
  - `undo()` drops the last turn (the last `user` message and everything after it) and reports whether there was one.
  - `rewind()` is what `/retry` uses: it drops the last turn if it was saved and returns its prompt. `Context` tracks `last_prompt` (set by `draft()`) and `last_saved` (set by `commit()`, cleared by `draft()`, `undo()` and `reset()`), so all history changes and the flag live in one class.
  - Trimming, inside `draft()`, estimates each item's tokens once with `estimate_tokens()` (UTF-8 bytes of its JSON ÷ 4, tokenizer-agnostic; a reasoning item's opaque `encrypted_content` is left out, because the API counts it as the few tokens of reasoning it holds, not its length: 18 tokens for 1,636 characters on gpt-5.5, against about 444 estimated) and drops whole turns from the front, each up to the next `user` message, until under budget. It never drops the most recent user turn, so a single oversized prompt can still exceed the limit.
- **`@tool`** (`tools.py`) turns a plain function into a tool, as in FastMCP. The name is the function's name, the description is its docstring (all of it, so write it for the model), and the parameters come from its type hints and defaults; `Annotated[int, Field(le=200, description="...")]` adds limits and per-parameter descriptions. It returns `str`. Mistakes fail at import: no docstring, a non-`str` return, an unannotated parameter, `*args`/`**kwargs`/positional-only/async, a duplicate name. A tool is `read_only` by default; `@tool(read_only=False)` declares one that changes things. Nothing enforces the flag yet: it is for the confirmation step that non-read-only tools will need The decorated name is a `Tool`, so call `add.function(...)` or `add.call('{"a": 1}')` in a test. `Tool.call` takes the model's JSON text and never raises: bad JSON, invalid arguments and any exception from the function come back as text (`Invalid arguments: ...`, `Error: Type: message`) for the model to correct, except Ctrl-C. Output over `MAX_OUTPUT_CHARS` is cut with a marker.
- **`Agent`** owns the config, the client and the `Context`, and does no terminal I/O. `run_turn(prompt, on_text, on_notice)` sends one turn and passes reply text to `on_text`: each delta when streaming, or the whole reply once otherwise (before the status check in both modes, so partial text is always shown). Status messages such as trimming go to `on_notice`. It raises `TurnError` for incomplete, failed, refused or empty responses. `run_turn` only applies the policy: `send_response()` gets the final response (the only place that calls the API), and `_status_problem()` / `_text_problem()` turn a response into an error message or `None`. They are separate so that a tool loop can tell a tool call (no text, which is fine) from a final response with no text (an error). The reply is saved when there is text and the turn either worked or was cut off (`incomplete`); then the problem, if any, is raised. The request is built once as the SDK's own `ResponseCreateParamsBase` and unpacked into both `create` and `stream`, so pyright checks the keys and they are written once; `reasoning` is added only when `REASONING_EFFORT` is set. `stream()` is kept over `create(stream=True)` because it rebuilds the final output when a compatible server leaves it out of `response.completed`. The system prompt plus environment info (OS, current date) is passed via `instructions`, not stored in the history. When streaming, `send_response()` takes the final response from the terminal stream event, because `get_final_response()` raises on incomplete or failed streams. If the stream ends with no final response, the `TurnError` carries the reason from the server's last `error` event (`Stream error: CODE: MESSAGE`), or says the stream ended without a final response if it sent none.
  - Tools: `Agent(config, client, tools)` is given the tools to offer and never reads the `TOOLS` registry itself; `cli.py` gives it every registered tool. `run_turn` is a loop of at most `max_turns` model calls: `send_response()` gets a response, and if it calls tools the model's own items (reasoning included) are kept in the turn's working list, `_run_tools()` answers every call, and the model is called again. It stops at a response with no calls, with a problem, at the cap (the turn is then saved, closed with an assistant note `[Stopped after N model calls without a final answer]`, so the model still knows the question and the work done; every call has its output, so history stays valid), or when the turn's estimated size passes `max_context_tokens` after a round (a `TurnError` each time; the budget stop saves nothing, as the turn no longer fits): nothing else bounds how far a turn grows while it runs, and every request resends all of it. Every call gets a `function_call_output`, even a failed or unknown one, as text for the model to read and correct. Each call is reported through `on_notice` before it runs, as `name(arguments)`, and its result after, as `→ result`; both are shortened to one line of `NOTICE_CHARS` characters, so a failed call's error is visible to the user too.
  - The turn is saved whole by `Context.commit`: the working list (user message, the model's call items, their outputs) plus the final reply as a plain `{"role": "assistant", "content": text}` message. Later turns therefore resend the tool calls and outputs; compacting them is a possible future change.
- **`Console`** (`console.py`) is the only thing that prints replies and status lines. `write()` prints text as it is (sanitized), `notice()` prints a bracketed status line, and `blank()`, `end_line()` and `clear()` do what they say. It tracks `at_line_start`, so a status line starts a new line first if a reply was cut off mid-line. `sanitize()` strips control characters and bidi overrides, since replies and notices carry text from the model or server. `/clear` only prints its escape codes when stdout is a terminal.
- **`Repl`** owns the input loop and dispatch, and prints through `self.console`.
  - `handle()` splits input at the first space: the first word picks a command, and the rest (stripped, `""` if none) is passed as `args`. A single word starting with `/` that isn't a command shows "Unknown command" instead of reaching the model. Anything else goes to `ask()`. A handler that raises is shown as `Command error: ...`, and Ctrl-C inside one as `Cancelled`; either way the session continues, so a new command needs no error handling of its own.
  - `ask()` runs one turn, passing `console.write` and `console.notice` to the agent, then shows any `TurnError`, `Agent error: ...` or `Cancelled` as a notice. Ctrl-C cancels just that turn, and the loop continues.
  - Slash commands live in the module-level `COMMANDS` registry in `commands.py` as `Command(handler, description)`; `main()` passes that dict to `Repl(agent, commands)`, which keeps it as `Repl.commands`, matched case-insensitively, and `/help` lists it. A handler is `handler(repl: Repl, args: str) -> str | None`: it returns a short status message, which `handle()` shows as a notice followed by a blank line, or `None` for no message. To add a command, write a handler with a docstring and decorate it with `@command("/name")`: the first line of the docstring is the description `/help` shows, as a tool's docstring is its description. A name must be a slash, then lowercase letters, digits and hyphens, because input is lowercased before the lookup and a leading slash keeps a plain prompt from matching; a bad name, a duplicate or a missing docstring fails at import (`ValueError`). Stack decorators for aliases (`/exit` and `/quit`).
  - `/model` and `/effort` change `repl.agent.config` for the rest of the session, and `/tools` only lists `repl.agent.tools`; `/effort` validates against the SDK's `ReasoningEffort` (`EFFORTS` lists its values). `/retry` calls `context.rewind()` and resends the prompt it returns.

## Conventions

- The code uses Python 3.14 syntax, such as the unparenthesized `except KeyboardInterrupt, EOFError:` (PEP 758). ruff takes its target from `requires-python`; `[tool.ruff]` only sets `src = ["src", "tests"]` so `muagent` and `fake_api` sort as first-party imports.

- Google-style docstrings (`Args:` / `Returns:` / `Raises:`) on functions.
- Output is a plain terminal: the system prompt forbids LaTeX in favor of Unicode math. Keep that constraint if editing the prompt.
