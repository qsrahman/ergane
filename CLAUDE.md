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

- `tests/fake_api.py` is a fake Responses API, and the `fake_server` fixture in `tests/conftest.py` runs it on a local port, using the real SDK client and streaming parser. The last input message picks the outcome: `"incomplete"`, `"failed"`, `"empty"` (completed with no output), `"refusal"`, `"truncated"` (a stream with no terminal event), `"error_event"` and `"error_event_nocode"` (a stream `error` event, with or without a code, after the first delta, then the stream ends), or anything else for a completed `"Hello world"` reply. Request bodies are recorded in `fake_server.requests`.
- The autouse `isolate_config` fixture runs each test in an empty `tmp_path` that is also `XDG_CONFIG_HOME`, and clears the config environment variables (taken from `Config`'s fields, so a new setting needs no change here), so neither the project's `.env` nor the user config file leaks in. `TestLoadConfig` writes its config files into that `tmp_path`.
- The test files mirror the modules: `test_config.py`, `test_context.py`, `test_agent.py`, `test_commands.py`, `test_repl.py`, `test_console.py`, `test_cli.py`. Plain helpers (`make_config`, `make_context`, `run_repl`) live in `tests/helpers.py`; the `client` and `make_agent` fixtures are in `conftest.py`.
- Tests build objects directly instead of monkeypatching. `make_config(...)` builds a `Config`, and the `make_agent` fixture builds an `Agent` wired to the fake server. `Agent` tests check return values and `TurnError`, not printed output. `TestRepl` replaces `agent.run_turn` on a single instance, and `TestReplRendering` drives a real `Agent` through `run_repl()`.

The package is built with `uv_build`, which expects `src/muagent/__init__.py`. The `muagent` script is `muagent.cli:main` in `[project.scripts]` (`python -m muagent` runs `__main__.py`), and `uv sync` installs the package in editable mode. Tests import the modules directly, for example `from muagent.agent import Agent`.

## Configuration

`load_config()` builds `Config` (pydantic-settings) from, highest priority first: environment variables, a `.env` in the current working directory, and the user file `$XDG_CONFIG_HOME/muagent/.env` (default `~/.config/muagent/.env`). Missing files are skipped, and keys that aren't settings are ignored (`extra="ignore"`).

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `OPENAI_BASE_URL` | `None` | for OpenAI-compatible endpoints |
| `OPENAI_MODEL` | `gpt-5.4-mini` | |
| `STREAMING` | `true` | stream deltas vs. single `responses.create` |
| `MAX_CONTEXT_TOKENS` | `65536` | client-side history budget; excludes instructions and output |
| `REASONING_EFFORT` | unset | SDK `ReasoningEffort`; unset sends no `reasoning`, since non-reasoning models reject it |
| `MAX_TURNS` | `25` | validated but not used yet; reserved to cap model calls per prompt once tool calling adds a loop to `run_turn` |

`Config` itself only validates and raises `ValidationError`. `main()` prints the per-field errors and exits with status 1.

## Architecture

The modules, in dependency order (each imports only from those above it):

- `config.py`: `Config` (the settings model), `load_config()` and `EFFORTS`.
- `context.py`: `Context`, the conversation history and how it is trimmed.
- `agent.py`: `Agent`, `TurnError` and `SYSTEM_PROMPT`.
- `console.py`: `Console`, which owns all terminal output, and `sanitize()`.
- `commands.py`: the `@command` registry and the slash commands. It imports `Repl` for type checking only.
- `repl.py`: `Repl`, the input loop and dispatch.
- `cli.py`: `main()`, which `__main__.py` calls.

`main()` only loads the config, reporting errors, and wires the pieces together: `Repl(Agent(config)).run()`. Dependencies are passed in through constructors (`Agent(config, client=None)`, `Repl(agent)`); there is no global config lookup.

- **`Context`** holds the history (`items`). Every request uses `store=False`, so the full history is resent each turn.
  - `draft(prompt, max_tokens)` returns a copy of the history plus the new user message, trimmed to the budget, without changing `items`. The budget is an argument, not state: `Agent` passes `config.max_context_tokens` on every call, so every setting is read where it is used and none is copied at startup (except `api_key` and `base_url`, which build the client).
  - `commit(draft, reply)` is the only way a turn becomes history. `Agent` calls it after a `completed` (or status-less, as some compatible servers send) response with non-empty `output_text`, and also for an `incomplete` response with text, so the partial reply the user saw is kept. Failed or empty responses and exceptions leave history untouched, so user and assistant messages always alternate. Preserve this invariant when changing the turn flow.
  - `undo()` drops the last turn (the last `user` message and everything after it) and reports whether there was one.
  - `rewind()` is what `/retry` uses: it drops the last turn if it was saved and returns its prompt. `Context` tracks `last_prompt` (set by `draft()`) and `last_saved` (set by `commit()`, cleared by `draft()`, `undo()` and `reset()`), so all history changes and the flag live in one class.
  - Trimming, inside `draft()`, estimates each item's tokens once (UTF-8 bytes of its JSON ÷ 4, tokenizer-agnostic) and drops whole turns from the front, each up to the next `user` message, until under budget. It never drops the most recent user turn, so a single oversized prompt can still exceed the limit.
- **`Agent`** owns the config, the client and the `Context`, and does no terminal I/O. `run_turn(prompt, on_text, on_notice)` sends one turn and passes reply text to `on_text`: each delta when streaming, or the whole reply once otherwise (before the status check in both modes, so partial text is always shown). Status messages such as trimming go to `on_notice`. It raises `TurnError` for incomplete, failed, refused or empty responses. `run_turn` only applies the policy: `_respond()` gets the final response (the only place that calls the API), and `_problem()` turns a response into an error message or `None`. The reply is saved when there is text and the turn either worked or was cut off (`incomplete`); then the problem, if any, is raised. `create` and `stream` each get explicit keyword arguments, so pyright checks them; `reasoning` is the SDK's `omit` sentinel when `REASONING_EFFORT` is unset, which leaves it out of the request. `stream()` is kept over `create(stream=True)` because it rebuilds the final output when a compatible server leaves it out of `response.completed`. The system prompt plus environment info (OS, current date) is passed via `instructions`, not stored in the history. When streaming, `_respond()` takes the final response from the terminal stream event, because `get_final_response()` raises on incomplete or failed streams. If the stream ends with no final response, the `TurnError` carries the reason from the server's last `error` event (`Stream error: CODE: MESSAGE`), or says the stream ended without a final response if it sent none.
- **`Console`** (`console.py`) is the only thing that prints replies and status lines. `write()` prints text as it is (sanitized), `notice()` prints a bracketed status line, and `blank()`, `end_line()` and `clear()` do what they say. It tracks `at_line_start`, so a status line starts a new line first if a reply was cut off mid-line. `sanitize()` strips control characters and bidi overrides, since replies and notices carry text from the model or server. `/clear` only prints its escape codes when stdout is a terminal.
- **`Repl`** owns the input loop and dispatch, and prints through `self.console`.
  - `handle()` splits input at the first space: the first word picks a command, and the rest (stripped, `""` if none) is passed as `args`. A single word starting with `/` that isn't a command shows "Unknown command" instead of reaching the model. Anything else goes to `ask()`.
  - `ask()` runs one turn, passing `console.write` and `console.notice` to the agent, then shows any `TurnError`, `Agent error: ...` or `Cancelled` as a notice. Ctrl-C cancels just that turn, and the loop continues.
  - Slash commands live in the module-level `COMMANDS` registry in `commands.py` as `Command(handler, description)`; `Repl.commands` is that dict, matched case-insensitively, and `/help` lists it. A handler is `handler(repl: Repl, args: str) -> str | None`: it returns a short status message, which `handle()` shows as a notice followed by a blank line, or `None` for no message. To add a command, write a handler and decorate it with `@command("/name", "description")`; stack decorators for aliases (`/exit` and `/quit`). Registering a name twice raises `ValueError` at import.
  - `/model` and `/effort` change `repl.agent.config` for the rest of the session; `/effort` validates against the SDK's `ReasoningEffort` (`EFFORTS` lists its values). `/retry` calls `context.rewind()` and resends the prompt it returns.

## Conventions

- The code uses Python 3.14 syntax, such as the unparenthesized `except KeyboardInterrupt, EOFError:` (PEP 758). ruff takes its target from `requires-python`; `[tool.ruff]` only sets `src = ["src", "tests"]` so `muagent` and `fake_api` sort as first-party imports.

- Google-style docstrings (`Args:` / `Returns:` / `Raises:`) on functions.
- Output is a plain terminal: the system prompt forbids LaTeX in favor of Unicode math. Keep that constraint if editing the prompt.
