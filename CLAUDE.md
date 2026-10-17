# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`muagent` is a minimal interactive terminal chat agent built on the OpenAI **Responses API**. The whole program lives in `main.py`. No linter or formatter is configured.

## Commands

Uses `uv` with Python 3.14 (`.python-version`, `requires-python >= 3.14`).

```bash
uv sync              # install dependencies (including pytest) into .venv
uv run muagent       # start the REPL; same as `uv run main.py`
uv run pytest        # run the tests
uv run pytest tests/test_main.py::TestRepl::test_strips_input_and_skips_blank_lines  # single test
```

## Tests

- `tests/conftest.py` runs a fake Responses API (`fake_server` fixture) on a local port, using the real SDK client and streaming parser. The last input message picks the outcome: `"incomplete"`, `"failed"`, `"empty"` (completed with no output), or anything else for a completed `"Hello world"` reply. Request bodies are recorded in `fake_server.requests`.
- The autouse `isolate_config` fixture keeps `.env` and the config environment variables out of `Config`. Note that `Config.model_validate(...)` still reads `.env`, so don't remove the fixture.
- Tests build objects directly instead of monkeypatching. `make_config(...)` builds a `Config`, the `make_agent` fixture builds an `Agent` wired to the fake server, and REPL tests replace `agent.run_turn` on a single instance.

The `muagent` script comes from `[project.scripts]` and is built with hatchling, with only `main.py` in the wheel. `uv sync` installs it in editable mode. A non-editable install, such as `uv tool install`, would look for `.env` in site-packages, because `Config` finds `.env` relative to `main.py`.

## Configuration

Settings come from environment variables or `.env` next to `main.py`, loaded by `Config` (pydantic-settings):

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `OPENAI_BASE_URL` | `None` | for OpenAI-compatible endpoints |
| `OPENAI_MODEL` | `gpt-5.4-mini` | |
| `STREAMING` | `true` | stream deltas vs. single `responses.create` |
| `MAX_CONTEXT_TOKENS` | `65536` | client-side history budget; excludes instructions and output |

`Config.load()` prints per-field validation errors and exits with status 1.

## Architecture

`main()` only wires the pieces together: `Repl(Agent(Config.load())).run()`. Dependencies are passed in through constructors (`Agent(config, client=None)`, `Repl(agent)`); there is no global config lookup.

- **`Context`** holds the history (`items`) and the token budget. Every request uses `store=False`, so the full history is resent each turn.
  - `draft(prompt)` returns a copy of the history plus the new user message, trimmed to budget, without changing `items`.
  - `commit(draft, reply)` is the only way a turn becomes history. `Agent` calls it only after a `completed` response with non-empty `output_text`. Failed, incomplete or empty responses and exceptions leave history untouched, so user and assistant messages always alternate. Preserve this invariant when changing the turn flow.
  - Trimming (`_trim_start`) estimates each item's tokens once (UTF-8 bytes of its JSON ÷ 4, tokenizer-agnostic) and drops whole turns from the front, each up to the next `user` message, until under budget. It never drops the most recent user turn, so a single oversized prompt can still exceed the limit.
- **`Agent`** owns the config, the client and the `Context`. `run_turn(prompt)` sends one turn and prints the reply. The system prompt plus environment info (OS, current date) is passed via `instructions`, not stored in the history. `_stream()` takes the final response from the terminal stream event, because `get_final_response()` raises on incomplete or failed streams.
- **`Repl`** owns terminal I/O. Slash commands are a `commands` dict mapping each command to a method (`/reset`, `/clear`, `/exit`, `/quit`), matched case-insensitively. To add a command, add an entry. Anything else goes to `ask()`, where Ctrl-C cancels just that turn and other exceptions are printed as `Agent error: ...` while the loop continues.

## Conventions

- Google-style docstrings (`Args:` / `Returns:` / `Raises:`) on functions.
- Output is a plain terminal: the system prompt forbids LaTeX in favor of Unicode math. Keep that constraint if editing the prompt.
