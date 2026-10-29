# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`buddy` is a minimal interactive terminal chat agent built on the OpenAI **Responses API**. It's a `src`-layout package: the whole program lives in `src/buddy/main.py`.

## Commands

Uses `uv` with Python 3.14 (`.python-version`, `requires-python >= 3.14`).

```bash
uv sync              # install dependencies (including pytest) into .venv
uv run buddy       # start the REPL; same as `uv run python -m buddy.main`
uv run pytest        # run the tests
uv run pytest tests/test_main.py::TestRepl::test_strips_input_and_skips_blank_lines  # single test
uv run ruff check . && uv run ruff format --check .   # lint and format check
uvx pyright --pythonpath .venv/bin/python src tests   # type check (pyright isn't a dependency)
```

## Tests

- `tests/fake_api.py` is a fake Responses API, and the `fake_server` fixture in `tests/conftest.py` runs it on a local port, using the real SDK client and streaming parser. The last input message picks the outcome: `"incomplete"`, `"failed"`, `"empty"` (completed with no output), `"refusal"`, `"truncated"` (a stream with no terminal event), or anything else for a completed `"Hello world"` reply. Request bodies are recorded in `fake_server.requests`.
- The autouse `isolate_config` fixture keeps `.env` and the config environment variables out of `Config`. Note that `Config.model_validate(...)` still reads `.env`, so don't remove the fixture.
- Tests build objects directly instead of monkeypatching. `make_config(...)` builds a `Config`, and the `make_agent` fixture builds an `Agent` wired to the fake server. `Agent` tests check return values and `TurnError`, not printed output. `TestRepl` replaces `agent.run_turn` on a single instance, and `TestReplRendering` drives a real `Agent` through `run_repl()`.

The package is built with `uv_build`, which expects `src/buddy/__init__.py`. The `buddy` script is `buddy.main:main` in `[project.scripts]`, and `uv sync` installs the package in editable mode. Tests import it as `from buddy import main`.

## Configuration

Settings come from environment variables or a `.env` file in the current working directory, loaded by `Config` (pydantic-settings):

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `OPENAI_BASE_URL` | `None` | for OpenAI-compatible endpoints |
| `OPENAI_MODEL` | `gpt-5.4-mini` | |
| `STREAMING` | `true` | stream deltas vs. single `responses.create` |
| `MAX_CONTEXT_TOKENS` | `65536` | client-side history budget; excludes instructions and output |
| `REASONING_EFFORT` | unset | SDK `ReasoningEffort`; unset sends no `reasoning` (`omit`), since non-reasoning models reject it |

`Config` itself only validates and raises `ValidationError`. `main()` prints the per-field errors and exits with status 1.

## Architecture

`main()` only loads the config, reporting errors, and wires the pieces together: `Repl(Agent(config)).run()`. Dependencies are passed in through constructors (`Agent(config, client=None)`, `Repl(agent)`); there is no global config lookup.

- **`Context`** holds the history (`items`) and the token budget. Every request uses `store=False`, so the full history is resent each turn.
  - `draft(prompt)` returns a copy of the history plus the new user message, trimmed to budget, without changing `items`.
  - `commit(draft, reply)` is the only way a turn becomes history. `Agent` calls it after a `completed` (or status-less, as some compatible servers send) response with non-empty `output_text`, and also for an `incomplete` response with text, so the partial reply the user saw is kept. Failed or empty responses and exceptions leave history untouched, so user and assistant messages always alternate. Preserve this invariant when changing the turn flow.
  - Trimming, inside `draft()`, estimates each item's tokens once (UTF-8 bytes of its JSON ÷ 4, tokenizer-agnostic) and drops whole turns from the front, each up to the next `user` message, until under budget. It never drops the most recent user turn, so a single oversized prompt can still exceed the limit.
- **`Agent`** owns the config, the client and the `Context`, and does no terminal I/O. `run_turn(prompt, on_text, on_notice)` sends one turn and passes reply text to `on_text`: each delta when streaming, or the whole reply once otherwise (before the status check in both modes, so partial text is always shown). Status messages such as trimming go to `on_notice`. It raises `TurnError` for incomplete, failed or empty responses, with all status checks in `_check()`. `create` and `stream` each get explicit keyword arguments, so pyright checks them; `reasoning` is `omit` unless `REASONING_EFFORT` is set. `stream()` is kept over `create(stream=True)` because it rebuilds the final output when a compatible server leaves it out of `response.completed`. The system prompt plus environment info (OS, current date) is passed via `instructions`, not stored in the history. When streaming, `run_turn` takes the final response from the terminal stream event, because `get_final_response()` raises on incomplete or failed streams.
- **`Repl`** owns all terminal I/O. `commands` maps each slash command to `(handler, description)`, matched case-insensitively, and `/help` lists them from that dict. To add a command, add one entry. A single word starting with `/` that isn't a command prints "Unknown command" instead of reaching the model. Anything else goes to `ask()`, which prints the streamed reply via `write()`; `notice()` prints bracketed status lines (the trim notice from `on_notice`, any `TurnError`, `Agent error: ...`, `Cancelled`) and starts a new line first if the reply was cut off mid-line (`at_line_start`). Ctrl-C cancels just that turn, and the loop continues.

## Conventions

- ruff's `target-version = "py313"` is deliberate, even though `requires-python` is 3.14. With a 3.14 target, `ruff format` strips the parentheses from `except (A, B):`, which is the PEP 758 style; this code keeps them.

- Google-style docstrings (`Args:` / `Returns:` / `Raises:`) on functions.
- Output is a plain terminal: the system prompt forbids LaTeX in favor of Unicode math. Keep that constraint if editing the prompt.
