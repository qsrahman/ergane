# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`muagent` is a minimal interactive terminal chat agent built on the OpenAI **Responses API**. It's a `src`-layout package: the whole program lives in `src/muagent/main.py`.

## Commands

Uses `uv` with Python 3.14 (`.python-version`, `requires-python >= 3.14`).

```bash
uv sync              # install dependencies (including pytest) into .venv
uv run muagent      # start the REPL; same as `uv run python -m muagent.main`
uv run pytest        # run the tests
uv run pytest tests/test_main.py::TestRepl::test_strips_input_and_skips_blank_lines  # single test
uv run ruff check . && uv run ruff format --check .   # lint and format check
uv run pyright src tests   # type check (pyright is a dev dependency)
```

## Tests

- `tests/fake_api.py` is a fake Responses API, and the `fake_server` fixture in `tests/conftest.py` runs it on a local port, using the real SDK client and streaming parser. The last input message picks the outcome: `"incomplete"`, `"failed"`, `"empty"` (completed with no output), `"refusal"`, `"truncated"` (a stream with no terminal event), or anything else for a completed `"Hello world"` reply. Request bodies are recorded in `fake_server.requests`.
- The autouse `isolate_config` fixture runs each test in an empty `tmp_path` that is also `XDG_CONFIG_HOME`, and clears the config environment variables, so neither the project's `.env` nor the user config file leaks in. `TestLoadConfig` writes its config files into that `tmp_path`.
- Tests build objects directly instead of monkeypatching. `make_config(...)` builds a `Config`, and the `make_agent` fixture builds an `Agent` wired to the fake server. `Agent` tests check return values and `TurnError`, not printed output. `TestRepl` replaces `agent.run_turn` on a single instance, and `TestReplRendering` drives a real `Agent` through `run_repl()`.

The package is built with `uv_build`, which expects `src/muagent/__init__.py`. The `muagent` script is `muagent.main:main` in `[project.scripts]`, and `uv sync` installs the package in editable mode. Tests import it as `from muagent import main`.

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

`main()` only loads the config, reporting errors, and wires the pieces together: `Repl(Agent(config)).run()`. Dependencies are passed in through constructors (`Agent(config, client=None)`, `Repl(agent)`); there is no global config lookup.

- **`Context`** holds the history (`items`) and the token budget. Every request uses `store=False`, so the full history is resent each turn.
  - `draft(prompt)` returns a copy of the history plus the new user message, trimmed to budget, without changing `items`.
  - `commit(draft, reply)` is the only way a turn becomes history. `Agent` calls it after a `completed` (or status-less, as some compatible servers send) response with non-empty `output_text`, and also for an `incomplete` response with text, so the partial reply the user saw is kept. Failed or empty responses and exceptions leave history untouched, so user and assistant messages always alternate. Preserve this invariant when changing the turn flow.
  - Trimming, inside `draft()`, estimates each item's tokens once (UTF-8 bytes of its JSON ÷ 4, tokenizer-agnostic) and drops whole turns from the front, each up to the next `user` message, until under budget. It never drops the most recent user turn, so a single oversized prompt can still exceed the limit.
- **`Agent`** owns the config, the client and the `Context`, and does no terminal I/O. `run_turn(prompt, on_text, on_notice)` sends one turn and passes reply text to `on_text`: each delta when streaming, or the whole reply once otherwise (before the status check in both modes, so partial text is always shown). Status messages such as trimming go to `on_notice`. It raises `TurnError` for incomplete, failed or empty responses, with all status checks in `_check()`. `create` and `stream` each get explicit keyword arguments, so pyright checks them; `reasoning` is the SDK's `omit` sentinel when `REASONING_EFFORT` is unset, which leaves it out of the request. `stream()` is kept over `create(stream=True)` because it rebuilds the final output when a compatible server leaves it out of `response.completed`. The system prompt plus environment info (OS, current date) is passed via `instructions`, not stored in the history. When streaming, `run_turn` takes the final response from the terminal stream event, because `get_final_response()` raises on incomplete or failed streams.
- **`Repl`** owns all terminal I/O. Slash commands live in the module-level `COMMANDS` registry as `Command(handler, description)`; `Repl.commands` is that dict, matched case-insensitively, and `/help` lists it. To add a command, write a module-level `handler(repl: Repl) -> None` and decorate it with `@command("/name", "description")`; stack decorators for aliases (`/exit` and `/quit`). Registering a name twice raises `ValueError` at import. A single word starting with `/` that isn't a command prints "Unknown command" instead of reaching the model. Anything else goes to `ask()`, which prints the streamed reply via `write()`; `notice()` prints bracketed status lines (the trim notice from `on_notice`, any `TurnError`, `Agent error: ...`, `Cancelled`) through `write()` too, starting a new line first if the reply was cut off mid-line (`at_line_start`). `write()` runs everything through `sanitize()`, which strips control characters and bidi overrides, since replies and notices carry text from the model or server. Ctrl-C cancels just that turn, and the loop continues.

## Conventions

- The code uses Python 3.14 syntax, such as the unparenthesized `except KeyboardInterrupt, EOFError:` (PEP 758). ruff takes its target from `requires-python`; `[tool.ruff]` only sets `src = ["src", "tests"]` so `muagent` and `fake_api` sort as first-party imports.

- Google-style docstrings (`Args:` / `Returns:` / `Raises:`) on functions.
- Output is a plain terminal: the system prompt forbids LaTeX in favor of Unicode math. Keep that constraint if editing the prompt.
