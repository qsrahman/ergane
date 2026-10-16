# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`muagent` is a minimal interactive terminal chat agent built on the OpenAI **Responses API**. The whole program lives in `main.py`. No linter or formatter is configured.

## Commands

Uses `uv` with Python 3.14 (`.python-version`, `requires-python >= 3.14`).

```bash
uv sync              # install dependencies (including pytest) into .venv
uv run muagent       # start the REPL (/exit or /quit to leave); same as `uv run main.py`
uv run pytest        # run the tests
uv run pytest tests/test_main.py::TestRepl::test_strips_input_and_skips_blank_lines  # single test
```

## Tests

- `tests/conftest.py` runs a fake Responses API (`fake_server` fixture) on a local port, using the real SDK client and streaming parser. The last input message picks the outcome: `"incomplete"`, `"failed"`, `"empty"` (completed with no output), or anything else for a completed `"Hello world"` reply. Request bodies are recorded in `fake_server.requests`.
- The autouse `isolate_config` fixture keeps `.env` and the config environment variables out of `Config`. Note that `Config.model_validate(...)` still reads `.env`, so don't remove the fixture.
- Tests inject config by monkeypatching `main.get_config` (see the `use_config` fixture in `test_main.py`).

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

`get_config()` is `@cache`d; validation errors print per-field messages and exit with status 1.

## Architecture

- **Stateless API calls:** every request uses `store=False`; the full conversation is resent each turn as a `ResponseInputParam` list held in `main()`. The system prompt plus environment info (OS, current date) is passed via `instructions`, not stored in the history.
- **Transactional history:** `run_turn` builds `turn_context` as a copy of the history plus the new user message, and only commits it back to `context` (`context[:] = ...`) after a `completed` response with non-empty `output_text`. Failed, incomplete or empty responses and exceptions leave history untouched, so user and assistant messages always alternate. Preserve this invariant when changing the turn flow.
- **Context trimming:** `trim_context` estimates each item's tokens once (UTF-8 bytes of its JSON ÷ 4, tokenizer-agnostic) and drops whole turns from the front (each up to the next `user` message) until under budget. It never drops the most recent user turn, so a single oversized prompt can still exceed the limit.
- **REPL errors:** Ctrl-C during a turn cancels just that turn; other exceptions are printed as `Agent error: ...` and the loop continues.

## Conventions

- Targets Python 3.14 syntax, e.g. the unparenthesized `except KeyboardInterrupt, EOFError:` (PEP 758) is intentional, not a bug.
- Google-style docstrings (`Args:` / `Returns:` / `Raises:`) on functions.
- Output is a plain terminal: the system prompt forbids LaTeX in favor of Unicode math. Keep that constraint if editing the prompt.
