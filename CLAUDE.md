# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`muagent` is a minimal interactive terminal chat agent built on the OpenAI **Responses API**. The whole program lives in `main.py`. No linter or formatter is configured.

## Commands

Uses `uv` with Python 3.14 (`.python-version`, `requires-python >= 3.14`).

```bash
uv sync              # install dependencies (including pytest) into .venv
uv run main.py       # start the REPL (/exit or /quit to leave)
uv run pytest        # run the tests
uv run pytest tests/test_main.py::TestTrimContext::test_drops_oldest_turns_first  # single test
```

## Tests

- `tests/conftest.py` runs a fake Responses API (`fake_server` fixture) on a local port, using the real SDK client and streaming parser. The last input message picks the outcome: `"incomplete"`, `"failed"`, or anything else for a completed `"Hello world"` reply. Request bodies are recorded in `fake_server.requests`.
- The autouse `isolate_config` fixture keeps `.env` and the config environment variables out of `Config`. Note that `Config.model_validate(...)` still reads `.env`, so don't remove the fixture.
- Tests inject config by monkeypatching `main.get_config` (see the `use_config` fixture in `test_main.py`).

## Configuration

Settings come from environment variables or `.env` next to `main.py`, loaded by `Config` (pydantic-settings):

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `OPENAI_BASE_URL` | `None` | for OpenAI-compatible endpoints |
| `OPENAI_MODEL` | `gpt-5.4-mini` | also selects the tiktoken encoding (falls back to `o200k_base`) |
| `STREAMING` | `true` | stream deltas vs. single `responses.create` |
| `MAX_CONTEXT_TOKENS` | `65536` | client-side history budget |

`get_config()` is `@cache`d; validation errors print per-field messages and exit with status 1.

## Architecture

- **Stateless API calls:** every request uses `store=False`; the full conversation is resent each turn as a `ResponseInputParam` list held in `main()`. The system prompt plus environment info (OS, current date) is passed via `instructions`, not stored in the history.
- **Transactional history:** `run_turn` builds `turn_context` as a copy of the history plus the new user message, and only commits it back to `context` (`context[:] = ...`) after a `completed` response. Failed/incomplete responses or exceptions leave history untouched — preserve this invariant when changing the turn flow.
- **Context trimming:** `trim_context` estimates tokens by tiktoken-encoding the JSON-serialized context, and drops whole turns from the front (everything before the second `user` message) until under budget. It never drops the most recent user turn, so a single oversized prompt can still exceed the limit.
- **REPL errors:** Ctrl-C during a turn cancels just that turn; other exceptions are printed as `Agent error: ...` and the loop continues.

## Conventions

- Targets Python 3.14 syntax, e.g. the unparenthesized `except KeyboardInterrupt, EOFError:` (PEP 758) is intentional, not a bug.
- Google-style docstrings (`Args:` / `Returns:` / `Raises:`) on functions.
- Output is a plain terminal: the system prompt forbids LaTeX in favor of Unicode math. Keep that constraint if editing the prompt.
