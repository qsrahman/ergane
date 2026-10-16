# muagent

A minimal terminal chat agent built on the OpenAI [Responses API](https://platform.openai.com/docs/api-reference/responses). It works with OpenAI and with any compatible endpoint.

## Features

- Interactive REPL with line editing and history (via `readline`, where available)
- Streaming output, which can be turned off
- Client-side conversation history with token-based trimming, so long sessions stay within a configurable context budget
- Works with any OpenAI-compatible base URL

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

Create a `.env` file in the project root. It is git-ignored.

```dotenv
OPENAI_API_KEY=sk-...
# Optional:
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-5.4-mini
# STREAMING=true
# MAX_CONTEXT_TOKENS=65536
```

Environment variables take precedence over `.env`.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | API key (required) |
| `OPENAI_BASE_URL` | OpenAI default | Base URL for an OpenAI-compatible API |
| `OPENAI_MODEL` | `gpt-5.4-mini` | Model name |
| `STREAMING` | `true` | Stream tokens as they arrive |
| `MAX_CONTEXT_TOKENS` | `65536` | Token budget for the conversation history sent each turn. It excludes the system instructions and the reply, so keep it well below the model's context window. |

## Usage

```bash
uv run muagent
```

`uv run main.py` also works.

Type a prompt at `>` and press Enter. To quit, use `/exit`, `/quit`, Ctrl-D, or Ctrl-C at the prompt. Pressing Ctrl-C while a response is generating cancels only that turn.

## Tests

```bash
uv run pytest
```

The tests run against a local fake of the Responses API. They ignore your `.env` and make no external network calls.

## How it works

- Requests are sent with `store=False`, so the server keeps no conversation state. The full history is resent on each turn.
- A turn is added to the history only after the response completes with text. Failed, incomplete or empty responses leave the history unchanged.
- When the history exceeds `MAX_CONTEXT_TOKENS`, the oldest turns are dropped first. Tokens are estimated at about 4 bytes of UTF-8 per token, which works with any backend's tokenizer.
