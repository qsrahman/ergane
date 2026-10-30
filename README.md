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

Put your settings in `~/.config/muagent/.env` (or `$XDG_CONFIG_HOME/muagent/.env`), or in a `.env` in the directory you run `muagent` from, which overrides the user file. The project's `.env` is git-ignored.

```dotenv
OPENAI_API_KEY=sk-...
# Optional:
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-5.4-mini
# STREAMING=true
# MAX_CONTEXT_TOKENS=65536
# REASONING_EFFORT=low
```

Environment variables take precedence over both files. Keys that aren't settings are ignored.

| Variable             | Default               | Description                                                                                                                                                    |
| -------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OPENAI_API_KEY`     | —                     | API key (required)                                                                                                                                             |
| `OPENAI_BASE_URL`    | OpenAI default        | Base URL for an OpenAI-compatible API                                                                                                                          |
| `OPENAI_MODEL`       | `gpt-5.4-mini`        | Model name                                                                                                                                                     |
| `STREAMING`          | `true`                | Stream tokens as they arrive                                                                                                                                   |
| `MAX_CONTEXT_TOKENS` | `65536`               | Token budget for the conversation history sent each turn. It excludes the system instructions and the reply, so keep it well below the model's context window. |
| `REASONING_EFFORT`   | unset (model default) | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`. Leave unset for non-reasoning models, which reject it.                                           |
| `MAX_TURNS`          | `25`                  | Reserved: will cap model calls per prompt once tool calling is added. Not used yet.                                                                        |

## Usage

```bash
uv run muagent
```

`uv run python -m muagent.main` also works.

Type a prompt at `>` and press Enter. To quit, use `/exit`, `/quit`, Ctrl-D, or Ctrl-C at the prompt.

| Command          | Effect                                         |
| ---------------- | ---------------------------------------------- |
| `/help`          | List the commands                              |
| `/reset`         | Clear the conversation history and start fresh |
| `/clear`         | Clear the terminal screen; the history is kept |
| `/exit`, `/quit` | Quit                                           |

Commands are case-insensitive. A mistyped command such as `/rest` is reported instead of being sent to the model. Pressing Ctrl-C while a response is generating cancels only that turn.

## Tests

```bash
uv run pytest
```

The tests run against a local fake of the Responses API. They ignore your `.env` and user config file, and make no external network calls.

## How it works

- Requests are sent with `store=False`, so the server keeps no conversation state. The full history is resent on each turn.
- A turn is added to the history once the response completes with text. An incomplete response keeps the partial reply you already saw; failed or empty responses leave the history unchanged.
- When the history exceeds `MAX_CONTEXT_TOKENS`, the oldest turns are dropped first. Tokens are estimated at about 4 bytes of UTF-8 per token, which works with any backend's tokenizer.
