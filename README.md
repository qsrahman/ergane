# muagent

A minimal terminal chat agent built on the OpenAI [Responses API](https://platform.openai.com/docs/api-reference/responses). It works with OpenAI and with any compatible endpoint.

## Features

- Interactive REPL with line editing and history (via `readline`, where available)
- Streaming output, which can be turned off
- Client-side conversation history with token-based trimming, so long sessions stay within a configurable context budget
- Works with any OpenAI-compatible base URL
- Tool calling: the model can call tools, sees their errors, and can correct itself (only two demo tools so far)

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
# TOOLS=read
# MAX_TURNS=25
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
| `MAX_TURNS`          | `25`                  | Most model calls for one prompt: the cap on the tool loop.                                                                        |
| `TOOLS`              | `read`                | `off`, `read`, `write` or `execute`: the most access the tools offered to the model may have. Use `off` with a model that does not support tool calling. |

## Usage

```bash
uv run muagent
```

`uv run python -m muagent` also works.

Type a prompt at `>` and press Enter. To quit, use `/exit`, `/quit`, Ctrl-D, or Ctrl-C at the prompt.

| Command           | Effect                                                                           |
| ----------------- | -------------------------------------------------------------------------------- |
| `/help`           | List the commands                                                                |
| `/retry`          | Resend the last prompt; if its reply was saved, it is replaced                   |
| `/undo`           | Remove the last exchange from the history                                        |
| `/tools [LEVEL]`  | Show the tools offered to the model, or set their level (`off`, `read`, `write`, `execute`) |
| `/model [NAME]`   | Show the model, or switch to `NAME` for the rest of the session                  |
| `/effort [LEVEL]` | Show the reasoning effort, or set it (`none` … `max`, or `default` to leave it out) |
| `/reset`          | Clear the conversation history and start fresh                                   |
| `/clear`          | Clear the terminal screen; the history is kept                                   |
| `/exit`, `/quit`  | Quit                                                                             |

Command names are case-insensitive. A mistyped command such as `/rest` is reported instead of being sent to the model. Pressing Ctrl-C while a response is generating cancels only that turn.

## Tools

The model can call tools. When it does, `muagent` prints the call and a short result, such as `[add({"a":17,"b":25})]` then `[→ 42]`, and sends the result back, repeating until the model answers. A turn that grows past `MAX_CONTEXT_TOKENS` while it runs is stopped. If a call fails, the error text goes back to the model so it can correct the call. The Responses API must support function calling; set `TOOLS=off` if your model does not.

Only two placeholder tools exist so far, `add` and `get_time`, which stand in until the real tools arrive. Try: *"What is 17 plus 25, and what time is it in Tokyo?"*

A tool is a plain function with a docstring (its description, written for the model), type hints (its parameters) and `@tool`:

```python
@tool
def add(a: int, b: int) -> str:
    """Add two integers and return the sum."""
    return str(a + b)
```

## Tests

```bash
uv run pytest
```

The tests run against a local fake of the Responses API. They ignore your `.env` and user config file, and make no external network calls.

## How it works

- Requests are sent with `store=False`, so the server keeps no conversation state. The full history is resent on each turn.
- A turn, tool calls and their results included, is added to the history once the final response completes with text. An incomplete response keeps the partial reply you already saw; failed or empty responses leave the history unchanged.
- When the history exceeds `MAX_CONTEXT_TOKENS`, the oldest turns are dropped first. Tokens are estimated at about 4 bytes of UTF-8 per token, which works with any backend's tokenizer.
