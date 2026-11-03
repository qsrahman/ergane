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

To use [MCP servers](#mcp-servers), install the extra too: `uv sync --extra mcp` (or `pip install 'muagent[mcp]'`).

Put your settings in `~/.config/muagent/.env` (or `$XDG_CONFIG_HOME/muagent/.env`). A `.env` in the directory you run `muagent` from is not read, so a project you open cannot change where your API key and prompts are sent.

```dotenv
OPENAI_API_KEY=sk-...
# Optional:
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-5.4-mini
# STREAMING=true
# MAX_CONTEXT_TOKENS=65536
# REASONING_EFFORT=low
# MAX_TURNS=25
```

Environment variables take precedence over the file. Keys that aren't settings are ignored.

| Variable             | Default               | Description                                                                                                                                                    |
| -------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OPENAI_API_KEY`     | —                     | API key (required)                                                                                                                                             |
| `OPENAI_BASE_URL`    | OpenAI default        | Base URL for an OpenAI-compatible API                                                                                                                          |
| `OPENAI_MODEL`       | `gpt-5.4-mini`        | Model name                                                                                                                                                     |
| `STREAMING`          | `true`                | Stream tokens as they arrive                                                                                                                                   |
| `MAX_CONTEXT_TOKENS` | `65536`               | Token budget for the conversation history sent each turn. It excludes the system instructions and the reply, so keep it well below the model's context window. |
| `REASONING_EFFORT`   | unset (model default) | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`. Leave unset for non-reasoning models, which reject it.                                           |
| `MAX_TURNS`          | `25`                  | Most model calls for one prompt: the cap on the tool loop.                                                                        |

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
| `/tools`          | List the tools offered to the model, marking the read-only ones |
| `/model [NAME]`   | Show the model, or switch to `NAME` for the rest of the session                  |
| `/effort [LEVEL]` | Show the reasoning effort, or set it (`none` … `max`, or `default` to leave it out) |
| `/reset`          | Clear the conversation history and start fresh                                   |
| `/clear`          | Clear the terminal screen; the history is kept                                   |
| `/exit`, `/quit`  | Quit                                                                             |

Command names are case-insensitive. A mistyped command such as `/rest` is reported instead of being sent to the model. Pressing Ctrl-C while a response is generating cancels only that turn.

## Tools

The model can call tools. When it does, `muagent` prints the call, such as `[add({"a":17,"b":25})]`, and sends the result back to the model, which says what it found, repeating until the model answers. A turn that grows past `MAX_CONTEXT_TOKENS` while it runs is stopped. If a call fails, the error text goes back to the model so it can correct the call. The Responses API must support function calling; set `TOOLS=off` if your model does not.

Only two placeholder tools exist so far, `add` and `get_time`, which stand in until the real tools arrive. Try: *"What is 17 plus 25, and what time is it in Tokyo?"*

A tool is a plain function with a docstring (its description, written for the model), type hints (its parameters) and `@tool`:

```python
@tool
def add(a: int, b: int) -> str:
    """Add two integers and return the sum."""
    return str(a + b)
```

## MCP servers

`muagent` can use the tools of [MCP](https://modelcontextprotocol.io) servers, either a local program it runs (stdio) or a server it connects to over HTTP (streamable HTTP). List them in `~/.config/muagent/mcp.json` (or `$XDG_CONFIG_HOME/muagent/mcp.json`), in the usual format:

```json
{
  "mcpServers": {
    "files": {
      "command": "my-files-server",
      "args": ["--root", "/path/to/project"],
      "env": { "SOME_TOKEN": "${SOME_TOKEN}" },
      "cwd": "${HOME}"
    },
    "notes": { "command": "my-notes-server", "read_only": true },
    "remote": {
      "url": "https://example.com/mcp",
      "headers": { "Authorization": "Bearer ${REMOTE_TOKEN}" },
      "enable": false
    }
  }
}
```

| Key         | Description                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------ |
| `type`      | `"local"` for a `command`, `"remote"` for a `url`. Optional; if given it must agree with them.               |
| `enable`    | `false` keeps the entry but does not start the server. Default `true`.                                       |
| `command`   | The program that runs the server. Give this or `url`, not both.                                              |
| `args`      | Its arguments                                                                                                |
| `cwd`       | The directory it runs in. Default: where you run `muagent`.                                                  |
| `env`       | Environment variables for it. The server gets only a short list (`PATH`, `HOME`, ...) and these, so it does not see your `OPENAI_API_KEY`. |
| `url`       | The address of a server to connect to, instead of a `command`. `https://`, or `http://` for `localhost`, `127.0.0.1` and `[::1]` only. |
| `headers`   | Sent with every request to the `url`, for an `Authorization` or an API key. Not in the URL itself: a user name or password there is refused. |
| `read_only` | `true` lets the server's tools run without asking. Default `false`.                                          |
| `startup_timeout` | Seconds to wait for the server to start and list its tools. Default `20`.                              |
| `call_timeout` | Seconds a call to one of its tools may take before it is given up on. Default `60`.                       |

Write `${NAME}` in `args`, `env`, `headers` or `cwd` to use the environment variable `NAME`, so that keys stay out of the file: `"env": { "API_KEY": "${MY_API_KEY}" }`. A server whose variable is not set is skipped, with a notice that names the variable. The server gets only the variables you list, not the rest of your environment.

The names are letters, digits, `_` and `-`. A server's tools are offered to the model as `name__tool`; `/tools` lists them. The servers start at the same time, so a few slow ones do not add up. A server that fails to start is reported and left out; one that stops while you work makes its tools return an error that names it, and the others carry on. The stderr of a program goes to `~/.local/state/muagent/NAME.log` (or `$XDG_STATE_HOME/muagent/NAME.log`), where you will find why it failed.

A few things to know before you add a server:

- **A server is a program that runs as you, when `muagent` starts.** Only the user-level `mcp.json` is read, never one from the directory you are in, so a project you open cannot start programs.
- **Every MCP tool asks before it runs**, showing the full arguments, unless you set `read_only` for that server. A server's own claim that a tool is read-only is ignored: it is not something `muagent` can check.
- **A server at a URL receives what the model sends it**: the arguments of every call, which can hold file contents or other things from your conversation. Connect only to servers you trust with that. `mcp.json` holds the headers in plain text, so keep it to yourself (`chmod 600`). Redirects to another host are not followed, so the headers go only where you pointed them.
- **Tool descriptions come from the server** and are read by the model. Add servers you trust.
- The list of tools is read once, at start.

## Tests

```bash
uv run pytest
```

The tests run against a local fake of the Responses API. They ignore your `.env` and user config file, and make no external network calls.

## How it works

- Requests are sent with `store=False`, so the server keeps no conversation state. The full history is resent on each turn.
- A turn, tool calls and their results included, is added to the history once the final response completes with text. An incomplete response keeps the partial reply you already saw; failed or empty responses leave the history unchanged.
- When the history exceeds `MAX_CONTEXT_TOKENS`, the oldest turns are dropped first. Tokens are estimated at about 4 bytes of UTF-8 per token, which works with any backend's tokenizer.
