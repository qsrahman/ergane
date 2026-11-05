# muagent

A minimal terminal chat agent built on the OpenAI [Responses API](https://platform.openai.com/docs/api-reference/responses). It works with OpenAI and with any compatible endpoint.

## Features

- Interactive REPL built on [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/): line editing, a history that is kept between sessions (with a grey suggestion from it as you type), Tab completion of the slash commands and of `/effort`'s levels, and a pasted block (a stack trace, a function) is one prompt
- Streaming output, which can be turned off
- Client-side conversation history with token-based trimming, so long sessions stay within a configurable context budget
- Works with any OpenAI-compatible base URL
- Tool calling: the model can call the tools of your [MCP servers](#mcp-servers), sees their errors, and can correct itself

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
| `MAX_CONTEXT_TOKENS` | `65536`               | Token budget for one request: the system instructions, the definitions of the tools and the conversation history together (not the reply). Old turns are dropped to fit it, so many or long tool definitions leave less room for the history. Keep it well below the model's context window. |
| `REASONING_EFFORT`   | unset (model default) | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`. Leave unset for non-reasoning models, which reject it.                                           |
| `MAX_TURNS`          | `25`                  | Most model calls for one prompt: the cap on the tool loop.                                                                        |

## Usage

```bash
uv run muagent
```

`uv run python -m muagent` also works.

Type a prompt at `>` and press Enter. Up and Down recall earlier prompts, which are kept in `~/.local/state/muagent/history` (or `$XDG_STATE_HOME/muagent/history`) in plain text; delete the file to forget them. The directory (which also holds the server logs) is made readable by you only (`0700`, the file `0600`), since a prompt can hold whatever you pasted. Paste several lines and they stay one prompt; press Enter to send it. To add a line by hand, press Alt-Enter (Esc, then Enter) or Ctrl-J (the new lines start with `. `); Shift-Enter does it too in a terminal that sends it as its own key (kitty's keyboard protocol, or xterm's modifyOtherKeys), but most send a plain Enter for it, and then it sends the prompt. The question before a tool runs (`Allow … [y/N]`) is read by the same editor, but is not kept in the history. When input or output is redirected (`echo hi | muagent`), plain `input()` reads the prompts instead. To quit, use `/exit`, `/quit`, Ctrl-D, or Ctrl-C twice on an empty prompt: the conversation lives only in memory, so one stray Ctrl-C must not end it. A single Ctrl-C clears what you have typed, or, on an empty line, asks you to press it again. `/help` lists the keys as well as the commands.

| Command           | Effect                                                                           |
| ----------------- | -------------------------------------------------------------------------------- |
| `/help`           | List the commands                                                                |
| `/retry`          | Resend the last prompt; if its reply was saved, it is replaced                   |
| `/undo`           | Remove the last exchange from the history                                        |
| `/tools`          | List the tools offered to the model, by whether they ask first  |
| `/context`        | Show how much of the context budget the next request would use: the instructions, the tool definitions and the history (estimates), and the count the server gave for the last request |
| `/model [NAME]`   | Show the model, or switch to `NAME` for the rest of the session                  |
| `/effort [LEVEL]` | Show the reasoning effort, or set it (`none` … `max`, or `default` to leave it out) |
| `/reset`          | Clear the conversation history and start fresh                                   |
| `/clear`          | Clear the terminal screen; the history is kept                                   |
| `/exit`, `/quit`  | Quit                                                                             |

Command names are case-insensitive. A mistyped command such as `/rest` is reported instead of being sent to the model. Pressing Ctrl-C while a response is generating cancels only that turn; if tools had already run, the turn is kept with a note saying you stopped it, so the model knows what was done.

## Tools

The model can call tools. A tool that only reads runs at once, and `muagent` prints the call as a dimmed line (on a terminal, unless `NO_COLOR` is set), such as `[add({"a": 17, "b": 25})]`, never wider than the terminal: the middle of long values is left out. A tool that can change things asks first, in one line, such as `Allow save {"path": "notes.txt"}? [y/N/a]`; type `y` to allow this call, or `a` to allow that tool for the rest of the session (it then runs like a read-only one, with its call shown as a line, and `/tools` lists it as allowed; restart muagent to take it back). Anything else is a no. The line is never wider than the terminal: if the arguments do not fit, the middle of their long values is left out (the start and the end stay) and the prompt ends `[y/N/a/?]`; type `?` to see all of it, and it asks again. The result goes back to the model, which says what it found, repeating until the model answers. When a turn grows past `MAX_CONTEXT_TOKENS` while it runs, the oldest tool results are replaced by a short placeholder (the calls stay) so that it can go on; only if the latest results alone are over, it is stopped. If a call fails, the error text goes back to the model so it can correct the call. The Responses API must support function calling.

`muagent` has no tools of its own: they come from the MCP servers you list in `mcp.json` (see below). With none, it is a plain chat, and `/tools` says so.

To give it a tool written in Python, make a function with a docstring (its description, written for the model), type hints (its parameters) and `@tool`, and list it in `main()` in `cli.py`, next to the tools of the MCP servers (`Agent(config, tools=[add, *mcp_tools])`):

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
      "env": { "SOME_TOKEN": "${SOME_TOKEN}" }
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
| `command`   | The program that runs the server, in the directory where you run `muagent`. Give this or `url`, not both.    |
| `args`      | Its arguments                                                                                                |
| `env`       | Environment variables for it. The server gets only a short list (`PATH`, `HOME`, ...) and these, so it does not see your `OPENAI_API_KEY`. |
| `url`       | The address of a server to connect to, instead of a `command`. `https://`, or `http://` for `localhost`, `127.0.0.1` and `[::1]` only. |
| `headers`   | Sent with every request to the `url`, for an `Authorization` or an API key. Not in the URL itself: a user name or password there is refused. |
| `read_only` | `true` lets the server's tools run without asking. Default `false`.                                          |
| `tools`     | Offer only these tools of the server, by the server's own names (see `/tools` for what a server has). The definition of every tool is sent with every request, so a server with many tools you do not use costs context and money. A name the server does not have fails the server, with a notice. Default: all. |
| `startup_timeout` | Seconds to wait for the server to start and list its tools. Default `20`.                              |
| `call_timeout` | Seconds a call to one of its tools may take before it is given up on. Default `60`.                       |

Write `${NAME}` in `args`, `env` or `headers` to use the environment variable `NAME`, so that keys stay out of the file: `"env": { "API_KEY": "${MY_API_KEY}" }`. A server whose variable is not set is skipped, with a notice that names the variable. The server gets only the variables you list, not the rest of your environment.

The names are letters, digits, `_` and `-`. A server's tools are offered to the model as `name__tool`; `/tools` lists them. The servers start at the same time, so a few slow ones do not add up. A server that starts is not announced (the first line shows how many tools there are); one that fails to start is reported and left out; one that stops while you work makes its tools return an error that names it, and the others carry on. The stderr of a program goes to `~/.local/state/muagent/NAME.log` (or `$XDG_STATE_HOME/muagent/NAME.log`), where you will find why it failed.

A few things to know before you add a server:

- **A server is a program that runs as you, when `muagent` starts.** Only the user-level `mcp.json` is read, never one from the directory you are in, so a project you open cannot start programs.
- **Every MCP tool asks before it runs**, showing its arguments (all of them, with `?`), unless you set `read_only` for that server. A server's own claim that a tool is read-only is ignored: it is not something `muagent` can check.
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
- When the instructions, the tool definitions and the history together exceed `MAX_CONTEXT_TOKENS`, the oldest turns are dropped first. Tokens are estimated at about 4 bytes of UTF-8 per token, which works with any backend's tokenizer.

[TURN.md](TURN.md) follows one turn through the code: what is sent, what is kept, what you see, and every way a turn can end.
