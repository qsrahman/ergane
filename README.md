# muagent

A minimal terminal chat agent built on the OpenAI [Responses API](https://platform.openai.com/docs/api-reference/responses). It works with OpenAI and with any compatible endpoint.

## Features

- Interactive REPL built on [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/): line editing, a history that is kept between sessions (with a grey suggestion from it as you type), Tab completion of the slash commands and of `/effort`'s levels, and a pasted block (a stack trace, a function) is one prompt
- Streaming output, which can be turned off
- Client-side conversation history that stays within a configurable context budget: when it gets long, the oldest turns are summarized by the model (the newest are kept as they are), and a long tool result is shortened to its start and end when it does not fit
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

Put your settings in `~/.config/muagent/config.json` (or `$XDG_CONFIG_HOME/muagent/config.json`). A `config.json` in the directory you run `muagent` from is not read, so a project you open cannot change where your API key and prompts are sent.

```json
{
  "api_key": "${OPENAI_API_KEY}",
  "model": "gpt-5.4-mini",
  "base_url": "https://api.openai.com/v1",
  "streaming": true,
  "max_context_tokens": 65536,
  "reasoning_effort": "low",
  "max_turns": 25,
  "mcpServers": {}
}
```

Only the key is required, and it may be written out, or as `${NAME}` to take it from the environment variable `NAME` (a variable that is not set is an error that names it). The servers under `mcpServers` are explained [below](#mcp-servers). A key that is not a setting is an error that names it. There is no `.env` file: put the key in the environment or in this file.

Each setting comes from the first of these that sets it: **its command-line flag, `config.json`, its default.** The key is the one setting the environment can also give: the variable `OPENAI_API_KEY`, which the OpenAI SDK reads too, beats `api_key` in the file. There are no variables for the rest (not for the base URL, which would let a project's environment send your key and prompts elsewhere).

| File key             | Flag                     | Default               | Description |
| -------------------- | ------------------------ | --------------------- | ----------- |
| `api_key`            | none (it would show in the process list) | —       | API key (required); `OPENAI_API_KEY` in the environment wins over it |
| `base_url`           | `--base-url URL`         | OpenAI default        | Base URL for an OpenAI-compatible API |
| `model`              | `--model NAME`           | `gpt-5.4-mini`        | Model name |
| `streaming`          | `--streaming`, `--no-streaming` | `true`         | Stream tokens as they arrive |
| `max_context_tokens` | `--max-context-tokens N` | `65536`               | Token budget for one request: the system instructions, the definitions of the tools and the conversation history together (not the reply). Old turns are dropped to fit it, so many or long tool definitions leave less room for the history. Keep it well below the model's context window. |
| `reasoning_effort`   | `--reasoning-effort LEVEL` | unset (model default) | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`; the flag also takes `default`, to send none over a value in the file. Leave unset for non-reasoning models, which reject it. |
| `auto_compact`       | `--auto-compact`, `--no-auto-compact` | `true`   | Summarize the oldest turns when the history gets long (see [Long conversations](#long-conversations)). |
| `compact_at`         | `--compact-at SHARE`     | `0.75`                | Share of `max_context_tokens` at which to summarize. |
| `keep_recent_tokens` | `--keep-recent-tokens N` | `15000`               | About how many tokens of the newest turns stay as they are (at most half of where summarizing starts). |
| `max_turns`          | `--max-turns N`          | `25`                  | Most model calls for one prompt: the cap on the tool loop. |
| `mcpServers`         | `--no-mcp` (start none)  | none                  | The [MCP servers](#mcp-servers). |

Two more flags: `--config FILE` reads another file (there is no variable for it, so that a project's environment cannot point muagent at a file of its own), `--version` prints the version, and `--help` lists them all. A file you name that does not exist is an error; the default one may be missing, if the environment and flags give the key.

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
| `/compact [FOCUS]` | Summarize the earlier turns now; `FOCUS` says what the summary must keep |
| `/summary`        | Show the summary that stands for the earliest turns |
| `/context`        | Show how much of the context budget the next request would use: the instructions, the tool definitions and the history (estimates), and the count the server gave for the last request |
| `/model [NAME]`   | Show the model, or switch to `NAME` for the rest of the session                  |
| `/effort [LEVEL]` | Show the reasoning effort, or set it (`none` … `max`, or `default` to leave it out) |
| `/reset`          | Clear the conversation history and start fresh                                   |
| `/clear`          | Clear the terminal screen; the history is kept                                   |
| `/exit`, `/quit`  | Quit                                                                             |

Command names are case-insensitive. A mistyped command such as `/rest` is reported instead of being sent to the model. Pressing Ctrl-C while a response is generating cancels only that turn; if tools had already run, the turn is kept with a note saying you stopped it, so the model knows what was done.

## Tools

The model can call tools. A tool that only reads runs at once, and `muagent` prints the call as a dimmed line (on a terminal, unless `NO_COLOR` is set), such as `[add({"a": 17, "b": 25})]`, never wider than the terminal: the middle of long values is left out. A tool that can change things asks first, in one line, such as `Allow save {"path": "notes.txt"}? [y/N/a]`; type `y` to allow this call, or `a` to allow that tool for the rest of the session (it then runs like a read-only one, with its call shown as a line, and `/tools` lists it as allowed; restart muagent to take it back). Anything else is a no. The line is never wider than the terminal: if the arguments do not fit, the middle of their long values is left out (the start and the end stay) and the prompt ends `[y/N/a/?]`; type `?` to see all of it, and it asks again. The result goes back to the model, which says what it found, repeating until the model answers. When a turn grows past `max_context_tokens` while it runs, the oldest tool results are replaced by a short placeholder (the calls stay) so that it can go on; only if the latest results alone are over, it is stopped. If a call fails, the error text goes back to the model so it can correct the call. The Responses API must support function calling.

`muagent` has no tools of its own: they come from the MCP servers you list in `config.json` (see below). With none, it is a plain chat, and `/tools` says so.

To give it a tool written in Python, make a function with a docstring (its description, written for the model), type hints (its parameters) and `@tool`, and list it in `main()` in `cli.py`, next to the tools of the MCP servers (`Agent(config, tools=[add, *mcp_tools])`):

```python
@tool
def add(a: int, b: int) -> str:
    """Add two integers and return the sum."""
    return str(a + b)
```

## MCP servers

`muagent` can use the tools of [MCP](https://modelcontextprotocol.io) servers, either a local program it runs (stdio) or a server it connects to over HTTP (streamable HTTP). List them under `mcpServers` in `config.json`, in the usual format of other tools' MCP files:

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

The names are letters, digits, `_` and `-`. A server's tools are offered to the model as `name__tool`; `/tools` lists them. The servers start at the same time, so a few slow ones do not add up. A server that starts is not announced (the first line shows how many tools there are); one that fails to start is reported and left out; one that stops while you work (it exits, or a remote one goes away) makes the call that finds out return an error that names it, and is started again for the next call, up to three times in a session (the call that failed is not repeated, since it may have done part of its work); the others carry on. A server that never answers delays the start by its `startup_timeout`, not by the sum of them. The stderr of a program goes to `~/.local/state/muagent/NAME.log` (or `$XDG_STATE_HOME/muagent/NAME.log`), where you will find why it failed.

A few things to know before you add a server:

- **A server is a program that runs as you, when `muagent` starts.** Only the user-level `config.json` is read, never one from the directory you are in, so a project you open cannot start programs.
- **Every MCP tool asks before it runs**, showing its arguments (all of them, with `?`), unless you set `read_only` for that server. A server's own claim that a tool is read-only is ignored: it is not something `muagent` can check.
- **A server at a URL receives what the model sends it**: the arguments of every call, which can hold file contents or other things from your conversation. Connect only to servers you trust with that. `config.json` holds the headers (and perhaps the key) in plain text, so keep it to yourself (`chmod 600`). Redirects to another host are not followed, so the headers go only where you pointed them.
- **Tool descriptions come from the server** and are read by the model. Add servers you trust.
- The list of tools is read once, at start.

## Long conversations

Every request carries the whole history, so a long session would outgrow `max_context_tokens`. muagent deals with it in three steps, from the cheapest:

1. **A long tool result is shortened** when the turn does not fit: the first 4,000 and last 1,000 characters stay, with a note of how much is hidden, and the model can call the tool again. Only if that is not enough is a result replaced by a placeholder.
2. **Old turns are summarized.** When the next request would pass `compact_at` (75%) of the budget, the model is asked for a summary of the oldest turns, with headings: goal, constraints, progress, decisions, every request in order, exact references (paths, ids, errors), next steps. The newest turns (about `keep_recent_tokens`) stay as they are. The summary goes first in every later request, as notes and not instructions, and the next summary updates it. It uses the model you chat with, one extra request, and says so: `[Summarizing earlier messages…]`.
3. **What still does not fit is dropped,** oldest turns first, as before. This is also what happens if a summary fails: the failure is shown, the conversation goes on, and after three failures in a row summaries pause until `/compact` works.

`/compact [focus]` summarizes now, `/summary` shows it, `/context` shows where summaries start, and `--no-auto-compact` turns them off. If the server refuses a request as too long, muagent summarizes (or shortens the results) and sends it again, once.

A summary is written from everything in the conversation, web pages and tool output included, so it is sent as notes, not instructions, and nothing in it is trusted more than the text it came from. Summaries lose detail: anything that must survive exactly belongs in a file, not the chat.

## Tests

```bash
uv run pytest
```

The tests run against a local fake of the Responses API. They ignore your `config.json` and environment, and make no external network calls.

## How it works

- Requests are sent with `store=False`, so the server keeps no conversation state. The full history is resent on each turn.
- A turn, tool calls and their results included, is added to the history once the final response completes with text. An incomplete response keeps the partial reply you already saw; failed or empty responses leave the history unchanged.
- When the next request would pass `compact_at` of `max_context_tokens` (instructions, tool definitions and history together), the oldest turns are summarized and the newest kept; see [Long conversations](#long-conversations). Turns that still do not fit are dropped, oldest first. Tokens are estimated at about 4 bytes of UTF-8 per token, which works with any backend's tokenizer.

[TURN.md](TURN.md) follows one turn through the code: what is sent, what is kept, what you see, and every way a turn can end.

## License

[MIT](LICENSE)
