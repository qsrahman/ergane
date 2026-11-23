# ergane

A minimal terminal chat agent built on the OpenAI [Responses API](https://platform.openai.com/docs/api-reference/responses). It works with OpenAI and with any compatible endpoint.

## Features

- Interactive REPL built on [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/): line editing, a history that is kept between sessions (with a grey suggestion from it as you type), Tab completion of the slash commands and of their arguments (`/effort`'s levels, the servers `/reconnect` can start), and a pasted block (a stack trace, a function) is one prompt
- Streaming output, which can be turned off
- Client-side conversation history that stays within a configurable context budget: when it gets long, the oldest turns are summarized by the model (the newest are kept as they are), and a long tool result is shortened to its start and end when it does not fit
- Works with any OpenAI-compatible base URL
- Tool calling: the model can call the tools of your [MCP servers](#mcp-servers), sees their errors, and can correct itself

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)

## Setup

ergane is not on PyPI, so it is installed from a checkout.

**As a command**, so that `ergane` works from any directory:

```bash
uv tool install ".[mcp]"
ergane --version
```

`[mcp]` is the extra that [MCP servers](#mcp-servers) need; leave it out (`uv tool install .`) if you will use none, and a server in `config.json` is then reported as needing the `mcp` package. If the shell cannot find `ergane`, run `uv tool update-shell` and open a new shell: it puts uv's tool directory on your `PATH`.

- After pulling changes: `uv tool install --reinstall ".[mcp]"`.
- To have edits to the checkout take effect at once, install with `--editable`: `uv tool install --editable ".[mcp]"`.
- To remove it: `uv tool uninstall ergane`.

**From the checkout**, for development: `uv sync --extra mcp`, then `uv run ergane`.

Put your settings in `~/.config/ergane/config.json` (or `$XDG_CONFIG_HOME/ergane/config.json`). A `config.json` in the directory you run `ergane` from is not read, so a project you open cannot change where your API key and prompts are sent.

```json
{
  "api_key": "${OPENAI_API_KEY}",
  "model": "gpt-5.4-mini",
  "base_url": "https://api.openai.com/v1",
  "streaming": true,
  "max_context_tokens": 65536,
  "reasoning_effort": "low",
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
| `builtin_tools`      | `--no-builtin` (offer none) | `true`             | Which of ergane's own [tools](#tools) are offered: `true` for all, `false` for none, or a list of names such as `["read_file"]`. |
| `mcpServers`         | `--no-mcp` (start none)  | none                  | The [MCP servers](#mcp-servers). |

Two more flags: `--config FILE` reads another file (there is no variable for it, so that a project's environment cannot point ergane at a file of its own), `--resume [ID]` continues a [saved session](#sessions), `--version` prints the version, and `--help` lists them all. A file you name that does not exist is an error; the default one may be missing, if the environment and flags give the key.

## Usage

```bash
ergane
```

From a checkout that is not installed as a command: `uv run ergane` (or `uv run python -m ergane`).

Type a prompt at `❯` and press Enter. Up and Down recall earlier prompts, which are kept in `~/.local/state/ergane/history` (or `$XDG_STATE_HOME/ergane/history`) in plain text; delete the file to forget them. The directory (which also holds the server logs) is made readable by you only (`0700`, the file `0600`), since a prompt can hold whatever you pasted. Paste several lines and they stay one prompt; press Enter to send it. To add a line by hand, press Alt-Enter (Esc, then Enter) or Ctrl-J (the new lines start with `· `); Shift-Enter does it too in a terminal that sends it as its own key (kitty's keyboard protocol, or xterm's modifyOtherKeys), but most send a plain Enter for it, and then it sends the prompt. Commands are completed with a line saying what they do. The line under the prompt shows the model, the effort and how full the context is, as a meter that turns yellow and then red (`~` while it is an estimate). Colours are your terminal's own, and `NO_COLOR` turns them off. Ctrl-R searches the earlier prompts, Ctrl-X Ctrl-E edits the prompt in `$EDITOR` and sends it when the editor closes, and Ctrl-Z suspends ergane. The question before a tool runs is a menu of its own (see below), which is not kept in the history. When input or output is redirected (`echo hi | ergane`), plain `input()` reads the prompts instead. To quit, use `/exit`, `/quit`, Ctrl-D, or Ctrl-C twice on an empty prompt. A stray Ctrl-C must not end a conversation, so on an empty line it takes two, and once there is a conversation Ctrl-D must be pressed twice too (it quits at once when there is none, or when input is redirected). The conversation is [saved](#sessions) after each prompt, so quitting loses nothing. A single Ctrl-C clears what you have typed, or, on an empty line, asks you to press it again. `/help` lists the keys as well as the commands.

| Command           | Effect                                                                           |
| ----------------- | -------------------------------------------------------------------------------- |
| `/help`           | List the commands                                                                |
| `/retry`          | Resend the last prompt; if its reply was saved, it is replaced                   |
| `/undo`           | Remove the last exchange from the history                                        |
| `/tools`          | List the tools offered to the model, by whether they ask first  |
| `/reconnect [NAME]` | Start the MCP servers that failed again, or the server NAME (also one that runs); the conversation stays |
| `/servers`        | List the MCP servers: which run (with how many tools, and the tokens their definitions take), which failed and why, and which are off |
| `/compact [FOCUS]` | Summarize the earlier turns now; `FOCUS` says what the summary must keep |
| `/summary`        | Show the summary that stands for the earliest turns |
| `/todo`           | Show the model's todo list (see [Tools](#tools)) |
| `/context`        | Show how much of the context budget the next request would use: the instructions, the tool definitions and the history (estimates), the total (the server's count of the last turn while the history is as it was), and the count the server gave for the last request |
| `/model [NAME]`   | Show the model, or switch to `NAME` for the rest of the session                  |
| `/effort [LEVEL]` | Show the reasoning effort, or set it (`none` … `max`, or `default` to leave it out) |
| `/reset`          | Clear the conversation history and start fresh (the old one stays saved)         |
| `/sessions [N or ID]` | Continue a [saved session](#sessions) of this directory: choose one from the list, or give its number or id |
| `/clear`          | Clear the terminal screen; the history is kept                                   |
| `/exit`, `/quit`  | Quit                                                                             |

Command names are case-insensitive. A mistyped command such as `/rest` is reported instead of being sent to the model. Pressing Ctrl-C while a response is generating cancels only that turn; if tools had already run, the turn is kept with a note saying you stopped it, so the model knows what was done.

## Sessions

The conversation is saved after each prompt, in a file of its own, for the directory you run `ergane` in. Nothing is asked and there is no setting: it is how a conversation survives quitting.

- `ergane --resume` continues the session saved last in this directory. `ergane --resume ID` continues the one with that id, or whose id starts with it. With no session saved it says so and starts a new one; an id that names none is an error.
- `/sessions` lists the sessions of this directory, newest first, with what the first prompt was, how many turns, when it was saved and its id. At a terminal it is a menu (arrows and Enter; `q` cancels) of the ten newest; choosing one puts it in place of the conversation you are in, which is saved already, so nothing is lost. Anywhere else it prints the list, and `/sessions 2` or `/sessions ID` continues one. `/reset` starts a new session and leaves the old one in the list.
- A resumed session is the conversation as it was: the history, the summary of the turns before it, and the pictures of tool results (`read_file`'s, or an MCP server's with `images` on). It goes on being saved as the same session. What is not saved is what you allowed for the session (ergane asks again), and the tools, which are the ones configured now.
- The files are in `~/.local/state/ergane/sessions/` (or `$XDG_STATE_HOME/ergane/sessions/`), a directory for each working directory, readable by you only. Only the newest 50 sessions of a directory are kept. **They hold everything you typed or pasted and every result of a tool.** To forget one, delete its file (`<id>.jsonl`); to forget them all, delete the `sessions` directory.
- Sessions are read only from there, never from the project you are in: a session is played back to the model as what it said and what its tools returned, so a file a project shipped could give orders in its voice.
- Two `ergane`s that resume the same session overwrite each other; the last to save wins.

## Tools

The model can call tools. A tool that only reads runs at once, and `ergane` prints the call as a line (its name in bold and its arguments dim, on a terminal, unless `NO_COLOR` is set), such as `▸ add  {"a": 17, "b": 25}`, never wider than the terminal: the middle of long values is left out. A tool that can change things asks first, in a menu under a line such as `Allow save {"path": "notes.txt"}?`. Deny is first and the default, so Enter denies; the other options are Allow once and Always allow that tool for the rest of the session (it then runs like a read-only one, with its call shown as a line, and `/tools` lists it as allowed; restart ergane to take it back). Choose with the arrows and Enter, the option's number, or its key: `n`, `y`, `a`. The menu goes away when you answer and leaves a line of what you decided, such as `▸ save  {"path": "notes.txt"}  allowed`. The question is never wider than the terminal: if the arguments do not fit, the middle of their long values is left out (the start and the end stay) and there is a fourth option, Show full arguments (`?`), which prints all of it and asks again. When input or output is redirected, the question is one line, `Allow save {"path": "notes.txt"}? [y/N/a]` (`[y/N/a/?]` when it is cut), and you type `y`, `a` or `?`; anything else is a no. The result goes back to the model, which says what it found, repeating until the model answers. A turn has no limit on its model calls, but a model that makes the same tool step (the same calls with the same results) again and again is told so in its instructions the third time among its last ten steps (you see a notice, and it can change course), and the turn is stopped, and kept with a closing note, when one step fills more than five of them; Ctrl-C ends a turn at any time. When a turn grows past `max_context_tokens` while it runs, the oldest tool results are replaced by a short placeholder (the calls stay) so that it can go on; only if the latest results alone are over, it is stopped. If a call fails, the error text goes back to the model so it can correct the call. The Responses API must support function calling.

`ergane` has tools of its own, in `src/ergane/builtin/`, and more come from the MCP servers you list in `config.json` (see below). Its own so far are `read_file` (a text file, or a png, jpeg, webp or gif picture for the model to look at, up to about 3.7 MB; a model without vision is refused once, and ergane sends it no more pictures for the rest of the session), `list_directory` (a directory's entries) and `search_files` (files by name or content: a pattern that takes too long on one line is refused, and a search gives up after 30 seconds); the last two leave out what the repository's `.gitignore` files exclude (a file you name is still read); these only read, so they run without asking. `write_file` (create or overwrite a file) and `edit_file` (replace text in a file: the text must match once unless `replace_all` is set, and trailing whitespace and typographic quotes and dashes are forgiven) ask before each call, and the question has a "Show the change" option that prints the diff; they never write through a symbolic link. `run_command` runs a shell command (stdin closed, stopped with its child processes after `timeout_s`, 120 seconds by default; output past about 16,000 characters keeps its start and end, and the whole of it is saved to a file in the system temp directory that the result names, for `read_file`; those files are not deleted) and asks before each call; the command itself is not confined, only its `cwd` is, and it does not get `OPENAI_API_KEY` in its environment. The file tools work only inside the working directory and the system temp directory. `web_fetch` reads a web page as markdown (or raw HTML with `raw`), in parts when it is long (`max_length`, `start_index`); it asks before each call, because a page can tell the model to fetch a URL that carries your data. It refuses private, loopback and link-local addresses (checked on the connection itself and on every redirect, so DNS rebinding gets no chance), refuses binary content, reads at most 5 MB, and does not read `robots.txt`; what it returns is untrusted web text. Its code is ported from the `fetch` MCP server, whose licence is in `THIRD_PARTY_NOTICES.md`. `web_search` searches the web through DuckDuckGo's HTML page, with no key: it returns the top results (title, URL, snippet) and asks before each call, since the query leaves your machine. It is not an API, so DuckDuckGo may turn automated searches away or change the page; a refusal (HTTP 202, 403 or 429, or an empty answer) is reported as one, never as "no results". For a search service you can rely on, list a search MCP server instead. `builtin_tools` or `--no-builtin` turns them off. With none of either kind, it is a plain chat, and `/tools` says so. With any tool, the model also gets a `todo` tool for a task of three or more steps: it sends the whole checklist on each call (`pending`, `in_progress`, `completed`, `cancelled`), and a list with two `in_progress` steps is refused. You see one line when the list changes, `/todo` shows it, and while steps are open it is put in the instructions of every request, so it survives the old tool results being dropped. It lives in memory: `/reset` and resuming a session clear it, and it is not saved with a session.

To give it a tool written in Python, put a function with a docstring (its description, written for the model), type hints (its parameters) and `@tool` in a new file in `src/ergane/builtin/`. That is all: every tool in that package is offered, and nothing lists it.

```python
# src/ergane/builtin/arithmetic.py
from ergane.tools import tool


@tool
def add(a: int, b: int) -> str:
    """Add two integers and return the sum."""
    return str(a + b)
```

A tool runs without asking unless it says `@tool(read_only=False)`, which it must if it changes anything outside the program. A file whose name starts with `_` is left alone, for helpers; import a heavy dependency inside the function, so that starting `ergane` does not pay for it. [TOOLS.md](TOOLS.md) has the rest: parameters and their descriptions, what to return, errors, testing.

## MCP servers

`ergane` can use the tools of [MCP](https://modelcontextprotocol.io) servers, either a local program it runs (stdio) or a server it connects to over HTTP (streamable HTTP). List them under `mcpServers` in `config.json`, in the usual format of other tools' MCP files:

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
      "enabled": false
    }
  }
}
```

For files and a shell, [workshop-mcp](https://github.com/qsrahman/workshop-mcp) is a server made to go with `ergane`. It is a Node program that `npx` fetches on first use:

```json
{
  "mcpServers": {
    "workshop": {
      "command": "npx",
      "args": ["-y", "@qsrahman/workshop-mcp", "--no-fetch"]
    }
  }
}
```

`--no-fetch` leaves out its `fetch_url`; drop the flag if you want the model to read web pages, since `ergane` has no tool for that of its own. Without a directory in `args` it can reach any file you can, and its shell is on, so every call asks first (leave `read_only` unset); `--read-only`, `--no-shell` and directories narrow it, as its README says. It starts in the directory where you run `ergane`.

| Key         | Description                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------ |
| `type`      | `"local"` for a `command`, `"remote"` for a `url`. Optional; if given it must agree with them.               |
| `enabled`    | `false` keeps the entry but does not start the server. Default `true`.                                       |
| `command`   | The program that runs the server, in the directory where you run `ergane`. Give this or `url`, not both.    |
| `args`      | Its arguments                                                                                                |
| `env`       | Environment variables for it. The server gets only a short list (`PATH`, `HOME`, ...) and these, so it does not see your `OPENAI_API_KEY`. |
| `url`       | The address of a server to connect to, instead of a `command`. `https://`, or `http://` for `localhost`, `127.0.0.1` and `[::1]` only. |
| `headers`   | Sent with every request to the `url`, for an `Authorization` or an API key. Not in the URL itself: a user name or password there is refused. |
| `read_only` | `true` lets the server's tools run without asking. Default `false`.                                          |
| `images` | `true` gives the model the pictures a tool returns (png, jpeg, webp, gif; each up to about 3.7 MB), not just a note that there was one. Default `false`: a picture costs tokens in every request until it is dropped to make room. A model without vision rejects the request (Ollama: `400 … does not support multimodal requests`); ergane then takes the pictures out of the request, says so, and sends it again, so the turn goes on, and sends that model no more pictures this session (set `images` to `false` to stop the wasted request for a server's pictures). gemma4 reads them through Ollama. |
| `instructions` | `true` gives the model what the server says about using its tools, in its instructions. Default `false`: the text is the server's, and some models follow an order hidden in it. |
| `tools`     | Offer only these tools of the server, by the server's own names (see `/tools` for what a server has). The definition of every tool is sent with every request, so a server with many tools you do not use costs context and money. A name the server does not have fails the server, with a notice. Default: all. |
| `startup_timeout` | Seconds to wait for the server to start and list its tools. Default `30`.                              |
| `call_timeout` | Seconds a call to one of its tools may take before it is given up on. Default `300`.                      |

Write `${NAME}` in `args`, `env` or `headers` to use the environment variable `NAME`, so that keys stay out of the file: `"env": { "API_KEY": "${MY_API_KEY}" }`. A server whose variable is not set is skipped, with a notice that names the variable. The server gets only the variables you list, not the rest of your environment.

The names are letters, digits, `_` and `-`. A server's tools are offered to the model as `name__tool`; `/tools` lists them. The servers start at the same time, so a few slow ones do not add up. At a terminal the prompt comes up at once and the servers connect while you type (the line under the prompt says `starting MCP servers`, then how many tools there are); the first prompt you send waits for the ones that are not up yet. A server that starts is not announced; one that fails to start is reported before your first reply, shown in the line under the prompt, and left out (`/servers` tells it again, with the log to read, and `/reconnect` starts it again once the cause is fixed, without leaving ergane); one that stops while you work (it exits, or a remote one goes away) makes the call that finds out return an error that names it, and is started again for the next call, up to three times in a session (the call that failed is not repeated, since it may have done part of its work); the others carry on. A server that never answers delays the start by its `startup_timeout`, not by the sum of them. The stderr of a program goes to `~/.local/state/ergane/NAME.log` (or `$XDG_STATE_HOME/ergane/NAME.log`), where you will find why it failed.

A few things to know before you add a server:

- **A server is a program that runs as you, when `ergane` starts.** Only the user-level `config.json` is read, never one from the directory you are in, so a project you open cannot start programs.
- **Every MCP tool asks before it runs**, showing its arguments (all of them, with `?`), unless you set `read_only` for that server. A server's own claim that a tool is read-only is ignored: it is not something `ergane` can check.
- **A server at a URL receives what the model sends it**: the arguments of every call, which can hold file contents or other things from your conversation. Connect only to servers you trust with that. `config.json` holds the headers (and perhaps the key) in plain text, so keep it to yourself (`chmod 600`). Redirects to another host are not followed, so the headers go only where you pointed them.
- **Tool descriptions come from the server** and are read by the model. Add servers you trust.
- **What a server says about using its tools** (its `instructions`) is kept from the model unless you set `instructions` to `true` for that server. It is then given, framed as notes from the server and not as orders, up to 4,000 characters. The framing is not a guarantee: a model can follow an order hidden in it (a call that is not `read_only` still asks first). Turn it on for servers you trust. The tools are listed once, at start (`/reconnect` lists a server's again).

## Long conversations

Every request carries the whole history, so a long session would outgrow `max_context_tokens`. ergane deals with it in three steps, from the cheapest:

1. **A long tool result is shortened** when the turn does not fit: the first 4,000 and last 1,000 characters stay, with a note of how much is hidden, and the model can call the tool again. Only if that is not enough is a result replaced by a placeholder.
2. **Old turns are summarized.** When the next request would pass `compact_at` (75%) of the budget, the model is asked for a summary of the oldest turns, with headings: goal, constraints, progress, decisions, every request in order, exact references (paths, ids, errors), next steps. The newest turns (about `keep_recent_tokens`) stay as they are. The summary goes first in every later request, as notes and not instructions, and the next summary updates it. It uses the model you chat with, one extra request, and says so: `[Summarizing earlier messages…]`.
3. **What still does not fit is dropped,** oldest turns first, as before. This is also what happens if a summary fails: the failure is shown, the conversation goes on, and after three failures in a row summaries pause until `/compact` works.

`/compact [focus]` summarizes now, `/summary` shows it, `/context` shows where summaries start, and `--no-auto-compact` turns them off. If the server refuses a request as too long, ergane summarizes (or shortens the results) and sends it again, once.

A summary is written from everything in the conversation, web pages and tool output included, so it is sent as notes, not instructions, and nothing in it is trusted more than the text it came from. Summaries lose detail: anything that must survive exactly belongs in a file, not the chat.

## Tests

```bash
uv run pytest
```

The tests run against a local fake of the Responses API. They ignore your `config.json` and environment, and make no external network calls.

## How it works

- Requests are sent with `store=False`, so the server keeps no conversation state. The full history is resent on each turn.
- A turn, tool calls and their results included, is added to the history once the final response completes with text. An incomplete response keeps the partial reply you already saw. A failed, refused or empty response in the first round leaves the history unchanged; after a round of tool calls has run (they may have had effects), the turn is kept with a closing note such as `[Stopped after 2 model calls: Response failed: …]`, so the model still knows what was done: say "continue" to go on, or use `/retry` to start the prompt over. An empty response (no text, no call) is asked for again once before it is reported.
- When the next request would pass `compact_at` of `max_context_tokens` (instructions, tool definitions and history together), the oldest turns are summarized and the newest kept; see [Long conversations](#long-conversations). Turns that still do not fit are dropped, oldest first. Tokens are estimated at about 4 bytes of UTF-8 per token, which works with any backend's tokenizer.

[TURN.md](TURN.md) follows one turn through the code: what is sent, what is kept, what you see, and every way a turn can end.

## License

[MIT](LICENSE)
