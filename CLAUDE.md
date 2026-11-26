# CLAUDE.md

Guidance for working in this repository. The mechanics are in the modules' docstrings
(start with the one at the top of a file) and in [TURN.md](TURN.md), which walks one
turn through the code; this file keeps what a change must not break.

## Overview

`ergane` is a minimal interactive terminal chat agent on the OpenAI **Responses API**
(`src/ergane/`, `src` layout). Its tools come from MCP servers listed in
`config.json` and from the Python functions in `src/ergane/builtin/` (files, shell, web
fetch and web search); `--no-builtin` and no server is the plain chat, which has no tool
at all.

## Commands

Uses `uv` with Python 3.14 (`.python-version`, `requires-python >= 3.14`).

```bash
uv sync              # install dependencies (including pytest and mcp) into .venv
uv sync --extra mcp  # what a user needs for MCP servers; `mcp` is an optional extra
uv run ergane       # start the REPL; same as `uv run python -m ergane`
uv run pytest        # run the tests
uv run pytest tests/test_repl.py::TestRepl::test_strips_input_and_skips_blank_lines  # one test
uv run ruff check . && uv run ruff format --check .   # lint and format check
uv run pyright src tests   # type check
```

The package is built with `uv_build` (needs `src/ergane/__init__.py`); `uv sync` installs
it editable; the script is `ergane.cli:main`. Tests import modules directly
(`from ergane.agent import Agent`).

## Versions

`version` in `pyproject.toml` is the only place the version is written (`ergane --version`
reads it from the installed package). Every change that a user could notice (a flag, a
`config.json` key, what a tool or command does, a requirement) is a line under
Unreleased in [CHANGELOG.md](CHANGELOG.md), in the same commit. To release, follow the
steps at the top of that file.

## Tests

- `tests/fake_api.py` is a fake Responses API run on a local port by the `fake_server`
  fixture, through the real SDK client and streaming parser. The most recent user
  message picks the outcome; the scenarios are listed in `FakeResponsesHandler`'s
  docstring (failures, broken streams, tool calls, `status_N`, `overflow`, `usage`, and
  a summary that goes wrong in each way). Request bodies are in `fake_server.requests`.
- `tests/fake_mcp_server.py` is a stdio (or, with a port, HTTP) MCP server on the SDK's
  own `MCPServer`, with a tool for each case a client must handle; `test_mcp_client.py` and
  `test_mcp_hub.py` start it as a subprocess (`FAKE`, `serve_http` in `helpers.py`; the
  `make_server`, `http_url` and `tools` fixtures in `conftest.py`).
- `test_properties.py` holds Hypothesis tests of the functions that take outside text:
  what must hold for any input. A run tries 100 inputs each;
  `--hypothesis-profile=thorough --timeout=0` tries 3,000. An input that once broke one
  stays on it as an `@example`.
- The autouse `isolate_config` fixture runs each test in an empty `tmp_path` that is
  also `XDG_CONFIG_HOME`, and clears `OPENAI_API_KEY`, the one variable ergane reads.
- Test files mirror the modules (`test_mcp_config.py` tests the server models of `config.py`), plus `test_compaction.py` (summaries, shortened
  results, the too-long retry) and `test_fake_api.py`. Helpers (`make_config`,
  `make_context`, `run_repl`, `tool_with`) are in `tests/helpers.py`; the `client` and
  `make_agent` fixtures are in `conftest.py`.
- Build objects directly instead of monkeypatching: `make_agent(tools=[...],
  STREAMING=False)` wires an `Agent` to the fake server; tools a test defines come from
  `Tool.from_function`. `Agent` tests check return values and `TurnError`, not printed
  output. `TestRepl` replaces `agent.run_turn` on one instance; `TestReplRendering`
  drives a real `Agent` through `run_repl()` and pins the layout of an exchange.
- `test_console.py` gives `Console` an `output=Vt100_Output(io.StringIO(), …)` to see the
  codes of styled lines, and a `choose` function in place of the menu; the real menu is
  driven with a pipe input in `TestMenu` (`test_terminal.py`), like the prompt.
- A behaviour change gets a mutation check: break the code, see the test fail. Run it
  with a script that always restores the file (a `| head` that closes the pipe early
  once left a mutated file behind).
- The built-ins have a `test_builtin_*.py` each, on `tmp_path` (the autouse fixture makes
  it the working directory). The fetch and search tests serve from a local HTTP server
  and never touch the network; `test_builtin_shell.py` points `tempfile.tempdir` at
  `tmp_path` so that saved output does not pile up in the real temp directory.

## Configuration

Highest first: the command-line flags, the user's
`$XDG_CONFIG_HOME/ergane/config.json` (default `~/.config/ergane/`), the defaults.
ergane itself reads one variable for a setting, `OPENAI_API_KEY`, which beats `api_key`
in the file; there is none of its own for any other setting (a variable for `base_url`
would let a project's environment send the key and every prompt elsewhere, one for the
servers would start a program of its own, and `MODEL`, `BASE_URL` are exported by other
programs). The OpenAI SDK is not held to this: with `base_url` unset it uses
`OPENAI_BASE_URL`, and it always reads `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID` and
`OPENAI_CUSTOM_HEADERS`. This is a known gap, documented in the README, not fixed. Only
`--config` names another file. `load_config(path=None, **flags)` reads the file, puts
the key and the flags over it and validates the result as one `Config` (a plain pydantic
model; there is no pydantic-settings); `main()` prints the per-key errors and exits with
status 1. Read the docstring of `config.py` for how, and why.

The keys, their flags and their defaults are the table in
[README.md](README.md#setup); `Config` in `config.py` is where they are defined.

MCP servers are `mcpServers` in the same file (`Config.mcp_servers`), and only there: a
server is a program, so neither the environment nor a project's files may define one.
Each server's stderr goes to `$XDG_STATE_HOME/ergane/NAME.log`; the state directory holds
the prompt history and the saved sessions too and is made `0700`.

## Architecture

The modules, in dependency order (each imports only from those above it):

- `tools.py`: the `@tool` decorator and `Tool` (`name`, `description`, `parameters`,
  `check`, `runner`, `read_only`, `preview`: what the user may ask to see before
  allowing a call, a diff for a write). `Tool.call` never raises (except Ctrl-C): a bad call
  or an exception comes back as text for the model, and output is cut at
  `MAX_OUTPUT_CHARS`. `tools_in(package)` finds the tools of a package: a `@tool`
  function in a module of `builtin/` is offered with nothing else to edit; [TOOLS.md](TOOLS.md)
  is the guide to writing one.
- `config.py`: `Config`, `load_config()`, `EFFORTS`, the XDG directories, the model of a
  server in `config.json`: `LocalMcpServer` (a `command`) or `RemoteMcpServer` (a
  `url`), told apart by `type` or by having a `url`, as the union `McpServerConfig` (a
  field of the other kind is an error of its own, so there is no check of
  combinations), and what the MCP code shares with it: the rules for a server's name
  and the `${NAME}` placeholders (`fill_placeholders`).
- `todos.py`: `TodoList` and its `todo` tool (whole-list replace, one `in_progress` at a
  time, held in memory, not saved with a session). The agent offers it only when it has
  another tool, shows a change as a checklist (`on_todos`, not a notice), and puts a list with open steps into the
  instructions of each request, since old tool results are dropped from the history.
- `builtin/`: ergane's own tools, each module only `@tool` functions (and the helpers
  they share, whose names start with `_`, which `tools_in` skips). In dependency order:
  `_paths` (`resolve`: the one place a model's path becomes a real one), `_text`
  (the line, character and head-and-tail cuts, and reading a file by lines), `_images`
  (which files are pictures), `_http` (what the web tools call themselves), `_edit` (the
  text of an edit, pure), then `files` (`read_file`), `writes` (`write_file`, `edit_file`), `shell`
  (`run_command`), `fetch` (`web_fetch`) and `search` (`web_search`). A tool that needs
  a setting a test changes is made by a `make_*_tool` factory (the fetch and search
  ones), since tools cannot see `Config`. [TOOLS.md](TOOLS.md) is the guide; `PLAN.md`
  records why each tool is as it is.
- `mcp_tools.py`: what an MCP server's tools and results become for the model: names,
  the text and pictures of a result, the check of a call's arguments. Plain functions.
- `mcp_client.py`: `McpClient`, one connection used from synchronous code (on anyio's
  blocking portal, a thread with an event loop); its tools become `Tool`s. `start()`
  blocks until the server is up, so servers are started at once on threads of their
  own. Imports `mcp` only when a server starts. The design notes are in its docstring.
- `mcp_hub.py`: the servers together: `close_servers`, `McpHub` (starts
  them, at a terminal in the background, and starts one again for `/reconnect`), and a
  `ServerStatus` per server (`McpHub.statuses`, by name), which records what became of it
  for `/servers`; `reconnect` starts a server again through the same `_Startup`.
- `surrogates.py`: `without_surrogates`, for text that is about to be sent or saved.
  Imports nothing of ergane.
- `loops.py`: the loop guard's pure functions (`step_signature`, `is_looping`, `last_repeats`,
  `loop_note`). Imports nothing of ergane.
- `errors.py`, `prompts.py`: `TurnError`, `StreamCut`, `describe_error`,
  `is_context_overflow`, `is_images_rejected`; the three prompts as text. They import nothing of ergane.
- `context.py`: `Context` (the history and the summary) and `estimate_tokens()`.
- `sessions.py`: `SessionStore`: conversations saved to disk, one file each, for a working
  directory, in the user's state directory (`0700`, files `0600`); the newest 50 are kept.
  A file is two lines of JSON, the first of which lists it (`SessionInfo`). The format and
  the reasons are in its docstring.
- `outputs.py`: the results of tools in the history: their form (`function_output`,
  pictures) and how they are made to fit (`drop_images`, `shorten`, `drop_outputs`,
  `answer_open_calls`).
- `compaction.py`: `Compactor`: when to summarize the oldest turns, what to keep, and the
  summary itself, through a function that asks the model; `transcript` (the text it is
  given) and `extract_summary` (the check of the answer). It needs nothing of the agent.
- `responses.py`: `Responder` (one request to the API, streamed, sent again when the stream
  breaks before any text) and `response_problem`. It needs nothing of the agent.
- `agent.py`: `Agent`: `run_turn` (a loop of `_ask` and `_run_round` around a `_Turn`),
  `compact` (which is `Compactor`'s), and the tools (`add_tools`: the first of a name
  wins, here and wherever tools are added). It reads its settings from `Config`, each
  where it is used (as do `Responder` and `Compactor`), and imports nothing of MCP (a
  test pins this).
- `markdown.py`: `Markdown`, the light styling of a streamed reply (`**bold**`, `` `code` ``,
  `#` headings, dim fences) a piece at a time; the marks stay in the text. Imports nothing of
  ergane.
- `console.py`: all terminal output, `sanitize()`, and the permission question. A status
  line is styled pieces (prompt_toolkit's), printed by prompt_toolkit where there is
  colour; the question is a typed line, or a menu when `Repl` gives the console `choose`.
- `commands.py`: the `@command` registry and the slash commands (imports `Repl` for type
  checking only).
- `terminal.py`: the prompt_toolkit side: `terminal_session` (the prompt, history,
  completion and key bindings, and their styles), `toolbar` (the line under the prompt:
  model, effort, git branch, tools allowed for the session, servers and the context
  meter, the least important left out when the terminal is narrow, never the meter)
  and `choose` (the menu of the permission question). Imports `Repl` for type checking
  only.
- `repl.py`: `Repl`, the input loop: it saves the conversation after each prompt
  (`save_session`), and continues a saved one (`restore`, for `--resume` and
  `/sessions`).
- `cli.py`: `main()`.

Dependencies are passed in through constructors (`Agent(config, client=None, tools=(),
server_notes=None)`, `Repl(agent, commands, session=None, console=None, hub=None,
store=None)`); there is no global config lookup.
Every setting is read where it is used, so `/model` and `/effort` take effect at once.

What must stay true:

- **The history changes in five places only:** `Context.save_turn`, `undo` (which
  `rewind` uses), `reset`, `compact` and `restore` (a saved session replacing it). Each
  adds one to `Context.revision`, which `Repl.save_session` compares to know that the
  conversation changed. A turn is saved with its reply or not at all,
  apart from the exceptions in `agent.py`'s docstring. User and assistant messages
  alternate; the summary is kept apart (`Context.summary`), put first in each request as
  a `user` message framed as notes, never as instructions, since it is written from text
  that may hold a web page's orders.
- **No lone surrogate leaves the program:** `without_surrogates` is applied to
  every request (`Responder`) and every saved session, and what measures or hashes the
  history before then must take one. One can come from a reply, a web page or the input.
- **Every function call has an output** before anything is sent or saved; calls and
  outputs are dropped together; only the newest turn keeps its tool work.
- **A tool that is not `read_only` runs only if `confirm(tool, arguments)` returns true or
  `"always"`** (which allows that tool for the session). The default denies. A call whose
  arguments `check` rejects is answered without asking.
- **Only read-only calls run together** (on threads, `Agent._run_tools`); any other call
  runs alone, after the calls before it. The outputs are added in the order of the calls.
  `Console` is used from the main thread only, so a call that asks never asks at the same
  time as another.
- **A turn has no cap on its model calls; the loop guard ends a stuck one.** The same tool
  step (calls, arguments and outputs) more than twice in the last 10 puts a notice in the
  instructions of the next requests, which goes when the model changes course and never
  outlives the turn; more than 5 stops the turn, kept with a closing note. The notice is in
  the instructions, not in a tool result, which the prompt calls data. The budget does not
  bound a turn (it drops old results so the turn can go on), so a test that loops without
  the guard would never end: pytest's `timeout` (60 s, `pytest-timeout`) fails it instead.
  Use a scenario the guard stops, or `tool_loop_N`.
- **Trimming never drops the summary or the newest user message.** The budget is for the
  whole request; the in-turn check uses the server's `usage.input_tokens` plus an
  estimate of what was added, shortens old tool results before dropping them, then the
  latest round's, and fails only when even that does not fit; the turn is then kept with
  a note and its results dropped, like any turn after a completed round.
- **The agent prints nothing.** What the user should see goes through `on_text`,
  `on_notice`, `on_call` or `confirm`, and the console prints it through `_emit`.
- **Credentials, programs and sessions come only from the user's own files:** no config,
  `.env`, server list or saved session is read from the working directory. A session is
  played back as the model's own words, so one that a project shipped could give orders
  in its voice; `sessions.py` reads only the state directory.
- **A model's path becomes a real one in `_paths.resolve` only:** `..` is collapsed
  first, symlinks are followed, and the result must be inside the working directory or
  the system temp directory (a path that does not exist yet is judged by its nearest
  parent that does). Nothing else in `builtin/` opens a path the model wrote. ergane's own
  config and state directories are not singled out: the roots are the protection.
- **A file is never written through a symbolic link,** and only a regular file is
  overwritten (`writes.py`): the path the model wrote must not be a link, a new file is
  made with `O_EXCL`, and an existing one is opened with `O_NOFOLLOW` after an `lstat`
  (opening a named pipe would block). `edit_file` and its preview share one plan
  (`_plan_edit`), so the diff the user approves is the change that is made.
- **A tool that is not `read_only` asks, and these are:** `write_file`, `edit_file`,
  `run_command`, `web_fetch` and `web_search` (a page the model read can tell it what to
  fetch or search). `read_file`, which only reads, runs on a thread and cannot be
  stopped by Ctrl-C, so it is bounded without help: output caps, reads in chunks (a file
  of one huge line must not be read whole) that stop at what can be shown, and regular
  files only (`_paths.regular_size`: opening a named pipe would block for ever; an edit
  refuses one too).
- **`run_command` leaves nothing running and tells nothing secret:** each call is a
  fresh `sh -c` in its own process group, stopped whole on a timeout, an overflow or
  Ctrl-C; stdin is closed; `OPENAI_API_KEY` is not in its environment; only its `cwd` is
  confined, the command is not. A cut result is saved whole in the temp directory and
  the result names the file.
- **`web_fetch` connects only to an address it checked:** the lookup is made inside the
  connection (`_CheckedBackend`), every address must be public, and every redirect hop is
  a new check. Without this a page could send the model to an internal service, or a
  DNS answer that changes between a check and a connection could. It reads at most
  5 MB after decompression and refuses a binary type before reading the body.
- **Pictures cost a request to refuse, so a refusal is remembered:** `read_file` returns
  a picture only for the four kinds the API takes, by the file's first bytes, up to
  3.7 MB; a model that refuses one is in `Agent.no_pictures`, and later pictures for it
  are replaced by a note before the request is made.

## Conventions

- Python 3.14 syntax, such as `except KeyboardInterrupt, EOFError:` (PEP 758). ruff takes
  its target from `requires-python`; `[tool.ruff]` sets `src = ["src", "tests"]` and
  turns on `BLE001` (a broad `except` says it is meant: `# noqa: BLE001`), `B905` and
  `RUF100` (a noqa that silences nothing goes).
- Google-style docstrings (`Args:` / `Returns:` / `Raises:`) on functions.
- Show the user only what they need, to decide or to know that something went wrong. A
  call that asks is one line, the question (at a terminal a short menu under it, which
  goes away when answered and leaves one line of what was decided); a call that runs
  freely is one line; a result
  is not shown (the reply says what it found); a server that starts is not announced; the
  banner is one line. Before adding output, ask whether anyone would miss it.
- A notice has a level (`info`, `warning`, `error`: dim, yellow, red) that the caller
  gives; only a warning or an error is in brackets, so that colour is not the only thing
  that tells them from a routine line: the agent's own are routine, and `Repl` and `cli.py` mark the failures and the
  mistakes. A status line that only reports a call shows plain arguments as `key=value`;
  the permission question is always the exact JSON that will run. A change to a file is
  shown as a diff coloured by its lines (`diff-add`, `diff-del`, `diff-hunk`,
  `diff-file`), whose first line names the file once. Notices wrap at the terminal's
  width with a hanging indent, a status block after reply text is set apart by a blank
  line (`Console.after_text` and `trailing` keep track), and the dimmed fence lines of
  a code block, and a reply's `**bold**`, `` `code` `` and `#` headings, are found a line
  at a time (`Markdown`, `markdown.py`). A command returns `Warn`
  or `Fail` (strings) to say how it went wrong.
- **The look is the terminal's own.** Styles use ANSI names (`ansicyan`,
  `ansibrightblack`), never hex, so a theme decides the colours; text must still read
  without them (the toolbar meter writes its percentage too), and `NO_COLOR` is honoured.
  A style for something prompt_toolkit already styles says `fg` and `bg` both: its
  completion menu is black on grey, which a change of `bg` alone leaves black on a dark
  terminal. A dotted class inherits its parent's style (a `decision.denied` class came out dim
  red), so give independent looks independent names. Status lines go through
  `print_formatted_text` only on a terminal with colour: it ends lines with `\r\n`, which
  a file or a pipe would keep.
- Output is a plain terminal: the system prompt forbids LaTeX in favor of Unicode math
  and says Markdown is not rendered (a reply's marks are only coloured, and stay in the
  text). Keep those if editing the prompt.
- The system prompt follows OpenAI's guidance for agentic coding models and depends on how
  the program behaves: the user sees each tool call but not its result, a denied call
  comes back as `The user denied this call.`, tool results are untrusted text, and a
  scratch script goes in the system's temporary location and is deleted afterwards
  (a request, not a guarantee: ergane tracks no files). If one of these changes in the
  code, change the prompt; `test_the_prompt_*` points at the words. The with-tools
  section is absent from `SYSTEM_PROMPT_NO_TOOLS`, which never mentions calls.
- **Test a prompt change against the models, not by reading it.** A harmless-looking
  change can break one: "never write a tool call out as text" made gpt-5.5 do it (right
  1 time in 6, 6 in 6 once the sentence was gone); tool language in the no-tools prompt
  made gemma4 return an empty reply; "a new temporary directory" made it create
  `temp_x/` in the project until the prompt named the system's temporary location. Compare
  old and new on scenarios with a pass/fail check each, several runs, on the models you
  use. Test a rule that touches files with a simulated shell that records commands and
  gives realistic answers (one that answered with only an exit code made gemma4 repeat
  `git status` 25 times, which looked like a regression and was the harness). The same
  goes for the summary prompt: a recall check (plant five facts, summarize twice, ask)
  gave 5 of 5 on gpt-5.5 and gemma4, and showed that gemma4 does not write the
  `<summary>` tags, which a strict parser would have refused.
- A tool's description is a prompt too: a change to one is compared on the models
  (which tool is called first for a dozen requests, several runs each) before it is
  kept. Trimming `run_command`'s was checked that way; it is also
  where "use `read_file`, not `cat`" lives, since the system prompt names no tool but `todo`.
