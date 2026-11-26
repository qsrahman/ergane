# Changelog

All notable changes to ergane are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versions follow
[Semantic Versioning](https://semver.org/): while the major version is 0, a minor version
may change how ergane is used (a flag, a `config.json` key, what a tool does), and a patch
version only fixes things.

The version is `version` in `pyproject.toml`, and `ergane --version` prints it. To release:
move what is under Unreleased to a new heading with the version and the date, change
`version` in `pyproject.toml`, run `uv lock`, commit, and tag the commit `vX.Y.Z`.

## [Unreleased]

### Fixed

- `edit_file` no longer edits a blank line when `old_text` is only whitespace that is not
  in the file.
- `edit_file` reports `old_text` that matches two overlapping places as ambiguous,
  instead of editing the first.
- `edit_file` keeps the blank lines that `new_text` ends with when `old_text` matches only
  after trailing whitespace and typographic characters are ignored.
- A malformed link (such as `http://[`) on a page no longer fails the whole `web_fetch`
  or `web_search`.
- `web_fetch` reads a raw deflate response whose first piece is a single byte.

## [0.1.0] - 2026-11-25

The first release. ergane began on 2026-10-16 as `muagent`, and was renamed on 2026-11-19.

### Added

**Chat**
- A terminal chat agent on the OpenAI Responses API, for OpenAI and for any compatible
  endpoint (`--base-url`), with streaming (`--no-streaming` to turn it off) and a
  `reasoning_effort` setting.
- A history kept on the client within a token budget (`max_context_tokens`): when it gets
  long the oldest turns are summarized by the model and the newest kept (`auto_compact`,
  `compact_at`, `keep_recent_tokens`), a long tool result is shortened to its start and
  end, and a request that is refused as too long is sent again, shorter.
- A loop guard that tells a model it is repeating itself, and ends a turn that is stuck.
- A stream that breaks before any text is sent again, and a turn stopped with Ctrl-C keeps
  the work it had done.
- Saved conversations: one file for each, in the state directory, the newest 50 kept;
  `--resume [ID]` and `/sessions` continue one.

**Tools**
- Built-in tools: `read_file` (text, and png, jpeg, webp and gif pictures for models that
  take them), `write_file`, `edit_file`,
  `run_command`, `web_fetch` and `web_search` (DuckDuckGo, no key). `builtin_tools`
  (`true` or `false`) and `--no-builtin` offer all of them or none. There is no tool to
  list a directory or to search files: the model uses `run_command` (`ls`, `grep`), which
  asks first.
- A `todo` tool for tasks of three or more steps, shown as a checklist, and `/todo`.
- A tool that changes anything asks first, in a menu with the change shown as a diff;
  "always allow" lasts for the session. Calls that only read run at the same time.
- Tools in `src/ergane/builtin/` are found without being listed; `TOOLS.md` is the guide.

**MCP servers**
- Servers listed under `mcpServers` in `config.json`, run over stdio or reached over
  streamable HTTP, started at the same time and, at a terminal, in the background while
  you type. Per server: `enabled`, `read_only`, `tools`, `instructions`, `images`,
  `startup_timeout`, `call_timeout`, `env`, `headers`, and `${NAME}` placeholders.
- `/servers` and `/reconnect`; a server that stops is started again, up to three times.

**Terminal**
- A prompt with history, Tab completion of commands and their arguments, multi-line input
  and pasted blocks, a toolbar (model, effort, git branch, tools allowed, servers, and a
  context meter that fits the terminal's width), and colours that are the terminal's own
  (`NO_COLOR` turns them off).
- Notices in three levels (plain and dim, yellow, red), wrapped; a reply's `**bold**`,
  `` `code` ``, headings and code fences lightly styled; one-line tool calls.
- Commands: `/help`, `/tools`, `/servers`, `/reconnect`, `/context`, `/compact`,
  `/summary`, `/todo`, `/model`, `/effort`, `/retry`, `/undo`, `/reset`, `/sessions`,
  `/clear`, `/exit`, `/quit`.

**Configuration**
- `$XDG_CONFIG_HOME/ergane/config.json` and a flag for each setting, with `--config`
  naming another file. `OPENAI_API_KEY` is the only variable ergane itself reads for a
  setting (the OpenAI SDK reads others: see Known issues). Every flag and key is in the
  README, with examples.

### Security
- No configuration, server list or saved session is read from the working directory, so a
  project's files cannot start a program.
- File tools are confined to the working directory and the system temp directory, never
  write through a symbolic link.
- `web_fetch` connects only to addresses it has checked, again on every redirect;
  `run_command` does not get `OPENAI_API_KEY`; tool results are treated as untrusted text.

### Known issues
- With no `base_url` set, the OpenAI SDK sends the API key and prompts to
  `OPENAI_BASE_URL` if the environment has it, which a project's `.envrc` can set. The SDK
  also reads `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID` and `OPENAI_CUSTOM_HEADERS`. Set
  `base_url` in `config.json` to pin the destination; the README lists the variables.

### Requirements
- Python 3.14 or later; the `mcp` extra for MCP servers.

[Unreleased]: https://github.com/qsrahman/ergane/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/qsrahman/ergane/releases/tag/v0.1.0
