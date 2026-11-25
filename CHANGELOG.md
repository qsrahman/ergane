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
  take them), `list_directory`, `search_files` (by name, by content or both, on
  [ripgrep](https://github.com/BurntSushi/ripgrep)), `write_file`, `edit_file`,
  `run_command`, `web_fetch` and `web_search` (DuckDuckGo, no key). `builtin_tools` and
  `--no-builtin` choose which are offered; a tool whose program is missing is left out
  with a warning.
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
  naming another file. `OPENAI_API_KEY` is the only variable read. Every flag and key is
  in the README, with examples.

### Security
- Credentials, servers and saved sessions are read only from the user's own files, never
  from the working directory, so a project cannot change where your key and prompts go or
  start a program.
- File tools are confined to the working directory and the system temp directory, never
  write through a symbolic link, and `search_files` ignores a project's own ignore files
  and ripgrep configuration.
- `web_fetch` connects only to addresses it has checked, again on every redirect;
  `run_command` does not get `OPENAI_API_KEY`; tool results are treated as untrusted text.

### Requirements
- Python 3.14 or later; [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) for
  `search_files`; the `mcp` extra for MCP servers.

[Unreleased]: https://github.com/qsrahman/ergane/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/qsrahman/ergane/releases/tag/v0.1.0
