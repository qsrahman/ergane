# Plan: local tools for ergane

Seven built-in tools in `src/ergane/builtin/`, designed from a reading of
[workshop-mcp](https://github.com/qsrahman/workshop-mcp) (a stdio MCP server with file,
shell, fetch and search tools) and of ergane's own `tools` branch, which already has a
`fetch_url` port. Nothing here is built yet. The decisions at the end are open; each has
a default.

## What workshop-mcp does that is worth keeping

- **Path checks.** Every tool resolves its path first (`paths.ts`): it expands `~`,
  resolves symlinks and walks up to the nearest existing parent. Writes never go through
  a symlink: a new file is created exclusively, an existing one must be a regular file,
  opened without following links. Errors name the path as the model wrote it.
- **Reads are capped to protect the model's context**, not memory. A cut result ends with
  the offset to continue from.
- **`edit_file` fails on an ambiguous `oldText`** unless `replaceAll` is set, and applies
  nothing if any edit fails. It returns a diff and has a `dryRun`. Its rule: "a silent
  wrong-place edit is the worst agent failure".
- **The shell is not confined to any directory.** It runs with stdin closed, kills the
  whole process tree on timeout, strips terminal escape codes, turns binary output into a
  note and keeps the start and end of long output. It removes `BRAVE_API_KEY` from the
  command's environment, so an injected command cannot print the key.
- **`fetch_url` has deliberate safeguards.** Every address is checked inside the
  connection's own lookup (no DNS rebinding), every redirect hop is checked, the body is
  capped after decompression, `robots.txt` is obeyed and fails closed, the user agent is
  honest, and pages are cached for five minutes.
- **Search results and fetched pages are labelled untrusted** in the tool descriptions.
- **`list_directory` and `search_files` exist only when the shell is off**, to save
  tokens. Its `search_files` matches names only; content search is left to the shell.
- **Not worth porting:** `read_multiple_files` (ergane runs read-only calls in parallel),
  file locks (ergane runs writes one at a time), the filesystem-timeout wrapper, the mixed
  camelCase and snake_case parameter names, and the extra Brave options.

## Constraints from ergane that shape the design

1. **A tool result over 20 KiB is cut from the end** (`MAX_OUTPUT_CHARS`), so a footer
   such as "continue with offset=N" would be lost. Each tool caps itself lower, at about
   16,000 characters, and keeps its own footer.
2. **`read_only` decides whether a tool asks.** It is a static per-tool flag, not per
   call. A read-only call runs without asking, on a thread that Ctrl-C cannot interrupt,
   so those tools must be fast and thread-safe.
3. **Tools cannot see `Config`.** There is no global config lookup, and a built-in is a
   module-level `@tool`. A tool that needs a setting, such as the search key, is built by
   a factory that `cli.py` calls.
4. **Credentials come only from the user's own files.** `config.json` holds the API key,
   so a file tool must never be able to read it.
5. **A prompt change must be tested against models** (`CLAUDE.md`). The plan relies on
   tool descriptions and leaves the system prompt alone.

## The tools

| Tool | Parameters | Asks? | Notes |
|---|---|---|---|
| `read_file` | `path`, `offset`, `limit` | no | Text only (UTF-8, no NUL bytes); cut at about 16,000 characters with the offset to continue from |
| `list_directory` | `path`, `recursive` | no | Directories end in `/`, symlinks in `@`, never followed; `.git` and `node_modules` listed but not entered; capped |
| `search_files` | `path`, `pattern`, `exclude` | no | Glob over names, as in workshop (content search: decision 1) |
| `write_file` | `path`, `content` | yes | Creates parent directories; refuses a symlink or a non-regular file |
| `edit_file` | `path`, `edits[]`, `dry_run` | yes | Ordered replacements, atomic; ambiguity is an error; returns a unified diff |
| `run_command` | `command`, `cwd`, `timeout_s` | yes | A fresh process each call, stdin closed, process group killed on timeout |
| `web_fetch` | `url`, `max_length`, `start_index`, `raw` | decision 3 | From the existing port |
| `web_search` | `query`, `count`, `freshness` | decision 3 | Brave; offered only when a key is configured |

Parameters are `snake_case`, and each has a description (`Annotated[..., Field(description=...)]`).

## Shared pieces

Modules starting with `_` in `builtin/` are skipped by `tools_in`, so helpers can sit
beside the tools:

```
builtin/
  _paths.py   resolve(path, roots) -> real path, or an error naming the path as written
  _text.py    binary sniff, cap with a footer, head-and-tail cut
  files.py    read_file, list_directory, search_files, write_file, edit_file
  _edit.py    the edit engine
  shell.py    run_command
  fetch.py    web_fetch (from the tools branch, renamed)
  search.py   web_search_tool(api_key) factory; cli adds it only with a key
```

## Security design

- **Roots.** Files and `run_command`'s `cwd` are confined to the working directory plus
  the system temp directory (the prompt tells the model to put scratch scripts there).
  Anything else is refused with a message telling the model to ask the user.
- **Deny list.** ergane's config directory and state directory are refused outright, so
  the API key, the prompt history and the saved sessions cannot be read even when the
  working directory is the home directory.
- **Symlinks.** Resolve with `realpath` before the root check. Writes use exclusive
  create or `O_NOFOLLOW`.
- **`run_command`'s environment.** Remove `OPENAI_API_KEY` (and the Brave key) from the
  child's environment, as workshop does for its key. It is the one secret ergane holds.
- **Network.** `web_fetch` keeps every safeguard of the port. Both web tools mark their
  results as untrusted in their descriptions.
- **Exfiltration.** Auto-run read tools plus an auto-run fetch would let an injected page
  read a file and send it out. That is why decision 3 matters.

## `edit_file` engine

Smaller than workshop's 437 lines: an exact match with a uniqueness check, `replace_all`,
CRLF preserved, a UTF-8 check and a `difflib` unified diff. On no match it reports the
closest lines. The whitespace-tolerant fallback and the re-indentation are left out until
a model test shows they are needed.

## Libraries

Python 3.14's standard library covers most of what the tools need, and the web stack is
already installed. Measured on this machine (dependency counts from a dry run, behaviour
by running it).

| Need | Use | Instead of |
|---|---|---|
| Name globs with `**` (`search_files`) | `PurePath.full_match`, `glob.translate` | workshop's `minimatch`, or `wcmatch` (+2 packages) |
| Unified diff for `edit_file`, closest lines on a mismatch | `difflib` | workshop's `diff` package and its mismatch engine |
| Kill the process tree on timeout (`run_command`) | `subprocess` with `start_new_session=True`, then `os.killpg` | `psutil` |
| Safe writes | `os.open` with `O_NOFOLLOW` or `O_EXCL`, `os.replace` | `filelock`, `atomicwrites` |
| Root and symlink checks | `Path.resolve()`, `is_relative_to` | nothing |
| Stripping terminal escape codes | one `re` pattern | a library |
| HTTP client | `httpx2` and `httpcore2` (installed) | `requests` |

Optional or to reconsider:

- **`pathspec`** (+1 package) matches `.gitignore` rules, so `list_directory` and
  `search_files` could skip ignored files and not only `.git` and `node_modules`.
- **`ripgrep`** (`rg`) is installed on this machine; content search could use it when it is
  present and fall back to pure Python. It cannot be assumed on another machine.
- **`protego` may be unnecessary.** The standard `urllib.robotparser` on 3.14 handled the
  three cases tried: `Disallow: /private/*/secret`, `Disallow: /*.pdf$`, and an `Allow`
  after a broader `Disallow`, each the same as `protego`. Phase 4 should run the branch's
  `robots.txt` tests with it swapped in; if they pass, drop the dependency.
- **`trafilatura`** extracts the main content and outputs markdown in one call. It would
  replace `beautifulsoup4`, `markdownify` and the page-furniture code, but a dry run shows
  14 new packages (`dateparser`, `htmldate`, `courlan` and others) and a slower import.
  Extraction quality is untested. Consider it only if the branch's output is poor on pages
  that matter.
- **`markitdown`** pulls in `numpy` and `onnxruntime` (13 packages): skip.
- **`ddgs`** (+3 packages) would do `web_search` without a key by scraping DuckDuckGo. It is
  unofficial and can break: an optional fallback at most.

Kept custom: the SSRF-safe client (address check inside the connection's own lookup,
per-hop redirect checks, post-decompression body cap). No library was evaluated for it.
Brave is one GET with `httpx2`, parsed with pydantic.

## Build order

Each phase is its own commit, with tests and a mutation check.

0. `_paths.py` and `_text.py`, with traversal, sibling-prefix, symlink and deny-list cases.
1. `read_file`, `list_directory`, `search_files`.
2. `write_file` and `edit_file`.
3. `run_command`.
4. `web_fetch`. Rebase the `tools` branch commit (`a7c7aaf`: it still says `muagent`,
   uses `pydantic-settings`, which main dropped, and sits on an old base), rename the
   tool, and bring its tests and licence notices. Try the standard `urllib.robotparser`
   in place of `protego` (see Libraries).
5. `web_search` and the `brave_api_key` setting.
6. Docs (README, `TOOLS.md`, `CLAUDE.md`, `TURN.md`), then a model eval on gpt-5.5 and
   gemma4: picks the file tools over shell `cat`, reports an ambiguous edit instead of
   guessing, and writes nothing outside the roots.

## Verification

- A test that every parameter of every built-in has a description (workshop has one).
- File tests on `tmp_path`.
- Fake HTTP servers for fetch and Brave, never the real network.
- A test that each tool's footer survives the 20 KiB cut.

## Decisions

1. **`search_files` scope.** Names only (as workshop), or also content search (`contains`,
   a regex, in pure Python)? *Default: both*, since `run_command` asks on every call and
   `grep` would prompt constantly.
2. **Confinement.** Working directory plus temp only, or unrestricted? *Default:
   confined*, with an `allowed_dirs` setting only if it is needed.
3. **Do `web_fetch` and `web_search` ask on each call?** The `tools` branch chose to ask,
   because a page can tell the model to fetch a URL that carries the user's data.
   *Default: ask*, with "Always allow" available.
4. **`web_fetch` source and dependencies.** Rebase the branch rather than rewrite it?
   Keep `beautifulsoup4`, `lxml` and `markdownify` as normal dependencies (`protego` only if
   the standard `robotparser` fails the branch's tests), or put them in an optional `web`
   extra? *Default: rebase, normal dependencies.*
5. **Brave key.** A `brave_api_key` in `config.json` (it supports `${BRAVE_API_KEY}`
   placeholders), not a new environment variable, since ergane reads only
   `OPENAI_API_KEY` on purpose. *Default: the config file.*
6. **`edit_file` fallback.** Exact-only for now, or add the whitespace-tolerant match?
   *Default: exact-only.*
