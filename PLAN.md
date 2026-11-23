# Plan: local tools for ergane

Built-in tools in `src/ergane/builtin/`, designed from a reading of four codebases:
[workshop-mcp](https://github.com/qsrahman/workshop-mcp) (a stdio MCP server with file,
shell, fetch and search tools), ergane's own `tools` branch (a `fetch_url` port), and the
tools of opencode, pi and codex. Nothing here is built yet. The decisions at the end are
open; each has a default.

## What the other agents do

| | opencode | pi | codex | workshop-mcp |
|---|---|---|---|---|
| Tools on by default | `read` (files, directories, images, PDFs), `glob`, `grep`, `edit`, `write`, shell, `webfetch`, `websearch`; GPT-5+ models get `patch` instead of `edit` and `write` | `read`, `bash`, `edit`, `write`; `grep`, `find`, `ls` are opt-in | the shell (`exec_command`), `apply_patch`, `view_image`, and a web search the model provider runs; **no** read, list or search tools | read, write, edit, shell, fetch; list and search only when the shell is off |
| Finding files and text | `glob` and `grep`, over a managed ripgrep | `grep` and `find` (opt-in) | the shell; the prompt says to prefer `rg` | name glob only; content search left to the shell |
| Edit | one edit per call: `oldString`, `newString`, `replaceAll` | `edits[]`, each matched against the original file, unique and not overlapping, no replace-all | a patch language (`apply_patch`) | `edits[]`, ordered, with `replaceAll` and `dryRun` |
| Edit matching | exact, then trailing whitespace, then smart quotes, dashes and spaces; line endings kept; a miss shows the closest lines | exact, then the same normalizations | exact, as a patch | exact, then trimmed whole lines |
| What the model reads back from a write | one line (`Edited path (2 replacements)`); the diff goes to the UI | one line (`Successfully replaced N block(s)`); the diff goes to the UI | one line | the whole diff |
| Read limits | 2,000 lines, 50 KB, 2,000 characters a line | 2,000 lines, 50 KB | by tokens (10,000) | 100,000 characters |
| Shell | timeout moves the command to the background; long output saved to a file, a tail returned with its path | process group; no default timeout; tail kept, full output saved to a file | a sandbox plus an approval policy | stdin closed, tree killed on timeout, head and tail kept |
| Permissions | rules per action; a path outside the project is its own approval; the diff is shown in the prompt | none: "containerize it" | sandbox plus approvals | none; optional allowed directories |
| Web fetch | `format` (text, markdown, html), 5 MB cap, 30 s default; redirects followed by hand and checked each hop; no robots.txt | none | none | SSRF-safe, robots.txt, paginated, cached |
| Web search | a provider the user picks; the query is the permission | none | **hosted**: a tool the API runs, with modes cached, live, disabled | Brave, with a key |

What this changes in the plan:

- **Fewer tools.** `pi` ships four and `codex` almost none, because the shell covers the rest.
  Each tool's definition is paid for in every request. `opencode` folds listing a directory
  into `read`.
- **A write or edit tells the model one line,** not a diff. All three show the diff to the
  user, in the approval prompt. ergane's approval question shows only the arguments, cut in
  the middle, so a change cannot be reviewed before it is approved (see Permissions).
- **Editing is one change per call,** with `replace_all`. It is the simplest schema to get
  right, and a model makes several calls (ergane runs writes one at a time).
- **Matching is exact, then normalized.** Both `opencode` and `pi` fall back to trailing
  whitespace, smart quotes, Unicode dashes and spaces. That is evidence it matters in
  practice.
- **Web search is a normal function tool,** not OpenAI's hosted one. `opencode` asks per query
  and lets the user pick a provider; `codex` uses a hosted tool, which ergane does not (it
  would work only on an OpenAI endpoint, and ergane could not ask before a search). Brave,
  through `httpx2`, works on any endpoint and goes through the same approval as the other
  tools.

## Constraints from ergane that shape the design

1. **A tool result over 20 KiB is cut from the end** (`MAX_OUTPUT_CHARS`), so a footer such
   as "continue with offset=N" would be lost. Each tool caps itself lower, at about 16,000
   characters, and keeps its own footer.
2. **`read_only` decides whether a tool asks.** It is a static per-tool flag, not per call. A
   read-only call runs without asking, on a thread that Ctrl-C cannot interrupt, so those
   tools must be fast and thread-safe.
3. **Tools cannot see `Config`.** There is no global config lookup, and a built-in is a
   module-level `@tool`. A tool that needs a setting is built by a factory that `cli.py`
   calls.
4. **Credentials come only from the user's own files.** `config.json` holds the API key, so a
   file tool must never be able to read it.
5. **A prompt change must be tested against models** (`CLAUDE.md`). The plan relies on tool
   descriptions and leaves the system prompt alone.

## The tools

| Tool | Parameters | Asks? | Notes |
|---|---|---|---|
| `read_file` | `path`, `offset`, `limit` | no | A file as text (UTF-8, no NUL bytes) or, for a directory, a listing (directories first, ending in `/`); an image comes back as a picture; capped at about 16,000 characters with the offset to continue from; a missing path suggests the nearest name |
| `search_files` | `path`, `name`, `contains`, `limit` | no | `name` is a glob over paths, `contains` a regular expression over file contents; at least one; results are paths, or `path:line: text` |
| `write_file` | `path`, `content` | yes | Creates parent directories; refuses a symlink or a non-regular file |
| `edit_file` | `path`, `old_text`, `new_text`, `replace_all` | yes | `old_text` must match once unless `replace_all`; exact, then normalized; one line back |
| `run_command` | `command`, `cwd`, `timeout_s` | yes | A fresh process each call, stdin closed, process group killed on timeout, head and tail of the output |
| `web_fetch` | `url`, `max_length`, `start_index`, `raw` | decision 3 | From the `tools` branch port |
| `web_search` | `query`, `count`, `freshness` | decision 3 | Brave; built by a factory in `cli.py` and offered only when a `brave_api_key` is configured |

`list_directory` is not a tool of its own: `read_file` lists a directory (decision 1).
Parameters are `snake_case`, and each has a description (`Annotated[..., Field(description=...)]`).

## Shared pieces

Modules starting with `_` in `builtin/` are skipped by `tools_in`, so helpers can sit beside
the tools:

```
builtin/
  _paths.py   resolve(path, roots) -> real path, or an error naming the path as written
  _text.py    binary sniff, cap with a footer, head-and-tail cut
  files.py    read_file, search_files, write_file, edit_file
  _edit.py    exact and normalized matching, line endings
  shell.py    run_command
  fetch.py    web_fetch (from the tools branch, renamed)
  search.py   web_search_tool(api_key) factory; cli adds it only with a key
```

## Security design

- **Roots.** Files and `run_command`'s `cwd` are confined to the working directory plus the
  system temp directory (the prompt tells the model to put scratch scripts there). Anything
  else is refused, with a message telling the model to ask the user.
- **Deny list.** ergane's config directory and state directory are refused outright, so the
  API key, the prompt history and the saved sessions cannot be read even when the working
  directory is the home directory.
- **Symlinks.** Resolve with `realpath` before the root check. Writes use exclusive create or
  `O_NOFOLLOW`.
- **`run_command`'s environment.** Remove `OPENAI_API_KEY` and `BRAVE_API_KEY` from the child's
  environment: they are the secrets ergane holds.
- **Network.** `web_fetch` keeps the safeguards of the port (address check inside the
  connection's lookup, every hop checked, body capped after decompression). Its results are
  labelled untrusted in the description.
- **Exfiltration.** Auto-run read tools plus an auto-run fetch would let an injected page read
  a file and send it out. That is why decision 3 matters.

### Permissions: show the change

`confirm` shows the arguments of a call, cut in the middle, and "Show full arguments" prints
them as JSON. For `write_file` and `edit_file` that is not reviewable. A `Tool` gets an
optional `preview(arguments) -> str`; the console's "Show full arguments" option prints it
when there is one. For an edit it is a unified diff, computed without writing; for a write,
the diff against the existing file, or the new file's start. This is the one change to the
core that the tools need (`Tool`, `Console.confirm`, with tests).

## `edit_file` engine

- Try `old_text` exactly. If it is not found, try again after normalizing both sides: trailing
  whitespace on each line, smart quotes, Unicode dashes and spaces. A normalized match
  replaces the file's own text, keeping its indentation and line endings (CRLF stays CRLF).
- More than one match is an error unless `replace_all`, naming the lines. No match is an
  error that shows the closest lines (`difflib`).
- The file must be valid UTF-8. The model is told `Edited path (N replacements)`.
- Left out: `edits[]`, `dry_run`, the diff in the result, and re-indenting the replacement.

## `run_command`

- `subprocess.Popen(shell=True, start_new_session=True, stdin=DEVNULL)`, killed as a group on
  timeout (default 120 s) or Ctrl-C; the child's environment lacks `OPENAI_API_KEY` and `BRAVE_API_KEY`.
- Output: the exit code when not zero, stderr labelled when there is any, escape codes
  stripped, binary output as a note, and over about 16,000 characters the head and the tail
  with a line saying how much was left out. (`opencode` and `pi` save the full output to a
  file and return its path; that adds a file ergane would have to track and delete, so it
  is left out.)

## `web_search`

Brave Search, as in workshop-mcp: one GET with `httpx2`, the response parsed with pydantic,
results as numbered title, URL, age and description. Parameters `query`, `count` (1 to 20)
and `freshness`; the others (country, language, safesearch, offset) are left out, since each
costs tokens in every request.

- **The key** is `brave_api_key` in `config.json` (it supports `${BRAVE_API_KEY}`
  placeholders), not a new environment variable: ergane reads only `OPENAI_API_KEY`, on
  purpose. The key is never put in a URL, a result or an error, and `run_command` does not
  pass it on (the child's environment loses `BRAVE_API_KEY` as well as `OPENAI_API_KEY`).
- **The tool is offered only when the key is set.** `cli.py` builds it and adds it, so
  `tools_in` is unchanged.
- **It asks before each call** (decision 3): the query leaves the machine, and a page can tell
  the model what to search for.
- One retry when the rate limit resets within 5 seconds; a clear error otherwise. Results are
  labelled untrusted in the description.

## Libraries

Python 3.14's standard library covers most of what the tools need, and the web stack is
already installed. Measured on this machine (dependency counts from a dry run, behaviour by
running it).

| Need | Use | Instead of |
|---|---|---|
| Name globs with `**` (`search_files`) | `PurePath.full_match`, `glob.translate` | `minimatch`, or `wcmatch` (+2 packages) |
| Unified diff and closest lines (`edit_file`, previews) | `difflib` | the `diff` package and a mismatch engine |
| Kill the process tree on timeout (`run_command`) | `subprocess` with `start_new_session=True`, then `os.killpg` | `psutil` |
| Safe writes | `os.open` with `O_NOFOLLOW` or `O_EXCL` | `filelock`, `atomicwrites` |
| Root and symlink checks | `Path.resolve()`, `is_relative_to` | nothing |
| Stripping terminal escape codes | one `re` pattern | a library |
| HTTP client | `httpx2` and `httpcore2` (installed) | `requests` |

Optional or to reconsider:

- **`pathspec`** (+1 package) matches `.gitignore` rules, so `search_files` could skip ignored
  files and not only `.git` and `node_modules`.
- **`ripgrep`** (`rg`) is installed on this machine, and `opencode`, `pi` and `codex` all rely
  on it, but none can assume it on another machine. `search_files` is pure Python; using
  `rg` when present is a later optimization.
- **`protego` may be unnecessary.** The standard `urllib.robotparser` on 3.14 handled the
  three cases tried: `Disallow: /private/*/secret`, `Disallow: /*.pdf$`, and an `Allow` after
  a broader `Disallow`, each the same as `protego`. Phase 5 should run the branch's
  `robots.txt` tests with it swapped in.
- **`trafilatura`** would replace `beautifulsoup4`, `markdownify` and the page-furniture code,
  but a dry run shows 14 new packages. Extraction quality is untested. Consider it only if
  the branch's output is poor on pages that matter.
- **`markitdown`** pulls in `numpy` and `onnxruntime` (13 packages): skip.
- **`ddgs`** (+3 packages) searches DuckDuckGo without a key by scraping. Unofficial: an
  optional fallback at most. Brave needs no library.

Kept custom: the SSRF-safe client (address check inside the connection's own lookup, per-hop
redirect checks, post-decompression body cap). No library was evaluated for it.

## Build order

Each phase is its own commit, with tests and a mutation check.

0. `_paths.py` and `_text.py`, with traversal, sibling-prefix, symlink and deny-list cases.
1. `read_file` (files, directories, images) and `search_files`.
2. The `Tool.preview` hook and the console change, then `write_file` and `edit_file`.
3. `run_command`.
4. `web_search`: the Brave client, the `brave_api_key` setting, the factory in `cli.py`.
5. `web_fetch`. Rebase the `tools` branch commit (`a7c7aaf`: it still says `muagent`, uses
   `pydantic-settings`, which main dropped, and sits on an old base), rename the tool, and
   bring its tests and licence notices. Try the standard `urllib.robotparser` in place of
   `protego`.
6. Docs (README, `TOOLS.md`, `CLAUDE.md`, `TURN.md`), then a model eval on gpt-5.5 and
   gemma4: picks the file tools over shell `cat`, reports an ambiguous edit instead of
   guessing, writes nothing outside the roots, and edits as well with `edit_file` as with a
   patch. If `gpt-5.5` does clearly worse with `edit_file`, try an `apply_patch` function
   tool for it, as `opencode` does for GPT-5+ models.

## Verification

- A test that every parameter of every built-in has a description (workshop has one).
- File tests on `tmp_path`.
- Fake HTTP servers for fetch, never the real network.
- A test that each tool's footer survives the 20 KiB cut.
- The approval preview: an edit's diff is shown, nothing is written before approval.
- A fake Brave server for `web_search`: results, an empty result, a rate limit, a bad key,
  and a check that the key appears in no URL, result or error.

## Decisions

1. **`list_directory`.** Fold it into `read_file` (as `opencode` does), or keep it as a tool?
   *Default: fold*; it removes a tool definition from every request.
2. **Confinement.** Working directory plus temp only, or unrestricted? *Default: confined*,
   with an `allowed_dirs` setting only if it is needed.
3. **Do `web_fetch` and `web_search` ask on each call?** The `tools` branch chose
   to ask, because a page can tell the model to fetch a URL that carries the user's data.
   *Default: ask*, with "Always allow" available.
4. **`web_search` provider.** Brave only, with the key in `config.json`? *Default: Brave.*
5. **`web_fetch`.** Rebase the branch rather than rewrite it, keeping `robots.txt`? `opencode`,
   `pi` and `codex` do not obey it, so it could go to save code. *Default: rebase and keep it.*
   Dependencies: `beautifulsoup4`, `lxml` and `markdownify` as normal dependencies (`protego`
   only if the standard parser fails the branch's tests), or an optional `web` extra?
   *Default: normal dependencies.*
6. **`edit_file` matching.** Exact only, or exact then normalized (as `opencode` and `pi`
   do)? *Default: exact then normalized.* (This reverses the earlier default, on the
   evidence that both ship it.)
7. **Which tools are on by default.** All, or the core four (`read_file`, `write_file`,
   `edit_file`, `run_command`) plus `search_files`, with the web tools offered only when configured
   (`web_search` needs its key; `web_fetch` could sit behind a setting, as in `pi`)? Measure
   the definitions with `/context` first. *Default: core five on, `web_search` with a key,
   `web_fetch` on.*
