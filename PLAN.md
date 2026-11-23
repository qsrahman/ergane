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
- **Web search needs no key if it reads DuckDuckGo.** `opencode` makes the provider a plugin
  the user picks, and `codex` uses a hosted tool (which ergane does not: it works on one
  vendor's endpoint only, and ergane could not ask before a search). The
  [duckduckgo-mcp-server](https://github.com/nickclyde/duckduckgo-mcp-server) shows a keyless
  way: a form POST to DuckDuckGo's HTML endpoint, parsed with BeautifulSoup. ergane takes
  that approach as a small built-in (see `web_search`) and leaves MCP for other providers.

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
| `web_fetch` | `url`, `max_length`, `start_index`, `raw` | decision 3 | From the `tools` branch port; its dependencies are an optional extra |
| `web_search` | `query`, `max_results` | decision 3 | Keyless: DuckDuckGo's HTML endpoint; best effort; the same optional extra |

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
  fetch.py    web_fetch (from the tools branch, renamed); heavy imports on first use
  search.py   web_search: the DuckDuckGo form POST, the parser, the block check
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
- **`run_command`'s environment.** Remove `OPENAI_API_KEY` from the child's environment: it is
  the secret ergane holds. Variables the user exported for MCP servers (`${NAME}`
  placeholders) are part of the user's own environment and are inherited, as with `pi` and
  `opencode`; the docs say so.
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
  timeout (default 120 s) or Ctrl-C; the child's environment lacks `OPENAI_API_KEY`.
- Output: the exit code when not zero, stderr labelled when there is any, escape codes
  stripped, binary output as a note, and over about 16,000 characters the head and the tail
  with a line saying how much was left out. (`opencode` and `pi` save the full output to a
  file and return its path; that adds a file ergane would have to track and delete, so it
  is left out.)

## `web_search`: DuckDuckGo, no key

A built-in of about 70 lines, written from the page structure and not copied from
duckduckgo-mcp-server (MIT; if any line is copied after all, its notice goes in
`THIRD_PARTY_NOTICES.md`).

- **The request.** A form POST to `https://html.duckduckgo.com/html` with `q`, `kl` (region,
  empty by default) and `kp` (safe search, moderate), through `httpx2`, with an honest user
  agent (`ergane/<version>`) and a 15 s timeout.
- **The parse.** BeautifulSoup with `html.parser`: each `.result`, its `.result__title a` and
  `.result__snippet`; results whose link contains `y.js` are ads and skipped; a link of the
  form `//duckduckgo.com/l/?uddg=...` is unwrapped to the real URL. The output is numbered:
  title, URL, snippet; `max_results` 1 to 20, default 10.
- **A block is said aloud.** DuckDuckGo answers a client it dislikes with an empty page and
  status `202`, not an error. A `202`, `403` or `429`, or an empty body, returns "search was
  refused or returned nothing, try again later", never "no results". A parsed page with no
  results says there are none.
- **Measured** (three runs each, one machine, one moment): an honest user agent got `200`
  and 10 results every time; the other server's browser-like headers (a Chrome 139 user
  agent with `Sec-Ch-Ua` and `Sec-Fetch-*`) got `202` and no results every time.
- **Not taken from that server:** `curl_cffi` Chrome TLS impersonation (it evades bot filters,
  and adds a native dependency), the spoofed headers, its three rate limiters (one query at
  a time from a terminal needs, at most, a minimum interval), the `ref://` link tokens, and
  its transport, SSL and DNS-rebinding options.
- **It asks before each call** (decision 3): the query leaves the machine, and a page can
  tell the model what to search for. Results are third-party text, labelled untrusted in the
  description.
- **Risk.** It scrapes an endpoint that is not an API, so it can break or be blocked without
  notice, and DuckDuckGo's terms were not checked. It is best effort; a user who needs a
  reliable provider lists a search MCP server instead (offered with `tools`, the key in the
  server's `env` as `${NAME}`), and the README says how, with a generic example.

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
  a broader `Disallow`, each the same as `protego`. Phase 4 should run the branch's
  `robots.txt` tests with it swapped in.
- **`trafilatura`** would replace `beautifulsoup4`, `markdownify` and the page-furniture code,
  but a dry run shows 14 new packages. Extraction quality is untested. Consider it only if
  the branch's output is poor on pages that matter.
- **`markitdown`** pulls in `numpy` and `onnxruntime` (13 packages): skip.
- **`ddgs`** (+3 packages) wraps the same DuckDuckGo scraping. Not needed: `web_search` is
  about 70 lines on `httpx2` and `beautifulsoup4`.

Kept custom: the SSRF-safe client (address check inside the connection's own lookup, per-hop
redirect checks, post-decompression body cap). No library was evaluated for it.

## Build order

Each phase is its own commit, with tests and a mutation check.

0. `_paths.py` and `_text.py`, with traversal, sibling-prefix, symlink and deny-list cases.
1. `read_file` (files, directories, images) and `search_files`.
2. The `Tool.preview` hook and the console change, then `write_file` and `edit_file`.
3. `run_command`.
4. `web_fetch`. Rebase the `tools` branch commit (`a7c7aaf`: it still says `muagent`, uses
   `pydantic-settings`, which main dropped, and sits on an old base), rename the tool, and
   bring its tests and licence notices. Move `beautifulsoup4`, `lxml` and `markdownify` into
   an optional `web` extra, imported on first use, so that a missing one gives a clear error
   and not a failed start. Try the standard `urllib.robotparser` in place of `protego`.
5. `web_search`, with a saved DuckDuckGo page as a test fixture; it joins the `web` extra
   (`beautifulsoup4` is all it needs).
6. Docs (README, with a note on using a search MCP server for another provider, `TOOLS.md`,
   `CLAUDE.md`, `TURN.md`), then a model eval on gpt-5.5 and gemma4: picks the file tools over shell `cat`, reports an ambiguous edit instead of
   guessing, writes nothing outside the roots, and edits as well with `edit_file` as with a
   patch. If `gpt-5.5` does clearly worse with `edit_file`, try an `apply_patch` function
   tool for it, as `opencode` does for GPT-5+ models.

## Verification

- A test that every parameter of every built-in has a description (workshop has one).
- File tests on `tmp_path`.
- Fake HTTP servers for fetch, never the real network.
- A test that each tool's footer survives the 20 KiB cut.
- The approval preview: an edit's diff is shown, nothing is written before approval.
- `web_search` on a saved results page: ordinary results, an ad, a `uddg` redirect, and no
  results; a fake server answering `202`, `403`, `429` and an empty body each gives the
  "refused" message, not "no results".
- `web_fetch` and `web_search` without the extra installed: the tool is not offered, or says
  how to install it, and ergane still starts.

## Decisions

1. **`list_directory`.** Fold it into `read_file` (as `opencode` does), or keep it as a tool?
   *Default: fold*; it removes a tool definition from every request.
2. **Confinement.** Working directory plus temp only, or unrestricted? *Default: confined*,
   with an `allowed_dirs` setting only if it is needed.
3. **Do `web_fetch` and `web_search` ask on each call?** The `tools` branch chose to ask,
   because a page can tell the model to fetch a URL that carries the user's data.
   *Default: ask*, with "Always allow" available.
4. **`web_search`.** A keyless DuckDuckGo built-in (best effort), MCP only, or a built-in
   Brave tool with a `brave_api_key` in `config.json`? *Default: the DuckDuckGo built-in,
   with the MCP route documented.*
5. **`web_fetch`.** Rebase the branch rather than rewrite it, keeping `robots.txt`? `opencode`,
   `pi` and `codex` do not obey it, so it could go to save code. *Default: rebase and keep it.*
   Dependencies (`beautifulsoup4`, `lxml`, `markdownify`; `protego` only if the standard
   parser fails the branch's tests; `web_search` needs only `beautifulsoup4`): normal
   dependencies, or an optional `web` extra? *Default: optional extra.*
6. **`edit_file` matching.** Exact only, or exact then normalized (as `opencode` and `pi`
   do)? *Default: exact then normalized.* (This reverses the earlier default, on the
   evidence that both ship it.)
7. **Which tools are on by default.** All, or the core four (`read_file`, `write_file`,
   `edit_file`, `run_command`) plus `search_files`, with `web_fetch` and `web_search` offered when
   the extra is installed? Measure the definitions with `/context` first. *Default: core five on,
   the web tools when installed.*
