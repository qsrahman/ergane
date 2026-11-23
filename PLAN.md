# Plan: local tools for ergane

Built-in tools in `src/ergane/builtin/`, designed from a reading of four codebases:
[workshop-mcp](https://github.com/qsrahman/workshop-mcp) (a stdio MCP server with file,
shell, fetch and search tools), ergane's own `tools` branch (a `fetch_url` port), and the
tools of opencode, pi and codex. Nothing here is built yet. The decisions at the end were
settled in a round of questions and answers, and each is recorded there with what follows
from it.

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
| `read_file` | `path`, `offset`, `limit` | no | A file as text (UTF-8, no NUL bytes) or, for a directory, a listing (directories first, ending in `/`); capped at about 16,000 characters with the offset to continue from; a missing path suggests the nearest name. Pictures come later (see Build order) |
| `search_files` | `path`, `name`, `contains`, `limit` | no | `name` is a glob over paths, `contains` a regular expression over file contents; at least one; results are paths, or `path:line: text`; `.gitignore` is honoured |
| `write_file` | `path`, `content` | yes | Creates parent directories; refuses a symlink or a non-regular file |
| `edit_file` | `path`, `old_text`, `new_text`, `replace_all` | yes | `old_text` must match once unless `replace_all`; exact, then normalized; one line back |
| `run_command` | `command`, `cwd`, `timeout_s` | yes | A fresh process each call, stdin closed, process group killed on timeout, head and tail of the output |
| `web_fetch` | `url`, `max_length`, `start_index`, `raw` | yes | From the `tools` branch port, without `robots.txt`; ordinary dependencies |
| `web_search` | `query`, `max_results` | yes | Keyless: DuckDuckGo's HTML endpoint; best effort |

`list_directory` is not a tool of its own: `read_file` lists a directory (decision 1).
Parameters are `snake_case`, and each has a description (`Annotated[..., Field(description=...)]`).

### All seven are on, and can be switched off

Every built-in is offered by default. A `builtin_tools` setting in `config.json` takes `false`
(none) or a list of names (only those, like an MCP server's `tools`), and `--no-builtin`
turns them all off for one run. With no built-in and no MCP server, ergane is the plain chat
of `CLAUDE.md`, with `SYSTEM_PROMPT_NO_TOOLS`. `cli.py` does the filtering when it builds the
list it gives the `Agent`; `tools_in` is unchanged. Definitions cost tokens in every request:
measure them with `/context` once they exist.

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
  connection's lookup, every hop checked, body capped after decompression, binary content
  refused) and drops `robots.txt`. Its results are labelled untrusted in the description.
- **Exfiltration.** Auto-run read tools plus an auto-run fetch would let an injected page read
  a file and send it out. That is why both web tools ask on every call (decision 3).

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

- **`pathspec`** (+1 package, a normal dependency) matches `.gitignore` rules, so `search_files`
  and a recursive listing skip what git ignores. `.git` itself is never entered.
- **`ripgrep`** (`rg`) is installed on this machine, and `opencode`, `pi` and `codex` all rely
  on it, but none can assume it on another machine. `search_files` is pure Python; using
  `rg` when present is a later optimization.
- **`trafilatura`** would replace `beautifulsoup4`, `markdownify` and the page-furniture code,
  but a dry run shows 14 new packages. Extraction quality is untested. Consider it only if
  the branch's output is poor on pages that matter.
- **`markitdown`** pulls in `numpy` and `onnxruntime` (13 packages): skip.
- **`ddgs`** (+3 packages) wraps the same DuckDuckGo scraping. Not needed: `web_search` is
  about 70 lines on `httpx2` and `beautifulsoup4`.

Kept custom: the SSRF-safe client (address check inside the connection's own lookup, per-hop
redirect checks, post-decompression body cap). No library was evaluated for it.

## Build order

Each phase is its own commit, with tests and a mutation check. Start with phases 0 and 1.

0. `_paths.py` and `_text.py`, with traversal, sibling-prefix, symlink and deny-list cases.
1. `read_file` (files and directories, text only) and `search_files` (with `pathspec`), and the
   switch: the `builtin_tools` setting, `--no-builtin`, and the filtering in `cli.py`.
2. The `Tool.preview` hook and the console change, then `write_file` and `edit_file`.
3. `run_command`.
4. `web_fetch`. Rebase the `tools` branch commit (`a7c7aaf`: it still says `muagent`, uses
   `pydantic-settings`, which main dropped, and sits on an old base), rename the tool, and
   bring its tests and licence notices. Remove `robots.txt` (and with it `protego`). Add
   `beautifulsoup4`, `lxml` and `markdownify` to the dependencies.
5. `web_search`, with a saved DuckDuckGo page as a test fixture. It needs only
   `beautifulsoup4`.
6. Pictures in `read_file`, as a small commit: png, jpeg, webp and gif returned the way MCP
   pictures are, so a model that takes none has them dropped with the existing notice.
7. Docs (README, `TOOLS.md`, `CLAUDE.md`: the Overview's "none yet", the plain chat and the
   invariants; `TURN.md`; a note on using a search MCP server for another provider), then a
   model eval on gpt-5.5 and gemma4: picks the file tools over shell `cat`, reports an
   ambiguous edit instead of guessing, and writes nothing outside the roots.

## Verification

- A test that every parameter of every built-in has a description (workshop has one).
- File tests on `tmp_path`.
- Fake HTTP servers for fetch, never the real network.
- A test that each tool's footer survives the 20 KiB cut.
- The approval preview: an edit's diff is shown, nothing is written before approval.
- `web_search` on a saved results page: ordinary results, an ad, a `uddg` redirect, and no
  results; a fake server answering `202`, `403`, `429` and an empty body each gives the
  "refused" message, not "no results".
- `builtin_tools`: `false`, a list, and `--no-builtin` each give the right tools, and with
  none and no MCP server the request has no tools and `SYSTEM_PROMPT_NO_TOOLS`.
- `.gitignore`: an ignored file is neither listed nor found; `.git` is never entered.

## Decisions

Settled in a round of questions and answers. Each line is the choice and what it commits the
plan to.

| # | Question | Choice | What follows |
|---|---|---|---|
| 1 | Finding and listing files | `read_file` lists a directory; one `search_files` with a name glob and a content regex | No `list_directory`; content search is pure Python and never prompts |
| 2 | Where the file tools reach | The working directory plus the system temp directory | Everything else refused; ergane's config and state directories always refused |
| 3 | Do the web tools ask | Both ask on every call, "Always allow" available | `read_only` false for `web_fetch` and `web_search` |
| 4 | `web_search` source | A keyless DuckDuckGo built-in, best effort | About 70 lines; an MCP server is the documented route for another provider |
| 5 | `web_fetch` | Rebase the branch; drop `robots.txt`; ordinary dependencies | No `protego`; `beautifulsoup4`, `lxml` and `markdownify` join `dependencies`; no `web` extra |
| 6 | `edit_file` matching | Exact, then normalized | Trailing whitespace, smart quotes, Unicode dashes and spaces, file's own text kept |
| 7 | Which tools are offered | All seven, always | A `builtin_tools` setting and `--no-builtin` keep the plain chat reachable |
| 8 | `run_command` timeout | 120 seconds, `timeout_s` to raise it | The process group is killed at the limit |
| 9 | First phase | 0 and 1: the helpers, `read_file`, `search_files` and the switch | Read-only tools first, before anything writes |
| 10 | `apply_patch` for GPT models | Never: `edit_file` only | One editing mechanism for every model; no per-model tool choice |
| 11 | Pictures in `read_file` | Text only first, pictures in a later small commit | Phase 6 |
| 12 | Ignore rules in `search_files` | Honour `.gitignore` with `pathspec` | One more dependency |
| 13 | `run_command` environment | Inherit everything except `OPENAI_API_KEY` | Git credentials and toolchains work; variables exported for MCP servers are inherited and the docs say so |
| 14 | Long command output | Head and tail with a note, no file | Nothing for ergane to track or delete |
