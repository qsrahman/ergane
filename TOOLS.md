# Writing a tool

A tool is a Python function the model can call. This is everything you need to write
one, from the five-line version to what happens around a call. The code is in
[`tools.py`](src/ergane/tools.py); [TURN.md](TURN.md) walks a whole turn.

## The short version

Make a file in [`src/ergane/builtin/`](src/ergane/builtin/) with a function that has a
docstring, type hints and `@tool`:

```python
# src/ergane/builtin/arithmetic.py
from ergane.tools import tool


@tool
def add(a: int, b: int) -> str:
    """Add two integers and return the sum."""
    return str(a + b)
```

Start `ergane`: the tool is offered (`/tools` lists it). Nothing registers it: every
`Tool` at the top level of a module in `builtin/` is found when `ergane` starts
(`tools_in`, in `tools.py`). The order is the modules' names, then the order they are
defined in.

A tool you do not want to put in the repository can be an
[MCP server](README.md#mcp-servers) in any language instead.

## What the model sees

| Part | Where it comes from |
|---|---|
| name | the function's name; there is no way to give another |
| description | the function's docstring, which is required |
| parameters | the type hints, as a JSON schema |

A tool that returns a picture too is annotated `-> str | ToolOutput` and returns
`ToolOutput(text, (Image(base64_data, mime_type),))`, as `read_file` does for a png,
jpeg, webp or gif (`builtin/files.py`); the picture goes with the text of the result.

Write the docstring for the model, not for a programmer: what the tool does, when to use
it, what it returns, and what a wrong call looks like. The model decides from it alone
whether and how to call the tool. Every tool's definition goes into every request, so
keep it as short as that allows.

What OpenAI advises for function definitions (the function calling guide, "Best
practices for defining functions"; the GPT-5 and Codex prompting guides), as it applies
here:

- Say what the tool does, what each parameter is for (and its format), and what the
  result looks like. A parameter's words go in its `Field(description=...)`: ergane
  sends the whole docstring as the description, so an `Args:` section would be read as
  prose and say everything twice. (The OpenAI Agents SDK instead takes the summary as
  the description and the `Args:` entries as parameter descriptions.)
- Pass the intern test: could a person use the tool from these words alone? Put in what
  a wrong call looks like, and for a tool a shell command could stand in for, say which
  tool to use instead (`run_command` names the file tools); Codex's guide asks for a
  directive like that, or the model reaches for the terminal.
- Make invalid calls hard to write: `Literal` for a fixed set, `ge`/`le` for a range, a
  default for what the code knows. Do not make the model send what the program already
  has. A rule the schema cannot hold ("give at least one of name and contains") goes in
  the first lines of the description, and the error names the fix.
- Name tools by what they do (`web_search`, not `search`), and keep their output
  distinguishable from other tools'. OpenAI suggests fewer than 20 tools at the start
  of a turn (a soft limit).
- Descriptions are input tokens in every request: cut detail the model does not decide
  on (how a limit is implemented) and keep what it does (what is skipped, when output
  stops).
- Not adopted: `strict: true`. It needs every parameter required (an optional one
  becomes a nullable one), and ergane already checks the arguments with pydantic and
  sends back a message the model can correct from, on models that ignore `strict`.

The name must be letters, digits, `_` and `-` (the API refuses a request with any other),
and unique among the tools: when two tools have one name, the first is offered and
`ergane` says `[Tool NAME is defined twice; only the first is offered]`. Names of MCP
tools are `server__tool`, so a built-in name with `__` in it could clash with one; the
built-in is kept, and a notice says which tool was not offered.

## Parameters

Every parameter needs a type hint, and the function must be callable by keyword (no
`*args`, `**kwargs` or positional-only parameters). The schema is pydantic's, so what
pydantic understands works: `int`, `float`, `bool`, `str`, `list[str]`, `Literal["a",
"b"]`, `str | None`, nested models.

- A parameter with a default is optional; one without is required.
- A parameter's own description, and limits, go in `Annotated` with a pydantic `Field`:

  ```python
  from typing import Annotated
  from pydantic import Field


  @tool
  def read_file(
      path: str,
      max_chars: Annotated[int, Field(le=20_000, description="Longest text")] = 4_000,
  ) -> str:
      """Read a text file and return its start."""
      ...
  ```

- The model's arguments are checked against the hints before the function runs. Text
  that does not fit (`"x"` for an `int`) is not sent to the function: the model gets
  `Invalid arguments: a: …` and corrects the call. The user is not asked about such a
  call.
- A tool with no parameters is a function with none: `def look() -> str`.

## What it returns

The function must be synchronous and annotated `-> str`. Return the text the model
should read.

- **A failure is text too.** An exception that the function raises does not stop the turn:
  the model gets `Error: ValueError: message`, and can try again or say why it could not.
  So raise, with a message that says what is wrong and how to fix it, instead of
  returning a made-up result. Only Ctrl-C passes through.
- **Long output is cut** at 20 KiB (`MAX_OUTPUT_CHARS`), with a note saying how much was
  left out. It all goes into the next request, so return what is needed: a start, a
  count, a path to the rest.
- **Print nothing.** The user sees the call (one line, or the question) and not
  the result: the model's reply says what it found. A `print` lands in the middle of the
  terminal.
- The result is **untrusted text to the next step**: the model is told that a file or a
  page may contain orders, and not to follow them. Do not put secrets in a result.

A `@tool` function can only return text. A tool that returns pictures builds a `Tool`
from its parts (see `Tool` and `ToolOutput` in `tools.py`, which the MCP code does).

## Does it ask first? `read_only`

```python
@tool(read_only=False)
def save(path: str, text: str) -> str:
    """Write text to a file, replacing it."""
```

`read_only` is `True` unless you say otherwise. Say `False` for anything that changes
something outside the program: a file, a process, the network with a side effect. The
other option of `@tool` is `preview`, below.

- A tool that is not read-only asks the user before each call: the question is
  `Allow save {…}?` with the arguments, over a menu whose default is Deny. "Always allow"
  (`a`) allows that tool for the rest of the session. A read-only tool runs at once and is
  shown as a line.
- Read-only calls of one response run **at the same time**, on threads (up to 8), so such
  a tool must be safe to call from several threads. Any other call runs alone, after the
  ones before it.
- Ctrl-C stops a call that asks first, as any Ctrl-C in Python does. A read-only call runs
  on a thread, which cannot be interrupted: Ctrl-C ends the wait and the call runs on in
  the background. Keep calls short, or give them a timeout.

### Showing what a call would change: `preview`

The question shows the arguments, cut in the middle if they are long, which is not enough
to decide about a write. A tool that is not read-only may give `@tool` a `preview`: a
function with the same parameters, that returns what the call would change, as text.

```python
def show_save(path: str, text: str) -> str:
    return diff_of(path, text)  # reads, and changes nothing


@tool(read_only=False, preview=show_save)
def save(path: str, text: str) -> str:
    """Write text to a file, replacing it."""
```

The menu then always has a last option, "Show the change" (`?` at a typed question), which
prints that text, and asks again. The preview runs only when the user asks, in the main
thread, and must change nothing. If it raises, the user is told, and the question stays.
`write_file` and `edit_file` show a diff this way.

## Testing it

A `Tool` is called with the JSON text the model would send, and never raises:

```python
from ergane.builtin.arithmetic import add

assert add.call('{"a": 1, "b": 2}') == "3"
assert add.check('{"a": "x"}') is not None  # what is wrong, without running it
assert add.call('{"a": "x"}').startswith("Invalid arguments")
```

To see it through the agent, give it to `make_agent(tools=[add])` (see
[`tests/conftest.py`](tests/conftest.py)); the fake Responses API calls a tool named
`read_file`, which `tool_with` in [`tests/helpers.py`](tests/helpers.py) names a function
for. Put the tests for a built-in tool in `tests/`, and break the code to see one fail
(the project's rule for any behaviour change).

## What is checked when it is made

`@tool` refuses a function that cannot be a tool, when the module is imported, so
`ergane` does not start with one:

| Mistake | Error |
|---|---|
| no docstring | `ValueError: … needs a docstring` |
| not annotated `-> str` | `TypeError: … must be annotated to return str` |
| a parameter without a hint | `TypeError: … needs a type annotation` |
| `*args`, `**kwargs`, positional-only | `TypeError: … passable by keyword` |
| `async def` | `TypeError: … must not be async` |

A module that cannot be imported (a syntax error, a missing dependency) is left out with
a notice, `[Tools in NAME not loaded: …]`, before the first prompt; the other tools still
load.

## Keep it light

A module of `builtin/` is imported at every start. Import a heavy dependency inside the
function that needs it, not at the top, so that starting `ergane` does not pay for it. A module whose name starts with `_` (`_helpers.py`) is skipped, for code that is not a
tool.
