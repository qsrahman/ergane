# How a turn flows

A **turn** is everything between you pressing Enter on a prompt and the next `❯ `.
One turn can hold several **rounds**: a round is one request to the model, and the
model keeps asking for tool calls until it answers with text.

This page follows one turn through the code: what is sent, what is kept, what you
see, and every way a turn can end. Functions are named rather than numbered by
line, because lines move. Everything is in [`src/ergane/`](src/ergane/).

## At a glance

```
 you type a line
      │
      ▼
 Repl.run ── session.prompt(PROMPT) ──► Repl.handle
                                │
              /command ─────────┼──► the command's handler (no model call; /retry sends a prompt again)
                                │
              anything else     ▼
                            Repl.ask ── blank line ──► Agent.run_turn(prompt, on_text, on_notice, confirm, on_call)
                                                          │
       Compactor.compact_if_due: summarize if long ◄──────┤   (§2a)
       Context.begin_turn: summary + history + prompt ◄───┤
                                                          │
                       ┌──────────────────────────────────┤   until the model answers
                       │  Responder.send    one request; the reply text streams to the screen
                       │  response_problem  failed, incomplete or empty? ──► stop and report it
                       │  tool calls?       none ──► stop: this was the answer
                       │  _run_tools        one output per call (asks you first when it must)
                       │  loop guard        the same step again and again? ──► warn, then stop
                       │  budget check      turn too big? ──► drop older results; still too big? stop
                       └──────────────────────────────────┤
                                                          ▼
                       Context.save_turn (or nothing), then a TurnError becomes one notice line
```

The agent does no terminal I/O. Everything you see passes through four callbacks
that `Repl.ask` hands it, which are methods of `Console`:

| Callback    | Is                | Carries                                                       |
|-------------|-------------------|---------------------------------------------------------------|
| `on_text`   | `Console.write`   | the reply, as it arrives                                      |
| `on_notice` | `Console.notice`  | status line (in brackets when it is a warning or an error), wrapped at the terminal's width: a trim, a retry, a dropped result, a repeat warning |
| `confirm`   | `Console.confirm` | the permission question; returns whether the call may run     |
| `on_call`   | `Console.call`    | a call that runs without asking, as one line                  |
| `on_todos`  | `Console.todos`   | the model's checklist after it changed, with the finished steps dim and the current one bold |

## 1. The prompt (`Repl`)

- `Repl.run` reads a prompt with the prompt_toolkit session (plain `input("> ")` when input or
  output is redirected) and strips it. A pasted block arrives as one prompt, newlines
  included, and so does a prompt with lines added by hand (Alt-Enter, Ctrl-J). A blank line is skipped. End of input (Ctrl-D) ends ergane, but at a terminal, once there is a conversation, it asks for a second Ctrl-D first (`[Press Ctrl-D again, or type /exit, to quit]`, counted together with Ctrl-C). Ctrl-C clears the line being typed; on an empty line it
  asks for a second Ctrl-C in a row (`[Press Ctrl-C again, or type /exit, to quit]`), and any prompt
  in between starts the count again.
- `Repl.handle` lowercases the first word and looks it up among the slash commands.
  A lone word that starts with `/` and is not a command gives `[Unknown command: …]`.
  Anything else, including `/usr/bin is missing`, goes to the model.
- `Repl.ask` prints a blank line under what you typed, calls `Agent.run_turn`, and
  turns whatever comes out into at most one notice:

  | Raised               | Shown                              |
  |----------------------|------------------------------------|
  | `TurnError`          | its message                        |
  | `KeyboardInterrupt`  | `[Cancelled]`                      |
  | an `openai.APIError`  | `[API error 429 (code): message. hint]` (`describe_error`) |
  | any other exception  | `[Agent error: Type: message]`     |

  It then ends the line and prints a blank line, whatever happened.

## 2. Starting the turn (`Context.begin_turn`)

First, if the next request would be over `compact_at` (75%) of the budget, the oldest turns
are summarized (§2a). Then:

`begin_turn(prompt, max_tokens)` returns the **items** for the first request: the summary
message if there is one, a copy of the history, and `{"role": "user", "content": prompt}`. The history itself does
not change until the turn is saved, so a turn that fails leaves nothing behind. It
also remembers the prompt for `/retry` and marks it as not saved.

If the items are over `max_context_tokens`, whole turns are dropped from the front
(each up to the next user message) until they fit, and you see
`[Context trimmed: oldest turns dropped]`. The size is an estimate: the UTF-8 bytes
of an item's JSON divided by four, with a reasoning item's opaque
`encrypted_content` left out. The newest turn is never dropped, so one enormous
prompt can still be over the limit. The summary is never dropped. With summaries off, or
when one failed, this trimming is what keeps the request in the budget.

### 2a. Summarizing the oldest turns (`Compactor.compact`, in `compaction.py`)

The trigger is an estimate, before the turn: the instructions and tool definitions
(`_fixed_tokens`), the history with its summary, and the prompt, against `compact_at` of
`max_context_tokens`. Past it, `Context.plan_compaction` keeps the newest turns that
fit in `keep_recent_tokens` (at most half the trigger, and always the newest turn) and
cuts at the start of a turn, so a tool call stays with its output.

The turns before the cut go to the model as labelled text (`[User]:`, `[Assistant]:`,
`[Tool call]:`, `[Tool result]:` cut to 2,000 characters; reasoning left out), so that
it does not take them for a conversation to continue. The request has no tools, the
summarizing prompt in place of the system prompt, a cap of 8,000 tokens on the answer,
and `reasoning: low` where an effort is set. The answer has headings: goal, constraints,
progress, decisions, every request in order, exact references (paths, ids, errors),
next steps; an `<analysis>` scratchpad before it is dropped, and a missing
`<summary>` tag is forgiven, since not every model writes it. The previous summary is
an input, so there is always one summary, updated, never summaries of summaries.

The result is checked: empty, without headings, or longer than the text it replaces is
refused, and so is an answer that hit the cap. A refusal, an API error or a timeout is a
notice (`[Could not summarize earlier messages: …]`), the history stays whole, and the
turn goes on, with the trimming above as the fallback. After 3 failures in a row
summaries are not tried by themselves until `/compact` works. Ctrl-C leaves the history
as it was.

What the model then sees, first in every request, is
`<conversation_summary>` … `</conversation_summary>`, a `user` message that says it is
notes about what happened and not instructions: it was written from text that may hold a
web page's orders. It is not an item of the history (`Context.summary`), so the messages
still alternate and `/undo` and `/retry` work on the turns after it.

`max_context_tokens` is a budget for the whole request, not only the history, so the
items are given what is left of it after the part every request carries: the
instructions and the definitions of the tools (`Agent._fixed_tokens`, estimated the
same way). A long tool list therefore leaves less room for the history: the built-in
tools and `todo` are about 2,400 tokens, and the tool part of the prompt about 360 more,
so a request with the defaults carries about 2,800 more than a plain chat (4% of the
default budget). The reply is not counted.

## 3. A round: the request (`Responder.send`, in `responses.py`)

Every request is built the same way, and is the only place the API is called:

| Field          | Value                                                                                                  |
|----------------|--------------------------------------------------------------------------------------------------------|
| `model`        | the `model` setting                                                                                    |
| `instructions` | the system prompt (`SYSTEM_PROMPT` if the agent has tools, else `SYSTEM_PROMPT_NO_TOOLS`), then the OS and the date, then the todo list while steps are open and the agent has tools (`TodoList.notes`), then the notice to a model that repeats itself (§6) |
| `input`        | **all** the items: the history, this prompt, and everything this turn has produced so far              |
| `tools`        | the schema of every tool, if there are any (the built-ins that `builtin_tools` allows, then each server's; `cli.py`), and the `todo` tool with them (`Agent.offered`) |
| `reasoning`    | `{"effort": …}`, only if `reasoning_effort` is set                                                      |
| `store`        | `false`                                                                                                |

Because `store` is false, nothing lives on the server: **each round resends everything**.
That is why a turn with tools gets expensive, and why there is a budget check after
every round.

With `streaming` on the text deltas go to `on_text` as they arrive and the final
response is taken from the stream's last event (`completed`, `incomplete` or
`failed`). If the stream ends without one, the turn ends with the reason from the
server's last `error` event (`Stream error: CODE: MESSAGE`) or `Stream ended without
a final response`. With `streaming` off the whole text goes to `on_text` once. In
both modes the text is shown **before** the status is checked, so a partial reply is
never hidden.

Two layers retry a request. The SDK retries one that fails to start (connection
errors, 408, 409, 429 and 5xx, twice), but not one that breaks halfway. ergane does
that: if a stream **ends with no final event and no `error` event**, or **loses its
connection after it began**, and **no reply text has been shown yet**, the same
request is sent again, up to twice, with a notice such as `[Stream ended without a
final response; trying again (1 of 2)]`. This is safe because a tool runs only after
a complete response, so a retry repeats a request, never an action, and nothing was
on the screen to appear twice. It does not happen once text has been shown (the reply
would appear twice, perhaps differently), when the server ended the stream with an
`error` event (the same request would get the same answer), or when the request
never started (the SDK already tried).

## 4. Judging the response (`response_problem`)

| The response                                   | Result                                            |
|------------------------------------------------|---------------------------------------------------|
| status `incomplete`                            | `Response incomplete: REASON`                     |
| status `failed`                                | `Response failed: CODE: MESSAGE`                  |
| any other status but `completed` (or none)     | `Response not completed: STATUS`                  |
| has text, or has a tool call                   | fine                                              |
| neither, but a refusal                         | `Refused: …`                                      |
| neither                                        | `No text in response` (after one more try, `[The model sent an empty reply; trying again]`) |

A problem ends the turn on the spot, even if the response also asked for tool calls:
those calls are not run. A response with tool calls and no text is fine, because only
the response that ends the turn needs text.

## 5. Tool calls (`_run_tools`)

The calls of one response are answered in order, and each gets exactly one
`function_call_output`, whatever happens to it, because the API rejects a call with
no output (400) and an output with no call:

| The call                                   | On screen                       | The model is told                                  |
|--------------------------------------------|---------------------------------|----------------------------------------------------|
| names a tool that does not exist           | `▸ name  args`                  | `Unknown tool: NAME. Available tools: …`          |
| has arguments that fail `Tool.check`       | `▸ name  args`                  | `Invalid arguments: …` (you are never asked)       |
| is to a read-only tool                     | `▸ name  args`                  | its result                                         |
| is to a tool that asks, and you say yes    | `▸ name  args  allowed` ¹       | its result                                         |
| is to a tool that asks, and you say no     | `▸ name  args  denied` ¹        | `The user denied this call.`                       |
| is to a tool you answered `a` to earlier   | `▸ name  args`                  | its result                                         |

¹ At a terminal the question is a menu that goes away when answered, leaving this line;
with input or output redirected it is the typed line `Allow name {args}? [y/N/a] y`.

- The `▸` line (`Console.call`) is as wide as the terminal at most, the arguments on one
  line with the middle of long values left out. It only reports a call, so plain
  arguments are `key=value` (`▸ read_file  path=notes.txt`) and anything nested is JSON;
  the question that follows is always the exact JSON that will run. The permission question is cut the same
  way: it shows the arguments in full if they fit the terminal, otherwise with the middle
  of long values left out, and then offers to print everything and ask again (`?`: the
  menu's fourth option, "Show full arguments", or `[y/N/a/?]` typed). `a` allows the
  call and every later call to that tool until ergane ends (`Agent.always_allowed`, by
  tool name). At a terminal the question is a menu (`Console.choose`, which `Repl`
  supplies as `choose`, in `terminal.py`; Deny is first, so Enter denies); typed, anything but `y` and
  `a` is a no, and so is the end of input. Ctrl-C stops the turn (§8).
- A tool may add a **preview** to the question (`Tool.preview`): `write_file` and
  `edit_file` do, and the menu's "Show the change" option prints the diff of exactly the
  edit that would be made, or says why it would fail. The other built-ins that ask are
  `run_command`, `web_fetch` and `web_search`; `read_file`, `list_directory` and
  `search_files` only read. (`search_files` runs `rg`: at start, a tool whose program is not
  installed is left out and the user is told, so the model never sees it.)
- **Read-only calls that follow one another run at the same time**, each on a thread of its
  own (at most `MAX_PARALLEL_CALLS`, 8): they change nothing, so they cannot depend on one
  another, and a batch takes as long as its slowest call, not the sum. Any other call is a
  barrier: it waits for the calls before it, runs alone, and the user is asked about it
  alone. The `▸` lines of a batch print before it starts, in the order of the calls, and
  the outputs are added in that order whichever call ends first. Ctrl-C cancels the
  requests that are still waiting (`Tool.cancel`; for an MCP tool, the SDK tells the
  server) and keeps the outputs of the calls that had ended.
- **A result is never shown to you.** It goes to the model, whose reply is meant to
  say what it found. A result may carry a picture (`read_file` of a png): it goes in the
  same output item, and a model that takes none is refused once, then remembered
  (`Agent.no_pictures`, §8).
- **The `todo` tool** (`todos.py`) is offered with any other tool (`Agent.offered`; with
  none, the chat stays plain), runs without asking and has no `▸` line. Every
  call sends the whole list; two steps `in_progress` come back as an error and change
  nothing. After a round that changed the list, `on_todos` shows it (`Todos 2/5 done` and
  a line to a step). `/todo` shows the list.
- After the calls, the items grow by the model's own output (its reasoning and its
  `function_call` items, and any text it said first), then one output per call. The
  next round sends all of that back.

### Inside `Tool.call`

`check` the arguments, `runner` runs the tool, and any exception becomes the text
`Error: Type: message` for the model to read and correct (Ctrl-C is the exception to
that: it is not caught here, but `run_turn` keeps the turn, §8). A result over 20,480 characters is cut with
`[output truncated: N more characters]`.

For an MCP tool the runner hands the request to that server's own thread, where the
async SDK runs, and waits for the answer for at most `call_timeout` (300 s unless the
server's entry in `config.json` says otherwise). Text blocks are joined (so is a text file the server embedded; a result with only
structured content is that as JSON); other blocks become a placeholder such as
`[image: image/png]`; a server-side error gets an `Error:` prefix. A call that times out is cancelled and the connection stays usable
(`Error: TimeoutError: MCP server 'x' did not answer within N s`); a server whose connection
ended fails the call that finds it (`Error: McpServerError: MCP server 'x': …`) and is
started again by the next call (up to 3 times a session), which is not repeated for it. A call after 10 s of silence, or after a failed one, first pings; only an answer that says the connection is closed starts the server again, since one that answers with an error of its own is alive. See [README.md](README.md#mcp-servers).

## 6. When the rounds stop

After each round with tool calls the loop checks, in this order:

1. **The loop guard** (`loops.py`). A turn has no cap on its rounds. Each round is
   identified by its calls and what they returned. When the newest one has appeared more
   than twice among the last 10, the following requests carry a notice in their
   instructions (and you see one line) until the model takes a different step; when one
   step fills more than 5 of the last 10, the turn is saved with a closing assistant note
   and ends with `Stopped: the model keeps repeating the same tool calls`.
2. **The budget.** `_next_request_tokens` takes the `usage.input_tokens` the server
   reported for the request it just answered, which is exact, plus an estimate of the
   items added since (everything is estimated if the server sent no usage). Over
   `max_context_tokens` first shortens tool results: those of the rounds before the one
   that just ran, oldest first. A result over 8,000 characters keeps its first 4,000 and
   last 1,000 with `[... N characters hidden; call the tool again if you need them ...]`
   (the start says what it is, the end often how it went); what is still too much after
   that is replaced by `[output dropped to fit the context]`. The notice is
   `[Context full: older tool results dropped]`. The calls stay, so the model sees what
   it did and every call keeps an output, and the turn goes on (the model may call
   again for what it needs). If that is not enough, the latest round's own results
   are shortened the same way (start and end kept; notice `[Context full: this round's
   tool results shortened]`), never replaced by the placeholder, since the model has
   yet to read them. If even that is not enough, the calls have run and may have had
   effects, so the turn is kept, closed with `[Stopped after N model calls: its results
   passed the context budget]` and its results dropped (or every later request would
   carry them), and it ends with `This turn grew past the context budget (about X of Y
   tokens). Ask for less, or raise max_context_tokens`.

   The server's count includes the instructions and the tool definitions, so it
   measures the same thing as the trimming in §2, the whole request, but exactly. A
   one-word prompt is about 9 tokens of history and about 600 to the server with no
   tools, and the 42 small tools of the example below add about 3,000 more (by the
   estimate); real servers describe their tools at more length.

The loop also ends, without those checks, on a response with a problem or with no
tool calls (the answer).

## 7. Saving the turn (`Context.save_turn`)

Apart from the loop guard (§6) and a failure after a completed round (§8), a turn is saved only when its last response has text and
either nothing is wrong or the response was cut off (`incomplete`, so you saw the
partial reply it ends with). What is saved is the items, followed by the final reply as a plain
`{"role": "assistant", "content": text}` message. A turn with no reply is dropped:
saving a prompt without one would leave two user messages in a row.

`save_turn` then trims the history it keeps: **the reasoning, calls and outputs of
every turn but the newest are removed**, leaving the messages. The newest turn keeps
them, so you can ask a follow-up about what the tools found; the turn after that sees
only the replies. Calls and outputs are removed together, for the same reason as above.

### Saved to disk

Each of the five places the history changes (`save_turn`, `undo`, `reset`, `compact`,
`restore`) adds one to `Context.revision`. After `Repl.handle` has run a prompt or a
command, `Repl.save_session` compares it with what it last saved, and writes the
conversation (`SessionStore.save`, in `sessions.py`) when it differs: so a turn, a
Ctrl-C that kept a turn, `/undo`, `/retry` and `/compact` are all saved, and
a prompt that changed nothing is not. A conversation with nothing in it is not saved,
unless it was before, so that the file does not go on saying what was undone.
`/reset` forgets which session the conversation is, so the next save is a new one.

`--resume` and `/sessions` go the other way: `Repl.restore` hands the saved items and
summary to `Context.restore`, which leaves the history as `save_turn` would have, with
the last prompt known, so `/retry` and `/undo` work on it. What the server counted
of the earlier conversation is forgotten, and what the user allowed is not saved.

## 8. How a turn can end

| Ending                                                | Saved?                  | You see                                                       |
|-------------------------------------------------------|-------------------------|---------------------------------------------------------------|
| the model answers                                     | yes, the whole turn     | the reply                                                     |
| the reply is cut off after some text                  | yes, the partial reply  | the text, then `[Response incomplete: REASON]`                |
| incomplete with no text, `failed`, other status       | no, unless a round already ran (see below) | the matching notice from §4                |
| refused, or no text                                   | no, unless a round already ran | `[Refused: …]`, `[No text in response]`                |
| the stream breaks before any text: tried again, twice at most | if a try works, as usual | `[… trying again (1 of 2)]`, then the reply |
| the stream keeps breaking, or breaks after some text, or ends with an `error` event | no, unless a round already ran | `[Stream ended without a final response]`, `[Stream cut off: …]` or `[Stream error: …]` |
| the model repeats the same tool step (more than 5 of its last 10) | yes, with a closing note| `[The model is repeating itself; told to try something else]`, then `[Stopped: the model keeps repeating the same tool calls …]` |
| the turn outgrows `max_context_tokens`                | older, then the latest results are shortened and the turn goes on; if even that does not fit, kept with a note and its results dropped | `[Context full: older tool results dropped]`, `[… this round's tool results shortened]`, or `[This turn grew past the context budget …]` |
| Ctrl-C                                                | yes after a completed round or while tools run; no in the first model call | `[Cancelled]`                           |
| the server refuses a request because the model takes no pictures | as usual, if the retry works | `[The model does not take pictures, so they were dropped …]`: the pictures of tool results are taken out, the text stays (`[image removed: … tell the user you cannot see it]`), and the request is sent again; only if there was a picture to take out. The model is remembered (`Agent.no_pictures`, by name): later pictures are replaced by the same note before the request is made |
| the server refuses a request as too long (once a turn) | as usual, if the retry works | `[The request was too long for the model; trying again, shorter]`: before the first round the oldest turns are summarized, later the earlier results are shortened; then it is sent again |
| the API or the network fails (after the SDK's retries)| no, unless a round already ran | `[API error …]` or `[Agent error: Type: message]`                         |

"Unless a round already ran": once at least one round has completed (the model called
tools and they ran), these endings save the turn, closed with an assistant note
`[Stopped after N model calls: REASON]`, N being the rounds that completed and REASON
the notice you see (`Response failed: …`, `Stream error: …`) or, for an exception the
API client raised, its type (`APIConnectionError`). Every call has its output at that
point, so the history stays valid, and the model still knows the question and the work
done (some of it may have had effects). Say "continue" to go on; `/retry` drops this
turn like any saved one and starts the prompt over, so the model forgets that work.
A failure in the first round saves nothing, as before: there is no work to keep, and a
prompt with no reply would put two user messages in a row.

Three of these are worth knowing about:

- **Ctrl-C keeps the work.** Tools may already have changed things, so the model must
  know. Ctrl-C while waiting for a later round's response keeps the turn as it stands.
  Ctrl-C inside a round's tools gives every call without an output (the one that was
  running, which may have done part of its work, and those not yet run) the output
  `The user interrupted this call: it did not finish, or did not run.`, so the history
  stays valid. Either way the turn is closed with `[Stopped after N model calls:
  interrupted by the user]`. In the first round's model call nothing has run, and
  nothing is saved.
- **Failures and the loop guard save.** A turn that fails after a completed round is
  kept with a note, like the stop of the loop guard, for the same reason.
- **The budget drops results first.** A result is worth less than the work around it,
  so old results go before a turn is given up on; a turn is lost to the budget only
  when one round's results alone are over it.

## 9. Who sees what

| | You | The model |
|---|---|---|
| the system prompt and environment | no | yes, every request |
| the tool definitions | `/tools` lists names | yes, every request |
| your prompts and its replies | yes | yes (history) |
| tool calls | one line each | yes |
| tool results | **no** | yes |
| the todo list | `/todo`, and a line when it changes | yes, in the instructions while steps are open |
| a warning that it repeats itself | one line | yes, a notice in the instructions |
| its reasoning | no | yes, in the turn and the next one; then removed |
| a denial | your own answer | `The user denied this call.` |

The system prompt relies on two of these facts: that you see the calls but not their
results (so it tells the model to say what it found), and that a denied call comes
back as that text (so it tells the model not to repeat it). If either changes in the
code, change the prompt.

`Console` keeps the screen tidy with one rule in one place (`_emit`): a status line
(a notice, a call, the permission question) always starts on a fresh line, and plain
text that follows one, such as the reply, starts after a blank line. The reply is printed as
the model wrote it, with light colour for `**bold**`, `` `code` ``, headings and the fences of
code blocks (`Markdown` in `markdown.py`; the marks stay in the text). A status line is
made of styled pieces: prompt_toolkit prints them on a terminal with colour, and only
their text is printed when output is redirected or `NO_COLOR` is set.

## 10. Commands that touch a turn

| Command   | Effect                                                                                    |
|-----------|-------------------------------------------------------------------------------------------|
| `/retry`  | `Context.rewind()` drops the last turn if it was saved, then `ask` sends its prompt again  |
| `/undo`   | drops the last turn: the last user message and everything after it                         |
| `/reset`  | forgets the whole history, the summary too; the next save is a new session                 |
| `/sessions [N or ID]` | `Repl.restore` puts a saved session in place of the history (see §7)           |
| `/compact [focus]` | summarizes the oldest turns now (§2a); the focus says what the summary must keep |
| `/summary` | shows the summary                                                                          |
| `/todo`   | shows the model's todo list, which `/reset` and a restored session clear                   |
| `/model`, `/effort` | change the config; the next request reads it                                     |

## 11. What must stay true

- **User and assistant messages alternate** in the saved history. A turn is saved
  with its reply or not at all.
- **Every call has its output** before anything is sent or saved, and an output never
  outlives its call.
- **The history changes in five places only:** `save_turn`, `undo` (which `rewind`
  uses), `reset`, `compact` and `restore` (a saved session, from `--resume` or
  `/sessions`, replaces the whole history). `begin_turn` and the rounds work on a copy.
- **The newest user message is never trimmed away,** and neither is the summary.
- **A summary replaces turns only when it is good:** the history is whole after a failure.
- **A turn has no cap on its model calls.** The loop guard warns a model that repeats a
  step and stops one that goes on; the notice is in the instructions, never in a tool
  result, and does not outlive the turn.
- **The todo tool is offered only with another tool,** and its list reaches the
  instructions only then, so a chat with no tools never mentions one.
- **The agent prints nothing.** If a new message is for the user, it goes through one
  of the four callbacks.

## An example

This is real output, from three MCP servers that are copies of the test server
(14 tools each, so 42). `docs` is marked `read_only`; `files` and `notes` are not:

```
ergane · gemma4:cloud · /help for commands

❯ What is 1 plus 2? Use the docs__add tool. Answer in one short sentence.

▸ docs__add  {"a": 1, "b": 2}

1 plus 2 is 3.

❯ Now add 40 and 2 with the files__add tool. Answer in one short sentence.

▸ files__add  {"a": 40, "b": 2}  allowed

40 plus 2 is 42.

❯ Add 5 and 6 with the notes__add tool. Answer in one short sentence.

▸ notes__add  {"a": 5, "b": 6}  denied

The tool call was denied, but 5 plus 6 is 11.
```

The first turn took two rounds. In round one the model answered with a call to
`docs__add`; ergane ran it at once (the tool is read-only), printed the line, and
sent the call and its result `3` back. In round two the model answered with text, the
turn ended, and it was saved. The second turn is the same with a question in the
middle. In the third, the output of the call was `The user denied this call.`, and
the model still answered.

## Where to look

| What                                  | Where                                                        |
|---------------------------------------|--------------------------------------------------------------|
| reading input, commands, `ask`           | `Repl` in [`repl.py`](src/ergane/repl.py), [`commands.py`](src/ergane/commands.py) |
| the prompt, the toolbar, the permission menu | [`terminal.py`](src/ergane/terminal.py) |
| a reply's light styling                | `Markdown` in [`markdown.py`](src/ergane/markdown.py), used by `Console.write` |
| tools that need a program (`rg`)       | `Tool.requires`, `without_missing_programs` in [`tools.py`](src/ergane/tools.py); [`builtin/_rg.py`](src/ergane/builtin/_rg.py) |
| saved sessions                        | `SessionStore` in [`sessions.py`](src/ergane/sessions.py); `Repl.save_session` and `restore` |
| the turn, the rounds, the stops       | `Agent.run_turn`, `_ask`, `_run_round` in [`agent.py`](src/ergane/agent.py) |
| the request and the stream            | `Responder` in [`responses.py`](src/ergane/responses.py)    |
| calls and permissions                 | `Agent._run_tools`, `Tool` in [`tools.py`](src/ergane/tools.py) |
| the todo tool and its list            | `TodoList` in [`todos.py`](src/ergane/todos.py); `Agent.offered` |
| the loop guard                        | `Agent._watch_loop` in [`agent.py`](src/ergane/agent.py), the functions in [`loops.py`](src/ergane/loops.py) |
| history, summary, trimming, saving    | `Context` in [`context.py`](src/ergane/context.py)          |
| summarizing                           | `Compactor` in [`compaction.py`](src/ergane/compaction.py) |
| the too-long retry                    | `_shrink` in [`agent.py`](src/ergane/agent.py) |
| pruning results                       | `drop_outputs`, `drop_images` in [`outputs.py`](src/ergane/outputs.py) |
| transcript, summary check             | `transcript`, `extract_summary` in [`compaction.py`](src/ergane/compaction.py) |
| errors and how they are worded        | [`errors.py`](src/ergane/errors.py)                         |
| everything printed                    | `Console` in [`console.py`](src/ergane/console.py)          |
| MCP servers                           | [`mcp_client.py`](src/ergane/mcp_client.py) (one) and [`mcp_hub.py`](src/ergane/mcp_hub.py) (all of them), their config in [`config.py`](src/ergane/config.py) |
| the prompts                           | [`prompts.py`](src/ergane/prompts.py)                       |
| the layout of an exchange, pinned     | `TestReplRendering` in [`tests/test_repl.py`](tests/test_repl.py) |
