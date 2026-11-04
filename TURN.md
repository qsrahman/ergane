# How a turn flows

A **turn** is everything between you pressing Enter on a prompt and the next `> `.
One turn can hold several **rounds**: a round is one request to the model, and the
model keeps asking for tool calls until it answers with text.

This page follows one turn through the code: what is sent, what is kept, what you
see, and every way a turn can end. Functions are named rather than numbered by
line, because lines move. Everything is in [`src/muagent/`](src/muagent/).

## At a glance

```
 you type a line
      │
      ▼
 Repl.run ── session.prompt("> ") ──► Repl.handle
                                │
              /command ─────────┼──► the command's handler (no model call; /retry sends a prompt again)
                                │
              anything else     ▼
                            Repl.ask ── blank line ──► Agent.run_turn(prompt, on_text, on_notice, confirm)
                                                          │
       Context.begin_turn: history + prompt, trimmed ◄────┤
                                                          │
                       ┌──────────────────────────────────┤   at most MAX_TURNS rounds
                       │  _send_response    one request; the reply text streams to the screen
                       │  _response_status  failed, incomplete or empty? ──► stop and report it
                       │  tool calls?       none ──► stop: this was the answer
                       │  _run_tools        one output per call (asks you first when it must)
                       │  budget check      turn too big? ──► drop older results; still too big? stop
                       └──────────────────────────────────┤
                                                          ▼
                       Context.save_turn (or nothing), then a TurnError becomes one notice line
```

The agent does no terminal I/O. Everything you see passes through three callbacks
that `Repl.ask` hands it, which are methods of `Console`:

| Callback    | Is                | Carries                                                       |
|-------------|-------------------|---------------------------------------------------------------|
| `on_text`   | `Console.write`   | the reply, as it arrives                                      |
| `on_notice` | `Console.notice`  | one-line status: a call that runs, a trim, a problem          |
| `confirm`   | `Console.confirm` | the permission question; returns whether the call may run     |

## 1. The prompt (`Repl`)

- `Repl.run` reads a prompt with the prompt_toolkit session (plain `input("> ")` when input or
  output is redirected) and strips it. A pasted block arrives as one prompt, newlines
  included, and so does a prompt with lines added by hand (Alt-Enter, Ctrl-J). A blank line is skipped. Ctrl-C or end of input at this prompt ends muagent.
- `Repl.handle` lowercases the first word and looks it up among the slash commands.
  A lone word that starts with `/` and is not a command gives `[Unknown command: …]`.
  Anything else, including `/usr/bin is missing`, goes to the model.
- `Repl.ask` prints a blank line under what you typed, calls `Agent.run_turn`, and
  turns whatever comes out into at most one notice:

  | Raised               | Shown                              |
  |----------------------|------------------------------------|
  | `TurnError`          | its message                        |
  | `KeyboardInterrupt`  | `[Cancelled]`                      |
  | any other exception  | `[Agent error: Type: message]`     |

  It then ends the line and prints a blank line, whatever happened.

## 2. Starting the turn (`Context.begin_turn`)

`begin_turn(prompt, max_tokens)` returns the **items** for the first request: a copy
of the history plus `{"role": "user", "content": prompt}`. The history itself does
not change until the turn is saved, so a turn that fails leaves nothing behind. It
also remembers the prompt for `/retry` and marks it as not saved.

If the items are over `MAX_CONTEXT_TOKENS`, whole turns are dropped from the front
(each up to the next user message) until they fit, and you see
`[Context trimmed: oldest turns dropped]`. The size is an estimate: the UTF-8 bytes
of an item's JSON divided by four, with a reasoning item's opaque
`encrypted_content` left out. The newest turn is never dropped, so one enormous
prompt can still be over the limit.

`MAX_CONTEXT_TOKENS` is a budget for the whole request, not only the history, so the
items are given what is left of it after the part every request carries: the
instructions and the definitions of the tools (`Agent._fixed_tokens`, estimated the
same way). A long tool list therefore leaves less room for the history. The reply is
not counted.

## 3. A round: the request (`_send_response`)

Every request is built the same way, and is the only place the API is called:

| Field          | Value                                                                                                  |
|----------------|--------------------------------------------------------------------------------------------------------|
| `model`        | `OPENAI_MODEL`                                                                                         |
| `instructions` | the system prompt (`SYSTEM_PROMPT` if the agent has tools, else `SYSTEM_PROMPT_NO_TOOLS`), then the OS and the date |
| `input`        | **all** the items: the history, this prompt, and everything this turn has produced so far              |
| `tools`        | the schema of every tool, if there are any                                                             |
| `reasoning`    | `{"effort": …}`, only if `REASONING_EFFORT` is set                                                      |
| `store`        | `false`                                                                                                |

Because `store` is false, nothing lives on the server: **each round resends everything**.
That is why a turn with tools gets expensive, and why there is a budget check after
every round.

With `STREAMING=true` the text deltas go to `on_text` as they arrive and the final
response is taken from the stream's last event (`completed`, `incomplete` or
`failed`). If the stream ends without one, the turn ends with the reason from the
server's last `error` event (`Stream error: CODE: MESSAGE`) or `Stream ended without
a final response`. With `STREAMING=false` the whole text goes to `on_text` once. In
both modes the text is shown **before** the status is checked, so a partial reply is
never hidden.

Two layers retry a request. The SDK retries one that fails to start (connection
errors, 408, 409, 429 and 5xx, twice), but not one that breaks halfway. muagent does
that: if a stream **ends with no final event and no `error` event**, or **loses its
connection after it began**, and **no reply text has been shown yet**, the same
request is sent again, up to twice, with a notice such as `[Stream ended without a
final response; trying again (1 of 2)]`. This is safe because a tool runs only after
a complete response, so a retry repeats a request, never an action, and nothing was
on the screen to appear twice. It does not happen once text has been shown (the reply
would appear twice, perhaps differently), when the server ended the stream with an
`error` event (the same request would get the same answer), or when the request
never started (the SDK already tried).

## 4. Judging the response (`_response_status`)

| The response                                   | Result                                            |
|------------------------------------------------|---------------------------------------------------|
| status `incomplete`                            | `Response incomplete: REASON`                     |
| status `failed`                                | `Response failed: CODE: MESSAGE`                  |
| any other status but `completed` (or none)     | `Response not completed: STATUS`                  |
| has text, or has a tool call                   | fine                                              |
| neither, but a refusal                         | `Refused: …`                                      |
| neither                                        | `No text in response`                             |

A problem ends the turn on the spot, even if the response also asked for tool calls:
those calls are not run. A response with tool calls and no text is fine, because only
the response that ends the turn needs text.

## 5. Tool calls (`_run_tools`)

The calls of one response run one after another, in order. Each gets exactly one
`function_call_output`, whatever happens to it, because the API rejects a call with
no output (400) and an output with no call:

| The call                                   | On screen                       | The model is told                                  |
|--------------------------------------------|---------------------------------|----------------------------------------------------|
| names a tool that does not exist           | `[name(args)]`                  | `Unknown tool: NAME. Available tools: …`          |
| has arguments that fail `Tool.check`       | `[name(args)]`                  | `Invalid arguments: …` (you are never asked)       |
| is to a read-only tool                     | `[name(args)]`                  | its result                                         |
| is to a tool that asks, and you say yes    | `Allow name {args}? [y/N] y`    | its result                                         |
| is to a tool that asks, and you say no     | `Allow name {args}? [y/N] n`    | `The user denied this call.`                       |

- The arguments in the `[…]` line are made one line and cut at 80 characters. The permission
  line shows the arguments in full if they fit the terminal, otherwise with the middle
  of long values left out and `[y/N/?]`; `?` prints everything and asks again. End of
  input counts as no; Ctrl-C stops the turn (§8).
- **A result is never shown to you.** It goes to the model, whose reply is meant to
  say what it found.
- After the calls, the items grow by the model's own output (its reasoning and its
  `function_call` items, and any text it said first), then one output per call. The
  next round sends all of that back.

### Inside `Tool.call`

`check` the arguments, `runner` runs the tool, and any exception becomes the text
`Error: Type: message` for the model to read and correct (Ctrl-C is the exception to
that: it is not caught here, but `run_turn` keeps the turn, §8). A result over 20,480 characters is cut with
`[output truncated: N more characters]`.

For an MCP tool the runner hands the request to that server's own thread, where the
async SDK runs, and waits for the answer for at most `call_timeout` (60 s unless the
server's entry in `mcp.json` says otherwise). Text blocks are joined; other blocks
become a placeholder such as `[image: image/png]`; a server-side error gets an
`Error:` prefix. A call that times out is cancelled and the connection stays usable
(`Error: TimeoutError: MCP server 'x' did not answer within N s`); a server that has
died gives `Error: McpServerError: MCP server 'x': …`. See [README.md](README.md#mcp-servers).

## 6. When the rounds stop

After each round with tool calls the loop checks, in this order:

1. **The budget.** `_next_request_tokens` takes the `usage.input_tokens` the server
   reported for the request it just answered, which is exact, plus an estimate of the
   items added since (everything is estimated if the server sent no usage). Over
   `MAX_CONTEXT_TOKENS` first drops tool results: those of the rounds before the one
   that just ran, oldest first, each replaced by `[output dropped to fit the context]`
   until the estimate fits, with the notice `[Context full: older tool results
   dropped]`. The calls stay, so the model sees what it did and every call keeps an
   output, and the turn goes on (the model may call again for what it needs). If the
   latest round's results alone are over, nothing older is left to drop and the turn
   ends with `This turn grew past the context budget (about X of Y tokens). Ask for
   less, or raise MAX_CONTEXT_TOKENS`, saving nothing.

   The server's count includes the instructions and the tool definitions, so it
   measures the same thing as the trimming in §2, the whole request, but exactly. A
   one-word prompt is about 9 tokens of history and about 600 to the server with no
   tools, and the 42 small tools of the example below add about 3,000 more (by the
   estimate); real servers describe their tools at more length.
2. **The cap.** After `MAX_TURNS` rounds (25 by default) that all ended in tool calls,
   the turn is saved with a closing assistant note and ends with
   `Stopped after N model calls without a final answer`.

The loop also ends, without those checks, on a response with a problem or with no
tool calls (the answer).

## 7. Saving the turn (`Context.save_turn`)

Apart from the cap (§6) and a failure after a completed round (§8), a turn is saved only when its last response has text and
either nothing is wrong or the response was cut off (`incomplete`, so you saw the
partial reply it ends with). What is saved is the items, followed by the final reply as a plain
`{"role": "assistant", "content": text}` message. A turn with no reply is dropped:
saving a prompt without one would leave two user messages in a row.

`save_turn` then trims the history it keeps: **the reasoning, calls and outputs of
every turn but the newest are removed**, leaving the messages. The newest turn keeps
them, so you can ask a follow-up about what the tools found; the turn after that sees
only the replies. Calls and outputs are removed together, for the same reason as above.

## 8. How a turn can end

| Ending                                                | Saved?                  | You see                                                       |
|-------------------------------------------------------|-------------------------|---------------------------------------------------------------|
| the model answers                                     | yes, the whole turn     | the reply                                                     |
| the reply is cut off after some text                  | yes, the partial reply  | the text, then `[Response incomplete: REASON]`                |
| incomplete with no text, `failed`, other status       | no, unless a round already ran (see below) | the matching notice from §4                |
| refused, or no text                                   | no, unless a round already ran | `[Refused: …]`, `[No text in response]`                |
| the stream breaks before any text: tried again, twice at most | if a try works, as usual | `[… trying again (1 of 2)]`, then the reply |
| the stream keeps breaking, or breaks after some text, or ends with an `error` event | no, unless a round already ran | `[Stream ended without a final response]`, `[Stream cut off: …]` or `[Stream error: …]` |
| `MAX_TURNS` rounds of tool calls                      | yes, with a closing note| `[Stopped after N model calls without a final answer]`        |
| the turn outgrows `MAX_CONTEXT_TOKENS`                | older results are dropped and the turn goes on; no if the latest round alone is over | `[Context full: older tool results dropped]`, or `[This turn grew past the context budget …]` |
| Ctrl-C                                                | yes after a completed round or while tools run; no in the first model call | `[Cancelled]`                           |
| the API or the network fails (after the SDK's retries)| no, unless a round already ran | `[Agent error: Type: message]`                         |

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
- **Failures and the cap save.** A turn that fails after a completed round is kept with
  a note, like the cap, for the same reason.
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
| its reasoning | no | yes, in the turn and the next one; then removed |
| a denial | your own answer | `The user denied this call.` |

The system prompt relies on two of these facts: that you see the calls but not their
results (so it tells the model to say what it found), and that a denied call comes
back as that text (so it tells the model not to repeat it). If either changes in the
code, change the prompt.

`Console` keeps the screen tidy with one rule in one place (`_emit`): a status line
(a notice, the permission question) always starts on a fresh line, and plain text that
follows one, such as the reply, starts after a blank line.

## 10. Commands that touch a turn

| Command   | Effect                                                                                    |
|-----------|-------------------------------------------------------------------------------------------|
| `/retry`  | `Context.rewind()` drops the last turn if it was saved, then `ask` sends its prompt again  |
| `/undo`   | drops the last turn: the last user message and everything after it                         |
| `/reset`  | forgets the whole history                                                                  |
| `/model`, `/effort` | change the config; the next request reads it                                     |

## 11. What must stay true

- **User and assistant messages alternate** in the saved history. A turn is saved
  with its reply or not at all.
- **Every call has its output** before anything is sent or saved, and an output never
  outlives its call.
- **The history changes in three places only:** `save_turn`, `undo` (which `rewind`
  uses) and `reset`. `begin_turn` and the rounds work on a copy.
- **The newest user message is never trimmed away.**
- **The agent prints nothing.** If a new message is for the user, it goes through one
  of the three callbacks.

## An example

This is real output, from three MCP servers that are copies of the test server
(14 tools each, so 42). `docs` is marked `read_only`; `files` and `notes` are not:

```
muagent | model: gemma4:cloud | 42 tools | /help lists commands

> What is 1 plus 2? Use the docs__add tool. Answer in one short sentence.

[docs__add({"a":1,"b":2})]

1 plus 2 is 3.

> Now add 40 and 2 with the files__add tool. Answer in one short sentence.

Allow files__add {"a": 40, "b": 2}? [y/N] y

40 plus 2 is 42.

> Add 5 and 6 with the notes__add tool. Answer in one short sentence.

Allow notes__add {"a": 5, "b": 6}? [y/N] n

The tool call was denied, but 5 plus 6 is 11.
```

The first turn took two rounds. In round one the model answered with a call to
`docs__add`; muagent ran it at once (the tool is read-only), printed the line, and
sent the call and its result `3` back. In round two the model answered with text, the
turn ended, and it was saved. The second turn is the same with a question in the
middle. In the third, the output of the call was `The user denied this call.`, and
the model still answered.

## Where to look

| What                                  | Where                                                        |
|---------------------------------------|--------------------------------------------------------------|
| reading input, commands, `ask`        | `Repl` in [`repl.py`](src/muagent/repl.py), [`commands.py`](src/muagent/commands.py) |
| the turn, the rounds, the stops       | `Agent.run_turn` in [`agent.py`](src/muagent/agent.py)       |
| the request and the stream            | `Agent._send_response`                                       |
| calls and permissions                 | `Agent._run_tools`, `Tool` in [`tools.py`](src/muagent/tools.py) |
| history, trimming, saving, pruning    | `Context` in [`context.py`](src/muagent/context.py)          |
| everything printed                    | `Console` in [`console.py`](src/muagent/console.py)          |
| MCP servers                           | [`mcp_client.py`](src/muagent/mcp_client.py)                 |
| the system prompt                     | `SYSTEM_PROMPT` in `agent.py`                                |
| the layout of an exchange, pinned     | `TestReplRendering` in [`tests/test_repl.py`](tests/test_repl.py) |
