"""The agent: runs a turn against the model, with tools, inside a token budget.

`Agent` does no terminal I/O; what the user should see goes through the callbacks of
`run_turn`. TURN.md walks a turn through this module. What must stay true when you
change it:

- Every function call has its output before anything is sent or saved (the API
  rejects a call without one, and an output without its call). Rounds that fail or
  are interrupted are answered, never left open.
- A turn has no cap on its model calls. A model that repeats the same tool step is
  told in the instructions, then stopped (`loops.py`); the context budget bounds each
  request, and Ctrl-C ends a turn.
- A turn is saved whole or not at all, with one exception: once a round has
  completed (tools may have had effects, and the model must know), a failure, Ctrl-C
  or a stop by the loop guard keeps the turn with a closing note. A failure in the first
  round saves nothing.
- A stream that breaks before any text is shown is sent again (`STREAM_RETRIES`, in
  `responses.py`): Ollama's cloud proxy lost a connection 100 s into a request and a turn
  of 25 rounds was thrown away. Tools run only after a complete response, so a retry
  repeats a request, never an action.
- Summaries (`compact`, in `compaction.py`) change nothing unless the summary is good,
  and a failure never stops the turn.
"""

import platform
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, cast

from openai import OpenAI
from openai.types.responses import (
    FunctionToolParam,
    Response,
    ResponseFunctionToolCall,
    ResponseInputItemParam,
    ResponseInputParam,
)

from ergane.compaction import SUMMARY_MAX_TOKENS, Compactor
from ergane.config import Config
from ergane.context import Context, as_dicts, estimate_tokens
from ergane.errors import TurnError, is_context_overflow, is_images_rejected
from ergane.loops import (
    MAX_REPEATS,
    WARN_REPEATS,
    WINDOW,
    is_looping,
    last_repeats,
    loop_note,
    step_signature,
)
from ergane.outputs import (
    IMAGE_REFUSED,
    answer_open_calls,
    drop_images,
    drop_outputs,
    function_output,
)
from ergane.prompts import (
    COMPACT_PROMPT,
    SERVER_NOTES,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_NO_TOOLS,
)
from ergane.responses import NO_TEXT, Responder, response_problem
from ergane.todos import TodoList
from ergane.tools import Tool, ToolOutput

# The reason in the note that closes a turn the user stopped.
INTERRUPTED = "interrupted by the user"


# Calls to read-only tools that run at the same time, at most.
MAX_PARALLEL_CALLS = 8

# How long to wait, after Ctrl-C, for calls that were cancelled to end.
_CANCEL_SECONDS = 2

# How long to wait for the server's list of models, which is asked for once, without
# retries: the user is waiting for it, perhaps only for a name to be completed.
_MODELS_SECONDS = 5.0


class _Running:
    """A tool call running on a thread of its own, which ends when the call does.

    The thread is a daemon: a call that cannot be stopped (a plain function) must not
    keep ergane from exiting.
    """

    def __init__(
        self, tool: Tool, arguments: str, slots: threading.BoundedSemaphore
    ) -> None:
        self.tool = tool
        self.result: ToolOutput | None = None
        self.thread = threading.Thread(
            target=self._run,
            args=(arguments, slots),
            name=f"call-{tool.name}",
            daemon=True,
        )
        self.thread.start()

    def cancel(self) -> bool:
        """Ask the tool to stop this call, if it is still running and the tool can;
        whether it was asked to."""
        if self.thread.is_alive() and self.tool.cancel:
            self.tool.cancel()
            return True
        return False

    def _run(self, arguments: str, slots: threading.BoundedSemaphore) -> None:
        with slots:
            self.result = self.tool.call_output(arguments)


def _output_item(
    call: ResponseFunctionToolCall, result: ToolOutput | None, pictures: bool
) -> ResponseInputItemParam:
    """The item that answers a call: the output of a tool, or, for a call whose thread
    ended without one, a note that says so. A model that takes no `pictures` gets a
    note in their place instead of a request that would be refused."""
    result = result or ToolOutput("Error: the call ended without a result.")
    if result.images and not pictures:
        result = ToolOutput(f"{result.text}\n{IMAGE_REFUSED}")
    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": function_output(result.text, result.images),
    }


@dataclass(frozen=True)
class _Callbacks:
    """Where a turn reports to: the callbacks `run_turn` was given, together, since
    each round passes all of them down."""

    on_text: Callable[[str], None]
    on_notice: Callable[[str], None]
    on_call: Callable[[str, str], None]
    confirm: Callable[[Tool, str], bool | Literal["always"]]
    on_todos: Callable[[str], None]


@dataclass
class _Turn:
    """The turn that is running: what it has sent and done so far."""

    prompt: str
    items: ResponseInputParam  # the request's input, which grows with each round
    rounds: int = 0  # rounds that got a response and ran its tools
    refused: bool = False  # the server refused a request as too long, and we tried
    emptied: bool = False  # a response had nothing in it, and we asked again
    steps: list[str] = field(default_factory=list)  # signatures of the tool steps
    loop_note: str = ""  # what the instructions say while the model repeats itself


class Agent:
    """Sends conversation turns to the model. Does no terminal I/O of its own."""

    def __init__(
        self,
        config: Config,
        client: OpenAI | None = None,
        tools: Sequence[Tool] = (),
        server_notes: Mapping[str, str] | None = None,
    ) -> None:
        """Create an agent with an empty conversation.

        Args:
            config: The settings, which the agent reads each where it is used, so a
                change (`/model`) takes effect at once.
            client: The API client; built from the config if omitted.
            tools: The tools the model may call. The agent only ever uses the tools
                it is given here, and `update_server` changes them.
            server_notes: What MCP servers say about using their tools, by server.
                They go into the instructions, after the system prompt.

        Raises:
            ValueError: If two tools have the same name.
        """
        self.config = config
        self.client = client or OpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
        )
        self.responder = Responder(self.client, config)
        self._slots = threading.BoundedSemaphore(MAX_PARALLEL_CALLS)
        self.context = Context()
        # Tools the user allowed for the rest of the session, by name.
        self.always_allowed: set[str] = set()
        # The models that refused a request for its pictures: later pictures are not sent
        # to them, since each would cost a request that is refused. By model, since
        # `/model` can change it.
        self.no_pictures: set[str] = set()
        # The server's models as last asked for; None until then.
        self._models: list[str] | None = None
        # What the server counted in the last request it answered; None until then,
        # and when a compatible server sends no usage.
        self.last_input_tokens: int | None = None
        self.compactor = Compactor(self.context, config, self._summarize)
        self.tools: dict[str, Tool] = {}
        self.todos = TodoList()
        self._todo_tool = self.todos.tool()
        if clashes := self.add_tools(tools):
            raise ValueError(f"Tool names must be unique: {', '.join(clashes)}")
        self.server_notes = dict(server_notes or {})

    def run_turn(
        self,
        prompt: str,
        on_text: Callable[[str], None] = lambda text: None,
        on_notice: Callable[[str], None] = lambda text: None,
        confirm: Callable[
            [Tool, str], bool | Literal["always"]
        ] = lambda tool, arguments: False,
        on_call: Callable[[str, str], None] = lambda name, arguments: None,
        on_todos: Callable[[str], None] = lambda report: None,
    ) -> None:
        """Send one user message, passing the reply text to on_text.

        If the model calls tools, each call is run and answered, and the model is
        called again until it gives a reply. The whole turn
        (the calls and their outputs included) is added to the conversation if the
        final response completes with text, or ends incomplete after some text,
        which the user has seen, or if the loop guard stops a model that repeats
        itself, or if a round after the first fails or the user presses Ctrl-C
        (closed with a note saying why). Ctrl-C during the first model call
        saves nothing.

        Args:
            prompt: The user's input prompt.
            on_text: Called with each piece of reply text as it arrives when
                streaming, or once with the whole reply otherwise.
            on_notice: Called with status messages, such as old turns being
                dropped to fit the token budget.
            confirm: Called with a tool that is not read-only and the model's
                arguments for it, before it runs; it returns whether to allow
                the call (or "always", which allows it and every later call to that
                tool in this session). It is also what shows the user the call, so
                there is no other line for it. The default denies every such call.
            on_call: Called with the name and the arguments (JSON text) of each
                tool call that runs without asking, before it runs, to show it.
                A call that is asked about is shown by `confirm`, and a result is
                not shown at all: it goes to the model, and the reply says what it
                found. The `todo` tool's call is not shown either: `on_todos` is.
            on_todos: Called with `TodoList.report()` after a round that changed the
                model's checklist, for the user to follow it.

        Raises:
            TurnError: If the response is incomplete, failed, refused or has no
                text, if a stream ends without a final response, if the model
                keeps repeating the same tool calls, or if the tool results grow the
                turn past the context budget.
            Exception: Whatever the API client raises, such as an
                `openai.APIError`. Once a round has completed, the turn is kept
                (see above) before any of these is raised, except the budget one.
            KeyboardInterrupt: On Ctrl-C, after the turn is kept (see above).
        """
        io = _Callbacks(on_text, on_notice, on_call, confirm, on_todos)
        self.compactor.compact_if_due(
            self.request_tokens()
            + estimate_tokens({"role": "user", "content": prompt}),
            on_notice,
        )
        # The budget is for the whole request, so the history gets what is left of it
        # after the instructions and the tool definitions, which every request has.
        items, trimmed = self.context.begin_turn(prompt, self._room())
        if trimmed:
            on_notice("Context trimmed: oldest turns dropped")
        turn = _Turn(prompt, items)

        while True:
            response = self._ask(turn, io)
            problem = response_problem(response)
            calls = [item for item in response.output if item.type == "function_call"]
            if problem or not calls:
                break
            self._run_round(turn, response, calls, io)

        # Keep the turn if it worked, or was cut off after text the user has already
        # seen. A turn whose first response has a problem is dropped: saving a prompt
        # with no reply would leave two user messages in a row. After a completed
        # round it is kept with a note, like the stop of the loop guard.
        cut_off = response.status == "incomplete"
        if response.output_text and (problem is None or cut_off):
            reply: ResponseInputParam = [
                {"role": "assistant", "content": response.output_text}
            ]
            # The server's count of this request and the reply, which the next
            # request starts with (see Context.counted).
            counted = (
                self._next_request_tokens(
                    response, [*turn.items, *reply], len(turn.items)
                )
                if response.usage is not None
                else None
            )
            self.context.save_turn([*turn.items, *reply], counted, self.config.model)
        elif problem and turn.rounds:
            self._keep_stopped_turn(turn.items, turn.rounds, problem)
        if problem:
            raise TurnError(problem)

    def _room(self) -> int:
        """What is left of the budget for the history: the whole request is counted,
        and the instructions and the tool definitions are in every one."""
        return self.config.max_context_tokens - self._fixed_tokens()

    def _ask(self, turn: _Turn, io: _Callbacks) -> Response:
        """Send the turn's items and get the model's response.

        A request the server refuses as too long is shortened, once in a turn, one it
        refuses because the model takes no pictures is sent again without them, and a
        response with nothing in it is asked for again, once in a turn. Each happens a
        bounded number of times, so this ends. Any other failure ends the turn, and once a round has completed the turn is
        kept first: every call has its output at this point, so what the earlier
        rounds did (it may have had effects) is known.

        Raises:
            Exception: Whatever `Responder.send` raises.
            KeyboardInterrupt: On Ctrl-C.
        """
        while True:
            try:
                response = self.responder.send(
                    turn.items,
                    self._instructions(turn.loop_note),
                    io.on_text,
                    io.on_notice,
                    tools=self._tool_params(),
                )
            except (Exception, KeyboardInterrupt) as exc:
                # Only if there was a picture to take out: then the same request
                # cannot be refused for it again, and this does not go on for ever.
                if is_images_rejected(exc) and drop_images(turn.items):
                    self.no_pictures.add(self.config.model)
                    io.on_notice(
                        "The model does not take pictures, so they were dropped "
                        "(an MCP server's: set its images to false)"
                    )
                    continue
                if is_context_overflow(exc) and not turn.refused:
                    turn.refused = True
                    if self._shrink(turn, io.on_notice):
                        continue
                if turn.rounds:
                    if isinstance(exc, KeyboardInterrupt):
                        reason = INTERRUPTED
                    elif isinstance(exc, TurnError):
                        reason = str(exc)
                    else:
                        reason = type(exc).__name__
                    self._keep_stopped_turn(turn.items, turn.rounds, reason)
                raise
            if response.usage is not None:
                self.last_input_tokens = response.usage.input_tokens
            if response_problem(response) == NO_TEXT and not turn.emptied:
                turn.emptied = True
                io.on_notice("The model sent an empty reply; trying again")
                continue
            return response

    def _run_round(
        self,
        turn: _Turn,
        response: Response,
        calls: Sequence[ResponseFunctionToolCall],
        io: _Callbacks,
    ) -> None:
        """Add a response that calls tools to the turn, run the calls, and make sure
        the next request will fit.

        Raises:
            TurnError: If the turn is over the budget with nothing older to shorten.
            KeyboardInterrupt: On Ctrl-C, after the turn is kept.
        """
        turn.rounds += 1
        sent = len(turn.items)  # what the response answered
        listed = self.todos.revision
        try:
            turn.items += cast(
                ResponseInputParam,
                [item.model_dump(exclude_none=True) for item in response.output],
            )
            self._run_tools(calls, turn.items, io)
            if self.todos.revision != listed:
                io.on_todos(self.todos.report())
            self._watch_loop(turn, calls, io)
        except KeyboardInterrupt:
            # A call that was running, or had not run yet, has no output: say so, or
            # the API rejects the history. The call may have done part of its work,
            # and the model should know the user stopped it.
            answer_open_calls(turn.items)
            self._keep_stopped_turn(turn.items, turn.rounds, INTERRUPTED)
            raise
        budget = self.config.max_context_tokens
        used = self._next_request_tokens(response, turn.items, sent)
        # Results of earlier rounds go first, so the work goes on; the latest ones are
        # what the model needs to read next, so they are only shortened, keeping their
        # start and end.
        for upto, placeholders, notice in (
            (sent, True, "Context full: older tool results dropped"),
            (
                len(turn.items),
                False,
                "Context full: this round's tool results shortened",
            ),
        ):
            if used <= budget:
                break
            freed = drop_outputs(
                turn.items, upto, used - budget, placeholders=placeholders
            )
            if freed:
                io.on_notice(notice)
            used -= freed
        if used > budget:
            # Still over. The calls have run and may have had effects: keep the turn,
            # with its results dropped, or every later request would carry them.
            drop_outputs(turn.items, len(turn.items), used)
            self._keep_stopped_turn(
                turn.items, turn.rounds, "its results passed the context budget"
            )
            raise TurnError(
                f"This turn grew past the context budget (about {used} of "
                f"{budget} tokens). Ask for less, or raise max_context_tokens"
            )

    def _watch_loop(
        self,
        turn: _Turn,
        calls: Sequence[ResponseFunctionToolCall],
        io: _Callbacks,
    ) -> None:
        """Record the round just run; tell a model that repeats its steps (the next
        requests carry a notice, and the user sees one line), and stop one that goes
        on.

        Raises:
            TurnError: If one step fills too much of the recent ones. The turn is kept
                with a closing note first: every call has its output.
        """
        outputs = as_dicts(turn.items)[-len(calls) :]
        turn.steps.append(
            step_signature(
                [
                    (call.name, call.arguments, str(output["output"]))
                    for call, output in zip(calls, outputs, strict=True)
                ]
            )
        )
        if is_looping(turn.steps):
            self._keep_stopped_turn(
                turn.items, turn.rounds, "it kept repeating the same tool calls"
            )
            raise TurnError(
                "Stopped: the model keeps repeating the same tool calls (more than "
                f"{MAX_REPEATS} of its last {WINDOW} steps)"
            )
        repeats = last_repeats(turn.steps)
        if repeats > WARN_REPEATS:
            if not turn.loop_note:
                io.on_notice(
                    "The model is repeating itself; told to try something else"
                )
            turn.loop_note = loop_note(repeats)
        else:
            turn.loop_note = ""

    def _keep_stopped_turn(
        self, items: ResponseInputParam, rounds: int, reason: str
    ) -> None:
        """Save a turn that failed after `rounds` completed rounds, closed with a
        note, so the model still knows the work already done."""
        self._save_with_note(items, f"Stopped after {rounds} model calls: {reason}")

    def _save_with_note(self, items: ResponseInputParam, message: str) -> None:
        """Save a turn that has no final answer, closed with an assistant note.
        Every call in `items` must have its output."""
        self.context.save_turn(
            [*items, {"role": "assistant", "content": f"[{message}]"}]
        )

    def compact(
        self, on_notice: Callable[[str], None], focus: str | None = None
    ) -> bool:
        """Replace the oldest turns by a summary (see `Compactor.compact`)."""
        return self.compactor.compact(on_notice, focus)

    def _summarize(
        self, draft: ResponseInputParam, on_notice: Callable[[str], None]
    ) -> str:
        """Ask the model for the text of a summary.

        Raises:
            TurnError: If the response has a problem (see `response_problem`).
            Exception: Whatever the API client raises.
        """
        response = self.responder.send(
            draft,
            COMPACT_PROMPT,
            lambda text: None,
            on_notice,
            max_output_tokens=SUMMARY_MAX_TOKENS,
            low_effort=True,
        )
        if problem := response_problem(response):
            raise TurnError(problem)
        return response.output_text

    def _shrink(self, turn: _Turn, on_notice: Callable[[str], None]) -> bool:
        """Make the turn's request shorter after the server refused it as too long.

        Before the first round that means summarizing, whatever the trigger says; in
        a later round, shortening and dropping the results of the rounds before it.

        Returns:
            Whether the turn's items were made shorter, to be sent again.
        """
        if turn.rounds == 0:
            try:
                if not self.compact(on_notice):
                    return False
            except Exception:  # noqa: BLE001
                return False
            turn.items, _ = self.context.begin_turn(turn.prompt, self._room())
        else:
            results = sum(
                estimate_tokens(item)
                for item in as_dicts(turn.items)
                if item.get("type") == "function_call_output"
            )
            # Half of them: shortened first, dropped only if that is not enough.
            if not drop_outputs(turn.items, len(turn.items), results // 2 + 1):
                return False
        on_notice("The request was too long for the model; trying again, shorter")
        return True

    def models(self, *, refresh: bool = False) -> list[str]:
        """The names of the models the server offers, sorted, which it is asked for
        once and after that only with `refresh`. A name that could not be typed after
        `/model` (one with a space or a control character) is left out. A server that
        could not be asked is not asked again without `refresh`.

        Raises:
            Exception: Whatever the API client raises, such as an `openai.APIError`
                from a server that has no list.
        """
        if refresh or self._models is None:
            client = self.client.with_options(timeout=_MODELS_SECONDS, max_retries=0)
            try:
                listed = list(client.models.list())
            except Exception:
                self._models = []
                raise
            names = (getattr(model, "id", None) for model in listed)
            self._models = sorted(
                {
                    name
                    for name in names
                    if isinstance(name, str)
                    and name.isprintable()
                    and name.split() == [name]
                }
            )
        return self._models

    def add_tools(self, tools: Iterable[Tool]) -> list[str]:
        """Offer more tools. When a name is taken, the tool that has it stays: the
        first of a name wins, here and everywhere tools are added.

        Returns:
            The names of the tools that were not added because the name was taken.
        """
        skipped = []
        for tool in tools:
            if tool.name in self.tools:
                skipped.append(tool.name)
            else:
                self.tools[tool.name] = tool
        return skipped

    def update_server(
        self,
        removed: Iterable[str],
        tools: Iterable[Tool],
        name: str,
        notes: str | None,
    ) -> list[str]:
        """Take away the tools a server offered, add the ones it offers now, and set
        what it says about using them.

        A tool of a name that another tool has already is not added: the one that was
        there stays, as at start.

        Call it between turns: the tools of a turn that is running must not change.
        What the server counted for the last request does not hold any more.

        Args:
            removed: The names of the tools the server offered before.
            tools: The tools it offers now.
            name: The server.
            notes: What it says about using them, or None.

        Returns:
            The names of the tools that were not added because the name was taken.
        """
        for tool_name in removed:
            self.tools.pop(tool_name, None)
        skipped = self.add_tools(tools)
        if notes:
            self.server_notes[name] = notes
        else:
            self.server_notes.pop(name, None)
        self.context.counted = None
        return skipped

    def offered(self) -> dict[str, Tool]:
        """The tools of a request: the ones the agent was given, and the todo tool
        with them. With no other tool there is none, so that a chat without tools
        stays a plain chat. A given tool named `todo` takes the place of ours."""
        if not self.tools:
            return {}
        return {"todo": self._todo_tool, **self.tools}

    def _tool_params(self) -> list[FunctionToolParam]:
        """The definitions of the tools, as the request carries them."""
        return [tool.param() for tool in self.offered().values()]

    def _instructions(self, loop_note: str = "") -> str:
        """What every request carries as `instructions`: the system prompt, what the
        servers say, then the environment, so that the part that rarely changes
        comes first and can be cached. The todo list, if there are tools to keep it
        with, and the notice to a model that repeats itself (`loop_note`, of the turn
        that is running) go last."""
        prompt = SYSTEM_PROMPT if self.tools else SYSTEM_PROMPT_NO_TOOLS
        if self.server_notes and self.tools:
            sections = "\n".join(
                f"## {name}\n{text}\n" for name, text in self.server_notes.items()
            )
            prompt = f"{prompt}{SERVER_NOTES}{sections}"
        return (
            f"{prompt}\n"
            "Environment:\n"
            f"- Operating system: {platform.platform()}\n"
            f"- Current date: {date.today().isoformat()}\n"
            f"{self.todos.notes() if self.tools else ''}"
            f"{loop_note}"
        )

    def _fixed_estimate(self) -> tuple[int, int]:
        """Estimate the tokens that every request carries whatever the conversation:
        the instructions, and the definitions of the tools."""
        tools = self._tool_params()
        return (
            estimate_tokens({"instructions": self._instructions()}),
            estimate_tokens({"tools": tools}) if tools else 0,
        )

    def _fixed_tokens(self) -> int:
        """The instructions and the definitions of the tools, together."""
        return sum(self._fixed_estimate())

    def counted_tokens(self) -> int | None:
        """What the server counted for the instructions, the tools and the history at
        the end of the last turn, if that still holds: the history is as it was, and
        the model is the one that counted (another may count differently)."""
        context = self.context
        if context.counted is None or context.counted_model != self.config.model:
            return None
        return context.counted

    def reset(self) -> None:
        """Forget the whole conversation, the summary too."""
        self.context.reset()
        self.todos.clear()

    def restore(self, items: Sequence[Mapping[str, Any]], summary: str | None) -> None:
        """Replace the whole conversation by a saved one (see `Context.restore`)."""
        self.context.restore(cast(ResponseInputParam, items), summary)
        self.todos.clear()  # the list is not saved with a session
        self.last_input_tokens = None  # what the server counted was another request

    def undo(self) -> bool:
        """Drop the last turn; whether there was one."""
        return self.context.undo()

    def rewind(self) -> str | None:
        """Drop the last turn if it was saved, and return its prompt (None if there
        has not been one), to be sent again."""
        return self.context.rewind()

    @property
    def summary(self) -> str | None:
        """The summary that stands for the earliest turns, if there is one."""
        return self.context.summary

    @property
    def has_conversation(self) -> bool:
        """Whether there is anything in the history or the summary to lose."""
        return bool(self.context.items or self.context.summary)

    def request_tokens(self) -> int:
        """The tokens the next request would have, before the new prompt: what the
        server counted at the end of the last turn while the history is as it was
        (the instructions and the tools are in that count), otherwise an estimate."""
        counted = self.counted_tokens()
        if counted is not None:
            return counted
        return self._fixed_tokens() + self.context.estimate()

    def context_estimate(self) -> tuple[int, int, int]:
        """Estimate the tokens of the instructions, the tool definitions and the
        history, which together are what the next request would carry."""
        return (*self._fixed_estimate(), self.context.estimate())

    def _run_tools(
        self,
        calls: Sequence[ResponseFunctionToolCall],
        items: ResponseInputParam,
        io: _Callbacks,
    ) -> None:
        """Run the model's tool calls, adding each output to `items` in the order of
        the calls.

        Every call gets an output, even a failed, unknown or denied one: the output
        is text for the model, which can read the error and correct its call. A
        tool that is not read-only runs only if `confirm` allows the call.

        Calls to read-only tools that follow one another run at the same time, each on
        a thread of its own: they change nothing, so they cannot depend on each other.
        Any other call waits for the ones before it and runs alone, so the user is
        asked about one call at a time and a call that writes sees what came before.

        Args:
            calls: The function calls in the model's response.
            items: The turn's items; the outputs are added at the end, in order. If
                Ctrl-C interrupts, the outputs made so far are already there.
            io: Its `on_call` is called with each call that runs without asking,
                before it is run, and its `confirm` is asked before each call to a
                tool that is not read-only. A call that is asked about is shown by
                `confirm`, and a result is not shown at all: it goes to the model,
                and the reply says what it found.
        """
        together: list[tuple[ResponseFunctionToolCall, _Running]] = []

        def settle() -> None:
            """Wait for the calls that run together, and add their outputs."""
            try:
                for _, running in together:
                    running.thread.join()
            except KeyboardInterrupt:
                # Stop what can be stopped, and keep what is done: the rest is
                # answered as interrupted by the caller.
                cancelled = [r for _, r in together if r.cancel()]
                for running in cancelled:
                    running.thread.join(_CANCEL_SECONDS)
                for call, running in together:
                    if not running.thread.is_alive():
                        items.append(
                            _output_item(call, running.result, self._pictures())
                        )
                raise
            for call, running in together:
                items.append(_output_item(call, running.result, self._pictures()))
            together.clear()

        offered = self.offered()
        for call in calls:
            tool = offered.get(call.name)
            problem = tool.check(call.arguments) if tool else None
            if tool is not None and problem is None and tool.read_only:
                if tool is not self._todo_tool:  # the list is shown by `on_todos`
                    io.on_call(call.name, call.arguments)
                together.append((call, _Running(tool, call.arguments, self._slots)))
                continue
            settle()  # what came before is answered first
            items.append(
                _output_item(
                    call, self._answer(call, tool, problem, io), self._pictures()
                )
            )
        settle()

    def _pictures(self) -> bool:
        """Whether the model may be sent pictures: it has not refused any."""
        return self.config.model not in self.no_pictures

    def _answer(
        self,
        call: ResponseFunctionToolCall,
        tool: Tool | None,
        problem: str | None,
        io: _Callbacks,
    ) -> ToolOutput:
        """The output of a call that runs alone: the error for a tool that does not
        exist or arguments that do not fit, the note of a call the user denied, or the
        result of the tool.

        Args:
            call: The model's call.
            tool: The tool it names, if there is one.
            problem: What is wrong with its arguments (`Tool.check`), if anything.
            io: Where to show the call, and whom to ask.
        """
        # `confirm` shows the call it asks about, so such a call gets no line here.
        asks = (
            tool is not None
            and problem is None
            and tool.name not in self.always_allowed
        )
        if not asks:
            io.on_call(call.name, call.arguments)
        if tool is None:
            known = ", ".join(self.offered()) or "none"
            return ToolOutput(f"Unknown tool: {call.name}. Available tools: {known}.")
        if problem:
            # A call that would be rejected is not worth asking the user about.
            return ToolOutput(problem)
        answer = io.confirm(tool, call.arguments) if asks else True
        if answer == "always":
            self.always_allowed.add(tool.name)
        if not answer:
            return ToolOutput("The user denied this call.")
        return tool.call_output(call.arguments)

    @staticmethod
    def _next_request_tokens(
        response: Response, items: ResponseInputParam, sent: int
    ) -> int:
        """Estimate how many tokens the next request will have.

        The server's own count of the request it just answered is exact, so it is
        used when the response has one; the items added since are estimated. Some
        compatible servers send no usage, and then everything is estimated.

        Args:
            response: The response to the last request.
            items: The turn's items, including those added since that request.
            sent: How many of the items were in that request.

        Returns:
            The estimated number of tokens.
        """
        if response.usage is None:
            return sum(estimate_tokens(item) for item in items)
        return response.usage.input_tokens + sum(
            estimate_tokens(item) for item in items[sent:]
        )
