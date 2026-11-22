"""The model's todo list: a tool to keep a checklist of a multi-step task.

The tool tracks nothing the model could not track itself: it gives the model a place to
write the steps down, makes it say which one it is on, and shows the user. As in codex's
`update_plan` and opencode's `todowrite`, every call sends the whole list and replaces
the old one, so there are no ids to keep in step. Unlike them, a list with two
`in_progress` items is refused, which is the one rule that stops "too many tasks at
once"; the error says how to fix the call.

The list lives in memory, in `TodoList`, which the agent owns. The history keeps old
tool calls only for the newest turn, so the agent puts a list that still has open items
into the instructions of each request (`TodoList.notes`), where it survives dropped
tool results and summaries. It is not saved with a session.
"""

import threading
from typing import Literal

from pydantic import BaseModel

from ergane.tools import Tool

Status = Literal["pending", "in_progress", "completed", "cancelled"]

_MARKS: dict[str, str] = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "completed": "[x]",
    "cancelled": "[-]",
}


class TodoItem(BaseModel):
    """One step of the list."""

    content: str
    status: Status


class TodoList:
    """The list, and the tool that changes it."""

    def __init__(self) -> None:
        self.items: list[TodoItem] = []
        # Counts the updates, so that whoever shows the list knows it changed.
        self.revision = 0
        # Read-only calls run on threads, and one response may hold two of these.
        self._lock = threading.Lock()

    def tool(self) -> Tool:
        """The tool the model calls to replace the list."""
        return Tool.from_function(self.todo)

    def todo(self, todos: list[TodoItem]) -> str:
        """Keep a checklist for a task with three or more steps, or when the user gives
        you several tasks. Skip it for a simple task or a plain question.

        Send the whole list on every call; it replaces the previous one, and an empty
        list clears it. Mark a step `in_progress` when you start it (only one at a
        time) and `completed` as soon as it is done; use `cancelled` for a step you
        drop. Call it only when the list changes. The user sees the list, so do not
        repeat it in your reply.
        """
        if any(not item.content.strip() for item in todos):
            raise ValueError("every todo needs a non-empty content")
        active = [item.content for item in todos if item.status == "in_progress"]
        if len(active) > 1:
            raise ValueError(
                f"only one todo can be in_progress at a time, but {len(active)} are: "
                "finish one, or set the others back to pending, then send the list again"
            )
        with self._lock:
            self.items = list(todos)
            self.revision += 1
        return self.summary()

    def clear(self) -> None:
        """Forget the list, as `/reset` forgets the conversation."""
        with self._lock:
            if self.items:
                self.items = []
                self.revision += 1

    def summary(self) -> str:
        """The list in one line: how much is done, and what is being worked on."""
        items = self.items
        if not items:
            return "Todos cleared"
        parts = [
            f"{sum(i.status == 'completed' for i in items)}/{len(items)} completed"
        ]
        for status in ("in_progress", "pending", "cancelled"):
            if count := sum(i.status == status for i in items):
                parts.append(f"{count} {status.replace('_', ' ')}")
        text = f"Todos updated: {', '.join(parts)}"
        for item in items:
            if item.status == "in_progress":
                text += f". Now: {item.content}"
        return text

    def render(self) -> str:
        """The list as a checklist, one step to a line."""
        return "\n".join(f"{_MARKS[i.status]} {i.content}" for i in self.items)

    @property
    def open(self) -> bool:
        """Whether any step is still to do."""
        return any(i.status in ("pending", "in_progress") for i in self.items)

    def notes(self) -> str:
        """What goes into each request's instructions while steps are open, or ''."""
        if not self.open:
            return ""
        return (
            "\nYour todo list so far (your own notes; keep it up to date with the "
            f"todo tool):\n{self.render()}\n"
        )
