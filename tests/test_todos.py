import json

import pytest

from ergane.prompts import SYSTEM_PROMPT_NO_TOOLS
from ergane.todos import TodoItem, TodoList
from ergane.tools import Tool
from helpers import run_repl, tool_with


def read_file(path: str) -> str:
    """Read a text file."""
    return "x"


def call(todos: TodoList, *items: tuple[str, str]) -> str:
    """Call the tool as the model would, with its JSON arguments."""
    arguments = json.dumps({"todos": [{"content": c, "status": s} for c, s in items]})
    return todos.tool().call(arguments)


class TestTodoTool:
    def test_the_list_is_replaced_whole(self):
        todos = TodoList()

        call(todos, ("a", "pending"), ("b", "pending"))
        result = call(todos, ("a", "in_progress"))

        assert todos.items == [TodoItem(content="a", status="in_progress")]
        assert result == "Todos updated: 0/1 completed, 1 in progress. Now: a"

    def test_the_result_counts_the_steps_and_names_the_active_one(self):
        result = call(
            TodoList(),
            ("a", "completed"),
            ("b", "in_progress"),
            ("c", "pending"),
            ("d", "cancelled"),
        )

        assert result == (
            "Todos updated: 1/4 completed, 1 in progress, 1 pending, 1 cancelled. "
            "Now: b"
        )

    def test_an_empty_list_clears_it(self):
        todos = TodoList()
        call(todos, ("a", "pending"))

        assert call(todos) == "Todos cleared"
        assert todos.items == []

    def test_two_steps_in_progress_are_refused_and_the_list_is_kept(self):
        todos = TodoList()
        call(todos, ("a", "pending"))

        result = call(todos, ("a", "in_progress"), ("b", "in_progress"))

        assert result.startswith("Error: ValueError: only one todo can be in_progress")
        assert [i.content for i in todos.items] == ["a"]
        assert todos.items[0].status == "pending"

    def test_an_empty_step_is_refused(self):
        assert "non-empty content" in call(TodoList(), ("  ", "pending"))

    def test_a_status_that_does_not_exist_is_refused_before_the_tool_runs(self):
        tool = TodoList().tool()
        arguments = '{"todos": [{"content": "a", "status": "done"}]}'

        assert (tool.check(arguments) or "").startswith("Invalid arguments")

    def test_it_runs_without_asking(self):
        assert TodoList().tool().read_only

    def test_every_update_is_counted(self):
        todos = TodoList()
        call(todos, ("a", "pending"))
        call(todos, ("a", "in_progress"), ("b", "in_progress"))  # refused

        assert todos.revision == 1

    def test_clear_counts_only_when_there_was_something_to_forget(self):
        todos = TodoList()
        todos.clear()
        assert todos.revision == 0

        call(todos, ("a", "pending"))
        todos.clear()
        assert todos.revision == 2
        assert todos.items == []


class TestTodoNotes:
    def test_a_list_with_open_steps_is_written_out(self):
        todos = TodoList()
        call(todos, ("a", "completed"), ("b", "in_progress"), ("c", "pending"))

        assert "[x] a\n[>] b\n[ ] c" in todos.notes()

    @pytest.mark.parametrize(
        "items",
        [[], [("a", "completed")], [("a", "completed"), ("b", "cancelled")]],
    )
    def test_a_list_with_nothing_left_to_do_says_nothing(self, items):
        todos = TodoList()
        call(todos, *items)

        assert todos.notes() == ""


class TestAgentWithTodos:
    def test_the_todo_tool_is_offered_with_other_tools(self, make_agent, fake_server):
        make_agent(tools=[tool_with(read_file)]).run_turn("hi")

        names = [t["name"] for t in fake_server.requests[0]["tools"]]
        assert names == ["todo", "read_file"]

    def test_a_chat_without_tools_stays_a_plain_chat(self, make_agent, fake_server):
        make_agent().run_turn("hi")

        request = fake_server.requests[0]
        assert not request.get("tools")
        assert request["instructions"].startswith(SYSTEM_PROMPT_NO_TOOLS)
        assert "todo" not in request["instructions"]

    def test_open_steps_are_in_the_next_requests_instructions(
        self, make_agent, fake_server
    ):
        agent = make_agent(tools=[tool_with(read_file)])
        call(agent.todos, ("write it", "in_progress"))

        agent.run_turn("hi")

        assert "[>] write it" in fake_server.requests[0]["instructions"]

    def test_finished_steps_are_not_repeated_in_the_instructions(
        self, make_agent, fake_server
    ):
        agent = make_agent(tools=[tool_with(read_file)])
        call(agent.todos, ("write it", "completed"))

        agent.run_turn("hi")

        assert "write it" not in fake_server.requests[0]["instructions"]

    def test_a_change_made_in_a_round_is_shown_once(self, make_agent):
        agent = make_agent(tools=[])

        def read_file(path: str) -> str:
            """Stand-in for a call of the todo tool: it changes the list."""
            return call(agent.todos, ("a", "in_progress"))

        agent.add_tools([tool_with(read_file)])
        notices: list[str] = []

        agent.run_turn("tool_call", on_notice=notices.append)

        assert notices == ["Todos updated: 0/1 completed, 1 in progress. Now: a"]

    def test_a_round_that_leaves_the_list_alone_shows_nothing(self, make_agent):
        agent = make_agent(tools=[tool_with(read_file)])
        notices: list[str] = []

        agent.run_turn("tool_call", on_notice=notices.append)

        assert notices == []

    def test_reset_and_restore_forget_the_list(self, make_agent):
        agent = make_agent(tools=[tool_with(read_file)])
        call(agent.todos, ("a", "pending"))
        agent.reset()
        assert agent.todos.items == []

        call(agent.todos, ("a", "pending"))
        agent.restore([], None)
        assert agent.todos.items == []

    def test_a_given_tool_named_todo_takes_the_place_of_ours(self, make_agent):
        def todo(text: str) -> str:
            """Their own."""
            return "mine"

        agent = make_agent(tools=[tool_with(read_file)])
        agent.add_tools([Tool.from_function(todo)])

        assert agent.offered()["todo"].description == "Their own."


class TestTodoCommand:
    def test_it_shows_the_list(self, monkeypatch, make_agent, capsys):
        agent = make_agent()
        call(agent.todos, ("a", "completed"), ("b", "pending"))

        run_repl(monkeypatch, agent, ["/todo", "/exit"])

        assert "[x] a\n[ ] b" in capsys.readouterr().out

    def test_it_says_when_there_is_none(self, monkeypatch, make_agent, capsys):
        run_repl(monkeypatch, make_agent(), ["/todo", "/exit"])

        assert "No todos" in capsys.readouterr().out
