from ergane import builtin
from ergane.tools import tools_in


def test_every_built_in_has_a_description_and_every_parameter_too():
    # The model chooses a tool and fills its arguments from these words alone.
    for tool in tools_in(builtin):
        assert tool.description.strip(), tool.name
        for name, schema in tool.parameters["properties"].items():
            assert schema.get("description"), f"{tool.name}.{name}"


def test_the_built_ins_that_only_read_run_without_asking():
    by_name = {tool.name: tool for tool in tools_in(builtin)}

    assert by_name["read_file"].read_only


def test_the_web_tools_say_the_same_thing_about_themselves():
    from ergane.builtin import fetch, search

    assert fetch.USER_AGENT is search.USER_AGENT
    assert fetch.USER_AGENT.startswith("ergane/")


def test_parameter_names_are_snake_case():
    for tool in tools_in(builtin):
        for name in tool.parameters["properties"]:
            assert name == name.lower() and " " not in name, f"{tool.name}.{name}"


def test_the_built_ins_that_change_files_ask_first_and_can_show_the_change():
    by_name = {tool.name: tool for tool in tools_in(builtin)}

    for name in ("write_file", "edit_file"):
        assert not by_name[name].read_only, name
        assert by_name[name].preview is not None, name


def test_the_built_ins_that_run_commands_or_go_out_ask_first():
    by_name = {tool.name: tool for tool in tools_in(builtin)}

    for name in ("run_command", "web_fetch", "web_search"):
        assert not by_name[name].read_only, name


def test_the_tools_that_name_a_file_say_it_the_same_way():
    by_name = {tool.name: tool for tool in tools_in(builtin)}

    def described(tool: str) -> str:
        return by_name[tool].parameters["properties"]["path"]["description"]

    assert described("read_file") == described("write_file") == described("edit_file")
