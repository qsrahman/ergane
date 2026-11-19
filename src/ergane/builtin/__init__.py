"""ergane's own tools: add a tool by writing a function in a module here.

Every `Tool` at the top level of a module of this package is offered to the model
(`tools_in`), so nothing else needs editing:

    # src/ergane/builtin/arithmetic.py
    from ergane.tools import tool

    @tool
    def add(a: int, b: int) -> str:
        \"\"\"Add two integers and return the sum.\"\"\"
        return str(a + b)

The docstring is the description the model reads, the type hints are its parameters, and
`@tool(read_only=False)` marks a tool that changes anything outside the program, which
then asks the user first. A module whose name starts with `_` is not looked at, for
helpers. Import a heavy dependency inside the function, not at the top of the module, so
that starting ergane does not pay for it.
"""
