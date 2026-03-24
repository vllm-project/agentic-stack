from pydantic_ai import Tool

from agentic_stack.tools.ids import CODE_INTERPRETER_TOOL, WEB_SEARCH_TOOL

TOOLS: dict[str, Tool] = {}


def register(name_or_func=None):
    """
    Decorator to register a function as a tool.

    Can be used as:
    - @register - uses function name
    - @register("custom_name") - uses provided name
    """

    def decorator(func):
        # Determine the name to use
        tool_name = name_or_func if isinstance(name_or_func, str) else func.__name__
        # Wrap function in Tool and add to TOOLS
        TOOLS[tool_name] = Tool(func, takes_ctx=False, name=tool_name)
        return func

    # If called without parentheses (@register), name_or_func is the function
    if callable(name_or_func):
        func = name_or_func
        TOOLS[func.__name__] = Tool(func, takes_ctx=False, name=func.__name__)
        return func

    # If called with parentheses (@register() or @register("name"))
    return decorator


__all__ = [
    "CODE_INTERPRETER_TOOL",
    "TOOLS",
    "WEB_SEARCH_TOOL",
    "register",
]
