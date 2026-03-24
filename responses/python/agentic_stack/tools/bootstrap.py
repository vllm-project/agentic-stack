from __future__ import annotations

from agentic_stack.tools.code_interpreter import register_code_interpreter_tool
from agentic_stack.tools.web_search.tool import register_web_search_tool


def register_runtime_tool_handlers() -> None:
    register_code_interpreter_tool()
    register_web_search_tool()
