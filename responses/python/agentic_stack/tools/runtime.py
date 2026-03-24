from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentic_stack.configs.runtime import RuntimeConfig

if TYPE_CHECKING:
    from agentic_stack.tools.web_search.runtime import WebSearchToolRuntime


@dataclass(frozen=True, slots=True)
class ToolRuntimeContext:
    runtime_config: RuntimeConfig
    web_search: WebSearchToolRuntime | None = None


_REQUEST_TOOL_RUNTIME_CONTEXT: ContextVar[ToolRuntimeContext | None] = ContextVar(
    "tool_runtime_context",
    default=None,
)


@contextmanager
def bind_tool_runtime_context(tool_runtime_context: ToolRuntimeContext):
    token: Token[ToolRuntimeContext | None] = _REQUEST_TOOL_RUNTIME_CONTEXT.set(
        tool_runtime_context
    )
    try:
        yield
    finally:
        _REQUEST_TOOL_RUNTIME_CONTEXT.reset(token)


def get_tool_runtime_context() -> ToolRuntimeContext | None:
    return _REQUEST_TOOL_RUNTIME_CONTEXT.get()


def require_tool_runtime_context() -> ToolRuntimeContext:
    tool_runtime_context = get_tool_runtime_context()
    if tool_runtime_context is None:
        raise RuntimeError("Tool runtime context is not bound for this request.")
    return tool_runtime_context
