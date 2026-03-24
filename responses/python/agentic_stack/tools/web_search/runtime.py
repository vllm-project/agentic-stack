from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentic_stack.configs.runtime import RuntimeConfig
from agentic_stack.mcp.runtime_client import BuiltinMcpRuntimeClient
from agentic_stack.tools.base.runtime import bind_runtime_requirements
from agentic_stack.tools.base.types import BuiltinActionAdapter
from agentic_stack.tools.ids import WEB_SEARCH_TOOL
from agentic_stack.tools.profile_resolution import resolve_profiled_builtin_tool
from agentic_stack.tools.web_search.adapters import WEB_SEARCH_ADAPTER_SPECS
from agentic_stack.tools.web_search.config import resolve_request_config
from agentic_stack.tools.web_search.executor import WebSearchExecutor
from agentic_stack.tools.web_search.page_cache import WebSearchPageCache
from agentic_stack.utils.exceptions import BadInputError

if TYPE_CHECKING:
    from agentic_stack.types.openai import vLLMResponsesRequest


@dataclass(slots=True)
class WebSearchToolRuntime:
    executor: WebSearchExecutor


def build_web_search_tool_runtime(
    *,
    request: "vLLMResponsesRequest",
    enabled_builtin_tool_names: set[str],
    runtime_config: RuntimeConfig,
    builtin_mcp_runtime_client: BuiltinMcpRuntimeClient | None,
) -> WebSearchToolRuntime | None:
    if WEB_SEARCH_TOOL not in enabled_builtin_tool_names or not request.tools:
        return None
    from agentic_stack.types.openai import OpenAIResponsesWebSearchTool

    web_search_tools = [
        tool for tool in request.tools if isinstance(tool, OpenAIResponsesWebSearchTool)
    ]
    if not web_search_tools:
        return None
    if len(web_search_tools) > 1:
        raise BadInputError("Duplicate `web_search` tools are not allowed.")
    try:
        request_config = resolve_request_config(
            tool=web_search_tools[0],
            runtime_config=runtime_config,
        )
        resolved_tool = resolve_profiled_builtin_tool(
            tool_type=WEB_SEARCH_TOOL,
            profile_id=request_config.profile_id,
        )
        bound_requirements = bind_runtime_requirements(
            resolved_tool=resolved_tool,
            builtin_mcp_runtime_client=builtin_mcp_runtime_client,
        )
        adapter_by_action: dict[str, BuiltinActionAdapter] = {
            binding.action_name: WEB_SEARCH_ADAPTER_SPECS[binding.adapter_id].build_adapter()
            for binding in resolved_tool.action_bindings
        }
    except ValueError as exc:
        raise BadInputError(str(exc)) from exc
    return WebSearchToolRuntime(
        executor=WebSearchExecutor(
            request_config=request_config,
            resolved_tool=resolved_tool,
            bound_requirements=bound_requirements,
            adapter_by_action=adapter_by_action,
            page_cache=WebSearchPageCache(),
        )
    )
