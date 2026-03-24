from __future__ import annotations

from agentic_stack.mcp.runtime_client import BuiltinMcpRuntimeClient
from agentic_stack.tools.base.types import (
    BoundRuntimeRequirements,
    ResolvedProfiledBuiltinTool,
)


def bind_runtime_requirements(
    *,
    resolved_tool: ResolvedProfiledBuiltinTool,
    builtin_mcp_runtime_client: BuiltinMcpRuntimeClient | None,
) -> BoundRuntimeRequirements:
    builtin_mcp_server_labels = tuple(
        sorted(
            {
                requirement.key
                for requirement in resolved_tool.runtime_requirements
                if requirement.kind == "builtin_mcp_server"
            }
        )
    )
    if builtin_mcp_server_labels and builtin_mcp_runtime_client is None:
        raise ValueError(
            "Built-in MCP runtime is required for "
            f"{resolved_tool.tool_type!r} profile {resolved_tool.profile_id!r}."
        )
    return BoundRuntimeRequirements(
        builtin_mcp_runtime_client=(
            builtin_mcp_runtime_client if builtin_mcp_server_labels else None
        ),
        builtin_mcp_server_labels=builtin_mcp_server_labels,
    )
