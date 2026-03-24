from __future__ import annotations

from typing import Any, Callable, Mapping

from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

from agentic_stack.mcp.types import McpToolInfo

try:
    from fastmcp.client import Client
except ImportError as _fastmcp_import_error:  # pragma: no cover - import-time guard
    raise RuntimeError(
        "The `fastmcp` package is required for MCP execution."
    ) from _fastmcp_import_error


def build_fastmcp_toolset_from_server_entry(
    *,
    server_label: str,
    server_entry: Mapping[str, object],
    timeout_s: float | None = None,
    init_timeout_s: float | None = None,
) -> FastMCPToolset:
    mcp_config = {
        "mcpServers": {
            server_label: dict(server_entry),
        }
    }
    client = Client(
        transport=mcp_config,
        timeout=timeout_s,
        init_timeout=init_timeout_s,
    )
    return FastMCPToolset(
        client,
        max_retries=0,
        tool_error_behavior="error",
    )


def extract_mcp_tool_infos(
    tools: dict[str, ToolsetTool[Any]],
    *,
    schema_normalizer: Callable[[dict[str, object]], dict[str, object]] | None = None,
) -> dict[str, McpToolInfo]:
    out: dict[str, McpToolInfo] = {}
    for tool_name, tool in sorted(tools.items()):
        if not isinstance(tool_name, str) or not tool_name:
            continue
        tool_def = getattr(tool, "tool_def", None)
        description = getattr(tool_def, "description", None) if tool_def is not None else None
        schema_raw = (
            getattr(tool_def, "parameters_json_schema", {}) if tool_def is not None else {}
        )
        if not isinstance(schema_raw, dict):
            input_schema: dict[str, object] = {"type": "__invalid_schema__"}
        elif schema_normalizer is None:
            input_schema = schema_raw
        else:
            try:
                input_schema = schema_normalizer(dict(schema_raw))
            except ValueError:
                input_schema = {"type": "__invalid_schema__"}

        out[tool_name] = McpToolInfo(
            name=tool_name,
            description=description if isinstance(description, str) else None,
            input_schema=input_schema,
        )
    return out
