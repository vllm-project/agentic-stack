from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from agentic_stack.mcp.runtime_client import (
    BuiltinMcpRuntimeClient,
    BuiltinMcpRuntimeToolMissingError,
)

_DICT_ARGS_VALIDATOR = TypeAdapter(dict[str, Any]).validator


class BuiltinMcpRuntimeToolset(AbstractToolset[Any]):
    def __init__(
        self,
        *,
        server_label: str,
        runtime_client: BuiltinMcpRuntimeClient,
        id: str | None = None,
    ) -> None:
        self._server_label = server_label
        self._runtime_client = runtime_client
        self._id = id

    @property
    def id(self) -> str | None:
        return self._id

    async def get_tools(self, ctx) -> dict[str, ToolsetTool[Any]]:
        _ = ctx
        tool_infos = await self._runtime_client.list_tools(self._server_label)
        return {
            tool_info.name: ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name=tool_info.name,
                    description=tool_info.description,
                    parameters_json_schema=tool_info.input_schema,
                ),
                max_retries=0,
                args_validator=_DICT_ARGS_VALIDATOR,
            )
            for tool_info in tool_infos
        }

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx,
        tool: ToolsetTool[Any],
    ) -> Any:
        _ = ctx, tool
        try:
            result = await self._runtime_client.call_tool(
                server_label=self._server_label,
                tool_name=name,
                arguments=tool_args,
            )
        except BuiltinMcpRuntimeToolMissingError as exc:
            raise RuntimeError(
                f"MCP tool {name!r} is not available for server {self._server_label!r}."
            ) from exc
        if not result.ok:
            raise RuntimeError(result.error_text or "MCP tool execution failed.")
        return result.output_text or ""
