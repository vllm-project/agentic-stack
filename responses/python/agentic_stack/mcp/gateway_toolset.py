from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema import exceptions as jsonschema_exceptions
from pydantic import TypeAdapter
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from agentic_stack.mcp.types import McpExecutionResult, McpToolRef
from agentic_stack.mcp.utils import (
    build_mcp_tool_result_payload,
    canonicalize_output_text,
    is_mcp_tool_keyerror,
    redact_and_truncate_error_text,
)

_DICT_ARGS_VALIDATOR = TypeAdapter(dict[str, Any]).validator


@dataclass(frozen=True, slots=True)
class ResolvedMcpTool:
    internal_name: str
    ref: McpToolRef
    mcp_toolset: AbstractToolset[Any]
    mcp_tool_name: str
    description: str
    input_schema: dict[str, object]
    schema_validator: Draft202012Validator
    secret_values: tuple[str, ...] = ()
    mcp_tool: ToolsetTool[Any] | None = None


class McpGatewayToolset(AbstractToolset[Any]):
    def __init__(
        self,
        *,
        tools: list[ResolvedMcpTool],
        id: str | None = None,
    ) -> None:
        self._tools_by_name = {
            tool.internal_name: tool for tool in sorted(tools, key=lambda t: t.internal_name)
        }
        self._mcp_tool_cache: dict[str, ToolsetTool[Any]] = {
            tool.internal_name: tool.mcp_tool for tool in tools if tool.mcp_tool is not None
        }
        self._toolset_tools = {
            internal_name: ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name=internal_name,
                    description=tool.description,
                    parameters_json_schema=tool.input_schema,
                ),
                max_retries=0,
                args_validator=_DICT_ARGS_VALIDATOR,
            )
            for internal_name, tool in self._tools_by_name.items()
        }
        self._id = id

    @property
    def id(self) -> str | None:
        return self._id

    async def get_tools(self, ctx) -> dict[str, ToolsetTool[Any]]:
        _ = ctx
        return dict(self._toolset_tools)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx,
        tool: ToolsetTool[Any],
    ) -> Any:
        _ = tool
        resolved = self._tools_by_name.get(name)
        if resolved is None:  # pragma: no cover - defensive
            raise RuntimeError("MCP tool resolution failed for internal tool name.")

        validation_error = _validate_mcp_tool_arguments(
            validator=resolved.schema_validator,
            arguments=tool_args,
        )
        if validation_error is not None:
            return build_mcp_tool_result_payload(
                ref=resolved.ref,
                result=McpExecutionResult(
                    ok=False,
                    output_text=None,
                    error_text=validation_error,
                ),
            )

        result: McpExecutionResult
        try:
            mcp_tool = self._mcp_tool_cache.get(name)

            if mcp_tool is None:
                mcp_tool = await self._refresh_mcp_tool(name=name, resolved=resolved)
                if mcp_tool is None:
                    return build_mcp_tool_result_payload(
                        ref=resolved.ref,
                        result=McpExecutionResult(
                            ok=False,
                            output_text=None,
                            error_text=_tool_unavailable_error_text(resolved),
                        ),
                    )

            try:
                raw = await resolved.mcp_toolset.call_tool(
                    resolved.mcp_tool_name,
                    dict(tool_args),
                    ctx=ctx,
                    tool=mcp_tool,
                )
            except Exception as exc:
                if not is_mcp_tool_keyerror(exc, resolved.mcp_tool_name):
                    raise
                mcp_tool = await self._refresh_mcp_tool(name=name, resolved=resolved)
                if mcp_tool is None:
                    return build_mcp_tool_result_payload(
                        ref=resolved.ref,
                        result=McpExecutionResult(
                            ok=False,
                            output_text=None,
                            error_text=_tool_unavailable_error_text(resolved),
                        ),
                    )
                raw = await resolved.mcp_toolset.call_tool(
                    resolved.mcp_tool_name,
                    dict(tool_args),
                    ctx=ctx,
                    tool=mcp_tool,
                )

            result = McpExecutionResult(
                ok=True,
                output_text=canonicalize_output_text(raw),
                error_text=None,
            )
        except Exception as exc:
            result = McpExecutionResult(
                ok=False,
                output_text=None,
                error_text=redact_and_truncate_error_text(
                    text=str(exc).strip() or exc.__class__.__name__,
                    secret_values=resolved.secret_values,
                ),
            )

        return build_mcp_tool_result_payload(ref=resolved.ref, result=result)

    async def _refresh_mcp_tool(
        self,
        *,
        name: str,
        resolved: ResolvedMcpTool,
    ) -> ToolsetTool[Any] | None:
        refreshed = await resolved.mcp_toolset.get_tools(ctx=None)
        mcp_tool = refreshed.get(resolved.mcp_tool_name)
        if mcp_tool is None:
            self._mcp_tool_cache.pop(name, None)
            return None
        self._mcp_tool_cache[name] = mcp_tool
        return mcp_tool


def _validate_mcp_tool_arguments(
    *,
    validator: Draft202012Validator,
    arguments: dict[str, Any],
) -> str | None:
    try:
        validator.validate(arguments)
        return None
    except jsonschema_exceptions.ValidationError as exc:
        path = ".".join(str(part) for part in exc.path)
        if path:
            return f"input_validation_error: {exc.message} (path={path})"
        return f"input_validation_error: {exc.message}"


def _tool_unavailable_error_text(resolved: ResolvedMcpTool) -> str:
    return (
        f"MCP tool {resolved.ref.tool_name!r} is not available for "
        f"server {resolved.ref.server_label!r}."
    )
