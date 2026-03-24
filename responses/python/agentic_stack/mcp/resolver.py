from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping

from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from agentic_stack.mcp.fastmcp_runtime import (
    build_fastmcp_toolset_from_server_entry,
    extract_mcp_tool_infos,
)
from agentic_stack.mcp.policy import (
    build_request_remote_headers,
    request_remote_secret_values,
    validate_request_remote_server_url,
)
from agentic_stack.mcp.runtime_client import (
    BuiltinMcpRuntimeClient,
    BuiltinMcpRuntimeTransportError,
    BuiltinMcpRuntimeUnavailableServerError,
    BuiltinMcpRuntimeUnknownServerError,
)
from agentic_stack.mcp.runtime_toolset import BuiltinMcpRuntimeToolset
from agentic_stack.mcp.types import McpMode, McpToolInfo, RequestRemoteMcpServerBinding
from agentic_stack.mcp.utils import redact_and_truncate_error_text
from agentic_stack.utils.exceptions import BadInputError

if TYPE_CHECKING:
    from agentic_stack.types.openai import OpenAIResponsesMcpTool


@dataclass(frozen=True, slots=True)
class ResolvedMcpServerTools:
    mode: McpMode
    mcp_toolset: AbstractToolset[Any]
    secret_values: tuple[str, ...]
    allowed_tool_infos: dict[str, McpToolInfo]
    allowed_mcp_tools_by_name: dict[str, ToolsetTool[Any]] = field(default_factory=dict)


BuildRequestRemoteToolset = Callable[[RequestRemoteMcpServerBinding], AbstractToolset[Any]]


@dataclass(frozen=True, slots=True)
class _ServerInventory:
    toolset: AbstractToolset[Any]
    tool_map: dict[str, McpToolInfo]
    secret_values: tuple[str, ...]
    mcp_tools_by_name: dict[str, ToolsetTool[Any]]


def build_request_remote_toolset(
    binding: RequestRemoteMcpServerBinding,
) -> AbstractToolset[Any]:
    return build_fastmcp_toolset_from_server_entry(
        server_label=binding.server_label,
        server_entry={
            "url": binding.server_url,
            "headers": dict(binding.headers),
        },
    )


async def resolve_mcp_declarations(
    *,
    declarations: Mapping[str, OpenAIResponsesMcpTool],
    builtin_mcp_runtime_client: BuiltinMcpRuntimeClient | None,
    request_remote_enabled: bool,
    request_remote_url_checks_enabled: bool,
    request_remote_toolset_builder: BuildRequestRemoteToolset = build_request_remote_toolset,
) -> dict[str, ResolvedMcpServerTools]:
    resolved_servers: dict[str, ResolvedMcpServerTools] = {}

    for server_label, declaration in declarations.items():
        if declaration.connector_id is not None:
            raise BadInputError(
                "MCP `connector_id` declarations are not supported by this gateway."
            )
        if declaration.require_approval not in {None, "never"}:
            raise BadInputError("MCP `require_approval` supports `never` only in this gateway.")

        if declaration.server_url is None:
            if declaration.authorization is not None:
                raise BadInputError(
                    "MCP `authorization` is only supported when `server_url` is provided."
                )
            if declaration.headers is not None:
                raise BadInputError(
                    "MCP request-declared `headers` are only supported when `server_url` is provided."
                )
            mode = "hosted"
            inventory = await _resolve_builtin_server(
                server_label=server_label,
                builtin_mcp_runtime_client=builtin_mcp_runtime_client,
            )
        else:
            mode = "request_remote"
            inventory = await _resolve_request_remote_server(
                server_label=server_label,
                declaration=declaration,
                request_remote_enabled=request_remote_enabled,
                request_remote_url_checks_enabled=request_remote_url_checks_enabled,
                request_remote_toolset_builder=request_remote_toolset_builder,
            )

        allowed_tool_infos = _select_allowed_tool_infos(
            runtime_tool_map=inventory.tool_map,
            allowed_tools=declaration.allowed_tools,
        )
        if not allowed_tool_infos:
            raise BadInputError(
                f"MCP server {server_label!r} has an empty final allowed tool set."
            )

        resolved_servers[server_label] = ResolvedMcpServerTools(
            mode=mode,
            mcp_toolset=inventory.toolset,
            secret_values=inventory.secret_values,
            allowed_tool_infos=allowed_tool_infos,
            allowed_mcp_tools_by_name=inventory.mcp_tools_by_name,
        )

    return resolved_servers


async def _resolve_builtin_server(
    *,
    server_label: str,
    builtin_mcp_runtime_client: BuiltinMcpRuntimeClient | None,
) -> _ServerInventory:
    if builtin_mcp_runtime_client is None or not builtin_mcp_runtime_client.is_enabled():
        raise BadInputError(
            "Built-in MCP tools were provided but the Built-in MCP runtime is disabled."
        )

    try:
        tool_infos = await builtin_mcp_runtime_client.list_tools(server_label)
    except BuiltinMcpRuntimeUnknownServerError as exc:
        raise BadInputError(f"Unknown MCP server_label: {server_label}") from exc
    except BuiltinMcpRuntimeUnavailableServerError as exc:
        raise BadInputError(str(exc)) from exc
    except BuiltinMcpRuntimeTransportError as exc:
        raise BadInputError(
            f"MCP tool inventory resolution failed for server {server_label!r}: {exc}"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise BadInputError(
            f"MCP tool inventory resolution failed for server {server_label!r}: {exc}"
        ) from exc

    runtime_tool_map = {tool.name: tool for tool in tool_infos}
    toolset = BuiltinMcpRuntimeToolset(
        server_label=server_label,
        runtime_client=builtin_mcp_runtime_client,
    )
    return _ServerInventory(
        toolset=toolset,
        tool_map=runtime_tool_map,
        secret_values=(),
        mcp_tools_by_name={},
    )


async def _resolve_request_remote_server(
    *,
    server_label: str,
    declaration: OpenAIResponsesMcpTool,
    request_remote_enabled: bool,
    request_remote_url_checks_enabled: bool,
    request_remote_toolset_builder: BuildRequestRemoteToolset,
) -> _ServerInventory:
    if not request_remote_enabled:
        raise BadInputError("Request-remote MCP is disabled by gateway configuration.")
    if declaration.server_url is None:  # pragma: no cover - defensive
        raise BadInputError("MCP `server_url` is required for request-remote declarations.")

    if request_remote_url_checks_enabled:
        validate_request_remote_server_url(declaration.server_url)
    resolved_headers = build_request_remote_headers(
        authorization=declaration.authorization,
        request_headers=declaration.headers,
    )
    binding = RequestRemoteMcpServerBinding(
        mode="request_remote",
        server_label=server_label,
        server_url=declaration.server_url,
        authorization=declaration.authorization,
        headers=resolved_headers,
    )
    secret_values = request_remote_secret_values(
        authorization=declaration.authorization,
        headers=resolved_headers,
    )
    try:
        mcp_toolset = request_remote_toolset_builder(binding)
        mcp_tools = await mcp_toolset.get_tools(ctx=None)
        runtime_tool_map = extract_mcp_tool_infos(mcp_tools)
    except BadInputError:
        raise
    except Exception as exc:
        error_text = redact_and_truncate_error_text(
            text=str(exc).strip() or exc.__class__.__name__,
            secret_values=secret_values,
        )
        raise BadInputError(
            f"MCP tool inventory resolution failed for server {server_label!r}: {error_text}"
        ) from exc

    return _ServerInventory(
        toolset=mcp_toolset,
        tool_map=runtime_tool_map,
        secret_values=secret_values,
        mcp_tools_by_name=dict(mcp_tools),
    )


def _select_allowed_tool_infos(
    *,
    runtime_tool_map: dict[str, McpToolInfo],
    allowed_tools: list[str] | None,
) -> dict[str, McpToolInfo]:
    if allowed_tools is None:
        return dict(runtime_tool_map)

    allowed_set = set(allowed_tools)
    return {
        tool_name: tool_info
        for tool_name, tool_info in runtime_tool_map.items()
        if tool_name in allowed_set
    }
