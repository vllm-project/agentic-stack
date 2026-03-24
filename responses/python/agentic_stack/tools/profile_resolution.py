from __future__ import annotations

from agentic_stack.configs.sources import EnvSource
from agentic_stack.mcp.config import (
    McpRuntimeConfig,
    McpServerRuntimeConfig,
    split_hosted_server_entry,
)
from agentic_stack.tools.base.types import (
    ProfiledBuiltinProfileResolutionProvider,
    ResolvedProfiledBuiltinTool,
)
from agentic_stack.tools.ids import WEB_SEARCH_TOOL
from agentic_stack.tools.web_search.profiles import WEB_SEARCH_PROFILE_RESOLUTION_PROVIDER

PROFILE_RESOLUTION_PROVIDERS: dict[str, ProfiledBuiltinProfileResolutionProvider] = {
    WEB_SEARCH_TOOL: WEB_SEARCH_PROFILE_RESOLUTION_PROVIDER,
}


def _get_profile_resolution_provider(tool_type: str) -> ProfiledBuiltinProfileResolutionProvider:
    try:
        return PROFILE_RESOLUTION_PROVIDERS[tool_type]
    except KeyError as exc:
        raise ValueError(
            f"Built-in tool {tool_type!r} does not support profile resolution."
        ) from exc


def resolve_profiled_builtin_tool(
    *,
    tool_type: str,
    profile_id: str,
) -> ResolvedProfiledBuiltinTool:
    return _get_profile_resolution_provider(tool_type).resolve(profile_id)


def validate_profiled_builtin_profile(
    *,
    tool_type: str,
    profile_id: str | None = None,
) -> None:
    _get_profile_resolution_provider(tool_type).validate_profile(profile_id)


def resolve_required_builtin_mcp_server_labels(
    *,
    tool_type: str,
    profile_id: str | None = None,
) -> tuple[str, ...]:
    if profile_id is None:
        return ()
    resolved_tool = resolve_profiled_builtin_tool(tool_type=tool_type, profile_id=profile_id)
    return tuple(
        sorted(
            {
                requirement.key
                for requirement in resolved_tool.runtime_requirements
                if requirement.kind == "builtin_mcp_server"
            }
        )
    )


def profiled_builtin_requires_mcp(
    *,
    tool_type: str,
    profile_id: str | None = None,
) -> bool:
    return bool(
        resolve_required_builtin_mcp_server_labels(
            tool_type=tool_type,
            profile_id=profile_id,
        )
    )


def build_builtin_mcp_runtime_config(
    *,
    tool_type: str,
    profile_id: str | None = None,
    env: EnvSource | None = None,
) -> McpRuntimeConfig:
    env = EnvSource.from_env() if env is None else env
    definitions = tuple(
        _get_profile_resolution_provider(tool_type).required_mcp_definitions(profile_id)
    )
    if not definitions:
        return McpRuntimeConfig(enabled=False, mcp_servers={})

    mcp_servers: dict[str, McpServerRuntimeConfig] = {}
    for definition in definitions:
        if definition.build_server_entry is not None:
            server_entry = definition.build_server_entry(env)
        else:
            server_entry = dict(definition.server_entry or {})
        mcp_servers[definition.server_label] = McpServerRuntimeConfig(
            mcp_server_entry=split_hosted_server_entry(server_entry),
        )
    return McpRuntimeConfig(enabled=True, mcp_servers=mcp_servers)
