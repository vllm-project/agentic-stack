from __future__ import annotations

from agentic_stack.configs.sources import EnvSource
from agentic_stack.tools.base.types import BuiltinMcpServerDefinition

_EXA_SERVER_LABEL = "exa"
_FETCH_SERVER_LABEL = "fetch"


def _build_exa_server_entry(env: EnvSource) -> dict[str, object]:
    url = "https://mcp.exa.ai/mcp?tools=web_search_exa,crawling_exa"
    api_key = env.get_optional_str("EXA_API_KEY")
    if api_key:
        url = f"{url}&exaApiKey={api_key}"
    return {"url": url}


WEB_SEARCH_BUILTIN_MCP_SERVERS: dict[str, BuiltinMcpServerDefinition] = {
    _EXA_SERVER_LABEL: BuiltinMcpServerDefinition(
        server_label=_EXA_SERVER_LABEL,
        # Exa supports operator-supplied API keys in the MCP URL, so this
        # entry is materialized at runtime instead of being fully static.
        build_server_entry=_build_exa_server_entry,
    ),
    _FETCH_SERVER_LABEL: BuiltinMcpServerDefinition(
        server_label=_FETCH_SERVER_LABEL,
        # Shipped stdio MCP definition for page fetch/open_page support.
        server_entry={
            "command": "uvx",
            "args": ["mcp-server-fetch"],
        },
    ),
}
