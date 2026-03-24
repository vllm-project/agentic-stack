from __future__ import annotations

import httpx
import pytest
from pydantic import TypeAdapter
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import ToolsetTool

from agentic_stack.mcp.runtime_client import (
    MCP_TOOL_NOT_FOUND_PREFIX,
    BuiltinMcpRuntimeClient,
    BuiltinMcpRuntimeToolMissingError,
    BuiltinMcpRuntimeTransportError,
    BuiltinMcpRuntimeUnknownServerError,
)
from agentic_stack.mcp.runtime_toolset import BuiltinMcpRuntimeToolset


@pytest.mark.anyio
async def test_runtime_client_list_tools_maps_unknown_server_404() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/internal/mcp/servers/docs/tools":
            return httpx.Response(404, json={"detail": "Unknown MCP server_label: docs"})
        return httpx.Response(500, json={"detail": "unexpected"})

    client = BuiltinMcpRuntimeClient(base_url="http://runtime")
    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url="http://runtime"
    )  # type: ignore[attr-defined]
    try:
        with pytest.raises(BuiltinMcpRuntimeUnknownServerError, match="Unknown MCP server_label"):
            await client.list_tools("docs")
    finally:
        await client._http.aclose()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_runtime_client_call_tool_maps_missing_tool_404() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/internal/mcp/servers/docs/tools/search/call":
            return httpx.Response(
                404,
                json={
                    "detail": f"{MCP_TOOL_NOT_FOUND_PREFIX} MCP tool 'search' is not available."
                },
            )
        return httpx.Response(500, json={"detail": "unexpected"})

    client = BuiltinMcpRuntimeClient(base_url="http://runtime")
    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url="http://runtime"
    )  # type: ignore[attr-defined]
    try:
        with pytest.raises(BuiltinMcpRuntimeToolMissingError):
            await client.call_tool(server_label="docs", tool_name="search", arguments={})
    finally:
        await client._http.aclose()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_runtime_client_maps_transport_error() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = BuiltinMcpRuntimeClient(base_url="http://runtime")
    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url="http://runtime"
    )  # type: ignore[attr-defined]
    try:
        with pytest.raises(BuiltinMcpRuntimeTransportError, match="unavailable"):
            await client.list_servers()
    finally:
        await client._http.aclose()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_runtime_client_call_tool_404_without_prefix_maps_unknown_server() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/internal/mcp/servers/docs/tools/search/call":
            return httpx.Response(404, json={"detail": "Unknown MCP server_label: docs"})
        return httpx.Response(500, json={"detail": "unexpected"})

    client = BuiltinMcpRuntimeClient(base_url="http://runtime")
    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url="http://runtime"
    )  # type: ignore[attr-defined]
    try:
        with pytest.raises(BuiltinMcpRuntimeUnknownServerError):
            await client.call_tool(server_label="docs", tool_name="search", arguments={})
    finally:
        await client._http.aclose()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_runtime_client_does_not_retry_internal_http() -> None:
    calls = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"detail": "temporary"})

    client = BuiltinMcpRuntimeClient(
        base_url="http://runtime",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(_handler),
            base_url="http://runtime",
        ),
    )
    try:
        with pytest.raises(BuiltinMcpRuntimeTransportError):
            await client.list_servers()
        assert calls == 1
    finally:
        await client._http.aclose()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_runtime_client_parses_success_payloads() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/internal/mcp/servers":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "server_label": "docs",
                            "enabled": True,
                            "available": True,
                            "required": False,
                            "transport": "auto",
                        }
                    ],
                },
            )
        if request.url.path == "/internal/mcp/servers/docs/tools":
            return httpx.Response(
                200,
                json={
                    "server_label": "docs",
                    "available": True,
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search docs",
                            "input_schema": {"type": "object"},
                        }
                    ],
                },
            )
        return httpx.Response(500, json={"detail": "unexpected"})

    client = BuiltinMcpRuntimeClient(base_url="http://runtime")
    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url="http://runtime"
    )  # type: ignore[attr-defined]
    try:
        servers = await client.list_servers()
        tools = await client.list_tools("docs")
        assert servers[0].server_label == "docs"
        assert tools[0].name == "search"
    finally:
        await client._http.aclose()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_runtime_toolset_raises_runtimeerror_on_missing_tool() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/internal/mcp/servers/docs/tools/search/call":
            return httpx.Response(
                404,
                json={
                    "detail": f"{MCP_TOOL_NOT_FOUND_PREFIX} MCP tool 'search' is not available."
                },
            )
        if request.url.path == "/internal/mcp/servers/docs/tools":
            return httpx.Response(
                200,
                json={
                    "server_label": "docs",
                    "available": True,
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search docs",
                            "input_schema": {"type": "object"},
                        }
                    ],
                },
            )
        return httpx.Response(500, json={"detail": "unexpected"})

    client = BuiltinMcpRuntimeClient(base_url="http://runtime")
    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url="http://runtime"
    )  # type: ignore[attr-defined]
    toolset = BuiltinMcpRuntimeToolset(server_label="docs", runtime_client=client)
    args_validator = TypeAdapter(dict[str, object]).validator
    tool = ToolsetTool(
        toolset=toolset,
        tool_def=ToolDefinition(name="search", parameters_json_schema={"type": "object"}),
        max_retries=0,
        args_validator=args_validator,
    )
    try:
        with pytest.raises(RuntimeError, match="not available for server"):
            await toolset.call_tool("search", {}, ctx=None, tool=tool)
    finally:
        await client._http.aclose()  # type: ignore[attr-defined]
