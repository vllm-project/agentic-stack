from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import TypeAdapter
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import ToolsetTool

from agentic_stack.entrypoints.mcp_runtime import app as runtime_app


class _AlwaysMissingToolset:
    def __init__(self) -> None:
        args_validator = TypeAdapter(dict[str, Any]).validator
        self._tool = ToolsetTool(
            toolset=self,
            tool_def=ToolDefinition(name="search", parameters_json_schema={"type": "object"}),
            max_retries=0,
            args_validator=args_validator,
        )

    @property
    def id(self) -> str | None:
        return None

    async def get_tools(self, ctx) -> dict[str, ToolsetTool[Any]]:
        _ = ctx
        return {"search": self._tool}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx,
        tool: ToolsetTool[Any],
    ) -> Any:
        _ = name, tool_args, ctx, tool
        raise KeyError("search")


class _FakeRegistry:
    def __init__(self, toolset: _AlwaysMissingToolset) -> None:
        self._toolset = toolset

    def has_server(self, server_label: str) -> bool:
        return server_label == "docs"

    async def call_tool_with_refresh(
        self,
        *,
        server_label: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> object:
        _ = arguments
        assert server_label == "docs"
        assert tool_name == "search"
        tool = SimpleNamespace(tool_def=SimpleNamespace(name="search"))
        return await self._toolset.call_tool("search", {}, ctx=None, tool=tool)

    def get_server_secret_values(self, server_label: str) -> tuple[str, ...]:
        assert server_label == "docs"
        return ()


class _SecretFailingRegistry:
    def has_server(self, server_label: str) -> bool:
        return server_label == "docs"

    async def call_tool_with_refresh(
        self,
        *,
        server_label: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> object:
        _ = arguments
        assert server_label == "docs"
        assert tool_name == "search"
        raise RuntimeError("call failed with token super-secret-token")

    def get_server_secret_values(self, server_label: str) -> tuple[str, ...]:
        assert server_label == "docs"
        return ("super-secret-token",)


@pytest.mark.anyio
async def test_runtime_call_tool_keyerror_after_refresh_is_nonfatal_item_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_stack.entrypoints.mcp_runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "_get_registry",
        lambda: _FakeRegistry(_AlwaysMissingToolset()),
    )
    transport = httpx.ASGITransport(app=runtime_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
        resp = await client.post(
            "/internal/mcp/servers/docs/tools/search/call",
            json={"arguments": {"query": "q"}},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is False
    assert payload["output_text"] is None
    assert isinstance(payload.get("error_text"), str)


@pytest.mark.anyio
async def test_runtime_call_tool_redacts_secret_values_in_error_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_stack.entrypoints.mcp_runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "_get_registry",
        lambda: _SecretFailingRegistry(),
    )
    transport = httpx.ASGITransport(app=runtime_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
        resp = await client.post(
            "/internal/mcp/servers/docs/tools/search/call",
            json={"arguments": {"query": "q"}},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is False
    assert payload["output_text"] is None
    assert isinstance(payload.get("error_text"), str)
    assert "super-secret-token" not in payload["error_text"]
    assert "***" in payload["error_text"]


@pytest.mark.anyio
async def test_runtime_echoes_x_request_id_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_stack.entrypoints.mcp_runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "_get_registry",
        lambda: _FakeRegistry(_AlwaysMissingToolset()),
    )
    transport = httpx.ASGITransport(app=runtime_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
        resp = await client.post(
            "/internal/mcp/servers/docs/tools/search/call",
            json={"arguments": {"query": "q"}},
            headers={"x-request-id": "rid-123"},
        )

    assert resp.status_code == 200
    assert resp.headers.get("x-request-id") == "rid-123"
