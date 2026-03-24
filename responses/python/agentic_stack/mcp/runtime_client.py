from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from agentic_stack.mcp.types import McpExecutionResult, McpServerInfo, McpToolInfo

MCP_TOOL_NOT_FOUND_PREFIX = "MCP_TOOL_NOT_FOUND:"


class _ServerListItem(BaseModel):
    server_label: str
    enabled: bool
    available: bool
    required: bool = False
    transport: str


class _ServerListPayload(BaseModel):
    data: list[_ServerListItem]


class _ToolListItem(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None
    input_schema: dict[str, Any]


class _ToolListPayload(BaseModel):
    tools: list[_ToolListItem]


class BuiltinMcpRuntimeTransportError(RuntimeError):
    pass


class BuiltinMcpRuntimeUnknownServerError(RuntimeError):
    pass


class BuiltinMcpRuntimeUnavailableServerError(RuntimeError):
    pass


class BuiltinMcpRuntimeToolMissingError(KeyError):
    pass


class BuiltinMcpRuntimeClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = float(timeout_s)
        self._owns_http = http_client is None
        if http_client is not None:
            self._http = http_client
        else:
            # Internal runtime calls should be single-attempt to avoid nested retry behavior.
            self._http = httpx.AsyncClient(
                timeout=self._timeout_s,
                follow_redirects=False,
                max_redirects=0,
            )

    def is_enabled(self) -> bool:
        return True

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def list_servers(self) -> list[McpServerInfo]:
        payload = await self._request_json(
            method="GET",
            path="/internal/mcp/servers",
            timeout_s=10.0,
        )
        try:
            parsed = _ServerListPayload.model_validate(payload)
        except ValidationError as exc:
            raise BuiltinMcpRuntimeTransportError(
                "Invalid Built-in MCP runtime server list response."
            ) from exc

        return [
            McpServerInfo(
                server_label=item.server_label,
                enabled=item.enabled,
                available=item.available,
                required=item.required,
                transport=item.transport,
            )
            for item in parsed.data
        ]

    async def list_tools(self, server_label: str) -> list[McpToolInfo]:
        payload = await self._request_json(
            method="GET",
            path=f"/internal/mcp/servers/{server_label}/tools",
            timeout_s=15.0,
            server_label=server_label,
        )
        try:
            parsed = _ToolListPayload.model_validate(payload)
        except ValidationError as exc:
            raise BuiltinMcpRuntimeTransportError(
                "Invalid Built-in MCP runtime tool list response."
            ) from exc

        return [
            McpToolInfo(
                name=item.name,
                description=item.description,
                input_schema=item.input_schema,
            )
            for item in parsed.tools
        ]

    async def call_tool(
        self,
        *,
        server_label: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> McpExecutionResult:
        payload = await self._request_json(
            method="POST",
            path=f"/internal/mcp/servers/{server_label}/tools/{tool_name}/call",
            timeout_s=60.0,
            json_payload={"arguments": dict(arguments)},
            server_label=server_label,
            tool_name=tool_name,
        )
        return McpExecutionResult(
            ok=bool(payload.get("ok")),
            output_text=payload.get("output_text")
            if isinstance(payload.get("output_text"), str)
            else None,
            error_text=payload.get("error_text")
            if isinstance(payload.get("error_text"), str)
            else None,
        )

    async def _request_json(
        self,
        *,
        method: str,
        path: str,
        timeout_s: float,
        json_payload: dict[str, Any] | None = None,
        server_label: str | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = await self._http.request(
                method,
                url,
                json=json_payload,
                timeout=timeout_s,
            )
        except (httpx.HTTPError, OSError) as exc:
            raise BuiltinMcpRuntimeTransportError(
                f"Built-in MCP runtime is unavailable: {exc.__class__.__name__}: {exc}"
            ) from exc

        if response.status_code == 404:
            detail = _extract_detail(response)
            if tool_name is not None and detail.startswith(MCP_TOOL_NOT_FOUND_PREFIX):
                raise BuiltinMcpRuntimeToolMissingError(tool_name)
            if server_label is not None:
                raise BuiltinMcpRuntimeUnknownServerError(
                    detail or f"Unknown MCP server_label: {server_label}"
                )
            raise BuiltinMcpRuntimeUnknownServerError(detail or "Unknown MCP server_label.")

        if response.status_code == 409:
            detail = _extract_detail(response)
            raise BuiltinMcpRuntimeUnavailableServerError(
                detail or f"MCP server {server_label!r} is currently unavailable."
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise BuiltinMcpRuntimeTransportError(
                "Built-in MCP runtime request failed: "
                f"status={response.status_code} body={response.text[:500]!r}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise BuiltinMcpRuntimeTransportError(
                "Built-in MCP runtime returned a non-JSON response."
            ) from exc
        if not isinstance(payload, dict):
            raise BuiltinMcpRuntimeTransportError(
                "Built-in MCP runtime returned an invalid JSON object."
            )
        return payload


def _extract_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        return response.text.strip()
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
    return response.text.strip()
