from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from loguru import logger
from pydantic import BaseModel, Field

from agentic_stack.configs.builders import build_runtime_config_for_standalone
from agentic_stack.configs.sources import EnvSource
from agentic_stack.mcp.config import load_mcp_runtime_config, merge_mcp_runtime_configs
from agentic_stack.mcp.hosted_registry import (
    HostedMCPRegistry,
    HostedMcpStaleToolError,
    HostedMcpToolNotFoundError,
)
from agentic_stack.mcp.runtime_client import MCP_TOOL_NOT_FOUND_PREFIX
from agentic_stack.mcp.utils import canonicalize_output_text, redact_and_truncate_error_text
from agentic_stack.tools.ids import WEB_SEARCH_TOOL
from agentic_stack.tools.profile_resolution import build_builtin_mcp_runtime_config
from agentic_stack.utils import uuid7_str
from agentic_stack.utils.exceptions import BadInputError


class _CallToolRequest(BaseModel):
    arguments: dict[str, object] = Field(default_factory=dict)
    request_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    env = EnvSource.from_env()
    runtime_config = build_runtime_config_for_standalone(env=env)
    mcp_runtime_config = merge_mcp_runtime_configs(
        build_builtin_mcp_runtime_config(
            tool_type=WEB_SEARCH_TOOL,
            profile_id=runtime_config.web_search_profile,
            env=env,
        ),
        load_mcp_runtime_config(runtime_config.mcp_config_path),
    )
    registry = HostedMCPRegistry(
        config=mcp_runtime_config,
        startup_timeout_s=runtime_config.mcp_hosted_startup_timeout_sec,
        tool_timeout_s=runtime_config.mcp_hosted_tool_timeout_sec,
    )
    await registry.startup()
    app.state.hosted_mcp_registry = registry
    yield
    await registry.shutdown()


app = FastAPI(title="Built-in MCP Runtime", lifespan=lifespan)


@app.middleware("http")
async def add_request_id(request: Request, call_next) -> Response:
    request_id = request.headers.get("x-request-id", uuid7_str())
    request.state.request_id = request_id
    response: Response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


def _get_registry() -> HostedMCPRegistry:
    registry = getattr(app.state, "hosted_mcp_registry", None)
    if registry is None:  # pragma: no cover - defensive
        raise RuntimeError("Built-in MCP registry is not initialized.")
    return registry


def _missing_tool_http_exception(*, server_label: str, tool_name: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=(
            f"{MCP_TOOL_NOT_FOUND_PREFIX} "
            f"MCP tool {tool_name!r} is not available for server {server_label!r}."
        ),
    )


def _resolve_runtime_request_id(request: Request, body_request_id: str | None) -> str:
    if body_request_id is not None and body_request_id.strip():
        return body_request_id.strip()
    return request.state.request_id


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/internal/mcp/servers")
async def list_mcp_servers() -> dict[str, object]:
    registry = _get_registry()
    data = [
        {
            "server_label": info.server_label,
            "enabled": info.enabled,
            "available": info.available,
            "required": info.required,
            "transport": info.transport,
        }
        for info in registry.list_servers()
    ]
    return {"object": "list", "data": data}


@app.get("/internal/mcp/servers/{server_label}/tools")
async def list_mcp_server_tools(server_label: str) -> dict[str, object]:
    registry = _get_registry()
    if not registry.has_server(server_label):
        raise HTTPException(status_code=404, detail=f"Unknown MCP server_label: {server_label}")
    try:
        tools = await registry.list_tools(server_label)
    except BadInputError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    tools = sorted(tools, key=lambda item: item.name)
    return {
        "server_label": server_label,
        "available": True,
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ],
    }


@app.post("/internal/mcp/servers/{server_label}/tools/{tool_name}/call")
async def call_mcp_server_tool(
    request: Request,
    server_label: str,
    tool_name: str,
    body: _CallToolRequest,
) -> dict[str, object]:
    request_id = _resolve_runtime_request_id(request, body.request_id)
    registry = _get_registry()
    if not registry.has_server(server_label):
        raise HTTPException(status_code=404, detail=f"Unknown MCP server_label: {server_label}")
    try:
        raw = await registry.call_tool_with_refresh(
            server_label=server_label,
            tool_name=tool_name,
            arguments=dict(body.arguments),
        )
    except HostedMcpToolNotFoundError as exc:
        raise _missing_tool_http_exception(server_label=server_label, tool_name=tool_name) from exc
    except HostedMcpStaleToolError as exc:
        # Defensive: runtime registry should normally consume stale-tool misses in
        # `call_tool_with_refresh(...)`, but keep this mapping explicit at the HTTP boundary.
        raise _missing_tool_http_exception(server_label=server_label, tool_name=tool_name) from exc
    except BadInputError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning(
            f"{request_id} - Built-in MCP tool call failed: "
            f"server_label={server_label!r} tool_name={tool_name!r} error={exc.__class__.__name__}"
        )
        return {
            "ok": False,
            "output_text": None,
            "error_text": redact_and_truncate_error_text(
                text=str(exc).strip() or exc.__class__.__name__,
                secret_values=registry.get_server_secret_values(server_label),
            ),
        }

    return {
        "ok": True,
        "output_text": canonicalize_output_text(raw),
        "error_text": None,
    }


if __name__ == "__main__":
    raise SystemExit(
        "Direct execution of agentic_stack.entrypoints.mcp_runtime is unsupported. "
        "Use `agentic-stacks serve` or `vllm serve --responses`."
    )
