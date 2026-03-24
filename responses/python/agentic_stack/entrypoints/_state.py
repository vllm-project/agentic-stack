from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from starlette.requests import Request

from agentic_stack.types.api import UserAgent

CURRENT_REQUEST_ID: ContextVar[str | None] = ContextVar("current_request_id", default=None)

if TYPE_CHECKING:
    from asyncio.subprocess import Process

    from agentic_stack.configs.runtime import RuntimeConfig
    from agentic_stack.mcp.runtime_client import BuiltinMcpRuntimeClient
    from agentic_stack.routers.upstream_proxy import ProxyClientManager
    from agentic_stack.utils.cassette_replay import CassetteReplayer


@dataclass(slots=True)
class VRAppState:
    """Typed container for `FastAPI.app.state.agentic_stack`.

    Starlette's `app.state` is a dynamic attribute bag. We store all agentic_stack-owned state under a
    single stable attribute (`app.state.agentic_stack`) so the rest of the codebase can use direct
    attribute access without defensive `getattr(...)` checks.
    """

    code_interpreter_process: Process | None = None
    cassette_replayer: CassetteReplayer | None = None
    builtin_mcp_runtime_client: BuiltinMcpRuntimeClient | None = None
    proxy_client_manager: ProxyClientManager | None = None
    runtime_config: RuntimeConfig | None = None


@dataclass(slots=True)
class VRRequestState:
    """Typed container for `Request.state.agentic_stack`.

    This is initialized for every request by middleware so downstream code can safely access
    request-scoped values (e.g. request id) via `request.state.agentic_stack`.
    """

    id: str
    user_agent: UserAgent
    request_start_time: float = field(default_factory=perf_counter)
    timing: dict[str, float] = field(default_factory=dict)
    model_start_time: float | None = None
    billing: Any | None = None


def get_vr_request_state(request: Request) -> VRRequestState | None:
    state = getattr(request.state, "agentic_stack", None)
    return state if isinstance(state, VRRequestState) else None


def get_vr_app_state(app: FastAPI) -> VRAppState | None:
    state = getattr(app.state, "agentic_stack", None)
    return state if isinstance(state, VRAppState) else None


def require_vr_app_state(app: FastAPI) -> VRAppState:
    app_state = get_vr_app_state(app)
    if app_state is None:
        raise RuntimeError("agentic_stack app state is not initialized.")
    return app_state


def require_runtime_config(app: FastAPI) -> RuntimeConfig:
    runtime_config = require_vr_app_state(app).runtime_config
    if runtime_config is None:
        raise RuntimeError("agentic_stack runtime config is not initialized.")
    return runtime_config


def get_request_id(request: Request) -> str:
    request_state = get_vr_request_state(request)
    if request_state is not None and request_state.id:
        return request_state.id
    request_id = request.headers.get("x-request-id")
    if request_id:
        return request_id
    current_request_id = CURRENT_REQUEST_ID.get()
    return current_request_id or ""
