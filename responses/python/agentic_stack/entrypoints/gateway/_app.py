from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from contextvars import Token

from fastapi import APIRouter, BackgroundTasks, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from loguru import logger
from starlette.background import BackgroundTask

from agentic_stack.configs.runtime import INTERNAL_UPSTREAM_HEADER_NAME, RuntimeConfig
from agentic_stack.db import configure_db
from agentic_stack.entrypoints._state import (
    CURRENT_REQUEST_ID,
    VRAppState,
    VRRequestState,
    get_vr_app_state,
    get_vr_request_state,
    require_vr_app_state,
)
from agentic_stack.observability.metrics import (
    begin_gateway_http_request,
    configure_metrics,
    ensure_gateway_metrics_registered,
    finish_gateway_http_request,
    get_route_label,
    install_prometheus_metrics_endpoint,
    install_prometheus_metrics_middleware,
)
from agentic_stack.responses_core.store import configure_response_store
from agentic_stack.routers import mcp, serving, upstream_proxy
from agentic_stack.tools.bootstrap import register_runtime_tool_handlers
from agentic_stack.tools.code_interpreter import configure_code_interpreter
from agentic_stack.types.api import UserAgent
from agentic_stack.utils import uuid7_str
from agentic_stack.utils.exceptions import VRException
from agentic_stack.utils.handlers import (
    exception_handler,
    make_request_log_str,
    path_not_found_handler,
)

_LOGGED_REQUEST_METHODS = {"POST", "PATCH", "PUT", "DELETE"}


def _activate_runtime_config(runtime_config: RuntimeConfig) -> None:
    register_runtime_tool_handlers()
    configure_metrics(runtime_config)
    configure_db(runtime_config)
    configure_response_store(runtime_config)
    configure_code_interpreter(runtime_config)


def ensure_app_state(app: FastAPI) -> VRAppState:
    if get_vr_app_state(app) is None:
        app.state.agentic_stack = VRAppState()
    return require_vr_app_state(app)


def activate_gateway_runtime(app: FastAPI, *, runtime_config: RuntimeConfig) -> None:
    app_state = ensure_app_state(app)
    app_state.runtime_config = runtime_config
    _activate_runtime_config(runtime_config)


def _is_internal_upstream_request(
    request: Request,
    *,
    internal_upstream_header: str | None,
) -> bool:
    return bool(internal_upstream_header and request.headers.get(internal_upstream_header) == "1")


def _install_request_state(request: Request) -> tuple[str, Token[str | None]]:
    request_id = request.headers.get("x-request-id", uuid7_str())
    request.state.agentic_stack = VRRequestState(
        id=request_id,
        user_agent=UserAgent.from_user_agent_string(request.headers.get("user-agent", "")),
        timing=defaultdict(float),
    )
    request_id_token = CURRENT_REQUEST_ID.set(request_id)
    return request_id, request_id_token


def _merge_background_tasks(
    response: Response,
    tasks_to_add: Sequence[Callable[[], object]],
) -> None:
    if not tasks_to_add:
        return

    merged = BackgroundTasks()
    existing = response.background
    if isinstance(existing, BackgroundTasks):
        merged.tasks.extend(existing.tasks)
    elif isinstance(existing, BackgroundTask):
        merged.tasks.append(existing)

    for task in tasks_to_add:
        merged.add_task(task)
    response.background = merged


def _finalize_gateway_response(
    request: Request,
    response: Response,
    *,
    request_id: str,
    is_internal_upstream: bool,
    overhead_log_routes: set[str],
    log_response: bool = True,
) -> None:
    response.headers["x-request-id"] = request_id

    request_state = get_vr_request_state(request)
    if request_state is not None and request_state.billing is not None:
        _merge_background_tasks(response, [request_state.billing.process_all])

    path = request.url.path
    if log_response and not is_internal_upstream and "/health" not in path:
        logger.info(make_request_log_str(request, response.status_code))

    if request_state is None:
        return

    route_label = get_route_label(request)
    model_start_time = request_state.model_start_time
    runtime_config = require_vr_app_state(request.app).runtime_config
    if (
        runtime_config is not None
        and not is_internal_upstream
        and runtime_config.log_timings
        and model_start_time is not None
        and route_label in overhead_log_routes
    ):
        overhead = model_start_time - request_state.request_start_time
        breakdown = {key: f"{value * 1e3:,.1f} ms" for key, value in request_state.timing.items()}
        logger.info(
            f"{request_state.id} - Total overhead: {overhead * 1e3:,.1f} ms. Breakdown: {breakdown}"
        )


def _apply_gateway_openapi_customizations(app: FastAPI) -> None:
    original_openapi = app.openapi

    def _custom_openapi():
        if app.openapi_schema is None:
            openapi_schema = original_openapi()
            components = openapi_schema.setdefault("components", {})
            components["securitySchemes"] = {
                "Authentication": {"type": "http", "scheme": "bearer"},
            }
            openapi_schema["security"] = [{"Authentication": []}]
            openapi_schema["info"]["x-logo"] = {"url": ""}
            app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = _custom_openapi


class IntegratedGatewayRoute(APIRoute):
    def get_route_handler(self):  # type: ignore[override]
        route_handler = super().get_route_handler()
        overhead_log_routes = {self.path}

        async def integrated_route_handler(request: Request) -> Response:
            request_id, request_id_token = _install_request_state(request)
            is_internal_upstream = _is_internal_upstream_request(
                request,
                internal_upstream_header=INTERNAL_UPSTREAM_HEADER_NAME,
            )
            metrics_start_time = None
            status_code = 500
            log_response = True
            try:
                if not is_internal_upstream:
                    metrics_start_time = begin_gateway_http_request()
                if not is_internal_upstream and request.method in _LOGGED_REQUEST_METHODS:
                    logger.info(make_request_log_str(request))

                try:
                    response = await route_handler(request)
                except Exception as exc:
                    response = await exception_handler(request, exc, log=False)
                    log_response = False
                status_code = response.status_code
                _finalize_gateway_response(
                    request,
                    response,
                    request_id=request_id,
                    is_internal_upstream=is_internal_upstream,
                    overhead_log_routes=overhead_log_routes,
                    log_response=log_response,
                )
                return response
            finally:
                if not is_internal_upstream:
                    finish_gateway_http_request(
                        request=request,
                        status_code=status_code,
                        start_time=metrics_start_time,
                    )
                CURRENT_REQUEST_ID.reset(request_id_token)

        return integrated_route_handler


def _overhead_log_routes(*, include_upstream_proxy: bool) -> set[str]:
    routes = {route.path for route in serving.router.routes}
    if include_upstream_proxy:
        routes.update(route.path for route in upstream_proxy.router.routes)
    return routes


def install_standalone_gateway_request_middleware(
    app: FastAPI,
    *,
    include_upstream_proxy: bool,
    internal_upstream_header: str | None = None,
) -> None:
    overhead_log_routes = _overhead_log_routes(include_upstream_proxy=include_upstream_proxy)

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        request_id, request_id_token = _install_request_state(request)
        is_internal_upstream = _is_internal_upstream_request(
            request,
            internal_upstream_header=internal_upstream_header,
        )
        try:
            if not is_internal_upstream and request.method in _LOGGED_REQUEST_METHODS:
                logger.info(make_request_log_str(request))
            response: Response = await call_next(request)
            _finalize_gateway_response(
                request,
                response,
                request_id=request_id,
                is_internal_upstream=is_internal_upstream,
                overhead_log_routes=overhead_log_routes,
            )
            return response
        finally:
            CURRENT_REQUEST_ID.reset(request_id_token)


def augment_standalone_gateway_app(
    app: FastAPI,
    *,
    include_upstream_proxy: bool,
    include_metrics_route: bool,
    include_cors: bool,
    customize_openapi: bool,
    internal_upstream_header: str | None = None,
) -> FastAPI:
    ensure_app_state(app)
    ensure_gateway_metrics_registered()
    if include_metrics_route:
        install_prometheus_metrics_endpoint(app)
    install_prometheus_metrics_middleware(
        app,
        internal_upstream_header=internal_upstream_header,
    )
    app.include_router(serving.router, tags=["Serving"])
    if include_upstream_proxy:
        app.include_router(upstream_proxy.router, tags=["Upstream Proxy"])
    app.include_router(mcp.router, tags=["MCP"])
    app.add_exception_handler(VRException, exception_handler)
    app.add_exception_handler(Exception, exception_handler)
    app.add_exception_handler(404, path_not_found_handler)
    install_standalone_gateway_request_middleware(
        app,
        include_upstream_proxy=include_upstream_proxy,
        internal_upstream_header=internal_upstream_header,
    )
    if include_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    if customize_openapi:
        _apply_gateway_openapi_customizations(app)
    return app


def augment_integrated_gateway_app(app: FastAPI, *, runtime_config: RuntimeConfig) -> FastAPI:
    activate_gateway_runtime(app, runtime_config=runtime_config)
    ensure_gateway_metrics_registered()
    serving_router = APIRouter(route_class=IntegratedGatewayRoute)
    serving.install_routes(serving_router)
    app.include_router(serving_router, tags=["Serving"])
    mcp_router = APIRouter(route_class=IntegratedGatewayRoute)
    mcp.install_routes(mcp_router)
    app.include_router(mcp_router, tags=["MCP"])
    return app
