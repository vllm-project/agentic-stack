from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.exposition import generate_latest

from agentic_stack.configs.runtime import RuntimeConfig

HTTP_DURATION_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    20,
    30,
    60,
)

SSE_DURATION_BUCKETS = (
    0.25,
    0.5,
    0.75,
    1,
    1.5,
    2,
    3,
    5,
    7.5,
    10,
    15,
    20,
    30,
    45,
    60,
    90,
    120,
    180,
    240,
    300,
    420,
    600,
)

TOOL_DURATION_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    20,
    30,
    60,
    120,
)

ToolType = Literal["function", "code_interpreter", "mcp", "web_search"]


@dataclass(frozen=True)
class _GatewayMetrics:
    http_requests_total: Counter
    http_request_duration_seconds: Histogram
    http_in_flight_requests: Gauge

    sse_connections_in_flight: Gauge
    sse_stream_duration_seconds: Histogram

    tool_calls_requested_total: Counter
    tool_calls_executed_total: Counter
    tool_execution_duration_seconds: Histogram
    tool_errors_total: Counter
    mcp_server_startup_total: Counter


_METRICS: _GatewayMetrics | None = None
_METRICS_ENABLED = True
_METRICS_PATH = "/metrics"


def configure_metrics(runtime_config: RuntimeConfig) -> None:
    global _METRICS_ENABLED, _METRICS_PATH
    _METRICS_ENABLED = runtime_config.metrics_enabled
    _METRICS_PATH = runtime_config.metrics_path


def _get_metrics() -> _GatewayMetrics:
    global _METRICS
    if _METRICS is not None:
        return _METRICS

    # Metrics are defined once per process. In Prometheus multiprocess mode, the underlying
    # client uses mmap-backed files under PROMETHEUS_MULTIPROC_DIR.
    _METRICS = _GatewayMetrics(
        http_requests_total=Counter(
            "agentic_stack_http_requests_total",
            "Total HTTP requests completed.",
            labelnames=("method", "route", "status"),
        ),
        http_request_duration_seconds=Histogram(
            "agentic_stack_http_request_duration_seconds",
            "HTTP request handler duration in seconds (does not include SSE stream lifetime).",
            labelnames=("method", "route"),
            buckets=HTTP_DURATION_BUCKETS,
        ),
        http_in_flight_requests=Gauge(
            "agentic_stack_http_in_flight_requests",
            "Requests currently being handled by the worker (not open SSE streams).",
            multiprocess_mode="livesum",
        ),
        sse_connections_in_flight=Gauge(
            "agentic_stack_sse_connections_in_flight",
            "SSE connections currently open (stream iterators in-flight).",
            multiprocess_mode="livesum",
        ),
        sse_stream_duration_seconds=Histogram(
            "agentic_stack_sse_stream_duration_seconds",
            "SSE stream lifetime in seconds.",
            labelnames=("route",),
            buckets=SSE_DURATION_BUCKETS,
        ),
        tool_calls_requested_total=Counter(
            "agentic_stack_tool_calls_requested_total",
            "Tool calls requested by the model (seen in the model output stream).",
            labelnames=("tool_type",),
        ),
        tool_calls_executed_total=Counter(
            "agentic_stack_tool_calls_executed_total",
            "Tool calls executed by the gateway.",
            labelnames=("tool_type",),
        ),
        tool_execution_duration_seconds=Histogram(
            "agentic_stack_tool_execution_duration_seconds",
            "Tool execution duration in seconds (gateway-executed only).",
            labelnames=("tool_type",),
            buckets=TOOL_DURATION_BUCKETS,
        ),
        tool_errors_total=Counter(
            "agentic_stack_tool_errors_total",
            "Tool execution errors (gateway-executed only).",
            labelnames=("tool_type",),
        ),
        mcp_server_startup_total=Counter(
            "agentic_stack_mcp_server_startup_total",
            "Hosted MCP server startup outcomes.",
            labelnames=("server_label", "status"),
        ),
    )
    return _METRICS


def _derive_route_label(request: Request) -> str:
    scope_route = request.scope.get("route")
    route_path = getattr(scope_route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path
    return request.url.path


def _exposition_registry() -> CollectorRegistry | None:
    """
    Return a registry suitable for `/metrics`.

    If `PROMETHEUS_MULTIPROC_DIR` is set, we expose aggregated multiprocess metrics (single coherent view
    across Gunicorn workers). Otherwise, we fall back to the default in-process registry.
    """
    if not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        return None
    try:
        from prometheus_client import multiprocess
    except Exception:
        return None

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry


def ensure_gateway_metrics_registered() -> None:
    if not _METRICS_ENABLED:
        return
    _get_metrics()


def should_skip_gateway_http_metrics(
    request: Request,
    *,
    internal_upstream_header: str | None = None,
) -> bool:
    if request.url.path in {_METRICS_PATH, "/health"}:
        return True
    return bool(internal_upstream_header and request.headers.get(internal_upstream_header) == "1")


def begin_gateway_http_request() -> float | None:
    if not _METRICS_ENABLED:
        return None
    metrics = _get_metrics()
    metrics.http_in_flight_requests.inc()
    return time.perf_counter()


def finish_gateway_http_request(
    *,
    request: Request,
    status_code: int,
    start_time: float | None,
) -> None:
    if start_time is None or not _METRICS_ENABLED:
        return

    metrics = _get_metrics()
    duration_s = max(0.0, time.perf_counter() - start_time)
    method = request.method
    route = _derive_route_label(request)
    metrics.http_request_duration_seconds.labels(method=method, route=route).observe(duration_s)
    metrics.http_requests_total.labels(
        method=method,
        route=route,
        status=str(status_code),
    ).inc()
    metrics.http_in_flight_requests.dec()


def install_prometheus_metrics_endpoint(app: FastAPI) -> None:
    """
    Install `GET {VR_METRICS_PATH}` Prometheus scrape endpoint (not in OpenAPI schema).
    """
    if not _METRICS_ENABLED:
        return

    @app.get(_METRICS_PATH, include_in_schema=False)
    async def metrics_endpoint() -> Response:
        registry = _exposition_registry()
        body = generate_latest(registry) if registry is not None else generate_latest()
        return Response(content=body, media_type=CONTENT_TYPE_LATEST)


def install_prometheus_metrics_middleware(
    app: FastAPI,
    *,
    internal_upstream_header: str | None = None,
) -> None:
    """
    Install one HTTP middleware for low-cardinality Golden Signals.
    """
    if not _METRICS_ENABLED:
        return

    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next):
        if should_skip_gateway_http_metrics(
            request,
            internal_upstream_header=internal_upstream_header,
        ):
            return await call_next(request)

        start = begin_gateway_http_request()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            finish_gateway_http_request(
                request=request,
                status_code=status_code,
                start_time=start,
            )


def install_prometheus_metrics(
    app: FastAPI,
    *,
    internal_upstream_header: str | None = None,
) -> None:
    """
    Install:
    - `GET {VR_METRICS_PATH}` Prometheus scrape endpoint (not in OpenAPI schema)
    - one HTTP middleware for low-cardinality Golden Signals
    """
    install_prometheus_metrics_endpoint(app)
    install_prometheus_metrics_middleware(
        app,
        internal_upstream_header=internal_upstream_header,
    )


def instrument_sse_stream(
    *,
    route: str,
    agen: AsyncIterator[str],
) -> AsyncIterator[str]:
    """
    Wrap an SSE async iterator to record stream lifetime and connection in-flight gauge.

    Notes:
    - Must be called *before* the first `anext(...)` on the iterator to include the full lifetime.
    - Decrements gauges and records duration on normal completion, disconnect, and errors.
    """
    if not _METRICS_ENABLED:
        return agen

    metrics = _get_metrics()
    metrics.sse_connections_in_flight.inc()
    start = time.perf_counter()

    async def _wrapped() -> AsyncIterator[str]:
        try:
            async for chunk in agen:
                yield chunk
        finally:
            duration_s = max(0.0, time.perf_counter() - start)
            metrics.sse_stream_duration_seconds.labels(route=route).observe(duration_s)
            metrics.sse_connections_in_flight.dec()

    return _wrapped()


def record_tool_call_requested(tool_type: ToolType) -> None:
    if not _METRICS_ENABLED:
        return
    _get_metrics().tool_calls_requested_total.labels(tool_type=tool_type).inc()


def record_tool_executed(*, tool_type: ToolType, duration_s: float, errored: bool) -> None:
    if not _METRICS_ENABLED:
        return

    metrics = _get_metrics()
    metrics.tool_calls_executed_total.labels(tool_type=tool_type).inc()
    metrics.tool_execution_duration_seconds.labels(tool_type=tool_type).observe(
        max(0.0, float(duration_s))
    )
    if errored:
        metrics.tool_errors_total.labels(tool_type=tool_type).inc()


def record_mcp_server_startup(*, server_label: str, status: Literal["ok", "error"]) -> None:
    if not _METRICS_ENABLED:
        return
    _get_metrics().mcp_server_startup_total.labels(
        server_label=server_label,
        status=status,
    ).inc()


def get_route_label(request: Request) -> str:
    """
    Public helper for places that need a low-cardinality route label.
    """
    return _derive_route_label(request)
