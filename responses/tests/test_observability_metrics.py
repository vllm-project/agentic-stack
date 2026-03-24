from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from agentic_stack.observability.metrics import install_prometheus_metrics, instrument_sse_stream


@pytest.fixture
def metrics_app() -> FastAPI:
    app = FastAPI(title="agentic_stack observability (test)")
    install_prometheus_metrics(app)

    @app.get("/hello")
    async def hello() -> dict[str, str]:
        return {"ok": "1"}

    @app.get("/sse")
    async def sse() -> StreamingResponse:
        async def _gen() -> AsyncIterator[str]:
            yield "data: hello\n\n"
            await asyncio.sleep(0)
            yield "data: bye\n\n"

        return StreamingResponse(
            content=instrument_sse_stream(route="/sse", agen=_gen()),
            media_type="text/event-stream",
        )

    return app


@pytest.mark.anyio
async def test_metrics_endpoint_exposes_expected_names(metrics_app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=metrics_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        assert "version=" in resp.headers["content-type"]

        text = resp.text
        assert "agentic_stack_http_requests_total" in text
        assert "agentic_stack_http_request_duration_seconds" in text
        assert "agentic_stack_http_in_flight_requests" in text
        assert "agentic_stack_sse_connections_in_flight" in text
        assert "agentic_stack_sse_stream_duration_seconds" in text
        assert "agentic_stack_tool_calls_requested_total" in text
        assert "agentic_stack_tool_calls_executed_total" in text
        assert "agentic_stack_tool_execution_duration_seconds" in text
        assert "agentic_stack_tool_errors_total" in text
        assert "agentic_stack_mcp_server_startup_total" in text


@pytest.mark.anyio
async def test_http_metrics_include_route_template(metrics_app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=metrics_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        hello = await client.get("/hello")
        assert hello.status_code == 200

        metrics = await client.get("/metrics")
        assert metrics.status_code == 200
        assert (
            'agentic_stack_http_requests_total{method="GET",route="/hello",status="200"}'
            in metrics.text
        )


@pytest.mark.anyio
async def test_sse_metrics_record_stream_lifetime(metrics_app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=metrics_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("GET", "/sse") as resp:
            assert resp.status_code == 200
            _ = await resp.aread()

        metrics = await client.get("/metrics")
        assert metrics.status_code == 200
        match = re.search(
            r'^agentic_stack_sse_stream_duration_seconds_count\{route="/sse"\} ([0-9.]+)\s*$',
            metrics.text,
            flags=re.MULTILINE,
        )
        assert match is not None
        assert float(match.group(1)) >= 1.0
