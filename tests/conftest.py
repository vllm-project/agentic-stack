import json
from collections.abc import AsyncIterator

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from agentic_api.config.runtime import RuntimeConfig
from agentic_api.core.proxy import ProxyClientManager
from agentic_api.entrypoints.app import create_app


def build_test_runtime_config(
    *, llm_api_base: str = "http://upstream", openai_api_key: str | None = "test-key"
) -> RuntimeConfig:
    return RuntimeConfig(
        llm_api_base=llm_api_base,
        openai_api_key=openai_api_key,
        gateway_host="0.0.0.0",
        gateway_port=9000,
        gateway_workers=1,
        upstream_ready_timeout_s=5.0,
        upstream_ready_interval_s=0.1,
    )


def build_upstream_stub() -> FastAPI:
    """Minimal upstream vLLM stub that handles /v1/models and /v1/responses."""
    app = FastAPI(title="Upstream Stub")

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        payload = {
            "object": "list",
            "data": [{"id": "model-a", "object": "model"}],
            "query_a": request.query_params.getlist("a"),
            "authorization": request.headers.get("authorization"),
            "saw_proxy_authorization_header": (
                "proxy-authorization" in request.headers
            ),
        }
        body = json.dumps(payload, separators=(",", ":")).encode()
        return Response(
            status_code=200,
            content=body,
            headers={
                "content-type": "application/json",
                "content-length": str(len(body)),
                "x-upstream": "models",
                "connection": "keep-alive",
            },
        )

    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        body = await request.json()

        if body.get("echo_auth"):
            return Response(
                status_code=200,
                content=json.dumps(
                    {"authorization": request.headers.get("authorization")}
                ).encode(),
                headers={"content-type": "application/json", "x-upstream": "responses"},
            )

        if body.get("force_error") == 429:
            return JSONResponse(
                status_code=429,
                content={"error": {"message": "rate limited", "code": "rate_limit"}},
                headers={"x-upstream": "error"},
            )

        if body.get("stream") is True:

            async def _stream() -> AsyncIterator[bytes]:
                yield b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                _stream(),
                status_code=200,
                headers={
                    "content-type": "text/event-stream; charset=utf-8",
                    "x-upstream": "responses-stream",
                },
            )

        out = b'{"id":"resp_test","object":"response","status":"completed"}'
        return Response(
            status_code=200,
            content=out,
            headers={
                "content-type": "application/json",
                "content-length": str(len(out)),
                "x-upstream": "responses",
                "connection": "keep-alive",
            },
        )

    return app


class _FixedProxyClientManager(ProxyClientManager):
    """ProxyClientManager that always returns a pre-built client."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        super().__init__()
        self._fixed_client = client

    def get_client(self, *, allow_sse_passthrough: bool) -> httpx.AsyncClient:
        return self._fixed_client

    async def aclose(self) -> None:
        return


@pytest.fixture
async def gateway_client() -> AsyncIterator[httpx.AsyncClient]:
    upstream_app = build_upstream_stub()
    upstream_transport = httpx.ASGITransport(app=upstream_app)
    upstream_client = httpx.AsyncClient(
        transport=upstream_transport, base_url="http://upstream"
    )

    runtime_config = build_test_runtime_config(
        llm_api_base="http://upstream",
        openai_api_key="env-upstream-key",
    )
    gateway_app = create_app(runtime_config)

    async with LifespanManager(gateway_app):
        # Lifespan has run — app.state.runtime_config and proxy_client_manager are set.
        # Override the proxy client manager to route requests to the in-process upstream stub.
        gateway_app.state.proxy_client_manager = _FixedProxyClientManager(
            upstream_client
        )

        transport = httpx.ASGITransport(app=gateway_app)
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://gateway"
            ) as client:
                yield client
        finally:
            await upstream_client.aclose()
