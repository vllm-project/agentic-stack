from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from agentic_stack.configs.builders import build_runtime_config_for_standalone
from agentic_stack.configs.sources import EnvSource
from agentic_stack.entrypoints._state import VRAppState, VRRequestState
from agentic_stack.routers import mcp, serving, upstream_proxy
from agentic_stack.types.api import UserAgent
from agentic_stack.utils.exceptions import VRException
from agentic_stack.utils.handlers import exception_handler, path_not_found_handler


def _build_gateway_app() -> FastAPI:
    app = FastAPI(title="VR Gateway (proxy test)")
    app.state.agentic_stack = VRAppState(
        runtime_config=build_runtime_config_for_standalone(
            env=EnvSource(environ={"VR_LLM_API_BASE": "http://upstream/v1"})
        )
    )
    app.include_router(serving.router)
    app.include_router(upstream_proxy.router)
    app.include_router(mcp.router)

    @app.middleware("http")
    async def _init_request_state(request, call_next):
        request.state.agentic_stack = VRRequestState(
            id="test-request-id",
            user_agent=UserAgent.from_user_agent_string("pytest"),
            timing=defaultdict(float),
        )
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.agentic_stack.id
        return response

    app.add_exception_handler(VRException, exception_handler)
    app.add_exception_handler(Exception, exception_handler)
    app.add_exception_handler(404, path_not_found_handler)
    return app


def _build_upstream_app() -> FastAPI:
    app = FastAPI(title="Upstream Stub")

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        payload = {
            "object": "list",
            "data": [{"id": "model-a", "object": "model"}],
            "query_a": request.query_params.getlist("a"),
            "authorization": request.headers.get("authorization"),
            "saw_proxy_authorization_header": ("proxy-authorization" in request.headers),
        }
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
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

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body = await request.json()
        if body.get("force_error") == 429:
            return ORJSONResponse(
                status_code=429,
                content={"error": {"message": "rate limited", "code": "rate_limit"}},
                headers={"x-upstream": "error"},
            )
        if body.get("stream") is True:

            async def _stream() -> AsyncIterator[bytes]:
                yield b'data: {"id":"chatcmpl_stream","choices":[{"delta":{"content":"hello"}}]}\n\n'
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                _stream(),
                status_code=200,
                headers={
                    "content-type": "text/event-stream; charset=utf-8",
                    "x-upstream": "chat-stream",
                },
            )

        out = b'{"id":"chatcmpl_non_stream","object":"chat.completion"}'
        return Response(
            status_code=201,
            content=out,
            headers={
                "content-type": "application/json",
                "content-length": str(len(out)),
                "x-upstream": "chat",
                "connection": "keep-alive",
            },
        )

    return app


@pytest.fixture
async def proxy_gateway_client() -> AsyncIterator[httpx.AsyncClient]:
    upstream_app = _build_upstream_app()
    upstream_transport = httpx.ASGITransport(app=upstream_app)
    upstream_client = httpx.AsyncClient(transport=upstream_transport)

    gateway_app = _build_gateway_app()
    gateway_transport = httpx.ASGITransport(app=gateway_app)

    gateway_app.state.agentic_stack.runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={
                "VR_LLM_API_BASE": "http://upstream/v1",
                "VR_OPENAI_API_KEY": "env-upstream-key",
            }
        )
    )

    class _FixedManager:
        def __init__(self, client: httpx.AsyncClient) -> None:
            self._client = client

        def get_client(self, *, allow_sse_passthrough: bool) -> httpx.AsyncClient:
            return self._client

        async def aclose(self) -> None:
            return

    gateway_app.state.agentic_stack.proxy_client_manager = _FixedManager(upstream_client)
    try:
        async with httpx.AsyncClient(
            transport=gateway_transport,
            base_url="http://gateway",
        ) as gateway_client:
            yield gateway_client
    finally:
        gateway_app.state.agentic_stack.runtime_config = None
        await upstream_client.aclose()


@pytest.mark.anyio
async def test_proxy_models_passthrough_preserves_status_body_headers_and_query(
    proxy_gateway_client: httpx.AsyncClient,
) -> None:
    resp = await proxy_gateway_client.get(
        "/v1/models?a=1&a=2",
        headers={"Proxy-Authorization": "Basic abc123"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-upstream") == "models"
    assert "connection" not in {k.lower() for k in resp.headers}
    assert "content-length" in {k.lower() for k in resp.headers}

    payload = resp.json()
    assert payload["query_a"] == ["1", "2"]
    assert payload["saw_proxy_authorization_header"] is False


@pytest.mark.anyio
async def test_proxy_models_authorization_precedence(
    proxy_gateway_client: httpx.AsyncClient,
) -> None:
    resp_env = await proxy_gateway_client.get("/v1/models")
    assert resp_env.status_code == 200
    assert resp_env.json()["authorization"] == "Bearer env-upstream-key"

    resp_client = await proxy_gateway_client.get(
        "/v1/models",
        headers={"Authorization": "Bearer client-token"},
    )
    assert resp_client.status_code == 200
    assert resp_client.json()["authorization"] == "Bearer client-token"


@pytest.mark.anyio
async def test_proxy_chat_non_stream_passthrough(
    proxy_gateway_client: httpx.AsyncClient,
) -> None:
    resp = await proxy_gateway_client.post(
        "/v1/chat/completions",
        json={"model": "model-a", "messages": [], "stream": False},
    )
    assert resp.status_code == 201
    assert resp.headers.get("x-upstream") == "chat"
    assert "connection" not in {k.lower() for k in resp.headers}
    assert resp.content == b'{"id":"chatcmpl_non_stream","object":"chat.completion"}'


@pytest.mark.anyio
async def test_proxy_chat_stream_passthrough_and_done(
    proxy_gateway_client: httpx.AsyncClient,
) -> None:
    async with proxy_gateway_client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "model-a", "messages": [], "stream": True},
    ) as resp:
        chunks: list[str] = []
        async for chunk in resp.aiter_text():
            chunks.append(chunk)

    text = "".join(chunks)
    assert resp.status_code == 200
    assert resp.headers.get("x-upstream") == "chat-stream"
    assert resp.headers.get("x-accel-buffering") == "no"
    assert "data: [DONE]" in text
    assert "chatcmpl_stream" in text
    assert "content-length" not in {k.lower() for k in resp.headers}


@pytest.mark.anyio
async def test_proxy_chat_stream_mid_stream_failure_closes_without_synthetic_events() -> None:
    class _FakeStreamResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream; charset=utf-8", "x-upstream": "fake-stream"}

        async def aiter_raw(self) -> AsyncIterator[bytes]:
            yield b'data: {"id":"chatcmpl_stream","choices":[{"delta":{"content":"hello"}}]}\n\n'
            raise RuntimeError("simulated upstream stream failure")

        async def aclose(self) -> None:
            return

    class _MidStreamFailClient:
        def build_request(self, method: str, url: str, **kwargs) -> httpx.Request:
            return httpx.Request(method=method, url=url, headers=kwargs.get("headers"))

        async def send(self, request: httpx.Request, **kwargs):
            return _FakeStreamResponse()

    class _MidStreamFailManager:
        def get_client(self, *, allow_sse_passthrough: bool) -> _MidStreamFailClient:
            return _MidStreamFailClient()

        async def aclose(self) -> None:
            return

    app = _build_gateway_app()
    app.state.agentic_stack.proxy_client_manager = _MidStreamFailManager()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [], "stream": True},
        ) as resp:
            chunks: list[str] = []
            async for chunk in resp.aiter_text():
                chunks.append(chunk)

    text = "".join(chunks)
    assert resp.status_code == 200
    assert "chatcmpl_stream" in text
    assert "data: [DONE]" not in text
    assert "upstream_timeout" not in text
    assert "upstream_unavailable" not in text


@pytest.mark.anyio
async def test_proxy_upstream_http_error_payload_passthrough(
    proxy_gateway_client: httpx.AsyncClient,
) -> None:
    resp = await proxy_gateway_client.post(
        "/v1/chat/completions",
        json={"model": "model-a", "messages": [], "force_error": 429},
    )
    assert resp.status_code == 429
    assert resp.headers.get("x-upstream") == "error"
    assert resp.json() == {"error": {"message": "rate limited", "code": "rate_limit"}}


@pytest.mark.anyio
async def test_proxy_transport_connect_error_maps_to_502() -> None:
    class _ConnectErrorClient:
        def build_request(self, method: str, url: str, **kwargs) -> httpx.Request:
            return httpx.Request(method=method, url=url, headers=kwargs.get("headers"))

        async def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
            raise httpx.ConnectError("failed to connect", request=request)

    class _ConnectErrorManager:
        def get_client(self, *, allow_sse_passthrough: bool) -> _ConnectErrorClient:
            return _ConnectErrorClient()

        async def aclose(self) -> None:
            return

    app = _build_gateway_app()
    app.state.agentic_stack.proxy_client_manager = _ConnectErrorManager()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        resp = await client.get("/v1/models")

    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_unavailable"
    assert "x-request-id" in {k.lower() for k in resp.headers}


@pytest.mark.anyio
async def test_proxy_transport_timeout_maps_to_504() -> None:
    class _TimeoutClient:
        def build_request(self, method: str, url: str, **kwargs) -> httpx.Request:
            return httpx.Request(method=method, url=url, headers=kwargs.get("headers"))

        async def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

    class _TimeoutManager:
        def get_client(self, *, allow_sse_passthrough: bool) -> _TimeoutClient:
            return _TimeoutClient()

        async def aclose(self) -> None:
            return

    app = _build_gateway_app()
    app.state.agentic_stack.proxy_client_manager = _TimeoutManager()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        resp = await client.get("/v1/models")

    assert resp.status_code == 504
    assert resp.json()["error"]["code"] == "upstream_timeout"
    assert "x-request-id" in {k.lower() for k in resp.headers}


@pytest.mark.anyio
async def test_route_ownership_serving_and_mcp_unchanged(
    proxy_gateway_client: httpx.AsyncClient,
) -> None:
    bad_resp = await proxy_gateway_client.post("/v1/responses", json={})
    assert bad_resp.status_code == 422

    mcp_resp = await proxy_gateway_client.get("/v1/mcp/servers")
    assert mcp_resp.status_code == 200
    assert mcp_resp.json()["object"] == "list"
