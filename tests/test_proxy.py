from collections.abc import AsyncIterator

import httpx
import pytest
from asgi_lifespan import LifespanManager

from agentic_api.entrypoints.app import create_app
from tests.conftest import build_test_runtime_config


@pytest.mark.anyio
async def test_proxy_responses_non_stream_passthrough(
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={"model": "model-a", "input": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-upstream") == "responses"
    assert "connection" not in {k.lower() for k in resp.headers}
    assert resp.json()["id"] == "resp_test"


@pytest.mark.anyio
async def test_proxy_responses_stream_passthrough(
    gateway_client: httpx.AsyncClient,
) -> None:
    async with gateway_client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "model-a",
            "input": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ) as resp:
        chunks: list[str] = []
        async for chunk in resp.aiter_text():
            chunks.append(chunk)

    text = "".join(chunks)
    assert resp.status_code == 200
    assert resp.headers.get("x-upstream") == "responses-stream"
    assert resp.headers.get("x-accel-buffering") == "no"
    assert "data: [DONE]" in text
    assert "response.output_text.delta" in text
    assert "content-length" not in {k.lower() for k in resp.headers}


@pytest.mark.anyio
async def test_proxy_hop_by_hop_headers_stripped(
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={"model": "model-a", "input": []},
        headers={"Proxy-Authorization": "Basic abc123"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-upstream") == "responses"
    # hop-by-hop headers must not be forwarded to the client
    assert "connection" not in {k.lower() for k in resp.headers}


@pytest.mark.anyio
async def test_proxy_authorization_env_key_injected(
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={"model": "model-a", "input": [], "echo_auth": True},
    )
    assert resp.status_code == 200
    assert resp.json()["authorization"] == "Bearer env-upstream-key"


@pytest.mark.anyio
async def test_proxy_authorization_client_header_takes_precedence(
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={"model": "model-a", "input": [], "echo_auth": True},
        headers={"Authorization": "Bearer client-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["authorization"] == "Bearer client-token"


@pytest.mark.anyio
async def test_proxy_upstream_http_error_passthrough(
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={"model": "model-a", "input": [], "force_error": 429},
    )
    assert resp.status_code == 429
    assert resp.headers.get("x-upstream") == "error"
    assert resp.json() == {"error": {"message": "rate limited", "code": "rate_limit"}}


@pytest.mark.anyio
async def test_proxy_stream_mid_stream_failure_closes_without_synthetic_events() -> (
    None
):
    class _FakeStreamResponse:
        status_code = 200
        headers = {
            "content-type": "text/event-stream; charset=utf-8",
            "x-upstream": "fake-stream",
        }

        async def aiter_raw(self) -> AsyncIterator[bytes]:
            yield b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
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

    app = create_app(build_test_runtime_config())
    async with LifespanManager(app):
        app.state.proxy_client_manager = _MidStreamFailManager()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway"
        ) as client:
            async with client.stream(
                "POST",
                "/v1/responses",
                json={"model": "model-a", "input": [], "stream": True},
            ) as resp:
                chunks: list[str] = []
                async for chunk in resp.aiter_text():
                    chunks.append(chunk)

    text = "".join(chunks)
    assert resp.status_code == 200
    assert "response.output_text.delta" in text
    assert "data: [DONE]" not in text
    assert "upstream_timeout" not in text
    assert "upstream_unavailable" not in text


@pytest.mark.anyio
async def test_proxy_connect_error_maps_to_502() -> None:
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

    app = create_app(build_test_runtime_config())
    async with LifespanManager(app):
        app.state.proxy_client_manager = _ConnectErrorManager()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway"
        ) as client:
            resp = await client.post(
                "/v1/responses", json={"model": "model-a", "input": []}
            )

    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_unavailable"


@pytest.mark.anyio
async def test_proxy_timeout_maps_to_504() -> None:
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

    app = create_app(build_test_runtime_config())
    async with LifespanManager(app):
        app.state.proxy_client_manager = _TimeoutManager()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway"
        ) as client:
            resp = await client.post(
                "/v1/responses", json={"model": "model-a", "input": []}
            )

    assert resp.status_code == 504
    assert resp.json()["error"]["code"] == "upstream_timeout"
