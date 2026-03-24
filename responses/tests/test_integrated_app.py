from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import ORJSONResponse
from fastapi.routing import APIRoute
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

from agentic_stack.configs.builders import build_runtime_config_for_integrated
from agentic_stack.configs.sources import EnvSource
from agentic_stack.entrypoints.gateway._app import IntegratedGatewayRoute
from agentic_stack.entrypoints.vllm._runtime import _build_integrated_app
from agentic_stack.routers import serving, upstream_proxy
from agentic_stack.types.openai import OpenAIResponsesResponse


def _route_endpoint(app: FastAPI, *, method: str, path: str):
    for route in app.router.routes:
        methods = getattr(route, "methods", set())
        if method in methods and getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"missing route {method} {path}")


def _route_object(app_or_router: FastAPI | object, *, method: str, path: str) -> APIRoute:
    routes = getattr(app_or_router, "router", app_or_router).routes
    for route in routes:
        methods = getattr(route, "methods", set())
        if method in methods and getattr(route, "path", None) == path:
            assert isinstance(route, APIRoute)
            return route
    raise AssertionError(f"missing route {method} {path}")


def _counter_value(metric_name: str, labels: dict[str, str]) -> float:
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == metric_name and sample.labels == labels:
                return float(sample.value)
    return 0.0


def _install_fake_native_responses_router(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, list[str]]:
    attach_calls: list[str] = []
    responses_api_router = ModuleType("vllm.entrypoints.openai.responses.api_router")

    def attach_router(app: FastAPI) -> None:
        attach_calls.append("called")

        @app.post("/v1/responses")
        async def native_responses_create() -> ORJSONResponse:
            return ORJSONResponse({"native": "create"})

        @app.get("/v1/responses/{response_id}")
        async def native_responses_get(response_id: str) -> ORJSONResponse:
            return ORJSONResponse({"native": response_id})

        @app.post("/v1/responses/{response_id}/cancel")
        async def native_responses_cancel(response_id: str) -> ORJSONResponse:
            return ORJSONResponse({"cancelled": response_id})

    responses_api_router.attach_router = attach_router
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.openai.responses.api_router",
        responses_api_router,
    )
    return responses_api_router, attach_calls


def _fake_upstream_build_app(args: SimpleNamespace, supported_tasks: list[str]) -> FastAPI:
    _ = args
    _ = supported_tasks
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/v1/models")
    async def models() -> ORJSONResponse:
        return ORJSONResponse({"data": []})

    @app.post("/v1/chat/completions")
    async def chat() -> ORJSONResponse:
        return ORJSONResponse({"object": "chat.completion"})

    @app.post("/v1/completions")
    async def completions() -> ORJSONResponse:
        return ORJSONResponse({"object": "text_completion"})

    @app.post("/v1/messages")
    async def messages() -> ORJSONResponse:
        return ORJSONResponse({"object": "message"})

    @app.get("/native-error")
    async def native_error() -> ORJSONResponse:
        raise HTTPException(status_code=418, detail="native boom")

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    import_module("vllm.entrypoints.openai.responses.api_router").attach_router(app)
    return app


@pytest.fixture
def integrated_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    _install_fake_native_responses_router(monkeypatch)
    return _build_integrated_app(
        args=SimpleNamespace(),
        supported_tasks=["generate"],
        upstream_build_app=_fake_upstream_build_app,
        runtime_config=build_runtime_config_for_integrated(
            env=EnvSource(environ={}),
            host="127.0.0.1",
            port=8000,
            web_search_profile=None,
            code_interpreter_mode="disabled",
            code_interpreter_port=5970,
            code_interpreter_workers=0,
            code_interpreter_startup_timeout_s=30.0,
            mcp_config_path=None,
            mcp_builtin_runtime_url=None,
        ),
    )


@pytest.fixture
async def integrated_client(integrated_app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=integrated_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://integrated") as client:
        yield client


def test_build_integrated_app_delegates_once_and_suppresses_native_attach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses_api_router, attach_calls = _install_fake_native_responses_router(monkeypatch)
    build_calls: list[tuple[object, object]] = []
    original_attach_router = responses_api_router.attach_router

    def _counting_build_app(args: object, supported_tasks: object) -> FastAPI:
        build_calls.append((args, supported_tasks))
        return _fake_upstream_build_app(SimpleNamespace(), ["generate"])

    app = _build_integrated_app(
        args=SimpleNamespace(),
        supported_tasks=["generate"],
        upstream_build_app=_counting_build_app,
        runtime_config=build_runtime_config_for_integrated(
            env=EnvSource(environ={}),
            host="127.0.0.1",
            port=8000,
            web_search_profile=None,
            code_interpreter_mode="disabled",
            code_interpreter_port=5970,
            code_interpreter_workers=0,
            code_interpreter_startup_timeout_s=30.0,
            mcp_config_path=None,
            mcp_builtin_runtime_url=None,
        ),
    )

    assert isinstance(app, FastAPI)
    assert len(build_calls) == 1
    assert attach_calls == []
    assert responses_api_router.attach_router is original_attach_router


def test_build_integrated_app_owns_responses_route_family(integrated_app: FastAPI) -> None:
    assert (
        _route_endpoint(
            integrated_app,
            method="POST",
            path="/v1/responses",
        )
        is serving.create_model_response
    )
    assert (
        _route_endpoint(
            integrated_app,
            method="GET",
            path="/v1/responses/{response_id}",
        )
        is serving.retrieve_model_response
    )

    cancel_routes = [
        route
        for route in integrated_app.router.routes
        if "POST" in getattr(route, "methods", set())
        and getattr(route, "path", None) == "/v1/responses/{response_id}/cancel"
    ]
    assert cancel_routes == []
    assert getattr(integrated_app.state, "agentic_stack", None) is not None
    assert integrated_app.state.agentic_stack.runtime_config is not None
    assert integrated_app.state.agentic_stack.runtime_config.runtime_mode == "integrated"


def test_build_integrated_app_keeps_native_chat_and_models(integrated_app: FastAPI) -> None:
    assert (
        _route_endpoint(
            integrated_app,
            method="GET",
            path="/v1/models",
        )
        is not upstream_proxy.proxy_models
    )
    assert (
        _route_endpoint(
            integrated_app,
            method="POST",
            path="/v1/chat/completions",
        )
        is not upstream_proxy.proxy_chat_completions
    )


def test_build_integrated_app_uses_route_wrapper_without_mutating_shared_routers(
    integrated_app: FastAPI,
) -> None:
    integrated_route = _route_object(
        integrated_app,
        method="POST",
        path="/v1/responses",
    )
    shared_route = _route_object(
        serving.router,
        method="POST",
        path="/v1/responses",
    )

    assert isinstance(integrated_route, IntegratedGatewayRoute)
    assert type(shared_route) is APIRoute
    assert shared_route is not integrated_route


@pytest.mark.anyio
async def test_integrated_app_retrieve_miss_returns_openai_error(
    integrated_client: httpx.AsyncClient,
) -> None:
    resp = await integrated_client.get("/v1/responses/resp_missing")

    assert resp.status_code == 404
    assert resp.headers["x-request-id"]
    assert resp.json()["error"]["type"] == "invalid_request_error"
    assert resp.json()["error"]["param"] == "response_id"
    assert resp.json()["error"]["code"] == "response_not_found"


@pytest.mark.anyio
async def test_integrated_app_previous_response_id_miss_returns_openai_error(
    integrated_client: httpx.AsyncClient,
) -> None:
    resp = await integrated_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "previous_response_id": "resp_missing",
            "input": [{"role": "user", "content": "Hello"}],
            "tool_choice": "none",
        },
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
    assert resp.json()["error"]["param"] == "previous_response_id"
    assert resp.json()["error"]["code"] == "previous_response_not_found"


@pytest.mark.anyio
async def test_integrated_app_store_false_response_is_not_retrievable(
    integrated_app: FastAPI,
    integrated_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeLMEngine:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            _ = args
            _ = kwargs

        async def run(self) -> OpenAIResponsesResponse:
            return OpenAIResponsesResponse(
                id="resp_store_false",
                model="some-model",
                status="completed",
                store=False,
                output=[],
            )

    monkeypatch.setattr(serving, "LMEngine", _FakeLMEngine)

    create_resp = await integrated_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "store": False,
            "input": [{"role": "user", "content": "Hello"}],
            "tool_choice": "none",
        },
    )
    assert create_resp.status_code == 200

    retrieve_resp = await integrated_client.get("/v1/responses/resp_store_false")
    assert retrieve_resp.status_code == 404
    assert retrieve_resp.json()["error"]["code"] == "response_not_found"


@pytest.mark.anyio
async def test_integrated_native_vllm_errors_are_not_remapped(
    integrated_client: httpx.AsyncClient,
) -> None:
    resp = await integrated_client.get("/native-error")

    assert resp.status_code == 418
    assert resp.json() == {"detail": "native boom"}


@pytest.mark.anyio
async def test_integrated_gateway_http_metrics_cover_only_wrapped_routes(
    integrated_client: httpx.AsyncClient,
) -> None:
    wrapped_labels = {
        "method": "GET",
        "route": "/v1/responses/{response_id}",
        "status": "404",
    }
    native_labels = {
        "method": "GET",
        "route": "/native-error",
        "status": "418",
    }
    wrapped_before = _counter_value("agentic_stack_http_requests_total", wrapped_labels)
    native_before = _counter_value("agentic_stack_http_requests_total", native_labels)

    wrapped_resp = await integrated_client.get("/v1/responses/resp_missing_metrics")
    native_resp = await integrated_client.get("/native-error")

    assert wrapped_resp.status_code == 404
    assert native_resp.status_code == 418
    assert (
        _counter_value("agentic_stack_http_requests_total", wrapped_labels) == wrapped_before + 1
    )
    assert _counter_value("agentic_stack_http_requests_total", native_labels) == native_before


@pytest.mark.anyio
async def test_integrated_metrics_endpoint_exposes_shared_gateway_metrics(
    integrated_app: FastAPI,
    integrated_client: httpx.AsyncClient,
) -> None:
    metrics_routes = [
        route
        for route in integrated_app.router.routes
        if "GET" in getattr(route, "methods", set()) and getattr(route, "path", None) == "/metrics"
    ]
    assert len(metrics_routes) == 1

    before_scrape = await integrated_client.get("/metrics")
    assert before_scrape.status_code == 200
    assert before_scrape.headers["content-type"].startswith("text/plain")

    wrapped_resp = await integrated_client.get("/v1/responses/resp_missing_metrics_scrape")
    assert wrapped_resp.status_code == 404

    after_scrape = await integrated_client.get("/metrics")
    assert after_scrape.status_code == 200
    assert (
        'agentic_stack_http_requests_total{method="GET",route="/v1/responses/{response_id}",status="404"}'
        in after_scrape.text
    )
