from __future__ import annotations

import sys
import threading
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from agentic_stack.configs.sources import EnvSource
from agentic_stack.entrypoints._helper_runtime import SpawnCodeInterpreterSpec
from agentic_stack.entrypoints._state import CURRENT_REQUEST_ID
from agentic_stack.entrypoints.vllm._runtime import (
    HelperProcess,
    _start_helper_watchdog,
    run_integrated_serve,
)
from agentic_stack.entrypoints.vllm._spec import IntegratedServeSpec
from agentic_stack.utils.io import get_async_client


class _FakeStore:
    async def ensure_schema(self) -> None:
        return None


class _FakeProc:
    def __init__(self, poll_code: int | None = None) -> None:
        self.poll_code = poll_code
        self.stdout = None

    def poll(self) -> int | None:
        return self.poll_code


def _install_fake_vllm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    upstream_main,
) -> ModuleType:
    vllm_mod = ModuleType("vllm")
    entrypoints_mod = ModuleType("vllm.entrypoints")
    openai_mod = ModuleType("vllm.entrypoints.openai")
    cli_main_mod = ModuleType("vllm.entrypoints.cli.main")
    api_server_mod = ModuleType("vllm.entrypoints.openai.api_server")
    responses_api_router_mod = ModuleType("vllm.entrypoints.openai.responses.api_router")

    def attach_router(app: FastAPI) -> None:
        @app.post("/v1/responses")
        async def native_responses_create() -> ORJSONResponse:
            return ORJSONResponse({"native": "create"})

        @app.get("/v1/responses/{response_id}")
        async def native_responses_get(response_id: str) -> ORJSONResponse:
            return ORJSONResponse({"native": response_id})

        @app.post("/v1/responses/{response_id}/cancel")
        async def native_responses_cancel(response_id: str) -> ORJSONResponse:
            return ORJSONResponse({"cancelled": response_id})

    responses_api_router_mod.attach_router = attach_router

    def original_build_app(args, supported_tasks) -> FastAPI:  # type: ignore[no-untyped-def]
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

        import_module("vllm.entrypoints.openai.responses.api_router").attach_router(app)
        return app

    api_server_mod.build_app = original_build_app
    cli_main_mod.main = upstream_main

    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", entrypoints_mod)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai", openai_mod)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.cli.main", cli_main_mod)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.api_server", api_server_mod)
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.openai.responses.api_router",
        responses_api_router_mod,
    )
    return api_server_mod


def _patch_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    env: EnvSource,
    popen_factory=None,
) -> None:
    import agentic_stack.entrypoints._helper_runtime as helper_runtime_mod
    import agentic_stack.entrypoints.vllm._runtime as runtime_mod
    import agentic_stack.responses_core.store as store_mod

    monkeypatch.setattr(store_mod, "get_default_response_store", lambda: _FakeStore())
    monkeypatch.setattr(runtime_mod.EnvSource, "from_env", classmethod(lambda cls: env))
    monkeypatch.setattr(runtime_mod, "is_port_available", lambda *args, **kwargs: True)
    monkeypatch.setattr(helper_runtime_mod, "wait_http_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper_runtime_mod, "stream_lines", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper_runtime_mod, "terminate_process", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_mod, "terminate_process", lambda *args, **kwargs: None)
    if popen_factory is not None:
        monkeypatch.setattr(helper_runtime_mod.subprocess, "Popen", popen_factory)


def test_run_integrated_serve_starts_helpers_and_cleans_them_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_stack.entrypoints._helper_runtime as helper_runtime_mod
    import agentic_stack.entrypoints.vllm._runtime as runtime_mod

    seen: dict[str, object] = {}
    popen_calls: list[dict[str, object]] = []
    terminate_calls: list[str] = []

    def _fake_upstream_main() -> None:
        app = api_server_mod.build_app(SimpleNamespace(), ["generate"])
        seen["mcp_url"] = app.state.agentic_stack.runtime_config.mcp_builtin_runtime_url
        seen["argv"] = list(sys.argv)
        raise SystemExit(0)

    api_server_mod = _install_fake_vllm(monkeypatch, upstream_main=_fake_upstream_main)

    def _fake_popen(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        _ = args
        proc = _FakeProc(poll_code=None)
        popen_calls.append(
            {
                "cmd": [str(part) for part in cmd],
                "cwd": kwargs.get("cwd"),
                "env": kwargs.get("env"),
                "proc": proc,
            }
        )
        return proc

    _patch_runtime_dependencies(
        monkeypatch,
        env=EnvSource(environ={}),
        popen_factory=_fake_popen,
    )
    monkeypatch.setattr(
        runtime_mod,
        "build_code_interpreter_spawn_spec",
        lambda runtime_config, *, error_factory, error_prefix: SpawnCodeInterpreterSpec(
            cmd=["code-interpreter-server", "--port", "5970"],
            cwd=Path("/tmp"),
            port=5970,
            workers=0,
            ready_url="http://localhost:5970/health",
        ),
    )
    monkeypatch.setattr(
        helper_runtime_mod,
        "terminate_process",
        lambda proc, *, name, timeout_s=10.0: terminate_calls.append(name),
    )
    monkeypatch.setattr(
        runtime_mod,
        "terminate_process",
        lambda proc, *, name, timeout_s=10.0: terminate_calls.append(name),
    )

    spec = IntegratedServeSpec(
        vllm_args=["serve", "model", "--port", "8005"],
        code_interpreter_mode="spawn",
        code_interpreter_port=5970,
        code_interpreter_workers=0,
        code_interpreter_startup_timeout_s=30.0,
        mcp_config_path="/tmp/mcp.json",
        mcp_port=6101,
    )

    code = run_integrated_serve(spec)

    assert code == 0
    assert seen["argv"] == ["vllm", "serve", "model", "--port", "8005"]
    assert seen["mcp_url"] == "http://127.0.0.1:6101"
    assert len(popen_calls) == 2
    assert popen_calls[0]["cmd"] == ["code-interpreter-server", "--port", "5970"]
    assert popen_calls[0]["cwd"] == "/tmp"
    assert popen_calls[1]["cmd"][:4] == [
        sys.executable,
        "-m",
        "uvicorn",
        "agentic_stack.entrypoints.mcp_runtime:app",
    ]
    assert popen_calls[1]["env"]["VR_MCP_CONFIG_PATH"] == "/tmp/mcp.json"
    assert terminate_calls == ["mcp-runtime", "code-interpreter"]


def test_run_integrated_serve_cli_mcp_port_sets_runtime_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    popen_calls: list[dict[str, object]] = []

    def _fake_upstream_main() -> None:
        app = api_server_mod.build_app(SimpleNamespace(), ["generate"])
        seen["mcp_url"] = app.state.agentic_stack.runtime_config.mcp_builtin_runtime_url
        raise SystemExit(0)

    api_server_mod = _install_fake_vllm(monkeypatch, upstream_main=_fake_upstream_main)

    def _fake_popen(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        _ = args
        proc = _FakeProc(poll_code=None)
        popen_calls.append(
            {
                "cmd": [str(part) for part in cmd],
                "env": kwargs.get("env"),
            }
        )
        return proc

    _patch_runtime_dependencies(
        monkeypatch,
        env=EnvSource(environ={}),
        popen_factory=_fake_popen,
    )

    spec = IntegratedServeSpec(
        vllm_args=["serve", "model", "--port", "8005"],
        code_interpreter_mode="disabled",
        code_interpreter_port=5970,
        code_interpreter_workers=0,
        code_interpreter_startup_timeout_s=30.0,
        mcp_config_path="/tmp/mcp.json",
        mcp_port=6201,
    )

    code = run_integrated_serve(spec)

    assert code == 0
    assert seen["mcp_url"] == "http://127.0.0.1:6201"
    assert popen_calls[0]["cmd"][:8] == [
        sys.executable,
        "-m",
        "uvicorn",
        "agentic_stack.entrypoints.mcp_runtime:app",
        "--host",
        "127.0.0.1",
        "--port",
        "6201",
    ]


def test_run_integrated_serve_web_search_profile_spawns_builtin_mcp_runtime_without_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    popen_calls: list[dict[str, object]] = []

    def _fake_upstream_main() -> None:
        app = api_server_mod.build_app(SimpleNamespace(), ["generate"])
        seen["mcp_url"] = app.state.agentic_stack.runtime_config.mcp_builtin_runtime_url
        raise SystemExit(0)

    api_server_mod = _install_fake_vllm(monkeypatch, upstream_main=_fake_upstream_main)

    def _fake_popen(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        _ = args
        proc = _FakeProc(poll_code=None)
        popen_calls.append(
            {
                "cmd": [str(part) for part in cmd],
                "env": kwargs.get("env"),
            }
        )
        return proc

    _patch_runtime_dependencies(
        monkeypatch,
        env=EnvSource(environ={}),
        popen_factory=_fake_popen,
    )

    spec = IntegratedServeSpec(
        vllm_args=["serve", "model", "--port", "8005"],
        web_search_profile="duckduckgo_plus_fetch",
        code_interpreter_mode="disabled",
        code_interpreter_port=5970,
        code_interpreter_workers=0,
        code_interpreter_startup_timeout_s=30.0,
        mcp_config_path=None,
        mcp_port=None,
    )

    code = run_integrated_serve(spec)

    assert code == 0
    assert seen["mcp_url"] == "http://127.0.0.1:5981"
    assert popen_calls[0]["cmd"][:8] == [
        sys.executable,
        "-m",
        "uvicorn",
        "agentic_stack.entrypoints.mcp_runtime:app",
        "--host",
        "127.0.0.1",
        "--port",
        "5981",
    ]
    assert popen_calls[0]["env"]["VR_WEB_SEARCH_PROFILE"] == "duckduckgo_plus_fetch"
    assert "VR_MCP_CONFIG_PATH" not in popen_calls[0]["env"]


def test_run_integrated_serve_validates_external_code_interpreter_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_stack.entrypoints._helper_runtime as helper_runtime_mod

    ready_calls: list[dict[str, object]] = []

    def _fake_upstream_main() -> None:
        raise SystemExit(0)

    _install_fake_vllm(monkeypatch, upstream_main=_fake_upstream_main)
    _patch_runtime_dependencies(
        monkeypatch,
        env=EnvSource(environ={}),
    )
    monkeypatch.setattr(
        helper_runtime_mod,
        "wait_http_ready",
        lambda **kwargs: ready_calls.append(kwargs),
    )

    spec = IntegratedServeSpec(
        vllm_args=["serve", "model"],
        code_interpreter_mode="external",
        code_interpreter_port=6123,
        code_interpreter_workers=0,
        code_interpreter_startup_timeout_s=45.0,
    )

    code = run_integrated_serve(spec)

    assert code == 0
    assert len(ready_calls) == 1
    assert ready_calls[0]["name"] == "code-interpreter"
    assert ready_calls[0]["url"] == "http://localhost:6123/health"
    assert ready_calls[0]["timeout_s"] == 45.0


def test_run_integrated_serve_cleans_spawned_helpers_on_interrupt_during_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_stack.entrypoints._helper_runtime as helper_runtime_mod
    import agentic_stack.entrypoints.vllm._runtime as runtime_mod

    terminate_calls: list[str] = []

    def _fake_upstream_main() -> None:
        raise AssertionError("upstream main should not run")

    _install_fake_vllm(monkeypatch, upstream_main=_fake_upstream_main)

    def _fake_popen(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        _ = args
        _ = cmd
        _ = kwargs
        return _FakeProc(poll_code=None)

    _patch_runtime_dependencies(
        monkeypatch,
        env=EnvSource(environ={}),
        popen_factory=_fake_popen,
    )
    monkeypatch.setattr(
        runtime_mod,
        "build_code_interpreter_spawn_spec",
        lambda runtime_config, *, error_factory, error_prefix: SpawnCodeInterpreterSpec(
            cmd=["code-interpreter-server", "--port", "5970"],
            cwd=Path("/tmp"),
            port=5970,
            workers=0,
            ready_url="http://localhost:5970/health",
        ),
    )
    monkeypatch.setattr(
        helper_runtime_mod,
        "terminate_process",
        lambda proc, *, name, timeout_s=10.0: terminate_calls.append(name),
    )
    monkeypatch.setattr(
        runtime_mod,
        "terminate_process",
        lambda proc, *, name, timeout_s=10.0: terminate_calls.append(name),
    )
    monkeypatch.setattr(
        runtime_mod,
        "wait_for_code_interpreter_ready",
        lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    spec = IntegratedServeSpec(
        vllm_args=["serve", "model"],
        code_interpreter_mode="spawn",
        code_interpreter_port=5970,
        code_interpreter_workers=0,
        code_interpreter_startup_timeout_s=30.0,
    )

    code = run_integrated_serve(spec)

    assert code == 130
    assert terminate_calls == ["code-interpreter"]


def test_run_integrated_serve_rejects_mcp_port_without_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_upstream_main() -> None:
        raise AssertionError("upstream main should not run")

    _install_fake_vllm(monkeypatch, upstream_main=_fake_upstream_main)
    _patch_runtime_dependencies(
        monkeypatch,
        env=EnvSource(environ={}),
    )

    spec = IntegratedServeSpec(
        vllm_args=["serve", "model"],
        code_interpreter_mode="disabled",
        code_interpreter_port=5970,
        code_interpreter_workers=0,
        code_interpreter_startup_timeout_s=30.0,
        mcp_port=6201,
    )

    code = run_integrated_serve(spec)

    assert code == 2


def test_helper_watchdog_terminates_other_helpers_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_stack.entrypoints.vllm._runtime as runtime_mod

    terminated: list[str] = []
    exit_codes: list[int] = []
    exited = threading.Event()

    def _fake_exit(code: int) -> None:
        exit_codes.append(code)
        exited.set()

    monkeypatch.setattr(
        runtime_mod,
        "terminate_process",
        lambda proc, *, name, timeout_s=10.0: terminated.append(name),
    )
    monkeypatch.setattr(runtime_mod.os, "_exit", _fake_exit)

    stop_event = threading.Event()
    thread = _start_helper_watchdog(
        procs=[
            HelperProcess(name="code-interpreter", proc=_FakeProc(poll_code=7)),
            HelperProcess(name="mcp-runtime", proc=_FakeProc(poll_code=None)),
        ],
        stop_event=stop_event,
    )

    assert exited.wait(timeout=2.0)
    stop_event.set()
    thread.join(timeout=1.0)
    assert exit_codes == [7]
    assert terminated == ["mcp-runtime"]


@pytest.mark.asyncio
async def test_shared_http_client_propagates_request_id_from_context() -> None:
    seen_headers: dict[str, str] = {}

    async def _handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(200, json={"ok": True})

    client = get_async_client()
    client._transport = httpx.MockTransport(_handler)  # type: ignore[attr-defined]
    token = CURRENT_REQUEST_ID.set("req_123")
    try:
        response = await client.get("http://example.test/health")
    finally:
        CURRENT_REQUEST_ID.reset(token)
        await client.aclose()

    assert response.status_code == 200
    assert seen_headers["x-request-id"] == "req_123"
