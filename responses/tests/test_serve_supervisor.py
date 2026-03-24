from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from agentic_stack.configs.builders import build_runtime_config_for_supervisor
from agentic_stack.configs.sources import EnvSource
from agentic_stack.entrypoints._serve._runtime import run_serve_spec
from agentic_stack.entrypoints._serve._spec import (
    DisabledCodeInterpreterSpec,
    ExternalUpstreamSpec,
    GatewaySpec,
    McpRuntimeSpec,
    MetricsSpec,
    ServeSpec,
    TimeoutSpec,
)


@dataclass
class _FakeProc:
    poll_code: int | None
    stdout: object | None = None

    def poll(self) -> int | None:
        return self.poll_code

    def wait(self) -> int:
        return 0 if self.poll_code is None else self.poll_code


class _FakeStore:
    async def ensure_schema(self) -> None:
        return None


def _base_spec(
    *,
    mcp_runtime: McpRuntimeSpec | None,
    web_search_profile: str | None = None,
    mcp_config_path: str | None = None,
) -> ServeSpec:
    runtime_config = build_runtime_config_for_supervisor(
        args=SimpleNamespace(
            upstream="http://127.0.0.1:8457/v1",
            gateway_host="127.0.0.1",
            gateway_port=5969,
            gateway_workers=1,
            web_search_profile=web_search_profile,
            code_interpreter="disabled",
            code_interpreter_port=None,
            code_interpreter_workers=None,
            mcp_config=mcp_config_path if mcp_config_path is not None else None,
            mcp_port=None if mcp_runtime is None else mcp_runtime.port,
        ),
        env=EnvSource(environ={}),
    )
    return ServeSpec(
        runtime_config=runtime_config,
        notices=[],
        gateway=GatewaySpec(host="127.0.0.1", port=5969, workers=1),
        mcp_runtime=mcp_runtime,
        upstream=ExternalUpstreamSpec(
            base_url="http://127.0.0.1:8457/v1",
            ready_url="http://127.0.0.1:8457/v1/models",
            headers=None,
        ),
        code_interpreter=DisabledCodeInterpreterSpec(),
        code_interpreter_workers=0,
        metrics=MetricsSpec(enabled=False),
        timeouts=TimeoutSpec(
            upstream_ready_timeout_s=10.0,
            upstream_ready_interval_s=1.0,
            code_interpreter_startup_timeout_s=10.0,
        ),
    )


def _patch_supervisor_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    import agentic_stack.entrypoints._helper_runtime as helper_runtime_module
    import agentic_stack.entrypoints._serve._runtime as supervisor_module
    import agentic_stack.responses_core.store as store_module

    popen_calls: list[dict[str, object]] = []

    def _fake_popen(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        _ = args
        cmd_list = [str(c) for c in cmd]
        is_mcp_runtime = "agentic_stack.entrypoints.mcp_runtime:app" in cmd_list
        proc = _FakeProc(poll_code=None if is_mcp_runtime else 0)
        popen_calls.append(
            {
                "cmd": cmd_list,
                "env": kwargs.get("env"),
                "is_mcp_runtime": is_mcp_runtime,
                "proc": proc,
            }
        )
        return proc

    monkeypatch.setattr(store_module, "get_default_response_store", lambda: _FakeStore())
    monkeypatch.setattr(supervisor_module, "wait_http_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper_runtime_module, "wait_http_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(supervisor_module, "is_port_available", lambda *args, **kwargs: True)
    monkeypatch.setattr(helper_runtime_module.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(helper_runtime_module, "terminate_process", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper_runtime_module, "stream_lines", lambda *args, **kwargs: None)

    return popen_calls


def test_run_serve_spec_without_mcp_runtime_does_not_spawn_or_inject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popen_calls = _patch_supervisor_runtime_dependencies(monkeypatch)
    spec = _base_spec(mcp_runtime=None)

    code = run_serve_spec(spec)

    assert code == 0
    assert len(popen_calls) == 1
    assert popen_calls[0]["is_mcp_runtime"] is False
    gateway_env = popen_calls[0]["env"]
    assert isinstance(gateway_env, dict)
    assert "VR_RESPONSES_RUNTIME_MODE" not in gateway_env


def test_run_serve_spec_with_mcp_runtime_spawns_and_injects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popen_calls = _patch_supervisor_runtime_dependencies(monkeypatch)
    spec = _base_spec(
        mcp_runtime=McpRuntimeSpec(
            host="127.0.0.1",
            port=5981,
            ready_url="http://127.0.0.1:5981/health",
        ),
        web_search_profile="exa_mcp",
        mcp_config_path="/tmp/mcp.json",
    )

    code = run_serve_spec(spec)

    assert code == 0
    assert len(popen_calls) == 2
    assert any(call["is_mcp_runtime"] is True for call in popen_calls)
    mcp_runtime_calls = [call for call in popen_calls if call["is_mcp_runtime"] is True]
    assert len(mcp_runtime_calls) == 1
    assert mcp_runtime_calls[0]["env"]["VR_MCP_CONFIG_PATH"] == "/tmp/mcp.json"

    gateway_calls = [call for call in popen_calls if call["is_mcp_runtime"] is False]
    assert len(gateway_calls) == 1
    gateway_env = gateway_calls[0]["env"]
    assert isinstance(gateway_env, dict)
    assert gateway_env.get("VR_WEB_SEARCH_PROFILE") == "exa_mcp"
    assert gateway_env.get("VR_MCP_CONFIG_PATH") == "/tmp/mcp.json"
    assert gateway_env.get("VR_MCP_BUILTIN_RUNTIME_URL") == "http://127.0.0.1:5981"
    assert "VR_RESPONSES_RUNTIME_MODE" not in gateway_env


def test_run_serve_spec_with_web_search_profile_spawns_builtin_mcp_runtime_without_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popen_calls = _patch_supervisor_runtime_dependencies(monkeypatch)
    spec = _base_spec(
        mcp_runtime=McpRuntimeSpec(
            host="127.0.0.1",
            port=5981,
            ready_url="http://127.0.0.1:5981/health",
        ),
        web_search_profile="duckduckgo_plus_fetch",
        mcp_config_path=None,
    )

    code = run_serve_spec(spec)

    assert code == 0
    assert len(popen_calls) == 2
    mcp_runtime_calls = [call for call in popen_calls if call["is_mcp_runtime"] is True]
    assert len(mcp_runtime_calls) == 1
    assert "VR_MCP_CONFIG_PATH" not in mcp_runtime_calls[0]["env"]
    assert mcp_runtime_calls[0]["env"]["VR_WEB_SEARCH_PROFILE"] == "duckduckgo_plus_fetch"

    gateway_calls = [call for call in popen_calls if call["is_mcp_runtime"] is False]
    assert len(gateway_calls) == 1
    gateway_env = gateway_calls[0]["env"]
    assert isinstance(gateway_env, dict)
    assert gateway_env.get("VR_WEB_SEARCH_PROFILE") == "duckduckgo_plus_fetch"
    assert "VR_MCP_CONFIG_PATH" not in gateway_env
    assert gateway_env.get("VR_MCP_BUILTIN_RUNTIME_URL") == "http://127.0.0.1:5981"
