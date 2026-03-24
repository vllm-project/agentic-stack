from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, Request, Response

from agentic_stack.configs.builders import build_runtime_config_for_standalone
from agentic_stack.configs.sources import EnvSource
from agentic_stack.entrypoints._state import VRAppState, VRRequestState
from agentic_stack.entrypoints.gateway._app import (
    _finalize_gateway_response,
    activate_gateway_runtime,
    augment_standalone_gateway_app,
)
from agentic_stack.types.api import UserAgent


def test_standalone_api_imports_cleanly_in_fresh_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import agentic_stack.entrypoints.api",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_standalone_api_import_does_not_build_runtime_config_from_invalid_env() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import agentic_stack.entrypoints.api",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "VR_WEB_SEARCH_PROFILE": "missing_profile"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_mcp_runtime_import_does_not_build_runtime_config_from_invalid_env() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import agentic_stack.entrypoints.mcp_runtime",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "VR_WEB_SEARCH_PROFILE": "missing_profile"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_setup_logger_sinks_falls_back_when_enqueue_is_unavailable(monkeypatch, capsys) -> None:
    import agentic_stack.utils.logging as logging_mod

    configure_calls: list[list[dict[str, object]]] = []

    def _fake_configure(*, handlers):  # type: ignore[no-untyped-def]
        configure_calls.append(list(handlers))
        if len(configure_calls) == 1:
            raise PermissionError("semlock unavailable")

    monkeypatch.setattr(logging_mod.logger, "remove", lambda *args, **kwargs: None)
    monkeypatch.setattr(logging_mod.logger, "level", lambda *args, **kwargs: None)
    monkeypatch.setattr(logging_mod.logger, "configure", _fake_configure)

    logging_mod.setup_logger_sinks(None)

    assert len(configure_calls) == 2
    assert all(handler["enqueue"] is True for handler in configure_calls[0])
    assert all(handler["enqueue"] is False for handler in configure_calls[1])
    assert "falling back to synchronous Loguru handlers" in capsys.readouterr().err


def test_openapi_customization_stays_lazy_for_late_routes() -> None:
    app = FastAPI()
    runtime_config = build_runtime_config_for_standalone(env=EnvSource(environ={}))
    augment_standalone_gateway_app(
        app,
        include_upstream_proxy=False,
        include_metrics_route=False,
        include_cors=False,
        customize_openapi=True,
    )
    activate_gateway_runtime(app, runtime_config=runtime_config)

    assert app.openapi_schema is None

    @app.get("/late")
    async def late_route() -> dict[str, bool]:
        return {"ok": True}

    schema = app.openapi()

    assert "/late" in schema["paths"]
    assert schema["components"]["securitySchemes"]["Authentication"]["scheme"] == "bearer"


def test_finalize_gateway_response_matches_templated_routes_for_overhead_logging(
    monkeypatch,
) -> None:
    scope = {
        "type": "http",
        "app": FastAPI(),
        "method": "GET",
        "path": "/v1/responses/resp_123",
        "raw_path": b"/v1/responses/resp_123",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "scheme": "http",
        "root_path": "",
        "route": SimpleNamespace(path="/v1/responses/{response_id}"),
    }
    request = Request(scope)
    request.app.state.agentic_stack = VRAppState(
        runtime_config=build_runtime_config_for_standalone(
            env=EnvSource(environ={"VR_LOG_TIMINGS": "1"})
        )
    )
    request.state.agentic_stack = VRRequestState(
        id="req_123",
        user_agent=UserAgent(is_browser=False, agent=""),
    )
    request.state.agentic_stack.model_start_time = (
        request.state.agentic_stack.request_start_time + 0.01
    )
    request.state.agentic_stack.timing["normalize"] = 0.001
    response = Response()
    captured_logs: list[str] = []

    monkeypatch.setattr(
        "agentic_stack.entrypoints.gateway._app.logger.info",
        lambda message: captured_logs.append(message),
    )

    _finalize_gateway_response(
        request,
        response,
        request_id="req_123",
        is_internal_upstream=False,
        overhead_log_routes={"/v1/responses/{response_id}"},
    )

    assert response.headers["x-request-id"] == "req_123"
    assert any("Total overhead" in message for message in captured_logs)
