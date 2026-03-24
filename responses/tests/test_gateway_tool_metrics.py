from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from prometheus_client import REGISTRY

from agentic_stack.configs.builders import build_runtime_config_for_standalone
from agentic_stack.configs.sources import EnvSource
from agentic_stack.entrypoints import llm as mock_llm
from agentic_stack.entrypoints.gateway._app import (
    activate_gateway_runtime,
    augment_standalone_gateway_app,
)


def _sample_value(metric_name: str, labels: dict[str, str]) -> float:
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == metric_name and sample.labels == labels:
                return float(sample.value)
    return 0.0


@pytest.fixture
def gateway_metrics_app() -> FastAPI:
    app = FastAPI(title="VR Gateway Metrics (test)")
    augment_standalone_gateway_app(
        app,
        include_upstream_proxy=False,
        include_metrics_route=True,
        include_cors=False,
        customize_openapi=False,
    )
    activate_gateway_runtime(
        app,
        runtime_config=build_runtime_config_for_standalone(
            env=EnvSource(environ={"VR_LLM_API_BASE": "http://mock/v1"})
        ),
    )
    return app


@pytest.fixture
async def gateway_metrics_client(
    gateway_metrics_app: FastAPI,
) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=gateway_metrics_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        yield client


@pytest.mark.anyio
async def test_code_interpreter_tool_loop_updates_tool_metrics(
    patched_gateway_clients,
    gateway_metrics_client: httpx.AsyncClient,
    cassette_replayer_factory,
) -> None:
    requested_labels = {"tool_type": "code_interpreter"}
    executed_labels = {"tool_type": "code_interpreter"}
    before_requested = _sample_value(
        "agentic_stack_tool_calls_requested_total",
        requested_labels,
    )
    before_executed = _sample_value(
        "agentic_stack_tool_calls_executed_total",
        executed_labels,
    )
    before_duration_count = _sample_value(
        "agentic_stack_tool_execution_duration_seconds_count",
        executed_labels,
    )
    before_errors = _sample_value(
        "agentic_stack_tool_errors_total",
        executed_labels,
    )

    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "vllm-code_interpreter-step1-stream.yaml",
        "vllm-code_interpreter-step2-stream.yaml",
    )

    async with gateway_metrics_client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": True,
            "include": ["code_interpreter_call.outputs"],
            "input": [
                {
                    "role": "user",
                    "content": "You MUST call the code_interpreter tool now. Execute: 2+2.",
                }
            ],
            "tools": [{"type": "code_interpreter"}],
            "tool_choice": "auto",
        },
    ) as resp:
        assert resp.status_code == 200
        _ = await resp.aread()

    metrics_resp = await gateway_metrics_client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert (
        'agentic_stack_tool_calls_requested_total{tool_type="code_interpreter"}'
        in metrics_resp.text
    )
    assert (
        'agentic_stack_tool_calls_executed_total{tool_type="code_interpreter"}'
        in metrics_resp.text
    )

    assert (
        _sample_value("agentic_stack_tool_calls_requested_total", requested_labels)
        == before_requested + 1
    )
    assert (
        _sample_value("agentic_stack_tool_calls_executed_total", executed_labels)
        == before_executed + 1
    )
    assert (
        _sample_value("agentic_stack_tool_execution_duration_seconds_count", executed_labels)
        == before_duration_count + 1
    )
    assert _sample_value("agentic_stack_tool_errors_total", executed_labels) == before_errors
