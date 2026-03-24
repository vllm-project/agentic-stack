from __future__ import annotations

from dataclasses import dataclass

from agentic_stack.configs.runtime import RuntimeConfig
from agentic_stack.entrypoints._helper_runtime import (
    CodeInterpreterSpec,
    DisabledCodeInterpreterSpec,
    ExternalCodeInterpreterSpec,
    McpRuntimeSpec,
    build_code_interpreter_spawn_spec,
    build_mcp_runtime_spec,
)


class ServeSpecError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = int(exit_code)


@dataclass(frozen=True, slots=True)
class GatewaySpec:
    host: str
    port: int
    workers: int


@dataclass(frozen=True, slots=True)
class TimeoutSpec:
    upstream_ready_timeout_s: float
    upstream_ready_interval_s: float
    code_interpreter_startup_timeout_s: float


@dataclass(frozen=True, slots=True)
class MetricsSpec:
    enabled: bool


@dataclass(frozen=True, slots=True)
class ExternalUpstreamSpec:
    base_url: str
    ready_url: str
    headers: dict[str, str] | None


@dataclass(frozen=True, slots=True)
class ServeSpec:
    runtime_config: RuntimeConfig
    notices: list[str]
    gateway: GatewaySpec
    mcp_runtime: McpRuntimeSpec | None
    upstream: ExternalUpstreamSpec
    code_interpreter: CodeInterpreterSpec
    code_interpreter_workers: int
    metrics: MetricsSpec
    timeouts: TimeoutSpec


def build_serve_spec(runtime_config: RuntimeConfig) -> ServeSpec:
    openai_key = (runtime_config.openai_api_key or "").strip()
    upstream_headers = None if not openai_key else {"Authorization": f"Bearer {openai_key}"}
    upstream_base_url = runtime_config.llm_api_base

    if runtime_config.code_interpreter_mode == "disabled":
        code_interpreter: CodeInterpreterSpec = DisabledCodeInterpreterSpec()
    elif runtime_config.code_interpreter_mode == "external":
        if runtime_config.code_interpreter_port is None:
            raise ServeSpecError(
                "[serve] error: external code interpreter mode requires a configured port.",
                exit_code=2,
            )
        code_interpreter = ExternalCodeInterpreterSpec(
            port=runtime_config.code_interpreter_port,
            ready_url=f"http://localhost:{runtime_config.code_interpreter_port}/health",
        )
    elif runtime_config.code_interpreter_mode == "spawn":
        if runtime_config.code_interpreter_port is None:
            raise ServeSpecError(
                "[serve] error: spawn code interpreter mode requires a configured port.",
                exit_code=2,
            )
        code_interpreter = build_code_interpreter_spawn_spec(
            runtime_config,
            error_factory=ServeSpecError,
            error_prefix="[serve]",
        )
    else:  # pragma: no cover - defensive
        raise AssertionError(f"unreachable ci mode: {runtime_config.code_interpreter_mode!r}")

    return ServeSpec(
        runtime_config=runtime_config,
        notices=[],
        gateway=GatewaySpec(
            host=runtime_config.gateway_host,
            port=runtime_config.gateway_port,
            workers=runtime_config.gateway_workers,
        ),
        mcp_runtime=build_mcp_runtime_spec(
            runtime_config,
            error_factory=ServeSpecError,
            error_prefix="[serve]",
        ),
        upstream=ExternalUpstreamSpec(
            base_url=upstream_base_url,
            ready_url=f"{upstream_base_url}/models",
            headers=upstream_headers,
        ),
        code_interpreter=code_interpreter,
        code_interpreter_workers=int(runtime_config.code_interpreter_workers or 0),
        metrics=MetricsSpec(enabled=runtime_config.metrics_enabled),
        timeouts=TimeoutSpec(
            upstream_ready_timeout_s=runtime_config.upstream_ready_timeout_s,
            upstream_ready_interval_s=runtime_config.upstream_ready_interval_s,
            code_interpreter_startup_timeout_s=runtime_config.code_interpreter_startup_timeout_s,
        ),
    )
