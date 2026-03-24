from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RuntimeMode = Literal["standalone", "supervisor", "integrated", "mock_llm"]
CodeInterpreterMode = Literal["spawn", "external", "disabled"]

INTERNAL_UPSTREAM_HEADER_NAME = "x-vr-internal-upstream"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    runtime_mode: RuntimeMode

    gateway_host: str
    gateway_port: int
    gateway_workers: int
    gateway_max_concurrency: int
    log_dir: str

    llm_api_base: str
    openai_api_key: str | None
    web_search_profile: str | None

    code_interpreter_mode: CodeInterpreterMode
    code_interpreter_port: int | None
    code_interpreter_workers: int | None
    pyodide_cache_dir: str | None
    code_interpreter_dev_bun_fallback: bool
    code_interpreter_startup_timeout_s: float

    upstream_ready_timeout_s: float
    upstream_ready_interval_s: float

    mcp_config_path: str | None
    mcp_builtin_runtime_url: str | None
    mcp_request_remote_enabled: bool
    mcp_request_remote_url_checks: bool
    mcp_hosted_startup_timeout_sec: float
    mcp_hosted_tool_timeout_sec: float

    metrics_enabled: bool
    metrics_path: str
    log_timings: bool
    log_model_messages: bool

    tracing_enabled: bool
    otel_service_name: str
    tracing_sample_ratio: float
    opentelemetry_host: str
    opentelemetry_port: int

    db_path: str
    redis_host: str
    redis_port: int
    response_store_cache: bool
    response_store_cache_ttl_seconds: int

    internal_upstream_header_name: str = INTERNAL_UPSTREAM_HEADER_NAME

    @property
    def db_dialect(self) -> Literal["sqlite", "postgresql"]:
        if self.db_path.startswith("sqlite"):
            return "sqlite"
        if self.db_path.startswith("postgresql"):
            return "postgresql"
        raise ValueError(f'`db_path` "{self.db_path}" has an invalid dialect.')
