from __future__ import annotations

from dataclasses import dataclass

from agentic_stack.configs.runtime import CodeInterpreterMode


@dataclass(frozen=True, slots=True)
class RuntimeDefaults:
    llm_api_base: str = "http://localhost:8080/v1"
    web_search_profile: str | None = None
    host: str = "0.0.0.0"
    port: int = 5969
    workers: int = 1
    max_concurrency: int = 300
    log_dir: str = "logs"
    log_timings: bool = False
    log_model_messages: bool = False

    code_interpreter_mode: CodeInterpreterMode = "spawn"
    code_interpreter_port: int = 5970
    code_interpreter_workers: int = 0
    code_interpreter_dev_bun_fallback: bool = False
    code_interpreter_startup_timeout_s: float = 10 * 60.0
    upstream_ready_timeout_s: float = 30 * 60.0
    upstream_ready_interval_s: float = 5.0

    metrics_enabled: bool = True
    metrics_path: str = "/metrics"

    tracing_enabled: bool = False
    otel_service_name: str = "agentic-stacks"
    tracing_sample_ratio: float = 0.01
    opentelemetry_host: str = "otel-collector"
    opentelemetry_port: int = 4317

    mcp_request_remote_enabled: bool = True
    mcp_request_remote_url_checks: bool = True
    mcp_hosted_startup_timeout_sec: float = 10.0
    mcp_hosted_tool_timeout_sec: float = 60.0
    mcp_builtin_runtime_url: str = "http://127.0.0.1:5981"

    db_path: str = "sqlite+aiosqlite:///agentic_stack.db"
    redis_host: str = "localhost"
    redis_port: int = 6379
    response_store_cache: bool = False
    response_store_cache_ttl_seconds: int = 3600


RUNTIME_DEFAULTS = RuntimeDefaults()
