from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from agentic_stack.configs.defaults import RUNTIME_DEFAULTS
from agentic_stack.configs.runtime import CodeInterpreterMode, RuntimeConfig, RuntimeMode
from agentic_stack.configs.sources import EnvSource
from agentic_stack.configs.startup import (
    format_web_search_profile_choices,
    supervisor_responses_cli_labels,
    supervisor_responses_cli_raw_values,
    validate_responses_cli_args,
)
from agentic_stack.tools.ids import WEB_SEARCH_TOOL
from agentic_stack.tools.profile_resolution import (
    profiled_builtin_requires_mcp,
    validate_profiled_builtin_profile,
)
from agentic_stack.utils.urls import is_ready_url_host


class RuntimeConfigError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = int(exit_code)


def loopback_url_from_port(port: int) -> str:
    return f"http://127.0.0.1:{int(port)}"


def resolve_secret(*, env: EnvSource, env_key: str, default: str | None = None) -> str | None:
    raw, is_set = env.get(env_key)
    if not is_set or raw is None:
        return default
    value = raw.strip()
    return value or None


def resolve_code_interpreter_mode(raw: str) -> CodeInterpreterMode:
    normalized = raw.strip().lower()
    if normalized in {"spawn", "external", "disabled"}:
        return normalized
    raise ValueError(f"invalid code interpreter mode: {raw!r}")


def build_common_runtime_config(
    *,
    env: EnvSource,
    runtime_mode: RuntimeMode,
    gateway_host: str,
    gateway_port: int,
    gateway_workers: int,
    llm_api_base: str,
    web_search_profile: str | None,
    code_interpreter_mode: CodeInterpreterMode,
    code_interpreter_port: int | None,
    code_interpreter_workers: int | None,
    code_interpreter_startup_timeout_s: float = RUNTIME_DEFAULTS.code_interpreter_startup_timeout_s,
    upstream_ready_timeout_s: float = RUNTIME_DEFAULTS.upstream_ready_timeout_s,
    upstream_ready_interval_s: float = RUNTIME_DEFAULTS.upstream_ready_interval_s,
    mcp_config_path: str | None,
    mcp_builtin_runtime_url: str | None,
) -> RuntimeConfig:
    try:
        validate_profiled_builtin_profile(
            tool_type=WEB_SEARCH_TOOL,
            profile_id=web_search_profile,
        )
    except ValueError as exc:
        raise RuntimeConfigError(
            "[runtime] error: "
            f"unknown --web-search-profile={web_search_profile!r}; "
            f"expected one of: {format_web_search_profile_choices()}."
        ) from exc
    pyodide_cache_dir = env.get_optional_str("VR_PYODIDE_CACHE_DIR")
    if pyodide_cache_dir is None:
        xdg_cache_home = env.get_optional_str("XDG_CACHE_HOME")
        base_dir = Path(xdg_cache_home) if xdg_cache_home is not None else Path.home() / ".cache"
        pyodide_cache_dir = str(base_dir / "agentic-stacks" / "pyodide")

    resolved_mcp_config_path = mcp_config_path
    resolved_mcp_builtin_runtime_url = mcp_builtin_runtime_url
    requires_builtin_mcp = profiled_builtin_requires_mcp(
        tool_type=WEB_SEARCH_TOOL,
        profile_id=web_search_profile,
    )
    if (
        resolved_mcp_config_path is not None or requires_builtin_mcp
    ) and resolved_mcp_builtin_runtime_url is None:
        resolved_mcp_builtin_runtime_url = RUNTIME_DEFAULTS.mcp_builtin_runtime_url

    return RuntimeConfig(
        runtime_mode=runtime_mode,
        gateway_host=gateway_host,
        gateway_port=int(gateway_port),
        gateway_workers=int(gateway_workers),
        gateway_max_concurrency=env.get_int(
            "VR_MAX_CONCURRENCY",
            RUNTIME_DEFAULTS.max_concurrency,
        ),
        log_dir=env.get_str("VR_LOG_DIR", RUNTIME_DEFAULTS.log_dir),
        llm_api_base=llm_api_base.strip(),
        openai_api_key=resolve_secret(env=env, env_key="VR_OPENAI_API_KEY"),
        web_search_profile=web_search_profile,
        code_interpreter_mode=code_interpreter_mode,
        code_interpreter_port=code_interpreter_port,
        code_interpreter_workers=code_interpreter_workers,
        pyodide_cache_dir=pyodide_cache_dir,
        code_interpreter_dev_bun_fallback=env.get_bool(
            "VR_CODE_INTERPRETER_DEV_BUN_FALLBACK",
            RUNTIME_DEFAULTS.code_interpreter_dev_bun_fallback,
        ),
        code_interpreter_startup_timeout_s=float(code_interpreter_startup_timeout_s),
        upstream_ready_timeout_s=float(upstream_ready_timeout_s),
        upstream_ready_interval_s=float(upstream_ready_interval_s),
        mcp_config_path=resolved_mcp_config_path,
        mcp_builtin_runtime_url=resolved_mcp_builtin_runtime_url,
        mcp_request_remote_enabled=env.get_bool(
            "VR_MCP_REQUEST_REMOTE_ENABLED",
            RUNTIME_DEFAULTS.mcp_request_remote_enabled,
        ),
        mcp_request_remote_url_checks=env.get_bool(
            "VR_MCP_REQUEST_REMOTE_URL_CHECKS",
            RUNTIME_DEFAULTS.mcp_request_remote_url_checks,
        ),
        mcp_hosted_startup_timeout_sec=env.get_float(
            "VR_MCP_HOSTED_STARTUP_TIMEOUT_SEC",
            RUNTIME_DEFAULTS.mcp_hosted_startup_timeout_sec,
        ),
        mcp_hosted_tool_timeout_sec=env.get_float(
            "VR_MCP_HOSTED_TOOL_TIMEOUT_SEC",
            RUNTIME_DEFAULTS.mcp_hosted_tool_timeout_sec,
        ),
        metrics_enabled=env.get_bool("VR_METRICS_ENABLED", RUNTIME_DEFAULTS.metrics_enabled),
        metrics_path=env.get_str("VR_METRICS_PATH", RUNTIME_DEFAULTS.metrics_path),
        log_timings=env.get_bool("VR_LOG_TIMINGS", RUNTIME_DEFAULTS.log_timings),
        log_model_messages=env.get_bool(
            "VR_LOG_MODEL_MESSAGES",
            RUNTIME_DEFAULTS.log_model_messages,
        ),
        tracing_enabled=env.get_bool("VR_TRACING_ENABLED", RUNTIME_DEFAULTS.tracing_enabled),
        otel_service_name=env.get_str(
            "VR_OTEL_SERVICE_NAME",
            RUNTIME_DEFAULTS.otel_service_name,
        ),
        tracing_sample_ratio=env.get_float(
            "VR_TRACING_SAMPLE_RATIO",
            RUNTIME_DEFAULTS.tracing_sample_ratio,
        ),
        opentelemetry_host=env.get_str(
            "VR_OPENTELEMETRY_HOST",
            RUNTIME_DEFAULTS.opentelemetry_host,
        ),
        opentelemetry_port=env.get_int(
            "VR_OPENTELEMETRY_PORT",
            RUNTIME_DEFAULTS.opentelemetry_port,
        ),
        db_path=env.get_str("VR_DB_PATH", RUNTIME_DEFAULTS.db_path),
        redis_host=env.get_str("VR_REDIS_HOST", RUNTIME_DEFAULTS.redis_host),
        redis_port=env.get_int("VR_REDIS_PORT", RUNTIME_DEFAULTS.redis_port),
        response_store_cache=env.get_bool(
            "VR_RESPONSE_STORE_CACHE",
            RUNTIME_DEFAULTS.response_store_cache,
        ),
        response_store_cache_ttl_seconds=env.get_int(
            "VR_RESPONSE_STORE_CACHE_TTL_SECONDS",
            RUNTIME_DEFAULTS.response_store_cache_ttl_seconds,
        ),
    )


def build_runtime_config_for_standalone(
    *,
    env: EnvSource | None = None,
    runtime_mode: RuntimeMode = "standalone",
) -> RuntimeConfig:
    env = EnvSource.from_env() if env is None else env
    code_interpreter_mode = resolve_code_interpreter_mode(
        env.get_str("VR_CODE_INTERPRETER_MODE", RUNTIME_DEFAULTS.code_interpreter_mode)
    )
    code_interpreter_port = None
    code_interpreter_workers = None
    if code_interpreter_mode != "disabled":
        code_interpreter_port = env.get_int(
            "VR_CODE_INTERPRETER_PORT",
            RUNTIME_DEFAULTS.code_interpreter_port,
        )
        code_interpreter_workers = env.get_int(
            "VR_CODE_INTERPRETER_WORKERS",
            RUNTIME_DEFAULTS.code_interpreter_workers,
        )

    llm_api_base = env.get_str("VR_LLM_API_BASE", RUNTIME_DEFAULTS.llm_api_base).strip()
    return build_common_runtime_config(
        env=env,
        runtime_mode=runtime_mode,
        gateway_host=env.get_str("VR_HOST", RUNTIME_DEFAULTS.host),
        gateway_port=env.get_int("VR_PORT", RUNTIME_DEFAULTS.port),
        gateway_workers=env.get_int("VR_WORKERS", RUNTIME_DEFAULTS.workers),
        llm_api_base=llm_api_base,
        web_search_profile=env.get_optional_str(
            "VR_WEB_SEARCH_PROFILE", RUNTIME_DEFAULTS.web_search_profile
        ),
        code_interpreter_mode=code_interpreter_mode,
        code_interpreter_port=code_interpreter_port,
        code_interpreter_workers=code_interpreter_workers,
        code_interpreter_startup_timeout_s=env.get_float(
            "VR_CODE_INTERPRETER_STARTUP_TIMEOUT",
            RUNTIME_DEFAULTS.code_interpreter_startup_timeout_s,
        ),
        mcp_config_path=env.get_optional_str("VR_MCP_CONFIG_PATH"),
        mcp_builtin_runtime_url=env.get_optional_str("VR_MCP_BUILTIN_RUNTIME_URL"),
    )


def build_runtime_config_for_mock_llm(*, env: EnvSource | None = None) -> RuntimeConfig:
    return build_runtime_config_for_standalone(env=env, runtime_mode="mock_llm")


def build_runtime_config_for_supervisor(
    *,
    args: Namespace,
    env: EnvSource | None = None,
) -> RuntimeConfig:
    env = EnvSource.from_env() if env is None else env
    responses_cli = validate_responses_cli_args(
        raw_values=supervisor_responses_cli_raw_values(args),
        labels=supervisor_responses_cli_labels(),
        error_prefix="[serve]",
        error_factory=RuntimeConfigError,
    )
    upstream_arg = getattr(args, "upstream", None)
    if upstream_arg is not None:
        llm_api_base = str(upstream_arg)
    else:
        raise RuntimeConfigError(
            "[serve] error: no upstream configured. Provide `--upstream http://host:port/v1` with the exact upstream API base."
        )

    code_interpreter_mode = (
        RUNTIME_DEFAULTS.code_interpreter_mode
        if responses_cli.code_interpreter_mode is None
        else resolve_code_interpreter_mode(responses_cli.code_interpreter_mode)
    )
    code_interpreter_port = None
    code_interpreter_workers = None
    if code_interpreter_mode != "disabled":
        code_interpreter_port = (
            RUNTIME_DEFAULTS.code_interpreter_port
            if responses_cli.code_interpreter_port is None
            else int(responses_cli.code_interpreter_port)
        )
        code_interpreter_workers = (
            RUNTIME_DEFAULTS.code_interpreter_workers
            if responses_cli.code_interpreter_workers is None
            else int(responses_cli.code_interpreter_workers)
        )

    mcp_builtin_runtime_url = None
    if responses_cli.mcp_port is not None and (
        responses_cli.mcp_config_path is not None
        or profiled_builtin_requires_mcp(
            tool_type=WEB_SEARCH_TOOL,
            profile_id=responses_cli.web_search_profile,
        )
    ):
        mcp_builtin_runtime_url = loopback_url_from_port(int(responses_cli.mcp_port))

    gateway_host_arg = args.gateway_host
    gateway_port_arg = args.gateway_port
    gateway_workers_arg = args.gateway_workers

    return build_common_runtime_config(
        env=env,
        runtime_mode="supervisor",
        gateway_host=(
            RUNTIME_DEFAULTS.host if gateway_host_arg is None else str(gateway_host_arg)
        ),
        gateway_port=(
            RUNTIME_DEFAULTS.port if gateway_port_arg is None else int(gateway_port_arg)
        ),
        gateway_workers=(
            RUNTIME_DEFAULTS.workers if gateway_workers_arg is None else int(gateway_workers_arg)
        ),
        llm_api_base=llm_api_base,
        web_search_profile=responses_cli.web_search_profile,
        code_interpreter_mode=code_interpreter_mode,
        code_interpreter_port=code_interpreter_port,
        code_interpreter_workers=code_interpreter_workers,
        code_interpreter_startup_timeout_s=(
            RUNTIME_DEFAULTS.code_interpreter_startup_timeout_s
            if responses_cli.code_interpreter_startup_timeout_s is None
            else float(responses_cli.code_interpreter_startup_timeout_s)
        ),
        upstream_ready_timeout_s=(
            RUNTIME_DEFAULTS.upstream_ready_timeout_s
            if responses_cli.upstream_ready_timeout_s is None
            else float(responses_cli.upstream_ready_timeout_s)
        ),
        upstream_ready_interval_s=(
            RUNTIME_DEFAULTS.upstream_ready_interval_s
            if responses_cli.upstream_ready_interval_s is None
            else float(responses_cli.upstream_ready_interval_s)
        ),
        mcp_config_path=responses_cli.mcp_config_path,
        mcp_builtin_runtime_url=mcp_builtin_runtime_url,
    )


def build_runtime_config_for_integrated(
    *,
    env: EnvSource | None,
    host: str,
    port: int,
    web_search_profile: str | None,
    code_interpreter_mode: CodeInterpreterMode,
    code_interpreter_port: int,
    code_interpreter_workers: int,
    code_interpreter_startup_timeout_s: float,
    mcp_config_path: str | None,
    mcp_builtin_runtime_url: str | None,
) -> RuntimeConfig:
    env = EnvSource.from_env() if env is None else env
    effective_code_interpreter_port = None
    effective_code_interpreter_workers = None
    if code_interpreter_mode != "disabled":
        effective_code_interpreter_port = int(code_interpreter_port)
        effective_code_interpreter_workers = int(code_interpreter_workers)

    return build_common_runtime_config(
        env=env,
        runtime_mode="integrated",
        gateway_host=host,
        gateway_port=port,
        gateway_workers=1,
        llm_api_base=f"http://{is_ready_url_host(host)}:{port}/v1",
        web_search_profile=web_search_profile,
        code_interpreter_mode=code_interpreter_mode,
        code_interpreter_port=effective_code_interpreter_port,
        code_interpreter_workers=effective_code_interpreter_workers,
        code_interpreter_startup_timeout_s=code_interpreter_startup_timeout_s,
        mcp_config_path=mcp_config_path,
        mcp_builtin_runtime_url=mcp_builtin_runtime_url,
    )


__all__ = [
    "RuntimeConfigError",
    "build_runtime_config_for_integrated",
    "build_runtime_config_for_mock_llm",
    "build_runtime_config_for_standalone",
    "build_runtime_config_for_supervisor",
]
