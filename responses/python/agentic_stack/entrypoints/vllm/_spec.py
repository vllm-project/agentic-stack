from __future__ import annotations

from dataclasses import dataclass

from agentic_stack.configs.runtime import CodeInterpreterMode
from agentic_stack.configs.startup import (
    find_flag_value,
    format_integrated_responses_help_block,
    resolve_integrated_responses_cli,
)

_RESPONSES_FLAG = "--responses"


class IntegratedSpecError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = int(exit_code)


@dataclass(frozen=True, slots=True)
class IntegratedServeSpec:
    vllm_args: list[str]
    code_interpreter_mode: CodeInterpreterMode
    code_interpreter_port: int
    code_interpreter_workers: int
    code_interpreter_startup_timeout_s: float
    mcp_config_path: str | None = None
    mcp_port: int | None = None
    web_search_profile: str | None = None


def should_show_integrated_help(raw_args: list[str]) -> bool:
    return bool(
        raw_args
        and raw_args[0] == "serve"
        and _RESPONSES_FLAG in raw_args
        and "--help" in raw_args
    )


def format_integrated_help() -> str:
    return (
        "usage: vllm serve <MODEL_ID_OR_PATH> --responses [vllm args] [responses args]\n\n"
        "Integrated Responses mode augments the native `vllm serve` app with gateway-owned\n"
        "Responses routes and helper-runtime lifecycle.\n\n"
        f"{format_integrated_responses_help_block()}\n\n"
        "Native vLLM flags remain owned by upstream `vllm serve`.\n"
        "Use `vllm serve --help` for the upstream help surface.\n"
    )


def build_integrated_serve_spec(raw_args: list[str]) -> IntegratedServeSpec:
    if not raw_args or raw_args[0] != "serve":
        raise IntegratedSpecError(
            "[integrated] error: `--responses` is supported only for `vllm serve`.",
        )
    if _RESPONSES_FLAG not in raw_args:
        raise IntegratedSpecError("[integrated] error: integrated mode requires `--responses`.")

    filtered_args = [arg for arg in raw_args if arg not in {_RESPONSES_FLAG, "--help"}]
    if "--headless" in filtered_args:
        raise IntegratedSpecError(
            "[integrated] error: integrated mode does not support `--headless`. "
            "Use `agentic-stacks serve` for unsupported topologies.",
        )
    api_server_count_raw = find_flag_value(filtered_args, "--api-server-count")
    if api_server_count_raw is not None and _parse_api_server_count(api_server_count_raw) > 1:
        raise IntegratedSpecError(
            "[integrated] error: integrated mode requires a single API server "
            "(`--api-server-count` must be omitted or <= 1). "
            "Use `agentic-stacks serve` for unsupported topologies.",
        )

    resolved_cli = resolve_integrated_responses_cli(
        raw_args=filtered_args,
        error_prefix="[integrated]",
        error_factory=IntegratedSpecError,
    )

    return IntegratedServeSpec(
        vllm_args=resolved_cli.filtered_args,
        web_search_profile=resolved_cli.web_search_profile,
        code_interpreter_mode=resolved_cli.code_interpreter_mode,
        code_interpreter_port=resolved_cli.code_interpreter_port,
        code_interpreter_workers=resolved_cli.code_interpreter_workers,
        code_interpreter_startup_timeout_s=resolved_cli.code_interpreter_startup_timeout_s,
        mcp_config_path=resolved_cli.mcp_config_path,
        mcp_port=resolved_cli.mcp_port,
    )


def _parse_api_server_count(value: str) -> int:
    try:
        return int(value)
    except Exception:
        raise IntegratedSpecError(
            f"[integrated] error: invalid --api-server-count={value!r}."
        ) from None
