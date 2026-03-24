from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, ValidationError, ValidationInfo, field_validator

from agentic_stack.configs.defaults import RUNTIME_DEFAULTS
from agentic_stack.configs.runtime import CodeInterpreterMode
from agentic_stack.tools.ids import WEB_SEARCH_TOOL
from agentic_stack.tools.profile_resolution import (
    profiled_builtin_requires_mcp,
    validate_profiled_builtin_profile,
)
from agentic_stack.tools.web_search.profiles import get_web_search_profile_ids

_SUPPORTED_CODE_INTERPRETER_MODES = "{spawn,external,disabled}"


def format_web_search_profile_choices() -> str:
    return ", ".join(get_web_search_profile_ids())


def find_flag_value(args: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


@dataclass(frozen=True, slots=True)
class ResponsesCliFlagSpec:
    field_name: str
    supervisor_flag: str | None
    supervisor_dest: str | None
    integrated_flag: str | None
    metavar: str
    help: str


RESPONSES_CLI_FLAG_SPECS = (
    ResponsesCliFlagSpec(
        field_name="web_search_profile",
        supervisor_flag="--web-search-profile",
        supervisor_dest="web_search_profile",
        integrated_flag="--responses-web-search-profile",
        metavar="PROFILE",
        help=(
            "Gateway-owned web search profile to enable. Choices: "
            f"{format_web_search_profile_choices()}."
        ),
    ),
    ResponsesCliFlagSpec(
        field_name="code_interpreter_mode",
        supervisor_flag="--code-interpreter",
        supervisor_dest="code_interpreter",
        integrated_flag="--responses-code-interpreter",
        metavar=_SUPPORTED_CODE_INTERPRETER_MODES,
        help="Code interpreter runtime policy (default: spawn).",
    ),
    ResponsesCliFlagSpec(
        field_name="code_interpreter_port",
        supervisor_flag="--code-interpreter-port",
        supervisor_dest="code_interpreter_port",
        integrated_flag="--responses-code-interpreter-port",
        metavar="PORT",
        help="Code interpreter port (when spawn|external).",
    ),
    ResponsesCliFlagSpec(
        field_name="code_interpreter_workers",
        supervisor_flag="--code-interpreter-workers",
        supervisor_dest="code_interpreter_workers",
        integrated_flag="--responses-code-interpreter-workers",
        metavar="N",
        help="Bun server --workers (only meaningful when --code-interpreter=spawn).",
    ),
    ResponsesCliFlagSpec(
        field_name="code_interpreter_startup_timeout_s",
        supervisor_flag="--code-interpreter-startup-timeout",
        supervisor_dest="code_interpreter_startup_timeout",
        integrated_flag="--responses-code-interpreter-startup-timeout",
        metavar="SECONDS",
        help="Maximum time to wait for the code interpreter readiness check.",
    ),
    ResponsesCliFlagSpec(
        field_name="upstream_ready_timeout_s",
        supervisor_flag="--upstream-ready-timeout",
        supervisor_dest="upstream_ready_timeout",
        integrated_flag=None,
        metavar="SECONDS",
        help="Maximum time to wait for upstream readiness.",
    ),
    ResponsesCliFlagSpec(
        field_name="upstream_ready_interval_s",
        supervisor_flag="--upstream-ready-interval",
        supervisor_dest="upstream_ready_interval",
        integrated_flag=None,
        metavar="SECONDS",
        help="Polling interval for upstream readiness checks.",
    ),
    ResponsesCliFlagSpec(
        field_name="mcp_config_path",
        supervisor_flag="--mcp-config",
        supervisor_dest="mcp_config",
        integrated_flag="--responses-mcp-config",
        metavar="PATH",
        help="Built-in MCP runtime config file path.",
    ),
    ResponsesCliFlagSpec(
        field_name="mcp_port",
        supervisor_flag="--mcp-port",
        supervisor_dest="mcp_port",
        integrated_flag="--responses-mcp-port",
        metavar="PORT",
        help="Loopback port for the Built-in MCP runtime.",
    ),
)


class ResponsesCliArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    web_search_profile: str | None = None
    code_interpreter_mode: CodeInterpreterMode | None = None
    code_interpreter_port: int | None = None
    code_interpreter_workers: int | None = None
    code_interpreter_startup_timeout_s: float | None = None
    upstream_ready_timeout_s: float | None = None
    upstream_ready_interval_s: float | None = None
    mcp_config_path: str | None = None
    mcp_port: int | None = None

    @field_validator("code_interpreter_mode", mode="before")
    @classmethod
    def _normalize_code_interpreter_mode(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            return normalized or None
        return value

    @field_validator(
        "code_interpreter_port",
        "code_interpreter_workers",
        "code_interpreter_startup_timeout_s",
        "upstream_ready_timeout_s",
        "upstream_ready_interval_s",
        "mcp_port",
        mode="before",
    )
    @classmethod
    def _normalize_optional_numbers(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return None if not stripped else stripped
        return value

    @field_validator("mcp_config_path", mode="before")
    @classmethod
    def _normalize_mcp_config_path(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("web_search_profile", mode="before")
    @classmethod
    def _normalize_web_search_profile(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("web_search_profile")
    @classmethod
    def _validate_web_search_profile(cls, value: str | None) -> str | None:
        try:
            validate_profiled_builtin_profile(
                tool_type=WEB_SEARCH_TOOL,
                profile_id=value,
            )
        except ValueError as exc:
            raise ValueError("unknown_profile") from exc
        return value

    @field_validator("mcp_port")
    @classmethod
    def _validate_mcp_port_requires_config(
        cls, value: int | None, info: ValidationInfo
    ) -> int | None:
        if (
            value is not None
            and info.data.get("mcp_config_path") is None
            and not profiled_builtin_requires_mcp(
                tool_type=WEB_SEARCH_TOOL,
                profile_id=info.data.get("web_search_profile"),
            )
        ):
            raise ValueError("requires_mcp_config")
        return value


@dataclass(frozen=True, slots=True)
class ResolvedIntegratedResponsesCli:
    filtered_args: list[str]
    web_search_profile: str | None
    code_interpreter_mode: CodeInterpreterMode
    code_interpreter_port: int
    code_interpreter_workers: int
    code_interpreter_startup_timeout_s: float
    mcp_config_path: str | None
    mcp_port: int | None


def add_supervisor_responses_cli_arguments(parser: argparse.ArgumentParser) -> None:
    for spec in RESPONSES_CLI_FLAG_SPECS:
        if spec.supervisor_flag is None or spec.supervisor_dest is None:
            continue
        parser.add_argument(
            spec.supervisor_flag,
            dest=spec.supervisor_dest,
            type=str,
            default=None,
            help=spec.help,
        )


def format_integrated_responses_help_block() -> str:
    lines = ["Responses-owned flags:", "  --responses"]
    for spec in RESPONSES_CLI_FLAG_SPECS:
        if spec.integrated_flag is None:
            continue
        lines.append(f"  {spec.integrated_flag} {spec.metavar}".rstrip())
        lines.append(f"      {spec.help}")
    return "\n".join(lines)


def strip_integrated_responses_cli_flags(
    raw_args: list[str],
) -> tuple[dict[str, str | None], list[str]]:
    kept_args = list(raw_args)
    raw_values: dict[str, str | None] = {}
    for spec in RESPONSES_CLI_FLAG_SPECS:
        if spec.integrated_flag is None:
            continue
        value, kept_args = _parse_optional_flag(kept_args, spec.integrated_flag)
        raw_values[spec.field_name] = value
    return raw_values, kept_args


def supervisor_responses_cli_raw_values(args: Any) -> dict[str, object]:
    raw_values: dict[str, object] = {}
    for spec in RESPONSES_CLI_FLAG_SPECS:
        if spec.supervisor_dest is None:
            continue
        raw_values[spec.field_name] = getattr(args, spec.supervisor_dest, None)
    return raw_values


def supervisor_responses_cli_labels() -> dict[str, str]:
    return {
        spec.field_name: spec.supervisor_flag
        for spec in RESPONSES_CLI_FLAG_SPECS
        if spec.supervisor_flag is not None
    }


def integrated_responses_cli_labels(raw_values: Mapping[str, object]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for spec in RESPONSES_CLI_FLAG_SPECS:
        if spec.integrated_flag is None:
            continue
        labels[spec.field_name] = spec.integrated_flag
    return labels


def validate_responses_cli_args(
    *,
    raw_values: Mapping[str, object],
    labels: Mapping[str, str],
    error_prefix: str,
    error_factory: Callable[[str], Exception],
) -> ResponsesCliArgs:
    try:
        return ResponsesCliArgs.model_validate(dict(raw_values))
    except ValidationError as exc:
        raise error_factory(
            _format_responses_cli_error(
                exc=exc,
                raw_values=raw_values,
                labels=labels,
                error_prefix=error_prefix,
            )
        ) from None


def resolve_integrated_responses_cli(
    *,
    raw_args: list[str],
    error_prefix: str,
    error_factory: Callable[[str], Exception],
) -> ResolvedIntegratedResponsesCli:
    try:
        integrated_raw_values, filtered_args = strip_integrated_responses_cli_flags(raw_args)
    except ValueError as exc:
        raise error_factory(f"{error_prefix} error: {exc}.") from None

    raw_values = {
        "web_search_profile": integrated_raw_values["web_search_profile"],
        "code_interpreter_mode": (
            integrated_raw_values["code_interpreter_mode"]
            if integrated_raw_values["code_interpreter_mode"] is not None
            else RUNTIME_DEFAULTS.code_interpreter_mode
        ),
        "code_interpreter_port": (
            integrated_raw_values["code_interpreter_port"]
            if integrated_raw_values["code_interpreter_port"] is not None
            else str(RUNTIME_DEFAULTS.code_interpreter_port)
        ),
        "code_interpreter_workers": (
            integrated_raw_values["code_interpreter_workers"]
            if integrated_raw_values["code_interpreter_workers"] is not None
            else str(RUNTIME_DEFAULTS.code_interpreter_workers)
        ),
        "code_interpreter_startup_timeout_s": (
            integrated_raw_values["code_interpreter_startup_timeout_s"]
            if integrated_raw_values["code_interpreter_startup_timeout_s"] is not None
            else str(RUNTIME_DEFAULTS.code_interpreter_startup_timeout_s)
        ),
        "mcp_config_path": integrated_raw_values["mcp_config_path"],
        "mcp_port": integrated_raw_values["mcp_port"],
    }
    responses_cli = validate_responses_cli_args(
        raw_values=raw_values,
        labels=integrated_responses_cli_labels(integrated_raw_values),
        error_prefix=error_prefix,
        error_factory=error_factory,
    )
    code_interpreter_mode = responses_cli.code_interpreter_mode
    code_interpreter_port = responses_cli.code_interpreter_port
    code_interpreter_workers = responses_cli.code_interpreter_workers
    code_interpreter_startup_timeout_s = responses_cli.code_interpreter_startup_timeout_s
    if code_interpreter_mode is None:
        raise AssertionError("integrated code interpreter mode must be resolved")
    if code_interpreter_port is None:
        raise AssertionError("integrated code interpreter port must be resolved")
    if code_interpreter_workers is None:
        raise AssertionError("integrated code interpreter workers must be resolved")
    if code_interpreter_startup_timeout_s is None:
        raise AssertionError("integrated code interpreter timeout must be resolved")
    return ResolvedIntegratedResponsesCli(
        filtered_args=filtered_args,
        web_search_profile=responses_cli.web_search_profile,
        code_interpreter_mode=code_interpreter_mode,
        code_interpreter_port=int(code_interpreter_port),
        code_interpreter_workers=int(code_interpreter_workers),
        code_interpreter_startup_timeout_s=float(code_interpreter_startup_timeout_s),
        mcp_config_path=responses_cli.mcp_config_path,
        mcp_port=responses_cli.mcp_port,
    )


def _format_responses_cli_error(
    *,
    exc: ValidationError,
    raw_values: Mapping[str, object],
    labels: Mapping[str, str],
    error_prefix: str,
) -> str:
    error = exc.errors(include_url=False)[0]
    field_name = str(error["loc"][0]) if error.get("loc") else "value"
    label = labels.get(field_name, field_name)
    raw_value = raw_values.get(field_name)

    if field_name == "code_interpreter_mode" and error["type"] == "literal_error":
        return f"{error_prefix} error: {label} must be one of {_SUPPORTED_CODE_INTERPRETER_MODES}."

    if (
        field_name == "mcp_port"
        and error["type"] == "value_error"
        and "requires_mcp_config" in error["msg"]
    ):
        return (
            f"{error_prefix} error: {label} requires "
            f"{labels.get('mcp_config_path', 'mcp_config_path')} or a web_search profile "
            "that provisions Built-in MCP helpers."
        )

    if (
        field_name == "web_search_profile"
        and error["type"] == "value_error"
        and "unknown_profile" in error["msg"]
    ):
        return (
            f"{error_prefix} error: unknown {label}={raw_value!r}; "
            f"expected one of: {format_web_search_profile_choices()}."
        )

    return f"{error_prefix} error: invalid {label}={raw_value!r}."


def _parse_optional_flag(raw_args: list[str], flag: str) -> tuple[str | None, list[str]]:
    kept: list[str] = []
    value: str | None = None
    skip_next = False
    for index, arg in enumerate(raw_args):
        if skip_next:
            skip_next = False
            continue
        if arg == flag:
            if index + 1 >= len(raw_args):
                raise ValueError(f"{flag} requires a value")
            value = raw_args[index + 1]
            skip_next = True
            continue
        prefix = f"{flag}="
        if arg.startswith(prefix):
            value = arg[len(prefix) :]
            continue
        kept.append(arg)
    return value, kept
