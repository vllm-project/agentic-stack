from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pydantic_ai import ModelHTTPError, UnexpectedModelBehavior

from agentic_stack.responses_core.models import (
    CodeInterpreterCallStarted,
    FunctionCallStarted,
    McpCallArgumentsDone,
    McpCallCompleted,
    McpCallFailed,
    McpCallStarted,
)
from agentic_stack.utils.io import json_loads


def extract_openai_error_fields(
    err_body: dict[str, Any] | None,
    *,
    fallback_message: str,
) -> tuple[str, str, str]:
    """Best-effort parse an OpenAI-style error object.

    Expected shape: {"error": {"message": ..., "type": ..., "param": ..., "code": ...}}
    """

    body = err_body or {}
    err = body.get("error") if isinstance(body.get("error"), dict) else body
    if not isinstance(err, dict):
        return "", fallback_message, ""

    code_raw = err.get("code")
    # Upstreams vary: OpenAI uses string-or-null, but other providers sometimes return ints (e.g. 404).
    code = "" if code_raw is None else str(code_raw)
    message_raw = err.get("message")
    message = (
        str(message_raw) if isinstance(message_raw, str) and message_raw else fallback_message
    )
    param_raw = err.get("param")
    param = "" if param_raw is None else str(param_raw)
    return code, message, param


@dataclass(frozen=True, slots=True)
class FailureDetails:
    error_class: str
    code: str
    message: str
    param: str
    upstream_status_code: int | None
    upstream_error_raw: str | None


@dataclass(slots=True)
class _McpCallContext:
    mode: str
    server_label: str
    tool_name: str
    arguments_json: str


@dataclass(slots=True)
class FailureCounters:
    tool_call_parts_seen: int = 0
    mcp_failed_count_hosted: int = 0
    mcp_failed_count_request_remote: int = 0
    last_failed_mcp_signature: dict[str, str] | None = None
    _mcp_call_context_by_item: dict[str, _McpCallContext] = field(default_factory=dict)

    def observe(self, normalized_event: object) -> None:
        if isinstance(normalized_event, (FunctionCallStarted, CodeInterpreterCallStarted)):
            self.tool_call_parts_seen += 1
            return

        if isinstance(normalized_event, McpCallStarted):
            self.tool_call_parts_seen += 1
            self._mcp_call_context_by_item[normalized_event.item_key] = _McpCallContext(
                mode=normalized_event.mode,
                server_label=normalized_event.server_label,
                tool_name=normalized_event.name,
                arguments_json=normalized_event.initial_arguments_json,
            )
            return

        if isinstance(normalized_event, McpCallArgumentsDone):
            call_ctx = self._mcp_call_context_by_item.get(normalized_event.item_key)
            if call_ctx is not None:
                call_ctx.arguments_json = normalized_event.arguments_json
            return

        if isinstance(normalized_event, McpCallCompleted):
            self._mcp_call_context_by_item.pop(normalized_event.item_key, None)
            return

        if isinstance(normalized_event, McpCallFailed):
            call_ctx = self._mcp_call_context_by_item.pop(normalized_event.item_key, None)
            mode = "hosted"
            if call_ctx is not None and call_ctx.mode in {"hosted", "request_remote"}:
                mode = call_ctx.mode
            if mode == "request_remote":
                self.mcp_failed_count_request_remote += 1
            else:
                self.mcp_failed_count_hosted += 1
            if call_ctx is not None:
                self.last_failed_mcp_signature = {
                    "mode": mode,
                    "server_label": call_ctx.server_label,
                    "tool_name": call_ctx.tool_name,
                }


_FAILURE_ERROR_MESSAGE_MAX_CHARS = 512
_FAILURE_UPSTREAM_RAW_MAX_CHARS = 2048
_FAILURE_DEBUG_MESSAGE_MAX_CHARS = 8192


def _truncate_prefix(value: str, limit: int) -> str:
    return value[:limit]


def _upstream_error_raw(raw: Any) -> str | None:
    if raw is None:
        return None
    return _truncate_prefix(str(raw), _FAILURE_UPSTREAM_RAW_MAX_CHARS)


def _extract_error_body(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            parsed = json_loads(raw)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def extract_failure_details(e: ModelHTTPError | UnexpectedModelBehavior) -> FailureDetails:
    if isinstance(e, ModelHTTPError):
        err_body = _extract_error_body(e.body)
        err_code, err_message, err_param = extract_openai_error_fields(
            err_body,
            fallback_message=str(e),
        )
        return FailureDetails(
            error_class=e.__class__.__name__,
            code=err_code,
            message=err_message,
            param=err_param,
            upstream_status_code=e.status_code,
            upstream_error_raw=_upstream_error_raw(e.body),
        )

    err_body = _extract_error_body(e.body)
    err_code, err_message, err_param = extract_openai_error_fields(
        err_body,
        fallback_message=e.message,
    )
    return FailureDetails(
        error_class=e.__class__.__name__,
        code=err_code,
        message=err_message,
        param=err_param,
        upstream_status_code=None,
        upstream_error_raw=_upstream_error_raw(e.body),
    )


def classify_failure_log_level(*, error_class: str, upstream_status_code: int | None) -> str:
    if error_class == ModelHTTPError.__name__ and upstream_status_code is not None:
        if 400 <= upstream_status_code < 500:
            return "warning"
    return "error"


def log_failure_summary(
    *,
    response_id: str | None,
    failure_phase: str,
    error_class: str,
    log_level: str,
    upstream_status_code: int | None,
    error_message: str,
    messages: list[Any] | Any,
    counters: FailureCounters,
    upstream_error_raw: str | None,
    log_model_messages: bool,
) -> None:
    summary: dict[str, Any] = {
        "request_id": response_id,
        "failure_phase": failure_phase,
        "error_class": error_class,
        "log_level": log_level,
        "upstream_status_code": upstream_status_code,
        "error_message": _truncate_prefix(error_message, _FAILURE_ERROR_MESSAGE_MAX_CHARS),
        "total_messages": len(messages) if hasattr(messages, "__len__") else 0,
        "tool_call_parts_seen": counters.tool_call_parts_seen,
        "mcp_failed_count_total": (
            counters.mcp_failed_count_hosted + counters.mcp_failed_count_request_remote
        ),
        "mcp_failed_count_hosted": counters.mcp_failed_count_hosted,
        "mcp_failed_count_request_remote": counters.mcp_failed_count_request_remote,
    }
    if counters.last_failed_mcp_signature is not None:
        summary["last_failed_mcp_signature"] = counters.last_failed_mcp_signature
    if upstream_error_raw:
        summary["upstream_error_raw"] = upstream_error_raw

    log_fn = logger.warning if log_level == "warning" else logger.error
    log_fn("LMEngine failure summary: {}", summary)

    if not log_model_messages:
        return

    if not isinstance(messages, list):
        log_fn(
            "LMEngine captured messages debug dump unavailable for type: {}",
            type(messages).__name__,
        )
        return

    for i, entry in enumerate(messages):
        log_fn(
            "LMEngine captured_messages[{}]: {}",
            i,
            _truncate_prefix(repr(entry), _FAILURE_DEBUG_MESSAGE_MAX_CHARS),
        )
