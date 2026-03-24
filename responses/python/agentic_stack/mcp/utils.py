from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agentic_stack.mcp.types import McpExecutionResult, McpMode, McpToolRef

HOSTED_MCP_INTERNAL_PREFIX = "mcp__"
MCP_TOOL_RESULT_KIND = "agentic_stack_mcp_result"
HOSTED_MCP_MAX_TOOL_NAME_LEN = 64
HOSTED_MCP_ERROR_MAX_CHARS = 4096

_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")


class _ToolResultModel(BaseModel):
    kind: str = Field(min_length=1)
    server_label: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    mode: McpMode | None = None
    ok: bool
    output_text: str | None = None
    error_text: str | None = None


def truncate_error_text(text: str | None) -> str:
    candidate = (text or "").strip()
    if not candidate:
        candidate = "MCP tool call failed"
    return candidate[:HOSTED_MCP_ERROR_MAX_CHARS]


def redact_and_truncate_error_text(
    *,
    text: str | None,
    secret_values: tuple[str, ...],
) -> str:
    redacted = text or ""
    for secret_value in secret_values:
        if secret_value:
            redacted = redacted.replace(secret_value, "***")
    return truncate_error_text(redacted)


def is_mcp_tool_keyerror(exc: Exception, tool_name: str) -> bool:
    if not isinstance(exc, KeyError):
        return False
    if len(exc.args) != 1:
        return False
    missing_key = exc.args[0]
    return isinstance(missing_key, str) and missing_key == tool_name


def sanitize_internal_tool_name(name: str) -> str:
    sanitized = _SANITIZE_RE.sub("_", name)
    return sanitized[:HOSTED_MCP_MAX_TOOL_NAME_LEN]


def build_internal_mcp_tool_name(
    *,
    server_label: str,
    tool_name: str,
    existing_map: dict[str, McpToolRef],
) -> str:
    ref = McpToolRef(server_label=server_label, tool_name=tool_name)
    base_name = sanitize_internal_tool_name(
        f"{HOSTED_MCP_INTERNAL_PREFIX}{server_label}__{tool_name}"
    )

    if len(base_name) <= HOSTED_MCP_MAX_TOOL_NAME_LEN:
        existing = existing_map.get(base_name)
        if existing is None or existing == ref:
            return base_name

    hash_suffix = (
        "__" + hashlib.sha1(f"{server_label}:{tool_name}".encode("utf-8")).hexdigest()[:10]
    )
    prefix_len = HOSTED_MCP_MAX_TOOL_NAME_LEN - len(hash_suffix)
    candidate = base_name[:prefix_len] + hash_suffix

    existing = existing_map.get(candidate)
    if existing is not None and existing != ref:
        raise ValueError(
            "Deterministic hosted MCP tool-name collision: "
            f"{candidate!r} maps to both {existing!r} and {ref!r}."
        )
    return candidate


def build_mcp_tool_result_payload(
    *, ref: McpToolRef, result: McpExecutionResult
) -> dict[str, Any]:
    return {
        "kind": MCP_TOOL_RESULT_KIND,
        "server_label": ref.server_label,
        "tool_name": ref.tool_name,
        "mode": ref.mode,
        "ok": bool(result.ok),
        "output_text": result.output_text,
        "error_text": truncate_error_text(result.error_text) if not result.ok else None,
    }


def parse_mcp_tool_result_payload(payload: Any) -> tuple[McpToolRef, McpExecutionResult]:
    if not isinstance(payload, dict):
        raise ValueError("MCP tool result payload must be a JSON object.")
    try:
        parsed = _ToolResultModel.model_validate(payload)
    except ValidationError as exc:
        issue = exc.errors()[0] if exc.errors() else {"msg": "invalid payload"}
        raise ValueError(f"MCP tool result payload is invalid: {issue['msg']}.") from exc
    if parsed.kind != MCP_TOOL_RESULT_KIND:
        raise ValueError(f"MCP tool result payload has invalid kind: {parsed.kind!r}.")

    return McpToolRef(
        server_label=parsed.server_label,
        tool_name=parsed.tool_name,
        mode=parsed.mode or "hosted",
    ), McpExecutionResult(
        ok=parsed.ok,
        output_text=parsed.output_text,
        error_text=truncate_error_text(parsed.error_text) if not parsed.ok else None,
    )


def canonicalize_output_text(value: Any) -> str:
    # Keep output text deterministic across transports/providers. We normalize all
    # JSON-serializable non-string values (including scalars), not just objects/arrays.
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if value in ([], {}, ()):
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except TypeError:
        return str(value)


def normalize_mcp_input_schema(input_schema: dict[str, object]) -> dict[str, object]:
    """Canonicalize MCP input schema and ensure object root typing when missing."""
    normalized = json.loads(
        json.dumps(input_schema, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )
    if not isinstance(normalized, dict):
        raise ValueError("`input_schema` must be a JSON object.")

    if "type" not in normalized:
        normalized["type"] = "object"
    return normalized
