from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from agentic_stack.utils.exceptions import BadInputError


def validate_request_remote_server_url(server_url: str) -> str:
    """Validate request-remote `server_url` and return normalized host for policy checks."""
    try:
        parsed = urlsplit(server_url)
    except ValueError as exc:
        raise BadInputError(f"Invalid request-remote MCP `server_url`: {exc}") from exc

    if parsed.scheme.lower() != "https":
        raise BadInputError("Request-remote MCP `server_url` must use `https`.")

    host = parsed.hostname
    if not host:
        raise BadInputError("Request-remote MCP `server_url` must include a host.")

    normalized_host = host.lower().rstrip(".")
    if normalized_host == "localhost" or normalized_host.endswith(".localhost"):
        raise BadInputError(
            "Request-remote MCP host is denylisted; `localhost` and `.localhost` are not allowed."
        )

    if _is_ip_literal_host(normalized_host):
        raise BadInputError(
            "Request-remote MCP host is denylisted; IP-literal hosts are not allowed."
        )

    return normalized_host


def build_request_remote_headers(
    *,
    authorization: str | None,
    request_headers: dict[str, str] | None,
) -> dict[str, str]:
    """Build outbound request-remote headers with deterministic Authorization precedence."""
    source_headers = dict(request_headers or {})
    authorization_keys = [k for k in source_headers if k.lower() == "authorization"]

    if authorization is None and len(authorization_keys) > 1:
        raise BadInputError(
            "Request-remote MCP `headers` contains multiple `Authorization` header variants."
        )

    if authorization is not None:
        for key in authorization_keys:
            source_headers.pop(key, None)
        source_headers["Authorization"] = f"Bearer {authorization}"
    return source_headers


def request_remote_secret_values(
    *,
    authorization: str | None,
    headers: dict[str, str] | None,
) -> tuple[str, ...]:
    """Collect request-remote secret-like values for error redaction."""
    values: list[str] = []
    if authorization:
        values.append(authorization)
    if headers:
        for value in headers.values():
            if value:
                values.append(value)
    # Longest-first prevents partial replacements from hiding longer exact matches.
    return tuple(sorted(set(values), key=len, reverse=True))


def _is_ip_literal_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False
