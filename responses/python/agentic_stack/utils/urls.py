from __future__ import annotations


def is_ready_url_host(host: str) -> str:
    """
    Translate a bind address into a connectable host for local readiness checks.
    """
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host
