from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentic_stack.configs.runtime import RuntimeConfig


class _WebSearchRequestTool(Protocol):
    filters: object | None
    search_context_size: str
    user_location: object | None


@dataclass(frozen=True, slots=True)
class ResolvedWebSearchRequestConfig:
    profile_id: str
    allowed_domains: tuple[str, ...]
    search_context_size: str
    user_location: dict[str, str | None] | None


def resolve_request_config(
    *,
    tool: _WebSearchRequestTool,
    runtime_config: RuntimeConfig,
) -> ResolvedWebSearchRequestConfig:
    profile_id = (runtime_config.web_search_profile or "").strip()
    if not profile_id:
        raise ValueError("`web_search` is disabled by configuration.")

    allowed_domains = (
        tuple(tool.filters.allowed_domains)
        if tool.filters is not None and hasattr(tool.filters, "allowed_domains")
        else ()
    )
    user_location = None
    if tool.user_location is not None:
        timezone = tool.user_location.timezone
        user_location = {
            "city": tool.user_location.city,
            "country": tool.user_location.country,
            "region": tool.user_location.region,
            "timezone": str(timezone) if timezone is not None else None,
        }
    return ResolvedWebSearchRequestConfig(
        profile_id=profile_id,
        allowed_domains=allowed_domains,
        search_context_size=tool.search_context_size,
        user_location=user_location,
    )
