from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from agentic_stack.tools.web_search.types import CachedPage


def canonicalize_url(url: str) -> str:
    stripped = url.strip()
    parts = urlsplit(stripped)
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def url_hostname(url: str) -> str | None:
    parts = urlsplit(canonicalize_url(url))
    hostname = parts.hostname
    return hostname.lower() if hostname is not None else None


def is_url_allowed(url: str, allowed_domains: tuple[str, ...]) -> bool:
    if not allowed_domains:
        return True
    hostname = url_hostname(url)
    if hostname is None:
        return False
    normalized_allowed = tuple(domain.lower().lstrip(".") for domain in allowed_domains)
    return any(
        hostname == domain or hostname.endswith(f".{domain}") for domain in normalized_allowed
    )


class WebSearchPageCache:
    def __init__(self) -> None:
        self._pages: dict[str, CachedPage] = {}

    def get(self, url: str | None) -> CachedPage | None:
        if not url:
            return None
        return self._pages.get(canonicalize_url(url))

    def put(self, page: CachedPage) -> None:
        self._pages[canonicalize_url(page.url)] = page
