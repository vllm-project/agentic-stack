from __future__ import annotations

import re
from dataclasses import dataclass

from agentic_stack.mcp.runtime_client import BuiltinMcpRuntimeTransportError
from agentic_stack.tools.base.types import BuiltinActionAdapter
from agentic_stack.tools.ids import WEB_SEARCH_TOOL
from agentic_stack.tools.web_search.adapters.base import (
    OpenPageAdapter,
    SearchAdapter,
    SearchAdapterHintSupport,
    WebSearchAdapterContext,
)
from agentic_stack.tools.web_search.page_cache import (
    canonicalize_url,
    is_url_allowed,
    url_hostname,
)
from agentic_stack.tools.web_search.types import (
    ActionOutcome,
    OpenPageActionResult,
    SearchActionResult,
    SearchResultRecord,
    WebSearchRequestOptions,
    WebSearchSource,
)

_BLOCK_FIELD_RE = re.compile(r"^(Title|URL|Text):\s*(.*)$", re.MULTILINE)


def _require_runtime_client(ctx: WebSearchAdapterContext):
    if ctx.builtin_mcp_runtime_client is None:
        raise RuntimeError("Built-in MCP runtime client is not configured.")
    return ctx.builtin_mcp_runtime_client


def _apply_allowed_domains(
    results: list[SearchResultRecord], allowed_domains: tuple[str, ...]
) -> list[SearchResultRecord]:
    if not allowed_domains:
        return results
    filtered: list[SearchResultRecord] = []
    for result in results:
        if is_url_allowed(result.url, allowed_domains):
            filtered.append(result)
    return filtered


def _hostname(url: str) -> str | None:
    return url_hostname(url)


def _dedupe_sources(results: list[SearchResultRecord]) -> tuple[WebSearchSource, ...]:
    seen: set[str] = set()
    sources: list[WebSearchSource] = []
    for result in results:
        canonical = canonicalize_url(result.url)
        if canonical in seen:
            continue
        seen.add(canonical)
        sources.append(WebSearchSource(url=canonical))
    return tuple(sources)


def _parse_exa_search_output(output_text: str) -> list[SearchResultRecord]:
    results: list[SearchResultRecord] = []
    blocks = [block.strip() for block in re.split(r"\n{2,}", output_text) if block.strip()]
    for block in blocks:
        fields: dict[str, str] = {}
        current_key: str | None = None
        current_lines: list[str] = []
        for line in block.splitlines():
            match = _BLOCK_FIELD_RE.match(line)
            if match:
                if current_key is not None:
                    fields[current_key] = "\n".join(current_lines).strip()
                current_key = match.group(1)
                current_lines = [match.group(2)]
            elif current_key is not None:
                current_lines.append(line)
        if current_key is not None:
            fields[current_key] = "\n".join(current_lines).strip()
        url = fields.get("URL")
        if not url:
            continue
        results.append(
            SearchResultRecord(
                url=canonicalize_url(url),
                title=fields.get("Title") or None,
                snippet=fields.get("Text") or None,
            )
        )
    return results


@dataclass(frozen=True, slots=True)
class ExaMcpSearchAdapter(BuiltinActionAdapter, SearchAdapter):
    tool_type: str = WEB_SEARCH_TOOL
    action_name: str = "search"
    adapter_id: str = "exa_mcp_search"
    config_model: type | None = None
    server_label: str = "exa"
    hint_support: SearchAdapterHintSupport = SearchAdapterHintSupport()

    async def search(
        self,
        *,
        ctx: WebSearchAdapterContext,
        query: str,
        queries: tuple[str, ...],
        options: WebSearchRequestOptions,
    ) -> ActionOutcome[SearchActionResult]:
        client = _require_runtime_client(ctx)
        arguments: dict[str, object] = {"query": query}
        if options.allowed_domains:
            arguments["includeDomains"] = list(options.allowed_domains)
        try:
            result = await client.call_tool(
                server_label=self.server_label,
                tool_name="web_search_exa",
                arguments=arguments,
            )
        except (BuiltinMcpRuntimeTransportError, RuntimeError) as exc:
            return ActionOutcome(ok=False, error=str(exc))

        if not result.ok or not result.output_text:
            return ActionOutcome(ok=False, error=result.error_text or "Search failed.")

        normalized_results = _apply_allowed_domains(
            _parse_exa_search_output(result.output_text),
            options.allowed_domains,
        )
        sources = _dedupe_sources(normalized_results)
        return ActionOutcome(
            ok=True,
            data=SearchActionResult(
                query=query,
                queries=tuple(queries or (query,)),
                results=tuple(normalized_results),
                sources=sources,
            ),
        )


@dataclass(frozen=True, slots=True)
class ExaMcpOpenPageAdapter(BuiltinActionAdapter, OpenPageAdapter):
    tool_type: str = WEB_SEARCH_TOOL
    action_name: str = "open_page"
    adapter_id: str = "exa_mcp_open_page"
    config_model: type | None = None
    server_label: str = "exa"

    async def open_page(
        self,
        *,
        ctx: WebSearchAdapterContext,
        url: str,
        options: WebSearchRequestOptions,
    ) -> ActionOutcome[OpenPageActionResult]:
        _ = options
        client = _require_runtime_client(ctx)
        try:
            result = await client.call_tool(
                server_label=self.server_label,
                tool_name="crawling_exa",
                arguments={"url": url},
            )
        except (BuiltinMcpRuntimeTransportError, RuntimeError) as exc:
            return ActionOutcome(ok=False, error=str(exc))

        if not result.ok:
            return ActionOutcome(ok=False, error=result.error_text or "open_page failed.")
        return ActionOutcome(
            ok=True,
            data=OpenPageActionResult(
                url=canonicalize_url(url),
                title=None,
                text=(result.output_text or "").strip() or None,
            ),
        )
