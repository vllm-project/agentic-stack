from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

from agentic_stack.tools.base.types import BuiltinActionAdapter
from agentic_stack.tools.ids import WEB_SEARCH_TOOL
from agentic_stack.tools.web_search.adapters.base import (
    SearchAdapter,
    SearchAdapterHintSupport,
    WebSearchAdapterContext,
)
from agentic_stack.tools.web_search.page_cache import canonicalize_url, is_url_allowed
from agentic_stack.tools.web_search.types import (
    ActionOutcome,
    SearchActionResult,
    SearchResultRecord,
    WebSearchRequestOptions,
    WebSearchSource,
)

_RESULT_LIMIT_BY_CONTEXT = {
    "low": 5,
    "medium": 8,
    "high": 12,
}


def _apply_allowed_domains(
    results: list[SearchResultRecord], allowed_domains: tuple[str, ...]
) -> list[SearchResultRecord]:
    if not allowed_domains:
        return results
    return [result for result in results if is_url_allowed(result.url, allowed_domains)]


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


async def _run_duckduckgo_search(
    *,
    query: str,
    max_results: int,
) -> list[dict[str, str]]:
    tool = duckduckgo_search_tool(max_results=max_results)
    return await tool.function(query)


@dataclass(frozen=True, slots=True)
class DuckDuckGoCommonSearchAdapter(BuiltinActionAdapter, SearchAdapter):
    tool_type: str = WEB_SEARCH_TOOL
    action_name: str = "search"
    adapter_id: str = "duckduckgo_common_search"
    config_model: type | None = None
    hint_support: SearchAdapterHintSupport = SearchAdapterHintSupport(search_context_size=True)

    async def search(
        self,
        *,
        ctx: WebSearchAdapterContext,
        query: str,
        queries: tuple[str, ...],
        options: WebSearchRequestOptions,
    ) -> ActionOutcome[SearchActionResult]:
        _ = ctx
        try:
            raw_results = await _run_duckduckgo_search(
                query=query,
                max_results=_RESULT_LIMIT_BY_CONTEXT[options.search_context_size],
            )
        except Exception as exc:
            return ActionOutcome(ok=False, error=str(exc))

        normalized_results = [
            SearchResultRecord(
                url=canonicalize_url(result["href"]),
                title=result.get("title") or None,
                snippet=result.get("body") or None,
            )
            for result in raw_results
            if result.get("href")
        ]
        filtered_results = _apply_allowed_domains(normalized_results, options.allowed_domains)
        return ActionOutcome(
            ok=True,
            data=SearchActionResult(
                query=query,
                queries=tuple(queries or (query,)),
                results=tuple(filtered_results),
                sources=_dedupe_sources(filtered_results),
            ),
        )
