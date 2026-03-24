from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import cast

from loguru import logger

from agentic_stack.observability.metrics import record_tool_executed
from agentic_stack.tools.base.types import (
    BoundRuntimeRequirements,
    BuiltinActionAdapter,
    ResolvedProfiledBuiltinTool,
)
from agentic_stack.tools.web_search.adapters.base import (
    OpenPageAdapter,
    SearchAdapter,
    SearchAdapterHintSupport,
    WebSearchAdapterContext,
)
from agentic_stack.tools.web_search.config import ResolvedWebSearchRequestConfig
from agentic_stack.tools.web_search.page_cache import (
    WebSearchPageCache,
    canonicalize_url,
    is_url_allowed,
)
from agentic_stack.tools.web_search.types import (
    ActionOutcome,
    CachedPage,
    FindInPageActionPublic,
    FindInPageMatch,
    FindInPageToolResult,
    OpenPageActionPublic,
    OpenPageToolResult,
    SearchActionPublic,
    SearchToolResult,
    WebSearchActionRequest,
    WebSearchRequestOptions,
    WebSearchToolResult,
)


@dataclass(slots=True)
class WebSearchExecutor:
    request_config: ResolvedWebSearchRequestConfig
    resolved_tool: ResolvedProfiledBuiltinTool
    bound_requirements: BoundRuntimeRequirements
    adapter_by_action: dict[str, BuiltinActionAdapter]
    page_cache: WebSearchPageCache
    _warned_ignored_hints: set[str] = field(default_factory=set, init=False, repr=False)

    async def execute(self, action_request: WebSearchActionRequest) -> WebSearchToolResult:
        start = perf_counter()
        errored = False
        try:
            return await self._execute(action_request)
        except Exception as exc:
            errored = True
            logger.exception("web_search action failed: {}", exc)
            raise
        finally:
            record_tool_executed(
                tool_type="web_search",
                duration_s=perf_counter() - start,
                errored=errored,
            )

    async def _execute(self, action_request: WebSearchActionRequest) -> WebSearchToolResult:
        options = WebSearchRequestOptions(
            allowed_domains=self.request_config.allowed_domains,
            search_context_size=self.request_config.search_context_size,  # type: ignore[arg-type]
            user_location=self.request_config.user_location,
        )
        ctx = WebSearchAdapterContext(
            builtin_mcp_runtime_client=self.bound_requirements.builtin_mcp_runtime_client,
        )

        if action_request.action == "search":
            search_adapter = self.adapter_by_action.get("search")
            if search_adapter is None:
                return SearchToolResult(
                    action=SearchActionPublic(query=action_request.query or "", queries=[]),
                    error="No search adapter configured.",
                )
            self._warn_ignored_search_hints(cast(SearchAdapter, search_adapter), options)
            outcome = await cast(SearchAdapter, search_adapter).search(
                ctx=ctx,
                query=action_request.query or "",
                queries=action_request.queries,
                options=options,
            )
            return self._search_result(action_request=action_request, outcome=outcome)

        if action_request.action == "open_page":
            return await self._open_page_result(
                action_request=action_request,
                adapter=self.adapter_by_action.get("open_page"),
                ctx=ctx,
                options=options,
            )

        return await self._find_in_page_result(
            action_request=action_request,
            open_page_adapter=self.adapter_by_action.get("open_page"),
            ctx=ctx,
            options=options,
        )

    def _warn_ignored_search_hints(
        self, adapter: SearchAdapter, options: WebSearchRequestOptions
    ) -> None:
        hint_support = getattr(adapter, "hint_support", SearchAdapterHintSupport())

        if (
            options.user_location is not None
            and not hint_support.user_location
            and "user_location" not in self._warned_ignored_hints
        ):
            logger.warning(
                "web_search user_location is ignored by adapter={} for profile={}.",
                getattr(adapter, "adapter_id", adapter.__class__.__name__),
                self.resolved_tool.profile_id,
            )
            self._warned_ignored_hints.add("user_location")

        if (
            options.search_context_size != "medium"
            and not hint_support.search_context_size
            and "search_context_size" not in self._warned_ignored_hints
        ):
            logger.warning(
                "web_search search_context_size={} is ignored by adapter={} for profile={}.",
                options.search_context_size,
                getattr(adapter, "adapter_id", adapter.__class__.__name__),
                self.resolved_tool.profile_id,
            )
            self._warned_ignored_hints.add("search_context_size")

    def _search_result(
        self,
        *,
        action_request: WebSearchActionRequest,
        outcome: ActionOutcome,
    ) -> SearchToolResult:
        query = action_request.query or ""
        queries = list(action_request.queries or ((query,) if query else ()))
        if not outcome.ok or outcome.data is None:
            return SearchToolResult(
                action=SearchActionPublic(type="search", query=query, queries=queries or None),
                error=outcome.error or "Search failed.",
            )
        data = outcome.data
        return SearchToolResult(
            action=SearchActionPublic(
                type="search",
                query=data.query,
                queries=list(data.queries) or None,
                sources=list(data.sources),
            ),
            results=list(data.results),
        )

    async def _open_page_result(
        self,
        *,
        action_request: WebSearchActionRequest,
        adapter,
        ctx: WebSearchAdapterContext,
        options: WebSearchRequestOptions,
    ) -> OpenPageToolResult:
        url = action_request.url
        if not url:
            return OpenPageToolResult(
                action=OpenPageActionPublic(url=None),
                page={"url": None, "title": None, "text": None},
                error="`url` is required for open_page.",
            )
        if not is_url_allowed(url, options.allowed_domains):
            return OpenPageToolResult(
                action=OpenPageActionPublic(url=None),
                page={"url": None, "title": None, "text": None},
                error="URL is not allowed by `filters.allowed_domains`.",
            )
        if adapter is None:
            return OpenPageToolResult(
                action=OpenPageActionPublic(url=canonicalize_url(url)),
                page={"url": canonicalize_url(url)},
                error="No open_page adapter configured.",
            )
        outcome = await cast(OpenPageAdapter, adapter).open_page(
            ctx=ctx,
            url=url,
            options=options,
        )
        if not outcome.ok or outcome.data is None:
            return OpenPageToolResult(
                action=OpenPageActionPublic(url=canonicalize_url(url)),
                page={"url": canonicalize_url(url)},
                error=outcome.error or "open_page failed.",
            )
        page = outcome.data
        if page.url is not None and page.text is not None:
            self.page_cache.put(
                CachedPage(url=page.url, title=page.title, text=page.text),
            )
        return OpenPageToolResult(
            action=OpenPageActionPublic(url=page.url),
            page={"url": page.url, "title": page.title, "text": page.text},
        )

    async def _find_in_page_result(
        self,
        *,
        action_request: WebSearchActionRequest,
        open_page_adapter,
        ctx: WebSearchAdapterContext,
        options: WebSearchRequestOptions,
    ) -> FindInPageToolResult:
        _ = open_page_adapter, ctx
        url = action_request.url
        pattern = action_request.pattern or ""
        if not url or not pattern:
            return FindInPageToolResult(
                action=FindInPageActionPublic(url=url, pattern=pattern),
                error="`url` and `pattern` are required for find_in_page.",
            )
        if not is_url_allowed(url, options.allowed_domains):
            return FindInPageToolResult(
                action=FindInPageActionPublic(url=None, pattern=pattern),
                error="URL is not allowed by `filters.allowed_domains`.",
            )
        cached_page = self.page_cache.get(url)
        if cached_page is None:
            return FindInPageToolResult(
                action=FindInPageActionPublic(url=canonicalize_url(url), pattern=pattern),
                error="Page is not available in the request-local web_search page cache.",
            )

        matches: list[FindInPageMatch] = []
        if cached_page is not None and cached_page.text:
            haystack = cached_page.text
            needle = pattern.casefold()
            start_index = 0
            while True:
                pos = haystack.casefold().find(needle, start_index)
                if pos < 0:
                    break
                end = pos + len(pattern)
                context_start = max(0, pos - 60)
                context_end = min(len(haystack), end + 60)
                matches.append(
                    FindInPageMatch(
                        start=pos,
                        end=end,
                        text=haystack[context_start:context_end],
                    )
                )
                start_index = end

        return FindInPageToolResult(
            action=FindInPageActionPublic(
                url=canonicalize_url(url),
                pattern=pattern,
            ),
            matches=matches,
        )
