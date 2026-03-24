from __future__ import annotations

import re
from dataclasses import dataclass

from agentic_stack.mcp.runtime_client import BuiltinMcpRuntimeTransportError
from agentic_stack.tools.base.types import BuiltinActionAdapter
from agentic_stack.tools.ids import WEB_SEARCH_TOOL
from agentic_stack.tools.web_search.adapters.base import (
    OpenPageAdapter,
    WebSearchAdapterContext,
)
from agentic_stack.tools.web_search.page_cache import canonicalize_url
from agentic_stack.tools.web_search.types import (
    ActionOutcome,
    OpenPageActionResult,
    WebSearchRequestOptions,
)

_TRUNCATION_RE = re.compile(r"<error>Content truncated.*?</error>", re.DOTALL)


def _normalize_fetch_output(url: str, output_text: str) -> str:
    prefix = f"Contents of {url}:"
    text = output_text.strip()
    if text.startswith(prefix):
        text = text[len(prefix) :].lstrip()
    text = _TRUNCATION_RE.sub("", text).strip()
    return text


@dataclass(frozen=True, slots=True)
class FetchMcpOpenPageAdapter(BuiltinActionAdapter, OpenPageAdapter):
    tool_type: str = WEB_SEARCH_TOOL
    action_name: str = "open_page"
    adapter_id: str = "fetch_mcp_open_page"
    config_model: type | None = None
    server_label: str = "fetch"

    async def open_page(
        self,
        *,
        ctx: WebSearchAdapterContext,
        url: str,
        options: WebSearchRequestOptions,
    ) -> ActionOutcome[OpenPageActionResult]:
        _ = options
        if ctx.builtin_mcp_runtime_client is None:
            return ActionOutcome(ok=False, error="Built-in MCP runtime client is not configured.")
        try:
            result = await ctx.builtin_mcp_runtime_client.call_tool(
                server_label=self.server_label,
                tool_name="fetch",
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
                text=_normalize_fetch_output(url, result.output_text or ""),
            ),
        )
