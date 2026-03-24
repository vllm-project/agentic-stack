from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from agentic_stack.mcp.runtime_client import BuiltinMcpRuntimeClient
from agentic_stack.tools.base.types import RuntimeRequirement
from agentic_stack.tools.web_search.types import (
    ActionOutcome,
    OpenPageActionResult,
    SearchActionResult,
    WebSearchRequestOptions,
)


@dataclass(frozen=True, slots=True)
class WebSearchAdapterContext:
    builtin_mcp_runtime_client: BuiltinMcpRuntimeClient | None


@dataclass(frozen=True, slots=True)
class SearchAdapterHintSupport:
    search_context_size: bool = False
    user_location: bool = False


class SearchAdapter(Protocol):
    hint_support: SearchAdapterHintSupport

    async def search(
        self,
        *,
        ctx: WebSearchAdapterContext,
        query: str,
        queries: tuple[str, ...],
        options: WebSearchRequestOptions,
    ) -> ActionOutcome[SearchActionResult]: ...


class OpenPageAdapter(Protocol):
    async def open_page(
        self,
        *,
        ctx: WebSearchAdapterContext,
        url: str,
        options: WebSearchRequestOptions,
    ) -> ActionOutcome[OpenPageActionResult]: ...


def builtin_mcp_requirement(server_label: str) -> Sequence[RuntimeRequirement]:
    return (RuntimeRequirement(kind="builtin_mcp_server", key=server_label),)
