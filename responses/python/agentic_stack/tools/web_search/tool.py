from __future__ import annotations

from typing import Literal

from loguru import logger

from agentic_stack.tools import register
from agentic_stack.tools.ids import WEB_SEARCH_TOOL
from agentic_stack.tools.runtime import require_tool_runtime_context
from agentic_stack.tools.web_search.types import WebSearchActionRequest
from agentic_stack.utils.exceptions import BadInputError
from agentic_stack.utils.io import json_dumps


def register_web_search_tool() -> None:
    register(WEB_SEARCH_TOOL)(run_web_search)


def require_web_search_runtime():
    tool_runtime_context = require_tool_runtime_context()
    web_search_runtime = tool_runtime_context.web_search
    if web_search_runtime is None:
        raise BadInputError("`web_search` is not enabled for this request.")
    return web_search_runtime


def normalize_action_request(
    *,
    action: Literal["search", "open_page", "find_in_page"] | None,
    query: str | None,
    queries: list[str] | None,
    url: str | None,
    pattern: str | None,
) -> WebSearchActionRequest:
    effective_action = action
    if effective_action is None:
        if pattern:
            effective_action = "find_in_page"
        elif url:
            effective_action = "open_page"
        else:
            effective_action = "search"
    return WebSearchActionRequest(
        action=effective_action,
        query=query,
        queries=tuple(queries or ()),
        url=url,
        pattern=pattern,
    )


async def run_web_search(
    action: Literal["search", "open_page", "find_in_page"] | None = None,
    query: str | None = None,
    queries: list[str] | None = None,
    url: str | None = None,
    pattern: str | None = None,
) -> str:
    """
    Search the web, open a page, or find text inside an opened page.

    Use:
    - `action="search"` with `query`
    - `action="open_page"` with `url`
    - `action="find_in_page"` with `url` and `pattern`

    The tool returns JSON containing the public action summary plus normalized
    data for the model.
    """
    runtime = require_web_search_runtime()
    action_request = normalize_action_request(
        action=action,
        query=query,
        queries=queries,
        url=url,
        pattern=pattern,
    )
    result = await runtime.executor.execute(action_request)
    payload = result.model_dump(mode="python", exclude_none=True)
    logger.debug("web_search result: {}", payload)
    return json_dumps(payload)
