from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter

from agentic_stack.utils.io import json_loads


class WebSearchSource(BaseModel):
    type: Literal["url"] = "url"
    url: str


class SearchActionPublic(BaseModel):
    type: Literal["search"] = "search"
    query: str
    queries: list[str] | None = None
    sources: list[WebSearchSource] | None = None


class OpenPageActionPublic(BaseModel):
    type: Literal["open_page"] = "open_page"
    url: str | None = None


class FindInPageActionPublic(BaseModel):
    type: Literal["find_in_page"] = "find_in_page"
    url: str | None = None
    pattern: str


WebSearchActionPublic = SearchActionPublic | OpenPageActionPublic | FindInPageActionPublic


class SearchResultRecord(BaseModel):
    url: str
    title: str | None = None
    snippet: str | None = None


class OpenPageResultRecord(BaseModel):
    url: str | None = None
    title: str | None = None
    text: str | None = None


class FindInPageMatch(BaseModel):
    start: int
    end: int
    text: str


class SearchToolResult(BaseModel):
    action: SearchActionPublic
    results: list[SearchResultRecord] = Field(default_factory=list)
    error: str | None = None


class OpenPageToolResult(BaseModel):
    action: OpenPageActionPublic
    page: OpenPageResultRecord
    error: str | None = None


class FindInPageToolResult(BaseModel):
    action: FindInPageActionPublic
    matches: list[FindInPageMatch] = Field(default_factory=list)
    error: str | None = None


WebSearchToolResult = SearchToolResult | OpenPageToolResult | FindInPageToolResult
_WEB_SEARCH_TOOL_RESULT_ADAPTER = TypeAdapter(WebSearchToolResult)


@dataclass(frozen=True, slots=True)
class SearchActionResult:
    query: str
    queries: tuple[str, ...]
    results: tuple[SearchResultRecord, ...]
    sources: tuple[WebSearchSource, ...]


@dataclass(frozen=True, slots=True)
class OpenPageActionResult:
    url: str | None
    title: str | None
    text: str | None


@dataclass(frozen=True, slots=True)
class FindInPageActionResult:
    url: str | None
    pattern: str
    matches: tuple[FindInPageMatch, ...]


@dataclass(frozen=True, slots=True)
class ActionOutcome[T]:
    ok: bool
    data: T | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WebSearchRequestOptions:
    allowed_domains: tuple[str, ...] = ()
    search_context_size: Literal["low", "medium", "high"] = "medium"
    user_location: dict[str, str | None] | None = None


@dataclass(frozen=True, slots=True)
class WebSearchActionRequest:
    action: Literal["search", "open_page", "find_in_page"]
    query: str | None = None
    queries: tuple[str, ...] = ()
    url: str | None = None
    pattern: str | None = None


@dataclass(slots=True)
class CachedPage:
    url: str
    title: str | None = None
    text: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


def parse_web_search_tool_result(raw: str) -> WebSearchToolResult:
    payload = json_loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("web_search tool result must be a JSON object.")
    return _WEB_SEARCH_TOOL_RESULT_ADAPTER.validate_python(payload)
