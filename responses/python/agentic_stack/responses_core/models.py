from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ItemKind = Literal[
    "message",
    "reasoning",
    "function_call",
    "code_interpreter_call",
    "mcp_call",
    "web_search_call",
]


@dataclass(frozen=True, slots=True)
class MessageStarted:
    item_key: str


@dataclass(frozen=True, slots=True)
class MessageDelta:
    item_key: str
    delta: str


@dataclass(frozen=True, slots=True)
class MessageDone:
    item_key: str
    text: str


@dataclass(frozen=True, slots=True)
class ReasoningStarted:
    item_key: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    item_key: str
    delta: str


@dataclass(frozen=True, slots=True)
class ReasoningDone:
    item_key: str
    text: str


@dataclass(frozen=True, slots=True)
class FunctionCallStarted:
    item_key: str
    call_id: str
    name: str
    initial_arguments_json: str


@dataclass(frozen=True, slots=True)
class FunctionCallArgumentsDelta:
    item_key: str
    delta: str


@dataclass(frozen=True, slots=True)
class FunctionCallDone:
    item_key: str
    arguments_json: str


@dataclass(frozen=True, slots=True)
class CodeInterpreterCallStarted:
    item_key: str
    initial_code: str | None


@dataclass(frozen=True, slots=True)
class CodeInterpreterCallCodeDelta:
    item_key: str
    delta: str


@dataclass(frozen=True, slots=True)
class CodeInterpreterCallCodeDone:
    item_key: str
    code: str | None


@dataclass(frozen=True, slots=True)
class CodeInterpreterCallInterpreting:
    item_key: str


@dataclass(frozen=True, slots=True)
class CodeInterpreterCallCompleted:
    item_key: str
    stdout: str | None
    stderr: str | None
    result: str | None


@dataclass(frozen=True, slots=True)
class WebSearchCallStarted:
    item_key: str


@dataclass(frozen=True, slots=True)
class WebSearchCallSearching:
    item_key: str


@dataclass(frozen=True, slots=True)
class WebSearchCallCompleted:
    item_key: str
    action_type: Literal["search", "open_page", "find_in_page"]
    query: str | None = None
    queries: tuple[str, ...] = ()
    sources: tuple[dict[str, str], ...] = ()
    url: str | None = None
    pattern: str | None = None


@dataclass(frozen=True, slots=True)
class McpCallStarted:
    item_key: str
    server_label: str
    name: str
    initial_arguments_json: str
    mode: Literal["hosted", "request_remote"] = "hosted"


@dataclass(frozen=True, slots=True)
class McpCallArgumentsDelta:
    item_key: str
    delta: str


@dataclass(frozen=True, slots=True)
class McpCallArgumentsDone:
    item_key: str
    arguments_json: str


@dataclass(frozen=True, slots=True)
class McpCallCompleted:
    item_key: str
    output_text: str


@dataclass(frozen=True, slots=True)
class McpCallFailed:
    item_key: str
    error_text: str


@dataclass(frozen=True, slots=True)
class UsageFinal:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    incomplete_reason: Literal["max_output_tokens", "content_filter"] | None = None


NormalizedEvent = (
    MessageStarted
    | MessageDelta
    | MessageDone
    | ReasoningStarted
    | ReasoningDelta
    | ReasoningDone
    | FunctionCallStarted
    | FunctionCallArgumentsDelta
    | FunctionCallDone
    | CodeInterpreterCallStarted
    | CodeInterpreterCallCodeDelta
    | CodeInterpreterCallCodeDone
    | CodeInterpreterCallInterpreting
    | CodeInterpreterCallCompleted
    | WebSearchCallStarted
    | WebSearchCallSearching
    | WebSearchCallCompleted
    | McpCallStarted
    | McpCallArgumentsDelta
    | McpCallArgumentsDone
    | McpCallCompleted
    | McpCallFailed
    | UsageFinal
)
