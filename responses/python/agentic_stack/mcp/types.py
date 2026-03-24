from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

McpMode: TypeAlias = Literal["hosted", "request_remote"]


@dataclass(frozen=True, slots=True, eq=False)
class McpToolRef:
    server_label: str
    tool_name: str
    mode: McpMode = "hosted"

    def __hash__(self) -> int:
        return hash((self.mode, self.server_label, self.tool_name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, McpToolRef):
            return False
        return (self.mode, self.server_label, self.tool_name) == (
            other.mode,
            other.server_label,
            other.tool_name,
        )


@dataclass(frozen=True, slots=True)
class RequestRemoteMcpServerBinding:
    mode: Literal["request_remote"]
    server_label: str
    server_url: str
    authorization: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class McpExecutionResult:
    ok: bool
    output_text: str | None
    error_text: str | None


@dataclass(frozen=True, slots=True)
class McpToolInfo:
    name: str
    description: str | None
    input_schema: dict[str, object]


@dataclass(frozen=True, slots=True)
class McpServerInfo:
    server_label: str
    enabled: bool
    available: bool
    required: bool
    transport: str
