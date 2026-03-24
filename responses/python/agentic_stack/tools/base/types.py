from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from agentic_stack.configs.sources import EnvSource
from agentic_stack.mcp.runtime_client import BuiltinMcpRuntimeClient


@runtime_checkable
class BuiltinActionAdapter(Protocol):
    tool_type: str
    action_name: str
    adapter_id: str
    config_model: type[BaseModel] | None


@runtime_checkable
class ProfiledBuiltinProfileResolutionProvider(Protocol):
    def resolve(self, profile_id: str) -> "ResolvedProfiledBuiltinTool": ...

    def validate_profile(self, profile_id: str | None) -> None: ...

    def required_mcp_definitions(
        self,
        profile_id: str | None,
    ) -> Sequence["BuiltinMcpServerDefinition"]: ...


@dataclass(frozen=True, slots=True)
class ActionBindingSpec:
    action_name: str
    adapter_id: str


@dataclass(frozen=True, slots=True)
class RuntimeRequirement:
    kind: Literal["builtin_mcp_server"]
    key: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class BuiltinMcpServerDefinition:
    server_label: str
    # Raw hosted MCP server entry. This matches the JSON-object shape accepted
    # under `mcpServers.<label>` when the built-in definition is fully static.
    server_entry: dict[str, object] | None = None
    # Optional for built-ins whose final hosted entry depends on operator env,
    # such as API-key-backed MCP servers.
    build_server_entry: Callable[[EnvSource], dict[str, object]] | None = None

    def __post_init__(self) -> None:
        if (self.server_entry is None) == (self.build_server_entry is None):
            raise ValueError(
                "BuiltinMcpServerDefinition requires exactly one of "
                "`server_entry` or `build_server_entry`."
            )


@dataclass(frozen=True, slots=True)
class ResolvedActionBinding:
    action_name: str
    adapter_id: str
    requirement_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolvedProfiledBuiltinTool:
    tool_type: str
    profile_id: str
    action_bindings: tuple[ResolvedActionBinding, ...]
    runtime_requirements: tuple[RuntimeRequirement, ...]


@dataclass(frozen=True, slots=True)
class BoundRuntimeRequirements:
    builtin_mcp_runtime_client: BuiltinMcpRuntimeClient | None = None
    builtin_mcp_server_labels: tuple[str, ...] = ()
