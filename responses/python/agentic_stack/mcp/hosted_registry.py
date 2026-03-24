from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from agentic_stack.mcp.config import McpRuntimeConfig, McpServerRuntimeConfig
from agentic_stack.mcp.fastmcp_runtime import (
    build_fastmcp_toolset_from_server_entry,
    extract_mcp_tool_infos,
)
from agentic_stack.mcp.types import McpServerInfo, McpToolInfo
from agentic_stack.mcp.utils import (
    is_mcp_tool_keyerror,
    redact_and_truncate_error_text,
    truncate_error_text,
)
from agentic_stack.observability.metrics import record_mcp_server_startup
from agentic_stack.utils.exceptions import BadInputError


@dataclass(slots=True)
class _ServerState:
    config: McpServerRuntimeConfig
    server: AbstractToolset[Any] | None = None
    startup_error: str | None = None
    allowed_tools: dict[str, McpToolInfo] = field(default_factory=dict)
    allowed_mcp_tools_by_name: dict[str, ToolsetTool[Any]] = field(default_factory=dict)


class HostedMcpToolNotFoundError(KeyError):
    pass


class HostedMcpStaleToolError(KeyError):
    pass


class HostedMCPRegistry:
    def __init__(
        self,
        *,
        config: McpRuntimeConfig,
        startup_timeout_s: float,
        tool_timeout_s: float,
    ) -> None:
        self._config = config
        self._startup_timeout_s = float(startup_timeout_s)
        self._tool_timeout_s = float(tool_timeout_s)
        self._states: dict[str, _ServerState] = {
            label: _ServerState(config=server_cfg)
            for label, server_cfg in config.mcp_servers.items()
        }
        self._startup_lock = asyncio.Lock()
        self._started = False

    async def startup(self) -> None:
        if self._started:
            return

        async with self._startup_lock:
            if self._started:
                return
            if not self.is_enabled():
                self._started = True
                return

            for label, state in self._states.items():
                try:
                    toolset = self._build_toolset(server_label=label, config=state.config)
                    await toolset.__aenter__()
                    state.server = toolset

                    mcp_tools_by_name = await asyncio.wait_for(
                        toolset.get_tools(ctx=None), timeout=self._startup_timeout_s
                    )
                    tool_infos = extract_mcp_tool_infos(mcp_tools_by_name)
                    state.allowed_tools = dict(tool_infos)
                    if not state.allowed_tools:
                        raise BadInputError(
                            f"MCP server {label!r} has an empty final allowed tool set."
                        )
                    state.allowed_mcp_tools_by_name = {
                        tool_name: tool
                        for tool_name, tool in mcp_tools_by_name.items()
                        if tool_name in state.allowed_tools
                    }

                    state.startup_error = None
                    record_mcp_server_startup(server_label=label, status="ok")
                except asyncio.TimeoutError:
                    await self._fail_server_startup(
                        state=state,
                        server_label=label,
                        startup_error=truncate_error_text(
                            f"MCP server startup timed out after {self._startup_timeout_s:g}s."
                        ),
                    )
                except Exception as exc:
                    await self._fail_server_startup(
                        state=state,
                        server_label=label,
                        startup_error=redact_and_truncate_error_text(
                            text=str(exc).strip() or exc.__class__.__name__,
                            secret_values=state.config.mcp_server_entry.secret_values_for_redaction(),
                        ),
                    )

            self._started = True

    async def shutdown(self) -> None:
        for state in self._states.values():
            await self._exit_state_server(state)
        self._started = False

    def is_enabled(self) -> bool:
        return self._config.enabled

    def has_server(self, server_label: str) -> bool:
        return server_label in self._states

    def is_server_available(self, server_label: str) -> bool:
        state = self._states.get(server_label)
        return bool(
            self.is_enabled()
            and state is not None
            and state.server is not None
            and state.startup_error is None
        )

    def list_servers(self) -> list[McpServerInfo]:
        if not self.is_enabled():
            return []

        result = []
        for label in sorted(self._states.keys()):
            state = self._states[label]
            transport = state.config.mcp_server_entry.server_transport()
            result.append(
                McpServerInfo(
                    server_label=label,
                    enabled=True,
                    available=self.is_server_available(label),
                    required=False,
                    transport=transport,
                )
            )
        return result

    def get_server_startup_error(self, server_label: str) -> str | None:
        state = self._states.get(server_label)
        return state.startup_error if state is not None else None

    def get_server_secret_values(self, server_label: str) -> tuple[str, ...]:
        state = self._get_state(server_label)
        return state.config.mcp_server_entry.secret_values_for_redaction()

    async def list_tools(self, server_label: str) -> list[McpToolInfo]:
        await self._ensure_started()
        state = self._require_enabled_available_state(server_label)
        return [state.allowed_tools[name] for name in sorted(state.allowed_tools)]

    async def call_tool_with_refresh(
        self,
        *,
        server_label: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> Any:
        await self._ensure_started()
        state = self._require_enabled_available_state(server_label)
        toolset = state.server
        if toolset is None:  # pragma: no cover - defensive
            raise BadInputError(f"MCP server {server_label!r} is unavailable.")

        try:
            return await self._call_tool_once(
                state=state,
                tool_name=tool_name,
                arguments=arguments,
            )
        except HostedMcpStaleToolError:
            refreshed_tools = await toolset.get_tools(ctx=None)
            refreshed_tool = refreshed_tools.get(tool_name)
            if refreshed_tool is None:
                state.allowed_mcp_tools_by_name.pop(tool_name, None)
                state.allowed_tools.pop(tool_name, None)
                raise HostedMcpToolNotFoundError(tool_name) from None

            # Persist the refreshed handle/info so future calls do not repeatedly refresh.
            state.allowed_mcp_tools_by_name[tool_name] = refreshed_tool
            refreshed_tool_def = refreshed_tool.tool_def
            state.allowed_tools[tool_name] = McpToolInfo(
                name=refreshed_tool_def.name,
                description=refreshed_tool_def.description,
                input_schema=refreshed_tool_def.parameters_json_schema,
            )

            return await toolset.call_tool(
                tool_name,
                dict(arguments),
                ctx=None,
                tool=refreshed_tool,
            )

    async def _ensure_started(self) -> None:
        if not self._started:
            await self.startup()

    async def _fail_server_startup(
        self,
        *,
        state: _ServerState,
        server_label: str,
        startup_error: str,
    ) -> None:
        await self._exit_state_server(state)
        state.startup_error = startup_error
        state.allowed_tools.clear()
        state.allowed_mcp_tools_by_name.clear()
        record_mcp_server_startup(server_label=server_label, status="error")

    async def _call_tool_once(
        self,
        *,
        state: _ServerState,
        tool_name: str,
        arguments: dict[str, object],
    ) -> Any:
        toolset = state.server
        if toolset is None:  # pragma: no cover - defensive
            raise BadInputError("MCP server is unavailable.")

        mcp_tool = state.allowed_mcp_tools_by_name.get(tool_name)
        if mcp_tool is None:
            raise HostedMcpToolNotFoundError(tool_name)

        try:
            return await toolset.call_tool(
                tool_name,
                dict(arguments),
                ctx=None,
                tool=mcp_tool,
            )
        except KeyError as exc:
            if is_mcp_tool_keyerror(exc, tool_name):
                raise HostedMcpStaleToolError(tool_name) from exc
            raise

    def _get_state(self, server_label: str) -> _ServerState:
        state = self._states.get(server_label)
        if state is None:
            raise BadInputError(f"Unknown MCP server_label: {server_label}")
        return state

    def _require_enabled_available_state(self, server_label: str) -> _ServerState:
        state = self._get_state(server_label)
        if not self.is_server_available(server_label):
            startup_error = state.startup_error
            if startup_error:
                raise BadInputError(
                    f"MCP server {server_label!r} is currently unavailable: {startup_error}"
                )
            raise BadInputError(f"MCP server {server_label!r} is currently unavailable.")
        return state

    def _build_toolset(
        self,
        *,
        server_label: str,
        config: McpServerRuntimeConfig,
    ) -> AbstractToolset[Any]:
        return build_fastmcp_toolset_from_server_entry(
            server_label=server_label,
            server_entry=config.mcp_server_entry.model_dump(exclude_none=True, round_trip=True),
            timeout_s=self._tool_timeout_s,
            init_timeout_s=self._startup_timeout_s,
        )

    @staticmethod
    async def _exit_state_server(state: _ServerState) -> None:
        server = state.server
        if server is not None:
            try:
                await server.__aexit__(None, None, None)
            except Exception:
                # Best-effort cleanup during shutdown/startup unwind.
                pass
        state.server = None
