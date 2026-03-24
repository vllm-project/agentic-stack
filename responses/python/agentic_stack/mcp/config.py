from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypeAlias
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictStr

SERVER_LABEL_REGEX = re.compile(r"^[a-zA-Z0-9_-]+$")
_COMMAND_STYLE_KEYS = {"command", "args", "env", "cwd"}


class McpServerRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mcp_server_entry: McpServerEntry


class McpRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mcp_servers: dict[str, McpServerRuntimeConfig] = Field(default_factory=dict)


class McpServerEntryBase(BaseModel):
    model_config = ConfigDict(extra="allow")

    transport: StrictStr | None = None
    authentication: dict[StrictStr, object] | None = None

    def server_transport(self) -> str:
        return self.transport or "auto"

    @staticmethod
    def _append_secret_variants(value: str, out: list[str]) -> None:
        if not value:
            return
        out.append(value)
        if value.lower().startswith("bearer "):
            token = value[7:].strip()
            if token:
                out.append(token)

    @staticmethod
    def _append_nested_secret_variants(value: object, out: list[str]) -> None:
        if isinstance(value, str):
            McpServerEntryBase._append_secret_variants(value, out)
            return
        if isinstance(value, dict):
            for nested in value.values():
                McpServerEntryBase._append_nested_secret_variants(nested, out)
            return
        if isinstance(value, (list, tuple, set)):
            for nested in value:
                McpServerEntryBase._append_nested_secret_variants(nested, out)

    @staticmethod
    def _finalize_secret_values(values: list[str]) -> tuple[str, ...]:
        return tuple(sorted(set(values), key=len, reverse=True))


class HttpMcpServerEntry(McpServerEntryBase):
    url: StrictStr
    headers: dict[StrictStr, StrictStr] | None = None
    auth: StrictStr | None = None

    def secret_values_for_redaction(self) -> tuple[str, ...]:
        values: list[str] = []
        if self.headers is not None:
            for value in self.headers.values():
                self._append_secret_variants(value, values)
        if self.auth:
            self._append_secret_variants(self.auth, values)
        if self.authentication is not None:
            self._append_nested_secret_variants(self.authentication, values)
        return self._finalize_secret_values(values)


class StdioMcpServerEntry(McpServerEntryBase):
    command: StrictStr
    args: list[StrictStr] | None = None
    env: dict[StrictStr, StrictStr] | None = None
    cwd: StrictStr | None = None

    def server_transport(self) -> str:
        return "stdio"

    def secret_values_for_redaction(self) -> tuple[str, ...]:
        values: list[str] = []
        if self.env is not None:
            for value in self.env.values():
                if value:
                    values.append(value)
        if self.authentication is not None:
            self._append_nested_secret_variants(self.authentication, values)
        return self._finalize_secret_values(values)


McpServerEntry: TypeAlias = HttpMcpServerEntry | StdioMcpServerEntry


def split_hosted_server_entry(
    server_entry: object,
) -> McpServerEntry:
    if not isinstance(server_entry, dict):
        raise ValueError("Each hosted MCP server entry must be a JSON object.")

    mcp_server_entry = dict(server_entry)

    transport = mcp_server_entry.get("transport")
    if isinstance(transport, dict):
        transport_type = transport.get("type")
        if isinstance(transport_type, str) and transport_type.strip().lower() == "stdio":
            raise ValueError(
                "Hosted MCP does not accept nested `transport` objects. "
                "For stdio servers, move `command`/`args`/`env`/`cwd` to top-level."
            )
        raise ValueError(
            "Hosted MCP `transport` must be a string when provided; "
            "nested `transport` objects are not supported."
        )

    has_command_style_keys = any(key in mcp_server_entry for key in _COMMAND_STYLE_KEYS)
    if (
        not has_command_style_keys
        and "url" not in mcp_server_entry
        and isinstance(transport, str)
        and transport.strip().lower() == "stdio"
    ):
        raise ValueError(
            "Hosted MCP stdio transport requires command-style fields; provide at least `command`."
        )

    if has_command_style_keys:
        return _validate_stdio_entry(mcp_server_entry)
    return _validate_http_entry(mcp_server_entry)


def _validate_stdio_entry(mcp_server_entry: dict[str, object]) -> StdioMcpServerEntry:
    if "url" in mcp_server_entry or "headers" in mcp_server_entry:
        raise ValueError(
            "Hosted MCP stdio entry must not include HTTP-only fields `url` or `headers`."
        )

    command = mcp_server_entry.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("Hosted MCP stdio entry must include a non-empty `command`.")

    args = mcp_server_entry.get("args")
    if args is not None:
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError("Hosted MCP stdio `args` must be a list of strings when provided.")

    env = mcp_server_entry.get("env")
    if env is not None:
        if not isinstance(env, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in env.items()
        ):
            raise ValueError(
                "Hosted MCP stdio `env` must be an object of string-to-string pairs when provided."
            )

    cwd = mcp_server_entry.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise ValueError("Hosted MCP stdio `cwd` must be a string when provided.")

    transport = mcp_server_entry.get("transport")
    if transport is not None:
        if not isinstance(transport, str) or not transport.strip():
            raise ValueError("Hosted MCP `transport` must be a non-empty string when provided.")
        if transport.strip().lower() != "stdio":
            raise ValueError(
                "Hosted MCP stdio `transport` must be `stdio` when command-style fields are used."
            )

    return StdioMcpServerEntry.model_validate(mcp_server_entry)


def _validate_http_entry(mcp_server_entry: dict[str, object]) -> HttpMcpServerEntry:
    url = mcp_server_entry.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Hosted MCP server entry must include a non-empty `url`.")

    try:
        parsed_url = urlsplit(url)
    except ValueError as exc:
        raise ValueError(f"Invalid hosted MCP `url`: {exc}") from exc

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("Hosted MCP `url` must be absolute.")

    if parsed_url.scheme.lower() not in {"http", "https"}:
        raise ValueError("Hosted MCP `url` must use `http` or `https`.")

    headers = mcp_server_entry.get("headers")
    if headers is not None:
        if not isinstance(headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in headers.items()
        ):
            raise ValueError("Hosted MCP `headers` must be an object of string-to-string pairs.")

    auth = mcp_server_entry.get("auth")
    if auth is not None and not isinstance(auth, str):
        raise ValueError("Hosted MCP `auth` must be a string when provided.")

    transport = mcp_server_entry.get("transport")
    if transport is not None:
        if not isinstance(transport, str) or not transport.strip():
            raise ValueError("Hosted MCP `transport` must be a non-empty string when provided.")
        if transport.strip().lower() == "stdio":
            raise ValueError("URL-style hosted MCP entries must not set `transport` to `stdio`.")

    return HttpMcpServerEntry.model_validate(mcp_server_entry)


def load_mcp_runtime_config_from_obj(raw: object) -> McpRuntimeConfig:
    if not isinstance(raw, dict):
        raise ValueError("MCP config must be a JSON object.")

    if "mcpServers" not in raw:
        raise ValueError("Hosted config must use `mcpServers` root key.")

    raw_servers = raw.get("mcpServers")
    if not isinstance(raw_servers, dict):
        raise ValueError("`mcpServers` must be an object keyed by server label.")

    parsed_servers: dict[str, McpServerRuntimeConfig] = {}
    for server_label, server_obj in raw_servers.items():
        if not isinstance(server_label, str) or not SERVER_LABEL_REGEX.fullmatch(server_label):
            raise ValueError(
                "Invalid MCP server label. Labels must match "
                f"{SERVER_LABEL_REGEX.pattern!r}. Received: {server_label!r}"
            )

        mcp_server_entry = split_hosted_server_entry(server_obj)
        parsed_servers[server_label] = McpServerRuntimeConfig(
            mcp_server_entry=mcp_server_entry,
        )

    return McpRuntimeConfig(enabled=True, mcp_servers=parsed_servers)


def load_mcp_runtime_config(path: str | None) -> McpRuntimeConfig:
    if path is None or not path.strip():
        return McpRuntimeConfig(enabled=False, mcp_servers={})

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return load_mcp_runtime_config_from_obj(raw)


def merge_mcp_runtime_configs(*configs: McpRuntimeConfig) -> McpRuntimeConfig:
    merged_servers: dict[str, McpServerRuntimeConfig] = {}
    for config in configs:
        if not config.enabled:
            continue
        merged_servers.update(config.mcp_servers)
    return McpRuntimeConfig(enabled=bool(merged_servers), mcp_servers=merged_servers)
