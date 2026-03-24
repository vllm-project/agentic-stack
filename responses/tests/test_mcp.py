from __future__ import annotations

import json
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from openai import APIError as OpenAIAPIError
from openai import AsyncOpenAI
from pydantic import TypeAdapter
from pydantic_ai import UsageLimits
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from sse_test_utils import extract_completed_response, parse_sse_frames, parse_sse_json_events

from agentic_stack.configs.sources import EnvSource
from agentic_stack.entrypoints import llm as mock_llm
from agentic_stack.entrypoints._state import VRAppState
from agentic_stack.mcp.config import (
    McpRuntimeConfig,
    load_mcp_runtime_config,
    merge_mcp_runtime_configs,
    split_hosted_server_entry,
)
from agentic_stack.mcp.gateway_toolset import McpGatewayToolset
from agentic_stack.mcp.runtime_client import (
    BuiltinMcpRuntimeTransportError,
    BuiltinMcpRuntimeUnavailableServerError,
    BuiltinMcpRuntimeUnknownServerError,
)
from agentic_stack.mcp.types import (
    McpExecutionResult,
    McpServerInfo,
    McpToolInfo,
    McpToolRef,
    RequestRemoteMcpServerBinding,
)
from agentic_stack.mcp.utils import (
    build_mcp_tool_result_payload,
    canonicalize_output_text,
    parse_mcp_tool_result_payload,
)
from agentic_stack.responses_core.composer import ResponseComposer
from agentic_stack.responses_core.models import (
    McpCallArgumentsDelta,
    McpCallArgumentsDone,
    McpCallCompleted,
    McpCallFailed,
    McpCallStarted,
    UsageFinal,
)
from agentic_stack.responses_core.normalizer import PydanticAINormalizer
from agentic_stack.tools.ids import WEB_SEARCH_TOOL
from agentic_stack.tools.profile_resolution import (
    build_builtin_mcp_runtime_config,
    profiled_builtin_requires_mcp,
)
from agentic_stack.types.openai import OpenAIResponsesResponse, vLLMResponsesRequest
from agentic_stack.utils.cassette_replay import load_cassette_yaml


def _chat_completion_cassettes_dir() -> Path:
    return Path(__file__).resolve().parent / "cassettes" / "chat_completion"


# ==============================
# Fixture-shape guard
# ==============================


def test_mcp_hosted_step1_stream_cassette_has_tool_call_deltas() -> None:
    cassette = load_cassette_yaml(
        _chat_completion_cassettes_dir() / "mcp-hosted-step1-stream.yaml"
    )
    assert cassette.request.method == "POST"
    assert cassette.request.path == "/v1/chat/completions"
    assert bool(cassette.request.body.get("stream")) is True
    assert cassette.response.sse is not None

    sse = "\n".join(cassette.response.sse)
    assert '"tool_calls"' in sse
    assert '"finish_reason":"tool_calls"' in sse
    assert "[DONE]" in sse


def test_mcp_hosted_step2_stream_cassette_has_final_assistant_completion() -> None:
    cassette = load_cassette_yaml(
        _chat_completion_cassettes_dir() / "mcp-hosted-step2-stream.yaml"
    )
    assert cassette.request.method == "POST"
    assert cassette.request.path == "/v1/chat/completions"
    assert bool(cassette.request.body.get("stream")) is True
    assert cassette.response.sse is not None

    sse = "\n".join(cassette.response.sse)
    assert '"finish_reason":"stop"' in sse
    assert "[DONE]" in sse


# ==============================
# Codec
# ==============================


def test_parse_mcp_tool_result_payload_validates_structure() -> None:
    ref, result = parse_mcp_tool_result_payload(
        {
            "kind": "agentic_stack_mcp_result",
            "server_label": "local_fs",
            "tool_name": "list_directory",
            "ok": True,
            "output_text": "{}",
        }
    )
    assert ref == McpToolRef(server_label="local_fs", tool_name="list_directory")
    assert result.ok is True
    assert result.output_text == "{}"
    assert result.error_text is None

    with pytest.raises(ValueError, match="invalid"):
        parse_mcp_tool_result_payload(
            {
                "kind": "agentic_stack_mcp_result",
                "server_label": "",
                "tool_name": "list_directory",
                "ok": True,
            }
        )


def test_canonicalize_output_text_handles_scalars_and_fallbacks() -> None:
    assert canonicalize_output_text(42) == "42"
    assert canonicalize_output_text(True) == "true"
    assert canonicalize_output_text(None) == ""
    assert canonicalize_output_text({}) == ""

    class _Unserializable:
        def __str__(self) -> str:
            return "custom-object"

    assert canonicalize_output_text(_Unserializable()) == "custom-object"


# ==============================
# Config
# ==============================


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_mcp_runtime_config_disabled_when_path_unset() -> None:
    cfg = load_mcp_runtime_config(None)
    assert cfg.enabled is False
    assert cfg.mcp_servers == {}


def test_build_builtin_mcp_runtime_config_derives_shipped_profile_servers() -> None:
    fetch_cfg = build_builtin_mcp_runtime_config(
        tool_type=WEB_SEARCH_TOOL,
        profile_id="duckduckgo_plus_fetch",
    )
    exa_cfg = build_builtin_mcp_runtime_config(
        tool_type=WEB_SEARCH_TOOL,
        profile_id="exa_mcp",
    )
    disabled_cfg = build_builtin_mcp_runtime_config(
        tool_type=WEB_SEARCH_TOOL,
        profile_id=None,
    )

    assert fetch_cfg.enabled is True
    assert set(fetch_cfg.mcp_servers) == {"fetch"}
    assert fetch_cfg.mcp_servers["fetch"].mcp_server_entry.model_dump(
        exclude_none=True, round_trip=True
    ) == {"command": "uvx", "args": ["mcp-server-fetch"]}

    assert exa_cfg.enabled is True
    assert set(exa_cfg.mcp_servers) == {"exa"}
    assert exa_cfg.mcp_servers["exa"].mcp_server_entry.model_dump(
        exclude_none=True, round_trip=True
    ) == {"url": "https://mcp.exa.ai/mcp?tools=web_search_exa,crawling_exa"}

    assert disabled_cfg.enabled is False
    assert disabled_cfg.mcp_servers == {}


def test_build_builtin_mcp_runtime_config_injects_exa_api_key_from_env() -> None:
    exa_cfg = build_builtin_mcp_runtime_config(
        tool_type=WEB_SEARCH_TOOL,
        profile_id="exa_mcp",
        env=EnvSource(environ={"EXA_API_KEY": "exa-secret-key"}),
    )

    assert exa_cfg.enabled is True
    assert exa_cfg.mcp_servers["exa"].mcp_server_entry.model_dump(
        exclude_none=True, round_trip=True
    ) == {
        "url": "https://mcp.exa.ai/mcp?tools=web_search_exa,crawling_exa&exaApiKey=exa-secret-key"
    }


def test_merge_mcp_runtime_configs_generic_entries_override_builtin_defaults() -> None:
    builtin_cfg = build_builtin_mcp_runtime_config(
        tool_type=WEB_SEARCH_TOOL,
        profile_id="duckduckgo_plus_fetch",
    )
    explicit_fetch_entry = split_hosted_server_entry(
        {
            "command": "custom-fetch",
            "args": ["--stdio"],
        }
    )
    generic_cfg = McpRuntimeConfig(
        enabled=True,
        mcp_servers={
            "fetch": type(builtin_cfg.mcp_servers["fetch"])(
                mcp_server_entry=explicit_fetch_entry,
            ),
            "local_fs": type(builtin_cfg.mcp_servers["fetch"])(
                mcp_server_entry=split_hosted_server_entry(
                    {
                        "command": "uvx",
                        "args": ["mcp-server-filesystem", "/tmp"],
                    }
                ),
            ),
        },
    )

    merged_cfg = merge_mcp_runtime_configs(builtin_cfg, generic_cfg)

    assert merged_cfg.enabled is True
    assert set(merged_cfg.mcp_servers) == {"fetch", "local_fs"}
    assert merged_cfg.mcp_servers["fetch"].mcp_server_entry.model_dump(
        exclude_none=True, round_trip=True
    ) == {"command": "custom-fetch", "args": ["--stdio"]}


def test_profiled_builtin_requires_mcp_tracks_resolved_runtime_requirements() -> None:
    assert profiled_builtin_requires_mcp(tool_type=WEB_SEARCH_TOOL, profile_id="exa_mcp") is True
    assert (
        profiled_builtin_requires_mcp(
            tool_type=WEB_SEARCH_TOOL,
            profile_id="duckduckgo_plus_fetch",
        )
        is True
    )
    with pytest.raises(ValueError, match="Unknown profile"):
        profiled_builtin_requires_mcp(
            tool_type=WEB_SEARCH_TOOL,
            profile_id="missing_profile",
        )


def test_load_mcp_runtime_config_parses_valid_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "github_docs": {
                    "url": "https://mcp.example.com/mcp",
                    "headers": {"Authorization": "Bearer inline-token"},
                    "auth": "Bearer abc",
                },
                "docs_sse": {
                    "url": "https://mcp.example.com/sse",
                    "transport": "sse",
                },
            }
        },
    )

    cfg = load_mcp_runtime_config(str(cfg_path))
    assert cfg.enabled is True
    assert set(cfg.mcp_servers.keys()) == {"github_docs", "docs_sse"}
    github_docs_entry = cfg.mcp_servers["github_docs"].mcp_server_entry.model_dump(
        exclude_none=True, round_trip=True
    )
    docs_sse_entry = cfg.mcp_servers["docs_sse"].mcp_server_entry.model_dump(
        exclude_none=True, round_trip=True
    )
    assert github_docs_entry["url"] == "https://mcp.example.com/mcp"
    assert github_docs_entry["auth"] == "Bearer abc"
    assert docs_sse_entry["transport"] == "sse"


def test_load_mcp_runtime_config_rejects_wrong_root_key(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp-wrong-root.json"
    _write_json(
        cfg_path,
        {
            "mcp_servers": {
                "docs": {
                    "url": "https://mcp.example.com/mcp",
                }
            }
        },
    )

    with pytest.raises(ValueError, match="mcpServers"):
        load_mcp_runtime_config(str(cfg_path))


def test_load_mcp_runtime_config_rejects_invalid_server_label(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp-invalid-label.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "bad label": {
                    "url": "https://mcp.example.com/mcp",
                }
            }
        },
    )

    with pytest.raises(ValueError, match="Invalid MCP server label"):
        load_mcp_runtime_config(str(cfg_path))


def test_load_mcp_runtime_config_preserves_passthrough_server_fields(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp-preserve-passthrough-fields.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "docs": {
                    "url": "https://mcp.example.com/mcp",
                    "authentication": {"type": "bearer", "token": "secret"},
                    "custom_passthrough": {"feature": True},
                }
            }
        },
    )

    cfg = load_mcp_runtime_config(str(cfg_path))
    entry = cfg.mcp_servers["docs"].mcp_server_entry.model_dump(exclude_none=True, round_trip=True)
    assert entry["authentication"] == {"type": "bearer", "token": "secret"}
    assert entry["custom_passthrough"] == {"feature": True}


def test_load_mcp_runtime_config_preserves_unknown_server_fields(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp-preserve-unknown-server-fields.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "docs": {
                    "url": "https://mcp.example.com/mcp",
                    "required": True,
                    "startup_timeout_sec": 5,
                }
            }
        },
    )
    cfg = load_mcp_runtime_config(str(cfg_path))
    entry = cfg.mcp_servers["docs"].mcp_server_entry.model_dump(exclude_none=True, round_trip=True)
    assert entry["required"] is True
    assert entry["startup_timeout_sec"] == 5


def test_load_mcp_runtime_config_preserves_context7_shape_without_transport(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "mcp-context7.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "context7": {
                    "url": "https://mcp.context7.com/mcp",
                    "headers": {"CONTEXT7_API_KEY": "secret"},
                }
            }
        },
    )

    cfg = load_mcp_runtime_config(str(cfg_path))
    assert cfg.enabled is True
    mcp_server_entry = cfg.mcp_servers["context7"].mcp_server_entry.model_dump(
        exclude_none=True, round_trip=True
    )
    assert mcp_server_entry["url"] == "https://mcp.context7.com/mcp"
    assert mcp_server_entry["headers"] == {"CONTEXT7_API_KEY": "secret"}
    assert "transport" not in mcp_server_entry


def test_load_mcp_runtime_config_rejects_http_entry_with_stdio_transport(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp-reject-stdio.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "docs": {
                    "url": "https://mcp.example.com/mcp",
                    "transport": "stdio",
                }
            }
        },
    )
    with pytest.raises(ValueError, match="must not set `transport` to `stdio`"):
        load_mcp_runtime_config(str(cfg_path))


def test_load_mcp_runtime_config_rejects_stdio_transport_without_command_keys(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "mcp-stdio-without-command-keys.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "docs": {
                    "transport": "stdio",
                }
            }
        },
    )
    with pytest.raises(ValueError, match="stdio transport requires command-style fields"):
        load_mcp_runtime_config(str(cfg_path))


def test_load_mcp_runtime_config_rejects_nested_stdio_transport_object(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp-nested-stdio-transport.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "local_fs": {
                    "transport": {
                        "type": "stdio",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    }
                }
            }
        },
    )
    with pytest.raises(ValueError, match="does not accept nested `transport` objects"):
        load_mcp_runtime_config(str(cfg_path))


def test_load_mcp_runtime_config_rejects_nested_transport_object(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp-nested-transport.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "docs": {
                    "transport": {
                        "type": "http",
                    }
                }
            }
        },
    )
    with pytest.raises(ValueError, match="nested `transport` objects are not supported"):
        load_mcp_runtime_config(str(cfg_path))


def test_load_mcp_runtime_config_parses_stdio_entry(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp-stdio.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "docs": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    "env": {"API_TOKEN": "secret"},
                    "cwd": "/srv/mcp",
                    "custom_passthrough": {"feature": True},
                }
            }
        },
    )

    cfg = load_mcp_runtime_config(str(cfg_path))
    entry = cfg.mcp_servers["docs"].mcp_server_entry.model_dump(exclude_none=True, round_trip=True)
    assert entry["command"] == "npx"
    assert entry["args"] == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    assert entry["env"] == {"API_TOKEN": "secret"}
    assert entry["cwd"] == "/srv/mcp"
    assert entry["custom_passthrough"] == {"feature": True}


def test_load_mcp_runtime_config_rejects_stdio_http_mix(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp-stdio-http-mix.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "docs": {
                    "command": "npx",
                    "url": "https://mcp.example.com/mcp",
                }
            }
        },
    )
    with pytest.raises(ValueError, match="must not include HTTP-only fields"):
        load_mcp_runtime_config(str(cfg_path))


def test_load_mcp_runtime_config_rejects_stdio_invalid_args(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp-stdio-invalid-args.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "docs": {
                    "command": "npx",
                    "args": ["-y", 123],
                }
            }
        },
    )
    with pytest.raises(ValueError, match="stdio `args` must be a list of strings"):
        load_mcp_runtime_config(str(cfg_path))


def test_load_mcp_runtime_config_rejects_stdio_missing_command(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp-stdio-missing-command.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "docs": {
                    "args": ["-y"],
                }
            }
        },
    )
    with pytest.raises(ValueError, match="must include a non-empty `command`"):
        load_mcp_runtime_config(str(cfg_path))


def test_load_mcp_runtime_config_rejects_stdio_transport_mismatch(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp-stdio-transport-mismatch.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "docs": {
                    "command": "npx",
                    "transport": "sse",
                }
            }
        },
    )
    with pytest.raises(ValueError, match="stdio `transport` must be `stdio`"):
        load_mcp_runtime_config(str(cfg_path))


def test_load_mcp_runtime_config_rejects_non_http_scheme(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp-reject-scheme.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "docs": {
                    "url": "wss://mcp.example.com/mcp",
                }
            }
        },
    )
    with pytest.raises(ValueError, match="must use `http` or `https`"):
        load_mcp_runtime_config(str(cfg_path))


def test_load_mcp_runtime_config_rejects_non_string_auth(tmp_path: Path) -> None:
    cfg_path = tmp_path / "mcp-reject-auth-object.json"
    _write_json(
        cfg_path,
        {
            "mcpServers": {
                "docs": {
                    "url": "https://mcp.example.com/mcp",
                    "auth": {"token": "abc"},
                }
            }
        },
    )
    with pytest.raises(ValueError, match="`auth` must be a string"):
        load_mcp_runtime_config(str(cfg_path))


def test_split_hosted_server_entry_preserves_passthrough_fields() -> None:
    mcp_server_entry = split_hosted_server_entry(
        {
            "url": "https://mcp.example.com/mcp",
            "headers": {"Authorization": "Bearer token"},
            "custom_passthrough": {"x": 1},
        }
    )
    assert mcp_server_entry.model_dump(exclude_none=True, round_trip=True) == {
        "url": "https://mcp.example.com/mcp",
        "headers": {"Authorization": "Bearer token"},
        "custom_passthrough": {"x": 1},
    }


def test_split_hosted_server_entry_stdio_passthrough() -> None:
    mcp_server_entry = split_hosted_server_entry(
        {
            "command": "npx",
            "args": ["-y", "some-mcp"],
            "custom_passthrough": {"x": 1},
        }
    )
    assert mcp_server_entry.model_dump(exclude_none=True, round_trip=True) == {
        "command": "npx",
        "args": ["-y", "some-mcp"],
        "custom_passthrough": {"x": 1},
    }


# ==============================
# Manager
# ==============================


def _tool_info(
    name: str,
    *,
    description: str | None = None,
    input_schema: dict[str, object] | None = None,
) -> McpToolInfo:
    return McpToolInfo(
        name=name,
        description=description,
        input_schema=input_schema
        if input_schema is not None
        else {"type": "object", "properties": {}, "additionalProperties": True},
    )


class _FakeRuntimeServer:
    def __init__(
        self,
        *,
        tools: list[str] | list[McpToolInfo],
        startup_error: Exception | None = None,
        call_error: Exception | None = None,
        call_result=None,
    ) -> None:
        self._tools = tools
        self._startup_error = startup_error
        self._call_error = call_error
        self._call_result = call_result if call_result is not None else {"ok": True}
        self.enter_calls = 0
        self.exit_calls = 0
        self.get_tools_calls = 0
        self.call_tool_calls = 0

    @property
    def id(self) -> str | None:
        return None

    async def __aenter__(self) -> _FakeRuntimeServer:
        self.enter_calls += 1
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        _ = exc_type, exc, tb
        self.exit_calls += 1

    async def get_tools(self, ctx) -> dict[str, ToolsetTool[Any]]:
        _ = ctx
        self.get_tools_calls += 1
        if self._startup_error is not None:
            raise self._startup_error
        out: dict[str, ToolsetTool[Any]] = {}
        args_validator = TypeAdapter(dict[str, Any]).validator
        for tool in self._tools:
            tool_info: McpToolInfo
            if isinstance(tool, McpToolInfo):
                tool_info = tool
            else:
                tool_info = _tool_info(tool)
            out[tool_info.name] = ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name=tool_info.name,
                    description=tool_info.description,
                    parameters_json_schema=tool_info.input_schema,
                ),
                max_retries=0,
                args_validator=args_validator,
            )
        return out

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx,
        tool: ToolsetTool[Any],
    ) -> Any:
        _ = ctx, tool
        self.call_tool_calls += 1
        if self._call_error is not None:
            raise self._call_error
        if callable(self._call_result):
            return self._call_result(name=name, args=tool_args)
        return self._call_result


def _make_runtime_config(servers: dict) -> McpRuntimeConfig:
    parsed_servers: dict[str, object] = {}
    for server_label, server_entry in servers.items():
        mcp_server_entry = split_hosted_server_entry(server_entry)
        parsed_servers[server_label] = {
            "mcp_server_entry": mcp_server_entry,
        }
    return McpRuntimeConfig.model_validate({"enabled": True, "mcp_servers": parsed_servers})


# ==============================
# Discovery API
# ==============================


@dataclass
class _FakeDiscoveryRegistry:
    enabled: bool
    servers: dict[str, dict]

    def is_enabled(self) -> bool:
        return self.enabled

    def has_server(self, server_label: str) -> bool:
        return server_label in self.servers

    def is_server_available(self, server_label: str) -> bool:
        server = self.servers.get(server_label)
        if server is None:
            return False
        return bool(server.get("available", False))

    async def list_servers(self) -> list[McpServerInfo]:
        result = []
        for label in sorted(self.servers.keys()):
            data = self.servers[label]
            result.append(
                McpServerInfo(
                    server_label=label,
                    enabled=bool(data.get("enabled", True)),
                    available=bool(data.get("available", False)),
                    required=bool(data.get("required", False)),
                    transport=str(data.get("transport", "stdio")),
                )
            )
        return result

    def get_server_startup_error(self, server_label: str) -> str | None:
        server = self.servers.get(server_label)
        if server is None:
            return None
        return server.get("startup_error")

    async def list_tools(self, server_label: str) -> list[McpToolInfo]:
        server = self.servers.get(server_label)
        if server is None:
            raise BuiltinMcpRuntimeUnknownServerError(f"Unknown MCP server_label: {server_label}")
        if not bool(server.get("enabled", True)):
            raise BuiltinMcpRuntimeUnavailableServerError(
                f"MCP server {server_label!r} is disabled."
            )
        if not bool(server.get("available", False)):
            raise BuiltinMcpRuntimeUnavailableServerError(
                f"MCP server {server_label!r} is currently unavailable: "
                f"{server.get('startup_error')}"
            )
        tools: list[McpToolInfo] = []
        for tool in list(server.get("tools", [])):
            if isinstance(tool, McpToolInfo):
                tools.append(tool)
            else:
                tools.append(_tool_info(str(tool)))
        return tools


@pytest.fixture
def mcp_discovery_app() -> FastAPI:
    from agentic_stack.routers import mcp as mcp_router

    app = FastAPI(title="MCP discovery (test)")
    app.state.agentic_stack = VRAppState()
    app.include_router(mcp_router.router)
    return app


@pytest.mark.anyio
async def test_list_servers_returns_configured_inventory(mcp_discovery_app: FastAPI) -> None:
    mcp_discovery_app.state.agentic_stack.builtin_mcp_runtime_client = _FakeDiscoveryRegistry(
        enabled=True,
        servers={
            "github_docs": {
                "enabled": True,
                "available": True,
                "required": True,
                "transport": "streamable_http",
                "tools": ["search_docs", "get_page"],
            },
            "local_fs": {
                "enabled": True,
                "available": False,
                "required": False,
                "transport": "stdio",
                "tools": ["list_files"],
                "startup_error": "dial timeout",
            },
        },
    )

    transport = httpx.ASGITransport(app=mcp_discovery_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        resp = await client.get("/v1/mcp/servers")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["object"] == "list"
    assert [item["server_label"] for item in payload["data"]] == ["github_docs", "local_fs"]
    assert payload["data"][0]["available"] is True
    assert payload["data"][1]["available"] is False


@pytest.mark.anyio
async def test_list_server_tools_returns_404_and_409(mcp_discovery_app: FastAPI) -> None:
    mcp_discovery_app.state.agentic_stack.builtin_mcp_runtime_client = _FakeDiscoveryRegistry(
        enabled=True,
        servers={
            "github_docs": {
                "enabled": True,
                "available": False,
                "required": False,
                "transport": "streamable_http",
                "tools": ["search_docs"],
                "startup_error": "dial timeout",
            }
        },
    )

    transport = httpx.ASGITransport(app=mcp_discovery_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        not_found = await client.get("/v1/mcp/servers/unknown/tools")
        unavailable = await client.get("/v1/mcp/servers/github_docs/tools")

    assert not_found.status_code == 404
    assert unavailable.status_code == 409


@pytest.mark.anyio
async def test_list_server_tools_includes_schema_and_description_with_deterministic_order(
    mcp_discovery_app: FastAPI,
) -> None:
    mcp_discovery_app.state.agentic_stack.builtin_mcp_runtime_client = _FakeDiscoveryRegistry(
        enabled=True,
        servers={
            "github_docs": {
                "enabled": True,
                "available": True,
                "required": False,
                "transport": "streamable_http",
                "tools": [
                    _tool_info(
                        "zeta",
                        description=None,
                        input_schema={
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                            "required": ["q"],
                            "additionalProperties": False,
                        },
                    ),
                    _tool_info(
                        "alpha",
                        description="Search docs",
                        input_schema={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    ),
                ],
            }
        },
    )

    transport = httpx.ASGITransport(app=mcp_discovery_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        resp = await client.get("/v1/mcp/servers/github_docs/tools")

    assert resp.status_code == 200
    payload = resp.json()
    assert [tool["name"] for tool in payload["tools"]] == ["alpha", "zeta"]
    assert payload["tools"][0]["description"] == "Search docs"
    assert payload["tools"][1]["description"] is None
    assert payload["tools"][0]["input_schema"]["required"] == ["query"]


@pytest.mark.anyio
async def test_discovery_returns_empty_or_404_when_mcp_disabled(
    mcp_discovery_app: FastAPI,
) -> None:
    mcp_discovery_app.state.agentic_stack.builtin_mcp_runtime_client = _FakeDiscoveryRegistry(
        enabled=False, servers={}
    )

    transport = httpx.ASGITransport(app=mcp_discovery_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        list_resp = await client.get("/v1/mcp/servers")
        tools_resp = await client.get("/v1/mcp/servers/github_docs/tools")

    assert list_resp.status_code == 200
    assert list_resp.json() == {"object": "list", "data": []}
    assert tools_resp.status_code == 404


@pytest.mark.anyio
async def test_discovery_returns_503_when_builtin_runtime_unreachable(
    mcp_discovery_app: FastAPI,
) -> None:
    class _FakeRuntimeClient:
        def is_enabled(self) -> bool:
            return True

        async def list_servers(self) -> list[McpServerInfo]:
            raise BuiltinMcpRuntimeTransportError("runtime down")

        async def list_tools(self, server_label: str) -> list[McpToolInfo]:
            _ = server_label
            raise BuiltinMcpRuntimeTransportError("runtime down")

    mcp_discovery_app.state.agentic_stack.builtin_mcp_runtime_client = _FakeRuntimeClient()

    transport = httpx.ASGITransport(app=mcp_discovery_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        list_resp = await client.get("/v1/mcp/servers")
        tools_resp = await client.get("/v1/mcp/servers/github_docs/tools")

    assert list_resp.status_code == 503
    assert tools_resp.status_code == 503


@pytest.mark.anyio
async def test_discovery_list_servers_returns_503_when_runtime_returns_unknown_server_error(
    mcp_discovery_app: FastAPI,
) -> None:
    class _FakeRuntimeClient:
        def is_enabled(self) -> bool:
            return True

        async def list_servers(self) -> list[McpServerInfo]:
            raise BuiltinMcpRuntimeUnknownServerError("runtime endpoint mismatch")

        async def list_tools(self, server_label: str) -> list[McpToolInfo]:
            _ = server_label
            return []

    mcp_discovery_app.state.agentic_stack.builtin_mcp_runtime_client = _FakeRuntimeClient()

    transport = httpx.ASGITransport(app=mcp_discovery_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        list_resp = await client.get("/v1/mcp/servers")

    assert list_resp.status_code == 503


# ==============================
# Request contract
# ==============================


_DICT_ARGS_VALIDATOR = TypeAdapter(dict[str, Any]).validator


class _HostedRegistryNativeToolset(AbstractToolset[Any]):
    def __init__(self, *, manager: Any, server_label: str) -> None:
        self._manager = manager
        self._server_label = server_label

    @property
    def id(self) -> str | None:
        return None

    async def get_tools(self, ctx) -> dict[str, ToolsetTool[Any]]:  # noqa: ARG002
        out: dict[str, ToolsetTool[Any]] = {}
        tools = await self._manager.list_tools(self._server_label)
        for tool in sorted(tools, key=lambda item: item.name):
            out[tool.name] = ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.input_schema,
                ),
                max_retries=0,
                args_validator=_DICT_ARGS_VALIDATOR,
            )
        return out

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx,
        tool: ToolsetTool[Any],  # noqa: ARG002
    ) -> Any:
        _ = ctx
        result = await self._manager.call_tool(self._server_label, name, dict(tool_args))
        if not result.ok:
            raise RuntimeError(result.error_text or "MCP tool call failed")
        return result.output_text or ""


class _FakeRequestContractRegistry:
    def __init__(
        self,
        *,
        enabled: bool,
        servers: dict[str, list[str] | list[McpToolInfo]],
    ):
        self._enabled = enabled
        self._servers = servers

    def is_enabled(self) -> bool:
        return self._enabled

    def has_server(self, server_label: str) -> bool:
        return server_label in self._servers

    def is_server_available(self, server_label: str) -> bool:
        return server_label in self._servers

    def get_server_startup_error(self, server_label: str) -> str | None:
        return None

    async def list_tools(self, server_label: str) -> list[McpToolInfo]:
        tools = self._servers.get(server_label)
        if tools is None:
            raise BuiltinMcpRuntimeUnknownServerError(f"Unknown MCP server_label: {server_label}")
        out: list[McpToolInfo] = []
        for tool in tools:
            if isinstance(tool, McpToolInfo):
                out.append(tool)
            else:
                out.append(_tool_info(tool))
        return out

    async def call_tool(self, server_label: str, tool_name: str, arguments: dict[str, object]):
        raise AssertionError(
            "call_tool should not be reached in request-contract validation tests"
        )

    async def get_server_toolset(self, server_label: str) -> AbstractToolset[Any]:
        return _HostedRegistryNativeToolset(manager=self, server_label=server_label)

    async def get_server_mcp_tools_by_name(self, server_label: str) -> dict[str, ToolsetTool[Any]]:
        toolset = await self.get_server_toolset(server_label)
        return await toolset.get_tools(ctx=None)

    async def get_server_runtime(self, server_label: str):
        return SimpleNamespace(
            mcp_toolset=await self.get_server_toolset(server_label),
            allowed_tool_infos={tool.name: tool for tool in await self.list_tools(server_label)},
            allowed_mcp_tools_by_name=await self.get_server_mcp_tools_by_name(server_label),
        )


class _FakeRequestRemoteNativeToolset(AbstractToolset[Any]):
    def __init__(
        self,
        *,
        binding: RequestRemoteMcpServerBinding,
        tool_infos: dict[str, McpToolInfo],
        call_result: McpExecutionResult | None = None,
        list_error: Exception | None = None,
        require_auth_for_list: bool = False,
        require_auth_for_call: bool = False,
    ) -> None:
        self._binding = binding
        self._tool_infos = dict(tool_infos)
        self._call_result = call_result or McpExecutionResult(
            ok=True, output_text='{"ok":true}', error_text=None
        )
        self._list_error = list_error
        self._require_auth_for_list = require_auth_for_list
        self._require_auth_for_call = require_auth_for_call

    @property
    def id(self) -> str | None:
        return None

    async def get_tools(self, ctx) -> dict[str, ToolsetTool[Any]]:  # noqa: ARG002
        if self._list_error is not None:
            raise RuntimeError(str(self._list_error))
        if self._require_auth_for_list and not self._binding.authorization:
            raise RuntimeError("remote-list-auth-required")

        out: dict[str, ToolsetTool[Any]] = {}
        for tool_name, tool_info in sorted(self._tool_infos.items()):
            out[tool_name] = ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name=tool_name,
                    description=tool_info.description,
                    parameters_json_schema=tool_info.input_schema,
                ),
                max_retries=0,
                args_validator=_DICT_ARGS_VALIDATOR,
            )
        return out

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx,
        tool: ToolsetTool[Any],  # noqa: ARG002
    ) -> Any:
        _ = ctx, name, tool_args
        if self._require_auth_for_call and not self._binding.authorization:
            raise RuntimeError("remote-call-auth-required")
        if not self._call_result.ok:
            raise RuntimeError(self._call_result.error_text or "MCP tool call failed")
        return self._call_result.output_text or ""


@dataclass
class _RemoteBuilderProbe:
    seen_bindings: list[RequestRemoteMcpServerBinding]


def _install_request_remote_builder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    remote_tools: dict[str, list[McpToolInfo] | list[str]] | list[str] | None = None,
    remote_call_result: McpExecutionResult | None = None,
    remote_list_error: Exception | None = None,
    remote_require_auth_for_list: bool = False,
    remote_require_auth_for_call: bool = False,
) -> _RemoteBuilderProbe:
    import agentic_stack.types.openai as openai_types

    seen_bindings: list[RequestRemoteMcpServerBinding] = []

    remote_tools_by_server: dict[str, list[McpToolInfo] | list[str]]
    if remote_tools is None:
        remote_tools_by_server = {}
    elif isinstance(remote_tools, dict):
        remote_tools_by_server = remote_tools
    else:
        remote_tools_by_server = {"github_docs": list(remote_tools)}

    def _build(binding: RequestRemoteMcpServerBinding) -> AbstractToolset[Any]:
        seen_bindings.append(binding)

        tools = remote_tools_by_server.get(binding.server_label, [])
        tool_infos: dict[str, McpToolInfo] = {}
        for tool in tools:
            if isinstance(tool, McpToolInfo):
                tool_infos[tool.name] = tool
            else:
                name = str(tool)
                tool_infos[name] = _tool_info(name)

        return _FakeRequestRemoteNativeToolset(
            binding=binding,
            tool_infos=tool_infos,
            call_result=remote_call_result,
            list_error=remote_list_error,
            require_auth_for_list=remote_require_auth_for_list,
            require_auth_for_call=remote_require_auth_for_call,
        )

    monkeypatch.setattr(openai_types, "build_request_remote_toolset", _build)
    return _RemoteBuilderProbe(seen_bindings=seen_bindings)


@pytest.fixture(autouse=True)
def _default_request_remote_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_request_remote_builder(
        monkeypatch,
        remote_tools={"github_docs": ["search_docs"], "remote_docs": ["search_docs"]},
    )


def _assert_bad_input(resp: httpx.Response, expected_substring: str) -> None:
    assert resp.status_code == 422
    payload = resp.json()
    assert payload.get("error") == "bad_input"
    assert expected_substring in payload.get("message", "")


@contextmanager
def _override_attr(obj: object, attr: str, value: object):
    old_value = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, old_value)


@pytest.mark.anyio
async def test_unknown_hosted_server_label_is_rejected(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = (
        _FakeRequestContractRegistry(
            enabled=True,
            servers={"known_docs": ["search_docs"]},
        )
    )

    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "mcp", "server_label": "unknown_docs"}],
            "tool_choice": "auto",
        },
    )

    _assert_bad_input(resp, "Unknown MCP server_label")


@pytest.mark.anyio
async def test_duplicate_hosted_server_declarations_are_rejected(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = (
        _FakeRequestContractRegistry(
            enabled=True,
            servers={"github_docs": ["search_docs"]},
        )
    )

    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {"type": "mcp", "server_label": "github_docs"},
                {"type": "mcp", "server_label": "github_docs"},
            ],
            "tool_choice": "auto",
        },
    )

    _assert_bad_input(resp, "Duplicate MCP declarations")


@pytest.mark.anyio
async def test_tool_choice_mcp_without_matching_declaration_is_rejected(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tool_choice": {
                "type": "mcp",
                "server_label": "github_docs",
                "name": "search_docs",
            },
        },
    )

    _assert_bad_input(resp, "requires at least one MCP tool declaration")


@pytest.mark.anyio
async def test_function_tool_names_with_reserved_mcp_prefix_are_rejected(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "name": "mcp__reserved_name",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            ],
            "tool_choice": "auto",
        },
    )

    _assert_bad_input(resp, "reserved")


@pytest.mark.anyio
async def test_function_tool_names_with_reserved_mcp_prefix_rejected_when_tool_choice_none(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "name": "mcp__reserved_name",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            ],
            "tool_choice": "none",
        },
    )

    _assert_bad_input(resp, "reserved")


@pytest.mark.anyio
async def test_hosted_mcp_is_rejected_when_subsystem_disabled_even_with_tool_choice_none(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = (
        _FakeRequestContractRegistry(
            enabled=False,
            servers={"github_docs": ["search_docs"]},
        )
    )

    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "mcp", "server_label": "github_docs"}],
            "tool_choice": "none",
        },
    )

    _assert_bad_input(resp, "runtime is disabled")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "server_url",
    [
        "https://localhost/sse",
        "https://docs.localhost/mcp",
        "https://127.0.0.1/mcp",
        "https://[::1]/mcp",
    ],
)
async def test_request_remote_denylisted_hosts_are_rejected(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    server_url: str,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "mcp", "server_label": "github_docs", "server_url": server_url}],
            "tool_choice": "none",
        },
    )

    _assert_bad_input(resp, "denylisted")


@pytest.mark.anyio
async def test_request_remote_server_url_must_use_https(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "github_docs",
                    "server_url": "http://mcp.example.com/sse",
                }
            ],
            "tool_choice": "none",
        },
    )
    _assert_bad_input(resp, "https")


@pytest.mark.anyio
async def test_request_remote_connector_id_is_rejected(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "github_docs",
                    "server_url": "https://mcp.example.com/sse",
                    "connector_id": "conn_123",
                }
            ],
            "tool_choice": "none",
        },
    )

    _assert_bad_input(resp, "connector_id")


@pytest.mark.anyio
async def test_request_remote_headers_are_accepted(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "github_docs",
                    "server_url": "https://mcp.example.com/sse",
                    "headers": {"X-Test": "1"},
                }
            ],
            "tool_choice": "none",
        },
    )

    assert resp.status_code == 200


@pytest.mark.anyio
async def test_request_remote_duplicate_authorization_header_variants_are_rejected(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "github_docs",
                    "server_url": "https://mcp.example.com/sse",
                    "headers": {
                        "Authorization": "Bearer token-a",
                        "authorization": "Bearer token-b",
                    },
                }
            ],
            "tool_choice": "none",
        },
    )
    _assert_bad_input(resp, "Authorization")


@pytest.mark.anyio
async def test_mcp_require_approval_other_than_never_is_rejected(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "github_docs",
                    "server_url": "https://mcp.example.com/sse",
                    "require_approval": "always",
                }
            ],
            "tool_choice": "none",
        },
    )

    _assert_bad_input(resp, "require_approval")


@pytest.mark.anyio
async def test_request_shape_errors_for_mcp_tools_are_validation_errors(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "github_docs",
                    "allowed_tools": "not-a-list",
                }
            ],
            "tool_choice": "none",
        },
    )

    assert resp.status_code == 422
    payload = resp.json()
    if payload.get("error") is not None:
        assert payload.get("error") == "validation_error"
    else:
        assert isinstance(payload.get("detail"), list)


@pytest.mark.anyio
async def test_mcp_authorization_on_hosted_declaration_is_rejected(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = (
        _FakeRequestContractRegistry(
            enabled=True,
            servers={"github_docs": ["search_docs"]},
        )
    )

    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "github_docs",
                    "authorization": "token",
                }
            ],
            "tool_choice": "none",
        },
    )

    _assert_bad_input(resp, "authorization")


@pytest.mark.anyio
async def test_mcp_headers_on_hosted_declaration_is_rejected(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = (
        _FakeRequestContractRegistry(
            enabled=True,
            servers={"github_docs": ["search_docs"]},
        )
    )
    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "github_docs",
                    "headers": {"X-Test": "1"},
                }
            ],
            "tool_choice": "none",
        },
    )
    _assert_bad_input(resp, "headers")


@pytest.mark.anyio
async def test_duplicate_server_label_across_hosted_and_remote_is_rejected(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = (
        _FakeRequestContractRegistry(
            enabled=True,
            servers={"github_docs": ["search_docs"]},
        )
    )

    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {"type": "mcp", "server_label": "github_docs"},
                {
                    "type": "mcp",
                    "server_label": "github_docs",
                    "server_url": "https://mcp.example.com/sse",
                },
            ],
            "tool_choice": "none",
        },
    )

    _assert_bad_input(resp, "Duplicate MCP declarations")


@pytest.mark.anyio
async def test_mixed_hosted_and_request_remote_declarations_are_supported() -> None:
    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {"type": "mcp", "server_label": "hosted_docs"},
                {
                    "type": "mcp",
                    "server_label": "remote_docs",
                    "server_url": "https://mcp.example.com/sse",
                },
            ],
            "tool_choice": {
                "type": "mcp",
                "server_label": "remote_docs",
                "name": "search_docs",
            },
        }
    )
    run_settings, builtin_tools, mcp_tool_name_map = await req.as_run_settings(
        builtin_mcp_runtime_client=_FakeRequestContractRegistry(
            enabled=True,
            servers={"hosted_docs": ["search_docs"]},
        ),
        request_remote_enabled=True,
        request_remote_url_checks_enabled=True,
    )
    assert run_settings["toolsets"] is not None
    assert any(isinstance(toolset, McpGatewayToolset) for toolset in run_settings["toolsets"])
    assert len(builtin_tools) == 0
    assert len(mcp_tool_name_map) == 1
    ref = next(iter(mcp_tool_name_map.values()))
    assert ref.server_label == "remote_docs"
    assert ref.mode == "request_remote"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "server_url",
    [
        "https://mcp.example.com/sse",
        "https://mcp.example.com/mcp",
    ],
)
async def test_request_remote_server_url_accepts_sse_and_streamable_http_shapes(
    server_url: str,
) -> None:
    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "github_docs",
                    "server_url": server_url,
                }
            ],
            "tool_choice": {"type": "mcp", "server_label": "github_docs", "name": "search_docs"},
        }
    )
    _run_settings, builtin_tools, mcp_tool_name_map = await req.as_run_settings(
        request_remote_enabled=True,
        request_remote_url_checks_enabled=True,
    )
    assert len(builtin_tools) == 0
    assert len(mcp_tool_name_map) == 1
    ref = next(iter(mcp_tool_name_map.values()))
    assert ref.mode == "request_remote"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "server_url",
    [
        "https://mcp.example.com/sse",
        "https://mcp.example.com/mcp",
    ],
)
async def test_request_remote_binding_does_not_require_transport_inference_metadata(
    server_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _install_request_remote_builder(monkeypatch, remote_tools=["search_docs"])
    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "github_docs",
                    "server_url": server_url,
                }
            ],
            "tool_choice": {"type": "mcp", "server_label": "github_docs", "name": "search_docs"},
        }
    )
    await req.as_run_settings(
        request_remote_enabled=True,
        request_remote_url_checks_enabled=True,
    )
    assert probe.seen_bindings
    assert probe.seen_bindings[0].server_url == server_url
    assert not hasattr(probe.seen_bindings[0], "transport")


@pytest.mark.anyio
async def test_request_remote_empty_allowed_tools_after_filter_is_rejected(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "github_docs",
                    "server_url": "https://mcp.example.com/sse",
                    "allowed_tools": ["nonexistent_tool"],
                }
            ],
            "tool_choice": "none",
        },
    )

    _assert_bad_input(resp, "empty final allowed tool set")


class _GatewayNativeToolset(AbstractToolset[Any]):
    def __init__(
        self,
        *,
        raise_missing_once: bool = False,
    ) -> None:
        self.get_tools_calls = 0
        self.call_tool_calls = 0
        self._raise_missing_once = raise_missing_once
        self._stale_tool = ToolsetTool(
            toolset=self,
            tool_def=ToolDefinition(
                name="search_docs",
                description="stale",
                parameters_json_schema={"type": "object", "properties": {}},
            ),
            max_retries=0,
            args_validator=_DICT_ARGS_VALIDATOR,
        )
        self._mcp_tool = ToolsetTool(
            toolset=self,
            tool_def=ToolDefinition(
                name="search_docs",
                description="live",
                parameters_json_schema={"type": "object", "properties": {}},
            ),
            max_retries=0,
            args_validator=_DICT_ARGS_VALIDATOR,
        )
        self._tools: dict[str, ToolsetTool[Any]] = {"search_docs": self._mcp_tool}

    @property
    def id(self) -> str | None:
        return None

    def stale_tool(self) -> ToolsetTool[Any]:
        return self._stale_tool

    def clear_tools(self) -> None:
        self._tools = {}

    async def get_tools(self, ctx) -> dict[str, ToolsetTool[Any]]:  # noqa: ARG002
        self.get_tools_calls += 1
        return dict(self._tools)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx,
        tool: ToolsetTool[Any],
    ) -> Any:
        _ = name, tool_args, ctx
        self.call_tool_calls += 1
        if self._raise_missing_once:
            self._raise_missing_once = False
            raise KeyError("search_docs")
        if tool is self._stale_tool:
            raise KeyError("search_docs")
        return {"ok": True}


@pytest.mark.anyio
async def test_gateway_toolset_refreshes_and_avoids_per_call_inventory_fetch() -> None:
    from jsonschema import Draft202012Validator

    from agentic_stack.mcp.gateway_toolset import ResolvedMcpTool

    mcp_toolset = _GatewayNativeToolset()
    gateway_toolset = McpGatewayToolset(
        tools=[
            ResolvedMcpTool(
                internal_name="mcp__docs__search_docs",
                ref=McpToolRef(server_label="docs", tool_name="search_docs"),
                mcp_toolset=mcp_toolset,
                mcp_tool_name="search_docs",
                description="search docs",
                input_schema={"type": "object", "properties": {}, "additionalProperties": True},
                schema_validator=Draft202012Validator(
                    {"type": "object", "properties": {}, "additionalProperties": True}
                ),
                mcp_tool=mcp_toolset.stale_tool(),
            )
        ]
    )
    tools = await gateway_toolset.get_tools(ctx=None)
    tool = tools["mcp__docs__search_docs"]

    payload1 = await gateway_toolset.call_tool("mcp__docs__search_docs", {}, ctx=None, tool=tool)
    payload2 = await gateway_toolset.call_tool("mcp__docs__search_docs", {}, ctx=None, tool=tool)
    assert payload1["ok"] is True
    assert payload2["ok"] is True
    assert mcp_toolset.get_tools_calls == 1
    assert mcp_toolset.call_tool_calls == 3


@pytest.mark.anyio
async def test_gateway_toolset_returns_item_failure_when_refresh_misses_tool() -> None:
    from jsonschema import Draft202012Validator

    from agentic_stack.mcp.gateway_toolset import ResolvedMcpTool

    mcp_toolset = _GatewayNativeToolset()
    mcp_toolset.clear_tools()
    gateway_toolset = McpGatewayToolset(
        tools=[
            ResolvedMcpTool(
                internal_name="mcp__docs__search_docs",
                ref=McpToolRef(server_label="docs", tool_name="search_docs"),
                mcp_toolset=mcp_toolset,
                mcp_tool_name="search_docs",
                description="search docs",
                input_schema={"type": "object", "properties": {}, "additionalProperties": True},
                schema_validator=Draft202012Validator(
                    {"type": "object", "properties": {}, "additionalProperties": True}
                ),
                mcp_tool=None,
            )
        ]
    )
    tools = await gateway_toolset.get_tools(ctx=None)
    payload = await gateway_toolset.call_tool(
        "mcp__docs__search_docs", {}, ctx=None, tool=tools["mcp__docs__search_docs"]
    )
    assert payload["ok"] is False
    assert "not available" in str(payload["error_text"])
    assert mcp_toolset.get_tools_calls == 1
    assert mcp_toolset.call_tool_calls == 0


def test_missing_tool_classifier_matches_expected_tool_name() -> None:
    from agentic_stack.mcp.utils import is_mcp_tool_keyerror

    class _Exc(Exception):
        pass

    with_text = _Exc("tool not found")

    assert is_mcp_tool_keyerror(KeyError("search_docs"), "search_docs")
    assert is_mcp_tool_keyerror(KeyError("other"), "search_docs") is False
    assert is_mcp_tool_keyerror(with_text, "search_docs") is False


@pytest.mark.anyio
async def test_hosted_mcp_toolset_validates_arguments_and_calls_router() -> None:
    from agentic_stack.types.openai import vLLMResponsesRequest

    class _FakeRegistry:
        def __init__(self) -> None:
            self.call_count = 0

        def is_enabled(self) -> bool:
            return True

        def has_server(self, server_label: str) -> bool:
            return server_label == "local_fs"

        def is_server_available(self, server_label: str) -> bool:
            return server_label == "local_fs"

        def get_server_startup_error(self, server_label: str) -> str | None:
            return None

        async def list_tools(self, server_label: str) -> list[McpToolInfo]:
            return [
                _tool_info(
                    "list_directory",
                    description=None,
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                )
            ]

        async def call_tool(
            self,
            server_label: str,
            tool_name: str,
            arguments: dict[str, object],
        ) -> McpExecutionResult:
            self.call_count += 1
            return McpExecutionResult(ok=True, output_text="{}", error_text=None)

        async def get_server_toolset(self, server_label: str) -> AbstractToolset[Any]:
            return _HostedRegistryNativeToolset(manager=self, server_label=server_label)

        async def get_server_mcp_tools_by_name(
            self, server_label: str
        ) -> dict[str, ToolsetTool[Any]]:
            toolset = await self.get_server_toolset(server_label)
            return await toolset.get_tools(ctx=None)

        async def get_server_runtime(self, server_label: str):
            return SimpleNamespace(
                mcp_toolset=await self.get_server_toolset(server_label),
                allowed_tool_infos={
                    tool.name: tool for tool in await self.list_tools(server_label)
                },
                allowed_mcp_tools_by_name=await self.get_server_mcp_tools_by_name(server_label),
            )

    manager = _FakeRegistry()
    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "mcp", "server_label": "local_fs"}],
            "tool_choice": {
                "type": "mcp",
                "server_label": "local_fs",
                "name": "list_directory",
            },
        }
    )

    run_settings, builtin_tools, _ = await req.as_run_settings(
        builtin_mcp_runtime_client=manager,
        request_remote_enabled=True,
        request_remote_url_checks_enabled=True,
    )
    assert len(builtin_tools) == 0
    assert run_settings["toolsets"] is not None
    mcp_toolset = next(
        toolset for toolset in run_settings["toolsets"] if isinstance(toolset, McpGatewayToolset)
    )
    tools = await mcp_toolset.get_tools(ctx=None)
    assert len(tools) == 1
    tool_name = next(iter(tools.keys()))
    invalid_payload = await mcp_toolset.call_tool(
        tool_name,
        {},
        ctx=None,
        tool=tools[tool_name],
    )
    assert isinstance(invalid_payload, dict)
    assert invalid_payload["ok"] is False
    assert str(invalid_payload["error_text"]).startswith("input_validation_error:")
    assert manager.call_count == 0

    valid_payload = await mcp_toolset.call_tool(
        tool_name,
        {"path": "."},
        ctx=None,
        tool=tools[tool_name],
    )
    assert isinstance(valid_payload, dict)
    assert valid_payload["ok"] is True
    assert manager.call_count == 1


@pytest.mark.anyio
async def test_hosted_mcp_toolset_uses_mcp_description_and_normalizes_missing_type() -> None:
    from agentic_stack.types.openai import vLLMResponsesRequest

    class _FakeRegistry:
        def is_enabled(self) -> bool:
            return True

        def has_server(self, server_label: str) -> bool:
            return server_label == "local_fs"

        def is_server_available(self, server_label: str) -> bool:
            return server_label == "local_fs"

        def get_server_startup_error(self, server_label: str) -> str | None:
            return None

        async def list_tools(self, server_label: str) -> list[McpToolInfo]:
            return [
                _tool_info(
                    "list_directory",
                    description="List directory entries",
                    input_schema={},
                )
            ]

        async def call_tool(
            self,
            server_label: str,
            tool_name: str,
            arguments: dict[str, object],
        ) -> McpExecutionResult:
            return McpExecutionResult(ok=True, output_text="{}", error_text=None)

        async def get_server_toolset(self, server_label: str) -> AbstractToolset[Any]:
            return _HostedRegistryNativeToolset(manager=self, server_label=server_label)

        async def get_server_mcp_tools_by_name(
            self, server_label: str
        ) -> dict[str, ToolsetTool[Any]]:
            toolset = await self.get_server_toolset(server_label)
            return await toolset.get_tools(ctx=None)

        async def get_server_runtime(self, server_label: str):
            return SimpleNamespace(
                mcp_toolset=await self.get_server_toolset(server_label),
                allowed_tool_infos={
                    tool.name: tool for tool in await self.list_tools(server_label)
                },
                allowed_mcp_tools_by_name=await self.get_server_mcp_tools_by_name(server_label),
            )

    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "mcp", "server_label": "local_fs"}],
            "tool_choice": {
                "type": "mcp",
                "server_label": "local_fs",
                "name": "list_directory",
            },
        }
    )

    run_settings, builtin_tools, _ = await req.as_run_settings(
        builtin_mcp_runtime_client=_FakeRegistry(),
        request_remote_enabled=True,
        request_remote_url_checks_enabled=True,
    )
    assert len(builtin_tools) == 0
    assert run_settings["toolsets"] is not None
    mcp_toolset = next(
        toolset for toolset in run_settings["toolsets"] if isinstance(toolset, McpGatewayToolset)
    )
    tools = await mcp_toolset.get_tools(ctx=None)
    tool_name = next(iter(tools.keys()))
    tool_def = tools[tool_name].tool_def
    assert tool_def.description == "List directory entries"
    assert tool_def.parameters_json_schema["type"] == "object"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("allowed_tools",),
    [
        (
            [
                {"type": "mcp", "server_label": "github_docs"},
                {"type": "mcp", "server_label": "github_docs", "name": "search_docs"},
            ],
        ),
        (
            [
                {"type": "mcp", "server_label": "github_docs", "name": "search_docs"},
                {"type": "mcp", "server_label": "github_docs"},
            ],
        ),
    ],
)
async def test_allowed_tools_mcp_server_wide_entry_remains_monotonic(
    allowed_tools: list[dict[str, str]],
) -> None:
    from agentic_stack.types.openai import vLLMResponsesRequest

    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "mcp", "server_label": "github_docs"}],
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": allowed_tools,
            },
        }
    )

    run_settings, builtin_tools, mcp_tool_name_map = await req.as_run_settings(
        builtin_mcp_runtime_client=_FakeRequestContractRegistry(
            enabled=True,
            servers={"github_docs": ["search_docs", "get_page"]},
        ),
        request_remote_enabled=True,
        request_remote_url_checks_enabled=True,
    )

    assert builtin_tools == []
    assert {(ref.server_label, ref.tool_name) for ref in mcp_tool_name_map.values()} == {
        ("github_docs", "search_docs"),
        ("github_docs", "get_page"),
    }
    assert run_settings["toolsets"] is not None
    mcp_toolset = next(
        toolset for toolset in run_settings["toolsets"] if isinstance(toolset, McpGatewayToolset)
    )
    assert len(await mcp_toolset.get_tools(ctx=None)) == 2


@pytest.mark.anyio
async def test_mcp_rehydration_keeps_colliding_tool_names_distinct() -> None:
    from pydantic_ai import BuiltinToolCallPart, ModelResponse

    from agentic_stack.types.openai import vLLMResponsesRequest

    class _FakeRegistry:
        def is_enabled(self) -> bool:
            return True

        def has_server(self, server_label: str) -> bool:
            return server_label == "docs"

        def is_server_available(self, server_label: str) -> bool:
            return server_label == "docs"

        def get_server_startup_error(self, server_label: str) -> str | None:
            return None

        async def list_tools(self, server_label: str) -> list[McpToolInfo]:
            return [
                _tool_info("search.docs"),
                _tool_info("search_docs"),
            ]

        async def call_tool(
            self,
            server_label: str,
            tool_name: str,
            arguments: dict[str, object],
        ) -> McpExecutionResult:
            return McpExecutionResult(ok=True, output_text="{}", error_text=None)

        async def get_server_toolset(self, server_label: str) -> AbstractToolset[Any]:
            return _HostedRegistryNativeToolset(manager=self, server_label=server_label)

        async def get_server_mcp_tools_by_name(
            self, server_label: str
        ) -> dict[str, ToolsetTool[Any]]:
            toolset = await self.get_server_toolset(server_label)
            return await toolset.get_tools(ctx=None)

        async def get_server_runtime(self, server_label: str):
            return SimpleNamespace(
                mcp_toolset=await self.get_server_toolset(server_label),
                allowed_tool_infos={
                    tool.name: tool for tool in await self.list_tools(server_label)
                },
                allowed_mcp_tools_by_name=await self.get_server_mcp_tools_by_name(server_label),
            )

    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [
                {
                    "type": "mcp_call",
                    "id": "mcp_call_2",
                    "server_label": "docs",
                    "name": "search_docs",
                    "arguments": '{"query":"beta"}',
                    "status": "completed",
                    "output": "{}",
                },
                {
                    "type": "mcp_call",
                    "id": "mcp_call_1",
                    "server_label": "docs",
                    "name": "search.docs",
                    "arguments": '{"query":"alpha"}',
                    "status": "completed",
                    "output": "{}",
                },
            ],
            "tools": [{"type": "mcp", "server_label": "docs"}],
            "tool_choice": "auto",
        }
    )

    run_settings, builtin_tools, mcp_tool_name_map = await req.as_run_settings(
        builtin_mcp_runtime_client=_FakeRegistry(),
        request_remote_enabled=True,
        request_remote_url_checks_enabled=True,
    )

    assert len(builtin_tools) == 0
    assert len(mcp_tool_name_map) == 2

    call_name_by_id: dict[str, str] = {}
    for message in run_settings["message_history"] or []:
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if isinstance(part, BuiltinToolCallPart):
                call_name_by_id[part.tool_call_id] = part.tool_name

    assert call_name_by_id.keys() == {"mcp_call_1", "mcp_call_2"}
    assert call_name_by_id["mcp_call_1"] != call_name_by_id["mcp_call_2"]
    assert mcp_tool_name_map[call_name_by_id["mcp_call_1"]] == McpToolRef(
        server_label="docs",
        tool_name="search.docs",
    )
    assert mcp_tool_name_map[call_name_by_id["mcp_call_2"]] == McpToolRef(
        server_label="docs",
        tool_name="search_docs",
    )


@pytest.mark.anyio
async def test_invalid_hosted_mcp_input_schema_is_rejected(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = (
        _FakeRequestContractRegistry(
            enabled=True,
            servers={
                "github_docs": [
                    _tool_info(
                        "search_docs",
                        input_schema={
                            "type": "string",
                        },
                    )
                ]
            },
        )
    )

    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "mcp", "server_label": "github_docs"}],
            "tool_choice": {
                "type": "mcp",
                "server_label": "github_docs",
                "name": "search_docs",
            },
        },
    )

    _assert_bad_input(resp, "root `type` must be `object`")


# ==============================
# Normalizer
# ==============================


def test_normalizer_emits_mcp_events_for_mcp_tool_results() -> None:
    from pydantic_ai import (
        FunctionToolResultEvent,
        PartDeltaEvent,
        PartEndEvent,
        PartStartEvent,
        ToolCallPart,
        ToolCallPartDelta,
        ToolReturnPart,
    )

    internal_name = "mcp__github_docs__search_docs"
    ref = McpToolRef(server_label="github_docs", tool_name="search_docs")

    normalizer = PydanticAINormalizer(
        builtin_tool_names={internal_name},
        code_interpreter_tool_name="code_interpreter",
        mcp_tool_name_map={internal_name: ref},
    )

    result_payload = build_mcp_tool_result_payload(
        ref=ref,
        result=McpExecutionResult(
            ok=True,
            output_text='{"results":[{"title":"Migration Notes"}]}',
            error_text=None,
        ),
    )

    events = [
        PartStartEvent(
            index=0,
            part=ToolCallPart(tool_name=internal_name, args=None, tool_call_id="call_1"),
        ),
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(
                tool_call_id="call_1", args_delta='{"query":"migration notes"}'
            ),
        ),
        PartEndEvent(
            index=0,
            part=ToolCallPart(
                tool_name=internal_name,
                args='{"query":"migration notes"}',
                tool_call_id="call_1",
            ),
        ),
        FunctionToolResultEvent(
            result=ToolReturnPart(
                tool_name=internal_name,
                content=result_payload,
                tool_call_id="call_1",
            )
        ),
    ]

    out = []
    for event in events:
        out.extend(list(normalizer.on_event(event)))

    assert any(isinstance(event, McpCallStarted) for event in out)
    assert any(isinstance(event, McpCallArgumentsDelta) for event in out)
    done_events = [event for event in out if isinstance(event, McpCallArgumentsDone)]
    assert len(done_events) == 1
    assert done_events[0].arguments_json == '{"query":"migration notes"}'

    completed = [event for event in out if isinstance(event, McpCallCompleted)]
    assert len(completed) == 1
    assert "Migration Notes" in completed[0].output_text


def test_normalizer_maps_malformed_tool_result_to_mcp_failed() -> None:
    from pydantic_ai import (
        FunctionToolResultEvent,
        PartEndEvent,
        PartStartEvent,
        ToolCallPart,
        ToolReturnPart,
    )

    internal_name = "mcp__github_docs__search_docs"
    ref = McpToolRef(server_label="github_docs", tool_name="search_docs")

    normalizer = PydanticAINormalizer(
        builtin_tool_names={internal_name},
        code_interpreter_tool_name="code_interpreter",
        mcp_tool_name_map={internal_name: ref},
    )

    events = [
        PartStartEvent(
            index=0,
            part=ToolCallPart(tool_name=internal_name, args=None, tool_call_id="call_1"),
        ),
        PartEndEvent(
            index=0,
            part=ToolCallPart(
                tool_name=internal_name,
                args='{"query":"migration notes"}',
                tool_call_id="call_1",
            ),
        ),
        FunctionToolResultEvent(
            result=ToolReturnPart(
                tool_name=internal_name,
                content="not-a-dict",
                tool_call_id="call_1",
            )
        ),
    ]

    out = []
    for event in events:
        out.extend(list(normalizer.on_event(event)))

    failed = [event for event in out if isinstance(event, McpCallFailed)]
    assert len(failed) == 1
    assert "payload must be a JSON object" in failed[0].error_text


def test_normalizer_maps_mismatched_tool_ref_to_mcp_failed() -> None:
    from pydantic_ai import (
        FunctionToolResultEvent,
        PartEndEvent,
        PartStartEvent,
        ToolCallPart,
        ToolReturnPart,
    )

    internal_name = "mcp__github_docs__search_docs"
    expected_ref = McpToolRef(server_label="github_docs", tool_name="search_docs")
    wrong_ref = McpToolRef(server_label="github_docs", tool_name="get_page")

    normalizer = PydanticAINormalizer(
        builtin_tool_names={internal_name},
        code_interpreter_tool_name="code_interpreter",
        mcp_tool_name_map={internal_name: expected_ref},
    )

    result_payload = build_mcp_tool_result_payload(
        ref=wrong_ref,
        result=McpExecutionResult(ok=True, output_text='{"results":[]}', error_text=None),
    )

    events = [
        PartStartEvent(
            index=0,
            part=ToolCallPart(tool_name=internal_name, args=None, tool_call_id="call_1"),
        ),
        PartEndEvent(
            index=0,
            part=ToolCallPart(
                tool_name=internal_name,
                args='{"query":"migration notes"}',
                tool_call_id="call_1",
            ),
        ),
        FunctionToolResultEvent(
            result=ToolReturnPart(
                tool_name=internal_name,
                content=result_payload,
                tool_call_id="call_1",
            )
        ),
    ]

    out = []
    for event in events:
        out.extend(list(normalizer.on_event(event)))

    failed = [event for event in out if isinstance(event, McpCallFailed)]
    assert len(failed) == 1
    assert "mismatched tool ref" in failed[0].error_text.lower()


# ==============================
# Composer
# ==============================


def _drain(composer: ResponseComposer, events: Iterable[object]):
    out = []
    out.extend(composer.start())
    for event in events:
        out.extend(list(composer.feed(event)))  # type: ignore[arg-type]
    return out


def _index_of_type(events: list, event_type: str) -> int:
    for i, event in enumerate(events):
        if getattr(event, "type", None) == event_type:
            return i
    raise AssertionError(f"event type not found: {event_type}")


def test_composer_emits_hosted_mcp_success_event_sequence() -> None:
    composer = ResponseComposer(response=OpenAIResponsesResponse(model="test-model"))
    out = _drain(
        composer,
        [
            McpCallStarted(
                item_key="mcp:1",
                server_label="github_docs",
                name="search_docs",
                initial_arguments_json="",
            ),
            McpCallArgumentsDelta(item_key="mcp:1", delta='{"query":"migration notes"}'),
            McpCallArgumentsDone(item_key="mcp:1", arguments_json='{"query":"migration notes"}'),
            McpCallCompleted(
                item_key="mcp:1",
                output_text='{"results":[{"title":"Migration Notes"}]}',
            ),
            UsageFinal(
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            ),
        ],
    )

    assert _index_of_type(out, "response.output_item.added") < _index_of_type(
        out, "response.mcp_call.in_progress"
    )
    assert _index_of_type(out, "response.mcp_call_arguments.delta") < _index_of_type(
        out, "response.mcp_call_arguments.done"
    )
    assert _index_of_type(out, "response.mcp_call.completed") < _index_of_type(
        out, "response.output_item.done"
    )

    done_item = next(
        event.item
        for event in out
        if getattr(event, "type", None) == "response.output_item.done"
        and getattr(event.item, "type", None) == "mcp_call"
    )
    assert done_item.status == "completed"
    assert "Migration Notes" in (done_item.output or "")


def test_composer_emits_hosted_mcp_failed_item() -> None:
    composer = ResponseComposer(response=OpenAIResponsesResponse(model="test-model"))
    out = _drain(
        composer,
        [
            McpCallStarted(
                item_key="mcp:1",
                server_label="github_docs",
                name="search_docs",
                initial_arguments_json='{"query":"migration notes"}',
            ),
            McpCallArgumentsDone(item_key="mcp:1", arguments_json='{"query":"migration notes"}'),
            McpCallFailed(item_key="mcp:1", error_text="tools/call timeout after 60s"),
            UsageFinal(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            ),
        ],
    )

    failed_event = next(
        event for event in out if getattr(event, "type", None) == "response.mcp_call.failed"
    )
    assert "error" not in failed_event.model_dump(exclude_none=True)

    done_item = next(
        event.item
        for event in out
        if getattr(event, "type", None) == "response.output_item.done"
        and getattr(event.item, "type", None) == "mcp_call"
    )
    assert done_item.status == "failed"
    assert done_item.error == "tools/call timeout after 60s"


def test_composer_repeated_hosted_mcp_failures_stay_item_level_and_non_fatal() -> None:
    composer = ResponseComposer(response=OpenAIResponsesResponse(model="test-model"))
    out = _drain(
        composer,
        [
            McpCallStarted(
                item_key="mcp:1",
                server_label="github_docs",
                name="search_docs",
                initial_arguments_json='{"query":"migration notes"}',
            ),
            McpCallArgumentsDone(item_key="mcp:1", arguments_json='{"query":"migration notes"}'),
            McpCallFailed(item_key="mcp:1", error_text="tools/call timeout after 60s"),
            McpCallStarted(
                item_key="mcp:2",
                server_label="github_docs",
                name="search_docs",
                initial_arguments_json='{"query":"migration notes"}',
            ),
            McpCallArgumentsDone(item_key="mcp:2", arguments_json='{"query":"migration notes"}'),
            McpCallFailed(item_key="mcp:2", error_text="tools/call timeout after 60s"),
            UsageFinal(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            ),
        ],
    )

    failed_events = [
        event for event in out if getattr(event, "type", None) == "response.mcp_call.failed"
    ]
    assert len(failed_events) == 2
    assert not any(getattr(event, "type", None) == "response.failed" for event in out)
    completed = [event for event in out if getattr(event, "type", None) == "response.completed"]
    assert len(completed) == 1

    mcp_items = [
        item
        for item in (completed[0].response.output or [])
        if getattr(item, "type", None) == "mcp_call"
    ]
    assert len(mcp_items) == 2
    assert all(item.status == "failed" for item in mcp_items)
    assert completed[0].response.incomplete_details is None


@pytest.mark.anyio
async def test_max_tool_calls_is_not_mapped_to_runtime_usage_limits() -> None:
    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "input": [{"role": "user", "content": "Hello"}],
            "max_tool_calls": 7,
        }
    )
    run_settings, _builtin_tools, _mcp_map = await req.as_run_settings(
        builtin_mcp_runtime_client=None,
        request_remote_enabled=True,
        request_remote_url_checks_enabled=True,
    )
    assert run_settings["usage_limits"].tool_calls_limit is None
    assert run_settings["usage_limits"].request_limit == UsageLimits().request_limit


def test_extract_openai_error_fields_uses_verbatim_fallback_for_non_json_body() -> None:
    from agentic_stack.lm_failures import extract_openai_error_fields

    fallback = "status_code: 502, model_name: some-model, body: <html>bad gateway</html>"
    code, message, param = extract_openai_error_fields(None, fallback_message=fallback)
    assert code == ""
    assert message == fallback
    assert param == ""


# ==============================
# Gateway replay
# ==============================


class _FakeGatewayRegistry:
    def __init__(self, *, result: McpExecutionResult):
        self._result = result

    def is_enabled(self) -> bool:
        return True

    def has_server(self, server_label: str) -> bool:
        return server_label == "github_docs"

    def is_server_available(self, server_label: str) -> bool:
        return server_label == "github_docs"

    def get_server_startup_error(self, server_label: str) -> str | None:
        return None

    async def list_tools(self, server_label: str) -> list[McpToolInfo]:
        assert server_label == "github_docs"
        return [
            _tool_info(
                "search_docs",
                description="Search docs",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            )
        ]

    async def call_tool(
        self,
        server_label: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> McpExecutionResult:
        assert server_label == "github_docs"
        assert tool_name == "search_docs"
        assert isinstance(arguments, dict)
        return self._result

    async def get_server_toolset(self, server_label: str) -> AbstractToolset[Any]:
        return _HostedRegistryNativeToolset(manager=self, server_label=server_label)

    async def get_server_mcp_tools_by_name(self, server_label: str) -> dict[str, ToolsetTool[Any]]:
        toolset = await self.get_server_toolset(server_label)
        return await toolset.get_tools(ctx=None)

    async def get_server_runtime(self, server_label: str):
        return SimpleNamespace(
            mcp_toolset=await self.get_server_toolset(server_label),
            allowed_tool_infos={tool.name: tool for tool in await self.list_tools(server_label)},
            allowed_mcp_tools_by_name=await self.get_server_mcp_tools_by_name(server_label),
        )


class _CountingGatewayManager(_FakeGatewayRegistry):
    def __init__(self, *, result: McpExecutionResult):
        super().__init__(result=result)
        self.call_count = 0

    async def call_tool(
        self,
        server_label: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> McpExecutionResult:
        self.call_count += 1
        return await super().call_tool(
            server_label=server_label, tool_name=tool_name, arguments=arguments
        )


class _ValidationFailGatewayManager:
    def __init__(self) -> None:
        self.call_count = 0

    def is_enabled(self) -> bool:
        return True

    def has_server(self, server_label: str) -> bool:
        return server_label == "github_docs"

    def is_server_available(self, server_label: str) -> bool:
        return server_label == "github_docs"

    def get_server_startup_error(self, server_label: str) -> str | None:
        return None

    async def list_tools(self, server_label: str) -> list[McpToolInfo]:
        assert server_label == "github_docs"
        return [
            _tool_info(
                "search_docs",
                description="Search docs",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            )
        ]

    async def call_tool(
        self,
        server_label: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> McpExecutionResult:
        self.call_count += 1
        return McpExecutionResult(ok=True, output_text='{"ok":true}', error_text=None)

    async def get_server_toolset(self, server_label: str) -> AbstractToolset[Any]:
        return _HostedRegistryNativeToolset(manager=self, server_label=server_label)

    async def get_server_mcp_tools_by_name(self, server_label: str) -> dict[str, ToolsetTool[Any]]:
        toolset = await self.get_server_toolset(server_label)
        return await toolset.get_tools(ctx=None)

    async def get_server_runtime(self, server_label: str):
        return SimpleNamespace(
            mcp_toolset=await self.get_server_toolset(server_label),
            allowed_tool_infos={tool.name: tool for tool in await self.list_tools(server_label)},
            allowed_mcp_tools_by_name=await self.get_server_mcp_tools_by_name(server_label),
        )


def _hosted_mcp_request_payload() -> dict:
    return {
        "model": "some-model",
        "stream": True,
        "input": [
            {
                "role": "user",
                "content": "Call tool `mcp__github_docs__search_docs` with query='migration notes'. Do not answer directly before tool call.",
            }
        ],
        "tools": [{"type": "mcp", "server_label": "github_docs"}],
        "tool_choice": {"type": "mcp", "server_label": "github_docs", "name": "search_docs"},
    }


def _request_remote_mcp_request_payload(
    *,
    server_url: str = "https://mcp.example.com/sse",
    authorization: str | None = "remote-token",
) -> dict:
    tool: dict[str, object] = {
        "type": "mcp",
        "server_label": "github_docs",
        "server_url": server_url,
    }
    if authorization is not None:
        tool["authorization"] = authorization
    return {
        "model": "some-model",
        "stream": True,
        "input": [
            {
                "role": "user",
                "content": "Call tool `mcp__github_docs__search_docs` with query='migration notes'. Do not answer directly before tool call.",
            }
        ],
        "tools": [tool],
        "tool_choice": {"type": "mcp", "server_label": "github_docs", "name": "search_docs"},
    }


@pytest.mark.anyio
async def test_gateway_stream_hosted_mcp_success_emits_mcp_event_sequence(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
) -> None:
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = _FakeGatewayRegistry(
        result=McpExecutionResult(
            ok=True,
            output_text='{"results":[{"title":"Migration Notes","url":"https://example.test/migration"}]}',
            error_text=None,
        )
    )
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "mcp-hosted-step1-stream.yaml",
        "mcp-hosted-step2-stream.yaml",
    )

    async with gateway_client.stream(
        "POST", "/v1/responses", json=_hosted_mcp_request_payload()
    ) as resp:
        assert resp.status_code == 200
        body = await resp.aread()

    text = body.decode("utf-8", errors="replace")
    frames = parse_sse_frames(text)
    events = parse_sse_json_events(frames)

    event_types = [event.get("type") for event in events]
    assert "response.mcp_call.in_progress" in event_types
    assert "response.mcp_call_arguments.delta" in event_types
    assert "response.mcp_call_arguments.done" in event_types
    assert "response.mcp_call.completed" in event_types

    completed = extract_completed_response(events)
    mcp_item = next(
        (
            item
            for item in (completed.get("output") or [])
            if isinstance(item, dict) and item.get("type") == "mcp_call"
        ),
        None,
    )
    assert mcp_item is not None
    assert mcp_item["status"] == "completed"
    assert "Migration Notes" in (mcp_item.get("output") or "")


@pytest.mark.anyio
async def test_gateway_stream_hosted_mcp_timeout_emits_failed_item(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
) -> None:
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = (
        _FakeGatewayRegistry(
            result=McpExecutionResult(
                ok=False,
                output_text=None,
                error_text="tools/call timeout after 60s",
            )
        )
    )
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "mcp-hosted-step1-stream.yaml",
        "mcp-hosted-step2-stream.yaml",
    )

    async with gateway_client.stream(
        "POST", "/v1/responses", json=_hosted_mcp_request_payload()
    ) as resp:
        assert resp.status_code == 200
        body = await resp.aread()

    text = body.decode("utf-8", errors="replace")
    frames = parse_sse_frames(text)
    events = parse_sse_json_events(frames)

    failed_events = [event for event in events if event.get("type") == "response.mcp_call.failed"]
    assert len(failed_events) == 1
    assert "error" not in failed_events[0]

    completed = extract_completed_response(events)
    mcp_item = next(
        (
            item
            for item in (completed.get("output") or [])
            if isinstance(item, dict) and item.get("type") == "mcp_call"
        ),
        None,
    )
    assert mcp_item is not None
    assert mcp_item["status"] == "failed"
    assert "timeout" in (mcp_item.get("error") or "").lower()
    assert completed.get("incomplete_details") is None


@pytest.mark.anyio
async def test_gateway_stream_hosted_mcp_failed_event_is_openai_sdk_compatible(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
) -> None:
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = (
        _FakeGatewayRegistry(
            result=McpExecutionResult(
                ok=False,
                output_text=None,
                error_text="tools/call timeout after 60s",
            )
        )
    )
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "mcp-hosted-step1-stream.yaml",
        "mcp-hosted-step2-stream.yaml",
    )

    sdk_http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway_client._transport.app),
        base_url="http://gateway/v1",
    )
    sdk = AsyncOpenAI(
        api_key="test",
        base_url="http://gateway/v1",
        http_client=sdk_http_client,
        max_retries=0,
    )

    event_types: list[str] = []
    try:
        async with sdk.responses.stream(
            model="some-model",
            input=[
                {
                    "role": "user",
                    "content": (
                        "Call tool `mcp__github_docs__search_docs` with query='migration notes'. "
                        "Do not answer directly before tool call."
                    ),
                }
            ],
            tools=[{"type": "mcp", "server_label": "github_docs"}],
            tool_choice={"type": "mcp", "server_label": "github_docs", "name": "search_docs"},
        ) as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)
                if isinstance(event_type, str):
                    event_types.append(event_type)
            final = await stream.get_final_response()
    except OpenAIAPIError as exc:  # pragma: no cover - regression guard
        pytest.fail(f"Unexpected OpenAI SDK stream failure for MCP item-level failure: {exc}")
    finally:
        await sdk_http_client.aclose()

    assert "response.mcp_call.failed" in event_types
    assert final.status == "completed"
    mcp_item = next((item for item in (final.output or []) if item.type == "mcp_call"), None)
    assert mcp_item is not None
    assert mcp_item.status == "failed"
    assert "timeout" in (mcp_item.error or "").lower()


@pytest.mark.anyio
async def test_gateway_stream_request_remote_mcp_success_emits_mcp_event_sequence(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_request_remote_builder(
        monkeypatch,
        remote_tools=["search_docs"],
        remote_call_result=McpExecutionResult(
            ok=True,
            output_text='{"results":[{"title":"Remote Notes","url":"https://example.test/remote"}]}',
            error_text=None,
        ),
    )
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "mcp-hosted-step1-stream.yaml",
        "mcp-hosted-step2-stream.yaml",
    )

    async with gateway_client.stream(
        "POST", "/v1/responses", json=_request_remote_mcp_request_payload()
    ) as resp:
        assert resp.status_code == 200
        body = await resp.aread()

    text = body.decode("utf-8", errors="replace")
    events = parse_sse_json_events(parse_sse_frames(text))
    event_types = [event.get("type") for event in events]
    assert "response.mcp_call.completed" in event_types
    completed = extract_completed_response(events)
    mcp_item = next(
        (
            item
            for item in (completed.get("output") or [])
            if isinstance(item, dict) and item.get("type") == "mcp_call"
        ),
        None,
    )
    assert mcp_item is not None
    assert mcp_item["status"] == "completed"
    assert "Remote Notes" in (mcp_item.get("output") or "")


@pytest.mark.anyio
async def test_gateway_request_remote_inventory_failure_maps_to_bad_input(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_request_remote_builder(
        monkeypatch,
        remote_tools=["search_docs"],
        remote_list_error=RuntimeError("remote-inventory-sentinel"),
    )

    resp = await gateway_client.post(
        "/v1/responses",
        json={
            **_request_remote_mcp_request_payload(),
            "stream": False,
        },
    )
    _assert_bad_input(resp, "remote-inventory-sentinel")


@pytest.mark.anyio
async def test_request_remote_inventory_failure_redacts_authorization_token(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_request_remote_builder(
        monkeypatch,
        remote_tools=["search_docs"],
        remote_list_error=RuntimeError("upstream says token=super-secret-token"),
    )
    resp = await gateway_client.post(
        "/v1/responses",
        json={
            **_request_remote_mcp_request_payload(authorization="super-secret-token"),
            "stream": False,
        },
    )
    assert resp.status_code == 422
    payload = resp.json()
    assert payload.get("error") == "bad_input"
    assert "super-secret-token" not in payload.get("message", "")
    assert "***" in payload.get("message", "")


@pytest.mark.anyio
async def test_gateway_stream_request_remote_runtime_failure_is_item_level(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_request_remote_builder(
        monkeypatch,
        remote_tools=["search_docs"],
        remote_call_result=McpExecutionResult(
            ok=False,
            output_text=None,
            error_text="remote-call-sentinel",
        ),
    )
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "mcp-hosted-step1-stream.yaml",
        "mcp-hosted-step2-stream.yaml",
    )

    async with gateway_client.stream(
        "POST", "/v1/responses", json=_request_remote_mcp_request_payload()
    ) as resp:
        assert resp.status_code == 200
        body = await resp.aread()

    events = parse_sse_json_events(parse_sse_frames(body.decode("utf-8", errors="replace")))
    event_types = [event.get("type") for event in events]
    assert "response.mcp_call.failed" in event_types
    assert "response.failed" not in event_types
    assert "response.completed" in event_types
    failed_event = next(
        event for event in events if event.get("type") == "response.mcp_call.failed"
    )
    assert "error" not in failed_event
    completed = extract_completed_response(events)
    mcp_item = next(
        (
            item
            for item in (completed.get("output") or [])
            if isinstance(item, dict) and item.get("type") == "mcp_call"
        ),
        None,
    )
    assert mcp_item is not None
    assert "remote-call-sentinel" in (mcp_item.get("error") or "")


@pytest.mark.anyio
async def test_request_remote_missing_auth_can_fail_pre_run_discovery(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_request_remote_builder(
        monkeypatch,
        remote_tools=["search_docs"],
        remote_require_auth_for_list=True,
    )
    payload = _request_remote_mcp_request_payload(authorization=None)
    payload["stream"] = False
    resp = await gateway_client.post("/v1/responses", json=payload)
    _assert_bad_input(resp, "remote-list-auth-required")


@pytest.mark.anyio
async def test_request_remote_missing_auth_can_fail_at_runtime_item_level(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_request_remote_builder(
        monkeypatch,
        remote_tools=["search_docs"],
        remote_require_auth_for_call=True,
    )
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "mcp-hosted-step1-stream.yaml",
        "mcp-hosted-step2-stream.yaml",
    )
    payload = _request_remote_mcp_request_payload(authorization=None)

    async with gateway_client.stream("POST", "/v1/responses", json=payload) as resp:
        assert resp.status_code == 200
        body = await resp.aread()

    events = parse_sse_json_events(parse_sse_frames(body.decode("utf-8", errors="replace")))
    failed_event = next(
        event for event in events if event.get("type") == "response.mcp_call.failed"
    )
    assert "error" not in failed_event
    completed = extract_completed_response(events)
    mcp_item = next(
        (
            item
            for item in (completed.get("output") or [])
            if isinstance(item, dict) and item.get("type") == "mcp_call"
        ),
        None,
    )
    assert mcp_item is not None
    assert "remote-call-auth-required" in (mcp_item.get("error") or "")


@pytest.mark.anyio
async def test_gateway_stream_max_tool_calls_is_not_runtime_enforced(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
) -> None:
    manager = _CountingGatewayManager(
        result=McpExecutionResult(
            ok=True,
            output_text='{"results":[{"title":"Migration Notes","url":"https://example.test/migration"}]}',
            error_text=None,
        )
    )
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = manager
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "mcp-hosted-step1-stream.yaml",
        "mcp-hosted-step2-stream.yaml",
    )

    payload = _hosted_mcp_request_payload()
    payload["max_tool_calls"] = 0
    async with gateway_client.stream("POST", "/v1/responses", json=payload) as resp:
        assert resp.status_code == 200
        body = await resp.aread()

    events = parse_sse_json_events(parse_sse_frames(body.decode("utf-8", errors="replace")))
    event_types = [event.get("type") for event in events]
    assert "response.completed" in event_types
    assert "error" not in event_types
    assert "response.failed" not in event_types
    assert manager.call_count == 1

    completed = extract_completed_response(events)
    assert completed.get("status") == "completed"
    assert completed.get("error") is None
    assert completed.get("incomplete_details") is None
    assert completed.get("max_tool_calls") == 0
    assert any(
        isinstance(item, dict) and item.get("type") == "mcp_call"
        for item in (completed.get("output") or [])
    )


@pytest.mark.anyio
async def test_gateway_non_stream_max_tool_calls_is_not_runtime_enforced(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
) -> None:
    manager = _CountingGatewayManager(
        result=McpExecutionResult(
            ok=True,
            output_text='{"results":[{"title":"Migration Notes","url":"https://example.test/migration"}]}',
            error_text=None,
        )
    )
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = manager
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "mcp-hosted-step1-stream.yaml",
        "mcp-hosted-step2-stream.yaml",
    )

    payload = _hosted_mcp_request_payload()
    payload["stream"] = False
    payload["max_tool_calls"] = 0

    resp = await gateway_client.post("/v1/responses", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "completed"
    assert data.get("error") is None
    assert data.get("incomplete_details") is None
    assert data.get("max_tool_calls") == 0
    assert manager.call_count == 1
    assert any(
        isinstance(item, dict) and item.get("type") == "mcp_call"
        for item in (data.get("output") or [])
    )


def test_lm_failure_summary_includes_last_failed_mcp_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_stack.lm as lm
    import agentic_stack.lm_failures as lm_failures

    counters = lm._FailureCounters()
    counters.observe(
        McpCallStarted(
            item_key="item_1",
            server_label="docs",
            name="search_docs",
            initial_arguments_json='{"query":"first"}',
            mode="hosted",
        )
    )
    counters.observe(
        McpCallArgumentsDone(
            item_key="item_1",
            arguments_json='{"query":"final"}',
        )
    )
    counters.observe(
        McpCallFailed(
            item_key="item_1",
            error_text="upstream timeout",
        )
    )

    logged: list[str] = []

    def _fake_error(message, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered.format(*args)
        logged.append(rendered)

    monkeypatch.setattr(lm_failures.logger, "error", _fake_error)
    monkeypatch.setattr(lm_failures.logger, "warning", lambda *args, **kwargs: None)

    lm._log_failure_summary(
        response_id="resp_1",
        failure_phase="stream",
        error_class="ModelHTTPError",
        log_level="error",
        upstream_status_code=502,
        error_message="boom",
        messages=[],
        counters=counters,
        upstream_error_raw=None,
        log_model_messages=False,
    )

    merged = "\n".join(logged)
    assert "last_failed_mcp_signature" in merged
    assert "'mode': 'hosted'" in merged
    assert "'server_label': 'docs'" in merged
    assert "'tool_name': 'search_docs'" in merged
    assert "mcp_failed_count_total" in merged
    assert "mcp_failed_count_hosted" in merged
    assert "mcp_failed_count_request_remote" in merged
    assert "args_hash" not in merged


def test_lm_failure_summary_tracks_request_remote_mode_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_stack.lm as lm
    import agentic_stack.lm_failures as lm_failures

    counters = lm._FailureCounters()
    counters.observe(
        McpCallStarted(
            item_key="item_remote",
            server_label="github_docs",
            name="search_docs",
            initial_arguments_json='{"query":"x"}',
            mode="request_remote",
        )
    )
    counters.observe(McpCallFailed(item_key="item_remote", error_text="remote boom"))

    logged: list[str] = []

    def _fake_error(message, *args, **kwargs):
        rendered = str(message)
        if args:
            rendered = rendered.format(*args)
        logged.append(rendered)

    monkeypatch.setattr(lm_failures.logger, "error", _fake_error)
    monkeypatch.setattr(lm_failures.logger, "warning", lambda *args, **kwargs: None)

    lm._log_failure_summary(
        response_id="resp_remote",
        failure_phase="stream",
        error_class="ModelHTTPError",
        log_level="error",
        upstream_status_code=502,
        error_message="boom",
        messages=[],
        counters=counters,
        upstream_error_raw=None,
        log_model_messages=False,
    )

    merged = "\n".join(logged)
    assert "'mcp_failed_count_total': 1" in merged
    assert "'mcp_failed_count_hosted': 0" in merged
    assert "'mcp_failed_count_request_remote': 1" in merged
    assert "'mode': 'request_remote'" in merged


@pytest.mark.anyio
async def test_gateway_stream_hosted_mcp_input_validation_failure_emits_failed_item_without_transport_call(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
) -> None:
    manager = _ValidationFailGatewayManager()
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = manager
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "mcp-hosted-step1-stream.yaml",
        "mcp-hosted-step2-stream.yaml",
    )

    async with gateway_client.stream(
        "POST", "/v1/responses", json=_hosted_mcp_request_payload()
    ) as resp:
        assert resp.status_code == 200
        body = await resp.aread()

    text = body.decode("utf-8", errors="replace")
    events = parse_sse_json_events(parse_sse_frames(text))
    failed_events = [event for event in events if event.get("type") == "response.mcp_call.failed"]
    assert len(failed_events) >= 1
    assert "error" not in failed_events[0]
    completed = extract_completed_response(events)
    mcp_item = next(
        (
            item
            for item in (completed.get("output") or [])
            if isinstance(item, dict) and item.get("type") == "mcp_call"
        ),
        None,
    )
    assert mcp_item is not None
    assert (mcp_item.get("error") or "").startswith("input_validation_error:")
    assert manager.call_count == 0


@pytest.mark.anyio
async def test_gateway_non_stream_hosted_mcp_success_returns_mcp_call_item(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
) -> None:
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = _FakeGatewayRegistry(
        result=McpExecutionResult(
            ok=True,
            output_text='{"results":[{"title":"Migration Notes","url":"https://example.test/migration"}]}',
            error_text=None,
        )
    )
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "mcp-hosted-step1-stream.yaml",
        "mcp-hosted-step2-stream.yaml",
    )

    payload = _hosted_mcp_request_payload()
    payload["stream"] = False

    resp = await gateway_client.post("/v1/responses", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    mcp_item = next(
        (
            item
            for item in (data.get("output") or [])
            if isinstance(item, dict) and item.get("type") == "mcp_call"
        ),
        None,
    )
    assert mcp_item is not None
    assert mcp_item["status"] == "completed"
    assert "Migration Notes" in (mcp_item.get("output") or "")


@pytest.mark.anyio
async def test_mixed_tools_keep_function_calls_client_owned(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
) -> None:
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = (
        _FakeGatewayRegistry(
            result=McpExecutionResult(ok=True, output_text='{"ok":true}', error_text=None)
        )
    )
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "vllm-code_interpreter-step1-stream.yaml"
    )

    async with gateway_client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": True,
            "input": [
                {
                    "role": "user",
                    "content": "You MUST call the code_interpreter tool now. Execute: 2+2.",
                }
            ],
            "tools": [
                {"type": "mcp", "server_label": "github_docs"},
                {
                    "type": "function",
                    "name": "code_interpreter",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            ],
            "tool_choice": "auto",
        },
    ) as resp:
        assert resp.status_code == 200
        body = await resp.aread()

    text = body.decode("utf-8", errors="replace")
    events = parse_sse_json_events(parse_sse_frames(text))
    event_types = [event.get("type") for event in events]
    assert "response.function_call_arguments.done" in event_types
    assert "response.mcp_call.in_progress" not in event_types


# ==============================
# Gateway statefulness
# ==============================


class _TrackingGatewayManager:
    def __init__(self) -> None:
        self.list_tools_calls = 0

    def is_enabled(self) -> bool:
        return True

    def has_server(self, server_label: str) -> bool:
        return server_label == "github_docs"

    def is_server_available(self, server_label: str) -> bool:
        return server_label == "github_docs"

    def get_server_startup_error(self, server_label: str) -> str | None:
        return None

    async def list_tools(self, server_label: str) -> list[McpToolInfo]:
        assert server_label == "github_docs"
        self.list_tools_calls += 1
        return [
            _tool_info(
                "search_docs",
                description="Search docs",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            )
        ]

    async def call_tool(
        self,
        server_label: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> McpExecutionResult:
        assert server_label == "github_docs"
        assert tool_name == "search_docs"
        return McpExecutionResult(
            ok=True,
            output_text='{"results":[{"title":"Migration Notes","url":"https://example.test/migration"}]}',
            error_text=None,
        )

    async def get_server_toolset(self, server_label: str) -> AbstractToolset[Any]:
        return _HostedRegistryNativeToolset(manager=self, server_label=server_label)

    async def get_server_mcp_tools_by_name(self, server_label: str) -> dict[str, ToolsetTool[Any]]:
        toolset = await self.get_server_toolset(server_label)
        return await toolset.get_tools(ctx=None)

    async def get_server_runtime(self, server_label: str):
        return SimpleNamespace(
            mcp_toolset=await self.get_server_toolset(server_label),
            allowed_tool_infos={tool.name: tool for tool in await self.list_tools(server_label)},
            allowed_mcp_tools_by_name=await self.get_server_mcp_tools_by_name(server_label),
        )


def _extract_completed_response_id(sse_text: str) -> str:
    frames = parse_sse_frames(sse_text)
    events = parse_sse_json_events(frames)
    completed = extract_completed_response(events)
    return str(completed["id"])


@pytest.mark.anyio
async def test_previous_response_id_reuses_mixed_effective_tools_when_omitted(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
) -> None:
    manager = _TrackingGatewayManager()
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = manager

    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "mcp-hosted-step1-stream.yaml",
        "mcp-hosted-step2-stream.yaml",
        "text-single-stream.yaml",
    )

    first_payload = {
        "model": "some-model",
        "stream": True,
        "input": [
            {
                "role": "user",
                "content": "Call tool `mcp__github_docs__search_docs` with query='migration notes'. Do not answer directly before tool call.",
            }
        ],
        "tools": [
            {"type": "mcp", "server_label": "github_docs"},
            {
                "type": "function",
                "name": "get_billing_status",
                "parameters": {
                    "type": "object",
                    "properties": {"account_id": {"type": "string"}},
                    "required": ["account_id"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ],
        "tool_choice": {"type": "mcp", "server_label": "github_docs", "name": "search_docs"},
    }

    async with gateway_client.stream("POST", "/v1/responses", json=first_payload) as resp1:
        assert resp1.status_code == 200
        body1 = await resp1.aread()

    text1 = body1.decode("utf-8", errors="replace")
    r1 = _extract_completed_response_id(text1)
    events1 = parse_sse_json_events(parse_sse_frames(text1))
    assert any(event.get("type") == "response.mcp_call.completed" for event in events1)
    assert not any(
        event.get("type") == "response.function_call_arguments.delta" for event in events1
    )

    import agentic_stack.lm as lm

    store = lm.get_default_response_store()
    stored = await store.get(response_id=r1)
    assert stored is not None
    payload = stored.payload()
    effective_types = [tool.type for tool in (payload.effective_tools or [])]
    assert "mcp" in effective_types
    assert "function" in effective_types

    async with gateway_client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": True,
            "previous_response_id": r1,
            "input": [{"role": "user", "content": "Please summarize briefly."}],
        },
    ) as resp2:
        assert resp2.status_code == 200
        body2 = await resp2.aread()

    text2 = body2.decode("utf-8", errors="replace")
    assert "event: response.completed" in text2
    assert "data: [DONE]\n\n" in text2

    assert manager.list_tools_calls >= 2


@pytest.mark.anyio
async def test_request_remote_authorization_is_not_persisted_and_omit_tools_continuation_is_allowed(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_request_remote_builder(
        monkeypatch,
        remote_tools=["search_docs"],
        remote_require_auth_for_call=True,
    )
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "mcp-hosted-step1-stream.yaml",
        "mcp-hosted-step2-stream.yaml",
        "mcp-hosted-step1-stream.yaml",
        "mcp-hosted-step2-stream.yaml",
    )

    first_payload = _request_remote_mcp_request_payload(authorization="super-secret-token")
    async with gateway_client.stream("POST", "/v1/responses", json=first_payload) as resp1:
        assert resp1.status_code == 200
        body1 = await resp1.aread()

    response_id = _extract_completed_response_id(body1.decode("utf-8", errors="replace"))

    import agentic_stack.lm as lm

    store = lm.get_default_response_store()
    stored = await store.get(response_id=response_id)
    assert stored is not None
    payload = stored.payload()
    persisted_mcp_tools = [tool for tool in (payload.effective_tools or []) if tool.type == "mcp"]
    assert len(persisted_mcp_tools) == 1
    assert persisted_mcp_tools[0].authorization is None
    assert persisted_mcp_tools[0].headers is None

    second_payload = {
        "model": "some-model",
        "stream": True,
        "previous_response_id": response_id,
        "input": [{"role": "user", "content": "Try the same MCP tool call again."}],
    }
    async with gateway_client.stream("POST", "/v1/responses", json=second_payload) as resp2:
        assert resp2.status_code == 200
        body2 = await resp2.aread()

    events2 = parse_sse_json_events(parse_sse_frames(body2.decode("utf-8", errors="replace")))
    event_types = [event.get("type") for event in events2]
    assert "response.mcp_call.failed" in event_types
    assert "response.completed" in event_types


@pytest.mark.anyio
async def test_previous_response_id_fails_when_persisted_mcp_server_is_missing(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
) -> None:
    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = (
        _TrackingGatewayManager()
    )
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "mcp-hosted-step1-stream.yaml",
        "mcp-hosted-step2-stream.yaml",
    )

    async with gateway_client.stream(
        "POST", "/v1/responses", json=_hosted_mcp_request_payload()
    ) as resp1:
        assert resp1.status_code == 200
        body1 = await resp1.aread()

    response_id = _extract_completed_response_id(body1.decode("utf-8", errors="replace"))

    class _MissingServerGatewayManager:
        def is_enabled(self) -> bool:
            return True

        def has_server(self, server_label: str) -> bool:
            return False

        def is_server_available(self, server_label: str) -> bool:
            return False

        def get_server_startup_error(self, server_label: str) -> str | None:
            return "removed"

        async def list_tools(self, server_label: str) -> list[McpToolInfo]:
            raise BuiltinMcpRuntimeUnknownServerError(f"Unknown MCP server_label: {server_label}")

        async def call_tool(
            self,
            server_label: str,
            tool_name: str,
            arguments: dict[str, object],
        ) -> McpExecutionResult:
            raise AssertionError("call_tool should not be called for missing server.")

    gateway_client._transport.app.state.agentic_stack.builtin_mcp_runtime_client = (
        _MissingServerGatewayManager()
    )

    resp2 = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "previous_response_id": response_id,
            "input": [{"role": "user", "content": "Continue with latest status."}],
        },
    )
    assert resp2.status_code == 422
    payload = resp2.json()
    assert payload.get("error") == "bad_input"
    assert "Unknown MCP server_label: github_docs" in payload.get("message", "")
