from __future__ import annotations

from fastapi import FastAPI

from agentic_stack.configs.builders import (
    build_runtime_config_for_integrated,
    build_runtime_config_for_standalone,
    build_runtime_config_for_supervisor,
)
from agentic_stack.configs.sources import EnvSource
from agentic_stack.configs.startup import validate_responses_cli_args
from agentic_stack.entrypoints.gateway._app import (
    activate_gateway_runtime,
    augment_standalone_gateway_app,
)
from agentic_stack.lm import INTEGRATED_LM_CLIENT, LM_CLIENT, get_openai_provider
from agentic_stack.tools.ids import CODE_INTERPRETER_TOOL, WEB_SEARCH_TOOL


def test_build_runtime_config_for_standalone_reads_env_overrides() -> None:
    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={
                "VR_LLM_API_BASE": "http://127.0.0.1:9000/v1",
                "VR_OPENAI_API_KEY": "runtime-key",
                "VR_WEB_SEARCH_PROFILE": "exa_mcp",
                "VR_CODE_INTERPRETER_MODE": "external",
                "VR_CODE_INTERPRETER_PORT": "6111",
                "VR_CODE_INTERPRETER_WORKERS": "2",
                "VR_CODE_INTERPRETER_STARTUP_TIMEOUT": "12.5",
            }
        )
    )

    assert runtime_config.runtime_mode == "standalone"
    assert runtime_config.llm_api_base == "http://127.0.0.1:9000/v1"
    assert runtime_config.openai_api_key == "runtime-key"
    assert runtime_config.web_search_profile == "exa_mcp"
    assert runtime_config.code_interpreter_mode == "external"
    assert runtime_config.code_interpreter_port == 6111
    assert runtime_config.code_interpreter_workers == 2
    assert runtime_config.code_interpreter_startup_timeout_s == 12.5
    assert runtime_config.mcp_builtin_runtime_url == "http://127.0.0.1:5981"


def test_build_runtime_config_for_standalone_without_builtin_mcp_need_leaves_url_unset() -> None:
    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={
                "VR_LLM_API_BASE": "http://127.0.0.1:9000/v1",
                "VR_CODE_INTERPRETER_MODE": "external",
                "VR_CODE_INTERPRETER_PORT": "6111",
                "VR_CODE_INTERPRETER_WORKERS": "2",
            }
        )
    )

    assert runtime_config.web_search_profile is None
    assert runtime_config.mcp_builtin_runtime_url is None


def test_config_helpers_do_not_bootstrap_builtin_registries(
    monkeypatch,
) -> None:
    import agentic_stack.tools as tools_mod

    monkeypatch.setattr(tools_mod, "TOOLS", {})

    _ = validate_responses_cli_args(
        raw_values={
            "web_search_profile": None,
            "code_interpreter_mode": "disabled",
            "code_interpreter_port": None,
            "code_interpreter_workers": None,
            "code_interpreter_startup_timeout_s": None,
            "upstream_ready_timeout_s": None,
            "upstream_ready_interval_s": None,
            "mcp_config_path": None,
            "mcp_port": None,
        },
        labels={},
        error_prefix="[test]",
        error_factory=RuntimeError,
    )
    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={
                "VR_LLM_API_BASE": "http://127.0.0.1:9000/v1",
                "VR_CODE_INTERPRETER_MODE": "disabled",
            }
        )
    )

    assert runtime_config.web_search_profile is None
    assert tools_mod.TOOLS == {}


def test_build_runtime_config_for_supervisor_ignores_web_search_env_without_cli() -> None:
    from argparse import Namespace

    runtime_config = build_runtime_config_for_supervisor(
        args=Namespace(
            upstream="http://127.0.0.1:8457/v1",
            gateway_host=None,
            gateway_port=None,
            gateway_workers=None,
            web_search_profile=None,
            code_interpreter="disabled",
            code_interpreter_port=None,
            code_interpreter_workers=None,
            code_interpreter_startup_timeout=None,
            upstream_ready_timeout=None,
            upstream_ready_interval=None,
            mcp_config=None,
            mcp_port=None,
        ),
        env=EnvSource(environ={"VR_WEB_SEARCH_PROFILE": "exa_mcp"}),
    )

    assert runtime_config.web_search_profile is None


def test_activate_gateway_runtime_attaches_runtime_config() -> None:
    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(environ={"VR_LLM_API_BASE": "http://127.0.0.1:8457/v1"})
    )

    app = FastAPI()
    augment_standalone_gateway_app(
        app,
        include_upstream_proxy=False,
        include_metrics_route=False,
        include_cors=False,
        customize_openapi=False,
    )
    activate_gateway_runtime(app, runtime_config=runtime_config)

    attached = app.state.agentic_stack.runtime_config
    assert attached is not None
    assert attached.runtime_mode == "standalone"
    assert attached.llm_api_base == "http://127.0.0.1:8457/v1"


def test_activate_gateway_runtime_registers_runtime_tool_handlers(monkeypatch) -> None:
    import agentic_stack.tools as tools_mod

    monkeypatch.setattr(tools_mod, "TOOLS", {})

    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(environ={"VR_LLM_API_BASE": "http://127.0.0.1:8457/v1"})
    )

    app = FastAPI()
    augment_standalone_gateway_app(
        app,
        include_upstream_proxy=False,
        include_metrics_route=False,
        include_cors=False,
        customize_openapi=False,
    )
    activate_gateway_runtime(app, runtime_config=runtime_config)

    assert CODE_INTERPRETER_TOOL in tools_mod.TOOLS
    assert WEB_SEARCH_TOOL in tools_mod.TOOLS


def test_get_openai_provider_uses_integrated_http_client(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeProvider:
        def __init__(self, *, api_key, base_url, http_client) -> None:  # type: ignore[no-untyped-def]
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["http_client"] = http_client

    monkeypatch.setattr("agentic_stack.lm.OpenAIProvider", _FakeProvider)

    runtime_config = build_runtime_config_for_integrated(
        env=EnvSource(environ={"VR_OPENAI_API_KEY": "ctx-key"}),
        host="0.0.0.0",
        port=8000,
        web_search_profile="exa_mcp",
        code_interpreter_mode="disabled",
        code_interpreter_port=5970,
        code_interpreter_workers=0,
        code_interpreter_startup_timeout_s=30.0,
        mcp_config_path=None,
        mcp_builtin_runtime_url=None,
    )

    _ = get_openai_provider(runtime_config)

    assert captured["api_key"] == "ctx-key"
    assert captured["base_url"] == "http://127.0.0.1:8000/v1"
    assert captured["http_client"] is INTEGRATED_LM_CLIENT


def test_build_runtime_config_for_integrated_ignores_web_search_env_without_cli() -> None:
    runtime_config = build_runtime_config_for_integrated(
        env=EnvSource(environ={"VR_WEB_SEARCH_PROFILE": "exa_mcp"}),
        host="127.0.0.1",
        port=8000,
        web_search_profile=None,
        code_interpreter_mode="disabled",
        code_interpreter_port=5970,
        code_interpreter_workers=0,
        code_interpreter_startup_timeout_s=30.0,
        mcp_config_path=None,
        mcp_builtin_runtime_url=None,
    )

    assert runtime_config.web_search_profile is None


def test_get_openai_provider_defaults_to_standalone_client(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeProvider:
        def __init__(self, *, api_key, base_url, http_client) -> None:  # type: ignore[no-untyped-def]
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["http_client"] = http_client

    monkeypatch.setattr("agentic_stack.lm.OpenAIProvider", _FakeProvider)

    runtime_config = build_runtime_config_for_standalone(env=EnvSource(environ={}))

    _ = get_openai_provider(runtime_config)

    assert captured["http_client"] is LM_CLIENT
