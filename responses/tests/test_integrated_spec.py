from __future__ import annotations

import pytest

from agentic_stack.entrypoints.vllm._spec import (
    IntegratedSpecError,
    build_integrated_serve_spec,
    format_integrated_help,
    should_show_integrated_help,
)


def test_build_integrated_serve_spec_strips_responses_flags_and_uses_cli_values() -> None:
    spec = build_integrated_serve_spec(
        [
            "serve",
            "meta-llama/Llama-3.2-3B-Instruct",
            "--responses",
            "--responses-web-search-profile",
            "exa_mcp",
            "--responses-code-interpreter=external",
            "--responses-code-interpreter-port",
            "6111",
            "--responses-code-interpreter-workers",
            "3",
            "--responses-code-interpreter-startup-timeout",
            "42",
            "--responses-mcp-config",
            "/tmp/mcp.json",
            "--responses-mcp-port",
            "6201",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
    )

    assert spec.vllm_args == [
        "serve",
        "meta-llama/Llama-3.2-3B-Instruct",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    assert spec.web_search_profile == "exa_mcp"
    assert spec.code_interpreter_mode == "external"
    assert spec.code_interpreter_port == 6111
    assert spec.code_interpreter_workers == 3
    assert spec.code_interpreter_startup_timeout_s == 42.0
    assert spec.mcp_config_path == "/tmp/mcp.json"
    assert spec.mcp_port == 6201


def test_build_integrated_serve_spec_uses_builtin_defaults_when_flags_omitted() -> None:
    spec = build_integrated_serve_spec(
        ["serve", "model", "--responses", "--port", "8000"],
    )

    assert spec.vllm_args == ["serve", "model", "--port", "8000"]
    assert spec.web_search_profile is None
    assert spec.code_interpreter_mode == "spawn"
    assert spec.code_interpreter_port == 5970
    assert spec.code_interpreter_workers == 0
    assert spec.code_interpreter_startup_timeout_s == 600.0


def test_build_integrated_serve_spec_rejects_non_serve_commands() -> None:
    with pytest.raises(IntegratedSpecError, match=r"supported only for `vllm serve`"):
        build_integrated_serve_spec(["chat", "--responses"])


def test_build_integrated_serve_spec_rejects_headless_mode() -> None:
    with pytest.raises(IntegratedSpecError, match=r"does not support `--headless`"):
        build_integrated_serve_spec(["serve", "model", "--responses", "--headless"])


def test_build_integrated_serve_spec_rejects_multiple_api_servers() -> None:
    with pytest.raises(IntegratedSpecError, match=r"single API server"):
        build_integrated_serve_spec(
            ["serve", "model", "--responses", "--api-server-count", "2"],
        )


def test_build_integrated_serve_spec_rejects_invalid_numeric_cli_flag() -> None:
    with pytest.raises(
        IntegratedSpecError, match=r"invalid --responses-code-interpreter-port='abc'"
    ):
        build_integrated_serve_spec(
            ["serve", "model", "--responses", "--responses-code-interpreter-port", "abc"],
        )

    with pytest.raises(IntegratedSpecError, match=r"invalid --responses-mcp-port='abc'"):
        build_integrated_serve_spec(
            ["serve", "model", "--responses", "--responses-mcp-port", "abc"],
        )

    with pytest.raises(
        IntegratedSpecError,
        match=r"--responses-mcp-port requires --responses-mcp-config or a web_search profile",
    ):
        build_integrated_serve_spec(
            ["serve", "model", "--responses", "--responses-mcp-port", "6201"],
        )


def test_build_integrated_serve_spec_rejects_unknown_web_search_profile() -> None:
    with pytest.raises(
        IntegratedSpecError,
        match=(
            r"unknown --responses-web-search-profile='missing_profile'; expected one of: "
            r"duckduckgo_plus_fetch, exa_mcp"
        ),
    ):
        build_integrated_serve_spec(
            [
                "serve",
                "model",
                "--responses",
                "--responses-web-search-profile",
                "missing_profile",
            ],
        )


def test_build_integrated_serve_spec_allows_mcp_port_without_config_for_web_search() -> None:
    spec = build_integrated_serve_spec(
        [
            "serve",
            "model",
            "--responses",
            "--responses-web-search-profile",
            "exa_mcp",
            "--responses-mcp-port",
            "6201",
        ]
    )

    assert spec.web_search_profile == "exa_mcp"
    assert spec.mcp_config_path is None
    assert spec.mcp_port == 6201


def test_should_show_integrated_help_requires_help_and_responses() -> None:
    assert should_show_integrated_help(["serve", "model", "--responses", "--help"]) is True
    assert should_show_integrated_help(["serve", "model", "--responses"]) is False


def test_format_integrated_help_mentions_upstream_vllm_help() -> None:
    help_text = format_integrated_help()
    assert "--responses-web-search-profile" in help_text
    assert "Choices: duckduckgo_plus_fetch, exa_mcp." in help_text
    assert "--responses-mcp-config" in help_text
    assert "--responses-mcp-port" in help_text
    assert "vllm serve --help" in help_text
