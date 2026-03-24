from __future__ import annotations

import pytest

from agentic_stack.entrypoints.serve import _build_root_parser


def test_serve_parser_accepts_remote_upstream_flags() -> None:
    parser = _build_root_parser()
    ns = parser.parse_args(
        [
            "serve",
            "--gateway-port",
            "8458",
            "--upstream",
            "http://127.0.0.1:8000/v1",
            "--web-search-profile",
            "exa_mcp",
            "--code-interpreter-startup-timeout",
            "12.5",
            "--upstream-ready-timeout",
            "90",
            "--upstream-ready-interval",
            "2.5",
            "--mcp-config",
            "/tmp/mcp.json",
            "--mcp-port",
            "6101",
        ]
    )
    assert ns.command == "serve"
    assert ns.gateway_port == 8458
    assert ns.upstream == "http://127.0.0.1:8000/v1"
    assert ns.web_search_profile == "exa_mcp"
    assert ns.code_interpreter_startup_timeout == "12.5"
    assert ns.upstream_ready_timeout == "90"
    assert ns.upstream_ready_interval == "2.5"
    assert ns.mcp_config == "/tmp/mcp.json"
    assert ns.mcp_port == "6101"


def test_serve_parser_help_lists_web_search_profile_choices(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = _build_root_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["serve", "--help"])

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "--web-search-profile" in help_text
    assert "Choices:" in help_text
    assert "duckduckgo_plus_fetch, exa_mcp." in help_text
