from __future__ import annotations

import pytest

from agentic_stack.entrypoints import vllm_cli


def test_vllm_cli_delegates_when_responses_flag_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated: list[list[str]] = []

    monkeypatch.setattr(
        vllm_cli,
        "_delegate_to_upstream_vllm",
        lambda argv: delegated.append(list(argv)) or 0,
    )

    with pytest.raises(SystemExit) as excinfo:
        vllm_cli.main(["serve", "model", "--port", "8000"])

    assert excinfo.value.code == 0
    assert delegated == [["serve", "model", "--port", "8000"]]


def test_vllm_cli_runs_integrated_serve_when_responses_flag_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def _fake_run(spec) -> int:  # type: ignore[no-untyped-def]
        seen["spec"] = spec
        return 0

    monkeypatch.setattr(vllm_cli, "run_integrated_serve", _fake_run)

    with pytest.raises(SystemExit) as excinfo:
        vllm_cli.main(
            [
                "serve",
                "model",
                "--responses",
                "--responses-code-interpreter=disabled",
            ]
        )

    assert excinfo.value.code == 0
    spec = seen["spec"]
    assert spec.vllm_args == ["serve", "model"]
    assert spec.code_interpreter_mode == "disabled"


def test_vllm_cli_bootstraps_builtin_registries_before_integrated_spec_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_stack.tools as tools_mod

    seen: dict[str, object] = {}

    def _fake_run(spec) -> int:  # type: ignore[no-untyped-def]
        seen["spec"] = spec
        return 0

    monkeypatch.setattr(tools_mod, "TOOLS", {})
    monkeypatch.setattr(vllm_cli, "run_integrated_serve", _fake_run)

    with pytest.raises(SystemExit) as excinfo:
        vllm_cli.main(
            [
                "serve",
                "model",
                "--responses",
                "--responses-web-search-profile",
                "exa_mcp",
                "--responses-code-interpreter=disabled",
            ]
        )

    assert excinfo.value.code == 0
    assert seen["spec"].web_search_profile == "exa_mcp"


def test_vllm_cli_errors_for_responses_on_non_serve_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        vllm_cli.main(["chat", "--responses"])

    assert excinfo.value.code == 2
    assert "supported only for `vllm serve`" in capsys.readouterr().err


def test_vllm_cli_prints_integrated_help_for_responses_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        vllm_cli.main(["serve", "model", "--responses", "--help"])

    assert excinfo.value.code == 0
    assert "--responses-code-interpreter" in capsys.readouterr().out
