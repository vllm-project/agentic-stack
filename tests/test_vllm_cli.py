import pytest

from agentic_stack.entrypoints import vllm_cli


def _fake_vllm_proc():
    """A minimal subprocess.Popen stand-in."""

    class _Proc:
        def terminate(self) -> None:
            pass

        def wait(self) -> int:
            return 0

    return _Proc()


def test_vllm_cli_delegates_when_flag_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    delegated: list[list[str]] = []

    def _fake_delegate(argv: list[str]) -> None:
        delegated.append(list(argv))

    monkeypatch.setattr(vllm_cli, "_delegate_to_vllm", _fake_delegate)

    vllm_cli.main(["serve", "model", "--port", "8000"])

    assert delegated == [["serve", "model", "--port", "8000"]]


def test_vllm_cli_runs_serve_when_flag_present(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(vllm_cli, "_spawn_vllm", lambda argv: _fake_vllm_proc())
    monkeypatch.setattr(vllm_cli, "run", lambda config: seen.update({"config": config}))

    vllm_cli.main(
        [
            "serve",
            "model",
            "--agentic-stack",
            "--port",
            "8457",
            "--gateway-port",
            "9000",
        ]
    )

    assert "config" in seen
    config = seen["config"]
    assert config.llm_api_base == "http://127.0.0.1:8457"
    assert config.gateway_port == 9000


def test_vllm_cli_infers_llm_api_base_from_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(vllm_cli, "_spawn_vllm", lambda argv: _fake_vllm_proc())
    monkeypatch.setattr(vllm_cli, "run", lambda config: seen.update({"config": config}))

    vllm_cli.main(["serve", "model", "--agentic-stack", "--port", "9999"])

    assert seen["config"].llm_api_base == "http://127.0.0.1:9999"


def test_vllm_cli_infers_default_port_when_no_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(vllm_cli, "_spawn_vllm", lambda argv: _fake_vllm_proc())
    monkeypatch.setattr(vllm_cli, "run", lambda config: seen.update({"config": config}))

    vllm_cli.main(["serve", "model", "--agentic-stack"])

    assert seen["config"].llm_api_base == "http://127.0.0.1:8000"


def test_vllm_cli_normalizes_v1_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(vllm_cli, "_spawn_vllm", lambda argv: _fake_vllm_proc())
    monkeypatch.setattr(vllm_cli, "run", lambda config: seen.update({"config": config}))

    vllm_cli.main(
        [
            "serve",
            "model",
            "--agentic-stack",
            "--llm-api-base",
            "http://127.0.0.1:8000/v1",
        ]
    )

    assert seen["config"].llm_api_base == "http://127.0.0.1:8000"


def test_vllm_cli_prints_help_for_agentic_stack_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        vllm_cli.main(["serve", "model", "--agentic-stack", "--help"])

    assert excinfo.value.code == 0
    assert "--gateway-port" in capsys.readouterr().out
