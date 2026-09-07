from __future__ import annotations

import builtins
from dataclasses import asdict

import pytest

from agentic_api import __version__
from agentic_api.cli import ServeOptions, build_parser, main


def parse_serve_args(*args: str) -> ServeOptions:
    parser = build_parser()
    namespace = parser.parse_args(["serve", *args])
    assert namespace.command == "serve"
    assert isinstance(namespace.options, ServeOptions)
    return namespace.options


def test_serve_with_model_uses_local_mode_defaults() -> None:
    options = parse_serve_args("--model", "Qwen/Qwen3-4B")

    assert asdict(options) == {
        "mode": "local",
        "model": "Qwen/Qwen3-4B",
        "vllm_base_url": None,
        "host": "0.0.0.0",
        "port": 9000,
        "startup_timeout_s": 600.0,
        "shutdown_timeout_s": 10.0,
        "vllm_port": 8000,
        "gateway_api_key_env": "OPENAI_API_KEY",
        "vllm_api_key_env": "AGENTIC_VLLM_API_KEY",
        "vllm_args": [],
    }


def test_serve_with_vllm_base_url_uses_remote_mode() -> None:
    options = parse_serve_args("--vllm-base-url", "http://existing-vllm:8000")

    assert options.mode == "remote"
    assert options.model is None
    assert options.vllm_base_url == "http://existing-vllm:8000"


@pytest.mark.parametrize(
    "url",
    [
        "http://user:password@existing-vllm:8000",
        "http://existing-vllm:99999",
        "http://[invalid",
        "http://existing vllm:8000",
    ],
)
def test_serve_rejects_unsafe_or_malformed_remote_base_urls(url: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["serve", "--vllm-base-url", url])

    assert exc_info.value.code == 2
    assert "--vllm-base-url" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "exactly one of --model or --vllm-base-url is required"),
        (
            ["--model", "Qwen/Qwen3-4B", "--vllm-base-url", "http://existing-vllm:8000"],
            "exactly one of --model or --vllm-base-url is required",
        ),
    ],
)
def test_serve_requires_exactly_one_source_option(args: list[str], message: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["serve", *args])

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("model", ["", "   "])
def test_serve_rejects_empty_model(capsys: pytest.CaptureFixture[str], model: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["serve", "--model", model])

    assert exc_info.value.code == 2
    assert "--model must not be empty" in capsys.readouterr().err


def test_serve_preserves_passthrough_after_double_dash() -> None:
    options = parse_serve_args(
        "--model",
        "Qwen/Qwen3-4B",
        "--",
        "--dtype",
        "bfloat16",
        "--max-model-len=32768",
    )

    assert options.vllm_args == ["--dtype", "bfloat16", "--max-model-len=32768"]


def test_remote_serve_rejects_vllm_passthrough_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(
            ["serve", "--vllm-base-url", "http://existing-vllm:8000", "--", "--dtype", "bfloat16"]
        )

    assert exc_info.value.code == 2
    assert "only supported with --model" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "Qwen/Qwen3-4B", "--", "--host", "127.0.0.1"],
        ["--model", "Qwen/Qwen3-4B", "--", "--host=127.0.0.1"],
        ["--model", "Qwen/Qwen3-4B", "--", "--port", "9999"],
        ["--model", "Qwen/Qwen3-4B", "--", "--port=9999"],
        ["--model", "Qwen/Qwen3-4B", "--", "--api-key", "secret"],
        ["--model", "Qwen/Qwen3-4B", "--", "--api-key=secret"],
        ["--model", "Qwen/Qwen3-4B", "--", "--api_key", "secret"],
        ["--model", "Qwen/Qwen3-4B", "--", "--api_key=secret"],
        ["--model", "Qwen/Qwen3-4B", "--", "--uds", "/tmp/vllm.sock"],
        ["--model", "Qwen/Qwen3-4B", "--", "--uds=/tmp/vllm.sock"],
        ["--model", "Qwen/Qwen3-4B", "--", "--por", "9999"],
        ["--model", "Qwen/Qwen3-4B", "--", "--api-ke", "secret"],
        ["--model", "Qwen/Qwen3-4B", "--", "--ud", "/tmp/vllm.sock"],
    ],
)
def test_serve_rejects_launcher_owned_vllm_passthrough_flags(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["serve", *args])

    assert exc_info.value.code == 2
    assert "reserved for the launcher" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--port", "0"),
        ("--port", "65536"),
        ("--vllm-port", "-1"),
        ("--vllm-port", "65536"),
    ],
)
def test_serve_rejects_ports_outside_tcp_range(
    flag: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["serve", "--model", "Qwen/Qwen3-4B", f"{flag}={value}"])

    assert exc_info.value.code == 2
    assert "must be between 1 and 65535" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--startup-timeout-s", "nan"),
        ("--startup-timeout-s", "inf"),
        ("--startup-timeout-s", "-inf"),
        ("--startup-timeout-s", "0"),
        ("--startup-timeout-s", "-1"),
        ("--startup-timeout-s", "86400.1"),
        ("--shutdown-timeout-s", "nan"),
        ("--shutdown-timeout-s", "inf"),
        ("--shutdown-timeout-s", "0"),
        ("--shutdown-timeout-s", "-1"),
        ("--shutdown-timeout-s", "86400.1"),
    ],
)
def test_serve_rejects_non_finite_non_positive_or_excessive_timeouts(
    flag: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["serve", "--model", "Qwen/Qwen3-4B", f"{flag}={value}"])

    assert exc_info.value.code == 2
    assert "must be finite and greater than 0" in capsys.readouterr().err


def test_serve_accepts_port_and_timeout_boundaries() -> None:
    options = parse_serve_args(
        "--model",
        "Qwen/Qwen3-4B",
        "--port",
        "1",
        "--vllm-port",
        "65535",
        "--startup-timeout-s",
        "0.001",
        "--shutdown-timeout-s",
        "86400",
    )

    assert options.port == 1
    assert options.vllm_port == 65535
    assert options.startup_timeout_s == 0.001
    assert options.shutdown_timeout_s == 86400.0


def test_doctor_subcommand_accepts_optional_mode() -> None:
    parser = build_parser()

    namespace = parser.parse_args(["doctor", "--mode", "local"])
    assert namespace.command == "doctor"
    assert namespace.mode == "local"

    namespace = parser.parse_args(["doctor"])
    assert namespace.command == "doctor"
    assert namespace.mode is None


def test_version_subcommand_exits_successfully(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("agentic_api.cli.version_report", lambda: "version report")

    exit_code = main(["version"])

    assert exit_code == 0
    assert capsys.readouterr().out == "version report\n"


def test_top_level_version_flag_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--version"])

    assert exit_code == 0
    assert capsys.readouterr().out == f"agentic-api {__version__}\n"


def test_version_subcommand_reports_missing_binary_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "agentic_api.cli.version_report",
        lambda: (_ for _ in ()).throw(FileNotFoundError("agentic-server not found")),
    )

    exit_code = main(["version"])

    assert exit_code == 1
    assert capsys.readouterr().err == "agentic-server not found\n"


def test_doctor_json_flag_is_forwarded_to_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str | None, bool]] = []
    monkeypatch.setattr("agentic_api.cli.doctor", lambda mode, *, json_output: calls.append((mode, json_output)) or 0)

    assert main(["doctor", "--mode", "remote", "--json"]) == 0
    assert calls == [("remote", True)]


def test_serve_reports_a_missing_packaged_launcher_with_remediation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real_import = builtins.__import__

    def missing_launcher(name: str, *args: object, **kwargs: object) -> object:
        if name == "agentic_api.launcher":
            raise ModuleNotFoundError("agentic_api.launcher", name="agentic_api.launcher")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_launcher)

    exit_code = main(["serve", "--vllm-base-url", "http://existing-vllm:8000"])

    assert exit_code == 1
    assert capsys.readouterr().err == (
        "agentic-api serve is unavailable because the Python launcher is missing; "
        "reinstall agentic-api for this platform.\n"
    )
