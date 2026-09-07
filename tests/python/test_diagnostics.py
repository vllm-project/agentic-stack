from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from agentic_api import diagnostics
from agentic_api.compatibility import SUPPORTED_VLLM_VERSION


@pytest.fixture(autouse=True)
def default_to_linux_for_local_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentic_api.diagnostics.platform.system", lambda: "Linux")


def test_remote_doctor_is_healthy_without_vllm(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rust_binary = tmp_path / "agentic-server"
    rust_binary.write_text("")
    rust_binary.chmod(0o755)

    monkeypatch.setattr("agentic_api.diagnostics.find_packaged_binary", lambda name: rust_binary)
    monkeypatch.setattr("agentic_api.diagnostics.read_binary_version", lambda path: "agentic-server 0.4.0")
    monkeypatch.setattr("agentic_api.diagnostics.metadata_version", _metadata_version_without_vllm)
    monkeypatch.setattr(
        "agentic_api.diagnostics.find_active_environment_executable",
        lambda name: (_ for _ in ()).throw(FileNotFoundError(name)),
    )

    exit_code = diagnostics.doctor("remote")
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Selected mode: remote" in output
    assert "Installed vLLM version: not installed" in output
    assert "Local mode health: unavailable" in output
    assert "Remote mode health: ok" in output


def test_doctor_without_mode_succeeds_for_a_healthy_base_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rust_binary = tmp_path / "agentic-server"
    rust_binary.write_text("")
    rust_binary.chmod(0o755)

    monkeypatch.setattr("agentic_api.diagnostics.find_packaged_binary", lambda name: rust_binary)
    monkeypatch.setattr("agentic_api.diagnostics.read_binary_version", lambda path: "agentic-server 0.4.0")
    monkeypatch.setattr("agentic_api.diagnostics.metadata_version", _metadata_version_without_vllm)
    monkeypatch.setattr(
        "agentic_api.diagnostics.find_active_environment_executable",
        lambda name: (_ for _ in ()).throw(FileNotFoundError(name)),
    )

    exit_code = diagnostics.doctor(None)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Selected mode: all" in output
    assert "Remote mode health: ok" in output


def test_local_doctor_reports_missing_vllm_installation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rust_binary = tmp_path / "agentic-server"
    rust_binary.write_text("")
    rust_binary.chmod(0o755)

    monkeypatch.setattr("agentic_api.diagnostics.find_packaged_binary", lambda name: rust_binary)
    monkeypatch.setattr("agentic_api.diagnostics.read_binary_version", lambda path: "agentic-server 0.4.0")
    monkeypatch.setattr("agentic_api.diagnostics.metadata_version", _metadata_version_without_vllm)
    monkeypatch.setattr(
        "agentic_api.diagnostics.find_active_environment_executable",
        lambda name: (_ for _ in ()).throw(FileNotFoundError(name)),
    )
    monkeypatch.setattr("agentic_api.diagnostics.platform.system", lambda: "Linux")

    exit_code = diagnostics.doctor("local")
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Selected mode: local" in output
    assert "Local mode health: unavailable" in output
    assert "Install the agentic-api wheel artifact with its local extra" in output
    assert "file:///path/to/agentic_api-PLATFORM.whl" in output


@pytest.mark.parametrize(
    ("rust_binary_details", "expected_message"),
    [
        (
            ("not found", False, "error: agentic-server not found"),
            "The packaged Rust gateway executable is missing or not executable.",
        ),
        (
            ("/venv/bin/agentic-server", True, "error: version probe failed"),
            "The packaged Rust gateway executable could not report its version.",
        ),
    ],
)
def test_local_doctor_requires_a_healthy_packaged_gateway(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    rust_binary_details: tuple[str, bool, str],
    expected_message: str,
) -> None:
    vllm_binary = tmp_path / "vllm"
    vllm_binary.write_text("")
    vllm_binary.chmod(0o755)

    monkeypatch.setattr("agentic_api.diagnostics._rust_binary_details", lambda: rust_binary_details)
    monkeypatch.setattr("agentic_api.diagnostics.metadata_version", _metadata_version_with_supported_vllm)
    monkeypatch.setattr("agentic_api.diagnostics.find_active_environment_executable", lambda name: vllm_binary)

    exit_code = diagnostics.doctor("local")
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Local mode health: unavailable" in output
    assert f"Local mode details: {expected_message}" in output


def test_local_doctor_explains_linux_only_extra_on_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rust_binary = tmp_path / "agentic-server"
    rust_binary.write_text("")
    rust_binary.chmod(0o755)

    monkeypatch.setattr("agentic_api.diagnostics.find_packaged_binary", lambda name: rust_binary)
    monkeypatch.setattr("agentic_api.diagnostics.read_binary_version", lambda path: "agentic-server 0.5.0")
    monkeypatch.setattr("agentic_api.diagnostics.metadata_version", _metadata_version_without_vllm)
    monkeypatch.setattr(
        "agentic_api.diagnostics.find_active_environment_executable",
        lambda name: (_ for _ in ()).throw(FileNotFoundError(name)),
    )
    monkeypatch.setattr("agentic_api.diagnostics.platform.system", lambda: "Darwin")

    exit_code = diagnostics.doctor("local")
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Local mode health: unavailable" in output
    assert "Local mode is currently supported only on Linux" in output
    assert "remote mode" in output


def test_local_doctor_rejects_supported_vllm_on_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rust_binary = tmp_path / "agentic-server"
    rust_binary.write_text("")
    rust_binary.chmod(0o755)
    vllm_binary = tmp_path / "vllm"
    vllm_binary.write_text("")
    vllm_binary.chmod(0o755)

    monkeypatch.setattr("agentic_api.diagnostics.find_packaged_binary", lambda name: rust_binary)
    monkeypatch.setattr("agentic_api.diagnostics.read_binary_version", lambda path: "agentic-server 0.5.0")
    monkeypatch.setattr("agentic_api.diagnostics.metadata_version", _metadata_version_with_supported_vllm)
    monkeypatch.setattr("agentic_api.diagnostics.find_active_environment_executable", lambda name: vllm_binary)
    monkeypatch.setattr("agentic_api.diagnostics.platform.system", lambda: "Darwin")

    assert diagnostics.doctor("local") == 1
    output = capsys.readouterr().out
    assert "Local mode health: unavailable" in output
    assert "Local mode is currently supported only on Linux" in output


def test_local_doctor_reports_incompatible_vllm_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rust_binary = tmp_path / "agentic-server"
    rust_binary.write_text("")
    rust_binary.chmod(0o755)
    vllm_binary = tmp_path / "vllm"
    vllm_binary.write_text("")
    vllm_binary.chmod(0o755)

    monkeypatch.setattr("agentic_api.diagnostics.find_packaged_binary", lambda name: rust_binary)
    monkeypatch.setattr("agentic_api.diagnostics.read_binary_version", lambda path: "agentic-server 0.4.0")
    monkeypatch.setattr("agentic_api.diagnostics.metadata_version", _metadata_version_with_incompatible_vllm)
    monkeypatch.setattr("agentic_api.diagnostics.find_active_environment_executable", lambda name: vllm_binary)

    exit_code = diagnostics.doctor("local")
    output = capsys.readouterr().out

    assert exit_code == 1
    assert f"Supported vLLM version: {SUPPORTED_VLLM_VERSION}" in output
    assert "Installed vLLM version: 0.12.0" in output
    assert "Local mode health: unavailable" in output
    assert "Installed vLLM does not match the tested version" in output


def test_local_doctor_finds_vllm_in_active_environment_when_it_is_not_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rust_binary = tmp_path / "agentic-server"
    rust_binary.write_text("")
    rust_binary.chmod(0o755)
    scripts_dir = tmp_path / "environment" / "bin"
    scripts_dir.mkdir(parents=True)
    vllm_binary = scripts_dir / "vllm"
    vllm_binary.write_text("")
    vllm_binary.chmod(0o755)

    monkeypatch.setattr("agentic_api.diagnostics.find_packaged_binary", lambda name: rust_binary)
    monkeypatch.setattr("agentic_api.diagnostics.read_binary_version", lambda path: "agentic-server 0.4.0")
    monkeypatch.setattr("agentic_api.diagnostics.metadata_version", _metadata_version_with_supported_vllm)
    monkeypatch.setattr("agentic_api.binary.sysconfig.get_path", lambda name: str(scripts_dir))
    monkeypatch.setattr("agentic_api.binary.sys.executable", str(scripts_dir / "python"))
    monkeypatch.setenv("PATH", "")

    report = diagnostics.collect_doctor_report()

    assert report.local_ok is True
    assert report.vllm_executable_path == str(vllm_binary)


def test_doctor_report_includes_platform_package_and_binary_details(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rust_binary = tmp_path / "agentic-server"
    rust_binary.write_text("")
    rust_binary.chmod(0o755)
    vllm_binary = tmp_path / "vllm"
    vllm_binary.write_text("")
    vllm_binary.chmod(0o755)

    monkeypatch.setattr("agentic_api.diagnostics.find_packaged_binary", lambda name: rust_binary)
    monkeypatch.setattr("agentic_api.diagnostics.read_binary_version", lambda path: "agentic-server 0.4.0")
    monkeypatch.setattr("agentic_api.diagnostics.metadata_version", _metadata_version_with_supported_vllm)
    monkeypatch.setattr("agentic_api.diagnostics.find_active_environment_executable", lambda name: vllm_binary)
    monkeypatch.setattr("agentic_api.diagnostics.platform.system", lambda: "Linux")
    monkeypatch.setattr("agentic_api.diagnostics.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("agentic_api.diagnostics.platform.python_version", lambda: "3.12.4")

    exit_code = diagnostics.doctor(None)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Python version: 3.12.4" in output
    assert "Platform: Linux x86_64" in output
    assert "agentic-api version: 0.4.0" in output
    assert f"Supported vLLM version: {SUPPORTED_VLLM_VERSION}" in output
    assert f"Rust binary path: {rust_binary}" in output
    assert "Rust binary executable: yes" in output
    assert "Rust binary version: agentic-server 0.4.0" in output
    assert f"vLLM executable path: {vllm_binary}" in output


def test_python_module_entrypoint_delegates_to_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("agentic_api.__main__")
    called: list[object] = []

    monkeypatch.setattr("agentic_api.cli.main", lambda argv=None: called.append(argv) or 7)

    assert module.main() == 7
    assert called == [None]


def test_doctor_json_is_machine_readable_and_preserves_exit_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rust_binary = tmp_path / "agentic-server"
    rust_binary.write_text("")
    rust_binary.chmod(0o755)

    monkeypatch.setattr("agentic_api.diagnostics.find_packaged_binary", lambda name: rust_binary)
    monkeypatch.setattr("agentic_api.diagnostics.read_binary_version", lambda path: "agentic-server 0.4.0")
    monkeypatch.setattr("agentic_api.diagnostics.metadata_version", _metadata_version_without_vllm)
    monkeypatch.setattr(
        "agentic_api.diagnostics.find_active_environment_executable",
        lambda name: (_ for _ in ()).throw(FileNotFoundError(name)),
    )

    exit_code = diagnostics.doctor("remote", json_output=True)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["selected_mode"] == "remote"
    assert payload["remote_ok"] is True
    assert payload["local_ok"] is False
    assert payload["package_version"] == "0.4.0"


def _metadata_version_without_vllm(name: str) -> str:
    if name == "agentic-api":
        return "0.4.0"
    raise diagnostics.PackageNotFoundError(name)


def _metadata_version_with_incompatible_vllm(name: str) -> str:
    if name == "agentic-api":
        return "0.4.0"
    if name == "vllm":
        return "0.12.0"
    raise diagnostics.PackageNotFoundError(name)


def _metadata_version_with_supported_vllm(name: str) -> str:
    if name == "agentic-api":
        return "0.4.0"
    if name == "vllm":
        return SUPPORTED_VLLM_VERSION
    raise diagnostics.PackageNotFoundError(name)
