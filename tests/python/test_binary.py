from __future__ import annotations

from pathlib import Path

import pytest

from agentic_api.binary import (
    PackagedBinaryNotFoundError,
    find_active_environment_executable,
    find_packaged_binary,
    read_binary_version,
)


def test_find_packaged_binary_prefers_scripts_directory_over_global_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripts_dir = tmp_path / "env" / "bin"
    scripts_dir.mkdir(parents=True)
    local_binary = scripts_dir / "agentic-server"
    local_binary.write_text("#!/bin/sh\nexit 0\n")
    local_binary.chmod(0o755)

    global_binary = tmp_path / "global" / "agentic-server"
    global_binary.parent.mkdir(parents=True)
    global_binary.write_text("#!/bin/sh\nexit 0\n")
    global_binary.chmod(0o755)

    monkeypatch.setattr("agentic_api.binary.sysconfig.get_path", lambda name: str(scripts_dir))
    monkeypatch.setattr("agentic_api.binary.sys.executable", str(tmp_path / "env" / "bin" / "python"))
    monkeypatch.setattr("agentic_api.binary.shutil.which", lambda name: str(global_binary))

    assert find_packaged_binary("agentic-server") == local_binary


@pytest.mark.parametrize("path_exists", [False, True])
def test_find_packaged_binary_reports_remediation_when_packaged_binary_is_missing_or_not_executable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path_exists: bool
) -> None:
    scripts_dir = tmp_path / "env" / "bin"
    scripts_dir.mkdir(parents=True)
    local_binary = scripts_dir / "agentic-server"
    if path_exists:
        local_binary.write_text("#!/bin/sh\nexit 0\n")
        local_binary.chmod(0o644)

    monkeypatch.setattr("agentic_api.binary.sysconfig.get_path", lambda name: str(scripts_dir))
    monkeypatch.setattr("agentic_api.binary.sys.executable", str(tmp_path / "env" / "bin" / "python"))
    monkeypatch.setattr("agentic_api.binary.shutil.which", lambda name: None)

    with pytest.raises(PackagedBinaryNotFoundError, match="Reinstall agentic-api for this platform"):
        find_packaged_binary("agentic-server")


def test_find_packaged_binary_does_not_use_an_ambient_path_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripts_dir = tmp_path / "env" / "bin"
    scripts_dir.mkdir(parents=True)
    ambient_binary = tmp_path / "unrelated" / "agentic-server"
    ambient_binary.parent.mkdir(parents=True)
    ambient_binary.write_text("#!/bin/sh\nexit 0\n")
    ambient_binary.chmod(0o755)

    monkeypatch.setattr("agentic_api.binary.sysconfig.get_path", lambda name: str(scripts_dir))
    monkeypatch.setattr("agentic_api.binary.sys.executable", str(tmp_path / "env" / "bin" / "python"))
    monkeypatch.setattr("agentic_api.binary.shutil.which", lambda name: str(ambient_binary))

    with pytest.raises(PackagedBinaryNotFoundError):
        find_packaged_binary("agentic-server")


def test_find_packaged_binary_finds_sibling_of_console_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripts_dir = tmp_path / "prefix" / "bin"
    scripts_dir.mkdir(parents=True)
    packaged_binary = scripts_dir / "agentic-server"
    packaged_binary.write_text("#!/bin/sh\nexit 0\n")
    packaged_binary.chmod(0o755)
    host_binary = tmp_path / "host" / "bin" / "agentic-server"
    host_binary.parent.mkdir(parents=True)
    host_binary.write_text("#!/bin/sh\nexit 0\n")
    host_binary.chmod(0o755)
    console_script = scripts_dir / "agentic-api"
    console_script.write_text("#!/bin/sh\nexit 0\n")
    console_script.chmod(0o755)

    monkeypatch.setattr("agentic_api.binary.sysconfig.get_path", lambda name: str(tmp_path / "host" / "bin"))
    monkeypatch.setattr("agentic_api.binary.sys.executable", str(tmp_path / "host" / "bin" / "python"))
    monkeypatch.setattr("agentic_api.binary.sys.argv", [str(console_script)])
    monkeypatch.setattr("agentic_api.binary.shutil.which", lambda name: None)

    assert find_packaged_binary("agentic-server") == packaged_binary


def test_find_active_environment_executable_does_not_use_ambient_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripts_dir = tmp_path / "env" / "bin"
    scripts_dir.mkdir(parents=True)
    ambient_binary = tmp_path / "unrelated" / "vllm"
    ambient_binary.parent.mkdir(parents=True)
    ambient_binary.write_text("#!/bin/sh\nexit 0\n")
    ambient_binary.chmod(0o755)

    monkeypatch.setattr("agentic_api.binary.sysconfig.get_path", lambda name: str(scripts_dir))
    monkeypatch.setattr("agentic_api.binary.sys.executable", str(tmp_path / "env" / "bin" / "python"))
    monkeypatch.setattr("agentic_api.binary.shutil.which", lambda name: str(ambient_binary))

    with pytest.raises(FileNotFoundError, match="active environment"):
        find_active_environment_executable("vllm")


def test_read_binary_version_returns_first_line_from_version_output(tmp_path: Path) -> None:
    binary = tmp_path / "agentic-server"
    binary.write_text("#!/bin/sh\nprintf 'agentic-server 0.4.0\\nextra detail\\n'\n")
    binary.chmod(0o755)

    assert read_binary_version(binary) == "agentic-server 0.4.0"


def test_read_binary_version_reports_a_hung_binary_without_waiting_forever(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "agentic-server"
    binary.write_text("#!/bin/sh\nsleep 1\n")
    binary.chmod(0o755)
    monkeypatch.setattr("agentic_api.binary.BINARY_VERSION_TIMEOUT_S", 0.01)

    with pytest.raises(RuntimeError, match="timed out while reporting its version"):
        read_binary_version(binary)
