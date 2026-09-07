from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as metadata_version
from pathlib import Path

from agentic_api import __version__
from agentic_api.binary import (
    PackagedBinaryNotFoundError,
    PackagedBinaryVersionError,
    find_active_environment_executable,
    find_packaged_binary,
    read_binary_version,
)
from agentic_api.compatibility import SUPPORTED_VLLM_VERSION


@dataclass(frozen=True)
class DoctorReport:
    python_version: str
    platform_summary: str
    package_version: str
    rust_binary_path: str
    rust_binary_executable: bool
    rust_binary_version: str
    supported_vllm_version: str
    installed_vllm_version: str
    vllm_executable_path: str
    local_ok: bool
    remote_ok: bool
    local_message: str
    remote_message: str


def doctor(mode: str | None, *, json_output: bool = False) -> int:
    report = collect_doctor_report()
    print(render_doctor_report(report, mode, json_output=json_output))

    if mode == "local":
        return 0 if report.local_ok else 1
    if mode == "remote":
        return 0 if report.remote_ok else 1
    return 0 if report.remote_ok else 1


def collect_doctor_report() -> DoctorReport:
    rust_binary_path, rust_binary_executable, rust_binary_version = _rust_binary_details()
    package_version = _package_version("agentic-api")
    installed_vllm_version = _package_version("vllm")
    try:
        vllm_executable_path = str(find_active_environment_executable("vllm"))
    except FileNotFoundError:
        vllm_executable_path = "not found"
    local_runtime_ok, local_message = _local_health(
        installed_vllm_version,
        vllm_executable_path,
        platform_name=platform.system(),
    )
    remote_ok = rust_binary_executable and not rust_binary_version.startswith("error:")
    remote_message = _remote_health_message(rust_binary_executable, rust_binary_version)
    if local_runtime_ok and not remote_ok:
        local_message = remote_message

    return DoctorReport(
        python_version=platform.python_version(),
        platform_summary=f"{platform.system()} {platform.machine()}",
        package_version=package_version,
        rust_binary_path=rust_binary_path,
        rust_binary_executable=rust_binary_executable,
        rust_binary_version=rust_binary_version,
        supported_vllm_version=SUPPORTED_VLLM_VERSION,
        installed_vllm_version=installed_vllm_version,
        vllm_executable_path=vllm_executable_path,
        local_ok=local_runtime_ok and remote_ok,
        remote_ok=remote_ok,
        local_message=local_message,
        remote_message=remote_message,
    )


def render_doctor_report(report: DoctorReport, mode: str | None, *, json_output: bool = False) -> str:
    if json_output:
        payload = asdict(report)
        payload["selected_mode"] = mode or "all"
        return json.dumps(payload, sort_keys=True)

    local_health = "ok" if report.local_ok else "unavailable"
    remote_health = "ok" if report.remote_ok else "unavailable"

    lines = [
        f"Selected mode: {mode or 'all'}",
        f"Python version: {report.python_version}",
        f"Platform: {report.platform_summary}",
        f"agentic-api version: {report.package_version}",
        f"Rust binary path: {report.rust_binary_path}",
        f"Rust binary executable: {'yes' if report.rust_binary_executable else 'no'}",
        f"Rust binary version: {report.rust_binary_version}",
        f"Supported vLLM version: {report.supported_vllm_version}",
        f"Installed vLLM version: {report.installed_vllm_version}",
        f"vLLM executable path: {report.vllm_executable_path}",
        f"Local mode health: {local_health}",
        f"Local mode details: {report.local_message}",
        f"Remote mode health: {remote_health}",
        f"Remote mode details: {report.remote_message}",
    ]
    return "\n".join(lines)


def _package_version(name: str) -> str:
    try:
        return metadata_version(name)
    except PackageNotFoundError:
        return "not installed"


def _rust_binary_details() -> tuple[str, bool, str]:
    try:
        path = find_packaged_binary("agentic-server")
    except PackagedBinaryNotFoundError as error:
        return ("not found", False, f"error: {error}")

    executable = path.is_file() and os.access(path, os.X_OK)
    try:
        version = read_binary_version(path)
    except PackagedBinaryVersionError as error:
        version = f"error: {error}"
    return (str(path), executable, version)


def _local_health(
    installed_vllm_version: str,
    vllm_executable_path: str,
    *,
    platform_name: str,
) -> tuple[bool, str]:
    if platform_name != "Linux":
        return (
            False,
            "Local mode is currently supported only on Linux because the [local] extra installs vLLM only on Linux. "
            "Use remote mode on this platform, or run local mode on Linux.",
        )

    install_hint = (
        "Install the agentic-api wheel artifact with its local extra: "
        '`uv pip install "agentic-api[local] @ file:///path/to/agentic_api-PLATFORM.whl"`.'
    )
    if installed_vllm_version == "not installed":
        return (False, install_hint)
    if installed_vllm_version != SUPPORTED_VLLM_VERSION:
        return (
            False,
            f"Installed vLLM does not match the tested version {SUPPORTED_VLLM_VERSION}. {install_hint}",
        )
    if vllm_executable_path == "not found":
        return (
            False,
            f"The installed vLLM package does not provide a `vllm` executable in the active Python environment. "
            f"{install_hint}",
        )
    return (True, "The tested local vLLM package and executable are available.")


def _remote_health_message(rust_binary_executable: bool, rust_binary_version: str) -> str:
    if not rust_binary_executable:
        return "The packaged Rust gateway executable is missing or not executable."
    if rust_binary_version.startswith("error:"):
        return "The packaged Rust gateway executable could not report its version."
    return "The packaged Rust gateway executable is available; remote mode does not require vLLM."
