#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: scripts/check-python-wheel.sh <wheel-path>" >&2
  exit 1
fi

wheel_path="$1"
check_python="${AGENTIC_API_CHECK_PYTHON:-python}"
expected_wheel_tag="${AGENTIC_API_EXPECTED_WHEEL_TAG:-}"
scripts_dir="${AGENTIC_API_CHECK_SCRIPTS_DIR:-}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/.." && pwd -P)"
cargo_manifest_path="$repo_root/Cargo.toml"

if [ -n "${AGENTIC_API_EXPECTED_VERSION:-}" ]; then
  expected_version="$AGENTIC_API_EXPECTED_VERSION"
else
  cargo_metadata="$(cargo metadata --format-version 1 --no-deps --manifest-path "$cargo_manifest_path")"
  expected_version="$(printf '%s\n' "$cargo_metadata" | "$check_python" -c '
import json
import sys

packages = json.load(sys.stdin)["packages"]
print(next(package["version"] for package in packages if package["name"] == "agentic-server"))
')"
fi

if [ ! -f "$wheel_path" ]; then
  echo "wheel file not found: $wheel_path" >&2
  exit 1
fi

"$check_python" - "$wheel_path" "$expected_version" "$scripts_dir" "$cargo_manifest_path" "$expected_wheel_tag" <<'PY'
from __future__ import annotations

import importlib
import json
import os
import stat
import subprocess
import sys
import sysconfig
import zipfile
from email.parser import Parser
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


FORBIDDEN_SUBSTRINGS = (
    "/vllm/",
    "/torch/",
    "libtorch",
    "/transformers/",
    "/nvidia/",
    "/nvidia/cublas/",
    "libcublas.so",
    "libcublaslt.so",
    "libcuda.so",
    "libcudart.so",
    "libcufft.so",
    "libcurand.so",
    "libcusolver.so",
    "libcusparse.so",
    "libnccl.so",
    "libnvrtc",
    "cuda",
    "cudnn",
    "rocm",
    "/hip/",
    "libamdhip64.so",
    "libhipblas.so",
    "librocblas.so",
    "libhsa-runtime64.so",
    "librocm",
)


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def normalize(path: str) -> str:
    return path.replace("\\", "/").lower()


wheel_path = Path(sys.argv[1])
expected_version = sys.argv[2]
scripts_dir_override = sys.argv[3]
cargo_manifest_path = Path(sys.argv[4])
expected_wheel_tag = sys.argv[5]
cargo_metadata_path_override = os.environ.get("AGENTIC_API_CHECK_CARGO_METADATA_JSON")

if expected_wheel_tag:
    expected_wheel_name = f"agentic_api-{expected_version}-{expected_wheel_tag}.whl"
    ensure(
        wheel_path.name == expected_wheel_name,
        f"wheel tag must be exactly {expected_wheel_tag}; found {wheel_path.name}",
    )


def load_cargo_metadata() -> dict[str, object]:
    if cargo_metadata_path_override:
        try:
            return json.loads(Path(cargo_metadata_path_override).read_text(encoding="utf-8"))
        except OSError as error:
            fail(f"unable to read cargo metadata fixture {cargo_metadata_path_override}: {error}")
        except json.JSONDecodeError as error:
            fail(f"invalid cargo metadata fixture {cargo_metadata_path_override}: {error}")

    try:
        completed = subprocess.run(
            [
                "cargo",
                "metadata",
                "--format-version",
                "1",
                "--no-deps",
                "--manifest-path",
                str(cargo_manifest_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        fail(
            "cargo metadata check: unable to launch "
            f"cargo metadata --format-version 1 --no-deps --manifest-path {cargo_manifest_path}: {error}"
        )
    except subprocess.CalledProcessError as error:
        output = (error.stdout or error.stderr or "").strip()
        if output:
            fail(f"cargo metadata check failed: {output}")
        fail(f"cargo metadata check exited with status {error.returncode}")

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        fail(f"cargo metadata produced invalid JSON: {error}")


def check_workspace_package_versions() -> None:
    metadata = load_cargo_metadata()
    packages = metadata.get("packages")
    workspace_members = metadata.get("workspace_members")

    ensure(isinstance(packages, list), "cargo metadata packages must be a list")
    ensure(isinstance(workspace_members, list), "cargo metadata workspace_members must be a list")

    workspace_member_ids = {member for member in workspace_members if isinstance(member, str)}
    workspace_packages = [
        package
        for package in packages
        if isinstance(package, dict) and package.get("id") in workspace_member_ids
    ]
    ensure(workspace_packages, "cargo metadata returned no workspace packages")

    for package in workspace_packages:
        name = package.get("name")
        version = package.get("version")
        ensure(isinstance(name, str), f"workspace package has invalid name: {package!r}")
        ensure(isinstance(version, str), f"workspace package {name!r} has invalid version: {package!r}")
        ensure(
            version == expected_version,
            f"workspace package {name} version must be {expected_version}; found {version!r}",
        )


check_workspace_package_versions()

with zipfile.ZipFile(wheel_path) as archive:
    names = archive.namelist()
    normalized_names = [normalize(name) for name in names]

    ensure(
        any(name == "agentic_api/__init__.py" or name.startswith("agentic_api/") for name in normalized_names),
        "wheel missing agentic_api package",
    )

    metadata_path = next((name for name in names if name.endswith(".dist-info/METADATA")), None)
    ensure(metadata_path is not None, "wheel missing dist-info metadata")
    metadata = Parser().parsestr(archive.read(metadata_path).decode("utf-8"))
    ensure(metadata.get("Name") == "agentic-api", "wheel metadata Name must be agentic-api")
    ensure(
        metadata.get("Version") == expected_version,
        f"wheel metadata Version must be {expected_version}; found {metadata.get('Version')!r}",
    )

    entry_points_text = ""
    entry_points_path = next((name for name in names if name.endswith(".dist-info/entry_points.txt")), None)
    if entry_points_path is not None:
        entry_points_text = archive.read(entry_points_path).decode("utf-8")

    def has_packaged_script(script_name: str) -> bool:
        return any(
            name.endswith(f".data/scripts/{script_name}")
            for name in normalized_names
        )

    has_agentic_api_console = has_packaged_script("agentic-api") or (
        "agentic-api" in entry_points_text
    )
    ensure(has_agentic_api_console, "wheel missing agentic-api console script")

    for binary_name in ("agentic", "agentic-server"):
        ensure(
            has_packaged_script(binary_name),
            f"wheel missing packaged executable: {binary_name}",
        )

    forbidden_entry = next(
        (
            original_name
            for original_name, lowered_name in zip(names, normalized_names, strict=True)
            if any(marker in f"/{lowered_name}" for marker in FORBIDDEN_SUBSTRINGS)
        ),
        None,
    )
    ensure(forbidden_entry is None, f"forbidden wheel payload: {forbidden_entry}")

try:
    installed_distribution = distribution("agentic-api")
except PackageNotFoundError as error:
    fail(f"installed distribution not found: {error}")

ensure(
    installed_distribution.version == expected_version,
    f"installed distribution version must be {expected_version}; found {installed_distribution.version!r}",
)
ensure(
    installed_distribution.metadata.get("Version") == expected_version,
    (
        "installed distribution metadata Version must be "
        f"{expected_version}; found {installed_distribution.metadata.get('Version')!r}"
    ),
)

agentic_api = importlib.import_module("agentic_api")
installed_module_version = getattr(agentic_api, "__version__", None)
ensure(
    installed_module_version == expected_version,
    f"agentic_api.__version__ must be {expected_version}; found {installed_module_version!r}",
)

scripts_dir = Path(scripts_dir_override) if scripts_dir_override else Path(sysconfig.get_path("scripts") or "")
ensure(scripts_dir.is_dir(), f"scripts directory not found: {scripts_dir}")

for command_name in ("agentic-api", "agentic", "agentic-server"):
    command_path = scripts_dir / command_name
    ensure(command_path.is_file(), f"installed executable not found: {command_path}")
    ensure(os.access(command_path, os.X_OK), f"installed executable is not executable: {command_path}")


def run_command(command: list[str], failure_prefix: str) -> str:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except OSError as error:
        fail(f"{failure_prefix}: unable to launch {' '.join(command)}: {error}")
    except subprocess.CalledProcessError as error:
        output = (error.stdout or error.stderr or "").strip()
        if output:
            fail(f"{failure_prefix}: {' '.join(command)} failed: {output}")
        fail(f"{failure_prefix}: {' '.join(command)} exited with status {error.returncode}")

    output = (completed.stdout or completed.stderr).strip()
    ensure(output, f"{failure_prefix}: {' '.join(command)} produced no output")
    return output


agentic_version_output = run_command([str(scripts_dir / "agentic"), "--version"], "agentic version check")
ensure(
    expected_version in agentic_version_output.splitlines()[0],
    f"agentic version output missing {expected_version}: {agentic_version_output}",
)

server_version_output = run_command([str(scripts_dir / "agentic-server"), "--version"], "agentic-server version check")
ensure(
    f"agentic-server {expected_version}" in server_version_output.splitlines()[0],
    f"agentic-server version output missing {expected_version}: {server_version_output}",
)

launcher_version_output = run_command([str(scripts_dir / "agentic-api"), "version"], "agentic-api version check")
ensure(
    f"agentic-api version: {expected_version}" in launcher_version_output,
    f"agentic-api version output missing {expected_version}: {launcher_version_output}",
)
ensure(
    f"Rust binary version: agentic-server {expected_version}" in launcher_version_output,
    (
        "agentic-api version output missing packaged Rust binary version "
        f"{expected_version}: {launcher_version_output}"
    ),
)

print(f"wheel validation passed: {wheel_path.name}")
PY
