from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import textwrap
import zipfile
from collections.abc import Mapping
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check-python-wheel.sh"
EXPECTED_VERSION = "0.4.0"
WORKSPACE_VERSION = re.search(
    r'(?ms)^\[workspace\.package\].*?^version\s*=\s*"([^"]+)"',
    (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8"),
).group(1)


def test_check_python_wheel_accepts_expected_wheel_and_installed_environment(tmp_path: Path) -> None:
    wheel_path = _write_fake_wheel(tmp_path / "agentic_api-0.4.0-py3-none-any.whl")
    cargo_metadata_path = _write_fake_cargo_metadata(tmp_path / "cargo-metadata.json")
    site_packages = _write_fake_site_packages(tmp_path / "site-packages")
    scripts_dir = _write_fake_scripts(tmp_path / "bin")

    result = _run_check_script(wheel_path, site_packages, scripts_dir, cargo_metadata_path=cargo_metadata_path)

    assert result.returncode == 0, result.stderr
    assert "wheel validation passed" in result.stdout


def test_check_python_wheel_derives_version_for_bare_invocation(tmp_path: Path) -> None:
    wheel_path = _write_fake_wheel(
        tmp_path / f"agentic_api-{WORKSPACE_VERSION}-py3-none-any.whl",
        version=WORKSPACE_VERSION,
    )
    cargo_metadata_path = _write_fake_cargo_metadata(
        tmp_path / "cargo-metadata.json",
        package_versions={name: WORKSPACE_VERSION for name in ("agentic-praxis", "agentic-server-core", "agentic-server")},
    )
    site_packages = _write_fake_site_packages(tmp_path / "site-packages", version=WORKSPACE_VERSION)
    scripts_dir = _write_fake_scripts(tmp_path / "bin", version=WORKSPACE_VERSION)

    result = _run_check_script(
        wheel_path,
        site_packages,
        scripts_dir,
        cargo_metadata_path=cargo_metadata_path,
        expected_version=None,
    )

    assert result.returncode == 0, result.stderr


def test_check_python_wheel_rejects_vllm_payloads(tmp_path: Path) -> None:
    wheel_path = _write_fake_wheel(
        tmp_path / "agentic_api-0.4.0-py3-none-any.whl",
        extra_entries={"agentic_api-0.4.0.data/purelib/vllm/__init__.py": "# forbidden\n"},
    )
    cargo_metadata_path = _write_fake_cargo_metadata(tmp_path / "cargo-metadata.json")
    site_packages = _write_fake_site_packages(tmp_path / "site-packages")
    scripts_dir = _write_fake_scripts(tmp_path / "bin")

    result = _run_check_script(wheel_path, site_packages, scripts_dir, cargo_metadata_path=cargo_metadata_path)

    assert result.returncode != 0
    assert "forbidden wheel payload" in result.stderr
    assert "vllm/__init__.py" in result.stderr


def test_check_python_wheel_rejects_nvidia_payloads(tmp_path: Path) -> None:
    wheel_path = _write_fake_wheel(
        tmp_path / "agentic_api-0.4.0-py3-none-any.whl",
        extra_entries={
            "agentic_api-0.4.0.data/purelib/nvidia/cublas/__init__.py": "# forbidden\n",
            "agentic_api.libs/libcublas.so.12": "",
            "agentic_api.libs/libnccl.so.2": "",
        },
    )
    cargo_metadata_path = _write_fake_cargo_metadata(tmp_path / "cargo-metadata.json")
    site_packages = _write_fake_site_packages(tmp_path / "site-packages")
    scripts_dir = _write_fake_scripts(tmp_path / "bin")

    result = _run_check_script(wheel_path, site_packages, scripts_dir, cargo_metadata_path=cargo_metadata_path)

    assert result.returncode != 0
    assert "forbidden wheel payload" in result.stderr
    assert (
        "nvidia/cublas/__init__.py" in result.stderr
        or "libcublas.so.12" in result.stderr
        or "libnccl.so.2" in result.stderr
    )


def test_check_python_wheel_rejects_amd_payloads(tmp_path: Path) -> None:
    wheel_path = _write_fake_wheel(
        tmp_path / "agentic_api-0.4.0-py3-none-any.whl",
        extra_entries={"agentic_api.libs/libamdhip64.so": ""},
    )
    cargo_metadata_path = _write_fake_cargo_metadata(tmp_path / "cargo-metadata.json")
    site_packages = _write_fake_site_packages(tmp_path / "site-packages")
    scripts_dir = _write_fake_scripts(tmp_path / "bin")

    result = _run_check_script(wheel_path, site_packages, scripts_dir, cargo_metadata_path=cargo_metadata_path)

    assert result.returncode != 0
    assert "forbidden wheel payload" in result.stderr
    assert "libamdhip64.so" in result.stderr


def test_check_python_wheel_rejects_workspace_version_mismatch(tmp_path: Path) -> None:
    wheel_path = _write_fake_wheel(tmp_path / "agentic_api-0.4.0-py3-none-any.whl")
    cargo_metadata_path = _write_fake_cargo_metadata(
        tmp_path / "cargo-metadata.json",
        package_versions={"agentic-server-core": "0.3.0"},
    )
    site_packages = _write_fake_site_packages(tmp_path / "site-packages")
    scripts_dir = _write_fake_scripts(tmp_path / "bin")

    result = _run_check_script(wheel_path, site_packages, scripts_dir, cargo_metadata_path=cargo_metadata_path)

    assert result.returncode != 0
    assert "workspace package agentic-server-core version must be 0.4.0" in result.stderr


def test_check_python_wheel_rejects_an_unexpected_platform_tag(tmp_path: Path) -> None:
    wheel_path = _write_fake_wheel(tmp_path / "agentic_api-0.4.0-py3-none-linux_x86_64.whl")
    cargo_metadata_path = _write_fake_cargo_metadata(tmp_path / "cargo-metadata.json")
    site_packages = _write_fake_site_packages(tmp_path / "site-packages")
    scripts_dir = _write_fake_scripts(tmp_path / "bin")

    result = _run_check_script(
        wheel_path,
        site_packages,
        scripts_dir,
        cargo_metadata_path=cargo_metadata_path,
        expected_wheel_tag="py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64",
    )

    assert result.returncode != 0
    assert "wheel tag must be exactly" in result.stderr


def test_check_python_wheel_validates_workspace_versions_outside_repository_cwd(tmp_path: Path) -> None:
    wheel_path = _write_fake_wheel(
        tmp_path / f"agentic_api-{WORKSPACE_VERSION}-py3-none-any.whl",
        version=WORKSPACE_VERSION,
    )
    site_packages = _write_fake_site_packages(tmp_path / "site-packages", version=WORKSPACE_VERSION)
    scripts_dir = _write_fake_scripts(tmp_path / "bin", version=WORKSPACE_VERSION)
    outside_repo = tmp_path / "outside-repo"
    outside_repo.mkdir()

    result = _run_check_script(
        wheel_path,
        site_packages,
        scripts_dir,
        cwd=outside_repo,
        expected_version=WORKSPACE_VERSION,
    )

    assert result.returncode == 0, result.stderr
    assert "wheel validation passed" in result.stdout


def test_check_python_wheel_does_not_accept_a_similar_script_name(tmp_path: Path) -> None:
    wheel_path = _write_fake_wheel(
        tmp_path / "agentic_api-0.4.0-py3-none-any.whl",
        packaged_scripts=("agentic-server",),
        extra_entries={"agentic_api-0.4.0.data/scripts/agentic-api": ""},
    )
    cargo_metadata_path = _write_fake_cargo_metadata(tmp_path / "cargo-metadata.json")
    site_packages = _write_fake_site_packages(tmp_path / "site-packages")
    scripts_dir = _write_fake_scripts(tmp_path / "bin")

    result = _run_check_script(wheel_path, site_packages, scripts_dir, cargo_metadata_path=cargo_metadata_path)

    assert result.returncode != 0
    assert "wheel missing packaged executable: agentic" in result.stderr


def _run_check_script(
    wheel_path: Path,
    site_packages: Path,
    scripts_dir: Path,
    *,
    cargo_metadata_path: Path | None = None,
    expected_wheel_tag: str | None = None,
    cwd: Path = REPO_ROOT,
    expected_version: str | None = EXPECTED_VERSION,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["AGENTIC_API_CHECK_PYTHON"] = sys.executable
    env["AGENTIC_API_CHECK_SCRIPTS_DIR"] = str(scripts_dir)
    if expected_version is not None:
        env["AGENTIC_API_EXPECTED_VERSION"] = expected_version
    env["PYTHONPATH"] = str(site_packages)
    if cargo_metadata_path is not None:
        env["AGENTIC_API_CHECK_CARGO_METADATA_JSON"] = str(cargo_metadata_path)
    if expected_wheel_tag is not None:
        env["AGENTIC_API_EXPECTED_WHEEL_TAG"] = expected_wheel_tag

    return subprocess.run(
        [str(CHECK_SCRIPT), str(wheel_path)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_fake_wheel(
    path: Path,
    extra_entries: dict[str, str] | None = None,
    *,
    version: str = EXPECTED_VERSION,
    packaged_scripts: tuple[str, ...] = ("agentic", "agentic-server"),
) -> Path:
    dist_info = f"agentic_api-{version}.dist-info"
    data_dir = f"agentic_api-{version}.data"
    entries = {
        "agentic_api/__init__.py": f"__version__ = '{version}'\n",
        f"{dist_info}/METADATA": textwrap.dedent(
            f"""\
            Metadata-Version: 2.3
            Name: agentic-api
            Version: {version}
            """
        ),
        f"{dist_info}/entry_points.txt": textwrap.dedent(
            """\
            [console_scripts]
            agentic-api = agentic_api.cli:main
            """
        ),
    }
    entries.update({f"{data_dir}/scripts/{script_name}": "" for script_name in packaged_scripts})
    if extra_entries is not None:
        entries.update(extra_entries)

    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


def _write_fake_site_packages(path: Path, *, version: str = EXPECTED_VERSION) -> Path:
    package_dir = path / "agentic_api"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(f"__version__ = '{version}'\n", encoding="utf-8")

    dist_info = path / f"agentic_api-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        textwrap.dedent(
            f"""\
            Metadata-Version: 2.3
            Name: agentic-api
            Version: {version}
            """
        ),
        encoding="utf-8",
    )
    return path


def _write_fake_cargo_metadata(
    path: Path,
    *,
    package_versions: Mapping[str, str] | None = None,
) -> Path:
    workspace_names = ("agentic-praxis", "agentic-server-core", "agentic-server")
    version_overrides = dict(package_versions or {})
    packages = []
    workspace_members = []

    for name in workspace_names:
        version = version_overrides.get(name, EXPECTED_VERSION)
        package_id = f"path+file:///workspace/{name}#{version}"
        packages.append({"name": name, "version": version, "id": package_id})
        workspace_members.append(package_id)

    path.write_text(
        json.dumps({"packages": packages, "workspace_members": workspace_members}),
        encoding="utf-8",
    )
    return path


def _write_fake_scripts(path: Path, *, version: str = EXPECTED_VERSION) -> Path:
    path.mkdir(parents=True)
    _write_executable(
        path / "agentic",
        f"""
        #!/usr/bin/env python3
        import sys

        if sys.argv[1:] == ["--version"]:
            print("agentic {version}")
            raise SystemExit(0)
        raise SystemExit(1)
        """,
    )
    _write_executable(
        path / "agentic-server",
        f"""
        #!/usr/bin/env python3
        import sys

        if sys.argv[1:] == ["--version"]:
            print("agentic-server {version}")
            raise SystemExit(0)
        raise SystemExit(1)
        """,
    )
    _write_executable(
        path / "agentic-api",
        f"""
        #!/usr/bin/env python3
        import sys

        if sys.argv[1:] == ["version"]:
            print("agentic-api version: {version}")
            print("Rust binary version: agentic-server {version}")
            print("Supported vLLM version: 0.11.0")
            print("Installed vLLM version: not installed")
            raise SystemExit(0)
        raise SystemExit(1)
        """,
    )
    return path


def _write_executable(path: Path, source: str) -> None:
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
