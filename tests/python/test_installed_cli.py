from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from agentic_api import __version__


COMMAND_TIMEOUT_S = 20


def scripts_dir() -> Path:
    path = sysconfig.get_path("scripts")
    assert path is not None
    return Path(path).resolve()


def require_wheel_install() -> None:
    if os.environ.get("AGENTIC_API_TEST_WHEEL") is None:
        pytest.skip("set AGENTIC_API_TEST_WHEEL to exercise installed wheel commands")


def test_installed_wheel_exposes_expected_commands() -> None:
    require_wheel_install()
    directory = scripts_dir()

    for name in ("agentic-api", "agentic", "agentic-server"):
        path = directory / name
        assert path.is_file(), f"expected installed executable at {path}"
        assert os.access(path, os.X_OK), f"expected executable permissions on {path}"


def test_agentic_api_serve_uses_discovered_packaged_server_without_importing_vllm(tmp_path: Path) -> None:
    require_wheel_install()
    directory = scripts_dir()
    server_path = directory / "agentic-server"
    backup_path = tmp_path / "agentic-server.real"
    record_path = tmp_path / "agentic-server-record.json"
    guard_path = tmp_path / "sitecustomize.py"

    shutil.move(server_path, backup_path)
    try:
        server_path.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env python3",
                    "from __future__ import annotations",
                    "import json",
                    "import os",
                    "import sys",
                    "",
                    "record_path = os.environ['AGENTIC_TEST_RECORD_PATH']",
                    "with open(record_path, 'w', encoding='utf-8') as handle:",
                    "    json.dump(",
                    "        {",
                    "            'argv': sys.argv[1:],",
                    "            'openai_api_key': os.environ.get('OPENAI_API_KEY'),",
                    "        },",
                    "        handle,",
                    "    )",
                    "raise SystemExit(0)",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        server_path.chmod(0o755)

        guard_path.write_text(
            "\n".join(
                [
                    "from __future__ import annotations",
                    "import builtins",
                    "",
                    "_real_import = builtins.__import__",
                    "",
                    "def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):",
                    "    if name == 'vllm' or name.startswith('vllm.'):",
                    "        raise RuntimeError(f'unexpected vllm import: {name}')",
                    "    return _real_import(name, globals, locals, fromlist, level)",
                    "",
                    "builtins.__import__ = guarded_import",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["AGENTIC_TEST_RECORD_PATH"] = str(record_path)
        env["CUSTOM_GATEWAY_KEY"] = "wheel-remote-key"
        env["PYTHONPATH"] = str(tmp_path)

        result = subprocess.run(
            [
                str(directory / "agentic-api"),
                "serve",
                "--vllm-base-url",
                "https://upstream.example.test/base",
                "--host",
                "127.0.0.1",
                "--port",
                "7777",
                "--gateway-api-key-env",
                "CUSTOM_GATEWAY_KEY",
            ],
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
            env=env,
        )

        assert result.returncode == 1, result.stderr
        assert "agentic-server exited unexpectedly with status 0" in result.stderr
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record == {
            "argv": [
                "--llm-api-base",
                "https://upstream.example.test/base",
                "--llm-ready-timeout-s",
                "600.0",
                "--gateway-host",
                "127.0.0.1",
                "--gateway-port",
                "7777",
            ],
            "openai_api_key": "wheel-remote-key",
        }
    finally:
        if backup_path.exists():
            if server_path.exists():
                server_path.unlink()
            shutil.move(backup_path, server_path)


@pytest.mark.skipif(os.name != "posix", reason="prefix script layout is POSIX-specific")
def test_agentic_api_entry_point_works_in_prefix_without_sibling_python(tmp_path: Path) -> None:
    wheel_value = os.environ.get("AGENTIC_API_TEST_WHEEL")
    if wheel_value is None:
        pytest.skip("set AGENTIC_API_TEST_WHEEL to exercise the built wheel")

    wheel_path = Path(wheel_value).resolve()
    assert wheel_path.is_file(), f"wheel does not exist: {wheel_path}"
    uv = shutil.which("uv")
    assert uv is not None, "uv is required for the prefix-install regression"

    prefix = tmp_path / "prefix"
    install = subprocess.run(
        [uv, "pip", "install", "--python", sys.executable, "--prefix", str(prefix), str(wheel_path)],
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_S,
    )
    assert install.returncode == 0, install.stderr

    prefix_bin = prefix / "bin"
    command = prefix_bin / "agentic-api"
    assert command.is_file()
    assert not (prefix_bin / "python").exists()

    site_packages = list((prefix / "lib").glob("python*/site-packages"))
    assert len(site_packages) == 1
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join((str(prefix_bin), env.get("PATH", "")))
    env["PYTHONPATH"] = str(site_packages[0])

    result = subprocess.run(
        [str(command), "version"],
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_S,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert f"agentic-api version: {__version__}" in result.stdout
