from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from agentic_api import __version__
from agentic_api.compatibility import SUPPORTED_VLLM_VERSION
from agentic_api.version import version_report


def test_version_report_includes_package_rust_and_vllm_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "agentic-server"
    binary.write_text("#!/bin/sh\nprintf 'agentic-server 0.4.0\\n'\n")
    binary.chmod(0o755)

    monkeypatch.setattr("agentic_api.version.find_packaged_binary", lambda name: binary)
    monkeypatch.setattr(
        "agentic_api.version.metadata_version",
        lambda name: (_ for _ in ()).throw(PackageNotFoundError(name)),
    )

    report = version_report()

    assert f"agentic-api version: {__version__}" in report
    assert "Rust binary version: agentic-server 0.4.0" in report
    assert f"Supported vLLM version: {SUPPORTED_VLLM_VERSION}" in report
    assert "Installed vLLM version: not installed" in report
