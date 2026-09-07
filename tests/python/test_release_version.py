from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = REPO_ROOT / "scripts" / "validate-python-release-version.sh"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-python.yml"
PYTHON_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "python.yml"
BUILD_CONSTRAINTS = REPO_ROOT / "python-build-constraints.txt"
WORKSPACE_VERSION = re.search(
    r"(?ms)^\[workspace\.package\].*?^version\s*=\s*\"([^\"]+)\"", (REPO_ROOT / "Cargo.toml").read_text()
).group(1)


def test_release_version_validator_accepts_build_only_version() -> None:
    env = os.environ.copy()
    env["AGENTIC_API_RELEASE_VERSION"] = WORKSPACE_VERSION

    result = subprocess.run(["/bin/bash", str(VALIDATOR)], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


def test_release_version_validator_rejects_shell_payload_without_executing_it(tmp_path: Path) -> None:
    marker = tmp_path / "injected"
    env = os.environ.copy()
    env["AGENTIC_API_RELEASE_VERSION"] = f"{WORKSPACE_VERSION}; touch {marker}"

    result = subprocess.run(["/bin/bash", str(VALIDATOR)], env=env, capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert not marker.exists()
    assert f"{WORKSPACE_VERSION} build-only workflow" in result.stderr


def test_release_workflow_keeps_dispatch_version_out_of_shell_source() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "AGENTIC_API_RELEASE_VERSION: ${{ inputs.version }}" in workflow
    run_blocks = _workflow_run_blocks(workflow)
    assert run_blocks
    assert all("${{ inputs.version }}" not in block for block in run_blocks)


def test_python_workflows_pin_build_tools_and_manylinux_artifact_contract() -> None:
    release_workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    python_workflow = PYTHON_WORKFLOW.read_text(encoding="utf-8")
    constraints = BUILD_CONSTRAINTS.read_text(encoding="utf-8")

    assert "maturin==1.14.1" in constraints
    assert "pytest==9.1.1" in constraints
    assert "uv==0.11.21" in constraints
    assert "PyO3/maturin-action@86b9d133d34bc1b40018696f782949dac11bd380" in release_workflow
    assert (
        "quay.io/pypa/manylinux2014_x86_64@"
        "sha256:95440e0e72dd3a81dc8d2cf59a84d57af661456620f5bc821ff92048d0e54ff9"
    ) in release_workflow
    assert "manylinux: \"2014\"" in release_workflow
    assert "wheel-tag: py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64" in release_workflow
    assert "args: --release --locked" in release_workflow
    assert 'uv pip install --python .venv/bin/python "$wheel_path"' in release_workflow
    assert 'AGENTIC_API_TEST_WHEEL="$wheel_path" .venv/bin/python -m pytest tests/python -q' in release_workflow
    assert "AGENTIC_API_CHECK_PYTHON=.venv/bin/python" in release_workflow
    assert "AGENTIC_API_CHECK_SCRIPTS_DIR=.venv/bin" in release_workflow
    assert "hashFiles('Cargo.lock', 'python-build-constraints.txt')" in release_workflow
    assert "hashFiles('Cargo.lock', 'python-build-constraints.txt')" in python_workflow


def _workflow_run_blocks(workflow: str) -> list[str]:
    lines = workflow.splitlines()
    blocks: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("run:"):
            continue

        indent = len(line) - len(stripped)
        block = [stripped.removeprefix("run:").strip()]
        for candidate in lines[index + 1 :]:
            candidate_stripped = candidate.lstrip()
            candidate_indent = len(candidate) - len(candidate_stripped)
            if candidate_stripped and candidate_indent <= indent:
                break
            block.append(candidate)
        blocks.append("\n".join(block))
    return blocks
