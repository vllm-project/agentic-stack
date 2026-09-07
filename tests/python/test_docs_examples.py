from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
DOCS_INDEX = REPO_ROOT / "docs" / "index.md"
INSTALL_GUIDE = REPO_ROOT / "docs" / "guides" / "python-installation.md"


def test_documented_python_install_commands_respect_release_publication_gate() -> None:
    guide = INSTALL_GUIDE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    index = DOCS_INDEX.read_text(encoding="utf-8")

    combined = "\n".join((readme, index, guide))

    assert 'uv pip install "$WHEEL_PATH"' in combined
    assert 'agentic-api[local] @ file://$WHEEL_PATH' in combined
    assert "This release produces wheel artifacts" in combined
    assert "After PyPI publication" in combined
    assert "after the PyPI publication gate" in combined
    assert "Planned for 0.5.0" not in combined
    assert "0.4.0 build-only release" not in combined
    assert "future 0.5.0 public-index gate" not in combined
    assert "uv pip install agentic-api" in combined
    assert 'uv pip install "agentic-api[local]"' in combined
    assert "uvx --from agentic-api agentic-api doctor" in combined
    assert "uvx --from agentic-api agentic-api serve --vllm-base-url http://existing-vllm:8000" in combined
    assert "uvx pip install" not in combined


def test_python_install_guide_covers_workflows_and_backend_language() -> None:
    guide = INSTALL_GUIDE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    assert "agentic-api serve --vllm-base-url http://existing-vllm:8000" in guide
    assert "agentic-api serve --model Qwen/Qwen3-30B-A3B-FP8" in guide
    assert "agentic-api doctor --mode remote" in guide
    assert "agentic run codex --model MODEL_ID" in guide
    assert "agentic run claude --model SERVED_MODEL_ALIAS" in guide
    assert "vLLM is one supported backend, not part of the Agentic API product name" in readme
    assert "The Rust-native `agentic` CLI remains supported" in guide


def test_python_install_guide_documents_known_good_model_profiles() -> None:
    guide = INSTALL_GUIDE.read_text(encoding="utf-8")

    assert "documentation data, not an allowlist" in guide
    assert "Qwen/Qwen3-30B-A3B-FP8" in guide
    assert "qwen3-30b-a3b-fp8" in guide
    assert "vllm serve Qwen/Qwen3-30B-A3B-FP8 --reasoning-parser deepseek_r1 --port 5050" in guide
    assert "vllm serve Qwen/Qwen3-30B-A3B-FP8 --tool-call-parser hermes --enable-auto-tool-choice --port 5050" in guide
    assert "Qwen/Qwen3.5-35B-A3B-FP8" in guide
    assert "Qwen/Qwen3.8-27B-FP8" in guide
