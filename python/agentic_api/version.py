from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as metadata_version

from agentic_api import __version__
from agentic_api.binary import find_packaged_binary, read_binary_version
from agentic_api.compatibility import SUPPORTED_VLLM_VERSION


def version_report() -> str:
    binary_path = find_packaged_binary("agentic-server")
    rust_version = read_binary_version(binary_path)
    installed_vllm_version = _installed_vllm_version()
    return "\n".join(
        (
            f"agentic-api version: {__version__}",
            f"Rust binary version: {rust_version}",
            f"Supported vLLM version: {SUPPORTED_VLLM_VERSION}",
            f"Installed vLLM version: {installed_vllm_version}",
        )
    )


def _installed_vllm_version() -> str:
    try:
        return metadata_version("vllm")
    except PackageNotFoundError:
        return "not installed"
