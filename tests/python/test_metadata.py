from importlib.metadata import distribution

from agentic_api import __version__
from agentic_api.compatibility import SUPPORTED_VLLM_VERSION


def test_installed_package_version_matches_package_constant() -> None:
    installed_distribution = distribution("agentic-api")
    assert installed_distribution.version == __version__
    assert installed_distribution.metadata["Version"] == __version__
    assert __version__ == installed_distribution.version


def test_supported_vllm_is_exactly_declared_in_linux_local_extra() -> None:
    metadata = distribution("agentic-api").metadata
    assert (
        f"vllm=={SUPPORTED_VLLM_VERSION} ; platform_system == 'Linux' and extra == 'local'"
        in metadata.get_all("Requires-Dist")
    )
