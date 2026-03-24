from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover
    _bdist_wheel = None


_BUNDLED_BINARY = Path(
    "python/agentic_stack/tools/code_interpreter/bin/linux/x86_64/code-interpreter-server"
)


def _has_bundled_code_interpreter_binary() -> bool:
    target = Path(__file__).resolve().parent / _BUNDLED_BINARY
    return target.is_file() and os.access(target, os.X_OK)


cmdclass: dict[str, type] = {}

if _bdist_wheel is not None:

    class bdist_wheel(_bdist_wheel):  # type: ignore[misc,valid-type]
        """Emit a platform wheel when a Linux server binary is bundled."""

        def finalize_options(self) -> None:
            super().finalize_options()
            if _has_bundled_code_interpreter_binary():
                self.root_is_pure = False

        def get_tag(self) -> tuple[str, str, str]:
            python_tag, abi_tag, platform_tag = super().get_tag()
            if _has_bundled_code_interpreter_binary():
                return ("py3", "none", platform_tag)
            return (python_tag, abi_tag, platform_tag)

    cmdclass["bdist_wheel"] = bdist_wheel


setup(cmdclass=cmdclass)
