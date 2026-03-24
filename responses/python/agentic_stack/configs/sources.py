from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class EnvSource:
    environ: dict[str, str]

    @classmethod
    def from_env(cls) -> "EnvSource":
        return cls(environ=dict(os.environ))

    def get(self, key: str) -> tuple[str | None, bool]:
        if key in self.environ:
            return self.environ[key], True
        return None, False

    def get_typed(self, key: str, default: T, parse) -> T:
        value, is_set = self.get(key)
        if not is_set or value is None or value.strip() == "":
            return default
        try:
            return parse(value)
        except Exception:
            raise ValueError(f"invalid {key}={value!r}") from None

    def get_str(self, key: str, default: str) -> str:
        return self.get_typed(key, default, lambda value: value)

    def get_optional_str(self, key: str, default: str | None = None) -> str | None:
        value, is_set = self.get(key)
        if not is_set or value is None:
            return default
        stripped = value.strip()
        return stripped or None

    def get_int(self, key: str, default: int) -> int:
        return self.get_typed(key, default, int)

    def get_float(self, key: str, default: float) -> float:
        return self.get_typed(key, default, float)

    def get_bool(self, key: str, default: bool) -> bool:
        def _parse(value: str) -> bool:
            normalized = value.strip().lower()
            if normalized in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "f", "no", "n", "off"}:
                return False
            raise ValueError("invalid boolean")

        return self.get_typed(key, default, _parse)
