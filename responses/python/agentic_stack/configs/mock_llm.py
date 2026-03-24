from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class MockLLMConfig(BaseSettings):
    """Configuration for the mock upstream LLM server (`agentic_stack.entrypoints.llm`).

    Notes:
    - We intentionally keep this separate from the gateway runtime-config builders because:
      - `VR_MOCK_LLM_SCENARIOS` can be large (avoid printing/logging it).
      - The gateway runtime does not need mock-LLM settings.
    - Env vars use the `VR_MOCK_LLM_` prefix to make scope unambiguous.
    """

    model_config = SettingsConfigDict(
        env_prefix="vr_mock_llm_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        cli_parse_args=False,
    )

    mode: Literal["synthetic", "replay"] = "synthetic"
    cassette_dir: str = "responses/tests/cassettes/chat_completion"
    scenarios: SecretStr = SecretStr("")
    default_scenario: str = "default"
    strict: bool = False

    @property
    def cassette_dir_path(self) -> Path:
        return Path(self.cassette_dir)

    @property
    def scenarios_json(self) -> str | None:
        """
        JSON string mapping scenario name -> ordered cassette filename list.

        This is stored as a `SecretStr` even though it is not a secret:
        the goal is to avoid accidental logging / repr of a potentially large JSON blob.
        """
        value = self.scenarios.get_secret_value().strip()
        return value or None

    def scenarios_mapping(self) -> dict[str, list[str]]:
        raw = self.scenarios_json
        if not raw:
            return {}
        data: Any = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("VR_MOCK_LLM_SCENARIOS must be a JSON object mapping name->list.")
        out: dict[str, list[str]] = {}
        for name, files in data.items():
            if not isinstance(name, str):
                raise ValueError("Scenario names must be strings.")
            if not isinstance(files, list) or not all(isinstance(p, str) for p in files):
                raise ValueError(f"Scenario '{name}' must map to a list of cassette filenames.")
            out[name] = list(files)
        return out
