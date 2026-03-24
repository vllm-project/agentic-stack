from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import yaml

ScenarioName: TypeAlias = str
CassetteFilename: TypeAlias = str
# Env/config shape for replay scenarios.
# IMPORTANT: the list order is the replay/consumption order (i.e. an ordered queue).
ScenarioSpec: TypeAlias = dict[ScenarioName, list[CassetteFilename]]


@dataclass(frozen=True, slots=True)
class CassetteRequest:
    method: str
    path: str
    query_params: dict[str, Any]
    body: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CassetteResponse:
    status_code: int
    headers: dict[str, str]
    body: dict[str, Any] | None
    sse: list[str] | None

    @property
    def is_stream(self) -> bool:
        return self.sse is not None


@dataclass(frozen=True, slots=True)
class Cassette:
    filename: str
    request_id: str
    request: CassetteRequest
    response: CassetteResponse


def load_cassette_yaml(path: Path) -> Cassette:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    request = raw.get("request") or {}
    response = raw.get("response") or {}
    return Cassette(
        filename=str(raw.get("filename") or path.stem),
        request_id=str(raw.get("request_id") or ""),
        request=CassetteRequest(
            method=str((request.get("method") or "POST")).upper(),
            path=str(request.get("path") or ""),
            query_params=dict(request.get("query_params") or {}),
            body=dict((request.get("body") or {}) or {}),
        ),
        response=CassetteResponse(
            status_code=int(response.get("status_code") or 200),
            headers={str(k): str(v) for k, v in (response.get("headers") or {}).items()},
            body=response.get("body"),
            sse=response.get("sse"),
        ),
    )


class CassetteReplayError(RuntimeError):
    pass


class CassetteQueue:
    """Ordered, stateful cassette queue for deterministic replay.

    A "scenario" in this repo is implemented as an ordered queue:
    - Each incoming request consumes exactly one cassette (in order).
    - The queue is stateful (cursor advances); exhaustion is an error.

    Notes:
    - This is intentionally *not* a request-matching replayer. Tool loops and client libraries
      often produce slightly different request bodies; queue replay is robust and deterministic.
    - Concurrency: a single queue represents a single deterministic interaction flow.
      Use scenario selection (e.g. request header) to isolate concurrent flows if needed.
    """

    def __init__(self, *, name: str, cassettes: list[Cassette]) -> None:
        if not cassettes:
            raise ValueError("Scenario must contain at least one cassette.")
        self.name = name
        self._cassettes = cassettes
        self._cursor = 0
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._cursor = 0

    def consume_next(self) -> Cassette:
        with self._lock:
            if self._cursor >= len(self._cassettes):
                raise CassetteReplayError(
                    f"Scenario '{self.name}' exhausted (len={len(self._cassettes)})."
                )
            cassette = self._cassettes[self._cursor]
            self._cursor += 1
            return cassette


class CassetteReplayer:
    """Deterministically replay Chat Completions cassettes.

    MVP behavior:
    - choose a scenario (default unless request header overrides it)
    - for each incoming request, consume the next cassette in that scenario
    - validate minimal invariants (method/path/stream flag)
    """

    def __init__(
        self,
        *,
        scenarios: dict[ScenarioName, CassetteQueue],
        default_scenario: str,
        strict: bool = False,
    ) -> None:
        if default_scenario not in scenarios:
            raise ValueError(f"Default scenario '{default_scenario}' not found.")
        self._scenarios = scenarios
        self._default = default_scenario
        self._strict = strict

    @property
    def default_scenario(self) -> str:
        return self._default

    def reset(self) -> None:
        for scenario in self._scenarios.values():
            scenario.reset()

    def _get_scenario(self, name: str | None) -> CassetteQueue:
        if not name:
            return self._scenarios[self._default]
        if name not in self._scenarios:
            raise CassetteReplayError(
                f"Unknown scenario '{name}'. Known: {sorted(self._scenarios.keys())}"
            )
        return self._scenarios[name]

    def next_response(
        self,
        *,
        scenario: str | None,
        method: str,
        path: str,
        stream: bool,
        request_body: dict[str, Any] | None = None,
    ) -> CassetteResponse:
        scenario_obj = self._get_scenario(scenario)
        cassette = scenario_obj.consume_next()

        expected_method = cassette.request.method.upper()
        if method.upper() != expected_method:
            raise CassetteReplayError(
                f"Scenario '{scenario_obj.name}' expected method {expected_method}, got {method}."
            )
        if path != cassette.request.path:
            raise CassetteReplayError(
                f"Scenario '{scenario_obj.name}' expected path {cassette.request.path}, got {path}."
            )

        cassette_stream = bool(cassette.request.body.get("stream"))
        if stream != cassette_stream:
            raise CassetteReplayError(
                f"Scenario '{scenario_obj.name}' expected stream={cassette_stream}, got stream={stream}."
            )

        if self._strict and request_body is not None:
            # Minimal strictness: enforce model match if provided.
            expected_model = cassette.request.body.get("model")
            got_model = request_body.get("model")
            if expected_model and got_model and expected_model != got_model:
                raise CassetteReplayError(
                    f"Scenario '{scenario_obj.name}' expected model '{expected_model}', got '{got_model}'."
                )

        if cassette.response.is_stream and cassette.response.sse is None:
            raise CassetteReplayError(
                f"Cassette '{cassette.filename}' marked stream but has no response.sse."
            )
        if (not cassette.response.is_stream) and cassette.response.body is None:
            raise CassetteReplayError(
                f"Cassette '{cassette.filename}' marked non-stream but has no response.body."
            )
        return cassette.response

    @classmethod
    def from_env(
        cls,
        *,
        cassette_dir: Path,
        scenarios_json: str,
        default_scenario: str,
        strict: bool = False,
    ) -> CassetteReplayer:
        scenarios_raw: Any = json.loads(scenarios_json)
        if not isinstance(scenarios_raw, dict):
            raise ValueError("VR_MOCK_LLM_SCENARIOS must be a JSON object mapping name->list.")
        scenarios: dict[ScenarioName, CassetteQueue] = {}
        for name, files in scenarios_raw.items():
            if not isinstance(files, list) or not all(isinstance(p, str) for p in files):
                raise ValueError(f"Scenario '{name}' must map to a list of cassette filenames.")
            cassettes = [load_cassette_yaml(cassette_dir / Path(p)) for p in files]
            scenarios[name] = CassetteQueue(name=name, cassettes=cassettes)
        return cls(scenarios=scenarios, default_scenario=default_scenario, strict=strict)


async def stream_sse_chunks(chunks: list[str]) -> AsyncIterator[str]:
    for ch in chunks:
        yield ch
