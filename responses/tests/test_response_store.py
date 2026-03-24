from __future__ import annotations

from pathlib import Path

import pytest

from agentic_stack.configs.builders import build_runtime_config_for_standalone
from agentic_stack.configs.sources import EnvSource
from agentic_stack.responses_core.store import DBResponseStore
from agentic_stack.types.openai import (
    OpenAIOutputItem,
    OpenAIOutputTextContent,
    OpenAIResponsesResponse,
    vLLMResponsesRequest,
)
from agentic_stack.utils.exceptions import BadInputError


@pytest.mark.anyio
async def test_store_put_and_get_roundtrip(tmp_path: Path):
    db_path = tmp_path / "state.db"
    store = DBResponseStore.from_db_url(db_url=f"sqlite+aiosqlite:///{db_path}")

    req = vLLMResponsesRequest(
        model="test-model",
        input=[{"role": "user", "content": "hi"}],
        tool_choice="none",
    )
    hydrated_req = req
    resp = OpenAIResponsesResponse(
        model="test-model",
        status="completed",
        output=[
            OpenAIOutputItem(
                role="assistant",
                status="completed",
                id="msg_1",
                content=[OpenAIOutputTextContent(text="hello")],
            )
        ],
    )

    await store.put_completed(request=req, hydrated_request=hydrated_req, response=resp)

    stored = await store.get(response_id=resp.id)
    assert stored is not None
    payload = stored.payload()
    assert payload.response.id == resp.id
    assert payload.response.output[0].type == "message"
    assert payload.hydrated_input[0].role == "user"


@pytest.mark.anyio
async def test_store_put_and_get_roundtrip_incomplete_terminal(tmp_path: Path):
    db_path = tmp_path / "state.db"
    store = DBResponseStore.from_db_url(db_url=f"sqlite+aiosqlite:///{db_path}")

    req = vLLMResponsesRequest(
        model="test-model",
        input=[{"role": "user", "content": "hi"}],
        tool_choice="none",
    )
    resp = OpenAIResponsesResponse(
        model="test-model",
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
        output=[],
    )

    await store.put_completed(request=req, hydrated_request=req, response=resp)

    stored = await store.get(response_id=resp.id)
    assert stored is not None
    payload = stored.payload()
    assert payload.response.id == resp.id
    assert payload.response.status == "incomplete"
    assert payload.response.incomplete_details is not None
    assert payload.response.incomplete_details.reason == "max_output_tokens"


@pytest.mark.anyio
async def test_store_skips_non_terminal_response_status(tmp_path: Path):
    db_path = tmp_path / "state.db"
    store = DBResponseStore.from_db_url(db_url=f"sqlite+aiosqlite:///{db_path}")

    req = vLLMResponsesRequest(
        model="test-model",
        input=[{"role": "user", "content": "hi"}],
        tool_choice="none",
    )
    resp = OpenAIResponsesResponse(
        model="test-model",
        status="in_progress",
        output=[],
    )

    await store.put_completed(request=req, hydrated_request=req, response=resp)

    stored = await store.get(response_id=resp.id)
    assert stored is None


class _StubCache:
    def __init__(self) -> None:
        self.data: dict[str, object] = {}
        self.set_calls: int = 0

    async def get_json(self, key: str) -> object | None:
        return self.data.get(key)

    async def set_json(self, key: str, value: object, **_kwargs) -> None:
        self.data[key] = value
        self.set_calls += 1


def _install_store_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cache_enabled: bool,
    cache_ttl_seconds: int = 3600,
    workers: int = 1,
) -> None:
    import agentic_stack.responses_core.store as store_mod

    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={
                "VR_LLM_API_BASE": "http://mock/v1",
                "VR_RESPONSE_STORE_CACHE": "1" if cache_enabled else "0",
                "VR_RESPONSE_STORE_CACHE_TTL_SECONDS": str(cache_ttl_seconds),
                "VR_WORKERS": str(workers),
                "VR_DB_PATH": "sqlite+aiosqlite:///ignored.db",
            }
        )
    )
    monkeypatch.setattr(store_mod, "_STORE_RUNTIME_CONFIG", runtime_config)


@pytest.mark.anyio
async def test_store_get_prefers_cache_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import agentic_stack.responses_core.store as store_mod

    db_path = tmp_path / "state.db"
    store = DBResponseStore.from_db_url(db_url=f"sqlite+aiosqlite:///{db_path}")
    cache = _StubCache()

    monkeypatch.setattr(store_mod, "_STORE_CACHE", cache)
    _install_store_runtime_config(monkeypatch, cache_enabled=True)

    req = vLLMResponsesRequest(
        model="test-model", input=[{"role": "user", "content": "hi"}], tool_choice="none"
    )
    resp = OpenAIResponsesResponse(
        model="test-model",
        status="completed",
        output=[
            OpenAIOutputItem(
                role="assistant",
                status="completed",
                id="msg_1",
                content=[OpenAIOutputTextContent(text="hello")],
            )
        ],
    )

    await store.put_completed(request=req, hydrated_request=req, response=resp)

    # Guard: if the cache-hit path regresses and touches the DB, this will fail the test.
    async def _boom() -> None:
        raise AssertionError("ensure_schema() should not be called on cache hit")

    monkeypatch.setattr(store, "ensure_schema", _boom)

    stored = await store.get(response_id=resp.id)
    assert stored is not None
    assert stored.payload().response.id == resp.id
    assert cache.set_calls >= 1

    await store.aclose()


@pytest.mark.anyio
async def test_store_get_populates_cache_on_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import agentic_stack.responses_core.store as store_mod

    db_path = tmp_path / "state.db"
    store = DBResponseStore.from_db_url(db_url=f"sqlite+aiosqlite:///{db_path}")
    cache = _StubCache()

    monkeypatch.setattr(store_mod, "_STORE_CACHE", cache)
    _install_store_runtime_config(monkeypatch, cache_enabled=False)

    req = vLLMResponsesRequest(
        model="test-model", input=[{"role": "user", "content": "hi"}], tool_choice="none"
    )
    resp = OpenAIResponsesResponse(model="test-model", status="completed", output=[])

    # Store in DB while cache is disabled (so we can test cache-aside on get()).
    await store.put_completed(request=req, hydrated_request=req, response=resp)

    _install_store_runtime_config(monkeypatch, cache_enabled=True)

    stored = await store.get(response_id=resp.id)
    assert stored is not None
    assert cache.set_calls == 1

    await store.aclose()


@pytest.mark.anyio
async def test_store_get_falls_back_to_db_on_cache_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import agentic_stack.responses_core.store as store_mod

    class _ExplodingCache(_StubCache):
        async def get_json(self, key: str) -> object | None:
            raise RuntimeError("boom")

    db_path = tmp_path / "state.db"
    store = DBResponseStore.from_db_url(db_url=f"sqlite+aiosqlite:///{db_path}")
    cache = _ExplodingCache()

    monkeypatch.setattr(store_mod, "_STORE_CACHE", cache)
    _install_store_runtime_config(monkeypatch, cache_enabled=True)

    req = vLLMResponsesRequest(
        model="test-model", input=[{"role": "user", "content": "hi"}], tool_choice="none"
    )
    resp = OpenAIResponsesResponse(model="test-model", status="completed", output=[])
    await store.put_completed(request=req, hydrated_request=req, response=resp)

    stored = await store.get(response_id=resp.id)
    assert stored is not None
    assert stored.payload().response.id == resp.id

    await store.aclose()


@pytest.mark.anyio
async def test_store_false_does_not_persist_response(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    store = DBResponseStore.from_db_url(db_url=f"sqlite+aiosqlite:///{db_path}")

    req = vLLMResponsesRequest(
        model="test-model",
        input=[{"role": "user", "content": "hi"}],
        tool_choice="none",
        store=False,
    )
    resp = OpenAIResponsesResponse(model="test-model", status="completed", output=[])

    await store.put_completed(request=req, hydrated_request=req, response=resp)

    stored = await store.get(response_id=resp.id)
    assert stored is None

    await store.aclose()


@pytest.mark.anyio
async def test_store_false_does_not_populate_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentic_stack.responses_core.store as store_mod

    db_path = tmp_path / "state.db"
    store = DBResponseStore.from_db_url(db_url=f"sqlite+aiosqlite:///{db_path}")
    cache = _StubCache()

    monkeypatch.setattr(store_mod, "_STORE_CACHE", cache)
    _install_store_runtime_config(monkeypatch, cache_enabled=True)

    req = vLLMResponsesRequest(
        model="test-model",
        input=[{"role": "user", "content": "hi"}],
        tool_choice="none",
        store=False,
    )
    resp = OpenAIResponsesResponse(model="test-model", status="completed", output=[])

    await store.put_completed(request=req, hydrated_request=req, response=resp)

    assert cache.set_calls == 0
    assert await store.get(response_id=resp.id) is None

    await store.aclose()


@pytest.mark.anyio
async def test_store_response_id_immutable(tmp_path: Path):
    db_path = tmp_path / "state.db"
    store = DBResponseStore.from_db_url(db_url=f"sqlite+aiosqlite:///{db_path}")

    req = vLLMResponsesRequest(
        model="test-model",
        input=[{"role": "user", "content": "hi"}],
        tool_choice="none",
    )
    resp = OpenAIResponsesResponse(model="test-model", status="completed", output=[])
    fixed_id = "resp_fixed"
    resp.id = fixed_id

    await store.put_completed(request=req, hydrated_request=req, response=resp)
    with pytest.raises(BadInputError):
        await store.put_completed(request=req, hydrated_request=req, response=resp)


@pytest.mark.anyio
async def test_hydration_appends_previous_input_and_output(tmp_path: Path):
    db_path = tmp_path / "state.db"
    store = DBResponseStore.from_db_url(db_url=f"sqlite+aiosqlite:///{db_path}")

    step1_req = vLLMResponsesRequest(
        model="test-model",
        input=[{"role": "user", "content": "hi"}],
        tool_choice="none",
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object"},
                "strict": True,
            }
        ],
    )
    step1_resp = OpenAIResponsesResponse(
        model="test-model",
        status="completed",
        output=[
            OpenAIOutputItem(
                role="assistant",
                status="completed",
                id="msg_1",
                content=[OpenAIOutputTextContent(text="hello")],
            )
        ],
    )
    await store.put_completed(request=step1_req, hydrated_request=step1_req, response=step1_resp)

    step2_req = vLLMResponsesRequest(
        model="test-model",
        previous_response_id=step1_resp.id,
        input=[{"type": "function_call_output", "call_id": "call_1", "output": "ok"}],
        tool_choice="auto",
    )
    hydrated = await store.rehydrate_request(request=step2_req)

    assert hydrated.previous_response_id is None
    assert len(hydrated.input) == 3
    assert hydrated.input[0].role == "user"
    assert hydrated.input[1].type == "message"
    assert hydrated.input[2].type == "function_call_output"
    # tools omitted in step2 => reuse stored tools
    assert hydrated.tools is not None
    assert hydrated.tools[0].type == "function"
