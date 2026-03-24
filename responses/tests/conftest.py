from __future__ import annotations

from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Response

from agentic_stack.configs.builders import build_runtime_config_for_standalone
from agentic_stack.configs.sources import EnvSource
from agentic_stack.entrypoints import llm as mock_llm
from agentic_stack.entrypoints._state import VRAppState, VRRequestState
from agentic_stack.responses_core.store import DBResponseStore
from agentic_stack.routers import serving
from agentic_stack.tools.bootstrap import register_runtime_tool_handlers
from agentic_stack.types.api import UserAgent
from agentic_stack.utils.cassette_replay import (
    CassetteQueue,
    CassetteReplayer,
    load_cassette_yaml,
)
from agentic_stack.utils.exceptions import VRException
from agentic_stack.utils.handlers import exception_handler, path_not_found_handler

register_runtime_tool_handlers()


@pytest.fixture
def chat_completion_cassettes_dir() -> Path:
    # `responses/tests/...` → `responses/` → `responses/tests/cassettes/chat_completion`
    return Path(__file__).resolve().parent / "cassettes" / "chat_completion"


@pytest.fixture
def cassette_replayer_factory(
    chat_completion_cassettes_dir: Path,
) -> Callable[[str], CassetteReplayer]:
    def _make(*filenames: str) -> CassetteReplayer:
        cassettes = [
            load_cassette_yaml(chat_completion_cassettes_dir / name) for name in filenames
        ]
        return CassetteReplayer(
            scenarios={"default": CassetteQueue(name="default", cassettes=cassettes)},
            default_scenario="default",
            strict=False,
        )

    return _make


@pytest.fixture
def stub_code_interpreter_app() -> FastAPI:
    app = FastAPI(title="Stub Code Interpreter")

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"pyodide_loaded": True}

    @app.post("/python")
    async def python(body: dict) -> Response:
        # Return the exact JSON string our recorded vLLM tool-loop cassettes use.
        code = str(body.get("code", "")).strip()
        if code == "2+2":
            payload = '{"status":"success","result":"4","execution_time_ms":8}'
        else:
            payload = '{"status":"success","result":null,"execution_time_ms":1}'
        return Response(content=payload, media_type="application/json")

    return app


@pytest.fixture
def gateway_app() -> FastAPI:
    app = FastAPI(title="VR Gateway (test)")
    app.state.agentic_stack = VRAppState(
        runtime_config=build_runtime_config_for_standalone(
            env=EnvSource(environ={"VR_LLM_API_BASE": "http://mock/v1"})
        )
    )
    app.include_router(serving.router)

    @app.middleware("http")
    async def _init_request_state(request, call_next):
        request.state.agentic_stack = VRRequestState(
            id="test-request-id",
            user_agent=UserAgent.from_user_agent_string("pytest"),
            timing=defaultdict(float),
        )
        return await call_next(request)

    app.add_exception_handler(VRException, exception_handler)
    app.add_exception_handler(Exception, exception_handler)
    app.add_exception_handler(404, path_not_found_handler)
    return app


@pytest.fixture
async def patched_gateway_clients(
    monkeypatch: pytest.MonkeyPatch,
    stub_code_interpreter_app: FastAPI,
    tmp_path: Path,
) -> AsyncIterator[None]:
    """
    Patch:
    - upstream LLM calls → in-process ASGI mock (`agentic_stack.entrypoints.llm.app`)
    - code interpreter HTTP calls → in-process ASGI stub
    """
    # Keep mock LLM cassette replay isolated per test to avoid cross-test scenario
    # cursor leakage (Scenario "default" exhaustion).
    previous_replayer = mock_llm.app.state.agentic_stack.cassette_replayer
    mock_llm.app.state.agentic_stack.cassette_replayer = None

    llm_transport = httpx.ASGITransport(app=mock_llm.app)
    llm_client = httpx.AsyncClient(transport=llm_transport, base_url="http://mock/v1")

    tool_transport = httpx.ASGITransport(app=stub_code_interpreter_app)
    tool_client = httpx.AsyncClient(transport=tool_transport, base_url="http://localhost:5970")

    from pydantic_ai.providers.openai import OpenAIProvider

    def _provider_override(runtime_config, *args, **kwargs):
        _ = runtime_config
        _ = args
        _ = kwargs
        return OpenAIProvider(api_key="test", base_url="http://mock/v1", http_client=llm_client)

    import agentic_stack.lm as lm
    import agentic_stack.responses_core.store as store_mod
    import agentic_stack.routers.serving as serving_mod
    import agentic_stack.tools.code_interpreter as code_interpreter

    store = DBResponseStore.from_db_url(
        db_url=f"sqlite+aiosqlite:///{tmp_path / 'responses_state.db'}"
    )

    monkeypatch.setattr(lm, "get_openai_provider", _provider_override)
    monkeypatch.setattr(code_interpreter, "HTTP_ACLIENT", tool_client)
    monkeypatch.setattr(lm, "get_default_response_store", lambda: store)
    monkeypatch.setattr(store_mod, "get_default_response_store", lambda: store)
    monkeypatch.setattr(serving_mod, "get_default_response_store", lambda: store)

    try:
        yield
    finally:
        mock_llm.app.state.agentic_stack.cassette_replayer = previous_replayer
        await store.aclose()
        await llm_client.aclose()
        await tool_client.aclose()


@pytest.fixture
async def gateway_client(
    gateway_app: FastAPI,
) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=gateway_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        yield client
