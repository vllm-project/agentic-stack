from __future__ import annotations

import httpx
import pytest
from sse_test_utils import extract_completed_response, parse_sse_frames, parse_sse_json_events

from agentic_stack.entrypoints import llm as mock_llm


def _extract_completed_response(sse_text: str) -> dict:
    frames = parse_sse_frames(sse_text)
    events = parse_sse_json_events(frames)
    return extract_completed_response(events)


def _extract_completed_response_id(sse_text: str) -> str:
    resp = _extract_completed_response(sse_text)
    return str(resp["id"])


@pytest.mark.anyio
async def test_previous_response_id_statefulness_across_requests(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
):
    # We replay two upstream chat completion streams deterministically; the main point here is
    # to validate that the gateway's shared ResponseStore enables `previous_response_id` hydration.
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "text-single-stream.yaml",
        "text-single-stream.yaml",
    )

    async with gateway_client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": True,
            "input": [{"role": "user", "content": "Hello"}],
            "tool_choice": "none",
        },
    ) as resp1:
        assert resp1.status_code == 200
        body1 = await resp1.aread()
    text1 = body1.decode("utf-8", errors="replace")
    r1 = _extract_completed_response_id(text1)

    async with gateway_client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": True,
            "previous_response_id": r1,
            "input": [{"role": "user", "content": "And Germany?"}],
            "tool_choice": "none",
        },
    ) as resp2:
        assert resp2.status_code == 200
        body2 = await resp2.aread()
    text2 = body2.decode("utf-8", errors="replace")
    assert "event: response.completed" in text2
    assert "data: [DONE]\n\n" in text2


@pytest.mark.anyio
async def test_previous_response_id_custom_function_tool_loop_omit_tools(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
):
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "vllm-code_interpreter-step1-stream.yaml",
        "vllm-code_interpreter-step2-stream.yaml",
    )

    async with gateway_client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": True,
            "input": [
                {
                    "role": "user",
                    "content": "You MUST call the code_interpreter tool now. Execute: 2+2.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "code_interpreter",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            ],
            "tool_choice": "auto",
        },
    ) as resp1:
        assert resp1.status_code == 200
        body1 = await resp1.aread()
    text1 = body1.decode("utf-8", errors="replace")

    completed1 = _extract_completed_response(text1)
    response_id = str(completed1["id"])
    call_id = next(
        (
            str(item.get("call_id"))
            for item in (completed1.get("output") or [])
            if isinstance(item, dict) and item.get("type") == "function_call"
        ),
        "",
    )
    assert call_id

    async with gateway_client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": True,
            "previous_response_id": response_id,
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": '{"status":"success","result":"4","execution_time_ms":8}',
                }
            ],
        },
    ) as resp2:
        assert resp2.status_code == 200
        body2 = await resp2.aread()
    text2 = body2.decode("utf-8", errors="replace")

    assert "event: response.completed" in text2
    assert "data: [DONE]\n\n" in text2
    assert "4" in text2


@pytest.mark.anyio
async def test_retrieve_response_returns_stored_response(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
) -> None:
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "text-single-stream.yaml"
    )

    create_resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "Hello"}],
            "tool_choice": "none",
        },
    )
    assert create_resp.status_code == 200
    created = create_resp.json()

    retrieve_resp = await gateway_client.get(f"/v1/responses/{created['id']}")
    assert retrieve_resp.status_code == 200
    retrieved = retrieve_resp.json()
    assert retrieved["id"] == created["id"]
    assert retrieved["status"] == created["status"]


@pytest.mark.anyio
async def test_retrieve_response_missing_returns_openai_error(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.get("/v1/responses/resp_missing")

    assert resp.status_code == 404
    assert resp.json() == {
        "error": {
            "message": "No response found with id 'resp_missing'.",
            "type": "invalid_request_error",
            "param": "response_id",
            "code": "response_not_found",
        }
    }


@pytest.mark.anyio
async def test_previous_response_id_missing_returns_openai_error(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
) -> None:
    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "previous_response_id": "resp_missing",
            "input": [{"role": "user", "content": "Hello"}],
            "tool_choice": "none",
        },
    )

    assert resp.status_code == 400
    assert resp.json() == {
        "error": {
            "message": "No response found with id 'resp_missing'.",
            "type": "invalid_request_error",
            "param": "previous_response_id",
            "code": "previous_response_not_found",
        }
    }


@pytest.mark.anyio
async def test_store_false_response_is_not_retrievable_or_continuable(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
) -> None:
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "text-single-stream.yaml"
    )

    create_resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "store": False,
            "input": [{"role": "user", "content": "Hello"}],
            "tool_choice": "none",
        },
    )
    assert create_resp.status_code == 200
    response_id = create_resp.json()["id"]

    retrieve_resp = await gateway_client.get(f"/v1/responses/{response_id}")
    assert retrieve_resp.status_code == 404
    assert retrieve_resp.json()["error"]["code"] == "response_not_found"

    continue_resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "previous_response_id": response_id,
            "input": [{"role": "user", "content": "Continue"}],
            "tool_choice": "none",
        },
    )
    assert continue_resp.status_code == 400
    assert continue_resp.json()["error"]["code"] == "previous_response_not_found"
