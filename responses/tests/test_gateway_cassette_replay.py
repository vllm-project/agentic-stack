from __future__ import annotations

import httpx
import pytest
from sse_test_utils import (
    extract_completed_response,
    index_of_event_type,
    parse_sse_frames,
    parse_sse_json_events,
    sse_has_done_marker,
)

from agentic_stack.entrypoints import llm as mock_llm


@pytest.mark.anyio
async def test_responses_stream_text_single_from_chat_completion_cassette(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
):
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "text-single-stream.yaml"
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
    ) as resp:
        assert resp.status_code == 200
        body = await resp.aread()

    text = body.decode("utf-8", errors="replace")
    frames = parse_sse_frames(text)
    events = parse_sse_json_events(frames)
    assert sse_has_done_marker(frames) is True

    created_i = index_of_event_type(events, "response.created")
    completed_i = index_of_event_type(events, "response.completed")
    assert created_i < completed_i

    completed = extract_completed_response(events)
    # Sanity: ensure the upstream content propagated into the completed response.
    assert any(
        "Paris" in (part.get("text") or "")
        for item in (completed.get("output") or [])
        if isinstance(item, dict)
        for part in (item.get("content") or [])
        if isinstance(part, dict)
    )

    # OpenResponses streaming schema requires `logprobs` on output_text delta/done events.
    for event in events:
        if event.get("type") in {"response.output_text.delta", "response.output_text.done"}:
            assert "logprobs" in event
            assert isinstance(event["logprobs"], list)


@pytest.mark.anyio
async def test_responses_stream_upstream_http_error_emits_error_then_failed(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
):
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "model-not-found-stream.yaml"
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
    ) as resp:
        assert resp.status_code == 200
        body = await resp.aread()

    text = body.decode("utf-8", errors="replace")
    frames = parse_sse_frames(text)
    events = parse_sse_json_events(frames)
    assert sse_has_done_marker(frames) is True

    created_i = index_of_event_type(events, "response.created")
    error_i = index_of_event_type(events, "error")
    failed_i = index_of_event_type(events, "response.failed")
    assert created_i < error_i < failed_i


@pytest.mark.anyio
async def test_responses_non_stream_text_single_from_streaming_upstream_cassette(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
):
    # NOTE: LMEngine currently uses `pydantic_ai.Agent.run_stream_events(...)` even when the
    # downstream Responses request is non-streaming, so the upstream Chat Completions call is streaming.
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "text-single-stream.yaml"
    )

    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "Hello"}],
            "tool_choice": "none",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["object"] == "response"
    assert any(
        "Paris" in (c.get("text") or "")
        for o in data.get("output", [])
        if isinstance(o, dict)
        for c in (o.get("content") or [])
    )


@pytest.mark.anyio
async def test_responses_stream_finish_reason_length_emits_incomplete(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
):
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "text-max_tokens-length-stream.yaml"
    )

    async with gateway_client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": True,
            "max_output_tokens": 1,
            "input": [{"role": "user", "content": "Hello"}],
            "tool_choice": "none",
        },
    ) as resp:
        assert resp.status_code == 200
        body = await resp.aread()

    text = body.decode("utf-8", errors="replace")
    frames = parse_sse_frames(text)
    events = parse_sse_json_events(frames)
    assert sse_has_done_marker(frames) is True

    created_i = index_of_event_type(events, "response.created")
    incomplete_i = index_of_event_type(events, "response.incomplete")
    assert created_i < incomplete_i
    assert not any(e.get("type") == "response.completed" for e in events)
    assert not any(e.get("type") == "response.failed" for e in events)

    terminal = events[incomplete_i]
    response = terminal.get("response") or {}
    assert response.get("status") == "incomplete"
    assert (response.get("incomplete_details") or {}).get("reason") == "max_output_tokens"


@pytest.mark.anyio
async def test_responses_non_stream_finish_reason_length_returns_incomplete(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
):
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "text-max_tokens-length-stream.yaml"
    )

    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "max_output_tokens": 1,
            "input": [{"role": "user", "content": "Hello"}],
            "tool_choice": "none",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "incomplete"
    assert (data.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
    assert data["error"] is None


@pytest.mark.anyio
async def test_responses_stream_code_interpreter_tool_loop_from_vllm_cassettes(
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
            "include": ["code_interpreter_call.outputs"],
            "input": [
                {
                    "role": "user",
                    "content": "You MUST call the code_interpreter tool now. Execute: 2+2.",
                }
            ],
            "tools": [{"type": "code_interpreter"}],
            "tool_choice": "auto",
        },
    ) as resp:
        assert resp.status_code == 200
        body = await resp.aread()

    text = body.decode("utf-8", errors="replace")
    frames = parse_sse_frames(text)
    events = parse_sse_json_events(frames)
    assert sse_has_done_marker(frames) is True

    completed = extract_completed_response(events)
    ci_item = next(
        (
            item
            for item in (completed.get("output") or [])
            if isinstance(item, dict) and item.get("type") == "code_interpreter_call"
        ),
        None,
    )
    assert ci_item is not None
    outputs = ci_item.get("outputs") or []
    assert any(
        isinstance(o, dict) and o.get("type") == "logs" and str(o.get("logs", "")).strip() == "4"
        for o in outputs
    )
