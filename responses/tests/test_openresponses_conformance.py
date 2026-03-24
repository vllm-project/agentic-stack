from __future__ import annotations

import httpx
import pytest
from sse_test_utils import extract_completed_response, parse_sse_frames, parse_sse_json_events

from agentic_stack.entrypoints import llm as mock_llm


def _extract_completed_response(sse_text: str) -> dict:
    frames = parse_sse_frames(sse_text)
    events = parse_sse_json_events(frames)
    return extract_completed_response(events)


def _assert_openresponses_required_shape(resp: dict) -> None:
    # A minimal, schema-focused subset aligned with OpenResponses harness failures.
    # We avoid full OpenAPI validation here, but pin the invariants the harness is strict about.
    for key in (
        "id",
        "object",
        "created_at",
        "completed_at",
        "status",
        "model",
        "output",
        "tools",
        "tool_choice",
        "truncation",
        "text",
        "top_p",
        "temperature",
        "presence_penalty",
        "frequency_penalty",
        "top_logprobs",
        "store",
        "background",
        "service_tier",
        "parallel_tool_calls",
        "reasoning",
        "metadata",
    ):
        assert key in resp, f"missing key: {key}"

    assert resp["object"] == "response"
    assert isinstance(resp["created_at"], int)
    assert resp["status"] in {
        "completed",
        "in_progress",
        "failed",
        "incomplete",
        "queued",
        "cancelled",
    }

    assert isinstance(resp["completed_at"], int), (
        "completed_at should be set for completed responses"
    )

    assert isinstance(resp["tools"], list)
    assert resp["truncation"] in {"auto", "disabled"}

    text = resp["text"]
    assert isinstance(text, dict)
    assert isinstance(text.get("format"), dict)
    assert text["format"].get("type") == "text"

    assert isinstance(resp["presence_penalty"], (int, float))
    assert isinstance(resp["frequency_penalty"], (int, float))
    assert isinstance(resp["top_logprobs"], int)
    assert isinstance(resp["store"], bool)
    assert isinstance(resp["background"], bool)
    assert isinstance(resp["service_tier"], str)
    assert isinstance(resp["metadata"], dict)

    output = resp["output"]
    assert isinstance(output, list)
    assert output, "expected at least one output item"

    reasoning = next(
        (o for o in output if isinstance(o, dict) and o.get("type") == "reasoning"), None
    )
    if reasoning is not None:
        assert reasoning.get("summary") == []
        assert "status" not in reasoning
        assert "encrypted_content" not in reasoning
        content = reasoning.get("content")
        assert isinstance(content, list), "reasoning.content must be an array when present"
        assert all(isinstance(p, dict) and p.get("type") == "reasoning_text" for p in content)

    message = next((o for o in output if isinstance(o, dict) and o.get("type") == "message"), None)
    assert message is not None, "expected an assistant message output item"
    content = message.get("content") or []
    assert isinstance(content, list)
    text_parts = [p for p in content if isinstance(p, dict) and p.get("type") == "output_text"]
    assert text_parts, "expected at least one output_text content part"
    for part in text_parts:
        assert isinstance(part.get("annotations"), list)
        assert isinstance(part.get("logprobs"), list)


@pytest.mark.anyio
async def test_openresponses_conformance_basic_text_response_shape(
    patched_gateway_clients,
    gateway_client: httpx.AsyncClient,
    cassette_replayer_factory,
):
    mock_llm.app.state.agentic_stack.cassette_replayer = cassette_replayer_factory(
        "text-single-stream.yaml"
    )

    resp = await gateway_client.post(
        "/v1/responses",
        json={
            "model": "some-model",
            "stream": False,
            "input": [
                {"type": "message", "role": "user", "content": "Say hello in exactly 3 words."}
            ],
            "tool_choice": "none",
        },
    )
    assert resp.status_code == 200
    _assert_openresponses_required_shape(resp.json())


@pytest.mark.anyio
async def test_openresponses_conformance_streaming_response_shape(
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
            "input": [{"type": "message", "role": "user", "content": "Count from 1 to 5."}],
            "tool_choice": "none",
        },
    ) as resp:
        assert resp.status_code == 200
        body = await resp.aread()

    text = body.decode("utf-8", errors="replace")
    completed = _extract_completed_response(text)
    _assert_openresponses_required_shape(completed)
