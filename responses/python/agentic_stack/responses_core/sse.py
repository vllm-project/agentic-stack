from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import Protocol


class _ResponsesChunk(Protocol):
    type: str

    def as_responses_chunk(self) -> str: ...


DONE_MARKER = "data: [DONE]\n\n"
TERMINAL_EVENT_TYPES = {"response.completed", "response.failed"}


async def stream_responses_sse(events: AsyncIterable[_ResponsesChunk]) -> AsyncIterator[str]:
    """Encode typed Responses stream events into SSE frames, including the spec terminal marker.

    Layer-3 contract ownership:
    - The router should only stream the returned strings.
    - The terminal `[DONE]` marker is appended here (spec-first), not in the router.
    - Operational truth: recorded OpenAI Responses streams in this repo omit `[DONE]`; we still emit it for
      Open Responses spec conformance. Keep this divergence noted here (single point of truth).
    """

    done_emitted = False
    async for event in events:
        yield event.as_responses_chunk()
        if not done_emitted and event.type in TERMINAL_EVENT_TYPES:
            yield DONE_MARKER
            done_emitted = True

    # Defensive: if upstream ended without an explicit terminal lifecycle event, still close the SSE stream
    # with the spec marker to avoid hanging clients.
    if not done_emitted:
        yield DONE_MARKER
