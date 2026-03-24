from typing import AsyncGenerator

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from agentic_stack.entrypoints._state import require_vr_app_state
from agentic_stack.lm import LMEngine
from agentic_stack.observability.metrics import get_route_label, instrument_sse_stream
from agentic_stack.responses_core.store import get_default_response_store
from agentic_stack.types.openai import OpenAIResponsesResponse, vLLMResponsesRequest
from agentic_stack.utils.exceptions import ResponsesAPIError

router = APIRouter()


async def _empty_async_generator():
    """Returns an empty asynchronous generator."""
    return
    # This line is never reached, but makes it an async generator
    yield


async def create_model_response(
    request: Request,
    # session: Annotated[AsyncSession, Depends(yield_async_session)],
    body: vLLMResponsesRequest,
) -> Response:
    # as_responses_chunk()
    app_state = require_vr_app_state(request.app)
    builtin_mcp_runtime_client = app_state.builtin_mcp_runtime_client
    runtime_config = app_state.runtime_config
    if runtime_config is None:
        raise RuntimeError("agentic_stack runtime config is not initialized.")
    engine = LMEngine(
        body=body,
        builtin_mcp_runtime_client=builtin_mcp_runtime_client,
        runtime_config=runtime_config,
    )
    if body.stream:
        agen: AsyncGenerator[str, None] = await engine.run()
        agen = instrument_sse_stream(
            route=get_route_label(request),
            agen=agen,
        )
        try:
            # Get the first chunk outside of the loop so that errors can be raised immediately
            # Otherwise, streaming requests will always return 200
            chunk = await anext(agen)
        except StopAsyncIteration:
            return StreamingResponse(
                content=_empty_async_generator(),
                status_code=200,
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no"},
            )

        async def _generate():
            nonlocal chunk
            yield chunk
            async for chunk in agen:
                yield chunk

        response = StreamingResponse(
            content=_generate(),
            status_code=200,
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )
    else:
        response = await engine.run()
    return response


async def retrieve_model_response(
    response_id: str,
) -> OpenAIResponsesResponse:
    stored = await get_default_response_store().get(response_id=response_id)
    if stored is None:
        raise ResponsesAPIError(
            f"No response found with id '{response_id}'.",
            status_code=404,
            param="response_id",
            code="response_not_found",
        )
    return stored.payload().response


def install_routes(router: APIRouter) -> None:
    """Register the Responses API routes on the provided router."""
    router.add_api_route(
        "/v1/responses",
        create_model_response,
        methods=["POST"],
        summary="Create a model response.",
        description=(
            "Creates a model response. "
            "Provide text or image inputs to generate text or JSON outputs. "
            "Have the model call your own custom code or use built-in tools like code interpreter."
        ),
    )
    router.add_api_route(
        "/v1/responses/{response_id}",
        retrieve_model_response,
        methods=["GET"],
        summary="Retrieve a model response.",
        description="Retrieves a stored model response by its response ID.",
        response_model=OpenAIResponsesResponse,
    )


install_routes(router)
