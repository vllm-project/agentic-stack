from fastapi import APIRouter, Request
from fastapi.responses import Response

from agentic_api.core.proxy import ProxyClientManager, proxy_responses

router = APIRouter()


@router.post("/v1/responses")
async def create_response(request: Request) -> Response:
    runtime_config = request.app.state.runtime_config
    proxy_client_manager: ProxyClientManager = request.app.state.proxy_client_manager
    return await proxy_responses(
        request=request,
        runtime_config=runtime_config,
        proxy_client_manager=proxy_client_manager,
    )
