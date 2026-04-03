from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_api.config.runtime import RuntimeConfig
from agentic_api.core.proxy import ProxyClientManager
from agentic_api.routers import responses


def create_app(runtime_config: RuntimeConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime_config = runtime_config
        app.state.proxy_client_manager = ProxyClientManager()
        yield
        await app.state.proxy_client_manager.aclose()

    app = FastAPI(
        title="Agentic API",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(responses.router)
    return app
