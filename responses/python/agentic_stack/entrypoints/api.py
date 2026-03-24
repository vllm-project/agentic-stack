import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from loguru import logger

from agentic_stack.configs.builders import build_runtime_config_for_standalone
from agentic_stack.configs.sources import EnvSource
from agentic_stack.entrypoints._state import require_vr_app_state
from agentic_stack.entrypoints.gateway._app import (
    activate_gateway_runtime,
    augment_standalone_gateway_app,
)
from agentic_stack.mcp.runtime_client import BuiltinMcpRuntimeClient
from agentic_stack.observability.tracing import configure_tracing
from agentic_stack.responses_core.store import get_default_response_store
from agentic_stack.routers import upstream_proxy
from agentic_stack.tools.code_interpreter import start_server
from agentic_stack.utils.io import HTTP_ACLIENT
from agentic_stack.utils.logging import setup_logger_sinks, suppress_logging_handlers

# Setup logging
setup_logger_sinks(None)
suppress_logging_handlers(["uvicorn", "litellm", "pottery"], True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state = require_vr_app_state(app)
    runtime_config = app_state.runtime_config
    if runtime_config is None:
        runtime_config = build_runtime_config_for_standalone(env=EnvSource.from_env())
    activate_gateway_runtime(app, runtime_config=runtime_config)
    logger.info(f"Using runtime config: {runtime_config}")

    tracing_shutdown = configure_tracing(runtime_config, app)

    # Ensure the ResponseStore schema exists.
    #
    # Multi-worker policy:
    # - `agentic-stacks serve` initializes the schema once in the supervisor and sets `VR_DB_SCHEMA_READY=1`
    #   for all workers. In that mode this call is a cheap no-op.
    # - If you start the gateway without `agentic-stacks serve` and set `VR_WORKERS > 1` with SQLite,
    #   schema init is not safe (race) and `ensure_schema()` will raise with guidance.
    await get_default_response_store().ensure_schema()

    app_state.builtin_mcp_runtime_client = None
    if (
        runtime_config.mcp_builtin_runtime_url is not None
        and runtime_config.mcp_builtin_runtime_url.strip()
    ):
        app_state.builtin_mcp_runtime_client = BuiltinMcpRuntimeClient(
            base_url=runtime_config.mcp_builtin_runtime_url.strip(),
        )
    if app_state.proxy_client_manager is None:
        app_state.proxy_client_manager = upstream_proxy.ProxyClientManager()

    if runtime_config.code_interpreter_mode == "spawn":
        if runtime_config.gateway_workers > 1:
            raise RuntimeError(
                "VR_CODE_INTERPRETER_MODE=spawn is not allowed when VR_WORKERS > 1. "
                "Use VR_CODE_INTERPRETER_MODE=external (recommended with Gunicorn), "
                "or run `agentic-stacks serve` to supervise a single shared code-interpreter process."
            )
        app_state.code_interpreter_process = await start_server(
            port=runtime_config.code_interpreter_port,
            workers=runtime_config.code_interpreter_workers or 0,
        )

    yield
    logger.info("Shutting down...")

    tracing_shutdown()

    # Shutdown code interpreter server
    code_interpreter_process = app_state.code_interpreter_process
    if code_interpreter_process:
        logger.info("Stopping code interpreter server...")
        try:
            code_interpreter_process.terminate()
            await asyncio.wait_for(code_interpreter_process.wait(), timeout=10.0)
            logger.info("Code interpreter server stopped.")
        except asyncio.TimeoutError:
            logger.warning("Code interpreter server did not stop gracefully, forcing kill...")
            code_interpreter_process.kill()
            await code_interpreter_process.wait()
        except Exception as e:
            logger.warning(f"Error stopping code interpreter server: {repr(e)}")

    runtime_client = app_state.builtin_mcp_runtime_client
    if runtime_client is not None:
        try:
            await runtime_client.aclose()
        except Exception as e:
            logger.warning(f"Error closing Built-in MCP runtime client: {repr(e)}")

    # Close DB connection
    # NOTE: the DB engine is cached for the process lifetime; explicit disposal is not required here.

    # Close HTTPX clients
    await HTTP_ACLIENT.aclose()
    proxy_client_manager = app_state.proxy_client_manager
    if proxy_client_manager is not None:
        await proxy_client_manager.aclose()
    # Ensure Loguru's background queue (enqueue=True) is fully drained before process exit.
    # Without this, interactive `Ctrl+C` shutdown can require a second interrupt.
    try:
        logger.complete()
    except Exception as e:
        logger.warning(f"Failed to flush logger queue: {repr(e)}")
    logger.info("Shutdown complete.")


app = FastAPI(
    title="vllm Responses API",
    logger=logger,
    default_response_class=ORJSONResponse,  # Should be faster
    openapi_url="/public/openapi.json",
    docs_url="/public/docs",
    redoc_url="/public/redoc",
    # license_info={
    #     "name": "Apache 2.0",
    #     "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    # },
    # servers=[dict(url="https://api.jamaibase.com")],
    lifespan=lifespan,
)
augment_standalone_gateway_app(
    app,
    include_upstream_proxy=True,
    include_metrics_route=True,
    include_cors=True,
    customize_openapi=True,
)


@app.get("/health", tags=["Health"])
async def health() -> ORJSONResponse:
    """Health check."""
    return ORJSONResponse(status_code=200, content={})


if __name__ == "__main__":
    raise SystemExit(
        "Direct execution of agentic_stack.entrypoints.api is unsupported. "
        "Use `agentic-stacks serve` or `vllm serve --responses`."
    )
