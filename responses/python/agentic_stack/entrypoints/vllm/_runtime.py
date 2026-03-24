from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from loguru import logger
from starlette.routing import BaseRoute, Route

from agentic_stack.configs.builders import (
    build_runtime_config_for_integrated,
)
from agentic_stack.configs.runtime import RuntimeConfig
from agentic_stack.configs.sources import EnvSource
from agentic_stack.configs.startup import find_flag_value
from agentic_stack.entrypoints._helper_runtime import (
    HelperProcess,
    McpRuntimeSpec,
    SpawnCodeInterpreterSpec,
    build_code_interpreter_spawn_spec,
    build_mcp_runtime_cmd,
    build_mcp_runtime_spec,
    cleanup_helper_processes,
    spawn_logged_process,
    wait_for_code_interpreter_ready,
    wait_for_helper_ready,
)
from agentic_stack.entrypoints._serve_utils import (
    is_port_available,
    terminate_process,
)
from agentic_stack.entrypoints._state import get_vr_app_state, require_vr_app_state
from agentic_stack.entrypoints.vllm._adapter import (
    load_api_server_module,
    run_upstream_cli,
    suppress_native_responses_attach,
)
from agentic_stack.entrypoints.vllm._spec import IntegratedServeSpec
from agentic_stack.responses_core.store import configure_response_store
from agentic_stack.tools.ids import WEB_SEARCH_TOOL
from agentic_stack.tools.profile_resolution import profiled_builtin_requires_mcp
from agentic_stack.utils.logging import setup_logger_sinks
from agentic_stack.utils.urls import is_ready_url_host

_RESPONSES_ROUTE_FAMILY = {
    ("POST", "/v1/responses"),
    ("GET", "/v1/responses/{response_id}"),
    ("POST", "/v1/responses/{response_id}/cancel"),
}


class _IntegratedRuntimeError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = int(exit_code)


def derive_integrated_llm_api_base(*, host: str, port: int) -> str:
    return f"http://{is_ready_url_host(host)}:{port}/v1"


def _resolve_vllm_bind(spec: IntegratedServeSpec) -> tuple[str, int]:
    host = find_flag_value(spec.vllm_args, "--host") or "127.0.0.1"
    port = int(find_flag_value(spec.vllm_args, "--port") or "8000")
    return host, port


def _route_matches(route: BaseRoute, *, method: str, path: str) -> bool:
    methods = getattr(route, "methods", None)
    route_path = getattr(route, "path", None)
    return bool(methods) and method in methods and route_path == path


def _is_agentic_stack_endpoint(route: BaseRoute, endpoint: Callable[..., Any]) -> bool:
    return isinstance(route, Route) and route.endpoint is endpoint


def _assert_responses_route_ownership(app: FastAPI) -> None:
    from agentic_stack.routers import serving

    post_routes = [
        route
        for route in app.router.routes
        if _route_matches(route, method="POST", path="/v1/responses")
    ]
    if len(post_routes) != 1 or not _is_agentic_stack_endpoint(
        post_routes[0], serving.create_model_response
    ):
        raise RuntimeError("Integrated mode requires one authoritative POST /v1/responses route.")

    get_routes = [
        route
        for route in app.router.routes
        if _route_matches(route, method="GET", path="/v1/responses/{response_id}")
    ]
    if len(get_routes) != 1 or not _is_agentic_stack_endpoint(
        get_routes[0], serving.retrieve_model_response
    ):
        raise RuntimeError(
            "Integrated mode requires one authoritative GET /v1/responses/{response_id} route."
        )

    cancel_routes = [
        route
        for route in app.router.routes
        if _route_matches(route, method="POST", path="/v1/responses/{response_id}/cancel")
    ]
    if cancel_routes:
        raise RuntimeError(
            "Integrated mode must not expose POST /v1/responses/{response_id}/cancel."
        )


def _cleanup_responses_family_routes(app: FastAPI) -> None:
    from agentic_stack.routers import serving

    cleaned_routes: list[BaseRoute] = []
    for route in app.router.routes:
        matched = False
        for method, path in _RESPONSES_ROUTE_FAMILY:
            if _route_matches(route, method=method, path=path):
                matched = True
                break
        if not matched:
            cleaned_routes.append(route)
            continue
        if _is_agentic_stack_endpoint(
            route, serving.create_model_response
        ) or _is_agentic_stack_endpoint(route, serving.retrieve_model_response):
            cleaned_routes.append(route)
    app.router.routes[:] = cleaned_routes


def _install_integrated_lifespan(app: FastAPI) -> None:
    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _lifespan(inner_app: FastAPI):
        async with previous_lifespan(inner_app):
            try:
                yield
            finally:
                app_state = get_vr_app_state(inner_app)
                runtime_client = (
                    None if app_state is None else app_state.builtin_mcp_runtime_client
                )
                if runtime_client is not None:
                    await runtime_client.aclose()

    app.router.lifespan_context = _lifespan


def _build_integrated_app(
    args: Any,
    supported_tasks: Any,
    *,
    upstream_build_app: Callable[[Any, Any], FastAPI],
    runtime_config: RuntimeConfig,
) -> FastAPI:
    from agentic_stack.entrypoints.gateway._app import augment_integrated_gateway_app
    from agentic_stack.mcp.runtime_client import BuiltinMcpRuntimeClient

    with suppress_native_responses_attach():
        app = upstream_build_app(args, supported_tasks)

    if not isinstance(app, FastAPI):
        raise RuntimeError("Integrated mode expected vLLM build_app(...) to return a FastAPI app.")

    augment_integrated_gateway_app(app, runtime_config=runtime_config)
    builtin_runtime_url = (runtime_config.mcp_builtin_runtime_url or "").strip()
    if builtin_runtime_url:
        require_vr_app_state(app).builtin_mcp_runtime_client = BuiltinMcpRuntimeClient(
            base_url=builtin_runtime_url
        )
    _install_integrated_lifespan(app)

    try:
        _assert_responses_route_ownership(app)
    except RuntimeError:
        _cleanup_responses_family_routes(app)
        _assert_responses_route_ownership(app)
    return app


def _ensure_port_available(*, name: str, host: str, port: int, hint: str | None = None) -> None:
    if is_port_available(host, port):
        return
    message = f"[integrated] error: {name} port already in use: {host}:{port}"
    if hint is not None:
        message = f"{message}\n[integrated] hint: {hint}"
    raise _IntegratedRuntimeError(message, exit_code=2)


def _start_integrated_helpers(
    *,
    spec: IntegratedServeSpec,
    procs: list[HelperProcess],
    base_env: dict[str, str],
    runtime_config: RuntimeConfig,
    mcp_runtime: McpRuntimeSpec | None,
) -> None:
    helper_env = dict(base_env)
    if runtime_config.web_search_profile is not None:
        helper_env["VR_WEB_SEARCH_PROFILE"] = runtime_config.web_search_profile
    else:
        helper_env.pop("VR_WEB_SEARCH_PROFILE", None)
    if runtime_config.mcp_config_path is not None:
        helper_env["VR_MCP_CONFIG_PATH"] = runtime_config.mcp_config_path
    else:
        helper_env.pop("VR_MCP_CONFIG_PATH", None)

    if spec.code_interpreter_mode == "spawn":
        code_interpreter_spec = build_code_interpreter_spawn_spec(
            runtime_config,
            error_factory=_IntegratedRuntimeError,
            error_prefix="[integrated]",
        )
        assert isinstance(code_interpreter_spec, SpawnCodeInterpreterSpec)
        _ensure_port_available(
            name="code-interpreter",
            host="127.0.0.1",
            port=code_interpreter_spec.port,
            hint="choose another port via --responses-code-interpreter-port.",
        )
        code_interpreter = spawn_logged_process(
            log_prefix="[integrated]",
            name="code-interpreter",
            cmd=code_interpreter_spec.cmd,
            cwd=code_interpreter_spec.cwd,
            env=helper_env,
        )
        procs.append(code_interpreter)
        wait_for_code_interpreter_ready(
            error_factory=_IntegratedRuntimeError,
            error_prefix="[integrated]",
            ready_url=code_interpreter_spec.ready_url,
            startup_timeout_s=spec.code_interpreter_startup_timeout_s,
            proc=code_interpreter.proc,
        )
        logger.info(
            f"[integrated] code interpreter ready: mode=spawn port={code_interpreter_spec.port}"
        )
    elif spec.code_interpreter_mode == "external":
        ready_url = f"http://localhost:{spec.code_interpreter_port}/health"
        wait_for_code_interpreter_ready(
            error_factory=_IntegratedRuntimeError,
            error_prefix="[integrated]",
            ready_url=ready_url,
            startup_timeout_s=spec.code_interpreter_startup_timeout_s,
            proc=None,
        )
        logger.info(
            f"[integrated] code interpreter ready: mode=external port={spec.code_interpreter_port}"
        )

    if mcp_runtime is not None:
        _ensure_port_available(
            name="mcp-runtime",
            host=mcp_runtime.host,
            port=mcp_runtime.port,
        )
        mcp_runtime_proc = spawn_logged_process(
            log_prefix="[integrated]",
            name="mcp-runtime",
            cmd=build_mcp_runtime_cmd(host=mcp_runtime.host, port=mcp_runtime.port),
            env=helper_env,
        )
        procs.append(mcp_runtime_proc)
        wait_for_helper_ready(
            error_factory=_IntegratedRuntimeError,
            error_prefix="[integrated]",
            name="mcp-runtime",
            ready_url=mcp_runtime.ready_url,
            timeout_s=60.0,
            interval_s=2.0,
            proc=mcp_runtime_proc.proc,
        )
        logger.info(
            f"[integrated] mcp-runtime ready: url=http://{mcp_runtime.host}:{mcp_runtime.port}"
        )


def _start_helper_watchdog(
    *,
    procs: list[HelperProcess],
    stop_event: threading.Event,
) -> threading.Thread:
    def _watch() -> None:
        while not stop_event.is_set():
            for helper in list(procs):
                code = helper.proc.poll()
                if code is None:
                    continue
                if stop_event.is_set():
                    return
                exit_code = code or 1
                if code == 0:
                    print(
                        f"[integrated] {helper.name} exited (code=0). terminating integrated server.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[integrated] {helper.name} exited unexpectedly (code={code}). terminating integrated server.",
                        file=sys.stderr,
                    )
                stop_event.set()
                for other_helper in reversed(list(procs)):
                    if other_helper.proc is helper.proc:
                        continue
                    terminate_process(other_helper.proc, name=other_helper.name)
                os._exit(exit_code)
            time.sleep(0.25)

    thread = threading.Thread(target=_watch, name="integrated-helper-watchdog", daemon=True)
    thread.start()
    return thread


def run_integrated_serve(spec: IntegratedServeSpec) -> int:
    setup_logger_sinks(None)
    api_server = load_api_server_module()

    host, port = _resolve_vllm_bind(spec)
    env_source = EnvSource.from_env()
    mcp_runtime: McpRuntimeSpec | None = None
    helper_procs: list[HelperProcess] = []
    helper_watchdog_stop = threading.Event()
    helper_watchdog: threading.Thread | None = None
    original_build_app = api_server.build_app

    try:
        if (
            spec.mcp_port is not None
            and spec.mcp_config_path is None
            and not profiled_builtin_requires_mcp(
                tool_type=WEB_SEARCH_TOOL,
                profile_id=spec.web_search_profile,
            )
        ):
            raise _IntegratedRuntimeError(
                "[integrated] error: --responses-mcp-port requires --responses-mcp-config "
                "or a web_search profile that provisions Built-in MCP helpers.",
                exit_code=2,
            )
        cli_mcp_runtime_url = (
            None if spec.mcp_port is None else f"http://127.0.0.1:{spec.mcp_port}"
        )
        runtime_config = build_runtime_config_for_integrated(
            env=env_source,
            host=host,
            port=port,
            web_search_profile=spec.web_search_profile,
            code_interpreter_mode=spec.code_interpreter_mode,
            code_interpreter_port=spec.code_interpreter_port,
            code_interpreter_workers=spec.code_interpreter_workers,
            code_interpreter_startup_timeout_s=spec.code_interpreter_startup_timeout_s,
            mcp_config_path=spec.mcp_config_path,
            mcp_builtin_runtime_url=cli_mcp_runtime_url,
        )
        mcp_runtime = build_mcp_runtime_spec(
            runtime_config,
            error_factory=_IntegratedRuntimeError,
            error_prefix="[integrated]",
        )
        from agentic_stack.responses_core.store import get_default_response_store

        configure_response_store(runtime_config)
        asyncio.run(get_default_response_store().ensure_schema())
        _start_integrated_helpers(
            spec=spec,
            procs=helper_procs,
            base_env=env_source.environ,
            runtime_config=runtime_config,
            mcp_runtime=mcp_runtime,
        )
        if helper_procs:
            helper_watchdog = _start_helper_watchdog(
                procs=helper_procs,
                stop_event=helper_watchdog_stop,
            )

        def _patched_build_app(args: Any, supported_tasks: Any) -> FastAPI:
            return _build_integrated_app(
                args,
                supported_tasks,
                upstream_build_app=original_build_app,
                runtime_config=runtime_config,
            )

        api_server.build_app = _patched_build_app
        run_upstream_cli(spec.vllm_args)
    except _IntegratedRuntimeError as exc:
        logger.error(str(exc))
        return exc.exit_code
    except KeyboardInterrupt:
        return 130
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:
        logger.exception(f"[integrated] error: {exc!r}")
        return 1
    finally:
        helper_watchdog_stop.set()
        if helper_watchdog is not None:
            helper_watchdog.join(timeout=1.0)
        cleanup_helper_processes(helper_procs)
        api_server.build_app = original_build_app
        try:
            logger.complete()
        except Exception:
            pass
    return 0
