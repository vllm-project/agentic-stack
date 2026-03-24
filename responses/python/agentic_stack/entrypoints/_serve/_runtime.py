from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

from loguru import logger

from agentic_stack.entrypoints._helper_runtime import (
    HelperProcess,
    SpawnCodeInterpreterSpec,
    build_mcp_runtime_cmd,
    cleanup_helper_processes,
    spawn_logged_process,
    wait_for_code_interpreter_ready,
    wait_for_helper_ready,
)
from agentic_stack.entrypoints._serve._spec import (
    DisabledCodeInterpreterSpec,
    ServeSpec,
)
from agentic_stack.entrypoints._serve_utils import (
    cleanup_prometheus_multiproc_dir,
    cleanup_stale_prometheus_multiproc_dirs,
    create_prometheus_multiproc_dir,
    is_port_available,
    wait_http_ready,
)
from agentic_stack.responses_core.store import (
    configure_response_store,
    get_default_response_store,
)
from agentic_stack.utils.logging import setup_logger_sinks
from agentic_stack.utils.urls import is_ready_url_host


class _ServeRuntimeError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = int(exit_code)


def _build_gateway_worker_env(
    *,
    spec: ServeSpec,
    prometheus_multiproc_dir: Path | None,
) -> dict[str, str]:
    gateway_env = dict(os.environ)
    runtime_config = spec.runtime_config

    gateway_env["VR_LLM_API_BASE"] = spec.upstream.base_url
    gateway_env["VR_HOST"] = spec.gateway.host
    gateway_env["VR_PORT"] = str(spec.gateway.port)
    gateway_env["VR_WORKERS"] = str(spec.gateway.workers)
    gateway_env["VR_DB_SCHEMA_READY"] = "1"

    if runtime_config.web_search_profile is not None:
        gateway_env["VR_WEB_SEARCH_PROFILE"] = runtime_config.web_search_profile
    else:
        gateway_env.pop("VR_WEB_SEARCH_PROFILE", None)

    if runtime_config.mcp_config_path is not None:
        gateway_env["VR_MCP_CONFIG_PATH"] = runtime_config.mcp_config_path
    else:
        gateway_env.pop("VR_MCP_CONFIG_PATH", None)

    if spec.mcp_runtime is not None:
        gateway_env["VR_MCP_BUILTIN_RUNTIME_URL"] = (
            f"http://{spec.mcp_runtime.host}:{spec.mcp_runtime.port}"
        )
    else:
        gateway_env.pop("VR_MCP_BUILTIN_RUNTIME_URL", None)

    if prometheus_multiproc_dir is not None:
        gateway_env["PROMETHEUS_MULTIPROC_DIR"] = str(prometheus_multiproc_dir)

    if isinstance(spec.code_interpreter, DisabledCodeInterpreterSpec):
        gateway_env["VR_CODE_INTERPRETER_MODE"] = "disabled"
        gateway_env.pop("VR_CODE_INTERPRETER_PORT", None)
        gateway_env.pop("VR_CODE_INTERPRETER_WORKERS", None)
    else:
        gateway_env["VR_CODE_INTERPRETER_MODE"] = "external"
        gateway_env["VR_CODE_INTERPRETER_PORT"] = str(spec.code_interpreter.port)
        gateway_env["VR_CODE_INTERPRETER_WORKERS"] = str(spec.code_interpreter_workers)

    return gateway_env


def run_serve_spec(spec: ServeSpec) -> int:
    setup_logger_sinks(None)
    procs: list[HelperProcess] = []
    cleaned_up = False
    prometheus_multiproc_dir: Path | None = None

    def _cleanup() -> None:
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True
        cleanup_helper_processes(procs)
        if prometheus_multiproc_dir is not None:
            cleanup_prometheus_multiproc_dir(prometheus_multiproc_dir)

    previous_signal_handlers: dict[int, object] = {}

    def _install_signal_handlers() -> None:
        def _handler(signum: int, frame) -> None:  # type: ignore[no-untyped-def]
            _ = frame
            try:
                sig_name = signal.Signals(signum).name
            except Exception:
                sig_name = str(signum)
            print(f"[serve] received {sig_name}. shutting down.", file=sys.stderr)
            raise SystemExit(128 + signum)

        for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
            if sig is None:
                continue
            previous_signal_handlers[int(sig)] = signal.getsignal(sig)
            signal.signal(sig, _handler)

    def _restore_signal_handlers() -> None:
        for signum, handler in previous_signal_handlers.items():
            try:
                signal.signal(signum, handler)  # type: ignore[arg-type]
            except Exception:
                continue

    try:
        _install_signal_handlers()

        for line in spec.notices:
            logger.info(line)

        if spec.metrics.enabled:
            cleanup_stale_prometheus_multiproc_dirs()
            prometheus_multiproc_dir = create_prometheus_multiproc_dir(supervisor_pid=os.getpid())

        try:
            configure_response_store(runtime_config=spec.runtime_config)
            asyncio.run(get_default_response_store().ensure_schema())
        except Exception as e:
            logger.error(f"[serve] failed to initialize DB schema: {e!r}")
            return 2

        if not is_port_available(spec.gateway.host, spec.gateway.port):
            logger.error(
                f"[serve] gateway port already in use: {spec.gateway.host}:{spec.gateway.port}"
            )
            logger.info("[serve] hint: choose another port via --gateway-port.")
            return 2

        if isinstance(spec.code_interpreter, SpawnCodeInterpreterSpec):
            if not is_port_available("127.0.0.1", spec.code_interpreter.port):
                logger.error(
                    "[serve] code-interpreter port already in use: "
                    f"127.0.0.1:{spec.code_interpreter.port}"
                )
                logger.info("[serve] hint: choose another port via --code-interpreter-port.")
                return 2

        if spec.mcp_runtime is not None and not is_port_available(
            spec.mcp_runtime.host, spec.mcp_runtime.port
        ):
            logger.error(
                "[serve] error: mcp-runtime port already in use: "
                f"{spec.mcp_runtime.host}:{spec.mcp_runtime.port}"
            )
            return 2

        wait_http_ready(
            name="upstream",
            url=spec.upstream.ready_url,
            timeout_s=spec.timeouts.upstream_ready_timeout_s,
            interval_s=spec.timeouts.upstream_ready_interval_s,
            headers=spec.upstream.headers,
        )
        upstream_base_url = spec.upstream.base_url
        logger.info(f"[serve] upstream ready: {upstream_base_url}")

        if isinstance(spec.code_interpreter, DisabledCodeInterpreterSpec):
            pass
        else:
            code_interpreter_proc: HelperProcess | None = None
            if isinstance(spec.code_interpreter, SpawnCodeInterpreterSpec):
                code_interpreter_proc = spawn_logged_process(
                    log_prefix="[serve]",
                    name="code-interpreter",
                    cmd=spec.code_interpreter.cmd,
                    cwd=spec.code_interpreter.cwd,
                )
                procs.append(code_interpreter_proc)

            wait_for_code_interpreter_ready(
                error_factory=_ServeRuntimeError,
                error_prefix="[serve]",
                ready_url=spec.code_interpreter.ready_url,
                startup_timeout_s=spec.timeouts.code_interpreter_startup_timeout_s,
                proc=None if code_interpreter_proc is None else code_interpreter_proc.proc,
            )
            mode = (
                "spawn"
                if isinstance(spec.code_interpreter, SpawnCodeInterpreterSpec)
                else "external"
            )
            logger.info(
                f"[serve] code interpreter ready: mode={mode} port={spec.code_interpreter.port}"
            )

        if spec.mcp_runtime is not None:
            mcp_runtime_env = dict(os.environ)
            if spec.runtime_config.web_search_profile is not None:
                mcp_runtime_env["VR_WEB_SEARCH_PROFILE"] = spec.runtime_config.web_search_profile
            else:
                mcp_runtime_env.pop("VR_WEB_SEARCH_PROFILE", None)
            if spec.runtime_config.mcp_config_path is not None:
                mcp_runtime_env["VR_MCP_CONFIG_PATH"] = spec.runtime_config.mcp_config_path
            else:
                mcp_runtime_env.pop("VR_MCP_CONFIG_PATH", None)
            mcp_runtime_proc = spawn_logged_process(
                log_prefix="[serve]",
                name="mcp-runtime",
                cmd=build_mcp_runtime_cmd(
                    host=spec.mcp_runtime.host,
                    port=spec.mcp_runtime.port,
                ),
                env=mcp_runtime_env,
            )
            procs.append(mcp_runtime_proc)

            wait_for_helper_ready(
                error_factory=_ServeRuntimeError,
                error_prefix="[serve]",
                name="mcp-runtime",
                ready_url=spec.mcp_runtime.ready_url,
                timeout_s=60.0,
                interval_s=2.0,
                proc=mcp_runtime_proc.proc,
            )

        gateway_env = _build_gateway_worker_env(
            spec=spec,
            prometheus_multiproc_dir=prometheus_multiproc_dir,
        )

        gateway_cmd = [
            sys.executable,
            "-m",
            "gunicorn",
            "--config",
            str(Path(__file__).resolve().parents[1] / "gunicorn_conf.py"),
            "--bind",
            f"{spec.gateway.host}:{spec.gateway.port}",
            "--workers",
            str(spec.gateway.workers),
            "--worker-class",
            "uvicorn.workers.UvicornWorker",
            "agentic_stack.entrypoints.api:app",
        ]

        gateway_proc = spawn_logged_process(
            log_prefix="[serve]",
            name="gateway",
            cmd=gateway_cmd,
            env=gateway_env,
        )
        procs.append(gateway_proc)

        wait_http_ready(
            name="gateway",
            url=f"http://{is_ready_url_host(spec.gateway.host)}:{spec.gateway.port}/health",
            timeout_s=60.0,
            interval_s=2.0,
            abort_proc=gateway_proc.proc,
        )

        ready_bind = f"{spec.gateway.host}:{spec.gateway.port}"
        if spec.gateway.host in {"0.0.0.0", "::"}:
            ready_local = f"http://127.0.0.1:{spec.gateway.port}/v1/responses"
            logger.info(f"[serve] ready: gateway_bind={ready_bind} endpoint={ready_local}")
        else:
            logger.info(
                f"[serve] ready: gateway=http://{spec.gateway.host}:{spec.gateway.port}/v1/responses"
            )

        return gateway_proc.proc.wait()
    except _ServeRuntimeError as exc:
        logger.error(str(exc))
        return exc.exit_code
    except KeyboardInterrupt:
        return 130
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:
        logger.exception(f"[serve] error: {exc!r}")
        return 1
    finally:
        _cleanup()
        _restore_signal_handlers()
        try:
            logger.complete()
        except Exception:
            pass
