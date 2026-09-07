from __future__ import annotations

import os
import signal
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from agentic_api.binary import (
    PackagedBinaryNotFoundError,
    find_active_environment_executable,
    find_packaged_binary,
)
from agentic_api.cli import ServeOptions
from agentic_api.compatibility import SUPPORTED_VLLM_VERSION
from agentic_api.process import ChildResult, ProcessSupervisor, ShutdownRequested, wait_for_vllm_ready


READY_INTERVAL_S = 2.0


def run_serve(options: ServeOptions) -> int:
    supervisor = ProcessSupervisor()
    signal_exit_code: int | None = None
    previous_handlers: list[tuple[int, object]] = []

    def begin_shutdown(signum: int, _frame: object) -> None:
        nonlocal signal_exit_code
        if signal_exit_code is None:
            signal_exit_code = _signal_exit_code(signum)
        supervisor.request_shutdown(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers.append((signum, signal.signal(signum, begin_shutdown)))

    try:
        if options.mode == "local":
            result = _run_local_mode(supervisor, options)
        elif options.mode == "remote":
            result = _run_remote_mode(supervisor, options)
        else:
            raise ValueError(f"unsupported serve mode: {options.mode}")

        if signal_exit_code is not None:
            return signal_exit_code
        return _normalize_exit_code(result)
    except ShutdownRequested:
        if signal_exit_code is not None:
            return signal_exit_code
        raise
    except KeyboardInterrupt:
        signal_exit_code = _signal_exit_code(signal.SIGINT)
        supervisor.request_shutdown(signal.SIGINT)
        return signal_exit_code
    except (OSError, PackagedBinaryNotFoundError, RuntimeError, ValueError) as error:
        if signal_exit_code is not None:
            return signal_exit_code
        print(str(error), file=sys.stderr)
        return 1
    finally:
        supervisor.terminate_all(options.shutdown_timeout_s)
        for signum, previous in previous_handlers:
            signal.signal(signum, previous)


def _run_local_mode(supervisor: ProcessSupervisor, options: ServeOptions) -> ChildResult:
    if options.model is None:
        raise ValueError("--model is required in local mode")
    if sys.platform != "linux":
        raise RuntimeError(
            "agentic-api local mode is currently supported only on Linux; "
            "use remote mode on this platform"
        )

    rust_binary = find_packaged_binary("agentic-server")
    vllm_path = find_active_environment_executable("vllm")
    installed_version = _installed_vllm_version()
    if installed_version != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"agentic-api local mode requires vllm=={SUPPORTED_VLLM_VERSION}; found {installed_version}"
        )

    vllm_api_key = os.environ.get(options.vllm_api_key_env)
    vllm_url = f"http://127.0.0.1:{options.vllm_port}"

    vllm_process = supervisor.start(
        [
            str(vllm_path),
            "serve",
            options.model,
            *options.vllm_args,
            "--host",
            "127.0.0.1",
            "--port",
            str(options.vllm_port),
        ],
        _vllm_environment(vllm_api_key),
    )
    wait_for_vllm_ready(
        base_url=vllm_url,
        api_key=vllm_api_key,
        process=vllm_process,
        timeout=options.startup_timeout_s,
        interval=READY_INTERVAL_S,
        shutdown_requested=supervisor.shutdown_requested,
    )

    supervisor.start(
        _rust_command(options, vllm_url, binary=rust_binary),
        _rust_environment(options.gateway_api_key_env, vllm_api_key),
    )
    return supervisor.wait_for_failure()


def _run_remote_mode(supervisor: ProcessSupervisor, options: ServeOptions) -> ChildResult:
    if options.vllm_base_url is None:
        raise ValueError("--vllm-base-url is required in remote mode")

    supervisor.start(
        _rust_command(options, options.vllm_base_url),
        _rust_environment(options.gateway_api_key_env, None),
    )
    return supervisor.wait_for_failure()


def _rust_command(options: ServeOptions, upstream_base_url: str, *, binary: Path | None = None) -> list[str]:
    binary = binary or find_packaged_binary("agentic-server")
    return [
        str(binary),
        "--llm-api-base",
        upstream_base_url,
        "--llm-ready-timeout-s",
        str(options.startup_timeout_s),
        "--gateway-host",
        options.host,
        "--gateway-port",
        str(options.port),
    ]


def _rust_environment(gateway_api_key_env: str, api_key_override: str | None) -> dict[str, str]:
    env = os.environ.copy()
    gateway_api_key = api_key_override if api_key_override is not None else os.environ.get(gateway_api_key_env)
    if gateway_api_key is not None:
        env["OPENAI_API_KEY"] = gateway_api_key
    elif gateway_api_key_env != "OPENAI_API_KEY":
        env.pop("OPENAI_API_KEY", None)
    return env


def _vllm_environment(api_key: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if api_key is not None:
        env["VLLM_API_KEY"] = api_key
    else:
        env.pop("VLLM_API_KEY", None)
    return env


def _installed_vllm_version() -> str:
    try:
        return version("vllm")
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"agentic-api local mode requires vllm=={SUPPORTED_VLLM_VERSION}; install the [local] extra first"
        ) from error


def _normalize_exit_code(result: ChildResult) -> int:
    if result.returncode == 0:
        print(f"{result.name} exited unexpectedly with status 0", file=sys.stderr)
        return 1
    if result.returncode >= 0:
        return result.returncode
    return _signal_exit_code(-result.returncode)


def _signal_exit_code(signum: int) -> int:
    return 128 + int(signum)
