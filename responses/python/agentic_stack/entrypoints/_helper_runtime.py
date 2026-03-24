from __future__ import annotations

import importlib.util
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NoReturn, TypeAlias
from urllib.parse import urlsplit

from agentic_stack.configs.runtime import RuntimeConfig
from agentic_stack.entrypoints._serve_utils import (
    stream_lines,
    terminate_process,
    wait_http_ready,
)
from agentic_stack.tools.ids import WEB_SEARCH_TOOL
from agentic_stack.tools.profile_resolution import profiled_builtin_requires_mcp


@dataclass(frozen=True, slots=True)
class McpRuntimeSpec:
    host: str
    port: int
    ready_url: str


@dataclass(frozen=True, slots=True)
class DisabledCodeInterpreterSpec:
    pass


@dataclass(frozen=True, slots=True)
class ExternalCodeInterpreterSpec:
    port: int
    ready_url: str


@dataclass(frozen=True, slots=True)
class SpawnCodeInterpreterSpec:
    cmd: list[str]
    cwd: Path
    port: int
    workers: int
    ready_url: str


CodeInterpreterSpec: TypeAlias = (
    DisabledCodeInterpreterSpec | ExternalCodeInterpreterSpec | SpawnCodeInterpreterSpec
)


@dataclass(frozen=True, slots=True)
class HelperProcess:
    name: str
    proc: subprocess.Popen[str]


def _raise_mode_error(
    *,
    error_factory: Callable[..., Exception],
    message: str,
    exit_code: int = 2,
) -> NoReturn:
    raise error_factory(message, exit_code=exit_code)


def _code_interpreter_dir_from_spec(
    *,
    error_factory: Callable[..., Exception],
    error_prefix: str,
) -> Path:
    spec = importlib.util.find_spec("agentic_stack.tools.code_interpreter")
    if spec is None or not spec.submodule_search_locations:
        _raise_mode_error(
            error_factory=error_factory,
            message=(
                f"{error_prefix} error: failed to locate "
                "`agentic_stack.tools.code_interpreter` package data. "
                "This installation may be incomplete. Try reinstalling `agentic-stacks`."
            ),
        )
    return Path(spec.submodule_search_locations[0]).resolve()


def build_code_interpreter_spawn_spec(
    runtime_config: RuntimeConfig,
    *,
    error_factory: Callable[..., Exception],
    error_prefix: str,
) -> SpawnCodeInterpreterSpec:
    code_interpreter_dir = _code_interpreter_dir_from_spec(
        error_factory=error_factory,
        error_prefix=error_prefix,
    )
    pyodide_cache_dir = runtime_config.pyodide_cache_dir
    if pyodide_cache_dir is None:
        _raise_mode_error(
            error_factory=error_factory,
            message=f"{error_prefix} error: code interpreter cache directory is not configured.",
        )

    ci_bin = code_interpreter_dir / "bin" / "linux" / "x86_64" / "code-interpreter-server"
    cmd: list[str] | None = None

    if ci_bin.exists():
        cmd = [
            str(ci_bin),
            "--port",
            str(runtime_config.code_interpreter_port),
            "--pyodide-cache",
            pyodide_cache_dir,
        ]
        if (runtime_config.code_interpreter_workers or 0) > 0:
            cmd.extend(["--workers", str(runtime_config.code_interpreter_workers)])
    elif (
        runtime_config.code_interpreter_dev_bun_fallback
        and (code_interpreter_dir / "src/index.ts").exists()
    ):
        bun_bin = shutil.which("bun")
        if not bun_bin:
            _raise_mode_error(
                error_factory=error_factory,
                message=(
                    f"{error_prefix} error: VR_CODE_INTERPRETER_DEV_BUN_FALLBACK=1 "
                    "but `bun` was not found on PATH."
                ),
            )
        cmd = [
            bun_bin,
            "src/index.ts",
            "--port",
            str(runtime_config.code_interpreter_port),
            "--pyodide-cache",
            pyodide_cache_dir,
        ]
        if (runtime_config.code_interpreter_workers or 0) > 0:
            cmd.extend(["--workers", str(runtime_config.code_interpreter_workers)])
    else:
        _raise_mode_error(
            error_factory=error_factory,
            message=(
                f"{error_prefix} error: no bundled code-interpreter binary was found for this "
                "platform.\n"
                "  - On Linux x86_64 PyPI wheels, this should be present.\n"
                "  - For source checkouts, you can set VR_CODE_INTERPRETER_DEV_BUN_FALLBACK=1 "
                "and install Bun.\n"
                "  - Or disable the tool via --code-interpreter=disabled.\n"
            ),
        )

    return SpawnCodeInterpreterSpec(
        cmd=cmd,
        cwd=code_interpreter_dir,
        port=int(runtime_config.code_interpreter_port or 0),
        workers=int(runtime_config.code_interpreter_workers or 0),
        ready_url=f"http://localhost:{runtime_config.code_interpreter_port}/health",
    )


def build_mcp_runtime_spec(
    runtime_config: RuntimeConfig,
    *,
    error_factory: Callable[..., Exception],
    error_prefix: str,
) -> McpRuntimeSpec | None:
    if runtime_config.mcp_config_path is None and not profiled_builtin_requires_mcp(
        tool_type=WEB_SEARCH_TOOL,
        profile_id=runtime_config.web_search_profile,
    ):
        return None

    runtime_url = runtime_config.mcp_builtin_runtime_url
    if runtime_url is None:
        _raise_mode_error(
            error_factory=error_factory,
            message=f"{error_prefix} error: Built-in MCP runtime URL is not configured.",
        )

    try:
        parsed = urlsplit(runtime_url)
    except ValueError as exc:
        _raise_mode_error(
            error_factory=error_factory,
            message=f"{error_prefix} error: invalid VR_MCP_BUILTIN_RUNTIME_URL={runtime_url!r}: {exc}",
        )

    if parsed.scheme.lower() != "http":
        _raise_mode_error(
            error_factory=error_factory,
            message=f"{error_prefix} error: VR_MCP_BUILTIN_RUNTIME_URL must use `http://`.",
        )

    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost"}:
        _raise_mode_error(
            error_factory=error_factory,
            message=(
                f"{error_prefix} error: VR_MCP_BUILTIN_RUNTIME_URL must use loopback host "
                "(`127.0.0.1` or `localhost`)."
            ),
        )

    try:
        parsed_port = parsed.port
    except ValueError as exc:
        _raise_mode_error(
            error_factory=error_factory,
            message=f"{error_prefix} error: invalid VR_MCP_BUILTIN_RUNTIME_URL={runtime_url!r}: {exc}",
        )

    if parsed_port is None:
        _raise_mode_error(
            error_factory=error_factory,
            message=f"{error_prefix} error: VR_MCP_BUILTIN_RUNTIME_URL must include an explicit port.",
        )

    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        _raise_mode_error(
            error_factory=error_factory,
            message=(
                f"{error_prefix} error: VR_MCP_BUILTIN_RUNTIME_URL must not include path, "
                "query, or fragment."
            ),
        )

    port = int(parsed_port)
    return McpRuntimeSpec(
        host="127.0.0.1",
        port=port,
        ready_url=f"http://127.0.0.1:{port}/health",
    )


def build_mcp_runtime_cmd(*, host: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "agentic_stack.entrypoints.mcp_runtime:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "error",
    ]


def spawn_logged_process(
    *,
    log_prefix: str,
    name: str,
    cmd: list[str],
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> HelperProcess:
    print(f"{log_prefix} starting {name}: {shlex.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    if proc.stdout is not None:
        threading.Thread(target=stream_lines, args=(f"{name}| ", proc.stdout), daemon=True).start()
    return HelperProcess(name=name, proc=proc)


def wait_for_helper_ready(
    *,
    error_factory: Callable[..., Exception],
    error_prefix: str,
    name: str,
    display_name: str | None = None,
    ready_url: str,
    timeout_s: float,
    interval_s: float,
    proc: subprocess.Popen[str] | None = None,
    check_json: Callable[[object | None], bool] | None = None,
) -> None:
    helper_name = display_name or name
    try:
        wait_http_ready(
            name=name,
            url=ready_url,
            timeout_s=timeout_s,
            interval_s=interval_s,
            check_json=check_json,
            abort_proc=proc,
        )
    except Exception as exc:
        if proc is not None:
            code = proc.poll()
            if code is not None:
                _raise_mode_error(
                    error_factory=error_factory,
                    message=(
                        f"{error_prefix} {helper_name} exited during startup "
                        f"(code={code}). shutting down."
                    ),
                    exit_code=code or 1,
                )
        _raise_mode_error(
            error_factory=error_factory,
            message=f"{error_prefix} error: {helper_name} readiness failed: {exc!r}",
            exit_code=1,
        )


def wait_for_code_interpreter_ready(
    *,
    error_factory: Callable[..., Exception],
    error_prefix: str,
    ready_url: str,
    startup_timeout_s: float,
    proc: subprocess.Popen[str] | None = None,
) -> None:
    wait_for_helper_ready(
        error_factory=error_factory,
        error_prefix=error_prefix,
        name="code-interpreter",
        display_name="code interpreter",
        ready_url=ready_url,
        timeout_s=startup_timeout_s,
        interval_s=5.0,
        proc=proc,
        check_json=lambda payload: isinstance(payload, dict)
        and bool(payload.get("pyodide_loaded")),
    )


def cleanup_helper_processes(procs: list[HelperProcess]) -> None:
    for helper in reversed(procs):
        terminate_process(helper.proc, name=helper.name)
