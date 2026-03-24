import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import pytest

from agentic_stack.configs.builders import build_runtime_config_for_standalone
from agentic_stack.configs.sources import EnvSource
from agentic_stack.tools.code_interpreter import (
    HTTP_ACLIENT,
    bind_runtime_config,
    configure_code_interpreter,
    run_code,
    start_server,
)
from agentic_stack.tools.runtime import (
    ToolRuntimeContext,
    bind_tool_runtime_context,
    get_tool_runtime_context,
)
from agentic_stack.utils.exceptions import BadInputError

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
async def code_interpreter_server() -> None:
    cache_dir = os.environ.get("VR_PYODIDE_CACHE_DIR", "").strip()
    if cache_dir:
        cache_path = Path(os.path.expanduser(cache_dir))
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
        if xdg:
            base = Path(os.path.expanduser(xdg))
        else:
            base = Path.home() / ".cache"
        cache_path = base / "agentic-stacks" / "pyodide"

    marker = cache_path / ".pyodide_version"
    if not marker.exists():
        repo_root = Path(__file__).resolve().parents[2]
        bootstrap = repo_root / "scripts" / "ci" / "bootstrap_pyodide_cache.py"
        raise RuntimeError(
            "Pyodide cache is not initialized. The code interpreter tests require Pyodide to be installed "
            "ahead of time (we do not auto-download ~400MB during tests).\n\n"
            "Bootstrap it with:\n"
            f"  VR_PYODIDE_CACHE_DIR={str(cache_path)!r} {sys.executable} {bootstrap}\n"
        )

    # Use multiple workers to cover the WorkerPool path (when supported by the runtime).
    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={
                "VR_CODE_INTERPRETER_MODE": "spawn",
                "VR_CODE_INTERPRETER_PORT": "5970",
                "VR_CODE_INTERPRETER_WORKERS": "2",
                "VR_PYODIDE_CACHE_DIR": str(cache_path),
            }
        )
    )
    configure_code_interpreter(runtime_config)
    process = await start_server(port=5970, workers=2)
    try:
        yield process
    finally:
        if process:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=10.0)
                # print("Graceful termination")
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                # print("Kill termination")
            except Exception as e:
                print(f"Error stopping code interpreter server: {repr(e)}")
        await asyncio.wait_for(process.wait(), timeout=10.0)


async def test_code_interpreter_numpy(code_interpreter_server) -> None:
    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={"VR_CODE_INTERPRETER_MODE": "external", "VR_CODE_INTERPRETER_PORT": "5970"}
        )
    )
    with bind_runtime_config(runtime_config):
        response = json.loads(await run_code("import numpy as np; np.array([1,2,3]).mean()"))
    assert response["status"] == "success"
    assert response["result"] == "2"
    assert response["stdout"] == ""
    assert response["stderr"] == ""


async def test_code_interpreter_ctypes_patch(code_interpreter_server) -> None:
    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={"VR_CODE_INTERPRETER_MODE": "external", "VR_CODE_INTERPRETER_PORT": "5970"}
        )
    )
    with bind_runtime_config(runtime_config):
        response = json.loads(await run_code('import ctypes; ctypes.CDLL(None).system(b"whoami")'))
    assert response["status"] == "exception"
    assert "'NoneType' object is not callable" in response["result"]


async def test_code_interpreter_captures_print_stdout(code_interpreter_server) -> None:
    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={"VR_CODE_INTERPRETER_MODE": "external", "VR_CODE_INTERPRETER_PORT": "5970"}
        )
    )
    with bind_runtime_config(runtime_config):
        response = json.loads(await run_code('print("P1"); print("P2"); 2+2'))
    assert response["status"] == "success"
    assert response["stdout"] == "P1\nP2\n"
    assert response["stderr"] == ""
    assert response["result"] == "4"


async def test_code_interpreter_base_eval_patch(code_interpreter_server) -> None:
    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={"VR_CODE_INTERPRETER_MODE": "external", "VR_CODE_INTERPRETER_PORT": "5970"}
        )
    )
    with bind_runtime_config(runtime_config):
        response = json.loads(
            await run_code(
                'import _pyodide; _pyodide._base.eval_code("import os; os.system(\\"whoami\\")")'
            )
        )
    assert response["status"] == "success"
    assert 'import os; os.system("whoami")' in response["result"]


async def test_run_code_reads_runtime_port_from_bound_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _fake_post(url: str, json: dict[str, str]):  # type: ignore[no-untyped-def]
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(200, text='{"status":"success"}')

    monkeypatch.setattr(HTTP_ACLIENT, "post", _fake_post)
    monkeypatch.setattr(HTTP_ACLIENT, "_transport", httpx.ASGITransport(app=None))
    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={"VR_CODE_INTERPRETER_MODE": "external", "VR_CODE_INTERPRETER_PORT": "6112"}
        )
    )

    with bind_runtime_config(runtime_config):
        response = await run_code("print('ok')")

    assert response == '{"status":"success"}'
    assert captured["url"] == "http://localhost:6112/python"
    assert captured["json"] == {"code": "print('ok')"}


async def test_run_code_rejects_disabled_runtime_config() -> None:
    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(environ={"VR_CODE_INTERPRETER_MODE": "disabled"})
    )

    with bind_runtime_config(runtime_config):
        with pytest.raises(BadInputError, match="disabled by configuration"):
            await run_code("print('blocked')")


async def test_bind_runtime_config_preserves_existing_tool_runtime_context() -> None:
    original_runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={"VR_CODE_INTERPRETER_MODE": "external", "VR_CODE_INTERPRETER_PORT": "6112"}
        )
    )
    replacement_runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={"VR_CODE_INTERPRETER_MODE": "external", "VR_CODE_INTERPRETER_PORT": "6113"}
        )
    )
    sentinel_web_search = object()

    with bind_tool_runtime_context(
        ToolRuntimeContext(
            runtime_config=original_runtime_config,
            web_search=sentinel_web_search,  # type: ignore[arg-type]
        )
    ):
        with bind_runtime_config(replacement_runtime_config):
            bound_context = get_tool_runtime_context()

    assert bound_context is not None
    assert bound_context.runtime_config.code_interpreter_port == 6113
    assert bound_context.web_search is sentinel_web_search


async def test_start_server_uses_runtime_config_startup_timeout(monkeypatch) -> None:
    import agentic_stack.tools.code_interpreter as code_interpreter_mod

    class _FakeProcess:
        def __init__(self) -> None:
            self.pid = 12345
            self.returncode: int | None = None

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            return 0 if self.returncode is None else self.returncode

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"", b"")

    process = _FakeProcess()
    perf_values = iter([0.0, 0.0, 0.0, 0.06])

    async def _fake_create_subprocess_exec(*args, **kwargs):  # type: ignore[no-untyped-def]
        return process

    async def _fake_get(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise httpx.ConnectError("not ready")

    async def _fake_sleep(*args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(
        code_interpreter_mod,
        "_get_spawn_command",
        lambda **kwargs: (["code-interpreter-server", "--port", "6001"], "/tmp"),
    )
    monkeypatch.setattr(
        code_interpreter_mod.asyncio,
        "create_subprocess_exec",
        _fake_create_subprocess_exec,
    )
    monkeypatch.setattr(code_interpreter_mod.HTTP_ACLIENT, "get", _fake_get)
    monkeypatch.setattr(code_interpreter_mod.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(code_interpreter_mod, "perf_counter", lambda: next(perf_values))

    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={
                "VR_CODE_INTERPRETER_MODE": "spawn",
                "VR_CODE_INTERPRETER_PORT": "6001",
                "VR_CODE_INTERPRETER_STARTUP_TIMEOUT": "0.05",
                "VR_PYODIDE_CACHE_DIR": "/tmp/pyodide-cache",
            }
        )
    )
    configure_code_interpreter(runtime_config)

    with pytest.raises(TimeoutError, match=r"0.05s"):
        await start_server(port=6001, workers=0)
