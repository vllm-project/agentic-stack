from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from agentic_stack.configs.sources import EnvSource
from agentic_stack.configs.startup import find_flag_value
from agentic_stack.entrypoints._serve_utils import (
    cleanup_prometheus_multiproc_dir,
    cleanup_stale_prometheus_multiproc_dirs,
    create_prometheus_multiproc_dir,
    is_port_available,
    wait_http_ready,
)
from agentic_stack.utils.urls import is_ready_url_host


def test_find_flag_value_supports_space_and_equals() -> None:
    args = ["model", "--port", "8456", "--host=0.0.0.0", "--dtype", "auto"]
    assert find_flag_value(args, "--port") == "8456"
    assert find_flag_value(args, "--host") == "0.0.0.0"
    assert find_flag_value(args, "--dtype") == "auto"
    assert find_flag_value(args, "--missing") is None


def test_is_ready_url_host() -> None:
    assert is_ready_url_host("0.0.0.0") == "127.0.0.1"
    assert is_ready_url_host("::") == "127.0.0.1"
    assert is_ready_url_host("127.0.0.1") == "127.0.0.1"
    assert is_ready_url_host("192.168.1.2") == "192.168.1.2"


def test_envlookup_reads_environment_values() -> None:
    lookup = EnvSource(environ={"A": "1", "B": "3"})
    assert lookup.get("A") == ("1", True)
    assert lookup.get("B") == ("3", True)
    assert lookup.get("C") == (None, False)

    assert lookup.get_int("A", 0) == 1
    assert lookup.get_int("B", 0) == 3
    assert lookup.get_int("C", 7) == 7
    assert lookup.get_bool("C", True) is True


def test_envlookup_typed_getters_handle_empty_and_invalid_values() -> None:
    lookup = EnvSource(environ={"A": "", "B": "not-an-int", "C": "nope"})
    assert lookup.get_int("A", 7) == 7
    with pytest.raises(ValueError, match=r"invalid B="):
        lookup.get_int("B", 0)
    with pytest.raises(ValueError, match=r"invalid C="):
        lookup.get_bool("C", True)


def test_is_port_available_false_when_bound() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        host, port = sock.getsockname()
        assert host == "127.0.0.1"
        assert is_port_available("127.0.0.1", port) is False
    finally:
        sock.close()


def test_is_port_available_true_when_free() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
        assert host == "127.0.0.1"
    finally:
        sock.close()

    # Best-effort: should be free after close.
    assert is_port_available("127.0.0.1", port) is True


def test_wait_http_ready_aborts_when_process_exits() -> None:
    class _ExitedProc:
        def poll(self):  # type: ignore[no-untyped-def]
            return 1

    proc = _ExitedProc()
    with pytest.raises(RuntimeError, match=r"exited while waiting for readiness"):
        wait_http_ready(
            name="vLLM",
            url="http://127.0.0.1:9/never",
            timeout_s=60.0,
            interval_s=60.0,
            abort_proc=proc,  # type: ignore[arg-type]
        )


def test_wait_http_ready_fails_fast_on_unauthorized_response() -> None:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # type: ignore[override]
            self.send_response(401)
            self.end_headers()

        def log_message(self, format, *args) -> None:  # type: ignore[no-untyped-def]
            # Silence noisy test output.
            return

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with pytest.raises(RuntimeError, match=r"requires auth|VR_OPENAI_API_KEY"):
            wait_http_ready(
                name="upstream",
                url=f"http://{host}:{port}/v1/models",
                timeout_s=10.0,
                interval_s=10.0,
            )
    finally:
        server.shutdown()
        server.server_close()


def test_prometheus_multiproc_dir_lifecycle(tmp_path: Path) -> None:
    created = create_prometheus_multiproc_dir(supervisor_pid=1234, root=tmp_path)
    assert created.exists()
    assert created.is_dir()
    assert created.name.startswith("1234-")

    cleanup_prometheus_multiproc_dir(created)
    assert not created.exists()


def test_cleanup_stale_prometheus_multiproc_dirs_removes_dead_pids(tmp_path: Path) -> None:
    root = tmp_path / "agentic_stack-prom-multiproc"
    root.mkdir()
    stale = root / "999999-abcdef"
    stale.mkdir()
    assert stale.exists()

    cleanup_stale_prometheus_multiproc_dirs(root)
    assert not stale.exists()


@pytest.mark.anyio
async def test_sqlite_engine_pragmas_do_not_crash(tmp_path):
    # Regression: SQLite PRAGMAs (notably `journal_mode=WAL`) must not fail due to running
    # "within a transaction" on async sqlite drivers.
    from agentic_stack import db as agentic_stack_db
    from agentic_stack.configs.builders import build_runtime_config_for_standalone

    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={
                "VR_LLM_API_BASE": "http://mock/v1",
                "VR_DB_PATH": f"sqlite+aiosqlite:///{tmp_path / 'pragmas.db'}",
            }
        )
    )
    agentic_stack_db.configure_db(runtime_config)
    engine = agentic_stack_db.create_db_engine_async(runtime_config)
    try:
        async with engine.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
    finally:
        await engine.dispose()
        agentic_stack_db.create_db_engine_async.cache_clear()
