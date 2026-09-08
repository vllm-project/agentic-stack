from __future__ import annotations

import os
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any

import pytest

from agentic_api.process import ChildResult, ProcessSupervisor, ShutdownRequested, wait_for_vllm_ready


class FakePopen:
    def __init__(self, pid: int, poll_values: list[int | None] | None = None) -> None:
        self.pid = pid
        self._poll_values = list(poll_values or [None])
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0
        self.wait_timeout_values: list[float | None] = []
        self.wait_should_timeout = False

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if len(self._poll_values) > 1:
            return self._poll_values.pop(0)
        return self._poll_values[0]

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        self.wait_timeout_values.append(timeout)
        if self.returncode is not None:
            return self.returncode
        if self.wait_should_timeout:
            self.wait_should_timeout = False
            raise TimeoutExpired(cmd=["fake"], timeout=timeout)
        polled = self.poll()
        if polled is None:
            self.returncode = 0
        else:
            self.returncode = polled
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


class DummyProcess:
    def __init__(self, poll_values: list[int | None]) -> None:
        self._poll_values = list(poll_values)
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if len(self._poll_values) > 1:
            result = self._poll_values.pop(0)
        else:
            result = self._poll_values[0]
        if result is not None:
            self.returncode = result
        return result


class ModelsHandler(BaseHTTPRequestHandler):
    requests_seen = 0
    auth_headers: list[str | None] = []
    response_statuses: list[int] = [200]
    response_delay_s = 0.0

    def do_GET(self) -> None:  # noqa: N802
        type(self).requests_seen += 1
        type(self).auth_headers.append(self.headers.get("Authorization"))
        if self.path != "/v1/models":
            self.send_response(404)
            self.end_headers()
            return

        if type(self).response_delay_s:
            time.sleep(type(self).response_delay_s)

        status = type(self).response_statuses[min(type(self).requests_seen - 1, len(type(self).response_statuses) - 1)]
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"data":[{"id":"model-a"}]}')

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        del format, args


@pytest.fixture(autouse=True)
def reset_models_handler() -> None:
    ModelsHandler.requests_seen = 0
    ModelsHandler.auth_headers = []
    ModelsHandler.response_statuses = [200]
    ModelsHandler.response_delay_s = 0.0


@pytest.fixture
def models_server() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield server, f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=1)
        server.server_close()


def reserve_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_start_uses_argument_list_and_inherits_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    popen_calls: list[tuple[list[str], dict[str, Any]]] = []
    fake_process = FakePopen(pid=1234)

    def fake_popen(command: list[str], **kwargs: Any) -> FakePopen:
        popen_calls.append((command, kwargs))
        return fake_process

    monkeypatch.setattr("agentic_api.process.subprocess.Popen", fake_popen)

    supervisor = ProcessSupervisor()
    process = supervisor.start(["vllm", "serve", "Qwen/Qwen3-4B"], {"KEY": "value"})

    assert process is fake_process
    assert popen_calls == [
        (
            ["vllm", "serve", "Qwen/Qwen3-4B"],
            {
                "env": {"KEY": "value"},
                "shell": False,
                "start_new_session": True,
            },
        )
    ]


def test_terminate_all_sends_posix_group_signals_then_force_kills(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_process = FakePopen(pid=4321)
    fake_process.wait_should_timeout = True
    received_signals: list[tuple[int, int]] = []

    def fake_killpg(pid: int, sig: int) -> None:
        if sig == 0:
            if fake_process.returncode is not None:
                raise ProcessLookupError
            return
        received_signals.append((pid, sig))
        if sig == 9:
            fake_process.returncode = -9

    monkeypatch.setattr("agentic_api.process.subprocess.Popen", lambda command, **kwargs: fake_process)
    monkeypatch.setattr("agentic_api.process.os.killpg", fake_killpg)

    supervisor = ProcessSupervisor()
    supervisor.start(["agentic-server"], {})
    supervisor.terminate_all(timeout=0.01)

    assert received_signals == [(4321, 15), (4321, 9)]
    assert fake_process.returncode == -9


def test_terminate_all_reaps_exited_group_leader_without_waiting_for_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_process = FakePopen(pid=5432)
    leader_exited = False
    original_poll = fake_process.poll

    def fake_poll() -> int | None:
        if leader_exited:
            fake_process.returncode = -signal.SIGTERM
        return original_poll()

    def fake_killpg(pid: int, sig: int) -> None:
        nonlocal leader_exited
        assert pid == fake_process.pid
        if sig == signal.SIGTERM:
            leader_exited = True
        elif sig == 0 and fake_process.returncode is not None:
            raise ProcessLookupError

    fake_process.poll = fake_poll  # type: ignore[method-assign]
    monkeypatch.setattr("agentic_api.process.subprocess.Popen", lambda command, **kwargs: fake_process)
    monkeypatch.setattr("agentic_api.process.os.killpg", fake_killpg)
    monkeypatch.setattr(
        "agentic_api.process.time.sleep",
        lambda _: pytest.fail("an exited, reaped process group should not wait for the shutdown deadline"),
    )

    supervisor = ProcessSupervisor()
    supervisor.start(["agentic-server"], {})
    supervisor.terminate_all(timeout=10.0)

    assert fake_process.returncode == -signal.SIGTERM


def test_terminate_all_uses_direct_child_signals_off_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_process = FakePopen(pid=99)
    fake_process.wait_should_timeout = True

    monkeypatch.setattr("agentic_api.process.subprocess.Popen", lambda command, **kwargs: fake_process)
    monkeypatch.setattr("agentic_api.process._IS_POSIX", False)

    supervisor = ProcessSupervisor()
    supervisor.start(["agentic-server"], {})
    supervisor.terminate_all(timeout=0.01)

    assert fake_process.terminate_calls == 1
    assert fake_process.kill_calls == 1


def test_wait_for_failure_returns_exited_child_status(monkeypatch: pytest.MonkeyPatch) -> None:
    first = FakePopen(pid=1, poll_values=[None, None, None])
    second = FakePopen(pid=2, poll_values=[None, 7])

    monkeypatch.setattr(
        "agentic_api.process.subprocess.Popen",
        lambda command, **kwargs: first if command[0] == "vllm" else second,
    )
    monkeypatch.setattr("agentic_api.process.time.sleep", lambda _: None)

    supervisor = ProcessSupervisor()
    supervisor.start(["vllm", "serve"], {})
    supervisor.start(["agentic-server"], {})

    assert supervisor.wait_for_failure() == ChildResult(
        name="agentic-server",
        command=("agentic-server",),
        returncode=7,
    )


def test_wait_for_vllm_ready_allows_a_slow_successful_probe(
    models_server: tuple[ThreadingHTTPServer, str],
) -> None:
    server, base_url = models_server
    ModelsHandler.response_delay_s = 0.2

    wait_for_vllm_ready(
        base_url=base_url,
        api_key=None,
        process=DummyProcess([None]),
        timeout=0.5,
        interval=0.01,
    )

    assert server is not None


def test_wait_for_failure_raises_shutdown_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_process = FakePopen(pid=77, poll_values=[None, None, None])

    monkeypatch.setattr("agentic_api.process.subprocess.Popen", lambda command, **kwargs: fake_process)
    supervisor = ProcessSupervisor()
    supervisor.start(["agentic-server"], {})
    supervisor.request_shutdown(signal.SIGTERM)

    with pytest.raises(ShutdownRequested):
        supervisor.wait_for_failure()


def test_start_terminates_new_child_immediately_after_shutdown_request(monkeypatch: pytest.MonkeyPatch) -> None:
    first = FakePopen(pid=100)
    second = FakePopen(pid=200)
    popen_results = iter([first, second])
    received_signals: list[tuple[int, int]] = []

    def fake_killpg(pid: int, sig: int) -> None:
        if sig == 0:
            process = first if pid == 100 else second
            if process.returncode is not None:
                raise ProcessLookupError
            return
        received_signals.append((pid, sig))
        if sig == signal.SIGTERM:
            if pid == 100:
                first.returncode = -15
            if pid == 200:
                second.returncode = -15

    monkeypatch.setattr("agentic_api.process.subprocess.Popen", lambda command, **kwargs: next(popen_results))
    monkeypatch.setattr("agentic_api.process.os.killpg", fake_killpg)

    supervisor = ProcessSupervisor()
    supervisor.start(["vllm"], {})
    supervisor.request_shutdown(signal.SIGTERM)
    supervisor.terminate_all(timeout=0.01)
    supervisor.start(["agentic-server"], {})

    assert received_signals == [(100, signal.SIGTERM), (200, signal.SIGTERM)]


@pytest.mark.skipif(os.name != "posix" or not hasattr(os, "fork"), reason="requires POSIX process groups and fork")
def test_terminate_all_force_kills_forked_descendant_after_session_leader_exits(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "descendant.pid"
    helper = tmp_path / "fork_descendant.py"
    helper.write_text(
        "\n".join(
            (
                "from __future__ import annotations",
                "import os",
                "import signal",
                "import sys",
                "from pathlib import Path",
                "",
                "child_pid_path = Path(sys.argv[1])",
                "if os.fork() != 0:",
                "    os._exit(0)",
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                "child_pid_path.write_text(str(os.getpid()), encoding='utf-8')",
                "while True:",
                "    signal.pause()",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    supervisor = ProcessSupervisor()
    leader = supervisor.start([sys.executable, str(helper), str(child_pid_path)], os.environ.copy())
    process_group_id = leader.pid

    try:
        assert leader.wait(timeout=2) == 0
        deadline = time.monotonic() + 2
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_pid_path.exists(), "forked descendant did not start"
        assert _process_group_exists(process_group_id)

        supervisor.terminate_all(timeout=0.1)

        deadline = time.monotonic() + 2
        while _process_group_exists(process_group_id) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _process_group_exists(process_group_id), "forked descendant survived supervisor cleanup"
    finally:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def test_wait_for_vllm_ready_accepts_authenticated_models_response(
    models_server: tuple[ThreadingHTTPServer, str]
) -> None:
    _, base_url = models_server

    wait_for_vllm_ready(
        base_url=base_url,
        api_key="secret-token",
        process=DummyProcess([None, None]),
        timeout=0.5,
        interval=0.01,
    )

    assert ModelsHandler.auth_headers == ["Bearer secret-token"]


def test_wait_for_vllm_ready_recovers_from_transient_connection_failures(
    monkeypatch: pytest.MonkeyPatch,
    models_server: tuple[ThreadingHTTPServer, str],
) -> None:
    _, base_url = models_server
    original_urlopen = urllib.request.urlopen
    attempts = 0

    def transient_urlopen(request: urllib.request.Request, *, timeout: float) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise urllib.error.URLError(ConnectionRefusedError("simulated connection refusal"))
        return original_urlopen(request, timeout=timeout)

    monkeypatch.setattr("agentic_api.process.urllib.request.urlopen", transient_urlopen)

    wait_for_vllm_ready(
        base_url=base_url,
        api_key=None,
        process=DummyProcess([None] * 20),
        timeout=0.5,
        interval=0.01,
    )

    assert attempts == 3
    assert ModelsHandler.requests_seen == 1


def test_wait_for_vllm_ready_times_out_without_leaking_api_key() -> None:
    port = reserve_tcp_port()

    with pytest.raises(TimeoutError, match="timed out waiting for vLLM readiness"):
        wait_for_vllm_ready(
            base_url=f"http://127.0.0.1:{port}",
            api_key="super-secret",
            process=DummyProcess([None] * 20),
            timeout=0.15,
            interval=0.02,
        )


def test_wait_for_vllm_ready_retries_request_timeouts_without_leaking_api_key(
    models_server: tuple[ThreadingHTTPServer, str]
) -> None:
    _, base_url = models_server
    ModelsHandler.response_delay_s = 0.5

    with pytest.raises(TimeoutError) as exc_info:
        wait_for_vllm_ready(
            base_url=base_url,
            api_key="super-secret",
            process=DummyProcess([None] * 20),
            timeout=0.2,
            interval=0.02,
        )

    assert "super-secret" not in str(exc_info.value)


def test_wait_for_vllm_ready_reports_backend_exit_without_leaking_api_key() -> None:
    port = reserve_tcp_port()

    with pytest.raises(RuntimeError, match="exited with status 17") as exc_info:
        wait_for_vllm_ready(
            base_url=f"http://127.0.0.1:{port}",
            api_key="super-secret",
            process=DummyProcess([None, 17]),
            timeout=0.5,
            interval=0.02,
        )

    assert "super-secret" not in str(exc_info.value)


def test_wait_for_vllm_ready_raises_shutdown_requested_before_reporting_child_exit() -> None:
    port = reserve_tcp_port()

    with pytest.raises(ShutdownRequested):
        wait_for_vllm_ready(
            base_url=f"http://127.0.0.1:{port}",
            api_key="super-secret",
            process=DummyProcess([17]),
            timeout=0.5,
            interval=0.02,
            shutdown_requested=lambda: True,
        )
