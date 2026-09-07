from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any, Callable

import pytest

from agentic_api.cli import ServeOptions
from agentic_api.process import ChildResult, ShutdownRequested


class FakeChild:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self) -> int | None:
        return None


class FakeSupervisor:
    instances: list["FakeSupervisor"] = []

    def __init__(self) -> None:
        self.starts: list[tuple[list[str], dict[str, str]]] = []
        self.terminate_timeout: float | None = None
        self.children = [FakeChild(pid=101), FakeChild(pid=202)]
        self.wait_result = ChildResult(name="agentic-server", command=("agentic-server",), returncode=0)
        self.wait_callback: Callable[[], None] | None = None
        self.shutdown_requests: list[int | None] = []
        type(self).instances.append(self)

    def start(self, command: list[str], env: dict[str, str]) -> FakeChild:
        self.starts.append((command, env))
        return self.children[len(self.starts) - 1]

    def terminate_all(self, timeout: float) -> None:
        self.terminate_timeout = timeout

    def request_shutdown(self, signal_number: int | None = None) -> None:
        self.shutdown_requests.append(signal_number)

    def shutdown_requested(self) -> bool:
        return bool(self.shutdown_requests)

    def wait_for_failure(self) -> ChildResult:
        if self.wait_callback is not None:
            self.wait_callback()
        return self.wait_result


def make_options(**overrides: Any) -> ServeOptions:
    values: dict[str, Any] = {
        "mode": "local",
        "model": "Qwen/Qwen3-4B",
        "vllm_base_url": None,
        "host": "0.0.0.0",
        "port": 9000,
        "startup_timeout_s": 600.0,
        "shutdown_timeout_s": 10.0,
        "vllm_port": 8000,
        "gateway_api_key_env": "OPENAI_API_KEY",
        "vllm_api_key_env": "AGENTIC_VLLM_API_KEY",
        "vllm_args": [],
    }
    values.update(overrides)
    return ServeOptions(**values)


@pytest.fixture(autouse=True)
def clear_fake_supervisors(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSupervisor.instances.clear()
    monkeypatch.setattr("agentic_api.launcher.sys.platform", "linux")


def test_run_serve_local_mode_starts_vllm_then_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    import agentic_api.launcher as launcher

    wait_calls: list[tuple[str, str | None, float, float]] = []
    signal_handlers: dict[int, Any] = {}

    monkeypatch.setattr(launcher, "ProcessSupervisor", FakeSupervisor)
    monkeypatch.setattr(launcher, "find_packaged_binary", lambda name: Path("/pkg/bin/agentic-server"))
    monkeypatch.setattr(launcher, "find_active_environment_executable", lambda name: Path("/venv/bin/vllm"))
    monkeypatch.setattr(launcher, "_installed_vllm_version", lambda: "0.11.0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENTIC_VLLM_API_KEY", raising=False)
    monkeypatch.setattr(
        launcher,
        "wait_for_vllm_ready",
        lambda base_url, api_key, process, timeout, interval, shutdown_requested=None: wait_calls.append(
            (base_url, api_key, timeout, interval)
        ),
    )
    monkeypatch.setattr(launcher.signal, "signal", lambda sig, handler: signal_handlers.setdefault(sig, handler))

    exit_code = launcher.run_serve(
        make_options(vllm_args=["--dtype", "bfloat16", "--max-model-len=32768"]),
    )

    supervisor = FakeSupervisor.instances[-1]

    assert exit_code == 1
    assert supervisor.starts[0][0] == [
        "/venv/bin/vllm",
        "serve",
        "Qwen/Qwen3-4B",
        "--dtype",
        "bfloat16",
        "--max-model-len=32768",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    assert "VLLM_API_KEY" not in supervisor.starts[0][1]
    assert supervisor.starts[1][0] == [
        "/pkg/bin/agentic-server",
        "--llm-api-base",
        "http://127.0.0.1:8000",
        "--llm-ready-timeout-s",
        "600.0",
        "--gateway-host",
        "0.0.0.0",
        "--gateway-port",
        "9000",
    ]
    assert "OPENAI_API_KEY" not in supervisor.starts[1][1]
    assert wait_calls == [("http://127.0.0.1:8000", None, 600.0, 2.0)]
    assert signal.SIGINT in signal_handlers
    assert signal.SIGTERM in signal_handlers
    assert supervisor.terminate_timeout == 10.0


def test_run_serve_local_mode_checks_packaged_gateway_before_starting_vllm(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import agentic_api.launcher as launcher

    supervisor = FakeSupervisor()
    monkeypatch.setattr(launcher, "ProcessSupervisor", lambda: supervisor)
    monkeypatch.setattr(
        launcher,
        "find_packaged_binary",
        lambda name: (_ for _ in ()).throw(launcher.PackagedBinaryNotFoundError("missing gateway")),
    )
    monkeypatch.setattr(launcher, "find_active_environment_executable", lambda name: Path("/venv/bin/vllm"))
    monkeypatch.setattr(launcher.signal, "signal", lambda sig, handler: handler)

    assert launcher.run_serve(make_options()) == 1
    assert supervisor.starts == []
    assert capsys.readouterr().err == "missing gateway\n"


def test_run_serve_local_mode_rejects_unsupported_platform_before_discovery(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import agentic_api.launcher as launcher

    supervisor = FakeSupervisor()
    monkeypatch.setattr(launcher, "ProcessSupervisor", lambda: supervisor)
    monkeypatch.setattr(launcher.sys, "platform", "darwin")
    monkeypatch.setattr(
        launcher,
        "find_packaged_binary",
        lambda name: (_ for _ in ()).throw(AssertionError("gateway discovery should not run")),
    )
    monkeypatch.setattr(
        launcher,
        "find_active_environment_executable",
        lambda name: (_ for _ in ()).throw(AssertionError("vLLM discovery should not run")),
    )
    monkeypatch.setattr(launcher.signal, "signal", lambda sig, handler: handler)

    assert launcher.run_serve(make_options()) == 1
    assert supervisor.starts == []
    assert "local mode is currently supported only on Linux" in capsys.readouterr().err


def test_run_serve_local_mode_returns_sigint_during_readiness_without_startup_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import agentic_api.launcher as launcher

    signal_handlers: dict[int, Any] = {}
    supervisor = FakeSupervisor()

    monkeypatch.setattr(launcher, "ProcessSupervisor", lambda: supervisor)
    monkeypatch.setattr(launcher, "find_packaged_binary", lambda name: Path("/pkg/bin/agentic-server"))
    monkeypatch.setattr(launcher, "find_active_environment_executable", lambda name: Path("/venv/bin/vllm"))
    monkeypatch.setattr(launcher, "_installed_vllm_version", lambda: "0.11.0")

    def fake_wait_for_vllm_ready(
        base_url: str,
        api_key: str | None,
        process: FakeChild,
        timeout: float,
        interval: float,
        shutdown_requested: Callable[[], bool] | None = None,
    ) -> None:
        del base_url, api_key, process, timeout, interval
        signal_handlers[signal.SIGINT](signal.SIGINT, None)
        assert shutdown_requested is not None
        assert shutdown_requested() is True
        raise ShutdownRequested()

    monkeypatch.setattr(launcher, "wait_for_vllm_ready", fake_wait_for_vllm_ready)
    monkeypatch.setattr(launcher.signal, "signal", lambda sig, handler: signal_handlers.setdefault(sig, handler))

    exit_code = launcher.run_serve(make_options())

    assert exit_code == 130
    assert capsys.readouterr().err == ""
    assert len(supervisor.starts) == 1
    assert supervisor.shutdown_requests == [signal.SIGINT]
    assert supervisor.terminate_timeout == 10.0


def test_run_serve_remote_mode_starts_only_rust_and_uses_selected_gateway_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_api.launcher as launcher

    monkeypatch.setattr(launcher, "ProcessSupervisor", FakeSupervisor)
    monkeypatch.setattr(launcher, "find_packaged_binary", lambda name: Path("/pkg/bin/agentic-server"))
    monkeypatch.setattr(launcher.signal, "signal", lambda sig, handler: handler)
    monkeypatch.setenv("CUSTOM_GATEWAY_KEY", "remote-secret")

    supervisor = FakeSupervisor()
    supervisor.wait_result = ChildResult(name="agentic-server", command=("agentic-server",), returncode=17)
    monkeypatch.setattr(launcher, "ProcessSupervisor", lambda: supervisor)

    exit_code = launcher.run_serve(
        make_options(
            mode="remote",
            model=None,
            vllm_base_url="https://upstream.example.com/base",
            gateway_api_key_env="CUSTOM_GATEWAY_KEY",
        )
    )

    assert exit_code == 17
    assert supervisor.starts[0][0] == [
        "/pkg/bin/agentic-server",
        "--llm-api-base",
        "https://upstream.example.com/base",
        "--llm-ready-timeout-s",
        "600.0",
        "--gateway-host",
        "0.0.0.0",
        "--gateway-port",
        "9000",
    ]
    assert supervisor.starts[0][1]["OPENAI_API_KEY"] == "remote-secret"
    assert supervisor.terminate_timeout == 10.0


def test_run_serve_reports_startup_failure_and_cleans_up_started_children(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import agentic_api.launcher as launcher

    supervisor = FakeSupervisor()

    monkeypatch.setattr(launcher, "ProcessSupervisor", lambda: supervisor)
    monkeypatch.setattr(launcher, "find_packaged_binary", lambda name: Path("/pkg/bin/agentic-server"))
    monkeypatch.setattr(launcher, "find_active_environment_executable", lambda name: Path("/venv/bin/vllm"))
    monkeypatch.setattr(launcher, "_installed_vllm_version", lambda: "0.11.0")
    monkeypatch.setattr(
        launcher,
        "wait_for_vllm_ready",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("timed out waiting for vLLM readiness")),
    )
    monkeypatch.setattr(launcher.signal, "signal", lambda sig, handler: handler)

    exit_code = launcher.run_serve(make_options())

    assert exit_code == 1
    assert len(supervisor.starts) == 1
    assert "timed out waiting for vLLM readiness" in capsys.readouterr().err
    assert supervisor.terminate_timeout == 10.0


def test_run_serve_reports_clean_child_exit_as_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import agentic_api.launcher as launcher

    supervisor = FakeSupervisor()
    monkeypatch.setattr(launcher, "ProcessSupervisor", lambda: supervisor)
    monkeypatch.setattr(launcher, "find_packaged_binary", lambda name: Path("/pkg/bin/agentic-server"))
    monkeypatch.setattr(launcher.signal, "signal", lambda sig, handler: handler)

    exit_code = launcher.run_serve(
        make_options(mode="remote", model=None, vllm_base_url="https://upstream.example.com/base")
    )

    assert exit_code == 1
    assert "agentic-server exited unexpectedly with status 0" in capsys.readouterr().err


def test_run_serve_reports_readiness_timeout_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import agentic_api.launcher as launcher

    supervisor = FakeSupervisor()

    monkeypatch.setattr(launcher, "ProcessSupervisor", lambda: supervisor)
    monkeypatch.setattr(launcher, "find_packaged_binary", lambda name: Path("/pkg/bin/agentic-server"))
    monkeypatch.setattr(launcher, "find_active_environment_executable", lambda name: Path("/venv/bin/vllm"))
    monkeypatch.setattr(launcher, "_installed_vllm_version", lambda: "0.11.0")
    monkeypatch.setattr(
        launcher,
        "wait_for_vllm_ready",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timed out waiting for vLLM readiness")),
    )
    monkeypatch.setattr(launcher.signal, "signal", lambda sig, handler: handler)

    exit_code = launcher.run_serve(make_options())

    assert exit_code == 1
    assert capsys.readouterr().err == "timed out waiting for vLLM readiness\n"
    assert len(supervisor.starts) == 1
    assert supervisor.terminate_timeout == 10.0


def test_run_serve_restores_prior_signal_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    import agentic_api.launcher as launcher

    previous_sigint = object()
    previous_sigterm = object()
    registered_handlers = {
        signal.SIGINT: previous_sigint,
        signal.SIGTERM: previous_sigterm,
    }

    def fake_signal(sig: int, handler: Any) -> Any:
        previous = registered_handlers[sig]
        registered_handlers[sig] = handler
        return previous

    supervisor = FakeSupervisor()
    monkeypatch.setattr(launcher, "ProcessSupervisor", lambda: supervisor)
    monkeypatch.setattr(launcher, "find_packaged_binary", lambda name: Path("/pkg/bin/agentic-server"))
    monkeypatch.setattr(launcher.signal, "signal", fake_signal)

    exit_code = launcher.run_serve(
        make_options(mode="remote", model=None, vllm_base_url="https://upstream.example.com/base")
    )

    assert exit_code == 1
    assert registered_handlers[signal.SIGINT] is previous_sigint
    assert registered_handlers[signal.SIGTERM] is previous_sigterm


def test_run_serve_uses_a_single_shutdown_path_for_repeated_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    import agentic_api.launcher as launcher

    supervisor = FakeSupervisor()
    signal_handlers: dict[int, Any] = {}

    monkeypatch.setattr(launcher, "ProcessSupervisor", lambda: supervisor)
    monkeypatch.setattr(launcher, "find_packaged_binary", lambda name: Path("/pkg/bin/agentic-server"))
    monkeypatch.setattr(launcher.signal, "signal", lambda sig, handler: signal_handlers.setdefault(sig, handler))

    def trigger_signal() -> None:
        signal_handlers[signal.SIGTERM](signal.SIGTERM, None)
        signal_handlers[signal.SIGTERM](signal.SIGTERM, None)

    supervisor.wait_callback = trigger_signal
    supervisor.wait_result = ChildResult(name="agentic-server", command=("agentic-server",), returncode=-15)

    exit_code = launcher.run_serve(
        make_options(mode="remote", model=None, vllm_base_url="https://upstream.example.com/base")
    )

    assert exit_code == 143
    assert supervisor.shutdown_requests == [signal.SIGTERM, signal.SIGTERM]
    assert supervisor.terminate_timeout == 10.0
