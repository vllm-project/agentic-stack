from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


POLL_INTERVAL_S = 0.05
READY_REQUEST_TIMEOUT_S = 5.0
_IS_POSIX = os.name == "posix"


class ShutdownRequested(RuntimeError):
    """Raised when launcher shutdown has been requested."""


@dataclass(frozen=True)
class ChildResult:
    name: str
    command: tuple[str, ...]
    returncode: int


@dataclass(frozen=True)
class _ManagedChild:
    name: str
    command: tuple[str, ...]
    process: subprocess.Popen[str]
    process_group_id: int | None


class ProcessSupervisor:
    def __init__(self) -> None:
        self._children: list[_ManagedChild] = []
        self._lock = Lock()
        self._shutdown_requested = False
        self._shutdown_signal: int | None = None

    def request_shutdown(self, signal_number: int | None = None) -> None:
        self._shutdown_requested = True
        if signal_number is not None and self._shutdown_signal is None:
            self._shutdown_signal = signal_number

    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def shutdown_signal(self) -> int | None:
        return self._shutdown_signal

    def start(self, command: Sequence[str], env: Mapping[str, str]) -> subprocess.Popen[str]:
        command_list = [str(part) for part in command]
        popen_kwargs: dict[str, object] = {
            "env": dict(env),
            "shell": False,
        }
        if _IS_POSIX:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(command_list, **popen_kwargs)
        child = _ManagedChild(
            name=Path(command_list[0]).name or command_list[0],
            command=tuple(command_list),
            process=process,
            process_group_id=process.pid if _IS_POSIX else None,
        )
        with self._lock:
            self._children.append(child)
            shutdown_requested = self._shutdown_requested
        if shutdown_requested:
            self._terminate_child(child)
        return process

    def terminate_all(self, timeout: float) -> None:
        self.request_shutdown()
        deadline = time.monotonic() + max(timeout, 0.0)
        seen_children: set[int] = set()

        while True:
            with self._lock:
                children = [child for child in self._children if id(child) not in seen_children]

            if not children:
                return

            for child in children:
                seen_children.add(id(child))
                self._terminate_child(child)

            pending: list[_ManagedChild] = []
            for child in children:
                self._wait_for_child_target(child, deadline)
                if self._child_target_exists(child):
                    pending.append(child)

            for child in pending:
                self._kill_child(child)

            force_kill_deadline = time.monotonic() + 1.0
            for child in pending:
                self._wait_for_child_target(child, force_kill_deadline)

    def wait_for_failure(self) -> ChildResult:
        while True:
            if self._shutdown_requested:
                raise ShutdownRequested()

            with self._lock:
                children = list(self._children)

            if not children:
                raise RuntimeError("no managed child processes")

            for child in children:
                returncode = child.process.poll()
                if returncode is not None:
                    return ChildResult(name=child.name, command=child.command, returncode=returncode)

            time.sleep(POLL_INTERVAL_S)

    def _terminate_child(self, child: _ManagedChild) -> None:
        if child.process_group_id is not None:
            try:
                os.killpg(child.process_group_id, signal.SIGTERM)
            except ProcessLookupError:
                pass
            return
        if child.process.poll() is None:
            child.process.terminate()

    def _kill_child(self, child: _ManagedChild) -> None:
        if child.process_group_id is not None:
            try:
                os.killpg(child.process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return
        if child.process.poll() is None:
            child.process.kill()

    def _wait_for_child_target(self, child: _ManagedChild, deadline: float) -> None:
        while True:
            child.process.poll()
            if not self._child_target_exists(child):
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(POLL_INTERVAL_S, remaining))
        child.process.poll()

    def _child_target_exists(self, child: _ManagedChild) -> bool:
        if child.process_group_id is None:
            return child.process.poll() is None
        try:
            os.killpg(child.process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def wait_for_vllm_ready(
    base_url: str,
    api_key: str | None,
    process: subprocess.Popen[str],
    timeout: float,
    interval: float,
    shutdown_requested: Callable[[], bool] | None = None,
) -> None:
    ready_url = f"{base_url.rstrip('/')}/v1/models"
    deadline = time.monotonic() + max(timeout, 0.0)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: str | None = None
    while True:
        if shutdown_requested is not None and shutdown_requested():
            raise ShutdownRequested()

        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"vLLM exited with status {returncode} before becoming ready")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        request = urllib.request.Request(ready_url, headers=headers)
        request_timeout = min(READY_REQUEST_TIMEOUT_S, max(remaining, 0.01))
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"HTTP {response.status}"
        except urllib.error.HTTPError as error:
            last_error = f"HTTP {error.code}"
        except (TimeoutError, socket.timeout):
            last_error = "request timeout"
        except urllib.error.URLError:
            last_error = "connection failure"

        if shutdown_requested is not None and shutdown_requested():
            raise ShutdownRequested()

        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"vLLM exited with status {returncode} before becoming ready")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(max(interval, 0.0), remaining))

    detail = f" ({last_error})" if last_error else ""
    raise TimeoutError(f"timed out waiting for vLLM readiness{detail}")
