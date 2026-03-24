"""
Internal utilities for the `agentic-stacks serve` supervisor.

This module is intentionally stdlib-first and provides small, testable helpers for:
- preflight port checks (best-effort)
- subprocess lifecycle management (process-group aware on POSIX)
- readiness polling via HTTP
"""

from __future__ import annotations

import errno
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import httpx


def stream_lines(prefix: str, stream) -> None:
    """
    Best-effort stream forwarding with a log prefix.

    Intended to run in a daemon thread.
    """
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            sys.stdout.write(f"{prefix}{line}")
            sys.stdout.flush()
    except Exception:
        # Never let log threads crash the supervisor.
        return


def terminate_process(
    proc: subprocess.Popen[str],
    name: str,
    timeout_s: float = 10.0,
) -> None:
    """
    Terminate a subprocess, escalating to killing its process group on POSIX if needed.

    Notes:
    - We prefer terminating the process itself first. For process supervisors like Gunicorn,
      the master process coordinates worker shutdown more cleanly than sending SIGTERM to the
      whole process group immediately (which can increase shutdown races and noisy logs).
    - As an escalation step (timeout), we kill the process group on POSIX to avoid orphaned children
      (subprocesses are started with `start_new_session=True` in the supervisor).
    """
    if proc.poll() is not None:
        return

    try:
        proc.terminate()
    except Exception:
        pass

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)

    try:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            time.sleep(0.25)
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass

    try:
        proc.wait(timeout=timeout_s)
    except Exception:
        return


def is_port_available(host: str, port: int) -> bool:
    """
    Best-effort check whether a TCP port can be bound on the given host.

    Notes:
    - This is a preflight check; it cannot eliminate races.
    - For hostnames like 'localhost', we treat EADDRINUSE on any resolved address as unavailable.
    - We set `SO_REUSEADDR` for the probe socket to avoid false negatives caused by recently-closed
      connections in `TIME_WAIT`. This does not allow binding over an active listener (that would
      require `SO_REUSEPORT`).
    """
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        infos = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (host, port))]

    attempted = 0
    success = False
    for family, socktype, proto, _, sockaddr in infos:
        try:
            sock = socket.socket(family, socktype, proto)
        except OSError:
            continue
        attempted += 1
        try:
            # Avoid EADDRINUSE from TIME_WAIT during quick restarts.
            # Do not set SO_REUSEPORT here (that can mask an active listener).
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except OSError:
            pass
        try:
            sock.bind(sockaddr)
            success = True
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                return False
            # Ignore non-"in use" errors for alternative addrinfo entries.
        finally:
            try:
                sock.close()
            except Exception:
                pass

    return attempted > 0 and success


def wait_http_ready(
    *,
    name: str,
    url: str,
    timeout_s: float,
    interval_s: float,
    check_json: Callable[[object | None], bool] | None = None,
    headers: dict[str, str] | None = None,
    abort_proc: subprocess.Popen[str] | None = None,
) -> None:
    """
    Poll an HTTP endpoint until it becomes ready or times out.
    """
    start = time.perf_counter()
    last_notice = 0.0
    with httpx.Client(timeout=httpx.Timeout(2.0), headers=headers) as client:
        while True:
            elapsed = time.perf_counter() - start
            if abort_proc is not None:
                code = abort_proc.poll()
                if code is not None:
                    raise RuntimeError(
                        f"{name} process exited while waiting for readiness (code={code}): {url}"
                    )
            if elapsed > timeout_s:
                raise TimeoutError(f"{name} did not become ready within {timeout_s:.0f}s: {url}")

            resp: httpx.Response | None
            try:
                resp = client.get(url)
            except Exception:
                resp = None

            if resp is not None:
                if resp.status_code in {401, 403}:
                    raise RuntimeError(
                        f"{name} readiness check returned {resp.status_code} for {url}. "
                        "If this endpoint requires auth, set VR_OPENAI_API_KEY (or use an upstream that allows GET /v1/models)."
                    )
                if resp.status_code == 200:
                    if check_json is None:
                        return
                    try:
                        payload = resp.json()
                    except Exception:
                        payload = None
                    if check_json(payload):
                        return

            if elapsed - last_notice >= interval_s:
                last_notice = elapsed
                print(
                    f"[serve] waiting for {name}... elapsed={elapsed:.0f}s url={url}",
                    file=sys.stderr,
                )

            time.sleep(0.25)


_PROMETHEUS_MULTIPROC_ROOT = Path("/tmp/agentic_stack-prom-multiproc")


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "posix":
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def cleanup_stale_prometheus_multiproc_dirs(root: Path = _PROMETHEUS_MULTIPROC_ROOT) -> None:
    """
    Best-effort cleanup of leftover PROMETHEUS_MULTIPROC_DIR directories.

    Safety rule:
    - directories are named `<pid>-<uuid>`
    - we only remove directories whose PID is not alive
    """
    try:
        if not root.exists():
            return
        for child in root.iterdir():
            if not child.is_dir():
                continue
            prefix = child.name.split("-", 1)[0]
            try:
                pid = int(prefix)
            except Exception:
                continue
            if _pid_is_alive(pid):
                continue
            try:
                import shutil

                shutil.rmtree(child, ignore_errors=True)
            except Exception:
                continue
    except Exception:
        return


def create_prometheus_multiproc_dir(
    *,
    supervisor_pid: int | None = None,
    root: Path = _PROMETHEUS_MULTIPROC_ROOT,
) -> Path:
    """
    Create a fresh multiprocess directory under `/tmp/` for Prometheus multi-worker aggregation.
    """
    pid = os.getpid() if supervisor_pid is None else int(supervisor_pid)
    path = root / f"{pid}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def cleanup_prometheus_multiproc_dir(path: Path) -> None:
    try:
        import shutil

        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        return
