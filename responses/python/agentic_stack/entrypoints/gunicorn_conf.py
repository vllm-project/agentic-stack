from __future__ import annotations

import os


def child_exit(server, worker) -> None:  # type: ignore[no-untyped-def]
    """
    This function is not imported by repo code directly.

    Gunicorn loads this module via `--config .../gunicorn_conf.py` and discovers hook
    functions like `child_exit` by name.

    Gunicorn hook to keep Prometheus multiprocess gauges correct.

    In multiprocess mode, Gauges configured with `multiprocess_mode="livesum"` require
    `prometheus_client.multiprocess.mark_process_dead(pid)` on worker exit.
    """
    if not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        return

    try:
        from prometheus_client import multiprocess
    except Exception:
        return

    try:
        multiprocess.mark_process_dead(worker.pid)
    except Exception as e:
        try:
            server.log.warning(f"prometheus multiprocess mark_process_dead failed: {e!r}")
        except Exception:
            return
