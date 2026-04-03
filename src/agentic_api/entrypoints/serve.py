import logging
import time

import httpx
import uvicorn

from agentic_api.config.runtime import RuntimeConfig
from agentic_api.entrypoints.app import create_app

logger = logging.getLogger(__name__)


def _wait_upstream_ready(runtime_config: RuntimeConfig) -> None:
    """Poll vLLM /health until it responds 200 or timeout is reached."""
    base = runtime_config.llm_api_base.rstrip("/")
    url = f"{base}/health"
    headers: dict[str, str] = {}
    if runtime_config.openai_api_key:
        headers["Authorization"] = f"Bearer {runtime_config.openai_api_key}"

    timeout_s = runtime_config.upstream_ready_timeout_s
    interval_s = runtime_config.upstream_ready_interval_s
    start = time.perf_counter()
    last_notice = 0.0

    with httpx.Client(timeout=httpx.Timeout(2.0), headers=headers) as client:
        while True:
            elapsed = time.perf_counter() - start
            if elapsed > timeout_s:
                raise TimeoutError(
                    f"vLLM did not become ready within {timeout_s:.0f}s: {url}"
                )

            try:
                resp = client.get(url)
                if resp.status_code == 200:
                    return
            except Exception:
                pass

            if elapsed - last_notice >= interval_s:
                last_notice = elapsed
                logger.info("waiting for upstream (%ds elapsed): %s", int(elapsed), url)

            time.sleep(interval_s)


def run(runtime_config: RuntimeConfig) -> None:
    _wait_upstream_ready(runtime_config)
    logger.info("upstream ready: %s", runtime_config.llm_api_base)

    app = create_app(runtime_config)
    uvicorn.run(
        app,
        host=runtime_config.gateway_host,
        port=runtime_config.gateway_port,
        workers=runtime_config.gateway_workers,
    )
