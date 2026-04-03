import json
from collections.abc import Iterable

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from agentic_api.config.runtime import RuntimeConfig

_REQUEST_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
_RESPONSE_HOP_BY_HOP_HEADERS = set(_REQUEST_HOP_BY_HOP_HEADERS)
_REQUEST_DROP_HEADERS = _REQUEST_HOP_BY_HOP_HEADERS | {"host", "content-length"}

_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)
_NON_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)


class ProxyClientManager:
    """App-scoped owner for proxy HTTP clients."""

    def __init__(self) -> None:
        self._stream_client: httpx.AsyncClient | None = None
        self._non_stream_client: httpx.AsyncClient | None = None

    def get_client(self, *, allow_sse_passthrough: bool) -> httpx.AsyncClient:
        if allow_sse_passthrough:
            if self._stream_client is None:
                self._stream_client = httpx.AsyncClient(
                    follow_redirects=False,
                    timeout=_STREAM_TIMEOUT,
                )
            return self._stream_client
        if self._non_stream_client is None:
            self._non_stream_client = httpx.AsyncClient(
                follow_redirects=False,
                timeout=_NON_STREAM_TIMEOUT,
            )
        return self._non_stream_client

    async def aclose(self) -> None:
        if self._stream_client is not None:
            await self._stream_client.aclose()
            self._stream_client = None
        if self._non_stream_client is not None:
            await self._non_stream_client.aclose()
            self._non_stream_client = None


def proxy_error(*, status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "api_error",
                "param": None,
                "code": code,
            }
        },
    )


def _upstream_url(*, path_suffix: str, runtime_config: RuntimeConfig) -> str:
    base = runtime_config.llm_api_base.rstrip("/")
    return f"{base}/{path_suffix.lstrip('/')}"


def _is_sse_content_type(content_type: str | None) -> bool:
    if not content_type:
        return False
    return content_type.lower().startswith("text/event-stream")


def _filter_request_headers(
    headers: Iterable[tuple[str, str]],
    *,
    runtime_config: RuntimeConfig,
) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for key, value in headers:
        if key.lower() in _REQUEST_DROP_HEADERS:
            continue
        filtered[key] = value

    has_auth = any(key.lower() == "authorization" for key in filtered)
    if not has_auth:
        api_key = (runtime_config.openai_api_key or "").strip()
        if api_key:
            filtered["Authorization"] = f"Bearer {api_key}"
    return filtered


def _filter_response_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for key, value in headers:
        if key.lower() in _RESPONSE_HOP_BY_HOP_HEADERS:
            continue
        filtered[key] = value
    return filtered


async def proxy_responses(
    *,
    request: Request,
    runtime_config: RuntimeConfig,
    proxy_client_manager: ProxyClientManager,
) -> Response:
    body = await request.body()
    upstream_headers = _filter_request_headers(
        request.headers.items(),
        runtime_config=runtime_config,
    )
    upstream_params = list(request.query_params.multi_items())

    try:
        is_streaming = bool(json.loads(body).get("stream", False)) if body else False
    except Exception:
        is_streaming = False

    client = proxy_client_manager.get_client(allow_sse_passthrough=is_streaming)
    req = client.build_request(
        method="POST",
        url=_upstream_url(path_suffix="/v1/responses", runtime_config=runtime_config),
        params=upstream_params,
        headers=upstream_headers,
        content=body,
    )

    try:
        upstream_resp = await client.send(req, stream=True)
    except httpx.TimeoutException:
        return proxy_error(
            status_code=504,
            code="upstream_timeout",
            message="Upstream timeout",
        )
    except httpx.RequestError:
        return proxy_error(
            status_code=502,
            code="upstream_unavailable",
            message="Upstream unavailable",
        )

    response_headers = _filter_response_headers(upstream_resp.headers.items())
    if _is_sse_content_type(upstream_resp.headers.get("content-type")):
        response_headers["X-Accel-Buffering"] = "no"

        async def _stream():
            try:
                async for chunk in upstream_resp.aiter_raw():
                    yield chunk
            except Exception:
                return
            finally:
                await upstream_resp.aclose()

        return StreamingResponse(
            _stream(),
            status_code=upstream_resp.status_code,
            headers=response_headers,
        )

    payload = await upstream_resp.aread()
    await upstream_resp.aclose()
    return Response(
        content=payload,
        status_code=upstream_resp.status_code,
        headers=response_headers,
    )
