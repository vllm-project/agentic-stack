import pickle
from typing import Any

import orjson
import yaml
from httpx import AsyncClient, HTTPStatusError, Response
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from agentic_stack.entrypoints._state import CURRENT_REQUEST_ID
from agentic_stack.utils.types import JSONInput, JSONOutput


def load_pickle(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def dump_pickle(out_path: str, obj: Any):
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)


def read_json(path: str) -> JSONOutput:
    """Reads a JSON file.

    Args:
        path (str): Path to the file.

    Returns:
        data (JSONOutput): The data.
    """
    with open(path, "r") as f:
        return orjson.loads(f.read())


def dump_json(data: JSONInput, path: str, **kwargs) -> str:
    """Writes a JSON file.

    Args:
        data (JSONInput): The data.
        path (str): Path to the file.
        **kwargs: Other keyword arguments to pass into `orjson.dumps`.

    Returns:
        path (str): Path to the file.
    """
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, **kwargs))
    return path


def json_loads(data: str) -> JSONOutput:
    return orjson.loads(data)


def json_dumps(data: JSONInput, **kwargs) -> str:
    return orjson.dumps(data, **kwargs).decode("utf-8")


def read_yaml(path: str) -> JSONOutput:
    """Reads a YAML file.

    Args:
        path (str): Path to the file.

    Returns:
        data (JSONOutput): The data.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_yaml(
    data: JSONInput,
    path: str,
    allow_unicode: bool = True,
    encoding: str = "utf-8",
    **kwargs,
) -> str:
    """Writes a YAML file.

    Args:
        data (JSONInput): The data.
        path (str): Path to the file.
        allow_unicode (bool, optional): See `yaml.dump`.
        encoding (str, optional): See `yaml.dump`.
        **kwargs: Other keyword arguments to pass into `yaml.dump`.

    Returns:
        path (str): Path to the file.
    """
    with open(path, "w") as f:
        yaml.dump(data, f, allow_unicode=allow_unicode, encoding=encoding, **kwargs)
    return path


def _should_retry_status(response: Response) -> None:
    """Raise exceptions for retryable HTTP status codes."""
    if response.status_code in (429, 502, 503, 504):
        response.raise_for_status()  # This will raise HTTPStatusError


async def _propagate_request_id(request) -> None:  # type: ignore[no-untyped-def]
    request_id = CURRENT_REQUEST_ID.get()
    if request_id and "x-request-id" not in request.headers:
        request.headers["x-request-id"] = request_id


def get_async_client(
    *,
    timeout: float = 30.0,
    follow_redirects: bool = False,  # Prevent redirect-based SSRF
    max_redirects: int = 0,
    retry_config: RetryConfig | None = None,
) -> AsyncClient:
    if retry_config is None:
        retry_config = RetryConfig(
            # Retry on HTTP errors and connection issues
            retry=retry_if_exception_type((HTTPStatusError, ConnectionError)),
            # Smart waiting: respects Retry-After headers, falls back to exponential backoff
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=1, max=60),
                max_wait=300,
            ),
            # Stop after 5 attempts
            stop=stop_after_attempt(5),
            # Re-raise the last exception if all retries fail
            reraise=True,
        )
    client = AsyncClient(
        timeout=timeout,
        follow_redirects=follow_redirects,
        max_redirects=max_redirects,
        event_hooks={"request": [_propagate_request_id]},
        transport=AsyncTenacityTransport(
            config=retry_config,
            validate_response=_should_retry_status,
        ),
    )
    return client


HTTP_ACLIENT = get_async_client()
