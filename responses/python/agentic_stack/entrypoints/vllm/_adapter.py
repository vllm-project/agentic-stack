from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib import import_module
from typing import Any, Protocol

from fastapi import FastAPI


class VllmApiServerModule(Protocol):
    build_app: Callable[[Any, Any], FastAPI]


def _load_upstream_cli_main() -> Callable[[], None]:
    try:
        from vllm.entrypoints.cli.main import main as upstream_main
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This operation requires the `vllm` package to be installed in the active environment."
        ) from exc
    return upstream_main


def run_upstream_cli(argv: list[str]) -> int:
    upstream_main = _load_upstream_cli_main()
    previous_argv = sys.argv
    sys.argv = ["vllm", *argv]
    try:
        upstream_main()
    finally:
        sys.argv = previous_argv
    return 0


def load_api_server_module() -> VllmApiServerModule:
    try:
        from vllm.entrypoints.openai import api_server
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Integrated mode requires the `vllm` package to be installed in the active environment."
        ) from exc
    return api_server


def load_responses_attach_router() -> Callable[[FastAPI], None]:
    try:
        responses_api_router = import_module("vllm.entrypoints.openai.responses.api_router")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Integrated mode requires the vLLM OpenAI Responses router seam at "
            "`vllm.entrypoints.openai.responses.api_router.attach_router`."
        ) from exc

    attach_router = getattr(responses_api_router, "attach_router", None)
    if not callable(attach_router):
        raise RuntimeError(
            "Integrated mode requires a callable "
            "`vllm.entrypoints.openai.responses.api_router.attach_router`."
        )
    return attach_router


@contextmanager
def suppress_native_responses_attach() -> Iterator[None]:
    original_attach_router = load_responses_attach_router()
    responses_api_router = import_module("vllm.entrypoints.openai.responses.api_router")

    def _suppress_native_responses(_app: FastAPI) -> None:
        return None

    responses_api_router.attach_router = _suppress_native_responses
    try:
        yield
    finally:
        responses_api_router.attach_router = original_attach_router
