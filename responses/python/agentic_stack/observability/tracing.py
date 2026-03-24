from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI
from loguru import logger

from agentic_stack.configs.runtime import RuntimeConfig
from agentic_stack.utils import uuid7_str

_TRACING_CONFIGURED = False
_HTTPX_INSTRUMENTED = False


def configure_tracing(
    runtime_config: RuntimeConfig,
    app: FastAPI | None = None,
) -> Callable[[], None]:
    """
    Configure OpenTelemetry tracing (OTLP gRPC exporter), gated by `VR_TRACING_ENABLED`.

    Returns a shutdown callback that flushes and shuts down providers.
    """
    if not runtime_config.tracing_enabled:
        return lambda: None

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    except Exception as e:
        logger.warning(
            f"OpenTelemetry tracing unavailable: {e!r}. Install optional dependencies via `agentic_stack[tracing]`."
        )
        return lambda: None

    global _TRACING_CONFIGURED, _HTTPX_INSTRUMENTED

    shutdown_callbacks: list[Callable[[], None]] = []

    if not _TRACING_CONFIGURED:
        endpoint = (
            f"http://{runtime_config.opentelemetry_host}:{runtime_config.opentelemetry_port}"
        )
        ratio = float(runtime_config.tracing_sample_ratio)
        ratio = 0.0 if ratio < 0.0 else 1.0 if ratio > 1.0 else ratio

        resource = Resource.create(
            {
                "service.name": runtime_config.otel_service_name,
                "service.instance.id": uuid7_str(),
            }
        )
        provider = TracerProvider(resource=resource, sampler=ParentBased(TraceIdRatioBased(ratio)))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(provider)
        _TRACING_CONFIGURED = True

        def _shutdown_provider() -> None:
            try:
                provider.shutdown()
            except Exception as e:
                logger.warning(f"Tracing provider shutdown failed: {e!r}")

        shutdown_callbacks.append(_shutdown_provider)

    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        except Exception as e:
            logger.warning(f"FastAPI tracing instrumentation unavailable: {e!r}")
        else:
            FastAPIInstrumentor.instrument_app(app)

    if not _HTTPX_INSTRUMENTED:
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        except Exception as e:
            logger.warning(f"HTTPX tracing instrumentation unavailable: {e!r}")
        else:
            HTTPXClientInstrumentor().instrument()
            _HTTPX_INSTRUMENTED = True

            def _shutdown_httpx() -> None:
                try:
                    HTTPXClientInstrumentor().uninstrument()
                except Exception:
                    return

            shutdown_callbacks.append(_shutdown_httpx)

    def _shutdown() -> None:
        for cb in shutdown_callbacks:
            try:
                cb()
            except Exception:
                continue

    return _shutdown
