# Observability

Monitor the gateway using Prometheus metrics and optional OpenTelemetry tracing.

## Overview

The gateway exposes operational data through:

- **Prometheus metrics** at `GET /metrics` (enabled by default)
- **Health checks** at `GET /health` for load balancer probes
- **OpenTelemetry tracing** via OTLP export (optional, disabled by default)

## Prometheus Metrics

### Scraping Configuration

The metrics endpoint is unauthenticated and available at:

- **Endpoint**: `GET /metrics` (configurable via `VR_METRICS_PATH`)
- **Port**: Same as the active public API listener
    - `agentic-stacks serve`: default `5969`
    - `vllm serve --responses`: default `8000` unless overridden by vLLM flags
- **Format**: Prometheus text exposition format

If you change `VR_METRICS_PATH`, update your Prometheus `metrics_path` accordingly.

Example Prometheus `scrape_configs`:

```yaml
scrape_configs:
  # Gateway metrics
  - job_name: "agentic-stacks"
    static_configs:
      - targets: ["gateway-host:5969"]
    metrics_path: "/metrics"
    scrape_interval: 15s

  # Also scrape vLLM metrics (if running locally)
  - job_name: "vllm"
    static_configs:
      - targets: ["localhost:8457"]
    metrics_path: "/metrics"
    scrape_interval: 15s
```

!!! note "Also scrape vLLM"

    Remember to scrape your vLLM instance separately (it is a different service, with its own metrics). See [vLLM metrics documentation](https://docs.vllm.ai/en/stable/design/metrics/) for details.

### Multi-Worker Support

When running with multiple workers (`--gateway-workers N`), metrics are automatically aggregated across all workers when using `agentic-stacks serve`.

**Using `agentic-stacks serve` (recommended):**

Multi-process metrics are handled automatically. The supervisor creates a temporary directory for Prometheus multi-process mode and configures the workers correctly.

We intentionally do not document manual multi-worker ASGI server setups here (Gunicorn/Uvicorn/etc.), because correct multi-process Prometheus aggregation requires additional lifecycle wiring (per-run multiprocess directories and worker-exit cleanup hooks).

### Available Metrics

All metrics use the `agentic_stack_` prefix.

#### HTTP Metrics

| Metric                                         | Type      | Labels                      | Description                                                          |
| ---------------------------------------------- | --------- | --------------------------- | -------------------------------------------------------------------- |
| `agentic_stack_http_requests_total`           | Counter   | `method`, `route`, `status` | Total HTTP requests completed                                        |
| `agentic_stack_http_request_duration_seconds` | Histogram | `method`, `route`           | HTTP handler duration (excludes SSE stream lifetime)                 |
| `agentic_stack_http_in_flight_requests`       | Gauge     | -                           | Requests currently being handled (does not include open SSE streams) |

#### SSE Streaming Metrics

| Metric                                       | Type      | Labels  | Description                                                                                |
| -------------------------------------------- | --------- | ------- | ------------------------------------------------------------------------------------------ |
| `agentic_stack_sse_connections_in_flight`   | Gauge     | -       | SSE connections currently open                                                             |
| `agentic_stack_sse_stream_duration_seconds` | Histogram | `route` | Full SSE stream lifetime from request start (including time-to-first-chunk) to termination |

SSE metrics capture the end-to-end streaming duration for the `/v1/responses` endpoint, which is the primary API path.

#### Tool Metrics

| Metric                                           | Type      | Labels                   | Description                          |
| ------------------------------------------------ | --------- | ------------------------ | ------------------------------------ |
| `agentic_stack_tool_calls_requested_total`      | Counter   | `tool_type`              | Tool calls requested by the model    |
| `agentic_stack_tool_calls_executed_total`       | Counter   | `tool_type`              | Tool calls executed by the gateway   |
| `agentic_stack_tool_execution_duration_seconds` | Histogram | `tool_type`              | Tool execution wall-clock duration   |
| `agentic_stack_tool_errors_total`               | Counter   | `tool_type`              | Tool execution errors                |
| `agentic_stack_mcp_server_startup_total`        | Counter   | `server_label`, `status` | Built-in MCP server startup outcomes |

**Tool types:**

- `function` - Client-executed function tools (request count only)
- `code_interpreter` - Gateway-executed code interpreter (request + execution/error metrics)
- `web_search` - Gateway-executed web search tool calls (request + execution/error metrics)
- `mcp` - MCP tool calls requested by the model (covers Built-in MCP and Remote MCP modes)

Notes:

- MCP metrics are not split by mode (Built-in MCP vs Remote MCP) in current metrics.
- `agentic_stack_mcp_server_startup_total` is hosted-only (there is no startup metric for Client-Specified Remote declarations).

#### Metric Labels

Labels are bounded to prevent cardinality explosion:

- `method`: HTTP method (GET, POST, etc.)
- `route`: Route template (e.g., `/v1/responses`), not raw paths
- `status`: HTTP status code (200, 400, 500, etc.)
- `tool_type`: Tool category (`function`, `code_interpreter`, `web_search`, `mcp`)
- `server_label`: Built-in MCP server label configured in runtime config
- `status` (MCP startup metric): `ok` or `error`

## OpenTelemetry Tracing

Tracing is disabled by default. Enable it for distributed tracing of requests through the gateway.

### Installation (optional dependencies)

OpenTelemetry tracing requires the `tracing` optional dependencies.

### Configuration

Set the following environment variables:

```bash
# Enable tracing
export VR_TRACING_ENABLED=true

# Sampling ratio (0.0 to 1.0)
export VR_TRACING_SAMPLE_RATIO=0.01

# OTLP endpoint (gRPC)
export VR_OPENTELEMETRY_HOST=otel-collector
export VR_OPENTELEMETRY_PORT=4317

# Service name in traces
export VR_OTEL_SERVICE_NAME=agentic_stack
```

### What Gets Traced

When enabled, the gateway instruments:

- **Inbound HTTP requests** (FastAPI/ASGI)
- **Outbound HTTP to vLLM** (HTTPX client)

### Prerequisites

You need an OTLP-compatible collector or backend to receive traces:

- [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/)
- [Jaeger](https://www.jaegertracing.io/)
- [Tempo](https://grafana.com/docs/tempo/)
- Any other OTLP gRPC-compatible backend

Example minimal OpenTelemetry Collector configuration:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [jaeger]
```

## Health Checks

The gateway exposes a health endpoint for load balancer health probes:

- **Endpoint**: `GET /health`
- **Response**: `200 OK` with empty JSON object `{}`
- **Authentication**: None

Example:

```bash
curl http://localhost:5969/health
```

In integrated mode, use the `vllm serve` host/port instead (for example `http://localhost:8000/health` with defaults).

Use this endpoint for:

- Kubernetes liveness and readiness probes
- AWS ALB/NLB health checks
- Docker health checks
- Load balancer health monitoring

## Configuration Reference

For a complete list of observability environment variables and their defaults, see the [Configuration Reference](../reference/configuration.md).
