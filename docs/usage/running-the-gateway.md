# Running the Gateway

This guide covers the different ways to run `vLLM Responses` in various environments.

For a task-first reading order, start here, then use the
[Command Reference](command-reference.md) for the full option-by-option CLI surface.

## Supported entrypoint (important)

The repo has two first-class runtime modes:

1. `agentic-stacks serve`
1. `vllm serve --responses`

Direct startup via `uvicorn` or `gunicorn` remains useful for development and tests, but it is not the
recommended product entrypoint.

## Operational Modes

### 1. Remote-Upstream Gateway Mode

Use `agentic-stacks serve` when the upstream model server is already managed elsewhere. This is ideal for:

- External vLLM deployments.
- Cloud-hosted OpenAI-compatible endpoints.
- Restarting the gateway independently from the model server.
- Multi-worker gateway deployments.

```bash
agentic-stacks serve --upstream http://127.0.0.1:8000/v1
```

### 2. Integrated Colocated Mode

Use `vllm serve --responses` for the single-command local vLLM + gateway experience. This is ideal for:

- Local development.
- Demos and experimentation.
- Operators who want vLLM and the gateway on one public API server.

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct --responses
```

______________________________________________________________________

## Configuration

On the supported entrypoints, CLI flags own runtime topology and helper wiring.
Environment variables remain available for deployment-scoped settings such as storage, metrics, tracing, auth, and cache.

For `agentic-stacks serve`, the gateway-owned CLI surface is:

| CLI Flag                             | Description                           |
| ------------------------------------ | ------------------------------------- |
| `--upstream`                         | Exact upstream API base URL           |
| `--upstream-ready-timeout`           | Upstream readiness timeout            |
| `--upstream-ready-interval`          | Upstream readiness polling interval   |
| `--gateway-host`                     | Bind host                             |
| `--gateway-port`                     | Bind port                             |
| `--gateway-workers`                  | Number of workers                     |
| `--web-search-profile`               | Enable a shipped `web_search` profile |
| `--code-interpreter`                 | Code interpreter runtime policy       |
| `--code-interpreter-port`            | Code interpreter port                 |
| `--code-interpreter-workers`         | Code interpreter worker count         |
| `--code-interpreter-startup-timeout` | Code interpreter readiness timeout    |
| `--mcp-config`                       | Built-in MCP runtime config path      |
| `--mcp-port`                         | Built-in MCP runtime loopback port    |

When `--mcp-config` is set, `agentic-stacks serve` starts a singleton Built-in MCP runtime process shared by all gateway workers.
`--mcp-port` overrides the loopback runtime port.
If `--mcp-port` is absent, `serve` uses `http://127.0.0.1:5981`.
`--upstream-ready-timeout` and `--upstream-ready-interval` control how long the supervisor waits for the external upstream to become ready.
Shipped `web_search` profiles that require Built-in MCP helper servers provision their default helper entries automatically, so `--mcp-config` is not required just to enable a shipped profile.
For the shipped `exa_mcp` profile, setting `EXA_API_KEY` in the gateway
environment appends the operator key to the default Exa MCP URL automatically.

For `vllm serve --responses`, bind/public port are owned by native vLLM flags such as `--host` and `--port`.
Gateway-owned helper/runtime flags use the namespaced `--responses-*` family, for example:

- `--responses-code-interpreter`
- `--responses-code-interpreter-port`
- `--responses-code-interpreter-workers`
- `--responses-code-interpreter-startup-timeout`
- `--responses-web-search-profile`
- `--responses-mcp-config`
- `--responses-mcp-port`

See [Configuration Reference](../reference/configuration.md) for env-only deployment settings.

______________________________________________________________________

## Health Checks

The gateway exposes a health check endpoint useful for load balancers (AWS ALB, Kubernetes probes).

- **Endpoint**: `GET /health`
- **Response**: `200 OK` (JSON: `{}`)

Base URL by mode:

- `agentic-stacks serve`: `http://127.0.0.1:5969/health` by default
- `vllm serve --responses`: same host/port as `vllm serve` (default `http://127.0.0.1:8000/health`)

```bash
curl http://127.0.0.1:5969/health
```

The gateway also exposes a `/metrics` endpoint for Prometheus scraping. See [Observability](../deployment/observability.md) for monitoring setup instructions.

## Compatibility Passthrough Endpoints

When the gateway is configured with an upstream OpenAI-compatible base URL via `--upstream`, you can also call:

- `GET /v1/models`
- `POST /v1/chat/completions`

This allows older Chat Completions clients to point to the gateway base URL directly while `POST /v1/responses` remains available.

## Graceful Shutdown

The gateway handles `SIGINT` (Ctrl+C) and `SIGTERM` gracefully:

1. It stops accepting new connections.
1. It waits for active requests to complete (within a timeout).
1. It terminates the Code Interpreter subprocess (if spawned).
1. It terminates the Built-in MCP runtime subprocess (if started).
