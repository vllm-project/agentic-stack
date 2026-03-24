# Configuration Reference

The gateway uses both CLI flags and environment variables.

- CLI owns operator-facing runtime topology and helper wiring on supported entrypoints.
- Environment variables own deployment-scoped settings, secrets, and process-level integrations.

All documented environment variables are prefixed with `VR_`.

## Core Configuration

| Variable                    | Description                                                                                  | Default |
| :-------------------------- | :------------------------------------------------------------------------------------------- | :------ |
| **`VR_MAX_CONCURRENCY`**    | Gunicorn/Uvicorn concurrency limit for standalone gateway startup.                           | `300`   |
| **`VR_LOG_TIMINGS`**        | Enable logging of request timings and overhead.                                              | `False` |
| **`VR_LOG_MODEL_MESSAGES`** | Enable logging of model-facing messages for debugging.                                       | `False` |
| **`VR_OPENAI_API_KEY`**     | Upstream bearer token used when the gateway or proxy path must authenticate to the upstream. | (unset) |

Notes:

- Supported entrypoints use CLI flags for upstream selection, bind address, worker count, and helper wiring.
- Integrated mode (`vllm serve --responses`) uses native vLLM `--host` / `--port` and requires a single API server.
- `VR_MAX_CONCURRENCY` applies to direct standalone startup paths used mainly for development/tests.

## Storage Configuration

| Variable                                  | Description                                                                        | Default                                 |
| :---------------------------------------- | :--------------------------------------------------------------------------------- | :-------------------------------------- |
| **`VR_DB_PATH`**                          | Database connection string. Use `sqlite+aiosqlite:///` or `postgresql+asyncpg://`. | `sqlite+aiosqlite:///agentic_stack.db` |
| **`VR_RESPONSE_STORE_CACHE`**             | Enable Redis caching for the ResponseStore.                                        | `False`                                 |
| **`VR_RESPONSE_STORE_CACHE_TTL_SECONDS`** | Cache TTL in seconds.                                                              | `3600`                                  |
| **`VR_REDIS_HOST`**                       | Redis host (if cache enabled).                                                     | `localhost`                             |
| **`VR_REDIS_PORT`**                       | Redis port.                                                                        | `6379`                                  |

## Code Interpreter Configuration

| Variable                                   | Description                                                                         | Default    |
| :----------------------------------------- | :---------------------------------------------------------------------------------- | :--------- |
| **`VR_PYODIDE_CACHE_DIR`**                 | Directory for the Pyodide runtime cache (download + extracted files).               | (see docs) |
| **`VR_CODE_INTERPRETER_DEV_BUN_FALLBACK`** | Development-only: if `1`, allow `bun` fallback when no bundled binary is available. | `0`        |

Notes:

- Supported entrypoints use CLI flags for code-interpreter mode, port, workers, and startup timeout.
- `0` (default) runs **in-process** (no Bun Workers): single-threaded execution.
- `1` enables the WorkerPool path, but does not add parallelism (useful mainly to validate worker mode).
- `2+` enables parallel execution via Bun Workers (experimental).
- Each worker initializes its own Pyodide runtime, so RAM usage and startup time scale with worker count.

## MCP Configuration (Built-in + Remote)

| Variable                                | Description                                                                                     | Default |
| --------------------------------------- | ----------------------------------------------------------------------------------------------- | ------- |
| **`VR_MCP_REQUEST_REMOTE_ENABLED`**     | Enable Remote MCP (`tools[].mcp.server_url`) handling.                                          | `True`  |
| **`VR_MCP_REQUEST_REMOTE_URL_CHECKS`**  | Enable Remote MCP URL policy checks (`https`, denylist hosts).                                  | `True`  |
| **`VR_MCP_HOSTED_STARTUP_TIMEOUT_SEC`** | Built-in MCP startup/discovery timeout in seconds (applies to all hosted servers).              | `10`    |
| **`VR_MCP_HOSTED_TOOL_TIMEOUT_SEC`**    | Built-in MCP call timeout in seconds (applies to all hosted servers).                           | `60`    |
| **`EXA_API_KEY`**                       | Optional Exa API key appended to the shipped `exa_mcp` helper URL when that profile is enabled. | (unset) |

Built-in MCP enablement is CLI-owned on supported entrypoints:

- `agentic-stacks serve --mcp-config /path/to/mcp.json [--mcp-port PORT]`
- `vllm serve ... --responses --responses-mcp-config /path/to/mcp.json [--responses-mcp-port PORT]`

Remote-upstream supervisor readiness controls are also CLI-owned:

- `agentic-stacks serve --upstream-ready-timeout SECONDS`
- `agentic-stacks serve --upstream-ready-interval SECONDS`

If the MCP config flag is omitted, Built-in MCP is disabled.
If `VR_MCP_REQUEST_REMOTE_ENABLED=false`, Remote MCP declarations are rejected while Built-in MCP remains available.
If `VR_MCP_REQUEST_REMOTE_URL_CHECKS=false`, gateway URL policy checks are fully disabled for Remote MCP declarations.

For the canonical `mcp.json` examples (URL + stdio styles), see
[MCP Examples -> Built-in MCP Runtime Config](../examples/hosted-mcp-examples.md#built-in-mcp-runtime-config-mcpjson).

Notes:

- Labels under `mcpServers` are request-visible `server_label` values.
- `EXA_API_KEY` is not a `VR_`-prefixed gateway setting because it is passed through to the upstream Exa MCP helper contract directly.
- Built-in MCP supports two server entry shapes:
    - URL-based HTTP: `url` (required, accepts `http://` or `https://`), `headers` (optional), `transport` (optional).
    - Command-style stdio: `command` (required), `args`/`env`/`cwd` (optional), `transport` optional but only `"stdio"`.
- Nested `transport` objects are rejected (for example, `"transport": {"type":"stdio", ...}`).
- `transport: "stdio"` without command-style keys is rejected.
- Mixing HTTP and stdio keys in one entry (for example `command` + `url`) is rejected.
- Hosted startup and tool timeouts are configured globally with:
    - `VR_MCP_HOSTED_STARTUP_TIMEOUT_SEC`
    - `VR_MCP_HOSTED_TOOL_TIMEOUT_SEC`
- Unknown non-runtime server fields are forwarded to FastMCP.
- In supported entrypoints, Built-in MCP always binds on loopback. The CLI port flags control only the port.

## Observability Configuration

| Variable                      | Description                                                           | Default          |
| :---------------------------- | :-------------------------------------------------------------------- | :--------------- |
| **`VR_METRICS_ENABLED`**      | Enable Prometheus-compatible metrics and the `GET /metrics` endpoint. | `True`           |
| **`VR_METRICS_PATH`**         | Metrics endpoint path.                                                | `/metrics`       |
| **`VR_TRACING_ENABLED`**      | Enable OpenTelemetry tracing (OTLP gRPC exporter).                    | `False`          |
| **`VR_OTEL_SERVICE_NAME`**    | Service name used in OpenTelemetry resources.                         | `agentic-stacks` |
| **`VR_TRACING_SAMPLE_RATIO`** | Trace sampling ratio in `[0.0, 1.0]` (ratio-based).                   | `0.01`           |
| **`VR_OPENTELEMETRY_HOST`**   | OTLP endpoint host (gRPC).                                            | `otel-collector` |
| **`VR_OPENTELEMETRY_PORT`**   | OTLP endpoint port (gRPC).                                            | `4317`           |

## Example Configurations

### Local Development (Default)

```bash
export VR_DB_PATH="sqlite+aiosqlite:///agentic_stack.db"
agentic-stacks serve --upstream http://127.0.0.1:8000/v1
```

### Production with PostgreSQL & Redis

```bash
export VR_DB_PATH="postgresql+asyncpg://user:pass@db-host:5432/agentic_stack"
export VR_RESPONSE_STORE_CACHE=1
export VR_REDIS_HOST="redis-host"
agentic-stacks serve \
  --upstream http://vllm-service:8000/v1 \
  --gateway-workers 8
```

### Enable Built-in MCP

```bash
agentic-stacks serve \
  --upstream http://127.0.0.1:8000/v1 \
  --mcp-config /etc/agentic-stacks/mcp.json
```

### Integrated Mode With Built-in MCP

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --responses \
  --responses-mcp-config /etc/agentic-stacks/mcp.json
```
