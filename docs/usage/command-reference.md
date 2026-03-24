# Command Reference

Comprehensive reference for the `agentic-stacks` CLI.

## Synopsis

```bash
agentic-stacks serve [OPTIONS]
```

## Description

`agentic-stacks serve` is the remote-upstream gateway supervisor.

It does not spawn vLLM. For the single-command colocated local stack, use `vllm serve --responses`.

It also manages:

1. the **Code Interpreter** runtime (unless disabled),
1. the singleton **Built-in MCP** runtime process (when `--mcp-config` is set).

______________________________________________________________________

## Options

### Upstream Configuration

These options control where the gateway finds the inference server.

#### `--upstream URL`

**Description**: Exact base URL of an external OpenAI-compatible server. **Default**: `None` **Example**: `--upstream http://127.0.0.1:8000/v1` **Notes**: Provide the exact API base URL you want the gateway to call.

#### `--upstream-ready-timeout SECONDS`

**Description**: Maximum time to wait for the upstream readiness check to succeed. **Default**: `1800` (30 minutes)

#### `--upstream-ready-interval SECONDS`

**Description**: Polling interval between upstream readiness checks. **Default**: `5`

### Gateway Configuration

These options control the `agentic-stacks` server itself.

#### `--gateway-host HOST`

**Description**: The interface to bind the gateway server to. **Default**: `0.0.0.0`

#### `--gateway-port PORT`

**Description**: The port to listen on. **Default**: `5969`

#### `--gateway-workers N`

**Description**: Number of Gunicorn workers to spawn. **Default**: `1` **Notes**: For production, use multiple workers (e.g., `2 * CPU_CORES + 1`).

### Web Search Configuration

#### `--web-search-profile PROFILE`

**Description**: Enable the gateway-owned `web_search` built-in using the selected profile. **Default**: disabled **Example**: `--web-search-profile exa_mcp`

**Notes**:

- This is a gateway-owned feature-selection flag.
- For `agentic-stacks serve`, web search enablement is CLI-owned.
- If the flag is omitted, `web_search` is disabled.
- Shipped MCP-backed profiles provision their helper runtime entries automatically.
- For the shipped `exa_mcp` profile, setting `EXA_API_KEY` in the gateway environment appends the operator key to the default Exa MCP URL automatically.
- `--mcp-config` remains optional for generic MCP inventory and explicit helper overrides; it is not required just to make shipped `web_search` profiles work.

### Code Interpreter Configuration

#### `--code-interpreter MODE`

**Description**: Runtime policy for the code interpreter. **Default**: `spawn` **Values**:

- `spawn`: The `agentic-stacks serve` supervisor starts and manages the Bun/Pyodide server, then wires gateway workers to it.
- `external`: Connects to an already-running server (supervisor does not spawn one).
- `disabled`: Disables the tool entirely.

!!! note "Developer-only fallback"

    On platforms without a bundled Code Interpreter binary (or when running from a source checkout), you can allow a
    Bun-based fallback by setting `VR_CODE_INTERPRETER_DEV_BUN_FALLBACK=1`. This is intended for development.

#### `--code-interpreter-port PORT`

**Description**: Port for the code interpreter server. **Default**: `5970`

#### `--code-interpreter-workers N`

**Description**: Worker pool size for the code interpreter service when `--code-interpreter=spawn`. **Default**: `0` (in-process; no workers) **Notes**:

- This uses a [Bun Worker](https://bun.com/docs/runtime/workers) pool.
- Use `2+` for actual parallelism. `1` enables worker mode but does not increase throughput.
- Each worker loads its own Pyodide runtime, so increasing workers increases RAM and startup time.

#### `--code-interpreter-startup-timeout SECONDS`

**Description**: Maximum time to wait for the code interpreter to become ready. **Default**: `600` (10 minutes)

______________________________________________________________________

## Configuration Precedence

`agentic-stacks serve` resolves config in this order:

1. CLI flags
1. Built-in defaults

Deployment-scoped environment variables such as storage, metrics, tracing, auth, and cache remain separate from this CLI surface.

Gateway-owned feature-selection flags on this command, including `--web-search-profile`, do not use environment-variable fallback.

Built-in MCP runtime configuration is CLI-owned in this command:

- Use `--mcp-config /path/to/mcp.json` to enable Built-in MCP.
- Use `--mcp-port PORT` to override the loopback port used by the Built-in MCP runtime.
- When enabled, `serve` starts one loopback Built-in MCP runtime and injects `VR_MCP_BUILTIN_RUNTIME_URL` into gateway workers.
- If `--mcp-port` is absent, `serve` uses `http://127.0.0.1:5981`.

Upstream selection precedence:

1. `--upstream` (external upstream exact API base URL).
1. Otherwise: configuration error ("no upstream configured").

## Examples

### Connect to External Server

```bash
--8<-- "snippets/serve_external_upstream_cmd.txt"
```

### Use the Colocated Single-Command Mode

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct --responses
```

### Run the Remote-Upstream Gateway With More Workers

```bash
agentic-stacks serve \
  --gateway-workers 4 \
  --upstream http://127.0.0.1:8000/v1
```

### Enable Web Search

```bash
agentic-stacks serve \
  --upstream http://127.0.0.1:8000/v1 \
  --web-search-profile exa_mcp
```
