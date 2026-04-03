# agentic-api
Stateful API logic for agentic applications using vLLM

A lightweight gateway that sits in front of vLLM and exposes the OpenAI Responses API (`POST /v1/responses`).

## Installation

```bash
uv sync
```

## Running the gateway

### Integrated mode (recommended)

Start vLLM and the gateway together with a single command. The `--agentic-api` flag activates the gateway; all other flags are passed through to vLLM.

```bash
vllm serve <MODEL> --agentic-api [--port <VLLM_PORT>] [--gateway-port <GATEWAY_PORT>]
```

Example:

```bash
vllm serve Qwen/Qwen3.5-0.8B --agentic-api --port 8000 --gateway-port 9000 --host 0.0.0.0
```

The gateway starts on port `9000` by default and waits up to 10 minutes for vLLM to become ready before accepting traffic.

### Standalone mode

If vLLM is already running separately, point the gateway at it:

```bash
agentic-api --llm-api-base http://127.0.0.1:8000
```

The `/v1` suffix on `--llm-api-base` is optional.

### Gateway options

| Flag | Default | Description |
|---|---|---|
| `--llm-api-base` | *(required in standalone)* | Base URL of the upstream vLLM server |
| `--openai-api-key` | `None` | API key forwarded to upstream |
| `--gateway-host` | `0.0.0.0` | Host the gateway binds to |
| `--gateway-port` | `9000` | Port the gateway listens on |
| `--gateway-workers` | `1` | Uvicorn worker count |
| `--upstream-ready-timeout` | `600` | Seconds to wait for vLLM to become ready |
| `--upstream-ready-interval` | `2` | Poll interval in seconds |

## Sending requests

The gateway forwards requests to `/v1/responses` directly to vLLM.

```bash
curl http://127.0.0.1:9000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-0.8B",
    "input": [{"role": "user", "content": "Hello!"}]
  }'
```

Streaming:

```bash
curl http://127.0.0.1:9000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-0.8B",
    "input": [{"role": "user", "content": "Count from 1 to 5."}],
    "stream": true
  }'
```

## Development

Run tests:

```bash
uv run pytest
```

Lint and format:

```bash
ruff check --fix .
ruff format .
```
