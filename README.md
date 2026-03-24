# vLLM Agentic Stack

FastAPI gateway that exposes an OpenAI-style **Responses API** (`/v1/responses`) in front of a vLLM **OpenAI-compatible** server (`/v1/chat/completions`), with:

- SSE streaming event shape + ordering
- `previous_response_id` statefulness (ResponseStore)
- gateway-executed built-in tool: `code_interpreter`
- gateway-hosted MCP tools (`tools[].type="mcp"` with configured `server_label`)

Current MCP boundary:

- `tools[].type="mcp"` is gateway-hosted MCP resolved via `VR_MCP_CONFIG_PATH`.
- Request-declared MCP targets (`server_url`, `connector_id`) are not supported yet.

**[📚 Full User Documentation](https://vllm-project.github.io/agentic-stack/)** (Guides, API Reference, Examples)

Design docs (maintainer-facing): `design_docs/index.md`.

## Install

The `agentic-stack` CLI is provided by the Python package in `responses/`.

**Prerequisites:** Python 3.12+ and `uv`.

### Install from a prebuilt wheel (Linux x86_64) (Recommended)

Download a prebuilt wheel (`agentic_stack-*.whl`) from GitHub Releases (preferred) or a CI run artifact, then install it:

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install vllm
uv pip install path/to/agentic_stack-*.whl
```

On Linux x86_64 wheels, the Code Interpreter server binary is bundled, so **Bun is not required**.
Currently, wheels are only built for Linux x86_64.

Installing `agentic-stack` provides:

- `agentic-stack` for the standalone supervisor mode
- `vllm` as a CLI shim that supports `vllm serve --responses` and delegates all non-Responses paths to the upstream
  `vllm` Python package

### Install from source (repo checkout) (Development)

```bash
git clone https://github.com/vllm-project/agentic-stack.git
cd agentic-stack

uv venv --python=3.12
source .venv/bin/activate
uv pip install vllm
uv pip install -e ./responses

# Development: enable Code Interpreter via Bun fallback
# - Required for source checkouts when running with `code_interpreter` enabled (default)
cd responses/python/agentic_stack/tools/code_interpreter
bun install
export VR_CODE_INTERPRETER_DEV_BUN_FALLBACK=1
cd -

agentic-stack --help
```

Verify installation:

```bash
agentic-stack --help
vllm --help
```

### Optional dependency sets (extras)

Install any combination via:

```bash
uv pip install -e './responses[<extra1>,<extra2>]'
```

Available extras:

- `docs`: MkDocs toolchain (contributors).
- `lint`: Ruff + Markdown formatting.
- `test`: Pytest + coverage + load testing tools.
- `tracing`: OpenTelemetry tracing support (only needed if you enable `VR_TRACING_ENABLED=true`).
- `build`: Package build/publish tools.
- `all`: Everything above.

## Build a wheel from source

If you want to produce a local wheel from this checkout, build from the
`responses/` package directory.

### Rebuild the bundled Code Interpreter binary (Linux x86_64 only)

This step is only needed if you want the wheel to include a freshly compiled
Code Interpreter binary.

```bash
bash scripts/ci/prebuild_code_interpreter_linux_x86_64.sh responses
```

The script writes the bundled executable under:

- `responses/python/agentic_stack/tools/code_interpreter/bin/linux/x86_64/code-interpreter-server`

### Build wheel and sdist

```bash
uv pip install -e './responses[build]'
cd responses
python -m build --wheel --sdist
```

Build artifacts are written to:

- `responses/dist/`

On Linux x86_64, wheels built after the prebuild step bundle the native Code
Interpreter binary. On other platforms, use the source-install Bun fallback or
disable Code Interpreter.

## Run

### remote-upstream gateway mode (`agentic-stack serve`)

Prereqs:

- If `code_interpreter` is enabled (default), the first start may download the Pyodide runtime (~400MB) into a cache
    directory (see `VR_PYODIDE_CACHE_DIR`). This requires `tar` to be installed.
- For non-Linux platforms (or source installs without the bundled binary), you can disable the tool via
    `--code-interpreter disabled`. For development you can also enable the Bun-based fallback via
    `VR_CODE_INTERPRETER_DEV_BUN_FALLBACK=1`.

External upstream (you start vLLM yourself; `/v1` is optional):

```bash
agentic-stack serve --upstream http://127.0.0.1:8457
```

The Responses endpoint is:

- `POST http://127.0.0.1:5969/v1/responses`

Remote access note:

- If you bind the gateway with `--gateway-host 0.0.0.0`, use the machine’s IP/hostname to connect (not `0.0.0.0`).

### integrated runtime (`vllm serve --responses`)

Prereq:

- install upstream `vllm` first, then install `agentic-stack` into the same environment

Example:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-0.8B \
  --responses \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 \
  --port 8457
```

CLI help:

- `vllm serve --help` shows upstream vLLM help
- `vllm serve --responses --help` shows the Responses-owned integrated flags

### Optional: ResponseStore hot cache (Redis)

`previous_response_id` hydration reads the previous response state from the DB. For multi-worker deployments, you can optionally enable a Redis-backed hot cache to reduce DB reads/latency.

Env vars (default off):

- `VR_RESPONSE_STORE_CACHE=1`
- `VR_RESPONSE_STORE_CACHE_TTL_SECONDS=3600`

Redis connection:

- `VR_REDIS_HOST`, `VR_REDIS_PORT`

## Quick smoke test (OpenAI Python SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5969/v1", api_key="dummy")

with client.responses.stream(
    model="MiniMaxAI/MiniMax-M2.1",
    input=[{"role": "user", "content": "You MUST call the code_interpreter tool. Execute: 2+2. Reply with ONLY the number."}],
    tools=[{"type": "code_interpreter"}],
    tool_choice="auto",
    include=["code_interpreter_call.outputs"],
) as stream:
    for evt in stream:
        if getattr(evt, "type", "").endswith(".delta"):
            continue
        print(getattr(evt, "type", evt))
    r1 = stream.get_final_response().id

with client.responses.stream(
    model="MiniMaxAI/MiniMax-M2.1",
    previous_response_id=r1,
    input=[{"role": "user", "content": "What number did you just compute? Reply with ONLY the number."}],
    tool_choice="none",
) as stream:
    for evt in stream:
        if getattr(evt, "type", "").endswith(".delta"):
            continue
        print(getattr(evt, "type", evt))
```
