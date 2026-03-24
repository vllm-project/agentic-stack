# Python Sandboxed REPL

A sandboxed Python REPL that compiles to a single binary, run entirely using WebAssembly. No installation required!

## What is this?

This project lets you run Python code in a secure, sandboxed environment using [Pyodide](https://pyodide.org/). It's perfect for:

- Running Python code without installing Python
- Creating a portable, self-contained Python environment
- Experimenting with Python in a controlled sandbox
- Distributing a Python REPL as a single executable file

## Features

- **No Python installation needed** - Everything runs in WebAssembly
- **Pre-loaded popular packages** including:
  - Data science: `numpy`, `pandas`, `matplotlib`, `scikit-image`
  - HTTP requests: `requests`, `httpx`, `aiohttp`
  - Image processing: `Pillow`, `opencv-python`
  - Data formats: `beautifulsoup4`, `pyyaml`, `orjson`
  - Math & symbolic: `sympy`, `tiktoken`
  - And many more!
- **HTTP support** - Make real HTTP requests from Python
- **Interactive REPL** - Standard Python prompt
- **HTTP API Server** - Expose Python execution via REST endpoints
- **Compile to binary** - Bundle everything into a single executable

## Prerequisites

- [Bun](https://bun.sh) runtime installed

## Quick Start

### Run the REPL without Compiling

```shell
# Install dependencies
bun install

# Run the REPL
bun src/index.ts

# To use another cache directory
bun src/index.ts --pyodide-cache my-pyodide-cache

# To reset global context after every call
bun src/index.ts --reset-globals
```

On first run, Pyodide (~420MB) will be automatically downloaded and extracted to `~/.pyodide-env`. Subsequent runs will use the cached files.

You'll see a Python prompt where you can enter code:

```python
>>> print("Hello from Python!")
Hello from Python!

>>> import numpy as np
>>> np.array([1, 2, 3]).mean()
2.0

>>> import httpx
>>> httpx.get("https://api.github.com").status_code
200
```

Press `Ctrl+C` to exit.

## CLI Options

### `--reset-globals`

When enabled, each line of Python code executes in a fresh, isolated context that is destroyed after execution. This means variables and state are not preserved between executions.

```bash
bun src/index.ts --reset-globals
```

```python
>>> x = 0
>>> x
NameError("name 'x' is not defined")
```

Without this flag (default behavior), the REPL maintains state between executions, allowing you to define variables and reuse them in subsequent commands.

```python
>>> x = 0
>>> x
0
```

### `--pyodide-cache <path>`

Specifies the directory where Pyodide files are cached. Defaults to `~/.pyodide-env`.

```bash
bun src/index.ts --pyodide-cache my-custom-cache
```

The `~` character is automatically expanded to your home directory. This is useful if you want to:

- Use a different cache location
- Share a Pyodide installation across multiple projects
- Store the cache on a different disk

### `--port <number>`

Start an HTTP API server instead of the interactive REPL. The server exposes Python execution via REST endpoints.

```bash
# Start server on port 3000
bun src/index.ts --port 3000

# With custom options
bun src/index.ts --port 3000 --reset-globals --pyodide-cache ~/my-cache
```

**Note:** The server takes 5-10 seconds to initialize Pyodide on startup. Once ready, all subsequent requests are fast.

### `--workers <number>`

Enable multi-threaded execution using Bun Workers when running in server mode. Each worker runs its own Pyodide instance, enabling parallel Python code execution.

```bash
# Start server with 4 workers for parallel execution
bun src/index.ts --port 3000 --workers 4
```

**Important:** This flag only works in server mode (requires `--port`).

**Benefits:**

- Parallel request processing across multiple CPU cores
- Higher throughput for concurrent requests
- Better resource utilization

**Behavior:**

- With 1 worker: Executions respect the `reset_globals` flag (no parallelism)
- With 2+ workers: All code executions automatically use `reset_globals=true` for predictable, isolated execution
- Each worker maintains its own independent Python environment
- Requests are randomly distributed across available workers

**Worker Count Guidelines:**

- `0` (default): Single-threaded mode, lowest memory (~500MB)
- `1`: Single worker mode, allows stateful execution with `reset_globals=false` (~500MB)
- `2-4`: Parallel execution for moderate concurrent load (~1-2GB)
- `8+`: High concurrent load, requires significant RAM (~4GB+)

**Memory Considerations:** Each worker uses approximately 500MB of memory. Plan accordingly:

- 2 workers = ~1GB
- 4 workers = ~2GB
- 8 workers = ~4GB

## HTTP API Server Mode

When started with the `--port` flag, the application runs as an HTTP API server that executes Python code via REST endpoints. Pyodide is initialized once as a global singleton to avoid cold start issues.

### Endpoints

#### `GET /health`

Returns server health status and Pyodide initialization state.

**Response:**

```json
{
  "status": "healthy",
  "pyodide_loaded": true,
  "uptime_seconds": 3600,
  "execution_count": 42
}
```

#### `POST /python`

Execute Python code and return the result.

**Request:**

```json
{
  "code": "1 + 1",
  "reset_globals": false
}
```

- `code` (required): Python code to execute
- `reset_globals` (optional): If `true`, execute in a fresh isolated context. Defaults to server's `--reset-globals` flag setting.

**Response (200):**

For expressions that return a value, eg `x = 2; x`:

```json
{
  "status": "success",
  "stdout": "",
  "stderr": "",
  "result": "2",
  "execution_time_ms": 5
}
```

For statements (like assignments) that don't return a value, eg `x = 2`:

```json
{
  "status": "success",
  "stdout": "",
  "stderr": "",
  "result": null,
  "execution_time_ms": 4
}
```

Python errors are returned with `exception` status and with the error message in the result:

```json
{
  "status": "exception",
  "stdout": "",
  "stderr": "",
  "result": "NameError: name 'x' is not defined",
  "execution_time_ms": 5
}
```

### API Examples

**Health Check:**

```bash
curl http://localhost:3000/health
```

**Execute Python Code:**

```bash
curl -X POST http://localhost:3000/python \
  -H "Content-Type: application/json" \
  -d '{"code": "import numpy as np; np.array([1,2,3]).mean()"}'
```

**Execute with Fresh Context:**

```bash
curl -X POST http://localhost:3000/python \
  -H "Content-Type: application/json" \
  -d '{"code": "x = 42", "reset_globals": true}'
```

**Test HTTP Libraries:**

```bash
curl -X POST http://localhost:3000/python \
  -H "Content-Type: application/json" \
  -d '{"code": "import httpx; httpx.get(\"https://httpbin.org/json\").json()"}'
```

### API Server Features

- **Global Pyodide Instance**: Initialized once on startup to avoid cold start delays
- **Parallel Execution**: Optional worker-based execution with `--workers` flag enables concurrent Python code execution across multiple CPU cores
- **Thread Safety**: Single-threaded mode processes requests serially; multi-worker mode processes requests in parallel with isolated contexts
- **CORS Enabled**: Cross-origin requests are allowed
- **Error Handling**: Python exceptions are captured and returned with proper HTTP status codes
- **Execution Tracking**: Health endpoint reports total execution count

### Build Standalone Binary

Compile the REPL into a single `woma` executable:

```bash
# Install dependencies
bun install
# Compile
bun run build
# Run
./woma

# Or run with options
./woma --reset-globals --pyodide-cache ~/my-custom-cache
```

The compiled binary is fully self-contained and can be distributed without requiring Bun or any other dependencies. It supports the same CLI options as the uncompiled version.

### Running Tests

Run the test suite:

```bash
bun test
```

Tests verify:

- Basic Python execution (1+1)
- HTTP requests with httpx
- HTTP API server endpoints (`/health`, `/python`)
- Error handling and reset_globals functionality
- Compiled binary functionality

## Technical Details

### XMLHttpRequest Polyfill

The project includes a custom `XMLHttpRequest` wrapper that:

- Enables synchronous XHR (required by some Python packages)
- Filters forbidden HTTP headers that cause warnings

### HTTP Library Patches

Several patches are applied to make HTTP libraries work in Bun:

- **pyodide-http**: Patches urllib, requests, and aiohttp
- **urllib3**: Disables Node.js detection
- **httpx**: Converts URL objects to strings to avoid proxy errors

## Project Structure

```
.
├── src/
│   ├── index.ts              # Main entry point (CLI mode)
│   ├── index.test.ts         # CLI test suite
│   ├── server.ts             # HTTP API server
│   ├── server.test.ts        # Server integration tests
│   ├── pyodide-manager.ts    # Shared Pyodide initialization and execution
│   ├── types.ts              # Shared TypeScript interfaces
│   └── xmlhttprequest-ssl.d.ts # TypeScript definitions
├── pyodide-env/              # Pyodide distribution (auto-downloaded on first run)
├── package.json              # Dependencies and scripts
└── woma                      # Compiled binary (after build)
```

## Limitations

- **Startup time**: First launch downloads Pyodide and takes additional time; subsequent launches take ~5-10 seconds to load Python runtime
- **Package size**: Pyodide distribution is ~420MB
- **No native extensions**: Only pure Python or pre-compiled WASM packages work
- **No filesystem**: Uses virtual filesystem
- **Memory**: [Limited to 2GB](https://github.com/pyodide/pyodide/issues/1513#issuecomment-823841440)

## License

This project is licensed under Apache 2.0.

Pyodide is [licensed under the Mozilla Public License Version 2.0](https://github.com/pyodide/pyodide/blob/main/LICENSE).

## Credits

Built with:

- [Pyodide](https://pyodide.org/)
- [Bun](https://bun.sh)
- [xmlhttprequest-ssl](https://github.com/mjwwit/node-XMLHttpRequest)
