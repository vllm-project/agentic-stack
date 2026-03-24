# Built-in Tools

The gateway can execute certain built-in tools on your behalf and stream the
results back through the Responses API.

Current built-in tools:

- **Code Interpreter**
- **Web Search**

For the dedicated `web_search` guide, see [Web Search](web-search.md).

## Code Interpreter

The Code Interpreter allows the model to write and execute Python code in a sandboxed environment. This is useful for:

- Mathematical calculations
- Data analysis and visualization
- String manipulation
- Solving logic puzzles

### Enabling the Tool

To use the code interpreter, you must:

1. Include it in the `tools` list with type `code_interpreter`.
1. (Optional but recommended) Add `code_interpreter_call.outputs` to the `include` list if you want to receive:
    - the captured stdout/stderr stream (e.g. `print(...)` output), and
    - the final expression display value (if any), in the response object.

```python
response = client.responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input=[{"role": "user", "content": "Calculate the 10th Fibonacci number."}],
    tools=[{"type": "code_interpreter"}],
    include=["code_interpreter_call.outputs"]
)
```

### Response Structure

When the model uses the code interpreter, the response will contain a `code_interpreter_call` output item.

```json
{
  "type": "code_interpreter_call",
  "id": "ci_123",
  "container_id": "pyodide-worker-1",
  "code": "def fib(n):...",
  "status": "completed",
  "outputs": [
    {
      "type": "logs",
      "logs": "P1\nP2\n"
    },
    {
      "type": "logs",
      "logs": "6"
    }
  ]
}
```

**Output types:**

- **Logs**: `{ "type": "logs", "logs": "stdout/stderr text" }`
- **Images**: `{ "type": "image", "url": "data:image/png;base64,..." }`

### Streaming Execution

One of the biggest benefits of the built-in runtime is **streaming**. You receive events _while the code is running_.

1. `response.code_interpreter_call.in_progress`: The tool call has started.
1. `response.code_interpreter_call_code.delta`: The model is writing the code.
1. `response.code_interpreter_call.interpreting`: The code is finished and is now executing.
1. `response.code_interpreter_call.completed`: Execution finished.
1. `response.output_item.done`: The item is finalized, including outputs.

### Security and Sandboxing

The code interpreter runs in a sandboxed environment:

- **Runtime**: [Pyodide](https://pyodide.org/) (Python compiled to WebAssembly) running inside a local Code Interpreter
    service. On Linux x86_64 wheels, the server is bundled as a native binary; for development/source installs it can run
    via [Bun](https://bun.sh/).
- **Isolation**: Runs in a WebAssembly sandbox with no direct host file system access.
- **Network Access**: HTTP requests are available via `httpx` (useful for API calls, data fetching).
- **Resource Limits**: Execution time is capped (configurable via startup flags).

!!! note "First start download"

    If the tool is enabled, the first start may download the Pyodide runtime (~400MB) into a cache directory and extract
    it. You can control the cache location via `VR_PYODIDE_CACHE_DIR`.

!!! note "Concurrency"

    If you need more code-interpreter throughput, you can configure a worker pool for the Code Interpreter service via
    `--code-interpreter-workers` under `agentic-stacks serve`, or
    `--responses-code-interpreter-workers` in integrated mode. This uses [Bun Workers
    **experimental**](https://bun.com/docs/runtime/workers). Use `2+` for actual parallelism; `1` enables worker mode
    but does not increase throughput. Each worker loads its own Pyodide runtime, so higher worker counts increase RAM
    usage and startup time.

!!! warning "Production Deployment"

    While the sandbox provides isolation, running arbitrary code from an LLM always carries risks. Ensure your deployment environment is properly secured and monitored.
