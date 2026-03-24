# Quickstart

Get your Responses API gateway running in under 5 minutes.

## Prerequisites

- Completed [Installation](installation.md)

______________________________________________________________________

## 1. Start the Gateway

=== "Integrated Colocated Mode"

    Start vLLM and the Responses gateway together on one public API server:

    ```bash
    vllm serve meta-llama/Llama-3.2-3B-Instruct --responses
    ```

=== "Remote-Upstream Gateway Mode"

    If you already have vLLM running on port 8457:

    ```bash
    --8<-- "snippets/serve_external_upstream_cmd.txt"
    ```

Base URL by mode:

- `agentic-stacks serve`: `http://127.0.0.1:5969` by default
- `vllm serve --responses`: same host/port as `vllm serve` (default `http://127.0.0.1:8000`)

______________________________________________________________________

## 2. Send a Request

Now, send a request to the **Responses API** endpoint (`/v1/responses`).

The cURL examples below use the default `agentic-stacks serve` URL (`http://127.0.0.1:5969`).
If you started integrated mode with `vllm serve --responses`, replace that base URL with your
vLLM bind address (default `http://127.0.0.1:8000`).

=== "cURL (streaming with Code Interpreter)"

    ```bash
    curl -X POST http://127.0.0.1:5969/v1/responses \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer dummy" \
      -d '{
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "input": [{"role": "user", "content": "Calculate the factorial of 5"}],
        "stream": true,
        "tools": [{"type": "code_interpreter"}],
        "include": ["code_interpreter_call.outputs"]
      }'
    ```

=== "cURL (non-streaming)"

    ```bash
    curl -X POST http://127.0.0.1:5969/v1/responses \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer dummy" \
      -d '{
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "input": [{"role": "user", "content": "Calculate the factorial of 5"}],
        "tools": [{"type": "code_interpreter"}],
        "include": ["code_interpreter_call.outputs"]
      }'
    ```

=== "Python (OpenAI SDK)"

    ```python
    --8<-- "snippets/openai_client_local_gateway.py"

    # For integrated mode, export:
    #   AGENTIC_STACK_BASE_URL=http://127.0.0.1:8000/v1
    #
    # For `agentic-stacks serve`, the default snippet base URL already matches:
    #   http://127.0.0.1:5969/v1

    with client.responses.stream(
        model="meta-llama/Llama-3.2-3B-Instruct",
        input=[{"role": "user", "content": "Calculate the factorial of 5"}],
        tools=[{"type": "code_interpreter"}],
        include=["code_interpreter_call.outputs"],
    ) as stream:
        for event in stream:
            print(event)
    ```

______________________________________________________________________

## 3. Observe the Response

If you used `stream=true`, you will see **Server-Sent Events (SSE)**. Unlike standard Chat Completions, the Responses API provides rich lifecycle events:

```text
event: response.created
data: {"response":{...}}

event: response.output_item.added
data: {"output_item":{"type":"message", ...}}

event: response.content_part.added
data: {"part":{"type":"text", "text":""}, ...}

event: response.output_text.delta
data: {"delta":"I am a large language model...", ...}

...

event: response.completed
data: {"response":{...}}
```

## 4. Optional: MCP Smoke Test (Built-in MCP)

If you enabled Built-in MCP on your active entrypoint, you can run a minimal forced tool call:

- `agentic-stacks serve ... --mcp-config /path/to/mcp.json`
- `vllm serve ... --responses --responses-mcp-config /path/to/mcp.json`

Need the Built-in MCP `mcp.json` format first? See:

- [MCP Examples -> Built-in MCP Runtime Config](../examples/hosted-mcp-examples.md#built-in-mcp-runtime-config-mcpjson)

```bash
curl -X POST http://127.0.0.1:5969/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "stream": true,
    "input": [{"role":"user","content":"Use the MCP docs tool to search for migration notes."}],
    "tools": [{"type":"mcp","server_label":"github_docs"}],
    "tool_choice": {"type":"mcp","server_label":"github_docs","name":"search_docs"}
  }'
```

If you are running integrated mode, replace `http://127.0.0.1:5969` with your `vllm serve`
base URL (default `http://127.0.0.1:8000`).

In the stream, you should see MCP lifecycle events such as:

- `response.mcp_call.in_progress`
- `response.mcp_call_arguments.done`
- `response.mcp_call.completed` (or `response.mcp_call.failed`)

## Next Steps

Now that you have the basic loop working, try the advanced features:

- **[Code Interpreter](../features/built-in-tools.md)**: Ask the model to write and execute code.
- **[Web Search](../features/web-search.md)**: Let the model search the web with a shipped gateway profile.
- **[Stateful Conversations](../features/statefulness.md)**: Use `previous_response_id` to continue a chat.
- **[MCP Integration](../features/hosted-mcp.md)**: Use Built-in MCP or Remote MCP declarations.
- **[Architecture](architecture.md)**: Learn how the gateway processes your request.
