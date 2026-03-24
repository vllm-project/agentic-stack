# MCP Examples (Built-in MCP + Remote MCP)

Use MCP tools through the Responses API in either Built-in MCP mode or Remote MCP mode.

## Built-in MCP Runtime Config (`mcp.json`)

Pass `--mcp-config` (or `--responses-mcp-config` in integrated mode) to point to an MCP runtime config file:

```bash
agentic-stacks serve \
  --upstream http://127.0.0.1:8000/v1 \
  --mcp-config /etc/agentic-stacks/mcp.json
```

Expected shape: top-level `mcpServers`, with each key as your `server_label` and each value as one MCP server entry.
This is intentionally close to common MCP client config formats, so you can usually copy existing entries directly.

Canonical example (`url` + `stdio` styles):

--8<-- "snippets/mcp_runtime_config_example.txt"

## Built-in MCP: Discover Available Servers and Tools

Before sending tool requests, inspect runtime availability:

Start with a supported entrypoint that enables Built-in MCP:

- `agentic-stacks serve --mcp-config ...`
- `vllm serve --responses --responses-mcp-config ...`

```bash
--8<-- "snippets/mcp_discover_servers_tools_curl.txt"
```

The snippet defaults to `http://127.0.0.1:5969`. In integrated mode, set
`AGENTIC_STACK_HTTP_BASE=http://127.0.0.1:8000` first, or replace the base URL with your
configured `vllm serve` address.

## Built-in MCP: Force an MCP Tool Call

Canonical request payload:

--8<-- "snippets/mcp_builtin_request_payload.txt"

Python SDK equivalent:

```python
response = client.responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    stream=True,
    input=[{"role": "user", "content": "Find migration notes in docs."}],
    tools=[
        {
            "type": "mcp",
            "server_label": "github_docs",
            "allowed_tools": ["search_docs"],
            "require_approval": "never",
        }
    ],
    tool_choice={
        "type": "mcp",
        "server_label": "github_docs",
        "name": "search_docs",
    },
)
```

Expected stream events include:

- `response.mcp_call.in_progress`
- `response.mcp_call_arguments.delta`
- `response.mcp_call_arguments.done`
- `response.mcp_call.completed` or `response.mcp_call.failed`

## Continue with `previous_response_id`

If `store=true` (default), the response ID from a terminal response can be reused:

```python
follow_up = client.responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    previous_response_id=response.id,
    input=[{"role": "user", "content": "Summarize that in one sentence."}],
)
```

## Remote MCP Mode: Quick Example

Use Remote MCP mode when you want to declare an MCP endpoint directly in the request instead of using Built-in MCP registry config.

```python
remote = client.responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input=[{"role": "user", "content": "Search remote docs for migration notes."}],
    tools=[
        {
            "type": "mcp",
            "server_label": "docs_remote",
            "server_url": "https://mcp.example.com/sse",
            "authorization": "YOUR_TOKEN",
            "allowed_tools": ["search_docs"],
            "require_approval": "never",
        }
    ],
    tool_choice={"type": "mcp", "server_label": "docs_remote", "name": "search_docs"},
)
```
