# MCP Integration (Built-in MCP + Remote MCP)

The gateway supports two MCP declaration modes in `tools`:

- Built-in MCP mode: reference a configured server by `server_label`.
- Remote MCP mode: provide request `server_url` (and request-scoped auth/headers).

This page focuses on Built-in MCP setup and call flow, then summarizes Remote MCP mode differences.

## Choose a Mode

| Mode         | Best When                                                 | Request Shape                                 |
| ------------ | --------------------------------------------------------- | --------------------------------------------- |
| Built-in MCP | You want centrally managed server inventory and policy    | `type: "mcp"` + `server_label`                |
| Remote MCP   | You want to point directly to an MCP endpoint per request | `type: "mcp"` + `server_label` + `server_url` |

## What It Solves

- Keep MCP execution inside the gateway request lifecycle.
- Use Responses-style streaming events for MCP call progress/results.
- Reuse response IDs with `previous_response_id` just like other tool flows.

## Prerequisites

1. Configure MCP runtime servers via `--mcp-config` (or `--responses-mcp-config` in integrated mode).
1. Ensure the target `server_label` is available (`GET /v1/mcp/servers`).
1. Start the gateway with either:
    - `agentic-stacks serve --mcp-config ...`, or
    - `vllm serve --responses --responses-mcp-config ...`

## Built-in MCP Setup

Pass the Built-in MCP config path on the entrypoint CLI:

```bash
agentic-stacks serve \
  --upstream http://127.0.0.1:8000/v1 \
  --mcp-config /etc/agentic-stacks/mcp.json
```

`mcp.json` follows the common MCP client-style shape: a top-level `mcpServers` object keyed by your server labels.
In most cases, you can copy an MCP server entry from another MCP client config and reuse it here with minimal changes.
For canonical examples (URL + stdio styles), see [MCP Examples -> Built-in MCP Runtime Config](../examples/hosted-mcp-examples.md#built-in-mcp-runtime-config-mcpjson).

Built-in URL-style entries accept both `http://` and `https://` URLs. This differs from Remote MCP request URLs, which are policy-checked as `https://` by default.

Verify server availability before requests:

```bash
--8<-- "snippets/mcp_discover_servers_tools_curl.txt"
```

For integrated mode, set `AGENTIC_STACK_HTTP_BASE=http://127.0.0.1:8000` first (or use your
custom `vllm serve` host/port).

Runtime architecture note:

- `agentic-stacks serve` starts one internal Built-in MCP runtime process on loopback.
- `vllm serve --responses --responses-mcp-config ...` also starts one loopback Built-in MCP helper for the combined app.
- All gateway workers share that runtime, so Built-in MCP startup/discovery/session state is not duplicated per worker.

## Built-in MCP Usage

Use one complete request payload including both MCP declaration and tool choice:

--8<-- "snippets/mcp_builtin_request_payload.txt"

### cURL

```bash
curl -X POST http://127.0.0.1:5969/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "stream": true,
    "input": [{"role":"user","content":"Find migration notes in docs."}],
    "tools": [{"type":"mcp","server_label":"github_docs","allowed_tools":["search_docs"],"require_approval":"never"}],
    "tool_choice": {"type":"mcp","server_label":"github_docs","name":"search_docs"}
  }'
```

If you are using integrated mode, replace `http://127.0.0.1:5969` with your `vllm serve`
base URL (default `http://127.0.0.1:8000`).

### OpenAI Python SDK

```python
--8<-- "snippets/openai_client_local_gateway.py"

# For integrated mode, set:
#   export AGENTIC_STACK_BASE_URL=http://127.0.0.1:8000/v1

with client.responses.stream(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input=[{"role": "user", "content": "Find migration notes in docs."}],
    tools=[
        {
            "type": "mcp",
            "server_label": "github_docs",
            "allowed_tools": ["search_docs"],
            "require_approval": "never",
        }
    ],
    tool_choice={"type": "mcp", "server_label": "github_docs", "name": "search_docs"},
) as stream:
    for event in stream:
        print(event.type)
```

## MCP Event Lifecycle

Both MCP modes stream these event types:

- `response.mcp_call.in_progress`
- `response.mcp_call_arguments.delta`
- `response.mcp_call_arguments.done`
- `response.mcp_call.completed` or `response.mcp_call.failed`

See [Events Reference](../reference/events.md) for payload details.

## Remote MCP Mode Notes

- Built-in MCP requests reference configured servers by `server_label` only.
- Remote MCP via request `server_url` does not require server registration in the Built-in MCP config file.
- Remote MCP transport selection is delegated to FastMCP from request `server_url` and headers.
- `require_approval` currently supports `never` only.
- Remote MCP host policy rejects `localhost`, `*.localhost`, and IP-literal hosts, and only `https` is accepted.
- For Remote MCP field compatibility (`server_url`, `connector_id`, `headers`), see [API Reference](../reference/api-reference.md#mcp-compatibility-matrix-current).

For end-to-end examples, see [MCP Examples](../examples/hosted-mcp-examples.md).
