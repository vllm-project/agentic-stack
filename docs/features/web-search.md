# Web Search

Use the gateway-owned `web_search` built-in when you want the model to search
the web, open a page, or search inside a page it already opened during the same
request.

`web_search` is one public built-in tool:

```json
{"type": "web_search"}
```

The gateway owns how that public tool is realized internally. Operators enable
one shipped profile at startup, and clients continue to use the same public
tool shape regardless of which profile is active.

## Shipped Profiles

Current shipped profiles:

- `exa_mcp`
- `duckduckgo_plus_fetch`

### `exa_mcp`

- Uses Exa-backed MCP tools for search and page opening.
- Requires the Built-in MCP runtime, but the shipped helper entry is provisioned
    automatically when the profile is enabled.
- If `EXA_API_KEY` is set in the gateway environment, the shipped Exa MCP entry
    appends it automatically to the Exa MCP URL.

### `duckduckgo_plus_fetch`

- Uses DuckDuckGo for search and the Fetch MCP server for page opening.
- Also requires the Built-in MCP runtime, with the shipped Fetch helper entry
    provisioned automatically when the profile is enabled.

## Principles

- `web_search` stays one public built-in tool even though the gateway may
    realize it internally with multiple actions.
- Profile selection is an operator decision made at startup, not a per-request
    client parameter.
- The gateway keeps backend/provider details out of the public Responses API
    request shape.
- `find_in_page` works over request-local cached page text from a prior
    `open_page` result in the same request.

## Enable the Tool

### `agentic-stacks serve`

```bash
agentic-stacks serve \
  --upstream http://127.0.0.1:8000/v1 \
  --web-search-profile exa_mcp
```

### `vllm serve --responses`

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --responses \
  --responses-web-search-profile exa_mcp
```

Notes:

- If the profile flag is omitted, `web_search` is disabled.
- For supported entrypoints, web-search enablement is CLI-owned.
- Shipped profiles that need Built-in MCP helper servers provision their
    default helper entries automatically. You do not need `--mcp-config` just to
    enable a shipped `web_search` profile.

## Use the Tool

Minimal Python SDK example:

```python
response = client.responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input=[{"role": "user", "content": "Find the latest migration notes for vLLM."}],
    tools=[{"type": "web_search"}],
)
```

When the model uses the tool, the response contains a `web_search_call` output
item.

If you want source expansion on search results, add:

```python
include=["web_search_call.action.sources"]
```

Streaming responses emit the `web_search_call` lifecycle family, including:

- `response.web_search_call.in_progress`
- `response.web_search_call.searching`
- `response.web_search_call.completed`

## Current Behavior and Limitations

- Profile selection happens at startup. Requests do not choose between shipped
    profiles.
- Current shipped profiles are limited to `exa_mcp` and
    `duckduckgo_plus_fetch`.
- Completed `web_search_call` output items follow normal Responses storage
    behavior when `store=true`.
- `find_in_page` depends on page text cached from an earlier `open_page`
    action in the same request, and that page cache is not persisted across
    requests or `previous_response_id` continuations.
- `include=["web_search_call.action.sources"]` controls source expansion for
    search results.
- Built-in MCP auto-provision covers the shipped helper defaults. Use
    `--mcp-config` or `--responses-mcp-config` when you also need extra MCP
    inventory or explicit overrides.
