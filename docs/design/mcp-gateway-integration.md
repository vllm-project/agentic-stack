# MCP Gateway Integration

Target: `crates/agentic-server-core/`

References:

- [OpenAI Responses MCP guide](https://developers.openai.com/api/docs/guides/tools-connectors-mcp)
- [MCP tools specification](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
- [Codex MCP client](https://github.com/openai/codex/tree/main/codex-rs/codex-mcp)
- [Codex MCP tool handler](https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/handlers/mcp.rs)

## Goal

Agentic API implements the OpenAI Responses `type: "mcp"` contract for tools exposed by a remote MCP server. The
gateway connects to each declared server, discovers tools with `tools/list`, presents those tools to the upstream
model as function tools, executes model-selected tools with `tools/call`, and restores the public Responses MCP
identity in output items and streaming events.

The request declaration has server identity and connection information, not a tool name:

```json
{
  "type": "mcp",
  "server_label": "counter",
  "server_url": "http://localhost:8000/mcp",
  "allowed_tools": ["increment", "get_value"],
  "require_approval": "never"
}
```

The server owns the tool names returned by `tools/list`. A client cannot put `name` on the MCP declaration to select
an operation.

## Supported MCP surface

The gateway supports the model-controlled MCP tool lifecycle:

```text
Responses type:mcp declaration
  -> connect and initialize MCP client
  -> tools/list
  -> filter allowed_tools
  -> normalize discovered tools for the upstream model
  -> model emits an internal function call
  -> tools/call
  -> restore public mcp_call identity and events
```

The OpenAI public contract uses:

- `mcp_list_tools` for discovery output
- `response.mcp_list_tools.in_progress`
- `response.mcp_list_tools.completed` or `response.mcp_list_tools.failed`
- `mcp_call` for a selected tool call
- `response.mcp_call.in_progress`
- `response.mcp_call_arguments.delta`
- `response.mcp_call_arguments.done`
- `response.mcp_call.completed` or `response.mcp_call.failed`

The internal function-tool representation is an implementation detail and must not leak as a public function call.
The gateway exposes discovery through the `mcp_list_tools` lifecycle before any selected tool is exposed through the
`mcp_call` lifecycle. Blocking responses include the completed discovery item in `output`. Streaming responses emit
the corresponding output-item and `response.mcp_list_tools.*` events. A discovery failure produces a failed
`mcp_list_tools` item with its error and does not register tools from that server.

Within a stored response or conversation chain, that public list-tools lifecycle is emitted once per server label.
Discovery may still be needed to rebuild executable handlers for a later request; public emission is a separate
decision based on continuation history.

## Components

### `McpClient`

`McpClient` is a thin asynchronous wrapper around an `rmcp` client service. Its gateway execution surface is limited
to MCP tools:

```rust
impl McpClient {
    pub async fn connect(
        server_url: &str,
        headers: Option<HashMap<String, String>>,
    ) -> Result<Self, McpError>;

    pub async fn connect_stdio(
        command: &str,
        args: &[String],
        env: Option<&HashMap<String, String>>,
        cwd: Option<&str>,
    ) -> Result<Self, McpError>;

    pub async fn list_tools(&self) -> Result<Vec<rmcp::model::Tool>, McpError>;

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> Result<rmcp::model::CallToolResult, McpError>;
}
```

### `McpClientPool`

`McpClientPool` owns clients keyed by `server_label`. Request-scoped HTTP clients are constructed from
`McpToolParam`. Gateway configuration can construct HTTP or stdio clients through an `McpServerEntry` under
`~/.agentic-api/config.toml`; a request selects one by declaring the configured `server_label` without a `server_url`.
Connection details from gateway configuration take precedence and cannot be overridden by a request. Configured
`allowed_tools` form a policy ceiling that a request may narrow but cannot expand, and configured
`require_approval = "never"` lets a request omit its approval setting.

Request-provided URLs allow loopback hosts by default. Additional trusted hostnames may be configured through
`AGENTIC_MCP_ALLOWED_HOSTS` from the process environment or the `config.toml` `[mcp].allowed_hosts` array. URL
validation, pinned DNS addresses, disabled automatic proxy discovery, and disabled redirects prevent later routing
changes from bypassing the configured trust boundary.

### `GatewayExecutors`

`GatewayExecutors` caches discovered MCP handlers by `server_label`. A server label may appear only once in a request.
For an uncached declaration it:

1. Builds an MCP connection from the declaration.
2. Calls `tools/list`.
3. Applies `allowed_tools`.
4. Creates one `McpDiscoveredHandler` per remaining tool.
5. Caches the handlers under the declaration's `server_label`.

An empty final allowed set is a configuration error.

### `McpHandler`

`McpHandler` has one responsibility: normalize and execute discovered MCP tools.

An executable handler contains the `McpClient` bound to the server that advertised the tool. A spec-only handler has
no client and exists only for `ResponsesTool::to_function_tools()` normalization. Each executable handler invokes a
discovered tool through `tools/call`, so it does not need an operation enum.

```rust
pub struct McpHandler {
    client: Option<Arc<McpClient>>,
}
```

During execution, the registry entry supplies `McpDiscoveredToolParam`, which contains:

- public `server_label`
- public MCP `tool_name`
- internal model-visible name
- the tool schema returned by `tools/list`

`McpHandler::execute()` reads that identity, validates the model arguments as a JSON object, calls `tools/call`, and
serializes the MCP result for the next inference round.

### `ToolRegistry`

`ToolRegistry::build_with_handlers()` owns the complete routing table, including discovered MCP tools. Discovery
happens before `RequestPayload::to_upstream_request()` normalizes the request:

```text
ResponsesTool::Mcp
  -> GatewayExecutors::mcp_server_tools()
  -> mcp_list_tools discovery output is retained
  -> declaration._agentic_discovered_tools is populated
  -> each discovered handler is inserted into ToolRegistry
  -> ResponsesTool::to_function_tools()
  -> McpHandler::spec_from_param(...).normalize(...)
```

The `_agentic_discovered_tools` field is internal state and is not part of the public request contract.

Each upstream function name includes both server and tool identity, for example
`mcp__counter__increment`. Names are sanitized and bounded to the upstream function-name limit. The registry retains
the original identity so public output uses:

```json
{
  "type": "mcp_call",
  "server_label": "counter",
  "name": "increment"
}
```

The registry also owns list-tools lifecycle state. Current and historical records
are grouped as `HashMap<String, Vec<McpListTools>>`, keyed by `server_label`.
Registry construction inserts the current discovery record first, rehydrated
`InputItem::McpListTools` records are appended only to labels present in the current
registry, and entries with more than one record are treated as already listed.
`mcp_list_tool_items()` exposes the remaining one-record entries directly to blocking
output assembly and `emit_mcp_discovery_lifecycle()`. Streaming clears the map after
round zero, so the lifecycle cannot repeat in later inference rounds.

This metadata follows the normal history pipeline rather than a side channel:

```text
stored OutputItem::McpListTools
  -> InOutItem::into_input_items
  -> InputItem::McpListTools in enriched continuation history
  -> ToolRegistry lifecycle cache
  -> ResponsesInput::model_input removes it before vLLM
```

Stored public `mcp_call` items are not reconstructed as model input. The
model-visible `function_call` and matching `function_call_output` pair is
persisted separately and remains the canonical continuation source.

## Turn execution

The gateway's existing tool loop handles MCP tools together with other gateway-executed built-in tools:

```text
build request-scoped registry
  -> discover MCP tools
  -> suppress an already-recorded list-tools lifecycle by server label
  -> normalize request for upstream inference
  -> receive internal function call
  -> GatewayRound resolves the registry binding
  -> McpClient tools/call
  -> append function call output for the next upstream round
  -> expose public mcp_call item/events to the Responses client
```

Tool execution failures become failed tool call output and are returned to the model for the next round; they do not
automatically fail the whole Responses request.
