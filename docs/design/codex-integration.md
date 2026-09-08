# Design: Codex CLI Integration

> **References:** [Issue #54](https://github.com/vllm-project/agentic-api/issues/54),
> [PR #67](https://github.com/vllm-project/agentic-api/pull/67)
> **Owner:** @haoshan98 for Codex compatibility. Latest `main` owns the shared tool framework lineage from PR #67.

---

## Summary

`agentic-api` can sit between Codex CLI and a vLLM-backed Responses-compatible model endpoint.

Codex can declare grouped tools with `type: "namespace"` and freeform tools with `type: "custom"`. Namespace members
need translation for upstreams that expose ordinary `type: "function"` calling. Custom tools must remain native because
their calls contain raw text instead of JSON function arguments. The gateway performs the namespace translation while
preserving custom declarations, calls, streaming events, outputs, and continuation history unchanged.

The important split:

- **Codex compatibility:** preserve Codex request/response shapes, namespace identity, and continuation state.
- **Shared framework:** provide generic tool normalization, registry ownership, gateway execution, and tool-loop
  orchestration.

---

## Implemented Scope

The current integration supports the typed executor path:

- `ResponsesTool::Namespace` preserves the public Codex namespace declaration.
- `ResponsesTool::Custom` preserves freeform declarations, including opaque `format` grammars.
- `CodexNamespaceHandler` owns Codex-specific namespace flattening and restoration.
- `RequestPayload::to_upstream_request()` flattens namespace function members to vLLM-compatible function tools.
- `RequestPayload::to_upstream_request()` forwards custom tools in their native wire shape.
- Namespaced `tool_choice` values are rewritten to the same flat names sent upstream.
- `ToolRegistry` builds a request-scoped namespace map once and uses it for final payload and streaming event
  restoration.
- `GatewayExecutors` provides the shared web-search handler and builds request-scoped MCP handlers. The gateway executes
  those tools while namespace and custom calls remain client-executed.
- Stateful continuation stores effective tools, tool choice, instructions, and response/conversation linkage for later
  `previous_response_id` or `conversation_id` turns.
- WebSocket Responses execution uses the same typed executor path and restores namespace tool-call events before sending
  them to clients.

Stateless `store=false` requests containing only ordinary `function` declarations remain byte-transparent raw proxy
requests. Requests with continuation IDs or any non-function tool use the typed executor, where namespace and
gateway-executed built-in tool normalization occurs.

---

## Namespace Flattening

The model-visible namespace member format is:

```text
agentic_ns__{namespace}__{member}
```

Names at or below the upstream 64-character function-name limit retain that exact form. Longer generated names keep a readable prefix and replace the tail with a deterministic 16-hex fingerprint, producing exactly 64 characters. The request-scoped namespace map records the result, so restoration never depends on parsing either form.

For example, Codex can send:

```json
{
  "type": "namespace",
  "name": "mcp__agentic_fixture",
  "tools": [
    { "type": "function", "name": "add_numbers" }
  ]
}
```

The upstream request receives:

```json
{
  "type": "function",
  "name": "agentic_ns__mcp__agentic_fixture__add_numbers"
}
```

When the model calls that flat function, the gateway restores:

```json
{
  "type": "function_call",
  "namespace": "mcp__agentic_fixture",
  "name": "add_numbers"
}
```

---

## Collision Handling

The `agentic_ns__` prefix marks gateway-generated namespace member names. If any other declared tool registers the same function-call name as a namespace member (including a function, MCP tool, or normalized built-in), or if two distinct namespace members generate the same flat name, the typed executor rejects the request as invalid before upstream inference or gateway tool setup. Forwarding either shape would make a later model call ambiguous and impossible to restore reliably to `{ namespace, name }`. Custom tools are excluded from this check because they use `custom_tool_call`, not `function_call`.

---

## Namespace Versus Custom Tools

These types differ at both the declaration and call boundaries:

| Property | `namespace` | `custom` |
|----------|-------------|----------|
| Declaration | A named group containing function members with JSON Schema parameters. | One named freeform tool with an optional format, commonly a Lark or regex grammar. |
| Upstream request | Flatten each member to a model-visible `type: "function"` declaration. | Forward the original `type: "custom"` declaration and `format` unchanged. |
| Model output | `function_call` with JSON text in `arguments`. | `custom_tool_call` with opaque text in `input`. |
| Streaming | `response.function_call_arguments.delta` and `.done`. | `response.custom_tool_call_input.delta` and `.done`. |
| Client result | `function_call_output`. | `custom_tool_call_output`. |
| Gateway execution | Never; the client owns the namespace member execution. | Never; the client owns the freeform tool execution. |

A client that declares a custom tool consumes the returned raw input, executes it locally, and submits a
`custom_tool_call_output` using the same `call_id`. On the stored WebSocket path, the gateway stores the assistant call
before exposing `response.completed`, so an immediate close cannot lose the continuation state.

The upstream model still decides whether to select a custom tool when `tool_choice` is `auto`. The gateway does not
rewrite that choice. The configured Qwen Responses endpoint also requires a `format` on custom declarations. This
matches the Responses API distinction between JSON-schema function tools and
[freeform custom tools](https://developers.openai.com/api/docs/guides/function-calling#custom-tools).

Tool availability is a client-version and configuration concern, separate from gateway protocol support. In the tested
Codex 0.144.3 integration, a captured request declared `apply_patch` as a native custom tool with a format, and the
gateway forwarded it unchanged. The `codex features list` line `apply_patch_freeform removed false` is not sufficient by
itself to determine the request shape; use a captured request or gateway debug log as the authoritative check.

---

## Compatibility Rules

The gateway should not detect requests by user agent, route, or "is this Codex?" heuristics. Compatibility is driven by
Responses tool shapes and execution semantics, so it can be always on.

| Shape | Behavior |
|-------|----------|
| `function` | Client-executed function tool. Preserve the declaration and return matching calls to the client. |
| `namespace` | Client-executed Codex grouping for function tools. Flatten members only for upstream requests, then restore returned calls. |
| `custom` | Client-executed custom tool. Preserve its opaque format and forward it natively. |
| `web_search_preview` | Gateway-executed built-in tool normalized to the web-search function tool. Without a usable provider, execution produces a failed tool result instead of returning the call for client execution. |
| `mcp` | Gateway-executed built-in tool. Normalize discovered MCP tools to model-visible function tools, execute calls with request-scoped MCP bindings, and expose public `mcp_call` items. Streaming emits `response.output_item.added`, `response.mcp_call.in_progress`, `response.mcp_call_arguments.delta`/`.done`, `response.mcp_call.completed` or `.failed`, and `response.output_item.done`. |
| `file_search`, `code_interpreter` | Accepted by the typed request parser but skipped during upstream normalization because no gateway handler is registered yet. |
| Unknown tool | Recognized and skipped on the typed path; opaque fields are not preserved or executed. Eligible raw-proxy requests remain byte-transparent. |

For response items:

| Response item | Behavior |
|---------------|----------|
| `function_call` | Preserve optional `namespace`; restore flat namespace calls before returning to Codex. |
| `custom_tool_call` | Preserve raw `input`; return it to Codex for local execution. |
| `web_search_call` | Result from the gateway-executed web-search tool. |
| `mcp_call` | Result from a gateway-executed MCP tool with `server_label`, discovered tool `name`, JSON-string `arguments`, and `status`; successful calls contain `output`, while failures contain a structured `mcp_tool_execution_error`. |
| Unknown output item | Recognized as an unknown unit variant on the typed path; opaque fields are not preserved or executed. |

---

## Continuation

Codex tool calls returned for client execution must survive response-store continuation.

Expected rehydration shape:

```text
prior context + assistant tool call + Codex tool output + new input
```

On a turn that returns client-executed tool calls, storage keeps the assistant call item. On the next turn, Codex submits
the matching tool output item, and `previous_response_id` rebuilds the full sequence while preserving effective tool
metadata from the previous response unless the client explicitly overrides it.

Public output items for gateway-executed web search and MCP tools are not reconstructed as model input during continuation. Their
internal `function_call` and matching `function_call_output` records are persisted as the canonical model-visible pair.
MCP list-tools output is different: it rehydrates as an internal `InputItem::McpListTools` record so the request-scoped
registry can avoid repeating that server label's public discovery lifecycle. `ResponsesInput::model_input()` removes
the record before the request is sent to vLLM.

---

## Manual Custom Tool Test

Run the commands in this section from the repository root.

First run the deterministic gateway lifecycle test. Its mock upstream emits the same custom-call streaming events as a
Responses provider, then the test sends the matching Codex output on a second WebSocket request and verifies the fully
rehydrated upstream input:

```bash
cargo test -p agentic-server --test responses_websocket_test \
  test_websocket_custom_tool_round_trip_and_continuation -- --nocapture
```

The following live smoke test additionally measures whether the configured model selects the custom tool correctly.
Start the gateway in one terminal:

```bash
RUST_LOG=agentic_core=debug,agentic_server=debug \
GATEWAY_PORT=3018 \
DATABASE_URL="sqlite:///tmp/agentic_api_codex_qwen36_custom.db" \
V_API_BASE="http://192.168.80.6:8396" \
V_API_KEY="" \
V_MODEL="Qwen/Qwen3.6-35B-A3B" \
./scripts/codex-start-gateway.sh
```

Before involving Codex, verify the gateway and upstream custom-tool protocol directly from a second terminal:

```bash
curl --max-time 60 -sS http://127.0.0.1:3018/v1/responses \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"Qwen/Qwen3.6-35B-A3B","input":"Call echo_raw with exactly CUSTOM_RAW_OK. Do not answer in prose.","tools":[{"type":"custom","name":"echo_raw","description":"Emit the requested raw token exactly.","format":{"type":"grammar","syntax":"lark","definition":"start: \"CUSTOM_RAW_OK\""}}],"tool_choice":"required","store":true,"stream":false}'
```

The response should contain a `custom_tool_call` named `echo_raw` whose `input` is `CUSTOM_RAW_OK`. This isolates native
gateway/upstream support from any client's tool-registration settings.

To test Codex's local custom-tool handler without exposing credentials from the normal Codex home, seed an isolated
temporary home with the gateway's credential-free model catalog, prepare the fixture, and run Codex there. The helper
writes both Codex's ordinary `models_cache.json` and an authoritative `model_catalog.json`:

```bash
mkdir -p /tmp/agentic-codex-smoke-home tmp
printf 'CUSTOM_OK\n' > tmp/custom-tool-smoke.txt

GATEWAY_URL=http://127.0.0.1:3018 \
MODEL="Qwen/Qwen3.6-35B-A3B" \
CODEX_HOME=/tmp/agentic-codex-smoke-home \
bash ./scripts/codex-seed-model-cache.sh

CODEX_HOME=/tmp/agentic-codex-smoke-home codex exec \
  --disable image_generation \
  --disable apps \
  --disable plugins \
  --sandbox workspace-write \
  -C "$PWD" \
  -m "Qwen/Qwen3.6-35B-A3B" \
  -c model_reasoning_effort=low \
  -c model_provider=agentic-local \
  -c 'model_providers.agentic-local.name="agentic-api local"' \
  -c 'model_providers.agentic-local.base_url="http://127.0.0.1:3018/v1"' \
  -c 'model_providers.agentic-local.wire_api="responses"' \
  -c 'model_providers.agentic-local.supports_websockets=true' \
  -c 'model_providers.agentic-local.requires_openai_auth=false' \
  -c 'model_catalog_json="/tmp/agentic-codex-smoke-home/model_catalog.json"' \
  "Use the native freeform apply_patch tool now. Do not call exec_command, shell, or any namespace/function tool. Send this exact text as apply_patch's raw input, preserving newlines:
*** Begin Patch
*** Update File: tmp/custom-tool-smoke.txt
@@
-CUSTOM_OK
+CUSTOM_TOOL_OK
*** End Patch
After it succeeds, reply only: CUSTOM_TOOL_OK"
```

Success means the file contains `CUSTOM_TOOL_OK`, the gateway log reports `forwarding native custom tool declaration
upstream` for `apply_patch`, and Codex replies `CUSTOM_TOOL_OK`.

The static catalog setting is required for repeatable use with the tested Codex 0.144.3 client. Its ordinary model cache
expires after 300 seconds. An unauthenticated custom provider does not trigger a remote catalog refresh, so after that
TTL Codex falls back to generic model metadata where `apply_patch_tool_type` is unset. That makes `apply_patch` disappear
from both `codex exec` and interactive sessions even though the gateway is healthy. `model_catalog_json` loads the same
gateway metadata as an authoritative startup catalog and has no cache TTL.

For an interactive session, use the same static catalog setting:

```bash
CODEX_HOME=/tmp/agentic-codex-smoke-home codex \
  --disable image_generation \
  --disable apps \
  --disable plugins \
  --sandbox workspace-write \
  -C "$PWD" \
  -m "Qwen/Qwen3.6-35B-A3B" \
  -c model_reasoning_effort=low \
  -c model_provider=agentic-local \
  -c 'model_providers.agentic-local.name="agentic-api local"' \
  -c 'model_providers.agentic-local.base_url="http://127.0.0.1:3018/v1"' \
  -c 'model_providers.agentic-local.wire_api="responses"' \
  -c 'model_providers.agentic-local.supports_websockets=true' \
  -c 'model_providers.agentic-local.requires_openai_auth=false' \
  -c 'model_catalog_json="/tmp/agentic-codex-smoke-home/model_catalog.json"'
```

Interactive Codex also performs a WebSocket startup prewarm by sending `response.create` with `generate: false`. The
gateway acknowledges that request locally with `response.created` and `response.completed`, persists its prompt and
tool context, and performs no upstream inference. Codex then reuses the returned response ID for the first real turn,
which may contain only incremental input. This behavior is covered by:

```bash
cargo test -p agentic-server --test responses_websocket_test \
  test_websocket_generate_false_prewarm_persists_context_without_inference -- --nocapture
```

For any Codex probe against this unauthenticated local provider (`V_API_KEY=""`), add the following provider setting:

```bash
-c 'model_providers.agentic-local.requires_openai_auth=false'
```

The official Codex [alternative-provider authentication documentation](https://learn.chatgpt.com/docs/auth#alternative-model-providers)
says this setting selects no authentication when no `env_key` is also configured. However, the reproduced Codex 0.144.3
model-catalog refresh still attached the saved OpenAI/ChatGPT bearer credential to `/v1/models`. The generic gateway
preserves client authorization headers by design, so the upstream rejected that incidental credential. The isolated
`CODEX_HOME` above avoids the credential path for this smoke test. Do not copy or forward the credential shown in the
diagnostic, and rotate it if it was reusable.

---

## Image Capability Resolution

Codex decides whether a request may carry image content by reading its **local** model catalog. When the entry for the
selected model does not list `image` in `input_modalities`, Codex strips image content client-side and sends a
placeholder instead. A vision-capable upstream is therefore not enough on its own: the catalog Codex reads has to say
the model accepts images.

Two catalogs exist and they must agree:

| Catalog | Produced by | Consumed by |
|---|---|---|
| `GET /v1/models?client_version=<ver>` | `handler/http/models.rs` | Codex refreshing its model list, and both `agentic` launchers |
| `$CODEX_HOME/model_catalog.json` | `agentic_harness::prepare_codex_home` (and `scripts/agentic-codex.sh`) | Codex reading an isolated session home |

### Resolution order

The gateway resolves `input_modalities` for every served model in this order:

1. An explicit `[models."<served-model-id>"] input_modalities` override in `~/.agentic-api/config.toml`.
2. Recognized upstream metadata: `capabilities: ["image"]` on the upstream `/v1/models` entry.
3. A conservative text-only fallback.

An explicit `["text"]` override wins over upstream image metadata, which is how a vision model gets pinned to text.
Capabilities are never inferred from a model name — an unrecognized capability string such as `vision` or `multimodal`
resolves to text-only. `supports_image_detail_original` stays `false`: the gateway does not relay image detail hints.

### How the launchers stay consistent

`agentic run codex` and `agentic harness codex` both fetch `GET {gateway}/v1/models?client_version=<ver>` before
writing the isolated home, and take the selected model **and** its modalities from that one response. The client
version is read from the Codex binary itself (`codex --version`, honoring `AGENTIC_CODEX_BIN`); set
`AGENTIC_CODEX_CLIENT_VERSION` to skip the probe where it cannot run. Only the modalities are copied — the isolated
catalog keeps its launcher-specific settings (`shell_type: "local"`, an omitted `apply_patch_tool_type`, and a
token-based truncation policy), which intentionally differ from the HTTP catalog.

`scripts/agentic-codex.sh` copies the gateway catalog verbatim, so it inherits the resolved modalities with no change.

`scripts/codex-smoke.sh` covers this end to end in CI: the replay server advertises `capabilities: ["image"]`, the real
Codex CLI attaches a generated 1x1 PNG with `--image`, and the capture assertion requires that exact payload by
SHA-256. With the capability removed, Codex instead sends `image content omitted because you do not support image
input` as text and the assertion fails — which is what makes client-side stripping distinguishable from gateway loss.
[Verifying image support against a live vision model](../guides/vision-model-verification.md) covers the live check.

### Failure behavior

A launcher never writes a catalog it could not verify. If the gateway cannot be reached, rejects the request, returns
an undecodable catalog, or does not list the selected model, the launch fails with an actionable error instead of
writing text-only metadata:

- Transport errors, `5xx`, `408`, `425`, `429`, and an empty catalog are retried until the readiness budget expires
  (`--llm-ready-timeout-s`/`--llm-ready-interval-s`; 30s/250ms when attaching to a running gateway).
- `401`/`403` fail immediately with a hint to pass `--api-key`. **A gateway behind OIDC now requires a credential for
  `agentic harness codex`**, because `/v1/models` is a protected route.
- A served, non-empty catalog that does not list the selected model is retried for 10 seconds and then fails, naming
  the models the gateway does serve. A catalog listing other models proves the upstream is warm, so a missing model is
  treated as a configuration error rather than a cold start.
- Catalog responses larger than 1 MiB are rejected.

### Regenerating existing session homes

`agentic run` and `agentic harness` create a fresh session home per invocation, so they pick up resolved modalities
automatically. A persistent home does not: delete and regenerate any `model_catalog.json` written before this change
(for example a directory pinned with `AGENTIC_CODEX_HOME`, or a hand-written `-c model_catalog_json=...` file), or
Codex will keep reading the stale text-only entry.

---

## Out Of Scope

- Raw proxy namespace flatten/restore.
- Gateway-side model aliasing.
- A gateway-side Codex runtime.
- Executing Codex namespace tools in the gateway. Codex still owns client-side tool execution.
