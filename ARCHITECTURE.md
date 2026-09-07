# Architecture

This document explains how `agentic-api` is put together: the crate boundaries, the
request lifecycle, and where to make a change for common contribution tasks. It
assumes you've read the crate overview in [AGENTS.md](AGENTS.md) and complements it —
AGENTS.md covers tooling and conventions, this document covers the mental model.

Background on *why* the system is shaped this way lives in the ADRs
([ADR-01](docs/adr/ADR-01_core.md), [ADR-02](docs/adr/ADR-02_response_store.md),
[ADR-03](docs/adr/ADR-03_gateway_integration.md)) and the design docs under
`docs/design/`. Those documents record proposals and drift over time; this document
describes the code as it exists today and will be kept current as the code moves.
Where a design doc's "as-built" notes and the code agree, this document just states
the outcome.

## Workspace layout

```
agentic-api/
  crates/
    agentic-server-core/   # "agentic_core" — pure Rust orchestration library
    agentic-server/        # axum HTTP/WS gateway + the `agentic` CLI launcher
    agentic-praxis/        # placeholder: future Praxis gateway adapter
```

- **`agentic-server-core`** (library crate name `agentic_core`) is where all domain
  logic lives: request/response types, SSE parsing, the agentic loop, tool framework,
  and the storage layer. It has no HTTP framework dependency.
- **`agentic-server`** is a thin transport layer: an axum binary that parses HTTP/WS,
  calls into `agentic_core`, and streams the result back. It also happens to host a
  second, unrelated binary — a CLI launcher (`agentic`) that spawns the gateway and a
  coding harness (Codex/Claude Code) as subprocesses for local use.
- **`agentic-praxis`** is currently a placeholder. Per ADR-03, the intent is for it to
  wrap each `agentic-server-core` public function as an `HttpFilter` so Praxis can
  compose the agentic loop declaratively instead of going through `agentic-server`'s
  axum router. Nothing is implemented there yet.

The dependency direction is one-way: `agentic-server` depends on `agentic-server-core`,
never the reverse. `agentic-server-core` has no knowledge of axum, HTTP, or WebSockets.

## Request flow at a glance

```
Client ──HTTP/WS──▶ agentic-server (handler/*)
                        │
                        ▼
              agentic_core::executor::ExecuteRequest::run()
                        │
          ┌─────────────┼──────────────────────────┐
          ▼             ▼                           ▼
     rehydrate()   upstream call(s) + tool loop   persist()
   (storage read)  (vLLM, gateway tool execution)  (storage write)
                        │
                        ▼
                 SSE / JSON back to client
```

Persistence uses `sqlx` against a driver-agnostic `Any` pool backed by SQLite or
Postgres (`storage::pool`). The upstream inference call targets vLLM's own stateless
Responses API — this project owns the state, vLLM owns tokenization and generation
(see ADR-01 §1.1).

One Responses turn may contain several inference rounds, but it is exposed and
persisted as one response:

```
rehydrate history
      │
      ▼
build ToolRegistry + discover MCP tools
      │
      ▼
┌─▶ optional compaction ─▶ one upstream inference round
│                              │
│                              ▼
│                    resolve calls by ownership
│                       │               │
│                       │               └─ client-owned calls stay unresolved
│                       ▼
│             GatewayScheduler plans and executes calls
│             with bounded fan-out and ordered results/events
│                       │
│                       ▼
│                  classify_round
│              │            │             │
│              │            │             └─ client/incomplete/done: finalize
│              │            └─ append calls/results to continuation input
└──────────────┘  (`Continue`, at most 10 rounds)
      │
      ▼
finalize one public response + persist one turn
```

A mixed round may contain both ownership classes. Gateway-owned calls still execute
and are recorded, while the response returns the unresolved client-owned calls for the
client to resolve. Streaming uses the same round loop and projects it through one
continuous SSE lifecycle.

## `agentic-server` — the transport layer

### Two binaries sharing one library

The crate produces a library plus two independent binaries. `src/lib.rs` exports
`agentic_cli`, `agentic_harness`, `agentic_output`, `agentic_process`, `app`, `auth`,
and `handler`. On top of that:

| Binary | Entry point | Uses |
|---|---|---|
| `agentic-server` (the gateway) | `src/main.rs` | `app`, `auth`, `handler`, plus binary-private `server.rs` and `config_file.rs` |
| `agentic` (the CLI launcher) | `src/bin/agentic.rs` | `agentic_cli`, `agentic_harness`, `agentic_output`, `agentic_process` only |

These are two unrelated concerns bundled in one crate. If you're working on request
handling, ignore `agentic_cli*`/`agentic_harness.rs`/`agentic_output.rs`/
`agentic_process.rs` entirely — they're the launcher that spawns the gateway binary and
a coding harness (Codex or Claude Code) as subprocesses for local, single-command use
(`agentic serve <model>`), and never touch the request path.

### `app.rs`, `server.rs`, `main.rs`

- **`app.rs`** (library) builds the router: `AppState` (the per-request-shared state:
  `exec_ctx: Arc<ExecutionContext>`, proxy state, readiness/websocket trackers, config)
  and `build_router_with_auth(state, server_config, authenticator)`, which wires every
  route and optionally layers OIDC auth (`auth::require_oidc`) onto the protected ones.
- **`server.rs`** (binary-private) owns process lifecycle: `build_state` (constructs
  `ExecutionContext::from_config`, i.e. where the DB pool actually gets created),
  `serve_gateway`/`serve_gateway_until_signal` (bind, serve, graceful shutdown with a
  bounded drain), and `run`/`run_with_llm` (standalone mode, optionally spawning a vLLM
  subprocess).
- **`main.rs`** (binary-private) is the `clap` CLI front end: parses config from
  flags/env/`config.toml`, then calls `server::run` or `server::run_with_llm`.

### HTTP handlers (`handler/http/`)

| Route | Handler | File |
|---|---|---|
| `POST /v1/responses` | `responses` | `handler/http/responses.rs` |
| `POST /v1/responses/compact` | `compact_response` | `handler/http/responses.rs` |
| `POST /v1/conversations` | `conversations` | `handler/http/conversations.rs` |
| `POST /v1/messages` | `messages` | `handler/http/messages.rs` |
| `POST /v1/messages/count_tokens` | `count_tokens` | `handler/http/messages.rs` |
| `GET /v1/models` | `models` | `handler/http/models.rs` |
| `GET /health` | `health` | `handler/http/models.rs` |
| `GET /ready` | `ready` | `handler/http/models.rs` |

Handlers make a request-scoped decision between two paths:
- **Stateful/executor path** — used when the request needs state (`store: true`,
  `previous_response_id`, `conversation_id`, compaction, or a gateway-owned tool).
  Builds an `ExecuteRequest` (Responses) or calls `run_messages_loop`/
  `run_messages_stream` (Messages) against `state.exec_ctx`.
- **Pass-through path** — everything else is forwarded to vLLM unchanged via
  `agentic_core::proxy`, with no state, no persistence.

The HTTP Responses handler first parses `RequestPayload<RawValue>` so routing fields
remain typed while provider-specific `text` formats stay opaque on the pass-through
path. Executor routes convert that raw field to `ResponseTextConfig` before
building `ExecuteRequest`, preserving strict validation for in-process execution.

### WebSocket transport (`handler/websocket/`)

`GET /v1/responses` upgrades to a WebSocket. Structurally this is not a one-shot
handler like the HTTP routes — `responses_ws_loop` is a long-lived session loop that
reads `response.create` messages off the socket, queues any that arrive while a
response is streaming, and drives the *same* `ExecuteRequest::run()` executor call the
HTTP handler uses. WebSocket sessions always force `stream: true, store: true`. Because
axum's built-in graceful shutdown doesn't wait for upgraded connections, `AppState`
carries a separate `WebSocketTracker` so shutdown can drain in-flight sessions.
Errors are modeled by a dedicated `WsError` enum (`handler/websocket/error.rs`) rather
than reusing the HTTP JSON-error path, since some failure modes (a dead socket) must
not attempt to write a response.

### `handler/common.rs`

Transport helpers shared by the HTTP and WS handlers: body reading with a shared size
cap, generic JSON parsing, bearer-token extraction, SSE response wrapping, and
rendering an `ExecutorError` as a JSON error body.

### `auth.rs`

OIDC bearer-token authentication: discovery, JWKS fetch/cache/refresh, and the
`require_oidc` axum middleware layered onto protected routes in `app.rs`. The
WebSocket handler also reads the resulting `AuthenticatedPrincipal` extension directly,
to detect token expiry mid-session.

### The hard boundary: no direct storage access

**Nothing in `agentic-server`'s request-handling code (`handler/*`, `app.rs`,
`server.rs`) imports `agentic_core::storage` directly.** All persistence goes through
`AppState.exec_ctx: Arc<ExecutionContext>` — e.g. `ExecuteRequest::run()`,
`create_conversation()`, `persist_turn()`, `rehydrate_conversation()`,
`ExecutionContext::storage_ready()`. `ExecutionContext::from_config` is the only place
that constructs the storage handlers, and it does so precisely so callers don't need
to depend on the storage layer:

```rust
let conv_handler = ConversationHandler::new(ConversationStore::new(pool.clone()));
let resp_handler = ResponseHandler::new(ResponseStore::new(pool.clone()));
```

Two narrow, deliberate exceptions: the `agentic validate` CLI subcommand
(`src/bin/agentic.rs`) calls `storage::create_pool_with_schema` directly as a
connectivity pre-flight check, outside the request path; and integration tests /
benches under `crates/agentic-server/{tests,benches}` import `ConversationStore`/
`ResponseStore` directly for fixture setup and assertions. Production request-handling
code should never do either.

## `agentic-server-core` — the orchestration core

Per [AGENTS.md](AGENTS.md), the internal dependency direction is: `types/` owns
wire/domain data → `events/` parses upstream events → `tool/` owns tool discovery,
routing, and execution → `executor/` orchestrates across inference, tools, and
storage → `storage/` owns persistence. Handlers call executor APIs; the executor
coordinates `events`, `tool`, and `storage`; those share contracts through `types`.

In `src/` code, reuse `utils::common` for JSON serialization/deserialization and
fallback behavior. Do not call `serde_json` directly when an existing strict,
optional, or defaulting helper expresses the required policy; add a focused helper
there when the policy is reused. Direct `serde_json` use is fine in tests, fixtures,
and cassette tooling. Keep Serde wire-format attributes on the owning type.

### `types/` — wire shapes, not behavior

This module's job is JSON ⇄ Rust type conversion and shape validation for the
Responses and Messages APIs. It is not where tool execution, state transitions, or DB
access happen — those live in `tool/`, `executor/`, and `storage/` respectively.

- **`types/request_response.rs`** — `RequestPayload` is the deserialized incoming
  request. Its `to_upstream_request(&self, stream: bool) -> Result<UpstreamRequest<'_>, ToolError>`
  is the seam between the OpenAI-shaped request and vLLM's contract. It: flattens Codex
  namespace tool members to model-visible names, validates every declared tool
  (`ResponsesTool::validate()`), and normalizes each supported model-visible tool to
  `UpstreamTool::Function` (`ResponsesTool::to_function_tools()`). File search, code
  interpreter, and unknown typed declarations currently normalize to no upstream
  tool; every declaration that does reach vLLM is `type: "function"`, because that's
  the only tool type it speaks. The conversion also resolves/validates `tool_choice`
  and applies `ResponsesInput::model_input()`. It's called from
  `executor/upstream.rs`'s `fetch_blocking_payload` and `fetch_stream_payload` — the
  two functions that actually build the outbound request to vLLM.
- **`types/io/`** — `input.rs` (inbound message/tool-call/tool-result shapes,
  `ResponsesInput`), `output.rs` (outbound output items: messages, function calls, web
  search/MCP calls, reasoning — plus the `ApplyDone` trait described below), `tools.rs`
  (the normalized `FunctionTool` and `ToolChoice`, distinct from tool *declarations*),
  `usage.rs` (token accounting structs). `ResponsesInput::model_input()` is the final
  model-visibility boundary used by `RequestPayload::to_upstream_request`: it removes
  orchestration-only `McpListTools` and `CompactionTrigger` input items. A persisted
  `Compaction` item is different: the latest checkpoint supersedes earlier model
  context and is converted into an assistant `output_text` summary, while canonical
  retained user messages and items after the checkpoint remain. This keeps rich
  continuation state available to orchestration without sending unsupported public
  item types to vLLM.
- **`types/tools/params.rs`** — the tool **declaration** shapes a client sends:
  `ResponsesTool` (tagged enum: `Function`, `Mcp`, `WebSearch`, `FileSearch`,
  `CodeInterpreter`, `Namespace`, `Custom`, `Unknown`) and each variant's param struct.
  This is a good concrete example of the module boundary: `ResponsesTool` is *defined*
  here as a pure shape, but its behavior — `validate()` and `to_function_tools()` — is
  implemented as an `impl ResponsesTool` block physically living in
  `tool/normalize.rs`, which delegates to per-type handlers. Types own the shape; tool
  owns what it means.
- **`types/messages/`** — a separate, parallel type layer for the Anthropic Messages
  API (`MessagesRequest`, `ContentBlock`, etc.). `tool_seam.rs` is the pure, I/O-free
  adapter that converts Anthropic tool blocks into the same internal `ResponsesTool`/
  `FunctionToolCall` vocabulary the Responses-side `ToolRegistry` already understands,
  so both APIs share one tool-routing mechanism without the Messages loop depending on
  `RequestPayload`/`ResponsePayload`.
- **`types/event.rs`** — small status enums (`ResponseStatus`, `MessageStatus`).

### `events/` — parsing upstream SSE, and how to add a new event type

This module normalizes raw upstream SSE lines into typed frames, decoupled from the
executor so the accumulator doesn't do inline JSON parsing.

- **`types.rs`** — `SSEEventType` (the wire event's `type`, covering both OpenAI's and
  vLLM's naming, e.g. `response.done` vs `response.completed`), `EventPayload` (the
  typed, extracted payload — falls back to `Raw(Value)` for events not deeply parsed
  yet), `WireEvent` (the raw pass-through shape, used for re-serialization),
  `EventFrame { event_type, payload, wire }` (the normalized output), `SSEItemType`
  (output-item kind: reasoning, function call, MCP call, etc.).
- **`normalize.rs`** — `normalize_sse_line(&str) -> Option<EventFrame>` parses a
  `data: ...` line and classifies it; `extract_payload` dispatches to small per-event
  `extract_*` helpers.

**To add support for a new SSE event**, the touch points are, in order:
1. `events/types.rs` — add the `SSEEventType` variant, its wire-string mapping both
   directions, and (if it carries structured data) an `EventPayload` variant.
2. `events/normalize.rs` — extend `extract_payload` and add an `extract_*` helper if
   the payload needs real parsing (otherwise it can fall through to `Raw`).
3. Extend the single ingestion transition dispatcher. Today that is
   `executor/accumulator.rs`'s `process_event`; [#243](https://github.com/vllm-project/agentic-api/issues/243) will
   consolidate the stable entry point. Do not create a caller-specific validator or folding path.
4. If the event is gateway-synthesized (a built-in tool's lifecycle event), construct a typed `EventFrame` in
   `executor/gateway.rs` and feed it through the same ingestion/relay boundaries. Function-call shape translation
   remains an ingestion concern, currently implemented by `executor/function_sse.rs`.

### `executor/` — the loop, and the server's only door into storage

This is the layer `agentic-server` talks to. It owns the request lifecycle: rehydrate,
call inference, run the tool loop, persist. `agentic-server` never reaches past it.

- **`request.rs`** — `RequestContext` (per-turn state: original + enriched request,
  response/conversation IDs) and `ExecutionContext` (long-lived deps: storage
  handlers, HTTP client, gateway tool executors, LLM base URL). `ExecutionContext` is
  what `AppState` holds; it exposes `conv_handler`/`resp_handler` (the `modes/`
  handlers below), never the raw stores.
- **`rehydrate.rs`** — `rehydrate_conversation()` loads prior history from either the
  conversation store or the response store depending on which ID the request carries,
  and builds the enriched `RequestContext`. Rehydration retains internal
  `InputItem::McpListTools` records so `ToolRegistry` can suppress repeated MCP
  discovery lifecycle output. After stored history and the new request input are
  combined, `pending_calls.rs` validates the complete continuation's function/custom
  call sequence. Every call and call output must have a non-empty `call_id`; call IDs
  must be unique across the sequence; and each output must resolve exactly one
  currently pending call of the same item kind. An output without a pending call, a
  second output for an already resolved call, or a function/custom kind mismatch is an
  invalid request rather than evidence that the call was resolved. Valid unresolved
  calls remain ordered by their original emission, and the first unresolved
  client-executed call produces the existing missing-output error. Gateway-executed
  built-in tool calls are resolved and recorded within their originating round, so
  they do not remain pending at this boundary.
- **`upstream.rs`** — `fetch_blocking_payload`/`fetch_stream_payload`: builds the
  `UpstreamRequest` (via `to_upstream_request`, see above) and drives one round of
  upstream inference, running the accumulator and `FunctionSseTranslator` over the
  response and feeding synthesized frames through `GatewayStreamAccumulator`.
- **`inference.rs`** — `call_inference()`: the raw HTTP/SSE transport to vLLM. No
  parsing beyond splitting `data: ...` lines and stopping at `[DONE]`.
- **`engine.rs`** — the top-level orchestrator: `ExecuteRequest`/`execute()`,
  `create_conversation()`, and — this is worth being precise about —
  **`run_gateway_tool_loop` is where the multi-round tool loop actually lives**, not in
  `gateway.rs`. It calls `upstream.rs` for each round, hands the resulting output to
  `gateway.rs`'s helpers, and applies its local `classify_round`/`LoopDecision` to
  decide whether to loop again, finish, hand back to the client, or return an
  incomplete response (capped at `MAX_GATEWAY_TOOL_ROUNDS = 10`). It accumulates
  output and token usage across inference rounds, changes continuation `tool_choice`
  to `auto`, and persists gateway function calls plus their outputs as model-facing
  `InputItem`s. Also home to `run_compaction_trigger`,
  `run_blocking`, and `run_stream` (spawns the loop, forwards events as SSE, persists
  before yielding the terminal event).
- **`persist.rs`** — `persist_response`/`persist_turn`, which route to
  `ConversationHandler` or `ResponseHandler` in `modes/` depending on whether the turn
  is conversation-scoped or response-scoped.
- **`compaction.rs`** — `compact_response()` (the explicit `/v1/responses/compact`
  path) and `maybe_compact_context()` (automatic, threshold-triggered, called from the
  round loop before each inference call).
- **`modes/conversation.rs`, `modes/response.rs`** — `ConversationHandler` and
  `ResponseHandler`. Thin, 1:1 wrappers around `storage::ConversationStore` /
  `storage::ResponseStore` that translate `RequestContext` into store calls and
  `StorageError` into `ExecutorError`. **This is the sanctioned boundary between the
  executor and the storage stores** — nothing above this layer touches
  `storage::conversation`/`storage::response` directly. Today they only cover what the
  pipeline needs (`get`, `get_or_create`, `create`, `rehydrate[_snapshot]`,
  `execute_turn`, `validate_exists`); **any new CRUD operation beyond persist/rehydrate
  belongs here**, added as a new method that delegates to the corresponding store.
- **`error.rs`** — `ExecutorError`, with the mapping methods (`http_status()`,
  `error_type()`, `into_response_body()`, ...) handlers use to render errors.

#### Target streaming pipeline and ownership boundaries

The streaming executor is converging on one linear pipeline under RFC
[#241](https://github.com/vllm-project/agentic-api/issues/241). The current implementation still has overlapping
validation, accumulation, translation, and emission responsibilities in `upstream.rs`, `accumulator.rs`,
`function_sse.rs`, and `gateway_accumulator.rs`; that overlap is migration state, not an extension pattern. New work
must move toward the following ownership model:

```text
upstream HTTP bytes
        │
        ▼
inference transport ──raw SSE data line──▶ event normalization
                                                │ EventFrame
                                                ▼
                                  synchronous ingestion state machine
                                                │ validated semantic events/items
                                                ▼
                                           stream relay ──▶ client

engine.rs surrounds the per-round path: inference rounds → tool loop → persistence
```

| Stage | Owns | Must not own |
| --- | --- | --- |
| Inference transport (`inference.rs`) | HTTP request/response I/O, byte-chunk handling, SSE framing, timeouts, and `[DONE]` detection | Typed semantic-event validation, output-item lifecycle, translation, or client ordering |
| Event normalization (`events/`) | Converting one raw SSE data line into one typed `EventFrame` | Cross-event lifecycle state, response assembly, or delivery |
| Synchronous ingestion ([#243](https://github.com/vllm-project/agentic-api/issues/243)) | One entry point for normalization policy, semantic-event lifecycle validation, typed output-item slots, delta folding, tool-call shape translation, and finalization | Async task placement, client backpressure, cross-round sequencing, or persistence |
| Stream relay ([#244](https://github.com/vllm-project/agentic-api/issues/244)) | Cross-round sequence numbers, public `output_index` rebasing, lifecycle suppression, deferred-event ordering, bounded client delivery, and disconnect propagation | Re-parsing SSE data, reconstructing output items, or deciding the tool loop |
| Orchestrator (`engine.rs`) | Turn and inference-round control, tool-loop decisions, terminal-response policy, and persistence | SSE framing/parsing or a second semantic-event state machine |

The boundary contract is **one owner and one path per concern**:

- Every streamed upstream response enters the same synchronous ingestion state machine. Rejecting and compatibility
  validation policies may choose different outcomes, but they must exercise the same typed transitions rather than
  maintaining separate validators.
- An output item's lifecycle is scoped to one inference round and keyed by validated `output_index`; item ID and kind
  must agree on every subsequent semantic event. Completed slots remain distinguishable from never-seen slots so
  index reuse and duplicate completion can be detected. Finalization consumes the round's ingest state.
- Each supported output-item kind has typed in-flight state and participates in exhaustive transition/finalization
  matches. Adding a kind extends those declared matches and their tests instead of adding a side path.
- Downstream stages consume the typed result of the preceding stage. They do not parse the raw line again, infer a
  second lifecycle from the wire object, or reconstruct response state already owned upstream in the pipeline.

Concurrency is a deployment choice around this synchronous semantic core, not part of the core itself. Introduce a
channel only at a real task-ownership boundary. Every channel needs a bounded entry count and either a byte budget or
a maximum item size that gives a known memory ceiling. Define what happens when it is full, when the receiver
disconnects, and when either task is cancelled or fails; carry cancellation through the whole producer/consumer path
and join spawned tasks. Instrument entry and byte occupancy when tuning a capacity.

Run ingestion inline unless representative measurements show that worker placement improves the complete request
path. Benchmark [#245](https://github.com/vllm-project/agentic-api/issues/245) owns that decision and must compare
equivalent semantics, realistic event sizes and pacing, concurrent requests, slow consumers, tail latency, CPU,
memory, thread count, and queue occupancy. `spawn_blocking` and a capacity such as 16 entries are hypotheses, not
architectural defaults.

#### `accumulator.rs` — `ResponseAccumulator`: a stability contract, not just a file

`ResponseAccumulator` is the SSE state machine that turns a stream of `EventFrame`s
into a `ResponsePayload`. Its public surface is intentionally small
(`new`, `from_json`, `from_stream`, `from_sse_lines`, `mark_incomplete`, `finalize`) and
**should not grow outside the approved [#243 consolidation](https://github.com/vllm-project/agentic-api/issues/243)**.
Until that consolidation lands, don't add a new public method and call it from another method on the struct — that's
another surface API change. Extend current behavior through the existing pattern while preserving the target
single-ingestion boundary above:

- Each output item arrives via `response.output_item.added` and is **parked** as an
  `InFlightEntry` in `self.in_flight: IndexMap<String, InFlightEntry>`, keyed by item
  ID, in insertion order.
- Items are **constructed** via real `TryFrom<&EventPayload>` impls
  (`ReasoningOutput::try_from`, `FunctionToolCall::try_from`, `CustomToolCall::try_from`,
  `OutputMessage::try_from`, `CompactionItem::try_from`, `McpCall::try_from`,
  `McpListTools::try_from`, all in `types/io/output.rs`) — not ad hoc field-by-field
  building in the accumulator.
- Streamed deltas mutate the parked entry's buffer in place.
- Items are **completed** via the `ApplyDone` trait
  (`fn apply_done(&mut self, payload: &EventPayload, buffer: &mut String)`), applied on
  the matching `*_done` event or on `response.output_item.done`.
- Parked entries are promoted into the final `output: Vec<OutputItem>` only once, in
  `finalize_all`, which drains `in_flight`, calls each item's `finalize()`, sorts by
  `output_index`, and appends to the output — invoked at end-of-stream or on a terminal
  `response.completed|failed|incomplete` event.

If you're adding a new output-item kind: give it a `TryFrom<&EventPayload>` impl and an
`ApplyDone` impl in `types/io/output.rs`, and add it to the `InFlight` enum and the
`start_output_item`/`finalize` match arms in `accumulator.rs`. Don't restructure the
public methods to accommodate it.

#### `gateway_accumulator.rs` — `GatewayStreamAccumulator`

This is a smaller, different job than the name's similarity to `ResponseAccumulator`
suggests: it holds no `OutputItem`/in-flight item state at all. Its purpose is to make
several upstream rounds of gateway tool execution look like **one continuous SSE
stream** to the client — it assigns monotonically increasing `sequence_number`s across
rounds, rebases `output_index` so gateway-tool output lands after prior output, and
deduplicates `response.created`/`response.in_progress` so they fire once per response
rather than once per round. `gateway.rs` and `upstream.rs` both feed frames through it
via `process_event`/`synthetic_event`/`emit_sse_frame`.

This is the current precursor to the stream-relay boundary in
[#244](https://github.com/vllm-project/agentic-api/issues/244). New delivery,
buffering, and backpressure behavior belongs in that relay consolidation rather than
in the response accumulator or inference transport.

#### `function_sse.rs` — `FunctionSseTranslator`

vLLM only ever emits `function_call` SSE events, regardless of which tool type the
call is routed to. This translator looks up each call's name in the tool registry and
reshapes the raw stream accordingly:
- **Custom tools** — rewritten into the public `custom_tool_call` event shape
  (`output_item.added` / `custom_tool_call_input.delta` / `.done` / `output_item.done`),
  reconstructing the `input` JSON incrementally from the streamed `arguments`.
- **Gateway-owned tools** (`Mcp`, `WebSearch`, `FileSearch`, `CodeInterpreter`) — raw
  frames are suppressed entirely. Their real client-visible events are synthesized
  later, once the call has actually executed, by `gateway.rs`.
- **Client-owned tools** (`Function`, `CodexNamespace`) — pass through unchanged.

It also buffers function-call events that arrive before the call's name is known
(bounded at 256 KiB) and replays them once the name resolves.

#### `gateway.rs` — the tool-loop's building blocks

As noted above, the round-by-round loop itself is `engine.rs::run_gateway_tool_loop`.
`gateway.rs` supplies what that loop calls each round:
- `GatewayScheduler::plan` creates one slot per gateway-owned function call. Each slot
  owns the original item index, public output index, typed `GatewayBinding`, and
  lifecycle projection; a missing executor is represented by an explicit slot rather
  than omitted from a parallel vector. `GatewayScheduler::execute` then returns one
  ordered `GatewayCallResult` per slot. A `futures::stream::buffered` sliding window
  bounds fan-out using `tools.max_concurrent_gateway_calls` (default `5`, configurable
  through `AGENTIC_MAX_CONCURRENT_GATEWAY_CALLS`). The setting is a nonzero value
  carried by the owning `ExecutionContext` into each scheduler, so independent
  contexts do not share process-global policy and `.buffered(0)` is unrepresentable.
  Completion may occur out of order, but the collected result order always matches
  model call order.
- A normalized `web_search` function call may batch at most five queries. The JSON
  Schema advertises the ceiling and the handler enforces it again because normalized
  web search currently uses non-strict arguments. Provider searches acquire a shared
  handler semaphore initialized from `tools.max_concurrent_gateway_calls`, preventing
  batched calls from multiplying the configured outbound concurrency. Results remain
  collected in query order for the public `web_search_call.action.queries` projection.
- Every call has an independent 60-second timeout. Timeout, execution, and tool-config
  failures become failed tool outputs that can be fed back to the model instead of
  failing the whole response. A tool registered as gateway-owned without an
  implementation (currently file search/code interpreter) likewise produces an error
  tool result.
- Parallel safety is a per-handler contract. `GatewayExecutor::supports_parallel_execution`
  defaults to `false`; registration turns that into a `GatewayBinding::self_exclusion`
  semaphore. The semaphore serializes only simultaneous calls to the **same
  model-visible tool name**. It never blocks different tools from running concurrently.
  MCP and web search opt into same-tool parallel execution.
- Each scheduler slot retains its `GatewayEventPlan`; `emit_gateway_start_events` and
  `emit_gateway_completed_events` synthesize the OpenAI lifecycle for gateway-executed
  web search/MCP calls from those same slots. The ordinary path emits all planned start
  events, executes the round concurrently, then emits ordered completed/failed events.
- Streaming may receive client-visible output interleaved with gateway calls. In that
  case `engine.rs::execute_and_emit_ordered_output_calls` temporarily groups deferred
  upstream frames by `output_index`, executes the same `GatewayScheduler` concurrently,
  and then interleaves synthetic gateway lifecycle events with released upstream
  frames in original output order. Concurrency and wire ordering are therefore
  separate concerns.
- `public_output_items` is the public projection: custom function calls become
  `custom_tool_call`; gateway-owned internal function calls become their handler's
  `web_search_call`/`mcp_call` output; client-owned function calls remain function
  calls. The original gateway function calls and `function_call_output` results are
  retained separately for continuation persistence.

The round decision remains in `engine.rs`, after gateway execution:

| Decision | Condition and state transition |
|---|---|
| `RequiresClientAction` | At least one client-owned call exists. Any gateway calls from the mixed round have already executed; their internal calls/results are recorded before returning. |
| `Done` | No gateway result and no client-owned call remains. Finalize accumulated output and usage. |
| `Continue` | Gateway calls ran and round budget remains. Append the upstream output plus gateway results, set `tool_choice: auto`, and infer again. |
| `Incomplete` | Gateway calls ran on the tenth round. Record the final calls/results and return `status: incomplete` instead of leaving a dangling call. |

`parallel_tool_calls` is an upstream model-generation preference, not a gateway
scheduler switch. It is forwarded to vLLM for all supported declaration mixtures and
defaults to `false` when omitted. Whatever calls the model emits are executed under
the global sliding window and each handler's same-tool safety policy.

#### `messages_loop.rs` / `messages_request.rs` / `messages_stream.rs`

A **parallel, independent implementation** of the same shape of loop for the Anthropic
Messages API. `messages_stream.rs`'s own header comment describes it as "structurally
the Anthropic-native analogue of `GatewayStreamAccumulator`, kept deliberately parallel
for a future consolidation" — it never touches `RequestPayload`/`ResponsePayload`/
`ResponseAccumulator`/`GatewayStreamAccumulator`/`FunctionSseTranslator`, operating
directly on Anthropic-shaped JSON. The two loops share only the protocol-neutral
pieces: `ToolRegistry::dispatch` and `types::messages::tool_seam`. The round/timeout
constants (`MAX_GATEWAY_TOOL_ROUNDS`, `GATEWAY_TOOL_TIMEOUT`) are duplicated and
manually kept in sync with the Responses-side ones rather than shared — a known seam,
not an oversight, per the future-consolidation note.

### `storage/` — persistence

- **`pool.rs`** — `DbPool = sqlx::Pool<sqlx::Any>`, driver-agnostic across SQLite and
  Postgres. `create_pool`/`create_pool_with_schema` and friends build and tune it
  (WAL mode + busy-timeout retry on SQLite, statement/lock timeouts on Postgres).
- **`backend.rs`** — `DatabaseBackend` (Postgres/Sqlite/Other) detection from a
  connection URL, plus URL redaction for safe logging.
- **`schema.rs`** — migrations and readiness (`PoolWithSchema::ensure_schema_ready`),
  including a path for a supervisor-managed schema that skips running migrations
  itself and just verifies compatibility.
- **`models/`** — raw `sqlx::FromRow` row structs per table (`Conversation`, `Item`,
  `Response`) plus their raw, transaction-aware SQL functions (`create_in_tx`, `get`,
  `lock_in_tx`, ...). This is the literal DB row shape: JSON columns are still strings
  here.
- **`types/`** — the conversion layer from those raw rows into business types, via
  `From`/`TryFrom` impls: `ConversationData`/`ConversationSnapshot`, `ResponseData`/
  `ResponseMetadata` (parses the JSON metadata column into a typed struct),
  `InOutItem` (parses an `Item.data` JSON blob back into a typed `InputItem` or
  `OutputItem`), and `StorageError`. `InOutItem::into_input_items` turns a full
  history into the `Vec<InputItem>` used for continuation processing: stored
  `InputItem`s pass through, while stored `OutputItem`s go through
  `OutputItem::to_input_item()`. Messages, reasoning, function/custom calls,
  compaction checkpoints, and MCP list-tools records are retained. Public
  `web_search_call` and `mcp_call` outputs are deliberately omitted because their
  model-facing function calls and results are already persisted as input items;
  reconstructing them here would duplicate and lose information from that canonical
  pair.

  This conversion is **not** the model visibility boundary. The resulting enriched
  history still contains `InputItem::Compaction`, `InputItem::CompactionTrigger`, and
  `InputItem::McpListTools` for executor/registry decisions. Immediately before an
  upstream request, `RequestPayload::to_upstream_request` calls
  `ResponsesInput::model_input()`: the latest compaction checkpoint is converted to an
  assistant summary and supersedes older context, while compaction triggers and MCP
  list-tools records are removed. In particular, MCP list-tools remains available long
  enough for the registry to remember which server labels have already been listed,
  but it is never serialized to vLLM.
- **`conversation.rs`, `response.rs`** — `ConversationStore` and `ResponseStore`: the
  CRUD-with-transactions layer (`create`, `get`, `get_or_create`, `rehydrate[_snapshot]`,
  `persist`/`persist_if_version` — each transactional, via `pool.begin()` /
  `tx.commit()`). **These are not to be called outside `executor/` and `storage/`
  themselves.** The only sanctioned callers are `executor/modes/conversation.rs` and
  `executor/modes/response.rs`, described above. (Integration tests and benches import
  them directly for fixtures — that's expected and fine; production code paths should
  not.)

### `tool/` — the tool framework

Wire shapes for tool declarations live in `types::tools` (see above); this module owns
the behavioral layer — routing, handler traits, normalization, and execution.

- **`normalize.rs`** — the `impl ResponsesTool` block with `validate()` and
  `to_function_tools()`. Both are a match over the tool-kind enum that delegates to
  each type's `ToolHandler`: e.g. `Function` → `FunctionHandler`, `Mcp` → `McpHandler`,
  `Namespace` → `CodexNamespaceHandler`, `Custom` → `CustomHandler`. `WebSearch` is
  normalized inline from a static builder (single fixed tool, no per-instance state).
  `FileSearch`/`CodeInterpreter` are declared but currently normalize to nothing — no
  handler is registered yet.
- **`handler.rs`** — the two traits every tool type reasons about:
  ```rust
  pub trait ToolHandler: Send + Sync {
      type ToolParams: Send + Sync;

      fn tool_type(&self) -> ToolType;
      fn validate(&self, params: &Self::ToolParams) -> Result<(), ToolError>;
      fn normalize(&self, params: &Self::ToolParams) -> Vec<FunctionTool>;
  }

  pub trait GatewayExecutor: ToolHandler + 'static {
      type ExecutionParams: Clone + Send + Sync + 'static;

      fn execute(
          &self,
          call_id: &str,
          tool_name: &str,
          arguments: &str,
          params: &Self::ExecutionParams,
      )
          -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>>;
      fn supports_parallel_execution(&self) -> bool;
      fn plan_gateway_events(
          &self,
          call: &FunctionToolCall,
          params: &Self::ExecutionParams,
      ) -> GatewayToolEventPlan;
      fn public_output(
          &self,
          call: &FunctionToolCall,
          output: &ToolOutput,
          status: GatewayCallStatus,
          params: &Self::ExecutionParams,
      ) -> Option<OutputItem>;
  }
  ```
  `GatewayExecutor` requires `ToolHandler`: every executable gateway handler supports
  typed validation and normalization, but not every `ToolHandler` is gateway-executable.
  `ToolParams` describes the public declaration; `ExecutionParams` describes one
  model-visible executable entry. They intentionally differ for MCP: an
  `McpToolParam` declares a server, while an `McpDiscoveredToolParam` identifies one
  tool returned by that server. Gateway-owned registry types may also lack an executor
  entirely. The trait owns three runtime hooks: `supports_parallel_execution()`
  controls same-tool self-exclusion, `plan_gateway_events()` creates the typed public
  lifecycle projection, and `public_output()` shapes the completed/failed
  client-visible item.
  - **Client-owned** tools implement only `ToolHandler`: see `function.rs`
    (`FunctionHandler`), `custom.rs` (`CustomHandler`), `codex.rs`
    (`CodexNamespaceHandler`). Their calls are returned for the client to resolve — the
    gateway never executes them.
  - **Gateway-owned / built-in** tools implement both traits: see `web_search.rs`
    (`WebSearchHandler`, backed by You.com) and `mcp/handler.rs` (`McpHandler`, backed
    by `mcp/client.rs`'s MCP protocol client and `mcp/pool.rs`'s connection pool).
- **`ownership.rs`** — `ToolOwnership::Client` versus
  `ToolOwnership::Gateway(Option<GatewayBinding>)`. A `GatewayBinding` combines the
  resolved executor, its typed `ExecutionParams`, and the optional same-tool semaphore
  derived from its parallel-safety declaration. A generic adapter checks the
  executor/parameter pair at construction and erases only that valid bound pair for
  heterogeneous registry storage. The scheduler therefore never handles untyped JSON
  configuration or downcasts. Keeping ownership explicit avoids inferring execution
  policy from whether a handler happens to be present.
- **`registry.rs`** — `ToolRegistry`, a request-scoped map from model-visible tool name
  to `ToolEntry { tool_type, server_label, ownership }`. Executable parameters live in
  the typed `GatewayBinding`, not as a serialized `Value` on every entry. Its constructor,
  ```rust
  pub async fn build_with_handlers(
      tools: &mut [ResponsesTool],
      executors: &mut GatewayExecutors,
  ) -> Result<Self, ToolError>
  ```
  is the stable entry point every caller (Responses and Messages) uses to build a
  registry for a request — **its signature should not change**. It resolves namespace
  members, inserts one entry per declared/discovered tool, and for `Mcp`/`WebSearch`
  pulls the actual executor from `GatewayExecutors` (discovering live MCP tools via
  `tools/list` in the process). `ToolRegistry::dispatch(call)` is the per-call routing
  method the Messages loop uses; the Responses `GatewayScheduler` resolves the same
  binding into one call plan so execution, self-exclusion, item position, and lifecycle
  hooks cannot drift apart.

  MCP discovery history is also request-scoped registry state:
  `mcp_list_tools_items: HashMap<String, Vec<McpListTools>>` groups records by
  `server_label`. Registry construction puts the current discovery item first;
  rehydration appends prior `InputItem::McpListTools` records only for labels already
  present in that map. `mcp_list_tool_items()` exposes entries whose vector still has
  exactly one element—the current item with no history—to both blocking output
  assembly and streaming lifecycle emission. Streaming clears the map after the first
  inference round. Consequently a server's list lifecycle is emitted only when no
  prior list record exists and never repeats across rounds.
- **`executors.rs`** — `GatewayExecutors`, a shared registry built once at startup and
  reused across requests, specifically for gateway tools that need **lazy, per-request
  connection setup**: MCP servers (connects and caches `McpClient`s keyed by server
  URL, falling back to connecting a fresh request-declared server) and the shared
  `WebSearchHandler`. As of today it only has slots for `ToolType::Mcp` and
  `ToolType::WebSearch`; `GatewayExecutorRegistration` has typed variants for those
  supported slots. Client-owned
  tools (`function`, `custom`, `namespace`) never touch this file; their registry
  entries are inserted with `ToolOwnership::Client` and no `GatewayExecutors`
  involvement.

**To add a new tool type:**
1. Implement `ToolHandler`, including its typed `ToolParams`, for it. If it's client-executed,
   stop there — see `function.rs`/`custom.rs` for the pattern.
2. If it's gateway-executed, also declare typed `GatewayExecutor::ExecutionParams`
   and implement `execute` — see `web_search.rs`/`mcp/handler.rs`.
3. Wire it into `tool/normalize.rs`'s `validate`/`to_function_tools` match arms.
4. Wire it into `tool/registry.rs`'s `build_with_handlers` (an `insert_*_entry` call).
5. If it needs lazy per-request connection setup, add a slot to `GatewayExecutors` in
   `tool/executors.rs` and reference it from the registry's match arm for that type.

## `agentic-praxis`

Currently a placeholder (`src/lib.rs` is a comment describing intent). Per ADR-03, this
crate will eventually provide `HttpFilter` implementations, one per
`agentic-server-core` public function, composed into a Praxis filter chain with branch
support for tool-call looping — an alternative orchestrator to `agentic-server`'s axum
router, reusing the same core logic in-process.

## Quick reference: "I want to..."

| Task | Where |
|---|---|
| Add a new HTTP or WebSocket route | `agentic-server/src/handler/{http,websocket}/`, wire it in `app.rs`'s `build_router_with_auth` |
| Support a new upstream SSE event | `events/types.rs` → `events/normalize.rs` → the single ingestion dispatcher tracked by [#243](https://github.com/vllm-project/agentic-api/issues/243); do not add a caller-specific path |
| Add a new tool type | `tool/handler.rs` impl(s) → `tool/normalize.rs` → `tool/registry.rs` → `tool/executors.rs` if it needs lazy connection setup |
| Change gateway-round concurrency or lifecycle ordering | `executor/gateway.rs` (`GatewayScheduler`/event plans) + `executor/engine.rs` (round decision/ordered streaming) + `tool/ownership.rs` (typed binding and same-tool safety) |
| Change client streaming order, buffering, or backpressure | The stream-relay boundary tracked by [#244](https://github.com/vllm-project/agentic-api/issues/244); do not add it to `inference.rs` or the response accumulator |
| Move streaming ingestion to a worker | Benchmark the equivalent inline and worker paths under [#245](https://github.com/vllm-project/agentic-api/issues/245) before changing executor placement |
| Change continuation history visibility | `storage/types/item.rs::into_input_items` → `types/io/output.rs::to_input_item` (preservation) → `types/io/input.rs::model_input` (upstream visibility) |
| Add a CRUD operation beyond persist/rehydrate | `executor/modes/conversation.rs` or `modes/response.rs`, backed by `storage/conversation.rs` / `storage/response.rs` |
| Change how output items are assembled from a stream | `executor/accumulator.rs` — respect the `TryFrom`/`ApplyDone` pattern, don't add new public methods |
| Add a new Responses/Messages wire field | `types/io/` or `types/messages/` — shape only, no behavior |

## Further reading

- [AGENTS.md](AGENTS.md) — module boundaries, lint/format rules, commit and PR conventions
- [TERMINOLOGY.md](TERMINOLOGY.md) — normative vocabulary for API/state/tool/streaming concepts
- [ROADMAP.md](ROADMAP.md) — project direction and near-term focus
- [docs/adr/](docs/adr/) — architecture decision records
- [docs/design/](docs/design/) — as-built design docs (tool framework, core public API, MCP integration, Codex integration)
