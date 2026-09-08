# Terminology

This file defines the preferred vocabulary for code, documentation, issues, and pull requests in this repository.
Use the exact protocol spelling for wire fields and item types. For prose, prefer the OpenAI term below over a local
synonym. Existing identifiers may retain older wording until they are changed for a separate technical reason.

OpenAI documentation is the primary terminology source. Project-specific terms are included only where this repository
adds a distinct implementation concept, such as rehydration or tool normalization.

## Preferred wording

| Prefer                                   | Avoid in new prose                   | Reason                                                                                                             |
| ---------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| built-in tool                            | gateway-owned tool, server-side tool | Matches OpenAI's category for tools supplied by the API platform. Add an execution-location qualifier when needed. |
| gateway-executed built-in tool           | gateway-owned tool                   | Separates the tool category from where this deployment executes it.                                                |
| client-executed function tool            | client-owned tool                    | Function tools are defined by the application; the qualifier states who executes a call.                           |
| upstream-hosted built-in tool            | provider-owned tool                  | Describes both the OpenAI-compatible tool category and its execution location.                                     |
| tool call output or function call output | tool result                          | Matches the Responses API item name. Use `tool_result` only when discussing the Anthropic wire type.               |
| item history                             | message history                      | Responses history can contain messages, tool calls, tool call outputs, and reasoning items.                        |
| streaming event                          | streaming chunk                      | Responses streaming uses typed semantic events. Use chunk only for an untyped transport fragment.                  |
| tool call or function call               | tool invocation                      | Matches OpenAI documentation and Responses wire types.                                                             |
| pass through                             | proxy transparently                  | Use either precise verb instead of ownership language when the gateway does not execute or transform a tool.       |

The execution-location qualifiers are deliberately independent of the tool type:

- A **gateway-executed built-in tool** is supplied and executed by vLLM Agentic API.
- A **client-executed function tool** is declared in the request, returned as a function call, and executed by the
  client or application before it submits a function call output.
- An **upstream-hosted built-in tool** is sent to vLLM or another upstream provider for that upstream to execute.

Use the shorter **built-in tool** or **function tool** when the execution location is irrelevant. Existing Rust names
such as `GatewayExecutor` and `is_gateway_owned` describe current implementation concepts; they do not establish the
preferred prose terminology.

## Core API and state

### Agentic API

An API surface that can coordinate model generation, tools, and state across one or more model calls. Use lowercase
**agentic API** for the general category and **vLLM Agentic API** or `agentic-api` for this project.

### Responses API

The OpenAI-compatible API centered on the `/v1/responses` endpoint and typed items. Capitalize **Responses API** and
use **Responses** only when the context clearly identifies the API.

### response

The typed object produced by one Responses API request. A response has an ID, status, output items, and other metadata.
Do not use response as a synonym for an assistant message; a message is one possible output item.

### item

A typed unit in Responses input, output, or conversation state. Messages, function calls, function call outputs, and
reasoning items are all items.

### input item

An item supplied in a Responses request. Input may be a string or a list of input items.

### output item

An item in a response's `output` array. Prefer **output item** over response item when referring specifically to that
array.

### message

An item containing role-based content. In Responses, a message is one item type rather than the container for every
model action.

### content part

A typed element inside a message's `content` array, such as `input_text`, `input_image`, or `output_text`. Prefer
**content part** over content item to avoid confusing it with a top-level Responses item.

### conversation

A durable object that stores an ordered sequence of items and can be used across responses, sessions, devices, or
jobs. Do not use conversation as a synonym for a single response.

### turn

The work associated with one new user input and the model/tool activity needed to answer it. A turn can contain
multiple internal inference rounds when tools are called.

### active turn

The turn currently being processed, including every inference round in its tool loop. In reasoning-context
discussions, same-turn reasoning remains part of the active turn even when it was produced by an earlier inference
round.

### conversation state

The prior items and metadata made available to a later turn. State may be managed with a conversation, chained with
`previous_response_id`, or replayed manually. WebSocket response state may also be
retained transiently by the active connection; it is distinct from durable conversation storage.

### stored response

A response retained by the service for later retrieval or continuation. Storage is controlled by the API's `store`
semantics.

### continuation

Starting a later response with prior response or conversation state. Prefer this general term over chaining when the
mechanism could be either `previous_response_id` or a conversation.

### previous response ID

The response identifier passed in the `previous_response_id` field to continue from prior response state. That state
may be durable or cached on the active WebSocket connection. In prose,
write **previous response ID**; in code and wire-format discussion, use `previous_response_id`.

### rehydration

The project-specific process of loading stored items, restoring their order and effective request settings, and
building the input for continuation. Rehydration is an implementation step, not an OpenAI wire object.

### persistence

Writing response or conversation state to durable storage. Use **storage** for the API behavior controlled by `store`
and **persistence** for the implementation that realizes it.

### stateful

Describes a flow in which the service retains or resolves prior state, such as Responses continuation through
`previous_response_id` or a conversation.

### stateless

Describes a flow in which the request supplies all required context and the service does not rely on retained response
or conversation state. `store: false` disables durable response storage, although callers may still replay prior items
or continue from the active WebSocket connection's transient checkpoint. An explicit conversation remains durable.

### compaction

Replacing a long item history with a smaller canonical window that preserves user messages and a model-generated
summary. Compaction is distinct from truncation: it uses an inference call to summarize context rather than silently
dropping items.

### compaction item

An item with `type: "compaction"` whose `encrypted_content` carries the context checkpoint consumed by a later model
request. Preserve the exact field name in wire-format discussion. In this project's local implementation the field is
plaintext despite its protocol name.

### compacted window

The canonical output of a compaction request: retained user-message items followed by one compaction item. Replay the
complete window as later Responses input; do not extract only the summary.

### context management

The Responses request policy expressed by `context_management`. A `type: "compaction"` entry with
`compact_threshold` asks the gateway to compact resolved input before inference when its estimated token count exceeds
the threshold.

## Tools and tool calling

### tool

Functionality made available to the model. A tool definition describes what the model may call; it is distinct from a
tool call made by the model.

### tool calling

The overall mechanism by which a model requests tool use and receives tool call outputs. **Function calling** is an
accepted synonym when the discussion is specifically about function tools.

### function tool

A tool defined by a JSON Schema for its arguments. The model emits a function call, and application-side code normally
executes it and submits a function call output.

### custom tool

An OpenAI tool type whose call input is free-form text rather than JSON-Schema-constrained function arguments. Do not
use custom tool as a generic synonym for any user-defined function tool.

### built-in tool

A tool supplied by the API platform, such as web search, file search, code interpreter, computer use, or the `mcp`
tool. In this project, use an execution-location qualifier if it matters whether Agentic API or the upstream serves
the built-in capability.

### tool call

A model-generated request to use a tool. **Function call** is the more precise term for an output item with
`type: "function_call"`.

### tool call output

The output produced by executing a tool call and returned to model context. Use **function call output** for an input
item with `type: "function_call_output"`. The output references its call through `call_id`.

### call ID

The identifier that associates a tool call output with the tool call it answers. Write **call ID** in prose and
`call_id` when referring to the wire field.

### tool choice

The request policy that controls whether the model may, must, must not, or must specifically use a tool. Use
`tool_choice` for the wire parameter.

### strict mode

Function-calling mode in which generated arguments must adhere to the function's schema. Do not use strict mode as a
synonym for general request validation or Rust deserialization strictness.

### parallel function calling

A model producing multiple function calls in one turn. Use **parallel tool calls** only when discussing the
`parallel_tool_calls` parameter or a broader implementation that includes non-function tools.

### tool loop

The orchestration sequence that sends tools to the model, receives tool calls, executes applicable calls, appends tool
call outputs, and invokes inference again until the turn completes or requires client action.

### inference round

One model invocation inside a turn or tool loop. Prefer **round** for this internal iteration and **turn** for the
user-visible unit of interaction.

### tool registry

The project-specific request-scoped mapping from model-visible tool names to their original type and explicit
ownership. Gateway-owned entries may contain a `GatewayBinding` that pairs an executor with its typed execution
parameters and same-tool concurrency policy. The registry routes calls after inference and also retains MCP
discovery-history metadata; it is not part of the Responses wire format.

### tool normalization

The project-specific conversion of heterogeneous tool declarations into the function-tool shape accepted by the
upstream inference server. Normalization changes the upstream representation, not the public tool's meaning.

### pass-through

Forwarding a request, field, tool declaration, call, response, or error without executing it locally. Use
**transparent proxying** when emphasizing preservation of the upstream protocol behavior.

## MCP

### Model Context Protocol (MCP)

The protocol used to connect models and applications to external tools and data sources. Spell out the name on first
use in a document, followed by **MCP**.

### MCP server

A service that exposes MCP capabilities such as tools and resources. Do not call the service itself an MCP tool.

### remote MCP server

An MCP server reached over a remote transport, normally Streamable HTTP or HTTP/SSE. **Remote** describes the
connection mode, not who operates the server.

### MCP tool

A callable capability advertised by an MCP server. When the Responses API accesses it through a declaration with
`type: "mcp"`, describe `mcp` as a built-in tool and the imported callable capability as an MCP tool.

### MCP resource

Data exposed by an MCP server and identified by a URI. Reading a resource is distinct from calling an MCP tool even if
the gateway offers a function tool such as `read_mcp_resource` to bridge that operation.

### MCP tool call

A request to execute an MCP tool. Use the exact wire item type required by the implemented API version when discussing
serialization; use **MCP tool call** in version-independent prose.

### MCP approval

An explicit authorization step before data is shared with, or an action is performed through, an MCP server. Do not
use approval to mean ordinary tool selection by the model.

## Streaming and transport

### streaming

Delivering incremental response events before the entire response is complete. For HTTP Responses, `stream: true`
uses server-sent events; Responses WebSocket mode uses a persistent WebSocket.

### semantic event

A typed Responses streaming event with a defined schema and lifecycle meaning, such as
`response.output_item.added`, `response.output_text.delta`, or `response.completed`.

### delta event

A semantic event carrying an incremental fragment of a field, such as output text or function-call arguments. A delta
is not a complete output item.

### terminal event

The final lifecycle event for a response, such as `response.completed`, `response.incomplete`, or `response.failed`.

### server-sent events (SSE)

The HTTP event-stream transport used for Responses streaming. Spell out the name on first use in a document, followed
by **SSE**. SSE is the transport; semantic events are the typed payloads carried over it.

### Responses WebSocket mode

The persistent WebSocket transport for repeated `response.create` events. It uses the same response and
`previous_response_id` concepts as HTTP Responses, but it is not SSE.

## Reasoning

### reasoning item

An output item with `type: "reasoning"` that carries opaque reasoning state and may carry a summary. Preserve relevant
reasoning items with function calls and function call outputs as required by the model and API contract.

### reasoning summary

A model-generated summary exposed in a reasoning item's `summary` array when requested and supported. It is not the
model's raw chain of thought.

### reasoning text

Plaintext reasoning content emitted by an upstream model or reasoning parser. This is distinct from OpenAI's opaque
reasoning state and from a reasoning summary. Use the exact `reasoning_text` name only for a wire event or content type
that defines it.

### reasoning context

The policy, expressed by `reasoning.context`, that controls which available reasoning items may be rendered into later
model context. Keep the wire values exact: `auto`, `current_turn`, and `all_turns`.

## Runtime boundaries

### gateway

The vLLM Agentic API process that accepts client-facing requests, manages state and orchestration, and communicates
with the upstream inference service. Prefer **vLLM Agentic API** in user-facing prose and **gateway** in architecture
or execution-location discussion.

### upstream

The service to which the gateway sends a model inference request, usually vLLM core but potentially another
OpenAI-compatible provider. Upstream is a request-flow role, not necessarily a third-party provider.

### inference

Model generation performed by the upstream. Tool execution, storage, transport handling, and rehydration are
orchestration rather than inference.

## Protocol-specific vocabulary

When documenting another protocol, preserve that protocol's exact names and state the mapping instead of silently
renaming its wire types. For example:

- Anthropic `tool_use` maps conceptually to a tool call.
- Anthropic `tool_result` maps conceptually to a tool call output.
- A Codex `namespace` is a grouping of function tools, not an MCP server or a tool executor.

## Sources

These definitions follow current OpenAI documentation:

- [Migrate to the Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses)
- [Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)
- [Function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [Using tools](https://developers.openai.com/api/docs/guides/tools)
- [MCP and Connectors](https://developers.openai.com/api/docs/guides/tools-connectors-mcp)
- [Streaming API responses](https://developers.openai.com/api/docs/guides/streaming-responses)
- [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning)
- [Compaction](https://developers.openai.com/api/docs/guides/compaction)
