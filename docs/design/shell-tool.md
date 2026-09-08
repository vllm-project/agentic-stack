# Shell tool execution

`{"type":"shell","environment":{"type":"local"}}` declares a client-executed
tool by default. The gateway returns a typed `shell_call`; the client runs the
commands in its own environment and submits `shell_call_output` with the same
`call_id`. Explicit `tool_choice: {"type":"shell"}` is supported.

The public shell items remain in stored history. Immediately before inference,
the declaration, selector, calls, and outputs are lowered to the matching `shell`
function representation. Completed calls report `completed` in both blocking
responses and streaming `response.output_item.done` events. Native shell streams
can also be consumed through `decode_upstream` and its strict lifecycle validation.

## Application-provided sandbox

An embedding application can opt into gateway execution without changing core:

```rust,ignore
use std::sync::Arc;
use agentic_core::tool::GatewayExecutorRegistration;

// sandbox implements agentic_core::tool::ShellExecutor in the application.
let context = context.with_gateway_executor(
    GatewayExecutorRegistration::Shell(Arc::new(sandbox)),
);
```

The `ShellExecutor` trait receives an owned, typed `ShellCall` and a
`CancellationToken`, and returns one `ShellCallOutputContent` per command.
Its action carries effective execution limits: a maximum 60-second timeout and
a maximum 1 MiB of total captured UTF-8 stdout/stderr bytes. Smaller requested
limits are preserved; omitted limits use these ceilings. The adapter must enforce
the output budget while capturing output, not accumulate an unbounded buffer.
Core additionally checks the returned output size and count and bounds the future
by the timeout. Dropping that future cancels the token, including on an outer
timeout or client disconnect. The sandbox must stop and reap its own subprocesses
on cancellation and enforce its own filesystem, network, and command policies.
Cancellation is cooperative; a token is not an OS-level process kill.

No local command runner is installed by core, configuration loading, or a request
declaration. Registration is a deployment-level grant for requests using that
`ExecutionContext`; restrict access and bind the appropriate per-tenant context.
Same-name shell calls are serialized by the existing gateway scheduler. Successful
typed command outputs become internal function outputs for the next inference
round; errors become error outputs and an `incomplete` public shell-call status.

`crates/agentic-server-core/tests/shell_tool_test.rs` implements an adapter outside
the library and exercises two-round blocking/streaming execution, strict public
stream replay, client continuation, and subsequent stored-history rehydration.
These are constructed regression fixtures, **not** live OpenAI/vLLM recordings.
