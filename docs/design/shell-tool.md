# Shell tool execution

`{"type":"shell","environment":{"type":"local"}}` declares a client-executed
tool. The gateway returns a typed `shell_call`; the client runs the
commands in its own environment and submits `shell_call_output` with the same
`call_id`. Explicit `tool_choice: {"type":"shell"}` is supported.

The public shell items remain in stored history. Stored calls use the standard
output-to-input conversion to become `shell` function calls during rehydration.
Typed input conversions also lower submitted shell calls and outputs; inference
normalizes the declaration and selector to the matching function representation.
Completed calls report `completed` in both blocking
responses and streaming `response.output_item.done` events. Native shell streams
can also be consumed through `decode_upstream` and its strict lifecycle validation.

For client execution, normalized function arguments are translated incrementally
into `response.shell_call_command.added`, `.delta`, and `.done` events with
`output_index` and `command_index`. The initial shell item has an empty commands
array; the completed item contains the full action and limits. JSON string
escapes (including split Unicode surrogate pairs) are decoded before command
deltas are emitted. Native shell command events use the same ingestion lifecycle
checks; malformed indices, repeated completion, and contradictory command text
are rejected. The OpenAI recordings in `tests/cassettes/shell` are replayed by
`tests/shell_tool_test.rs` alongside client-continuation tests.
