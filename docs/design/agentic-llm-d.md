# agentic-llm-d

## Scope

`agentic-llm-d` runs agentic-api as a pair of state services for the llm-d coordinator, which performs the inference
call itself. It decomposes one stateful Responses turn into two separately callable steps, so a caller that already
routes model traffic does not have to proxy that traffic back through the gateway.

Serving `previous_response_id` normally means reaching whichever engine holds the earlier turn. Keeping that history
behind an API removes the constraint: the request the coordinator forwards carries its own history, so any engine can
serve it. Cache-aware scoring can still prefer the engine that handled the previous turn without being bound to it.

## The two steps

`hydrate` takes the client's request, resolves `previous_response_id` against storage, and returns two things: the
upstream request body with the history inlined and every continuation and storage field removed, and a sealed context
describing the turn. The caller forwards the body to a model unchanged and echoes the context back.

`persist` takes that context together with the response the model produced, either a complete JSON body or the SSE
frames a streaming caller has already relayed to its own client, assembles the turn, stores it, and returns the
response envelope carrying the reserved `resp_` identifier. Nothing is re-emitted for a streamed turn, since the caller
has already sent the frames on.

`SplitContext` is the wire form of the in-process `RequestContext`. It omits the enriched request, which is already in
flight as the request body, and the derived input items; both are rebuilt when the context comes back. It travels as an
HMAC-signed token carrying an expiry and an audience, so `persist` can prove `hydrate` issued it rather than trusting a
caller-authored context. Callers treat it as opaque.

## Composition

Neither step reimplements the storage flow. `hydrate` calls `rehydrate_conversation` and `upstream_request`; `persist` calls
`decode_upstream` and `commit`. All four are core operations the in-process executor already uses or shares, so a change
to how a turn is rehydrated or stored reaches both paths at once. The llm-d crate contains no storage access or
request-building logic of its own.

What it does contain is the boundary itself: the check for what cannot be split, strict validation for the relayed
JSON or streaming events, the conversion between the live and wire context forms, and the sealing.

Core keeps what is not specific to one consumer. `decode_upstream` is the OpenAI JSON/SSE adapter; it validates a
caller-supplied body more strictly than the in-process parser, which defaults a missing status and drops items it
cannot read. `commit` takes a normalized `ResponsePayload` and owns the rest of the shared behaviour:
reserved-identifier and terminal-status validation, atomic duplicate detection during insertion, and conditional
persistence. A response still in progress is something an external caller can return but the in-process flow never
produces; storing it would hand back an identifier that could never be continued, so `commit` rejects it.

`ensure_splittable` reuses `RequestPayload::in_process_feature`, the predicate that already decides whether the gateway
runs the executor or passes a request through to vLLM. The passthrough proxy and the split boundary have the same
limits, so sharing one predicate stops them drifting apart as features are added.

## Boundary

`ensure_splittable` names the feature that prevents a request being split: `conversation_id`, gateway-executed built-in tools,
compaction input, or `context_management`. Each needs state that the in-process executor keeps between steps.

## The crate

The endpoints are served by a separate crate and binary that depends on `agentic-server-core` and not on the gateway.
It serves `/v1alpha/responses/hydrate`, `/v1alpha/responses/persist`, `/health` and `/ready`, and nothing else: the
passthrough proxy, `/v1`, the WebSocket transport, upstream readiness probing and vLLM subprocess management are all
absent, so these endpoints cannot be exposed on a listener that also serves `/v1`. Readiness reports whether storage
answers, since the coordinator owns the model fleet.

## Calling and deployment contract

Every hydrate or persist request must send the shared workload credential in
`x-agentic-workload-token`. Configure the token with `AGENTIC_LLM_D_API_TOKEN` and configure the independent context
signing key with `AGENTIC_LLM_D_SIGNING_KEY`. Each secret must contain at least 32 non-whitespace bytes. The binary
rejects missing, short, or reused values before opening storage or binding its listener.

The request budgets are deliberately asymmetric. A hydrate body is limited to 2 MiB. A returned sealed context is
limited to 6 MiB. Persist accepts a body up to 16 MiB, with at most 6 MiB of context and 4 MiB of decoded upstream
JSON or SSE. These component budgets leave room for the persist JSON envelope and ensure a context returned by
hydrate can be echoed back. A larger response is rejected with `413` and code `body_too_large`.

The workload token authenticates only the coordinator, not the end user or tenant. Deploy this listener only on an
encrypted, policy-restricted service network where every token holder is trusted with all stored response history.
Network policy is defense in depth, not tenant authorization. Per-response ownership remains tracked in
[#107](https://github.com/vllm-project/agentic-api/issues/107), and the endpoint must not be exposed across tenant
trust boundaries until that work lands.

A repeated persist for an already stored response ID returns `409` with code `response_already_stored`. Duplicate
detection happens atomically at insertion, so concurrent delivery has the same result. The service does not yet prove
that an identical retry matches the stored payload, so it does not return a prior success response.

## Remaining compatibility point

Request fields that `RequestPayload` does not model are dropped rather than forwarded, which narrows what reaches
vLLM compared with plain pass-through.
