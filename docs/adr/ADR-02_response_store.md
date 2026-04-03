# ADR-02: Response Store semantics for `previous_response_id`

> **Status:** Proposal
> **Related discussion:** [Issue #14](https://github.com/vllm-project/agentic-stack/issues/14)

## Intention

Frame the design choices behind `previous_response_id` as a set of decision points for the MVP, so `agentic-stack` can support multi-turn continuation without requiring clients to replay full history on every request.

This document is intentionally closer to an RFC-shaped ADR than to a finalized decision record. The goal is to make the tradeoffs explicit, narrow the open questions, and leave room for implementation feedback before the direction is locked.

## Context

`previous_response_id` is only useful if `agentic-stack` retains enough prior state to reconstruct the next turn.

Without a server-side continuation model:

- clients must resend the full prior conversation
- request payloads grow with every turn
- client and server state are easier to drift out of sync
- multi-turn behavior becomes less ergonomic and harder to reason about

At a high level, the continuation model under discussion is:

> next turn context = prior context + prior output + new input

That makes a stored response more than a plain output artifact. It is a continuation checkpoint, or at least a pointer to whatever state must be loaded to rebuild one.

## Scope

This ADR is meant to resolve or narrow discussion around five questions:

1. should continuation be self-contained and single-hop, or chain-based across multiple stored objects?
2. should the persisted model start as single-table or multi-table?
3. should the backing store be relational-first or non-relational-first?
4. what data must be stored to rehydrate a response correctly?
5. when should response state be persisted at all?

This ADR does not attempt to settle:

- long-term memory or summarization
- analytics or reporting
- exact infrastructure wiring
- final retention policy details

## Working assumptions

Some points appear much less controversial than others and can be treated as working assumptions for discussion:

- `agentic-stack` needs a durable continuation layer for `previous_response_id`
- the stored state should preserve enough structure to support more than plain text outputs
- the design should stay inspectable and debuggable
- the project should align with OpenAI Responses API behavior where that behavior is clear and relevant

The remaining sections are structured as decision points rather than settled outcomes.

## Decision point 1: Continuation shape

### Question

Should `agentic-stack` model continuation as a self-contained checkpoint retrievable in one hop, or as a linked chain that must be walked and reassembled at request time?

### Option A: Self-contained checkpoint

Each stored response contains, or can load in one lookup, everything needed to continue the next turn.

This usually pairs naturally with:

- flattened prior input history
- prior model output
- turn-level configuration needed for inheritance
- a storage model that favors checkpoint retrieval over normalized reconstruction

### Option B: Linked chain

Each stored response contains only turn-local data, and `agentic-stack` reconstructs the next turn by walking prior links.

This usually pairs naturally with:

- turn-by-turn storage
- a more normalized history model
- more complex request-time reconstruction rules

### Tradeoffs

**Self-contained checkpoint**

Pros:

- simpler rehydration path
- clearer continuation semantics
- easier realtime lookup behavior
- easier to reason about partial reconstruction failures

Cons:

- more duplicated state across turns
- larger stored payloads
- weaker normalization

**Linked chain**

Pros:

- less duplicated data
- cleaner normalized history model
- more direct mapping to turn-by-turn conversation structure

Cons:

- more request-time orchestration
- harder failure semantics
- more complex history retrieval
- less obvious performance behavior for hot paths

### Rationale from issue discussion

The issue comments pushed on two related concerns:

- single-hop retrieval is easier to explain and may better support realtime history access
- normalized multi-entity designs may be more extensible in the long run

Those concerns are both valid. The main question is whether MVP should optimize first for correctness and simple rehydration, or for normalized modeling and future lifecycle separation.

Observed OpenAI-compatible behavior suggests that continuation may still succeed when a follow-up request provides previous_response_id and tool output without resending tools. That implies the effective tool context from the prior turn must be recoverable during rehydration. That is a useful signal that the effective tool context from the prior turn must be recoverable somewhere, which in turn strengthens the case for a richer checkpoint and simpler one-hop rehydration.


## Decision point 2: Storage model shape

### Question

If continuation is checkpoint-oriented, should the persisted model begin as a single-table store, or should it start with separate first-class entities such as `responses`, `prompts`, and `conversations`?

### Option A: Single-table checkpoint-first model

A single `response_store` table holds a small set of queryable metadata columns plus a flexible payload for the stored checkpoint body.

This tends to align with single-hop continuation because the persisted row is close to the rehydration unit.

### Option B: Multi-table normalized model

Separate entities capture different concerns:

- `responses`
- `prompts`
- `conversations`

This tends to align with a more normalized conceptual model and clearer separation of input-side and output-side state.

### Tradeoffs

**Single-table**

Pros:

- simpler MVP schema
- simpler read path
- fewer joins or equivalent cross-entity lookups
- easier payload evolution early on

Cons:

- mixes concerns into one persisted object
- may become awkward if prompts or conversations later need their own lifecycle
- can feel denormalized

**Multi-table**

Pros:

- clearer conceptual separation
- better lifecycle management if conversations or prompts become first-class later
- stronger normalization

Cons:

- more coordination and orchestration
- more schema and migration work up front
- more opportunities for ambiguity over which object is canonical for continuation

### Rationale from issue discussion

The comments on [Issue #14](https://github.com/vllm-project/agentic-stack/issues/14) explicitly surfaced this tension:

- single-table is attractive because retrieval can stay close to O(1)-style single lookup behavior
- multi-table is attractive because it may extend more cleanly as the product grows

This is likely the most important architectural decision in the document because it shapes both the rehydration path and the future data model.

Another practical consideration is schema evolution. A single-table model with a small queryable envelope plus versioned payload can absorb early payload changes with less migration churn while continuation semantics are still settling.

## Decision point 3: Backend model

### Question

Should `agentic-stack` approach the response store as primarily relational, or should it optimize first for a non-relational backend?

### Option A: Relational-first

The logical model is expressed in relational terms, even if parts of the stored checkpoint remain flexible or JSON-shaped.

This usually pairs naturally with:

- queryable first-class metadata
- explicit tenant or project scoping
- clearer indexing choices
- easier future RBAC modeling

### Option B: Non-relational-first

The logical model is expressed around document or key-value storage first, with relational concerns treated as secondary.

This usually pairs naturally with:

- flexible payload storage
- simpler document persistence
- looser schema expectations

### Tradeoffs

**Relational-first**

Pros:

- stronger queryability for metadata
- clearer lifecycle management for expiry and scoping
- a cleaner fit for future RBAC and administrative controls
- easier to reason about first-class indexed fields

Cons:

- may feel heavier for an MVP
- can encourage schema design too early
- may be less natural if the checkpoint is mostly opaque payload

**Non-relational-first**

Pros:

- flexible checkpoint storage
- lower friction for evolving payload shape
- can feel closer to the stored object model

Cons:

- weaker structure for metadata-heavy queries
- less obvious fit for scope boundaries and RBAC
- can defer important data-model decisions rather than eliminate them

### Rationale from issue discussion

The issue comments raised this explicitly: non-relational storage is worth considering, but relational modeling may age better if the project needs stronger scope boundaries or RBAC later.

This is not just an infrastructure choice. It affects which fields are treated as first-class, what gets indexed, and how easy it is to grow beyond a pure checkpoint blob.

## Decision point 4: What to store

### Question

What data must `agentic-stack` preserve so that `previous_response_id` can rehydrate the next turn correctly?

### Working assumption

This point appears closer to consensus than the others.

A stored response likely needs, at minimum:

- a `schema_version` for the stored payload
- the flattened prior input history seen by the model
- the model output from the prior turn
- tool calls and tool results needed to preserve execution context
- the effective tool configuration from the prior turn, when relevant to continuation
- tool-choice information, subject to compatibility rules
- instructions or equivalent system/developer input, subject to compatibility rules
- basic metadata such as response ID, parent response ID, model, timestamps, and expiry

### Open compatibility note

OpenAI’s current Responses API docs indicate that:

- `store: false` disables storage
- the Responses API stores application state by default unless retention settings override that behavior
- when `previous_response_id` is used, prior `instructions` are not automatically carried forward

Observed OpenAI-compatible behavior from exploratory testing suggests that:

- omitted `tools` may still be recoverable across continuation when the prior turn established them
- an explicit `tool_choice` should not be assumed valid if the continuation request omits the corresponding `tools`
- prior `instructions` need to be provided on each new turn, so they should not be assumed to persist automatically even if they are stored.

That suggests two useful guardrails for this ADR:

1. the stored payload should be rich enough to reproduce continuation semantics without relying on client replay
2. inheritance rules should be documented carefully rather than assumed from internal implementation convenience
3. storage and inheritance should be treated as separate concerns: a field may need to be stored even if it should not automatically inherit

This is an inference from the OpenAI docs, not a claim that `agentic-stack` must exactly mirror every OpenAI implementation detail.

Sources:

- [OpenAI Responses API reference](https://platform.openai.com/docs/api-reference/responses/retrieve)
- [OpenAI data controls](https://platform.openai.com/docs/models/how-we-use-your-data)
- [OpenAI migrate to Responses guide](https://platform.openai.com/docs/guides/migrate-to-responses)

---

## Proposed Decisions

These reflect where things stand. Still open for discussion while we're in Draft.

| # | Decision | Status |
|---|----------|--------|
| D1 | Continuation shape | Proposed |
| D2 | Storage model shape | Proposed |
| D3 | Backend model | Proposed |
| D4 | What to store | Proposed |