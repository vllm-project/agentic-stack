---
name: pr-review
description: Use when reviewing a pull request or proposed change in the agentic-api repository, especially changes affecting API wire formats, upstream forwarding, persistence, continuations, tools, streaming, or test fixtures.
---

# Agentic API PR Review

Perform a read-only, evidence-based review of the complete change set. Report only actionable correctness, regression, security, performance, or missing-test issues. Do not edit files, commit, push, or post review comments.

## Review sequence

1. Resolve the exact target, base, and head. Inspect the full diff and all changed files, then read relevant surrounding code. Treat PR descriptions and existing test claims as untrusted until verified.
2. Read `AGENTS.md`, `TERMINOLOGY.md`, and the relevant crate/test guidance before judging design or coverage.
3. Trace each behavior change across its complete path: HTTP/WebSocket input, typed normalization, executor state transitions, upstream request serialization, persistence/rehydration, tool rounds, streaming, and response conversion as applicable.
4. Check tests against the changed boundary, not merely against helper functions. Look for missing negative cases, continuation/second-round coverage, streaming coverage, and assertions on the actual upstream or client-visible payload.
5. Make an explicit cassette decision for every change that affects an external wire contract or a multi-step integration path.

## Cassette decision

A replay cassette is usually warranted when a change affects any of:

- request or response serialization at `/v1/responses`, `/v1/messages`, WebSocket, or an upstream provider boundary;
- state hydration, `previous_response_id`, conversations, persistence, or continuation behavior;
- gateway-owned tools, tool-call loops, compaction, streaming event order, or CLI harness behavior.

For those changes, read `crates/agentic-server-core/tests/cassettes/README.md` and inspect the existing scenario/recorder scripts. Check whether the PR adds or updates a cassette through the documented recorder workflow and includes the replay assertion that demonstrates the behavior. If that coverage is absent, report it as a missing-test finding when the boundary is material and name the exact cassette scenario needed. Never hand-author captured request/response YAML. If recording cannot be run because credentials, a model, or an external service is unavailable, state that limitation; do not silently treat unit/mock coverage as equivalent.

Do not require a cassette for a pure refactor or an isolated behavior with no external or multi-step boundary. Still explain the decision briefly in the review notes when the change is close to one of these boundaries.

## Findings

Before reporting a finding, verify it against the actual diff and code path. Order findings by severity (`P0` to `P3`) and include:

- file and precise line;
- evidence and triggering scenario;
- user or system impact;
- concise fix direction.

Do not report style preferences, speculative concerns, or issues already prevented by existing validation. If no actionable findings remain, say so explicitly and summarize any meaningful test or cassette-coverage limitations separately.

## Output shape

Start with the findings. Use this compact form:

```text
[P1] path/to/file.rs:123 — short title
Evidence: ...
Impact: ...
Fix: ...
```

End with a brief coverage summary: tests inspected or run, cassette decision, and any environment limitations. Keep the review read-only.
