---
name: executor-architecture-review
description: Use when planning or reviewing agentic-api changes that affect executor streaming, upstream SSE handling, output-item lifecycle, stream delivery, backpressure, or worker placement.
---

# Executor Architecture Review

Perform a read-only review that protects the executor's streaming ownership boundaries. Do not edit files, create or
update issues, post comments, commit, or push.

Use `skills/pr-review/SKILL.md` instead for a broad pull-request review that does not involve these boundaries.

## Authoritative inputs

1. Read `AGENTS.md` and `TERMINOLOGY.md` completely.
2. Read `ARCHITECTURE.md`, especially "Target streaming pipeline and ownership boundaries" and the affected module
   sections. Treat it as the source of truth; this skill does not replace it.
3. Resolve the exact task, diff, or pull request and inspect the complete changed path with relevant surrounding code.

## Review workflow

1. Map each changed responsibility to exactly one owner: inference transport, event normalization, synchronous
   ingestion, stream relay, or engine orchestration. Flag logic that has no owner or appears in multiple stages.
2. Trace one representative streaming event from upstream bytes through client delivery. Check that each stage
   consumes the previous stage's typed result instead of re-parsing or reconstructing its state.
3. Check output-item lifecycle invariants: validated `output_index`, stable item ID and kind, typed active state,
   detectable index reuse/duplicate completion, exhaustive kind handling, and consuming terminal finalization.
4. Check concurrency boundaries. Every channel needs entry and memory bounds, full/disconnect behavior, cancellation
   propagation, and join/error handling. Treat `spawn_blocking`, worker placement, and proposed capacities as claims
   requiring representative measurements, not defaults.
5. Match tests to the changed owner. Consider malformed and out-of-order semantic events, ID/index mismatches,
   duplicate completion, incomplete terminal state, tiny capacities, oversized items, slow consumers, cancellation,
   and task failure when applicable. Do not demand unrelated cases.

## Findings

Report only actionable correctness, regression, performance, or missing-test issues. Verify each finding against the
actual path and use this form:

```text
[P1] path/to/file.rs:123 — short title
Boundary: <owner and violated responsibility>
Evidence: <specific path or scenario>
Impact: <observable failure>
Fix: <concise direction that restores one owner and one path>
```

Order findings from `P0` to `P3`. If none remain, say so explicitly. End with a short ownership map, tests or
benchmarks inspected or run, and any environment limitations.
