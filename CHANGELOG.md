# Changelog

All notable changes to Agentic API are documented here.

## [Unreleased]

### Added

- Verified image preservation through the Responses gateway end to end (#253): integration coverage for mixed
  text/image ordering, multiple images per turn, client-executed `view_image` tool output, `previous_response_id`
  continuation, `conversation_id` rehydration, stateless `store: false` proxying, and compaction of retained
  image-bearing user messages, over both the HTTP and WebSocket transports.
- Extended `scripts/codex-smoke.sh` so the pinned Codex CLI attaches a generated 1x1 PNG with `--image` and the
  capture assertion requires that exact payload by SHA-256, proving the image reaches the gateway unmodified. The
  replay server now serves a `/v1/models` listing advertising image support (#253).
- Added [Verifying image support against a live vision model](docs/guides/vision-model-verification.md), the opt-in
  manual guide for checking a real vision model (`Qwen/Qwen3-VL-8B-Instruct`) behind the gateway (#253).
- Added typed per-model input-modality overrides to `config.toml`
  (`[models."<served-model-id>"] input_modalities = ["text", "image"]`), validated at startup:
  unknown modality names, empty lists, duplicates, and image-only lists are rejected with the
  offending file and line (#252).

### Changed

- Modeled `input_file` as a first-class message content part, mirroring the media parts already modeled for
  client-executed tool output, so a file part keeps its structure instead of being treated as unknown (#253).
- Forwarded `parallel_tool_calls` as the model-generation preference for typed
  Responses requests, including built-in-only and mixed tool declarations (#181).
- Added bounded, configurable parallel execution for Responses gateway rounds,
  preserving model call order and applying per-handler same-tool safety.
- Preserved MCP list-tools records in continuation history for registry lifecycle
  decisions while excluding them from model input, preventing repeated public
  list-tools emission on later turns.
- Clarified Codex tool execution roles by replacing ambiguous ownership language
  with the preferred client-executed and gateway-executed terminology.
- Modeled the Codex model catalog and the upstream model listing as typed Rust structs instead of
  untyped JSON, and reported an undecodable upstream `/v1/models` payload as `502` rather than
  serving it as an empty catalog (#252).
- `agentic run codex` and `agentic harness codex` now resolve the model and its input modalities
  from a single gateway catalog snapshot before writing an isolated Codex home, retrying a warming
  gateway and failing with an actionable error when the catalog cannot be fetched or does not list
  the selected model. A gateway behind OIDC now requires `--api-key` for `agentic harness codex`.
  `agentic_harness::prepare_codex_home` requires the resolved modalities and is no longer public
  (#252).

### Fixed

- Dropped unrepresentable message content parts from model-bound input instead of forwarding a synthetic
  `{"type": "unknown"}` part that no client sent. The surrounding message and the order of the remaining parts are
  preserved, so a mixed text-and-image message is no longer mutated in transit (#253).
- Restored `scripts/codex-smoke.sh`, which could not launch after #252 made the Codex launcher require the gateway
  model catalog: the replay server served no `/v1/models`, so the launcher failed with
  `the gateway model catalog ... returned HTTP 404 Not Found` (#253).
- Resolved Codex image capabilities consistently: the HTTP model catalog and both launcher modes
  now advertise the same resolved `input_modalities`, so a vision-capable model no longer has image
  content stripped client-side because an isolated catalog hardcoded `["text"]`. Existing persistent
  Codex session homes must be regenerated to pick this up (#252).
- Rejected split-execution responses with missing, reused, or unstable tool call IDs before persistence, keeping the
  reserved response ID available for a corrected retry.
- Hardened split execution with atomic duplicate persistence, strict relayed-response validation, independent secret
  validation, bounded hydrate and persist payloads, stable error envelopes, and graceful shutdown error propagation.
- Forwarded Responses `text` generation settings through typed execution paths while preserving provider-specific
  text formats on stateless proxy requests and JSON Schema property order.
- Replaced `WebSearchActionSearch::new` and `WebSearchCall::new` with fallible
  `try_new(...)` constructors; callers now handle `WebSearchActionError` for
  empty query lists instead of risking a panic.

### Added

- Documented running Agentic API in front of NVIDIA Dynamo and recorded Dynamo cassettes for stateful and
  function-call flows.

### Testing

- Added Dynamo upstream replay tests, a generic cassette validator (`scripts/validate-cassettes.py`), and a dedicated
  CI job for them.

## [0.5.0] - 2026-08-25

### Changed

- Preserved Claude Code Messages transport fidelity across the gateway.
- Updated You.com web search integration to use GET query parameters.
- Aligned deployment and harness documentation with the 0.4.0 release.

### Testing

- Fixed web search test hangs in CI.

## [0.4.0] - 2026-08-23

### Added

- Added the Agentic API harness CLI for running Codex and Claude Code against Agentic API.
- Added home-based configuration and typed tool settings for standalone deployments.
- Added support for Codex CLI remote compaction V2.
- Added Kubernetes deployment guidance and architecture documentation.

### Changed

- Improved handling of Codex and Claude harness upstream configuration and compatible reasoning effort values.
- Preserved unsupported parallel tool calls through serialized upstream requests.
- Hardened MCP configuration and startup behavior.
- Improved Kubernetes health and readiness behavior for read-only container roots.

### Testing

- Added native Codex and Claude harness coverage and expanded compatibility tests.

## [0.3.0]

Initial documented release.
