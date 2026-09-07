# AGENTS.md

Instructions for AI coding agents working on this repository.

## Project Overview

This repository is Rust-first under the `vllm-project` GitHub organization.

- **Rust** -- primary and active implementation language at the repo root.
- **Docs** -- MkDocs documentation in `docs/`.
- **Python** -- a lightweight distribution and launcher for the packaged Rust gateway; the gateway implementation
  remains Rust-first.

## Terminology

- Read and follow [`TERMINOLOGY.md`](TERMINOLOGY.md) when naming API, state, tool, streaming, MCP, or reasoning
  concepts in code, documentation, issues, and pull requests.
- Treat `TERMINOLOGY.md` as normative for preferred prose. Preserve exact field names and item types when discussing
  a wire protocol.
- When local wording conflicts with current OpenAI documentation, prefer the OpenAI term unless this repository has a
  distinct implementation concept documented in `TERMINOLOGY.md`.

## Project Structure

```
.
├── TERMINOLOGY.md              # Normative project vocabulary
├── crates/agentic-server/       # Axum binary, transport handlers, and configuration
├── crates/agentic-server-core/  # Protocol types, execution, tools, and persistence
├── crates/agentic-praxis/       # Praxis integration
├── python/agentic_api/          # Python distribution, diagnostics, and launcher
├── tests/python/                # Python package and CLI tests
├── pyproject.toml               # Python wheel build metadata
├── Cargo.toml                    # Workspace manifest and shared dependencies/lints
└── docs/                         # Documentation (MkDocs)
```

## Setup

Install pre-commit hooks and build the project:

```bash
pre-commit install
cargo build
```

## Testing

```bash
cargo test

# Python distribution and CLI tests
uv run --python 3.12 --with maturin==1.14.1 --with pytest==9.1.1 python -m pytest tests/python

# Source-install CLI E2E (also exercises the packaged wheel build)
python3 scripts/tests/agentic-cli-e2e-test.py
```

- Before adding or updating replay cassettes, read `crates/agentic-server-core/tests/cassettes/README.md` and use its
  recorder workflow and existing scenario scripts; do not hand-author captured request/response YAML.

## Linting and Formatting

```bash
cargo clippy --all-targets -- -D warnings   # lint
cargo fmt                                     # format
cargo fmt -- --check                          # check formatting only
```

To run all pre-commit hooks manually:

```bash
pre-commit run --all-files
```

## Documentation

Install docs dependencies and run docs locally:

```bash
uv venv
uv pip install -r docs/requirements.txt
uv run mkdocs serve
```

## Code Style

- Rust edition: 2024.
- Maximum line length: 120 characters (configured in `rustfmt.toml`).
- `unsafe` code is forbidden (`unsafe_code = "forbid"` in `Cargo.toml`).
- Clippy `all` lints are denied; `pedantic` lints are warnings.
- Minimum supported Rust version (MSRV): 1.85.

### The agentic-server and core design architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full crate breakdown, request
lifecycle, module-by-module walkthrough, and contribution guide.

- `types/` owns wire/domain data; `events/` parses and normalizes upstream events; `tool/` owns tool discovery,
  routing, and execution; `executor/` orchestrates requests across inference, tools, and persistence; `storage/` owns
  database models and operations; `utils/` contains genuinely shared, domain-neutral helpers.
- Respect this dependency direction: handlers call core APIs; executor coordinates `events`, `tool`, and `storage`;
  those modules share contracts through `types`. Do not introduce transport concerns into core types or business logic.

## Rust Best Practices

- Do not use loose/untyped JSON signatures (`serde_json::Value` and similar) at a public API boundary. Request,
  response, and tool payloads must be modeled as proper types in `agentic-server-core/types` and serialized/
  deserialized through them. Functions and APIs introduced in this repo must be typed Rust; untyped JSON is not a
  substitute for a real type at a public boundary.
- Prefer borrowing (`&T`, `&str`, `&[T]`) and avoid `.clone()` unless ownership or lifetime requirements make it
  necessary. Move values when ownership is transferred; use `Arc` only for genuinely shared thread-safe state, and
  keep required clones explicit and close to task spawn.
- Return `Result` for recoverable failures and propagate with `?`. Use typed `thiserror` errors in library/core code,
  preserve sources during conversion, add useful boundary context, and avoid `unwrap`/`expect` in production paths
  except for documented, impossible invariants.
- Never hold a `Mutex`/`RwLock` guard across `.await`. Use Tokio async I/O, `spawn_blocking` for blocking or CPU-heavy
  work, bounded channels for backpressure, and `try_join!` for independent fallible work. Spawned tasks must have clear
  cancellation, shutdown, and join/error handling.
- Encode invariants with enums, newtypes, `Option`, and validated constructors. Prefer exhaustive matches and safe
  conversions (`From`/`TryFrom`) over stringly typed state, unchecked casts, or panics.
- Avoid speculative optimization: minimize allocations in hot paths with borrowing, slices, `Bytes`, and known
  capacities, then validate non-obvious optimizations with measurements. `unsafe` remains forbidden.

## Commits

- Always sign off commits with the `-s` flag (`git commit -s`).
- Use conventional commit prefixes:
  - `feat:` -- new feature
  - `fix:` -- bug fix
  - `ci:` -- CI/CD changes
  - `chore:` -- maintenance tasks (deps, config)
  - `docs:` -- documentation only

## Pull Requests

- Target the `main` branch.
- Include two sections in the PR description:
  - **Summary** -- what the PR does and why.
  - **Test Plan** -- how the changes were verified.
- Ensure all pre-commit hooks pass before opening the PR.
