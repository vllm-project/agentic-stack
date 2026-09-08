<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/White-Main-Logo.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/Black-Main-Logo.svg">
  <img alt="Agentic API" src="assets/Black-Main-Logo.svg" width="600">
</picture>

**The stateful, agentic API layer for [vLLM](https://github.com/vllm-project/vllm), written in Rust 🦀**

*Run OpenAI-grade agentic workloads (Responses API, server-side tools, Codex) on your own GPUs.*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.85%2B-orange.svg?logo=rust)](Cargo.toml)
[![CI](https://github.com/vllm-project/agentic-api/actions/workflows/rust.yml/badge.svg)](https://github.com/vllm-project/agentic-api/actions/workflows/rust.yml)
[![pre-commit](https://github.com/vllm-project/agentic-api/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/vllm-project/agentic-api/actions/workflows/pre-commit.yml)

</div>

______________________________________________________________________

## 🧠 Overview

vLLM gives you state-of-the-art inference throughput. But real agentic applications need more than raw tokens: they need **conversation state, tool-call loops, and multi-turn orchestration**. Today, all of that complexity lives in your client code.

**Agentic API moves it server-side.** It is a Rust-native gateway that sits in front of vLLM and owns the stateful agentic APIs, starting with an OpenAI-compatible [Responses API](https://platform.openai.com/docs/api-reference/responses). vLLM is one supported backend, not part of the Agentic API product name. Your application makes *one API call* and the server handles the rest: state hydration, tool execution, streaming, and continuation.

```mermaid
flowchart LR
    C(["🧑‍💻 Client<br/>Codex · SDKs · curl"]) -->|"📮 <code>POST /v1/responses</code><br/>🌐 HTTP&nbsp;&nbsp;📡 SSE&nbsp;&nbsp;🔌 WebSocket"| A
    subgraph A ["⚡ Agentic API (Rust 🦀)"]
        direction TB
        S["🔄 State hydration<br/><code>previous_response_id</code>"]
        T["🛠️ Server-side tools<br/>web search · functions"]
        P["💾 Persistence<br/>SQLite response store"]
    end
    A -->|"🚀 <code>POST /v1/responses</code><br/>⚙️ stateless&nbsp;&nbsp;🤝 OpenAI-compatible"| V(["🚀 vLLM core<br/>inference engine"])

    classDef client fill:#FFE8B3,stroke:#F59E0B,stroke-width:2px,color:#7C2D12
    classDef inner fill:#E0E7FF,stroke:#6366F1,stroke-width:2px,color:#312E81
    classDef engine fill:#DCFCE7,stroke:#22C55E,stroke-width:2px,color:#14532D

    class C client
    class S,T,P inner
    class V engine
    style A fill:#F5F3FF,stroke:#8B5CF6,stroke-width:2px,color:#5B21B6
    linkStyle 0 stroke:#F59E0B,stroke-width:2px
    linkStyle 1 stroke:#22C55E,stroke-width:2px
```

> [!TIP]
> Point [OpenAI Codex](https://github.com/openai/codex) at Agentic API and drive it entirely with open models served by vLLM. No OpenAI account required.

## ✨ Key Features

- 🔄 **Stateful conversations**: the server manages history via `previous_response_id`. No client-side message tracking, no replaying full transcripts.
- 🛠️ **Server-side tool execution**: an explicit tool-ownership model (gateway / client / provider) decides exactly what runs where. Web search ships today via [You.com](https://you.com), and the model executes multi-step tool chains automatically.
- 📡 **Every transport**: non-streaming HTTP, server-sent events for token streaming, and full **WebSocket** support for interactive clients.
- 🧰 **Codex-ready**: accepts Codex-shaped Responses traffic out of the box, preserving the tool declarations and response item shapes Codex depends on.
- 🏃 **Background execution**: fire-and-forget requests that keep processing server-side.
- ✅ **Compatibility tested**: validated against the [Open Responses](https://www.openresponses.org/) compatibility suite, with replay-cassette tests for real OpenAI and vLLM traffic.

## 🧭 API Surface

| Endpoint | Description | Status |
| --- | --- | --- |
| `POST /v1/responses` | OpenAI-compatible Responses API with state, tools, and streaming | ✅ |
| `GET /v1/responses` | WebSocket transport for the Responses API | ✅ |
| `POST /v1/conversations` | Conversation management | ✅ |
| `GET /v1/models` | Model listing proxied from vLLM | ✅ |
| `GET /health` · `GET /ready` | Liveness and readiness probes | ✅ |
| Messages API | Anthropic-style stateful messages on shared primitives | 🚧 Planned |
| Interactions API | Higher-level agentic workflow surface | ⏳ Planned |

## 🚀 Quickstart

### Agentic API CLI

Build the user-facing CLI and gateway binaries together:

```bash
cargo build -p agentic-server --bins
```

Launch Codex or Claude Code with an isolated Agentic API configuration:

```bash
./target/debug/agentic run codex --model Qwen/Qwen3-30B-A3B-FP8
./target/debug/agentic run claude --model Qwen/Qwen3-30B-A3B-FP8
```

To use an existing upstream, provide its `http://` or `https://` base URL. The harness model defaults to the first
model the upstream lists at `/v1/models`; pass `--model` to choose a different one:

```bash
./target/debug/agentic run codex --upstream http://127.0.0.1:5050
./target/debug/agentic run claude \
  --upstream http://127.0.0.1:5050 \
  --model Qwen/Qwen3-30B-A3B-FP8
```

SQLite is the default storage backend. Use PostgreSQL explicitly when the session is shared:

```bash
./target/debug/agentic run codex \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --database-url postgresql://user:password@localhost/agentic_api
```

Run preflight checks without launching a harness:

```bash
./target/debug/agentic validate \
  --upstream http://127.0.0.1:5050 \
  --model Qwen/Qwen3-30B-A3B-FP8 \
  --harness codex
```

Use `AGENTIC_CODEX_BIN` or `AGENTIC_CLAUDE_BIN` to override harness binary discovery. Add `--no-color` for scripts or
`--quiet` for minimal lifecycle output. Use `--yolo` only in an externally isolated environment; it skips Claude
permission checks and disables Codex approvals and sandboxing.

### Python distribution

The `agentic-api` wheel packages the Rust gateway and a small Python launcher. This release produces wheel artifacts
for 0.5.0 as a build-only release: download the wheel for your platform from the release workflow, then install that local file. It is
not published on PyPI yet.

```bash
WHEEL_PATH=/absolute/path/to/agentic_api-PLATFORM.whl
uv pip install "$WHEEL_PATH"
agentic-api serve --vllm-base-url http://existing-vllm:8000

uv pip install "agentic-api[local] @ file://$WHEEL_PATH"
agentic-api serve --model MODEL_ID
```

The base install is for remote mode and does not install vLLM. The `[local]` extra installs the pinned vLLM runtime so
the launcher can manage a local vLLM process on supported Linux hosts.

Use `agentic-api --version` for a quick install check and `agentic-api doctor --mode remote --json` when an agent or
script needs machine-readable diagnostics.

#### After PyPI publication

These public-index and `uvx` examples apply only after the PyPI publication gate for a future release:

```bash
uv pip install agentic-api
uv pip install "agentic-api[local]"
uvx --from agentic-api agentic-api doctor
uvx --from agentic-api agentic-api serve --vllm-base-url http://existing-vllm:8000
```

The Rust-native `agentic` CLI remains supported for `run codex`, `run claude`, `serve`, and `validate`. For the full
installation walkthrough, managed-vLLM passthrough examples, `doctor` output, and known-good model profiles, see
[Python installation and workflows](docs/guides/python-installation.md).

For Claude sessions, Agentic API always sets both `--effort medium` and `CLAUDE_CODE_EFFORT_LEVEL=medium`; the
environment variable is intentional because Claude Code gives it precedence over the command-line effort flag.
Qwen3.8-27B's vLLM chat template accepts `low`, `medium`, and `xhigh` reasoning effort values but not Claude Code's
default `high`. Override the pinned value with `AGENTIC_CLAUDE_EFFORT`. See the [Claude Code effort
configuration](https://code.claude.com/docs/en/model-config) and [vLLM reasoning output
documentation](https://docs.vllm.ai/en/latest/features/reasoning_outputs/) for the underlying behavior, and
[Harness CLI Testing](docs/guides/harness-cli-testing.md) for an end-to-end verification checklist.

**1. Serve a model with vLLM.** Any recipe from [recipes.vllm.ai](https://recipes.vllm.ai) works:

```bash
vllm serve Qwen/Qwen3-30B-A3B-FP8 \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --reasoning-parser qwen3 --port 5050
```

Serving through [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) instead of a standalone `vllm serve`? Point the
gateway at the Dynamo frontend the same way; see
[Running Agentic API in front of NVIDIA Dynamo](docs/guides/dynamo-upstream.md).

**2. Start Agentic API**, pointing it at the vLLM server (set the `YOU_*` variables to enable built-in web search):

```bash
YOU_API_KEY=<your-you.com-api-key> YOU_API_BASE_URL=<you.com-api-base-url> \
  cargo run -p agentic-server -- --llm-api-base http://0.0.0.0:5050
```

The default database is `~/.agentic-api/agentic_api.db`, so running an installed binary does not create state in the
current directory. Set `AGENTIC_API_HOME` to an absolute directory to move both the default database and user
configuration, or set `DATABASE_URL`/`--db-url` to select a different database.

**3. Make a stateful call:**

```bash
curl http://localhost:9000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-FP8",
    "input": "What is new in vLLM this month?",
    "tools": [{"type": "web_search"}]
  }'
```

Continue the conversation by passing the returned `id` as `previous_response_id`, and the server rehydrates everything for you.

## ⚙️ Agentic API home and configuration

On startup, Agentic API creates `~/.agentic-api` and loads `~/.agentic-api/config.toml` when that file exists. On the
first invocation with a resolved LLM base URL, a missing config file is generated from the effective
`--llm-api-base`/`LLM_API_BASE` and non-secret tool settings. It records the name of the web-search API-key environment
variable, never its value. The generated file is group-readable but not group-writable on Unix (so a container restart
under a different arbitrary UID sharing the same group can still read it) and is never overwritten on later
runs. CLI arguments and process environment variables take precedence over file settings. A standalone server can
therefore be started with just `agentic-server` after creating a config like this:

```toml
llm_api_base = "http://127.0.0.1:5050"
# database_url = "postgresql://agentic-api@localhost/agentic_api"

[web_search]
base_url = "https://api.ydc-index.io"
api_key_env = "YOU_API_KEY"

[mcp]
allowed_hosts = ["mcp.example.com"]

[server]
# Maximum serialized request size in bytes for HTTP bodies and WebSocket
# messages and frames. Must be greater than zero.
max_request_body_size_bytes = 10485760

[tools]
# Upper bound for gateway-owned calls within one Responses round and for
# provider requests inside a batched web-search call.
# Must be greater than zero.
max_concurrent_gateway_calls = 5

[mcp_servers.counter]
url = "https://mcp.example.com/mcp"
allowed_tools = ["tool_1_name", "tool_2_name"]
require_approval = "never"
```

`max_request_body_size_bytes` bounds the serialized request the gateway accepts on `/v1/responses`,
`/v1/responses/compact`, `/v1/conversations`, the Anthropic Messages endpoints, and the Responses WebSocket. It counts
encoded bytes — JSON overhead, replayed conversation history, and base64 image attachments included — and is unrelated
to the model's token context limit, so raising it does not raise what the upstream will accept. It defaults to
10 MiB (10485760); inline base64 attachments cost roughly a third more than the source image, so conversations that
replay several images may need a higher value. Oversized HTTP requests are answered with `413 Payload Too Large`;
oversized WebSocket messages and frames are rejected by the transport, which closes the connection before any JSON is
parsed. Order of precedence is `--max-request-body-size-bytes`, then `AGENTIC_MAX_REQUEST_BODY_SIZE_BYTES`, then this
file setting.

`api_key_env` names the process environment variable containing the web-search credential; it does not contain the
credential itself. `YOU_API_BASE_URL`, `AGENTIC_MCP_ALLOWED_HOSTS`, `AGENTIC_MAX_REQUEST_BODY_SIZE_BYTES`, and
`AGENTIC_MAX_CONCURRENT_GATEWAY_CALLS` can override their typed file settings. The concurrency value is a sliding-window
upper bound; handlers may further serialize calls to the same tool name. The MCP allowlist is used only for
request-declared remote MCP URLs; configured `[mcp_servers]` entries are trusted operator configuration.

With that file in place, inject only the secret when starting the server:

```bash
YOU_API_KEY="<your-you.com-api-key>" agentic-server
```

Restrict the file to the service account (for example, `chmod 600 ~/.agentic-api/config.toml`), especially if you add
credentialed `database_url`, MCP headers, or stdio MCP environment values. Prefer `DATABASE_URL`, referenced API-key
environment variables, and a deployment secret manager for secrets.

To use an operator-configured MCP server, declare its label without sending its connection details or secrets:

```json
{
  "type": "mcp",
  "server_label": "server_label"
}
```

Configured `allowed_tools` form the maximum tool set; request-provided `allowed_tools` may narrow it. A configured
`require_approval = "never"` lets requests omit that field. If a label exists in `config.toml`, a request cannot
override it with `server_url`; otherwise the existing request-declared HTTP MCP flow remains available.

## 🤖 Codex on your own GPUs

Agentic API speaks the Responses wire protocol Codex expects, including WebSockets, so you can run the full Codex experience against open models.

Add a provider to `~/.codex/config.toml`:

```toml
[model_providers.agentic-api]
name = "OpenAI"
base_url = "http://localhost:9000/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = true
```

Then launch Codex:

```bash
codex --disable image_generation -c model_provider=agentic-api -m Qwen/Qwen3-30B-A3B-FP8
```

If the gateway enables OIDC, configure Codex's supported command-backed bearer authentication instead of
`requires_openai_auth = false`:

```toml
[model_providers.agentic-api]
name = "OpenAI"
base_url = "http://localhost:9000/v1"
wire_api = "responses"
supports_websockets = true

[model_providers.agentic-api.auth]
command = "/absolute/path/to/print-oidc-token"
args = ["--audience", "agentic-api"]
refresh_interval_ms = 300000
```

The command must print only a current OIDC token to stdout. Codex refreshes it before expiry and sends it as the
provider bearer token. See the
[Codex custom-provider authentication reference](https://developers.openai.com/codex/config-advanced#custom-model-providers).
Keep the inference credential in the gateway's `OPENAI_API_KEY`; do not print that service credential from the token
command.
See [GitHub authentication with Dex](docs/deploying/github-oidc.md) for a complete GitHub login, token-helper, and
gateway setup.

## 🧑‍💻 Claude Code on your own GPUs

Agentic API serves the Anthropic Messages protocol at `/v1/messages`, so Claude Code (CLI or Agent SDK) runs against open models. Point it at the gateway:

```bash
export ANTHROPIC_BASE_URL="http://localhost:9000"
export ANTHROPIC_API_KEY="<your-key>"
export ANTHROPIC_MODEL="Qwen/Qwen3-30B-A3B-FP8"   # match the served model

claude -p "summarize the files in this directory"
```

With OIDC enabled, use Claude Code's bearer-token variable and leave its API-key variable unset so the identity token
is not also sent as an upstream `x-api-key`:

```bash
export ANTHROPIC_BASE_URL="http://localhost:9000"
export ANTHROPIC_AUTH_TOKEN="$(/absolute/path/to/print-oidc-token --audience agentic-api)"
unset ANTHROPIC_API_KEY

claude -p "summarize the files in this directory"
```

Refresh `ANTHROPIC_AUTH_TOKEN` before it expires. For supported dynamic credential helpers, see Anthropic's
[LLM gateway authentication guide](https://docs.anthropic.com/en/docs/claude-code/llm-gateway).
The same [GitHub authentication with Dex](docs/deploying/github-oidc.md) guide shows how to obtain the ID token
without embedding a client secret in Claude Code.

Claude Code's own tools (Bash, Edit, Read, …) stay **client-owned** — Claude Code runs them, as usual.

### Running Claude Code's web search on the gateway

Current Claude Code versions declare Anthropic's native `web_search_20250305` server tool. Agentic API translates that
declaration for the upstream model and executes the resulting search server-side against the configured search backend;
no MCP server or tool alias is required:

```bash
YOU_API_KEY=<you.com-key> YOU_API_BASE_URL=<you.com-base-url> \
  cargo run -p agentic-server -- --llm-api-base http://0.0.0.0:5050
```

The gateway supports the basic `web_search_20250305` contract, including `max_uses`, `allowed_domains`,
`blocked_domains`, and the country in `user_location`. Other versioned native web-search declarations are rejected rather
than forwarded in a shape the upstream cannot execute.

Older clients that declare a function tool named `WebSearch` can still opt in with
`MESSAGES_GATEWAY_TOOL_ALIASES="WebSearch=web_search"`. This variable maps a client tool name to a gateway executor
(`name=executor`, comma-separated) and remains empty by default. The gateway adapts the older `WebSearch` function's
`allowed_domains`/`blocked_domains` arguments to the executor's schema automatically.

> Note: allow and block domain lists are mutually exclusive, matching Anthropic's native tool contract.

## 🧩 Tool Ownership Model

Every tool call has exactly one execution path, so nothing runs by accident:

| Ownership | Who executes it | Examples |
| --- | --- | --- |
| **Gateway-owned** | Agentic API executes it server-side and continues the loop | Web search, file search, MCP-backed tools |
| **Client-owned** | Preserved and returned to the client | Codex shell / editor tools, your functions |
| **Provider-owned** | Passed through to vLLM or an upstream provider | Provider-native tools |

Unknown or ambiguous tool shapes are **never executed by default**. They are preserved and returned.

## 🏗️ Repository Layout

```
crates/
├── agentic-server/       # Axum binary, transport handlers (HTTP/SSE/WS), configuration
├── agentic-server-core/  # Protocol types, executor, tool framework, persistence
└── agentic-praxis/       # Praxis gateway integration
docs/                     # MkDocs documentation, ADRs, and design notes
```

## 🛠️ Developing

```bash
cargo build                                  # build
cargo test                                   # test
cargo clippy --all-targets -- -D warnings    # lint
cargo fmt -- --check                         # format check
```

Docs are built with MkDocs:

```bash
uv venv
uv pip install -r docs/requirements.txt
uv run mkdocs serve
```

Design and migration decisions are tracked as ADRs in [docs/adr/](docs/adr/), with deeper design notes in [docs/design/](docs/design/). See the full [ROADMAP](ROADMAP.md) for where the project is heading, and [CONTRIBUTING](CONTRIBUTING.md) to get involved.

## 🗺️ Roadmap at a Glance

- [x] **Responses API hydration**: stateful continuation with `previous_response_id`
- [x] **Codex support**: practical Codex sessions through the Responses API
- [x] **Server-side tool execution**: explicit ownership, web search built in
- [ ] **Messages API**: built on the same persistence and execution primitives
- [ ] **Interactions API**: durable, higher-level agentic workflows
- [ ] **Production hardening**: storage backends, observability, cached-prefix continuation

______________________________________________________________________

## 📄 License

Licensed under the [Apache License 2.0](LICENSE).

<div align="center">

**⭐ If Agentic API saves you from writing one more client-side tool loop, star the repo! ⭐**

</div>
