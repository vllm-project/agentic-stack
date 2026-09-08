# Testing Claude Code and Codex through the `agentic` CLI

This guide is the shareable checklist for exercising a coding harness (Claude Code or Codex) against Agentic API,
both with a local gateway started by the CLI and against a gateway already running on Kubernetes. It was written
while verifying [issue #190](https://github.com/vllm-project/agentic-api/issues/190) / [PR
#197](https://github.com/vllm-project/agentic-api/pull/197) and records the exact commands and expected output.

## Prerequisites

- A running OpenAI-compatible upstream. The examples use vLLM serving Qwen on `http://127.0.0.1:8000`:

  ```console
  vllm serve Qwen/Qwen3.8-27B-FP8 \
    --served-model-name Qwen/Qwen3.8-27B-FP8 ...
  ```

  Check what is served with:

  ```console
  curl -s http://127.0.0.1:8000/v1/models | python3 -c 'import sys,json; print([m["id"] for m in json.load(sys.stdin)["data"]])'
  ```

- The harness binary on `PATH`: `claude --version` or `codex --version`. Override discovery with `AGENTIC_CLAUDE_BIN`
  or `AGENTIC_CODEX_BIN`.

- Both Agentic API binaries built from this repository:

  ```console
  cargo build -p agentic-server --bins
  ```

CI pins Claude Code 2.1.245 and Codex 0.149.1 and runs both real CLIs through the attach commands against recorded
Qwen/vLLM streams. The Claude job verifies a gateway-owned web-search round trip; the Codex job verifies a completed
Responses answer and that a PNG attached with `codex exec --image` reaches the gateway byte for byte. Run the same
checks locally with `bash scripts/claude-code-smoke.sh` and `bash scripts/codex-smoke.sh` after building both binaries
with `cargo build -p agentic-server --bins`.

Neither job runs a model. To check that a real vision model renders what the gateway delivered, follow
[Verifying image support against a live vision model](vision-model-verification.md).

## CLI behavior worth knowing

| Behavior | Detail |
|---|---|
| `--upstream` must be a full URL | `http://` or `https://` with a host. A typo such as `http//127.0.0.1:8000` is rejected at parse time with `invalid upstream URL`. |
| `--model` is optional with `--upstream` | When omitted, the CLI calls `GET {upstream}/v1/models` and uses the first model listed. Pass `--model` to pick another when the upstream serves several. |
| Claude uses isolated settings and state | Per-run settings map Claude Code's canonical `claude-sonnet-4-5-20250929` identifier to the exact served model ID, while every default and small/fast model tier is pinned to that served model. Session history is isolated from the user's normal Claude home under `$AGENTIC_API_HOME/harnesses/claude` (default `~/.agentic-api/harnesses/claude`) so `--resume` and `--continue` work across invocations. Inherited Vertex, Bedrock, and Foundry routing switches are removed. |
| Claude effort is pinned to `medium` | Claude Code defaults to `high`, which Qwen's vLLM chat template rejects (`ValueError`). The CLI always passes `--effort medium` and sets `CLAUDE_CODE_EFFORT_LEVEL=medium` (the env var wins inside Claude Code). Override both with `AGENTIC_CLAUDE_EFFORT=low|medium|xhigh`. |
| Claude resource limits are pinned | The generated environment sets a 32,768-token context, 2,048 output tokens, and disables extended thinking. These conservative defaults fit the tested Qwen deployment. |
| Codex reads its model and image support from the gateway | Before writing an isolated Codex home, both `run codex` and `harness codex` fetch `GET {gateway}/v1/models?client_version=<ver>` and take the model and its `input_modalities` from that one response, so the isolated catalog always matches what the gateway serves. The client version comes from `codex --version` (honoring `AGENTIC_CODEX_BIN`); set `AGENTIC_CODEX_CLIENT_VERSION` to skip that probe. A gateway that is unreachable, rejects the request, or does not serve the selected model fails the launch with an actionable error instead of writing text-only metadata. Against an OIDC-protected gateway, pass `--api-key`. Configure image support with `[models."<id>"] input_modalities = ["text", "image"]`; see [Codex integration](../design/codex-integration.md). |
| `--yolo` | Adds `--dangerously-skip-permissions` (Claude) or `--dangerously-bypass-approvals-and-sandbox` (Codex). Use only in an externally isolated environment. |
| `--skip-llm-ready-check` | Skips the upstream `/health` probe. Avoid it while testing: the probe is what surfaces an unreachable upstream before the harness starts. |
| Arguments after `--` | Forwarded to the harness (`-p`, `--resume`, `exec`, ...). Claude's `--model`, `--settings`, `--setting-sources`, and `--bare` are rejected because they would bypass the generated model and provider isolation. Generated settings are temporary, but Claude session history persists in the isolated Agentic API home. |

## 1. Validate the configuration

```console
./target/debug/agentic validate --upstream http://127.0.0.1:8000 --harness claude
```

Expected: `Agentic API configuration looks valid.` This checks the gateway port is free, the database URL is usable,
the harness binary resolves, and the upstream URL is well formed.

## 2. Non-interactive smoke test

Claude Code:

```console
./target/debug/agentic run claude --upstream http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B-FP8 -- -p "Reply with exactly one word: pong"
```

Codex:

```console
./target/debug/agentic run codex --upstream http://127.0.0.1:8000 -- exec "Reply with exactly one word: pong"
```

Expected lifecycle output, then the harness answer and a clean exit:

```text
Starting Claude via http://127.0.0.1:3000
... agentic_server::server: LLM ready: http://127.0.0.1:8000
... agentic_server::server: gateway listening on 127.0.0.1:3000
Claude Code config: /tmp/agentic-api-session-... (gateway: http://127.0.0.1:3000, model: Qwen/Qwen3.8-27B-FP8)
pong
```

Claude Code's header may still display the canonical Sonnet label. Requests are routed to the Qwen model by the
generated `modelOverrides` entry; the display label is not evidence that Anthropic billing or an enterprise provider
is in use.

## 3. Tool-call round trip

Tool calls are where the parallel-tool-call handling from #190 is exercised, so run at least one. Start a normal
session and approve the permission prompt when the harness asks to run the command:

```console
./target/debug/agentic run claude --upstream http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B-FP8
> Run 'ls crates' with the Bash tool and list the directory names.
```

Expected: the harness runs the command and answers with `agentic-praxis`, `agentic-server`, `agentic-server-core`.
The gateway log must not contain `invalid tool config`.

Permission prompts are the default. Only for an unattended run in an externally isolated environment (CI, a
throwaway container) add `--yolo`, which forwards the harness's native bypass flag:

```console
./target/debug/agentic run claude --upstream http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B-FP8 --yolo \
  -- -p "Run 'ls crates' with the Bash tool and list the directory names."
```

## 4. Interactive session

```console
./target/debug/agentic run claude --upstream http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B-FP8
```

Inside the session, useful checks are a plain question (no tools), a file read, a multi-step edit, and `/model`
(should show the discovered or passed model). `Ctrl-C` stops the harness and the gateway together; confirm nothing
is left behind with `pgrep -fa agentic-server`.

## 5. Reproduce and verify #190 directly

The bug was a gateway-side rejection of `parallel_tool_calls: true` whenever a built-in tool was declared. Codex
always sends `true`, so every Codex session with a built-in tool failed:

```console
curl -s -w '\nHTTP %{http_code}\n' -H 'content-type: application/json' http://127.0.0.1:3000/v1/responses -d '{
  "model": "Qwen/Qwen3.8-27B-FP8",
  "input": "Reply with the single word: pong",
  "max_output_tokens": 64,
  "parallel_tool_calls": true,
  "tools": [{"type": "web_search_preview"}]
}'
```

| Gateway build | Result |
|---|---|
| Before PR #197 | `HTTP 400` — `invalid tool config: parallel_tool_calls must be false when using built-in tools` |
| PR #197 through pre-#181 | `HTTP 200`, `"status": "completed"`; the gateway forwarded `parallel_tool_calls: false` upstream regardless of the request, serializing tool calls |
| #181 or later | `HTTP 200`, `"status": "completed"`; the gateway forwards `parallel_tool_calls: true` upstream as requested, then executes emitted gateway-owned calls through its configured concurrency window and per-handler same-tool safety policy |

Also run the mixed shape (a `function` tool plus a built-in such as `code_interpreter`) with `parallel_tool_calls:
true`; it must return `HTTP 200` as well. Unit coverage lives in
`crates/agentic-server-core/src/types/request_response.rs` (`to_upstream_request_*parallel_tool_calls*`).

## 6. Against a Kubernetes deployment

Use the attach-only `agentic harness` command to test a cluster-hosted gateway without starting another local
gateway. The kind-based development cluster below is the one from
[Deploy agentic-api on Kubernetes](../deploying/kubernetes.md).

### Roll out a new image

```console
docker build -t agentic-api:kind .
kind load docker-image agentic-api:kind --name agentic-api
kubectl --namespace agentic-api rollout restart deploy/agentic-api
kubectl --namespace agentic-api rollout status deploy/agentic-api --timeout=180s
```

The Kubernetes guide deploys into the `agentic-api` namespace, so every `kubectl` command below passes
`--namespace agentic-api`; without it kubectl reports `deployments.apps "agentic-api" not found`.

### Port-forward and run the same checks

```console
kubectl --namespace agentic-api port-forward svc/agentic-api 9000:9000 &

# #190 repro (expect HTTP 200)
curl -s -w '\nHTTP %{http_code}\n' -H 'content-type: application/json' http://127.0.0.1:9000/v1/responses -d '{
  "model": "Qwen/Qwen3.8-27B-FP8", "input": "Reply with the single word: pong", "max_output_tokens": 64,
  "parallel_tool_calls": true, "tools": [{"type": "web_search_preview"}]
}'

# Claude Code through the cluster gateway (interactive)
agentic harness claude \
  --gateway-url http://127.0.0.1:9000 \
  --model Qwen/Qwen3.8-27B-FP8

# The same path as a one-shot smoke test
agentic harness claude \
  --gateway-url http://127.0.0.1:9000 \
  --model Qwen/Qwen3.8-27B-FP8 \
  -- -p "Reply with exactly one word: pong"
```

The Claude command waits for `/health` and `/ready` without following redirects, creates owner-only temporary settings,
removes inherited cloud-provider routing variables, and deletes the temporary settings when Claude exits. Claude's session
history persists under `$AGENTIC_API_HOME/harnesses/claude`, separate from the user's normal Claude Code configuration.
It advertises the compact `Bash,Edit,Read,WebSearch` tool set. There is intentionally no `--web-search` switch:
whether Claude Code's `WebSearch` function is gateway-owned is a deployment policy controlled by
`MESSAGES_GATEWAY_TOOL_ALIASES=WebSearch=web_search`.

For a raw-command diagnosis, the critical pieces are a persistent isolated `CLAUDE_CONFIG_DIR`, a temporary `settings.json` whose
`modelOverrides` maps the full canonical ID `claude-sonnet-4-5-20250929` to the served model, removal of inherited
`CLAUDE_CODE_USE_VERTEX`/`CLAUDE_CODE_USE_BEDROCK`/`CLAUDE_CODE_USE_FOUNDRY`, and passing the canonical ID to
`claude --model`. Short aliases such as `claude-sonnet-4-5` are not sufficient for this override.

```console
CLAUDE_SETTINGS_HOME=$(mktemp -d)
CLAUDE_STATE_HOME="${AGENTIC_API_HOME:-$HOME/.agentic-api}/harnesses/claude"
mkdir -p -m 700 "$CLAUDE_STATE_HOME"
trap 'rm -rf -- "$CLAUDE_SETTINGS_HOME"' EXIT
jq --null-input --arg model 'Qwen/Qwen3.8-27B-FP8' \
  '{modelOverrides: {"claude-sonnet-4-5-20250929": $model}}' \
  >"$CLAUDE_SETTINGS_HOME/settings.json"

env -u CLAUDE_CODE_USE_VERTEX \
  -u CLAUDE_CODE_USE_BEDROCK \
  -u CLAUDE_CODE_USE_FOUNDRY \
  -u CLAUDE_CODE_USE_ANTHROPIC_AWS \
  -u CLAUDE_CODE_USE_MANTLE \
  -u CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST \
  -u ANTHROPIC_VERTEX_PROJECT_ID \
  -u CLOUD_ML_REGION \
  CLAUDE_CONFIG_DIR="$CLAUDE_STATE_HOME" \
  ANTHROPIC_BASE_URL=http://127.0.0.1:9000 \
  ANTHROPIC_MODEL=Qwen/Qwen3.8-27B-FP8 \
  ANTHROPIC_SMALL_FAST_MODEL=Qwen/Qwen3.8-27B-FP8 \
  ANTHROPIC_DEFAULT_OPUS_MODEL=Qwen/Qwen3.8-27B-FP8 \
  ANTHROPIC_DEFAULT_SONNET_MODEL=Qwen/Qwen3.8-27B-FP8 \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=Qwen/Qwen3.8-27B-FP8 \
  ANTHROPIC_API_KEY=agentic-api-local \
  ANTHROPIC_AUTH_TOKEN=agentic-api-local \
  CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768 \
  CLAUDE_CODE_MAX_OUTPUT_TOKENS=2048 \
  MAX_THINKING_TOKENS=0 \
  CLAUDE_CODE_EFFORT_LEVEL=medium \
  claude --model claude-sonnet-4-5-20250929 \
    --tools Bash,Edit,Read,WebSearch --setting-sources user --effort medium \
    --settings "$CLAUDE_SETTINGS_HOME/settings.json"
```

Codex uses the same attach-only workflow. The CLI generates an isolated `CODEX_HOME`, Responses provider
configuration, and model catalog, then removes them when Codex exits:

```console
# Interactive Codex session through the cluster gateway
agentic harness codex \
  --gateway-url http://127.0.0.1:9000 \
  --model Qwen/Qwen3.8-27B-FP8

# One-shot smoke test
agentic harness codex \
  --gateway-url http://127.0.0.1:9000 \
  --model Qwen/Qwen3.8-27B-FP8 \
  -- exec --skip-git-repo-check "Reply with exactly one word: pong"
```

Codex uses the gateway's WebSocket transport (`supports_websockets = true`), so this also verifies `/v1/responses`
over WebSocket through the port-forward. On Linux hosts where Codex's sandbox (`codex-linux-sandbox`/bwrap) cannot
start, shell tool calls fail inside Codex itself; that is unrelated to the gateway and `--dangerously-bypass-approvals-and-sandbox`
(or the CLI's `--yolo`) confirms the gateway side. Codex declares web search structurally through the Responses API,
so it does not require `MESSAGES_GATEWAY_TOOL_ALIASES`.

Replace `agentic-api-local` with a real key when the deployment enforces inbound authentication. For `agentic harness`,
pass it explicitly with `--api-key` or set `AGENTIC_GATEWAY_API_KEY`; ambient `OPENAI_API_KEY` and
`ANTHROPIC_CUSTOM_HEADERS` are deliberately ignored because they commonly carry unrelated provider credentials. Finish by
confirming the gateway logged no errors during the run:

```console
kubectl --namespace agentic-api logs deploy/agentic-api --all-pods --since=10m | grep -c ' ERROR '   # expect 0
```

Match the log level rather than the word `error`: Codex closes its WebSocket without a closing handshake when
`exec` finishes, which the gateway logs as a `WARN ... WebSocket protocol error: Connection reset without closing
handshake` line per run. That line is expected; anything at `ERROR` level is not.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `invalid upstream URL` at startup | Malformed `--upstream` (missing `://`, no scheme, no host) | Pass a full `http://host:port` URL. |
| `HTTP 502` on every request | Gateway cannot reach the upstream, typically a wrong URL combined with `--skip-llm-ready-check` | Drop `--skip-llm-ready-check` so the readiness probe fails fast, then fix the URL. |
| `There's an issue with the selected model` | Model name does not match what the upstream serves | Omit `--model` to auto-discover, or copy the id from `/v1/models` exactly. |
| Claude opens with `Google Vertex AI`, Bedrock, or Foundry | Provider-selection variables leaked in from the user's normal Claude configuration | Use `agentic harness claude`; it creates an isolated config and removes the provider switches from the child environment. |
| Claude reports `There's an issue with the selected model (claude-sonnet-4-5)` | A short alias was overridden, but Claude Code requested the full canonical model ID | Map `claude-sonnet-4-5-20250929` to the served model; the CLI does this automatically. |
| Claude answers that it cannot browse even though `WebSearch` is listed | The deployment did not opt the PascalCase client function into gateway execution, or the model did not emit a tool call | Set `MESSAGES_GATEWAY_TOOL_ALIASES=WebSearch=web_search`, restart the Deployment, and verify the gateway logs contain an actual tool call. A model's prose claim is not a tool-call verification. |
| Template `ValueError` mentioning effort | Claude Code sent `high` | Do not override `AGENTIC_CLAUDE_EFFORT` with `high`; valid values for Qwen are `low`, `medium`, `xhigh`. |
| `HTTP 503` from a cluster gateway | `/ready` failing because the gateway cannot reach its upstream or database | `kubectl logs deploy/agentic-api` and look for `gateway dependencies not ready`. |
| `readiness.ready=false` warnings every minute or two while the upstream is healthy | Gateway build predates the readiness-client pooling fix ([#199](https://github.com/vllm-project/agentic-api/pull/199)): a pooled keep-alive connection that the upstream closed fails with `hyper::Error(IncompleteMessage)` | Rebuild from a tree that includes #199 and redeploy; add `agentic_server::handler::http::models=debug` to `RUST_LOG` to see the probe error. |
| Pods in `CrashLoopBackOff` with `failed to create temporary configuration file: Read-only file system` | Gateway build predates the read-only-home fix ([#199](https://github.com/vllm-project/agentic-api/pull/199)), and the base mounts a read-only root filesystem | Rebuild from a tree that includes #199 and redeploy; that base also mounts an `emptyDir` at `/var/lib/agentic-api`. Until then, mount a writable volume at `/var/lib/agentic-api` in your overlay. |
| `kubectl apply -k` fails with `cycle detected` | Overlay directory placed inside `deploy/kubernetes` | Move the overlay to a sibling directory such as `deploy/overlays/<env>` and reference `../../kubernetes` (a working kind example ships with #199). |
| Codex `exec` prints `Reading additional input from stdin...` and hangs | stdin is not a terminal | Append `</dev/null`. |
| Codex warns it could not create PATH aliases | Attach mode uses a temporary `CODEX_HOME` | The warning is non-fatal for an attach session. Codex still loads the generated provider and model catalog; use a persistent home under `$HOME` only for a manual configuration. |
| A long stream stops without a terminal event during a rollout | Bounded drain: 5 s `preStop` plus up to 8 s of in-flight draining, then the pod exits | Expected for responses longer than the drain window; clients should reconnect and continue with `previous_response_id`. |
| `parallel_tool_calls must be false when using built-in tools` | Gateway predates PR #197 | Rebuild and redeploy the image. |
