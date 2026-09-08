# Cassette Recorder

`record_cassette.py` runs an embedded proxy between the script and an upstream API (OpenAI, vLLM, or the agentic-api gateway). Every request and response is captured into a YAML cassette for use in replay tests.

## How it works

```
[record_cassette.py] -> [proxy :7070] -> [OpenAI | vLLM | gateway]
                         (cassette written here)
```

The proxy intercepts each turn, records the request body and response, then appends a `t<N>` entry to the output YAML.

The recorder is interactive. For each turn it prompts you to type the input message and waits for you to press Enter before sending the request. You can run it directly in your terminal and type the prompts by hand, or pipe them in from a script using `printf` or `echo` to feed all turns non-interactively:

```bash
# interactive -- type each prompt when asked
python tests/cassettes/record_cassette.py --mode responses --turns 2 --no-stream --vllm http://localhost:5050 --model Qwen/Qwen3-30B-A3B-FP8 --max-output-tokens 1024 --output out.yaml

# non-interactive -- pipe prompts in (one line per turn)
printf 'First prompt\nSecond prompt\n' | python tests/cassettes/record_cassette.py --mode responses --turns 2 --no-stream --vllm http://localhost:5050 --model Qwen/Qwen3-30B-A3B-FP8 --max-output-tokens 1024 --output out.yaml

# gateway-backed cassette -- records the gateway-facing request/response
printf 'Use web search to look up potato, then summarize in one sentence.\n' | python tests/cassettes/record_cassette.py --mode responses --turns 1 --no-stream --gateway http://localhost:9000 --model openai/gpt-oss-20b --output out.yaml

# structured single-turn input -- sends the JSON string or item array from input.json
python tests/cassettes/record_cassette.py --mode responses --turns 1 --no-stream --no-store --max-output-tokens 0 --input-file input.json --model gpt-4o --output out.yaml
```

The recorder scripts (`record_reasoning_cassettes.sh`, `record_tool_call_cassettes.sh`, etc.) use `printf` to feed fixed prompts per test so no manual input is needed.

## Coding-harness CLI acceptance tests

The literal fixture at `../fixtures/claude-code-cache-control-request.json` mirrors the cache-bearing parts of a
Claude Code Messages request: multi-block `system`, a structured user message, and both `WebSearch` and client-owned
tool declarations. It includes explicit `5m` and `1h` TTLs. The Messages HTTP and loop integration tests assert that
the fixture remains unchanged through transparent proxying and every streaming and non-streaming gateway-tool round.

The fixture is hand-checked test data, not a captured cassette. A dedicated GitHub Actions matrix runs the real,
pinned Claude Code and Codex CLIs through the `agentic harness` attach commands. Claude Code replays the streaming
vLLM Messages web-search cassette through a deterministic local search backend. Codex replays the streaming vLLM
Responses reasoning cassette. Neither job contacts Anthropic, OpenAI, or a live model.

To run the same end-to-end check locally, install the pinned dependencies, build the server, and run the harness:

```bash
npm install --global '@anthropic-ai/claude-code@2.1.245' '@openai/codex@0.149.1'
python -m pip install 'PyYAML==6.0.3'
cargo build -p agentic-server --bins
bash scripts/claude-code-smoke.sh
bash scripts/codex-smoke.sh
```

Each smoke script starts a replay server and Agentic API, then invokes the installed CLI through the corresponding
`agentic harness` command. The Claude job opts `WebSearch` into gateway execution with
`MESSAGES_GATEWAY_TOOL_ALIASES=WebSearch=web_search`; it asserts the recorded answer, two Messages rounds, one search
request, a hidden `tool_result`, cache-bearing system and user blocks, and the exact Qwen model requested by Claude
Code 2.1.245. The Codex job asserts the recorded `HELLO` answer, one streaming Responses request, the exact Qwen
model requested by Codex 0.149.1, and that the PNG attached with `codex exec --image` arrives as an inline
`input_image` data URL whose bytes hash to the digest of the generated file. The replay server serves a `/v1/models`
listing advertising `capabilities: ["image"]` so the gateway resolves text-and-image modalities; without it the Codex
launcher cannot resolve its catalog and Codex would strip the image before sending.

## Modes

| Mode | Description |
|------|-------------|
| `responses` | Chains turns via `previous_response_id`. Supported with `--vllm`. Common mode for gateway-backed built-in tool cassettes. |
| `messages` | Anthropic Messages API (`/v1/messages`). Stateless: resends the full `messages` history each turn. With `--tool-outputs`, a turn following a `tool_use` feeds back matching `tool_result` blocks (keyed by tool name) instead of prompting. Supported with `--vllm`. |
| `conv` | Creates a conversation object, passes `conversation` id each turn. |
| `isolation` | Two independent conversations (A and B) recorded into one cassette. |
| `mixed` | Turn 1 uses `conversation` id, turns 2+ switch to `previous_response_id`. |
| `store_true_then_store_false` | Turn 1: `store=true` with conversation id. Remaining turns: `store=false`, still pass conversation id. |

## CLI options

```
--turns N              Number of turns
--output PATH          Output YAML path
--mode MODE            responses | conv | isolation | mixed | store_true_then_store_false  (default: conv)
--stream / --no-stream Streaming or non-streaming (default: streaming)
--model NAME           Model name sent in requests
--no-store             Set store=false
--vllm URL             vLLM upstream, e.g. http://localhost:8000 (responses mode only)
--gateway URL          agentic-api gateway, e.g. http://localhost:9000
--openai URL           OpenAI upstream (default https://api.openai.com)
--tools FILE           JSON file containing a tools array (responses mode only)
--tool-choice VALUE    "auto", "none", "required", or JSON e.g. '{"type":"function","name":"foo"}'
--reasoning JSON       JSON object containing Responses reasoning settings
--input-file FILE       JSON string or item array for one HTTP Responses turn
--max-output-tokens N  max_output_tokens for Responses requests (default 1024; use 0 to omit)
--proxy-port PORT      Local proxy port (default 7070)
--branch-from TURN     Branch from this turn's response id (repeatable)
--branch-turn-number N First turn number for the corresponding branch (repeatable)
```

## Cassette YAML structure

Each cassette has a `turns` list. One entry is appended per request.

**Single turn (`--turns 1`, non-streaming):**

```yaml
turns:
- filename: t1
  request:
    method: POST
    path: /v1/responses
    body:
      model: Qwen/Qwen3-30B-A3B-FP8
      input: Reply with exactly one word: HELLO
      stream: false
      store: true
      max_output_tokens: 1024
    headers:
      content-type: application/json
    query_params: {}
  response:
    status_code: 200
    headers:
      content-type: application/json
    body:
      id: resp_abc123
      output: [...]
      usage: {...}
```

**Two turns (`--turns 2`, non-streaming) -- `t2` adds `previous_response_id`:**

```yaml
turns:
- filename: t1
  request:
    body:
      input: "Remember the word APPLE. Just say: OK"
      store: true
  response:
    body:
      id: resp_abc123

- filename: t2
  request:
    body:
      input: What word did I ask you to remember?
      previous_response_id: resp_abc123
  response:
    body:
      id: resp_def456
```

**Tool call turn -- `tool_choice` and `tools` appear in the request body:**

```yaml
turns:
- filename: t1
  request:
    body:
      input: What is the NVIDIA stock price?
      tool_choice: auto
      tools:
      - type: function
        name: get_stock_price
        description: ...
        parameters: {...}
  response:
    body:
      output:
      - type: function_call
        name: get_stock_price
        arguments: '{"ticker": "NVDA"}'
```

**Streaming turn -- `response.body` is replaced by `response.sse`, a list of raw SSE lines:**

```yaml
turns:
- filename: t1
  request:
    body:
      stream: true
  response:
    status_code: 200
    headers:
      content-type: text/event-stream; charset=utf-8
    sse:
    - "event: response.created\n"
    - "data: {...}\n"
    - "event: response.output_text.delta\n"
    - "data: {...}\n"
    - "event: response.completed\n"
    - "data: {...}\n"
```

## Recorder scripts

| Script | Cassettes | Backend |
|--------|-----------|---------|
| `record_text_only_cassettes.sh` | 10 text-only cassettes (responses + conv modes, streaming + non-streaming) | OpenAI (`OPENAI_API_KEY`) |
| `record_reasoning_cassettes.sh` | Matching explicit-reasoning cassettes (streaming + non-streaming) | gateway and OpenAI reference; optional direct vLLM |
| `record_tool_call_cassettes.sh` | 8 tool-call cassettes (4 tool_choice modes x streaming + non-streaming) | vLLM |
| `record_codex_cli_tool_call_cassettes.sh` | Codex function/namespace/custom-tool matrix | gateway, vLLM, and OpenAI |
| `record_custom_tool_cassettes.sh` | Matching two-turn custom-tool flows (streaming + non-streaming) | gateway and OpenAI reference |
| `record_mcp_cassettes.sh` | Native MCP counter tool discovery and calls (streaming + non-streaming) | gateway and OpenAI reference |
| `record_web_search_cassettes.sh` | Matching web-search calls (streaming + non-streaming) | gateway and OpenAI reference |
| `record_dynamo_cassettes.sh` | Stateful two-turn and client-executed function tool call cassettes (streaming + non-streaming) | NVIDIA Dynamo frontend |

### Text-only (OpenAI)

```bash
OPENAI_API_KEY=sk-... bash tests/cassettes/record_text_only_cassettes.sh
MODEL=gpt-4o-mini OPENAI_API_KEY=sk-... bash tests/cassettes/record_text_only_cassettes.sh
```

### Reasoning (gateway and OpenAI)

The default records the same explicit `reasoning` object against OpenAI and the
gateway for both response modes. The gateway fixture uses the same OpenAI model
as its reference so the comparison isolates gateway request and response
handling from model differences. Use `REASONING_RECORD_SET=gateway`,
`REASONING_RECORD_SET=openai`, or `REASONING_RECORD_SET=vllm` to record one
provider. The gateway recording requires a running gateway and reasoning-capable
upstream; the optional direct-vLLM set retains the legacy accumulator workflow.
Every selected recording is staged and validated before any final fixture is
replaced, so a failed provider or response cannot leave a partially refreshed
comparison set.

```bash
# Start the gateway against the same OpenAI ground-truth model in one terminal.
OPENAI_API_KEY=sk-... \
cargo run -p agentic-server -- \
  --llm-api-base https://api.openai.com \
  --skip-llm-ready-check

# Record the OpenAI-reference and gateway pairs from another terminal.
OPENAI_API_KEY=sk-... \
GATEWAY_URL=http://localhost:9000 \
MODEL=gpt-5.6 \
bash crates/agentic-server-core/tests/cassettes/record_reasoning_cassettes.sh

# To refresh only the gateway-facing pair instead:
REASONING_RECORD_SET=gateway \
GATEWAY_URL=http://localhost:9000 \
MODEL=gpt-5.6 \
bash crates/agentic-server-core/tests/cassettes/record_reasoning_cassettes.sh

vllm serve Qwen/Qwen3-30B-A3B-FP8 --reasoning-parser qwen3 --port 5050 > server.log 2>&1

REASONING_RECORD_SET=vllm \
VLLM_URL=http://0.0.0.0:5050 \
MODEL=Qwen/Qwen3-30B-A3B-FP8 \
bash crates/agentic-server-core/tests/cassettes/record_reasoning_cassettes.sh
```

### Tool calls (vLLM)

```bash
vllm serve Qwen/Qwen3-30B-A3B-FP8 --tool-call-parser hermes --enable-auto-tool-choice --port 5050 > server.log 2>&1

VLLM_URL=http://0.0.0.0:5050 MODEL=Qwen/Qwen3-30B-A3B-FP8 bash tests/cassettes/record_tool_call_cassettes.sh
```

### NVIDIA Dynamo (vLLM worker behind the Dynamo frontend)

Dynamo's `/v1/responses` rejects `previous_response_id` with `501`, so the recorder's own turn chaining cannot be
used. The script records turn 1 from a prompt, builds turn 2's input from turn 1's recorded assistant message (the
hydrated item history the gateway sends upstream), records it, and merges both into one cassette. See
[docs/guides/dynamo-upstream.md](../../../../docs/guides/dynamo-upstream.md) for the Dynamo launch commands.

```bash
DYNAMO_URL=http://127.0.0.1:8000 MODEL=openai/gpt-oss-20b bash tests/cassettes/record_dynamo_cassettes.sh
```

### Web search (gateway and OpenAI)

The default records both providers. Use `WEB_SEARCH_RECORD_SET=gateway` or
`WEB_SEARCH_RECORD_SET=openai` to record only one side.

```bash
OPENAI_API_KEY=sk-... \
bash crates/agentic-server-core/tests/cassettes/record_web_search_cassettes.sh
```

### Custom tool (gateway and OpenAI)

This records an unformatted freeform custom tool, including the
`custom_tool_call_output` continuation. Grammar-constrained custom tools are
covered separately by unit tests because the gateway intentionally rejects
formats that normalization cannot preserve.

```bash
OPENAI_API_KEY=sk-... \
bash crates/agentic-server-core/tests/cassettes/record_custom_tool_cassettes.sh
```

Use `CUSTOM_TOOL_RECORD_SET=gateway` or `CUSTOM_TOOL_RECORD_SET=openai` to
record only one provider.

### Codex custom tools (gateway, vLLM, and OpenAI)

The custom fixture uses a Lark grammar and records two turns: the model returns raw `custom_tool_call.input`, then the
recorder submits the matching `custom_tool_call_output` before the follow-up user message.

```bash
GATEWAY_URL=http://127.0.0.1:3018 \
V_MODEL=Qwen/Qwen3.6-35B-A3B \
bash tests/cassettes/record_codex_cli_tool_call_cassettes.sh gateway-custom

VLLM_URL=http://127.0.0.1:8000 \
V_MODEL=Qwen/Qwen3.6-35B-A3B \
bash tests/cassettes/record_codex_cli_tool_call_cassettes.sh direct-vllm-custom

OPENAI_API_KEY=sk-... \
OPENAI_CUSTOM_MODEL=gpt-5.6 \
bash tests/cassettes/record_codex_cli_tool_call_cassettes.sh openai-custom
```

### Compaction replay (OpenAI)

These recordings capture the non-streaming `/v1/responses` inference calls replayed by the compaction integration
tests. The JSON inputs contain the exact model-facing item arrays, including the context-checkpoint prompt. Use the
existing recorder directly from `crates/agentic-server-core`:

```bash
record_compaction() {
  input_name="$1"
  output_name="$2"
  uv run \
    --with click \
    --with fastapi \
    --with httpx \
    --with uvicorn \
    --with pyyaml \
    python tests/cassettes/record_cassette.py \
    --mode responses \
    --turns 1 \
    --no-stream \
    --no-store \
    --max-output-tokens 0 \
    --openai https://api.openai.com \
    --model gpt-4o \
    --input-file "tests/cassettes/compaction/inputs/${input_name}.json" \
    --output "tests/cassettes/compaction/compact-${output_name}-gpt-4o-nonstreaming.yaml"
}

export OPENAI_API_KEY=sk-...
record_compaction basic basic
record_compaction tool-prior-compaction tool-prior
record_compaction followup followup
```
