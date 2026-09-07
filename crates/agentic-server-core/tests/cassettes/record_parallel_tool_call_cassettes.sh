#!/usr/bin/env bash
# Records parallel-tool-call behavior for gateway-owned built-in tools
# (web_search_preview, mcp) mixed with client-owned function tools, per
# https://github.com/vllm-project/agentic-api/issues/181.
#
# Four files, each streaming + non-streaming, organized by theme. Each file's
# tool declarations live in a matching tools-*.json (request-shape, never
# changes at runtime):
#
#   parallel-builtin-only  (tools-builtin-only.json: web_search_preview + mcp)
#     Turn 1: two parallel web_search_preview calls, each batching two
#             different queries together (multi-query batching + parallel
#             dispatch, tested together).
#     Turn 2: two parallel calls to the same MCP tool, two different queries
#             (mirrors turn 1's batching test for the MCP tool type).
#     Turn 3: web_search_preview + a remote MCP tool call, mixed built-in types.
#
#   parallel-mixed  (tools-mixed.json: web_search_preview + get_weather + set_temperature_unit)
#     Success path only. Turn 1: get_weather + set_temperature_unit +
#     web_search, 3-way parallel. Turn 2: follow-up depending on
#     set_temperature_unit's effect.
#
#   parallel-client  (tools-client-only.json: get_weather + get_stock_price + set_temperature_unit)
#     Success path, pure client-owned tools -- both function and custom.
#     Turn 1: get_weather(Tokyo) + get_stock_price(AAPL) + set_temperature_unit
#             (custom, freeform), 3-way parallel -- request.
#     Turn 2: all three resolved successfully -- follow-up combining results.
#     Turn 3: get_weather(Tokyo) + get_weather(Paris) -- same tool called
#             twice in parallel with different arguments -- request.
#     Turn 4: both resolved successfully -- follow-up combining results.
#
#   parallel-failures  (tools-mixed.json, reused: web_search_preview + get_weather)
#     Exactly two failure modes -- an explicit error message, and a fully
#     omitted output -- plus a mixed (built-in + client) variant of the
#     omission. Each is its own independent two-turn conversation, recorded
#     as a separate leg appended into the same file (via record_cassette.py's
#     --append), because an omission turn may legitimately end in a provider
#     error (as OpenAI does), and a turn that errors cannot be recovered from
#     to continue the same conversation further -- so no leg can assume an
#     earlier one's conversation state survived.
#       Leg 1 (turns 1-2): get_weather(London) + get_weather(Tokyo), parallel
#         -- London resolved as an explicit error, Tokyo succeeds -- tests
#         that a failing call doesn't affect its concurrently-resolved
#         sibling.
#       Leg 2 (turns 3-4): get_weather(Tokyo) + get_weather(Atlantis),
#         parallel -- Tokyo resolved, Atlantis's output omitted entirely --
#         tests whether the provider rejects the turn or just complains about
#         the specific missing call_id.
#       Leg 3 (turns 5-6): web_search_preview + get_weather(Atlantis), mixed
#         parallel (web_search is already resolved server-side) --
#         Atlantis's output omitted entirely -- mixed built-in-resolved +
#         client-left-dangling omission.
#
# Fake client-tool outputs are computed by real Python functions in
# tool_outputs.py (loaded via record_cassette.py's --tool-outputs), not
# static JSON -- see that file for the sentinel argument values ("London"
# always produces an explicit error output; "Atlantis" is always omitted)
# used consistently across every turn above.
#
# Every request explicitly sends parallel_tool_calls=true so the recorded
# cassettes reflect the value the gateway would send once it stops forcing
# parallel_tool_calls=false upstream (see request_response.rs).
#
# Usage from the repository root:
#
#   # OpenAI reference only (default)
#   OPENAI_API_KEY=sk-... \
#     bash crates/agentic-server-core/tests/cassettes/record_parallel_tool_call_cassettes.sh
#
#   # Gateway only, against a locally running agentic-server + vLLM
#   PARALLEL_RECORD_SET=gateway GATEWAY_URL=http://localhost:9000 GATEWAY_MODEL=Qwen/Qwen3.5-35B-A3B-FP8 \
#     bash crates/agentic-server-core/tests/cassettes/record_parallel_tool_call_cassettes.sh
#
#   # Both
#   PARALLEL_RECORD_SET=all OPENAI_API_KEY=sk-... \
#     bash crates/agentic-server-core/tests/cassettes/record_parallel_tool_call_cassettes.sh

set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPTS_DIR/parallel_tool_calls"
TOOL_OUTPUTS="$BASE_DIR/tool_outputs.py"
PARALLEL_RECORD_SET="${PARALLEL_RECORD_SET:-openai}"
OPENAI_MODEL_NAME="${OPENAI_MODEL:-gpt-5.6}"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:9000}"
GATEWAY_MODEL="${GATEWAY_MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"

green()  { printf '\033[32m%s\033[0m\n' "$*"; }
bold()   { printf '\033[1m%s\033[0m\n'  "$*"; }
red()    { printf '\033[31m%s\033[0m\n' "$*"; }

case "$PARALLEL_RECORD_SET" in
  openai|gateway|all) ;;
  *)
    echo "ERROR: PARALLEL_RECORD_SET must be openai, gateway, or all" >&2
    exit 1
    ;;
esac

if [[ "$PARALLEL_RECORD_SET" == "openai" || "$PARALLEL_RECORD_SET" == "all" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY must be set for PARALLEL_RECORD_SET=$PARALLEL_RECORD_SET" >&2
    exit 1
  fi
fi


BUILTIN_TURN1_PROMPT='Do two separate web_search calls in parallel, in this single turn -- do not wait for one to finish before starting the other. The first web_search call must batch two exact queries together: "potato nutrition facts" and "tomato nutrition facts". The second web_search call must batch two different exact queries together: "cucumber nutrition facts" and "carrot nutrition facts". Issue both calls now, then summarize each of the four results in one sentence.'
BUILTIN_TURN2_PROMPT='Call gitmcp_tiktoken__search_tiktoken_documentation in parallel for two separate exact queries -- do not wait for one to finish before starting the other: (1) {"query":"encoding"} and (2) {"query":"tokenizer"}. Issue both calls now in this single turn, then summarize each result in one sentence.'
BUILTIN_TURN3_PROMPT='Do both of these right now, in parallel, in this single turn -- do not wait for one to finish before starting the other: (1) search the web for the exact query "latest vLLM release notes", and (2) call gitmcp_tiktoken__search_tiktoken_documentation with {"query":"encoding"}. Do not call any other tool.'

MIXED_TURN1_PROMPT='Do all three of these right now, in parallel, in this single turn -- do not wait for one before starting another: (1) call get_weather for "Tokyo", (2) call set_temperature_unit to set my preferred unit to "fahrenheit" for the rest of this conversation, and (3) search the web for the exact query "Tokyo weather today".'
MIXED_TURN2_PROMPT="Now report Tokyo's current temperature using the unit I just told you to prefer."

CLIENT_TURN1_PROMPT='Do all three of these right now, in parallel, in this single turn -- do not wait for one before starting another: (1) call get_weather for "Tokyo", (2) call get_stock_price for "AAPL", and (3) call set_temperature_unit to set my preferred unit to "fahrenheit" for the rest of this conversation.'
CLIENT_TURN2_PROMPT="What did you find for the weather and the stock price, and what temperature unit did you just set?"
CLIENT_TURN3_PROMPT='Do both of these right now, in parallel, in this single turn -- do not wait for one before starting another: (1) call get_weather for "Tokyo", and (2) call get_weather for "Paris".'
CLIENT_TURN4_PROMPT="What's the weather in each of those two cities?"

FAILURES_LEG1_TURN1_PROMPT='Do both of these right now, in parallel, in this single turn -- do not wait for one before starting another: (1) call get_weather for "London", and (2) call get_weather for "Tokyo".'
FAILURES_LEG1_TURN2_PROMPT="What did you find for each of those two cities?"
FAILURES_LEG2_TURN1_PROMPT='Do both of these right now, in parallel, in this single turn -- do not wait for one before starting another: (1) call get_weather for "Tokyo", and (2) call get_weather for "Atlantis".'
FAILURES_LEG2_TURN2_PROMPT="What's the weather in Tokyo?"
FAILURES_LEG3_TURN1_PROMPT='Do both of these right now, in parallel, in this single turn -- do not wait for one before starting another: (1) search the web for the exact query "Atlantis weather today", and (2) call get_weather for "Atlantis".'
FAILURES_LEG3_TURN2_PROMPT="Never mind the weather lookup -- just tell me one fun fact about Atlantis instead."


# Records an N-turn conversation. `prompts` is a single newline-separated
# string (one line per turn). A turn that legitimately produces a provider
# error (e.g. an omission turn) is still valuable recorded data, so this
# never aborts the script -- record_cassette.py's own exit status is just
# logged, and the proxy has already written every turn -- including an error
# response -- to the output file before any client-side exception could
# propagate. Returns 1 only if literally nothing was recorded.
record_case() {
  local endpoint_flag="$1" endpoint="$2" model="$3" turns="$4"
  local prompts="$5" tools_file="$6" stream_flag="$7" output="$8"
  local temporary_output
  temporary_output="$(mktemp "$BASE_DIR/.parallel-cassette.XXXXXX")"

  printf '%s\n' "$prompts" \
    | python "$SCRIPTS_DIR/record_cassette.py" \
        --mode responses \
        --turns "$turns" \
        "$stream_flag" \
        --model "$model" \
        "$endpoint_flag" "$endpoint" \
        --tools "$tools_file" \
        --tool-choice auto \
        --parallel-tool-calls true \
        --tool-outputs "$TOOL_OUTPUTS" \
        --max-output-tokens 2048 \
        --output "$temporary_output"

  if [[ ! -s "$temporary_output" ]]; then
    rm -f -- "$temporary_output"
    red "✗ no response was recorded at all for $output"
    return 1
  fi

  mv -- "$temporary_output" "$output"
  green "✓ recorded -> $output"
  return 0
}

# Records parallel-failures' three independent legs, each a fresh two-turn
# conversation appended into the same file with --append (see the header
# comment above for why they can't be one continuous conversation).
record_failures_case() {
  local endpoint_flag="$1" endpoint="$2" model="$3" tools_file="$4" stream_flag="$5" output="$6"
  local temporary_output
  temporary_output="$(mktemp -u "$BASE_DIR/.parallel-cassette.XXXXXX")"

  local leg
  for leg in \
    "$FAILURES_LEG1_TURN1_PROMPT|$FAILURES_LEG1_TURN2_PROMPT" \
    "$FAILURES_LEG2_TURN1_PROMPT|$FAILURES_LEG2_TURN2_PROMPT" \
    "$FAILURES_LEG3_TURN1_PROMPT|$FAILURES_LEG3_TURN2_PROMPT"
  do
    local turn1="${leg%%|*}" turn2="${leg##*|}"
    local append_flag=()
    [[ -e "$temporary_output" ]] && append_flag=(--append)
    printf '%s\n%s\n' "$turn1" "$turn2" \
      | python "$SCRIPTS_DIR/record_cassette.py" \
          --mode responses \
          --turns 2 \
          "$stream_flag" \
          --model "$model" \
          "$endpoint_flag" "$endpoint" \
          --tools "$tools_file" \
          --tool-choice auto \
          --parallel-tool-calls true \
          --tool-outputs "$TOOL_OUTPUTS" \
          --max-output-tokens 2048 \
          --output "$temporary_output" \
          "${append_flag[@]}"
  done

  if [[ ! -s "$temporary_output" ]]; then
    rm -f -- "$temporary_output"
    red "✗ no response was recorded at all for $output"
    return 1
  fi

  mv -- "$temporary_output" "$output"
  green "✓ recorded -> $output"
  return 0
}

record_provider_suite() {
  local provider_label="$1" endpoint_flag="$2" endpoint="$3" model="$4" output_suffix="$5"
  local model_slug
  model_slug="$(echo "$model" | tr '/: ' '---')"

  bold "═══ $provider_label ($endpoint) — model: $model ═══"

  for stream_flag in --stream --no-stream; do
    local stream_label="streaming"
    [[ "$stream_flag" == "--no-stream" ]] && stream_label="nonstreaming"

    bold "parallel-builtin-only ($stream_label)"
    record_case "$endpoint_flag" "$endpoint" "$model" 3 \
      "$(printf '%s\n%s\n%s' "$BUILTIN_TURN1_PROMPT" "$BUILTIN_TURN2_PROMPT" "$BUILTIN_TURN3_PROMPT")" \
      "$BASE_DIR/tools-builtin-only.json" "$stream_flag" \
      "$BASE_DIR/parallel-builtin-only-${output_suffix}-${model_slug}-${stream_label}.yaml"

    bold "parallel-mixed ($stream_label)"
    record_case "$endpoint_flag" "$endpoint" "$model" 2 \
      "$(printf '%s\n%s' "$MIXED_TURN1_PROMPT" "$MIXED_TURN2_PROMPT")" \
      "$BASE_DIR/tools-mixed.json" "$stream_flag" \
      "$BASE_DIR/parallel-mixed-${output_suffix}-${model_slug}-${stream_label}.yaml"

    bold "parallel-client ($stream_label)"
    record_case "$endpoint_flag" "$endpoint" "$model" 4 \
      "$(printf '%s\n%s\n%s\n%s' "$CLIENT_TURN1_PROMPT" "$CLIENT_TURN2_PROMPT" "$CLIENT_TURN3_PROMPT" "$CLIENT_TURN4_PROMPT")" \
      "$BASE_DIR/tools-client-only.json" "$stream_flag" \
      "$BASE_DIR/parallel-client-${output_suffix}-${model_slug}-${stream_label}.yaml"

    bold "parallel-failures ($stream_label)"
    record_failures_case "$endpoint_flag" "$endpoint" "$model" \
      "$BASE_DIR/tools-mixed.json" "$stream_flag" \
      "$BASE_DIR/parallel-failures-${output_suffix}-${model_slug}-${stream_label}.yaml"
  done
}

mkdir -p "$BASE_DIR"

if [[ "$PARALLEL_RECORD_SET" == "openai" || "$PARALLEL_RECORD_SET" == "all" ]]; then
  record_provider_suite "OpenAI" --openai "https://api.openai.com" "$OPENAI_MODEL_NAME" "openai-reference"
fi

if [[ "$PARALLEL_RECORD_SET" == "gateway" || "$PARALLEL_RECORD_SET" == "all" ]]; then
  record_provider_suite "Gateway" --gateway "$GATEWAY_URL" "$GATEWAY_MODEL" "gateway"
fi

echo
green "════════════════════════════════════════════════════════════════"
green "Parallel-tool-call cassettes recorded -> $BASE_DIR"
green "════════════════════════════════════════════════════════════════"
