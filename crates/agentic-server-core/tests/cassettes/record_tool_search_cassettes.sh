#!/usr/bin/env bash
# Records the checked-in client tool-search characterization against OpenAI or
# the gateway.
#
# Usage from the repository root:
#   TOOL_SEARCH_RECORD_SET=openai-reference OPENAI_API_KEY=sk-... \
#     bash crates/agentic-server-core/tests/cassettes/record_tool_search_cassettes.sh
#   TOOL_SEARCH_RECORD_SET=gateway-nonstreaming GATEWAY_URL=http://localhost:9000 \
#     bash crates/agentic-server-core/tests/cassettes/record_tool_search_cassettes.sh
#   TOOL_SEARCH_RECORD_SET=gateway-streaming GATEWAY_URL=http://localhost:9000 \
#     bash crates/agentic-server-core/tests/cassettes/record_tool_search_cassettes.sh
#   TOOL_SEARCH_RECORD_SET=gateway-websocket GATEWAY_URL=http://localhost:9000 \
#     bash crates/agentic-server-core/tests/cassettes/record_tool_search_cassettes.sh

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPTS_DIR/tool_search"
RETURNED_TOOLS="$BASE_DIR/returned_tools.json"
FUNCTION_OUTPUTS="$BASE_DIR/function_outputs.json"
PROMPTS="$BASE_DIR/prompts.txt"
OPENAI_TOOLS="$BASE_DIR/openai_tools.json"
OPENAI_TOOL_CHOICES="$BASE_DIR/openai_tool_choice_sequence.json"
GATEWAY_TOOL_CHOICES="$BASE_DIR/gateway_tool_choice_sequence.json"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.6}"
GATEWAY_MODEL="${GATEWAY_MODEL:-${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}}"
GATEWAY_URL="${GATEWAY_URL:-}"
TOOL_SEARCH_RECORD_SET="${TOOL_SEARCH_RECORD_SET:-all}"

model_slug() {
  printf '%s' "$1" | tr '/: ' '---'
}

record_scenario() {
  local endpoint_flag="$1"
  local endpoint="$2"
  local model="$3"
  local tools="$4"
  local projection="$5"
  local tool_choice_sequence="$6"
  local filename="$7"
  local recorder_args=("${@:8}")
  local temporary_output
  local continuation_args=()

  temporary_output="$(mktemp "$STAGING_DIR/.tool-search-cassette.XXXXXX")"
  if [[ "$projection" == "gateway-public" ]]; then
    continuation_args=(--no-store --manual-item-replay)
  fi
  if ! python "$SCRIPTS_DIR/record_cassette.py" \
    --mode responses \
    --turns 4 \
    "${recorder_args[@]}" \
    --model "$model" \
    "$endpoint_flag" "$endpoint" \
    --tools "$tools" \
    --tool-choice-sequence "$tool_choice_sequence" \
    --tool-outputs "$FUNCTION_OUTPUTS" \
    --tool-search-output-tools "$RETURNED_TOOLS" \
    "${continuation_args[@]}" \
    --max-output-tokens 4096 \
    --output "$temporary_output" < "$PROMPTS"
  then
    rm -f -- "$temporary_output"
    return 1
  fi

  mv -- "$temporary_output" "$STAGING_DIR/$filename"
  RECORDED_FILES+=("$filename")
  printf 'staged %s\n' "$filename"
}

record_provider() {
  local provider="$1"
  local endpoint_flag="$2"
  local endpoint="$3"
  local model="$4"
  local tools="$5"
  local projection="$6"
  local tool_choice_sequence="$7"
  local prefix="$8"
  local slug

  slug="$(model_slug "$model")"
  printf 'Recording %s blocking tool-search characterization\n' "$provider"
  record_scenario \
    "$endpoint_flag" "$endpoint" "$model" "$tools" "$projection" \
    "$tool_choice_sequence" \
    "${prefix}-${slug}-nonstreaming.yaml" --no-stream
  printf 'Recording %s streaming tool-search characterization\n' "$provider"
  record_scenario \
    "$endpoint_flag" "$endpoint" "$model" "$tools" "$projection" \
    "$tool_choice_sequence" \
    "${prefix}-${slug}-streaming.yaml" --stream
}

case "$TOOL_SEARCH_RECORD_SET" in
  openai-reference|openai|gateway-nonstreaming|gateway-streaming|gateway-websocket|gateway|all) ;;
  *)
    printf 'ERROR: TOOL_SEARCH_RECORD_SET must be openai-reference, gateway-nonstreaming, gateway-streaming, gateway-websocket, gateway, or all\n' >&2
    exit 1
    ;;
esac

for required_file in \
  "$RETURNED_TOOLS" \
  "$FUNCTION_OUTPUTS" \
  "$PROMPTS" \
  "$OPENAI_TOOLS" \
  "$OPENAI_TOOL_CHOICES" \
  "$GATEWAY_TOOL_CHOICES"
do
  if [[ ! -f "$required_file" ]]; then
    printf 'ERROR: required fixture does not exist: %s\n' "$required_file" >&2
    exit 1
  fi
done

if [[ "$TOOL_SEARCH_RECORD_SET" =~ ^(openai-reference|openai|all)$ ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  printf 'ERROR: OPENAI_API_KEY is required for %s\n' "$TOOL_SEARCH_RECORD_SET" >&2
  exit 1
fi

if [[ "$TOOL_SEARCH_RECORD_SET" =~ ^(gateway-nonstreaming|gateway-streaming|gateway-websocket|gateway|all)$ ]] && [[ -z "$GATEWAY_URL" ]]; then
  printf 'ERROR: GATEWAY_URL is required for %s\n' "$TOOL_SEARCH_RECORD_SET" >&2
  exit 1
fi

STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/agentic-tool-search-cassettes.XXXXXX")"
trap 'rm -rf -- "$STAGING_DIR"' EXIT
RECORDED_FILES=()
cp -a -- "$BASE_DIR/." "$STAGING_DIR/"

if [[ "$TOOL_SEARCH_RECORD_SET" =~ ^(openai-reference|openai|all)$ ]]; then
  record_provider \
    OpenAI --openai https://api.openai.com "$OPENAI_MODEL" \
    "$OPENAI_TOOLS" public-stored "$OPENAI_TOOL_CHOICES" tool-search-openai-reference
fi

if [[ "$TOOL_SEARCH_RECORD_SET" =~ ^(gateway-nonstreaming|gateway|all)$ ]]; then
  slug="$(model_slug "$GATEWAY_MODEL")"
  printf 'Recording gateway blocking tool-search flow\n'
  record_scenario \
    --gateway "$GATEWAY_URL" "$GATEWAY_MODEL" "$OPENAI_TOOLS" gateway-public \
    "$GATEWAY_TOOL_CHOICES" \
    "tool-search-gateway-${slug}-nonstreaming.yaml" --no-stream
fi

if [[ "$TOOL_SEARCH_RECORD_SET" =~ ^(gateway-streaming|gateway|all)$ ]]; then
  slug="$(model_slug "$GATEWAY_MODEL")"
  printf 'Recording gateway HTTP/SSE tool-search flow\n'
  record_scenario \
    --gateway "$GATEWAY_URL" "$GATEWAY_MODEL" "$OPENAI_TOOLS" public-stored \
    "$GATEWAY_TOOL_CHOICES" \
    "tool-search-gateway-${slug}-streaming.yaml" --stream
fi

if [[ "$TOOL_SEARCH_RECORD_SET" =~ ^(gateway-websocket|gateway|all)$ ]]; then
  slug="$(model_slug "$GATEWAY_MODEL")"
  printf 'Recording gateway WebSocket tool-search flow\n'
  record_scenario \
    --gateway "$GATEWAY_URL" "$GATEWAY_MODEL" "$OPENAI_TOOLS" public-stored \
    "$GATEWAY_TOOL_CHOICES" \
    "tool-search-gateway-${slug}-websocket.yaml" --stream --transport websocket
fi

printf 'Validating checked-in five-profile public tool-search matrix\n'
TOOL_SEARCH_CASSETTE_DIR="$STAGING_DIR" cargo test --manifest-path "$SCRIPTS_DIR/../../../../Cargo.toml" \
  -p agentic-server-core --test tool_search_characterization_test

for filename in "${RECORDED_FILES[@]}"; do
  chmod 664 "$STAGING_DIR/$filename"
  mv -- "$STAGING_DIR/$filename" "$BASE_DIR/$filename"
  printf 'recorded %s\n' "$BASE_DIR/$filename"
done
