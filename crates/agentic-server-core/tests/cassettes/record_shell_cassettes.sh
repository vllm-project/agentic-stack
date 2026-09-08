#!/usr/bin/env bash
# Records two-turn local-shell conversations against the gateway and OpenAI.
# Turn 1 requests a shell_call; turn 2 submits simulated shell_call_output and
# a follow-up user message, then captures the model's interpretation.
# Covers success, nonzero exit, timeout, and multiple commands in both modes.
# Model-generated commands are never executed. See shell/scenarios.py.
# Usage: SHELL_RECORD_SET=gateway GATEWAY_URL=http://localhost:9000 MODEL=... bash "$0"

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SHELL_OUTPUT_DIR:-$SCRIPTS_DIR/shell}"
TOOLS_FILE="$SCRIPTS_DIR/shell/tools.json"
SCENARIOS_FILE="$SCRIPTS_DIR/shell/scenarios.py"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:9000}"
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"
MODEL_SLUG="$(echo "$MODEL" | tr '/: ' '---')"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.6}"
OPENAI_MODEL_SLUG="$(echo "$OPENAI_MODEL" | tr '/: ' '---')"
SHELL_RECORD_SET="${SHELL_RECORD_SET:-all}"
record_scenario() {
  local endpoint_flag="$1"
  local endpoint="$2"
  local model="$3"
  local output="$4"
  local stream_flag="$5"
  local scenario="$6"
  echo "Recording $scenario ($stream_flag) against $endpoint with $model"
  if ! python "$SCENARIOS_FILE" prompts "$scenario" \
    | python "$SCRIPTS_DIR/record_cassette.py" \
        --mode responses \
        --turns 2 \
        "$stream_flag" \
        --model "$model" \
        "$endpoint_flag" "$endpoint" \
        --tools "$TOOLS_FILE" \
        --tool-outputs "$SCENARIOS_FILE" \
        --tool-choice auto \
        --max-output-tokens 4096 \
        --output "$output"
  then
    echo "ERROR: recording failed; captured YAML retained at $output" >&2
    return 1
  fi
  if ! python "$SCENARIOS_FILE" validate "$output" "$scenario" "$stream_flag"; then
    echo "ERROR: validation failed; captured YAML retained at $output" >&2
    return 1
  fi
}

record_provider_suite() {
  local endpoint_flag="$1"
  local endpoint="$2"
  local model="$3"
  local slug="$4"
  local provider="$5"
  local scenario

  for scenario in success nonzero-exit timeout multiple-commands; do
    record_scenario \
      "$endpoint_flag" "$endpoint" "$model" \
      "$BASE_DIR/shell-${provider}-${scenario}-${slug}-streaming.yaml" --stream "$scenario"
    record_scenario \
      "$endpoint_flag" "$endpoint" "$model" \
      "$BASE_DIR/shell-${provider}-${scenario}-${slug}-nonstreaming.yaml" --no-stream "$scenario"
  done
}

case "$SHELL_RECORD_SET" in
  gateway|openai|all) ;;
  *)
    echo "ERROR: SHELL_RECORD_SET must be gateway, openai, or all" >&2
    exit 1
    ;;
esac

if [[ "$SHELL_RECORD_SET" == "openai" || "$SHELL_RECORD_SET" == "all" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY must be set for SHELL_RECORD_SET=$SHELL_RECORD_SET" >&2
    exit 1
  fi
fi

mkdir -p "$BASE_DIR"

if [[ "$SHELL_RECORD_SET" == "openai" || "$SHELL_RECORD_SET" == "all" ]]; then
  record_provider_suite --openai https://api.openai.com "$OPENAI_MODEL" "$OPENAI_MODEL_SLUG" openai-reference
fi

if [[ "$SHELL_RECORD_SET" == "gateway" || "$SHELL_RECORD_SET" == "all" ]]; then
  record_provider_suite --gateway "$GATEWAY_URL" "$MODEL" "$MODEL_SLUG" gateway
fi

echo "Shell cassettes recorded and validated in $BASE_DIR"
