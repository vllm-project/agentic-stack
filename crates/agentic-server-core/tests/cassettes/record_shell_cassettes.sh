#!/usr/bin/env bash
# Records the same client-owned local-shell request against the gateway and
# OpenAI. The produced fixtures capture request normalization and blocking /
# streaming shell_call response lifecycles; the recorder never executes the
# returned command.

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPTS_DIR/shell"
TOOLS_FILE="$BASE_DIR/tools.json"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:9000}"
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"
MODEL_SLUG="$(echo "$MODEL" | tr '/: ' '---')"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.6}"
OPENAI_MODEL_SLUG="$(echo "$OPENAI_MODEL" | tr '/: ' '---')"
SHELL_RECORD_SET="${SHELL_RECORD_SET:-all}"
PROMPT='Use the shell tool to run exactly one command: pwd. Do not answer in prose.'

record_single_turn() {
  local endpoint_flag="$1"
  local endpoint="$2"
  local model="$3"
  local output="$4"
  local stream_flag="$5"
  local temporary_output

  temporary_output="$(mktemp "$BASE_DIR/.shell-cassette.XXXXXX")"
  if ! printf '%s\n' "$PROMPT" \
    | python "$SCRIPTS_DIR/record_cassette.py" \
        --mode responses \
        --turns 1 \
        "$stream_flag" \
        --model "$model" \
        "$endpoint_flag" "$endpoint" \
        --tools "$TOOLS_FILE" \
        --tool-choice required \
        --max-output-tokens 1024 \
        --output "$temporary_output"
  then
    rm -f -- "$temporary_output"
    return 1
  fi
  mv -- "$temporary_output" "$output"
}

record_provider_suite() {
  local endpoint_flag="$1"
  local endpoint="$2"
  local model="$3"
  local slug="$4"
  local provider="$5"

  record_single_turn \
    "$endpoint_flag" "$endpoint" "$model" \
    "$BASE_DIR/shell-${provider}-${slug}-streaming.yaml" --stream
  record_single_turn \
    "$endpoint_flag" "$endpoint" "$model" \
    "$BASE_DIR/shell-${provider}-${slug}-nonstreaming.yaml" --no-stream
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
  record_provider_suite --openai https://api.openai.com "$OPENAI_MODEL" "$OPENAI_MODEL_SLUG" openai-reference
fi

if [[ "$SHELL_RECORD_SET" == "gateway" || "$SHELL_RECORD_SET" == "all" ]]; then
  record_provider_suite --gateway "$GATEWAY_URL" "$MODEL" "$MODEL_SLUG" gateway
fi
