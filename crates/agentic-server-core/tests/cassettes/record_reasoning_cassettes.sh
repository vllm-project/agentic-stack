#!/usr/bin/env bash
# Records the same explicit Responses reasoning configuration against OpenAI
# and the gateway, with optional direct-vLLM recordings for the legacy
# accumulator fixtures.
#
# Each provider records one streaming and one non-streaming response. The
# default `all` set records the OpenAI ground truth and its gateway counterpart.
#
# Usage from the repository root:
#   OPENAI_API_KEY=sk-... \
#     bash crates/agentic-server-core/tests/cassettes/record_reasoning_cassettes.sh
#   REASONING_RECORD_SET=gateway GATEWAY_URL=http://localhost:9000 \
#     bash crates/agentic-server-core/tests/cassettes/record_reasoning_cassettes.sh
#   REASONING_RECORD_SET=vllm VLLM_URL=http://localhost:5050 \
#     bash crates/agentic-server-core/tests/cassettes/record_reasoning_cassettes.sh

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${REASONING_OUTPUT_DIR:-$SCRIPTS_DIR/reasoning/responses}"
RECORDER="${RECORDER:-$SCRIPTS_DIR/record_cassette.py}"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:9000}"
VLLM_URL="${VLLM_URL:-http://localhost:5050}"
MODEL="${MODEL:-gpt-5.6}"
MODEL_SLUG="$(echo "$MODEL" | tr '/: ' '---')"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.6}"
OPENAI_MODEL_SLUG="$(echo "$OPENAI_MODEL" | tr '/: ' '---')"
REASONING_RECORD_SET="${REASONING_RECORD_SET:-all}"
REASONING_CONFIG="${REASONING_CONFIG:-{\"effort\":\"high\",\"summary\":\"detailed\"}}"
PROMPT='Determine whether 47 is the unique two-digit positive integer whose digits sum to 11 and whose reversal is 27 larger. Analyze the constraints, then reply with exactly one word: VALID or INVALID.'
STAGING_DIR=""
STAGED_OUTPUTS=()
FINAL_OUTPUTS=()

green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n'  "$*"; }

cleanup_staging() {
  if [[ -n "$STAGING_DIR" ]]; then
    rm -rf -- "$STAGING_DIR"
  fi
}

trap cleanup_staging EXIT

validate_reasoning_config() {
  python - "$REASONING_CONFIG" <<'PY'
import json
import sys

try:
    value = json.loads(sys.argv[1])
except json.JSONDecodeError as error:
    raise SystemExit(f"ERROR: REASONING_CONFIG is not valid JSON: {error}") from error
if not isinstance(value, dict):
    raise SystemExit("ERROR: REASONING_CONFIG must contain a JSON object")
PY
}

validate_recorded_response() {
  local file="$1"
  local stream_flag="$2"

  python - "$file" "$stream_flag" "$REASONING_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
streaming = sys.argv[2] == "--stream"
expected_reasoning = json.loads(sys.argv[3])
document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
turns = document.get("turns") or []
if len(turns) != 1:
    raise SystemExit(f"ERROR: expected one recorded turn in {path}, found {len(turns)}")

turn = turns[0]
request = (turn.get("request") or {}).get("body") or {}
if request.get("reasoning") != expected_reasoning:
    raise SystemExit(
        f"ERROR: recorded reasoning configuration differs from REASONING_CONFIG: {request.get('reasoning')}"
    )
if request.get("stream") is not streaming:
    raise SystemExit(f"ERROR: recorded stream mode differs from {streaming}")

response = turn.get("response") or {}
status_code = response.get("status_code")
if status_code != 200:
    raise SystemExit(f"ERROR: recording returned HTTP {status_code}: {response.get('body')}")

if not streaming:
    terminal = response.get("body") or {}
else:
    events = []
    for raw in response.get("sse") or []:
        for line in raw.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                events.append(json.loads(line.removeprefix("data: ")))
            except json.JSONDecodeError:
                continue
    errors = [event.get("error") for event in events if event.get("type") == "error"]
    if errors:
        raise SystemExit(f"ERROR: streaming recording returned an error event: {errors[0]}")
    terminal = next(
        (
            event.get("response")
            for event in reversed(events)
            if event.get("type") == "response.completed"
        ),
        None,
    ) or {}

if terminal.get("status") != "completed":
    raise SystemExit(f"ERROR: recording did not complete: {terminal}")
output = terminal.get("output") or []
output_types = [item.get("type") for item in output]
if "reasoning" not in output_types:
    raise SystemExit(f"ERROR: completed response has no reasoning item: {output_types}")
if "message" not in output_types:
    raise SystemExit(f"ERROR: completed response has no message item: {output_types}")
message_text = "".join(
    part.get("text", "")
    for item in output
    if item.get("type") == "message"
    for part in item.get("content") or []
    if part.get("type") == "output_text"
)
if message_text.strip() != "VALID":
    raise SystemExit(f"ERROR: expected the deterministic answer VALID, got {message_text!r}")
PY
}

record_single_turn() {
  local endpoint_flag="$1"
  local endpoint="$2"
  local model="$3"
  local output="$4"
  local stream_flag="$5"
  local staged_output
  local temporary_output

  staged_output="$STAGING_DIR/$(basename "$output")"
  temporary_output="$(mktemp "$STAGING_DIR/.reasoning-cassette.XXXXXX")"

  if ! printf '%s\n' "$PROMPT" \
    | python "$RECORDER" \
        --mode responses \
        --turns 1 \
        "$stream_flag" \
        --model "$model" \
        "$endpoint_flag" "$endpoint" \
        --reasoning "$REASONING_CONFIG" \
        --max-output-tokens 2048 \
        --output "$temporary_output"
  then
    rm -f -- "$temporary_output"
    return 1
  fi

  if ! validate_recorded_response "$temporary_output" "$stream_flag"; then
    rm -f -- "$temporary_output"
    return 1
  fi
  mv -- "$temporary_output" "$staged_output"
  STAGED_OUTPUTS+=("$staged_output")
  FINAL_OUTPUTS+=("$output")
  green "✓ reasoning cassette validated -> $output"
}

promote_recorded_suite() {
  local index

  for index in "${!STAGED_OUTPUTS[@]}"; do
    mv -- "${STAGED_OUTPUTS[$index]}" "${FINAL_OUTPUTS[$index]}"
    green "✓ reasoning cassette promoted -> ${FINAL_OUTPUTS[$index]}"
  done
}

record_provider_suite() {
  local provider="$1"
  local endpoint_flag="$2"
  local endpoint="$3"
  local model="$4"
  local output_prefix="$5"

  bold "$provider reasoning cassettes"
  bold "Endpoint:  $endpoint"
  bold "Model:     $model"
  bold "Reasoning: $REASONING_CONFIG"

  bold "$provider streaming reasoning response"
  record_single_turn \
    "$endpoint_flag" "$endpoint" "$model" \
    "$BASE_DIR/${output_prefix}-streaming.yaml" \
    --stream

  bold "$provider non-streaming reasoning response"
  record_single_turn \
    "$endpoint_flag" "$endpoint" "$model" \
    "$BASE_DIR/${output_prefix}-nonstreaming.yaml" \
    --no-stream
}

case "$REASONING_RECORD_SET" in
  gateway|vllm|openai|all) ;;
  *)
    echo "ERROR: REASONING_RECORD_SET must be gateway, vllm, openai, or all" >&2
    exit 1
    ;;
esac

validate_reasoning_config

# Validate OpenAI requirements before making any live requests. Final fixtures
# remain unchanged until every selected recording has completed validation.
if [[ "$REASONING_RECORD_SET" == "openai" || "$REASONING_RECORD_SET" == "all" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY must be set for REASONING_RECORD_SET=$REASONING_RECORD_SET" >&2
    exit 1
  fi
fi

mkdir -p "$BASE_DIR"
STAGING_DIR="$(mktemp -d "$BASE_DIR/.reasoning-suite.XXXXXX")"

if [[ "$REASONING_RECORD_SET" == "openai" || "$REASONING_RECORD_SET" == "all" ]]; then
  record_provider_suite \
    OpenAI \
    --openai https://api.openai.com \
    "$OPENAI_MODEL" \
    "reasoning-openai-reference-${OPENAI_MODEL_SLUG}"
fi

if [[ "$REASONING_RECORD_SET" == "gateway" || "$REASONING_RECORD_SET" == "all" ]]; then
  record_provider_suite \
    Gateway \
    --gateway "$GATEWAY_URL" \
    "$MODEL" \
    "reasoning-gateway-${MODEL_SLUG}"
fi

if [[ "$REASONING_RECORD_SET" == "vllm" ]]; then
  record_provider_suite \
    vLLM \
    --vllm "$VLLM_URL" \
    "$MODEL" \
    "reasoning-single-${MODEL_SLUG}"
fi

promote_recorded_suite
