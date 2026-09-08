#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
real_codex="$(command -v codex || true)"
test_root="$(mktemp -d)"
trap 'rm -rf "$test_root"' EXIT

fake_bin="$test_root/bin"
capture_dir="$test_root/captures"
mkdir -p "$fake_bin" "$capture_dir"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_file_contains() {
  local file="$1"
  local expected="$2"
  grep -F -- "$expected" "$file" >/dev/null || fail "$file does not contain: $expected"
}

assert_catalog_modalities() {
  local file="$1"
  local expected="$2"
  local actual
  actual="$(jq -c '.models[0].input_modalities' "$file")"
  [[ "$actual" == "$expected" ]] || fail "$file advertises $actual, expected $expected"
}

assert_file_excludes() {
  local file="$1"
  local unexpected="$2"
  if grep -F -- "$unexpected" "$file" >/dev/null; then
    fail "$file unexpectedly contains: $unexpected"
  fi
}

cat >"$fake_bin/cargo" <<'EOF'
#!/usr/bin/env bash
{
  printf 'V_API_BASE=%s\n' "${V_API_BASE:-}"
  printf 'V_MODEL=%s\n' "${V_MODEL:-}"
  printf 'MESSAGES_MODEL_OVERRIDE=%s\n' "${MESSAGES_MODEL_OVERRIDE:-}"
  printf 'GATEWAY_PORT=%s\n' "${GATEWAY_PORT:-}"
  printf 'DATABASE_URL=%s\n' "${DATABASE_URL:-}"
  printf '%s\n' '--- args ---'
  printf '%s\n' "$@"
} >"$CAPTURE_DIR/gateway.txt"
EOF

cat >"$fake_bin/claude" <<'EOF'
#!/usr/bin/env bash
{
  printf 'ANTHROPIC_BASE_URL=%s\n' "${ANTHROPIC_BASE_URL:-}"
  printf 'ANTHROPIC_MODEL=%s\n' "${ANTHROPIC_MODEL:-}"
  printf 'ANTHROPIC_DEFAULT_OPUS_MODEL=%s\n' "${ANTHROPIC_DEFAULT_OPUS_MODEL:-}"
  printf 'CLAUDE_CODE_EFFORT_LEVEL=%s\n' "${CLAUDE_CODE_EFFORT_LEVEL:-}"
  printf 'ANTHROPIC_API_KEY=%s\n' "${ANTHROPIC_API_KEY:-}"
  printf 'ANTHROPIC_AUTH_TOKEN=%s\n' "${ANTHROPIC_AUTH_TOKEN:-}"
  if [[ -n "${CLAUDE_CODE_USE_VERTEX+x}" ]]; then printf 'CLAUDE_CODE_USE_VERTEX=set\n'; fi
  if [[ -n "${ANTHROPIC_VERTEX_PROJECT_ID+x}" ]]; then printf 'ANTHROPIC_VERTEX_PROJECT_ID=set\n'; fi
  printf '%s\n' '--- args ---'
  printf '%s\n' "$@"
} >"$CAPTURE_DIR/claude.txt"
EOF

cat >"$fake_bin/codex" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  echo 'codex-cli 0.148.0'
  exit 0
fi
{
  printf 'CODEX_HOME=%s\n' "${CODEX_HOME:-}"
  printf '%s\n' '--- args ---'
  printf '%s\n' "$@"
} >"$CAPTURE_DIR/codex.txt"
EOF

cat >"$fake_bin/curl" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" >"$CAPTURE_DIR/curl.txt"
cat <<'JSON'
{
  "models": [{
    "slug": "agentic-api",
    "display_name": "agentic-api",
    "auto_review_model_override": "agentic-api",
    "supported_in_api": true,
    "priority": 1,
    "shell_type": "shell_command",
    "visibility": "list",
    "base_instructions": "",
    "supported_reasoning_levels": [
      {"effort": "low", "description": "Fast responses with lighter reasoning"},
      {"effort": "medium", "description": "Balances speed and reasoning depth"},
      {"effort": "high", "description": "Greater reasoning depth for complex problems"}
    ],
    "default_reasoning_summary": "auto",
    "supports_reasoning_summaries": false,
    "support_verbosity": false,
    "default_verbosity": null,
    "apply_patch_tool_type": "freeform",
    "web_search_tool_type": "text",
    "truncation_policy": {"mode": "bytes", "limit": 100000},
    "supports_parallel_tool_calls": true,
    "supports_image_detail_original": false,
    "effective_context_window_percent": 95,
    "experimental_supported_tools": [],
    "supports_search_tool": true,
    "use_responses_lite": false,
    "tool_mode": null,
    "multi_agent_version": null,
    "input_modalities": ["text", "image"]
  }]
}
JSON
EOF

chmod +x "$fake_bin/cargo" "$fake_bin/claude" "$fake_bin/codex" "$fake_bin/curl"

PATH="$fake_bin:$PATH" CAPTURE_DIR="$capture_dir" \
  "$repo_root/scripts/agentic-api.sh" >/dev/null

assert_file_contains "$capture_dir/gateway.txt" 'V_API_BASE=http://127.0.0.1:8000/v1'
assert_file_contains "$capture_dir/gateway.txt" 'V_MODEL=agentic-api'
assert_file_contains "$capture_dir/gateway.txt" 'MESSAGES_MODEL_OVERRIDE=agentic-api'
assert_file_contains "$capture_dir/gateway.txt" 'GATEWAY_PORT=3020'
assert_file_contains "$capture_dir/gateway.txt" 'sqlite:///tmp/agentic_api_3020.db'
assert_file_contains "$capture_dir/gateway.txt" '--bin'
assert_file_contains "$capture_dir/gateway.txt" 'agentic-server'
assert_file_contains "$capture_dir/gateway.txt" 'http://127.0.0.1:8000/v1'

PATH="$fake_bin:$PATH" CAPTURE_DIR="$capture_dir" CLAUDE_BIN="$fake_bin/claude" \
  CLAUDE_CODE_USE_VERTEX=1 ANTHROPIC_VERTEX_PROJECT_ID=test-project ANTHROPIC_API_KEY= \
  "$repo_root/scripts/agentic-claude.sh" >/dev/null

assert_file_contains "$capture_dir/claude.txt" 'ANTHROPIC_BASE_URL=http://127.0.0.1:3020'
assert_file_contains "$capture_dir/claude.txt" 'ANTHROPIC_MODEL=agentic-api'
assert_file_contains "$capture_dir/claude.txt" 'ANTHROPIC_DEFAULT_OPUS_MODEL=agentic-api'
assert_file_contains "$capture_dir/claude.txt" 'CLAUDE_CODE_EFFORT_LEVEL=medium'
assert_file_contains "$capture_dir/claude.txt" 'ANTHROPIC_API_KEY=demo'
assert_file_contains "$capture_dir/claude.txt" 'ANTHROPIC_AUTH_TOKEN=demo'
assert_file_contains "$capture_dir/claude.txt" '--bare'
assert_file_excludes "$capture_dir/claude.txt" '--dangerously-skip-permissions'
assert_file_contains "$capture_dir/claude.txt" '--effort'
assert_file_contains "$capture_dir/claude.txt" 'medium'
assert_file_contains "$capture_dir/claude.txt" 'manual'
assert_file_excludes "$capture_dir/claude.txt" 'CLAUDE_CODE_USE_VERTEX='
assert_file_excludes "$capture_dir/claude.txt" 'ANTHROPIC_VERTEX_PROJECT_ID='

codex_home="$test_root/codex-home"
PATH="$fake_bin:$PATH" CAPTURE_DIR="$capture_dir" CODEX_BIN="$fake_bin/codex" \
  AGENTIC_CODEX_HOME="$codex_home" CODEX_HOME="$test_root/must-not-be-used" \
  "$repo_root/scripts/agentic-codex.sh" >/dev/null

assert_file_contains "$capture_dir/codex.txt" 'CODEX_HOME='
assert_file_contains "$capture_dir/codex.txt" '--search'
assert_file_excludes "$capture_dir/codex.txt" '--dangerously-bypass-approvals-and-sandbox'
assert_file_excludes "$capture_dir/codex.txt" 'workspace-write'
assert_file_excludes "$capture_dir/codex.txt" 'on-request'
assert_file_excludes "$capture_dir/codex.txt" 'must-not-be-used'
assert_file_contains "$codex_home/config.toml" 'model_provider = "agentic-api"'
assert_file_contains "$codex_home/config.toml" 'base_url = "http://127.0.0.1:3020/v1"'
assert_file_contains "$codex_home/config.toml" 'model = "agentic-api"'
assert_file_contains "$codex_home/config.toml" 'supports_websockets = true'
assert_file_contains "$codex_home/model_catalog.json" '"web_search_tool_type": "text"'
assert_file_contains "$codex_home/model_catalog.json" '"shell_type": "shell_command"'
assert_catalog_modalities "$codex_home/model_catalog.json" '["text","image"]'
assert_file_contains "$capture_dir/curl.txt" 'http://127.0.0.1:3020/v1/models?client_version=0.148.0'

PATH="$fake_bin:$PATH" CAPTURE_DIR="$capture_dir" AGENTIC_YOLO=1 CLAUDE_BIN="$fake_bin/claude" \
  "$repo_root/scripts/agentic-claude.sh" >/dev/null
assert_file_contains "$capture_dir/claude.txt" '--dangerously-skip-permissions'

PATH="$fake_bin:$PATH" CAPTURE_DIR="$capture_dir" AGENTIC_YOLO=1 CODEX_BIN="$fake_bin/codex" \
  AGENTIC_CODEX_HOME="$codex_home" "$repo_root/scripts/agentic-codex.sh" >/dev/null
assert_file_contains "$capture_dir/codex.txt" '--dangerously-bypass-approvals-and-sandbox'

if [[ -n "$real_codex" ]]; then
  CODEX_HOME="$codex_home" "$real_codex" features list >/dev/null
fi

echo 'agentic launcher tests passed'
