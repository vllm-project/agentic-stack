#!/usr/bin/env bash
set -euo pipefail

RESPONSES_DIR="${1:-$(pwd)}"
# Normalize to an absolute path so later `cd` calls don't accidentally make `--outfile`
# relative to a different working directory.
RESPONSES_DIR="$(cd "${RESPONSES_DIR}" && pwd)"
CODE_DIR="${RESPONSES_DIR}/python/agentic_stack/tools/code_interpreter"
TARGET="${CODE_DIR}/bin/linux/x86_64/code-interpreter-server"

if [[ ! -d "${CODE_DIR}" ]]; then
  echo "error: code interpreter dir not found: ${CODE_DIR}" >&2
  exit 2
fi

export BUN_INSTALL="${BUN_INSTALL:-/tmp/bun}"
export PATH="${BUN_INSTALL}/bin:${PATH}"

if ! command -v bun >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://bun.sh/install | bash
  else
    echo "error: curl is required to install bun in CI" >&2
    exit 2
  fi
fi

echo "[ci] bun: $(bun --version)"

cd "${CODE_DIR}"
bun install --frozen-lockfile

mkdir -p "$(dirname "${TARGET}")"
# The code interpreter uses Bun Workers when `--workers > 0`. Bun requires worker entrypoints
# to be explicitly passed to `bun build --compile` so they are bundled into the executable.
# Note: Bun's `--minify` currently breaks worker entrypoint bundling for `--compile` executables,
# causing runtime failures like "ModuleNotFound resolving \"/$bunfs/root/worker.ts\"".
# Prefer correctness/reliability over a slightly smaller binary.
bun build --compile src/index.ts worker.ts --outfile "${TARGET}"
chmod +x "${TARGET}"

echo "[ci] code-interpreter-server built: ${TARGET}"
ls -la "${TARGET}"
