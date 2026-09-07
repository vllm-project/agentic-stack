# Python installation and workflows

`agentic-api` is the Python distribution for the Rust-backed Agentic API gateway. vLLM is a supported inference
backend, not part of the Agentic API product name. Use the base wheel when you want a proxy-only install, and add the
`[local]` extra when you want the launcher to manage a local vLLM process.

The Rust-native `agentic` CLI remains supported for `run codex`, `run claude`, `serve`, and `validate`.

## Install the release artifact

This release produces wheel artifacts for 0.5.0 on supported platforms but does not publish them to PyPI yet.
Download the wheel for your platform from the release workflow and use its absolute path below:

```bash
WHEEL_PATH=/absolute/path/to/agentic_api-PLATFORM.whl
```

### Install the base package

Proxy-only installs use the base wheel:

```bash
uv pip install "$WHEEL_PATH"
agentic-api serve --vllm-base-url http://existing-vllm:8000
```

Use this mode when an external vLLM server is already running and Agentic API should only proxy to it.

### Install the local extra

On supported Linux hosts, local installs add the tested vLLM runtime candidate. The `file://` reference must use the
absolute wheel path assigned above:

```bash
uv pip install "agentic-api[local] @ file://$WHEEL_PATH"
agentic-api serve --model Qwen/Qwen3-30B-A3B-FP8
```

The launcher still accepts arbitrary `--model` values. The base package does not install vLLM; the `[local]` extra
supplies the tested vLLM dependency and makes the managed-vLLM workflow available.

Managed vLLM supports passthrough arguments after `--`:

```bash
agentic-api serve --model Qwen/Qwen3-30B-A3B-FP8 -- \
  --dtype bfloat16 \
  --max-model-len=32768
```

## After PyPI publication

The following public-index installation and `uvx` commands apply after the PyPI publication gate for a future release.
They do not work until the package is published:

```bash
uv pip install agentic-api
uv pip install "agentic-api[local]"
uvx --from agentic-api agentic-api doctor
uvx --from agentic-api agentic-api serve --vllm-base-url http://existing-vllm:8000
```

## Check the install

`doctor` reports whether the packaged Rust executable is present, whether the tested local vLLM wheel is installed, and
whether the current mode is healthy.

With no mode selected, `doctor` reports both local and remote health but uses remote health for its exit status, so the
base proxy-only install is considered healthy when its packaged gateway is available.

```bash
agentic-api doctor
agentic-api doctor --mode remote
agentic-api doctor --mode local
agentic-api doctor --mode remote --json
```

Use `--mode remote` when you only need the packaged Rust gateway checks. Use `--mode local` when you want to verify the
tested vLLM runtime and executable are available.

## Rust-native CLI usage

The Python package does not replace the Rust CLI. It complements it.

```bash
agentic run codex --model MODEL_ID
agentic run claude --model SERVED_MODEL_ALIAS
```

## Known-good model profiles

The matrix below is documentation data, not an allowlist. `agentic-api serve` still accepts arbitrary `--model`
values. The served alias column is only needed when Claude Code requires a slash-free model name, and the alias values
here are examples that should be revalidated on the target Linux GPU before promotion.

| Model identifier | Required hardware class | Served alias for Claude Code | Tested launch arguments |
| --- | --- | --- | --- |
| `Qwen/Qwen3-30B-A3B-FP8` | Linux GPU host that can serve a 30B FP8 model | `qwen3-30b-a3b-fp8` | `vllm serve Qwen/Qwen3-30B-A3B-FP8 --reasoning-parser deepseek_r1 --port 5050` and `vllm serve Qwen/Qwen3-30B-A3B-FP8 --tool-call-parser hermes --enable-auto-tool-choice --port 5050` |

Other documented model IDs already exercised in this repository include `Qwen/Qwen3.5-35B-A3B-FP8` and
`Qwen/Qwen3.8-27B-FP8`. Treat them as examples pending hardware revalidation rather than as a CLI allowlist.
