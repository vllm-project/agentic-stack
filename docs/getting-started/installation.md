# Installation

Get started with `vLLM Responses` by installing the package and its dependencies.

## Prerequisites

- **Python 3.12+**: Ensure you have a compatible Python version installed.
- **uv** (Recommended): We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.
- **tar**: If you use the built-in **Code Interpreter** (enabled by default), the first start may download the Pyodide
    runtime (~400MB) and extract it; `tar` must be available.
- **Bun** (Development): Required for source checkouts if you want the built-in Code Interpreter to work (enabled by
    default). Wheels for Linux x86_64 bundle a native Code Interpreter binary and do not require Bun.
- **vLLM**: Required if you want the integrated colocated runtime via `vllm serve --responses`.

## Install the CLI

We recommend setting up a virtual environment using `uv`.

### Install from a prebuilt wheel (Linux x86_64) (Recommended)

Download a prebuilt wheel (`agentic_stack-*.whl`) from GitHub Releases (preferred) or a CI run artifact, then install it:

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install path/to/agentic_stack-*.whl
agentic-stacks --help
```

On Linux x86_64 wheels, the Code Interpreter server binary is bundled, so **Bun is not required**.

!!! note "Non-Linux platforms"

    The gateway is a Python service and can run on other platforms, but the bundled Code Interpreter binary is currently
    only shipped in Linux x86_64 wheels. On other platforms, either disable the tool via `--code-interpreter disabled`,
    or run from a source checkout and use the (development-only) Bun fallback.

### Install from source (repo checkout)

If you are working from a source checkout and want the gateway to work with the default configuration (Code Interpreter
enabled), use the Bun fallback:

```bash
git clone https://github.com/EmbeddedLLM/agentic-stacks
cd agentic-stacks

uv venv --python=3.12
source .venv/bin/activate
uv pip install -e ./responses

cd responses/python/agentic_stack/tools/code_interpreter
bun install
export VR_CODE_INTERPRETER_DEV_BUN_FALLBACK=1
cd -

agentic-stacks --help
```

### First start: Pyodide download (Code Interpreter)

If `code_interpreter` is enabled (default), the first start may download the Pyodide runtime (~400MB) into a cache
directory and extract it. Subsequent starts reuse the cache.

- Default cache: `${XDG_CACHE_HOME:-$HOME/.cache}/agentic-stacks/pyodide`
- Override: set `VR_PYODIDE_CACHE_DIR` to a persistent directory with enough free disk space.

## Build a wheel from a source checkout

If you want to produce a local wheel from this repo, build from the
`responses/` package directory.

### Rebuild the bundled Code Interpreter binary (Linux x86_64 only)

This step is only needed when you want the built wheel to include a freshly
compiled native Code Interpreter binary.

```bash
bash scripts/ci/prebuild_code_interpreter_linux_x86_64.sh responses
```

### Build wheel and sdist

```bash
uv pip install -e './responses[build]'
cd responses
python -m build --wheel --sdist
```

Artifacts are written to:

- `responses/dist/`

On Linux x86_64, wheels built after the prebuild step bundle the native Code
Interpreter binary. On other platforms, use the source-install Bun fallback or
disable Code Interpreter.

## Optional dependency sets

Some features require additional optional dependencies.

### OpenTelemetry tracing (optional)

If you want to enable OpenTelemetry tracing (`VR_TRACING_ENABLED=true`), install with the `tracing` extra.

### Documentation toolchain (contributors)

If you want to build/serve the MkDocs site locally, install with the `docs` extra.
