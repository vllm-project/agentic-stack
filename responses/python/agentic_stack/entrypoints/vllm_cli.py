from __future__ import annotations

import sys

from agentic_stack.entrypoints.vllm._adapter import run_upstream_cli
from agentic_stack.entrypoints.vllm._runtime import run_integrated_serve
from agentic_stack.entrypoints.vllm._spec import (
    IntegratedSpecError,
    build_integrated_serve_spec,
    format_integrated_help,
    should_show_integrated_help,
)


def _delegate_to_upstream_vllm(argv: list[str]) -> int:
    return run_upstream_cli(argv)


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--responses" not in raw:
        sys.exit(_delegate_to_upstream_vllm(raw))

    if should_show_integrated_help(raw):
        print(format_integrated_help())
        sys.exit(0)

    try:
        spec = build_integrated_serve_spec(raw)
    except IntegratedSpecError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(exc.exit_code)

    sys.exit(run_integrated_serve(spec))


if __name__ == "__main__":
    main()
