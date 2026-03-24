from __future__ import annotations

import argparse
import sys

from agentic_stack.configs.builders import RuntimeConfigError, build_runtime_config_for_supervisor
from agentic_stack.configs.sources import EnvSource
from agentic_stack.configs.startup import add_supervisor_responses_cli_arguments
from agentic_stack.entrypoints._serve._runtime import run_serve_spec
from agentic_stack.entrypoints._serve._spec import ServeSpecError, build_serve_spec


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--upstream",
        type=str,
        default=None,
        help="Exact external upstream API base URL (for example, http://127.0.0.1:8000/v1).",
    )

    parser.add_argument("--gateway-host", type=str, default=None, help="Gateway bind host.")
    parser.add_argument("--gateway-port", type=int, default=None, help="Gateway bind port.")
    parser.add_argument("--gateway-workers", type=int, default=None, help="Gunicorn worker count.")
    add_supervisor_responses_cli_arguments(parser)


def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-stacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Remote-upstream runtime supervisor for the vLLM Responses gateway.\n\n"
            "Use `agentic-stacks serve` to run the gateway against an existing upstream.\n"
            "Use `vllm serve --responses` for the colocated single-command local stack."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser(
        "serve",
        help="Run the remote-upstream gateway supervisor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run the gateway against an existing OpenAI-compatible upstream.\n\n"
            "This command does not spawn vLLM.\n"
            "For the colocated single-command local stack, use `vllm serve --responses`."
        ),
        epilog=(
            "Examples:\n"
            "  External upstream:\n"
            "    agentic-stacks serve --upstream http://127.0.0.1:8000/v1\n\n"
            "  External upstream with more gateway workers:\n"
            "    agentic-stacks serve --gateway-workers 4 --upstream http://127.0.0.1:8000/v1\n"
        ),
    )
    _add_serve_arguments(serve)
    return parser


def _run_serve(args: argparse.Namespace) -> int:
    env = EnvSource.from_env()
    try:
        runtime_config = build_runtime_config_for_supervisor(args=args, env=env)
        spec = build_serve_spec(runtime_config)
    except (RuntimeConfigError, ServeSpecError) as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code

    return run_serve_spec(spec)


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = _build_root_parser()
    ns = parser.parse_args(raw)

    if ns.command != "serve":
        parser.error("unknown subcommand")

    sys.exit(_run_serve(ns))


if __name__ == "__main__":
    main()
