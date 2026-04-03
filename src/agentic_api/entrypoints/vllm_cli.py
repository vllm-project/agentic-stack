import argparse
import logging
import subprocess
import sys

from agentic_api.config.runtime import RuntimeConfig
from agentic_api.entrypoints.cli import _normalize_base_url
from agentic_api.entrypoints.serve import run

logger = logging.getLogger(__name__)

_FLAG = "--agentic-api"

_GATEWAY_DEFAULTS = {
    "gateway_host": "0.0.0.0",
    "gateway_port": 9000,
    "gateway_workers": 1,
    "upstream_ready_timeout_s": 600.0,
    "upstream_ready_interval_s": 2.0,
}


def _parse_gateway_args(argv: list[str]) -> tuple[RuntimeConfig, list[str]]:
    """Strip agentic-stack flags from argv, return (RuntimeConfig, remaining_vllm_args)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--llm-api-base", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--gateway-host", default=_GATEWAY_DEFAULTS["gateway_host"])
    parser.add_argument(
        "--gateway-port", type=int, default=_GATEWAY_DEFAULTS["gateway_port"]
    )
    parser.add_argument(
        "--gateway-workers", type=int, default=_GATEWAY_DEFAULTS["gateway_workers"]
    )
    parser.add_argument(
        "--upstream-ready-timeout",
        type=float,
        default=_GATEWAY_DEFAULTS["upstream_ready_timeout_s"],
        dest="upstream_ready_timeout_s",
    )
    parser.add_argument(
        "--upstream-ready-interval",
        type=float,
        default=_GATEWAY_DEFAULTS["upstream_ready_interval_s"],
        dest="upstream_ready_interval_s",
    )

    filtered = [a for a in argv if a != _FLAG]
    known, remaining = parser.parse_known_args(filtered)

    llm_api_base = known.llm_api_base
    if llm_api_base is None:
        port = _find_flag_value(remaining, "--port") or "8000"
        llm_api_base = f"http://127.0.0.1:{port}"

    runtime_config = RuntimeConfig(
        llm_api_base=_normalize_base_url(llm_api_base),
        openai_api_key=known.openai_api_key,
        gateway_host=known.gateway_host,
        gateway_port=known.gateway_port,
        gateway_workers=known.gateway_workers,
        upstream_ready_timeout_s=known.upstream_ready_timeout_s,
        upstream_ready_interval_s=known.upstream_ready_interval_s,
    )
    return runtime_config, remaining


def _find_flag_value(args: list[str], flag: str) -> str | None:
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _spawn_vllm(argv: list[str]) -> subprocess.Popen[bytes]:
    """Start vLLM serve as a subprocess so the gateway can run in the foreground."""
    # argv = ["serve", "MODEL", ...vllm flags...]
    # api_server expects --model MODEL rather than a positional, so convert.
    args = argv[1:]  # strip "serve"
    if args and not args[0].startswith("-"):
        model, rest = args[0], args[1:]
        args = ["--model", model] + rest
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"] + args
    return subprocess.Popen(cmd)


def _delegate_to_vllm(argv: list[str]) -> None:
    """Delegate to upstream vllm (no --agentic-api flag). Replaces the process."""
    try:
        from vllm.scripts import main as vllm_main  # type: ignore[import]

        sys.argv = ["vllm"] + argv
        vllm_main()
    except SystemExit:
        raise


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)

    if _FLAG not in raw:
        _delegate_to_vllm(raw)
        return

    if raw and raw[0] == "serve" and "--help" in raw:
        print(
            "usage: vllm serve <MODEL> --agentic-api [vllm-serve args] [gateway args]\n\n"
            "Gateway args:\n"
            "  --llm-api-base URL            upstream vLLM base URL (default: inferred from --port)\n"
            "  --openai-api-key KEY          API key forwarded to upstream\n"
            f"  --gateway-host HOST           gateway bind host (default: {_GATEWAY_DEFAULTS['gateway_host']})\n"
            f"  --gateway-port PORT           gateway bind port (default: {_GATEWAY_DEFAULTS['gateway_port']})\n"
            f"  --gateway-workers N           uvicorn worker count (default: {_GATEWAY_DEFAULTS['gateway_workers']})\n"
            f"  --upstream-ready-timeout S    seconds to wait for vLLM (default: {_GATEWAY_DEFAULTS['upstream_ready_timeout_s']})\n"
            f"  --upstream-ready-interval S   poll interval in seconds (default: {_GATEWAY_DEFAULTS['upstream_ready_interval_s']})\n"
        )
        sys.exit(0)

    runtime_config, vllm_args = _parse_gateway_args(raw)

    # Start vLLM in a subprocess so the gateway can run in the foreground.
    vllm_proc = _spawn_vllm(vllm_args)
    try:
        run(runtime_config)
    except TimeoutError as e:
        logger.error("%s", e)
        sys.exit(1)
    finally:
        vllm_proc.terminate()
        vllm_proc.wait()


if __name__ == "__main__":
    main()
