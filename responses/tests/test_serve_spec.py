from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from agentic_stack.configs.builders import (
    RuntimeConfigError,
    build_runtime_config_for_supervisor,
)
from agentic_stack.configs.sources import EnvSource
from agentic_stack.entrypoints._helper_runtime import SpawnCodeInterpreterSpec
from agentic_stack.entrypoints._serve._spec import (
    ExternalCodeInterpreterSpec,
    ExternalUpstreamSpec,
    ServeSpecError,
    build_serve_spec,
)


def _base_args(**overrides) -> argparse.Namespace:
    data = dict(
        upstream=None,
        gateway_host=None,
        gateway_port=None,
        gateway_workers=None,
        web_search_profile=None,
        code_interpreter="disabled",
        code_interpreter_port=None,
        code_interpreter_workers=None,
        code_interpreter_startup_timeout=None,
        upstream_ready_timeout=None,
        upstream_ready_interval=None,
        mcp_config=None,
        mcp_port=None,
    )
    data.update(overrides)
    return argparse.Namespace(**data)


def test_build_runtime_config_for_supervisor_requires_upstream() -> None:
    with pytest.raises(RuntimeConfigError, match=r"no upstream configured"):
        build_runtime_config_for_supervisor(
            args=_base_args(),
            env=EnvSource(environ={}),
        )


def test_build_runtime_config_for_supervisor_prefers_cli_upstream() -> None:
    runtime_config = build_runtime_config_for_supervisor(
        args=_base_args(upstream="http://127.0.0.1:8457"),
        env=EnvSource(environ={"VR_LLM_API_BASE": "http://example.invalid:9999"}),
    )
    assert runtime_config.llm_api_base == "http://127.0.0.1:8457"


def test_build_runtime_config_for_supervisor_ignores_env_upstream_without_cli() -> None:
    with pytest.raises(RuntimeConfigError, match=r"no upstream configured"):
        build_runtime_config_for_supervisor(
            args=_base_args(),
            env=EnvSource(environ={"VR_LLM_API_BASE": "http://example.invalid:9999"}),
        )


def test_build_runtime_config_for_supervisor_rejects_unknown_web_search_profile() -> None:
    with pytest.raises(
        RuntimeConfigError,
        match=(
            r"unknown --web-search-profile='missing_profile'; expected one of: "
            r"duckduckgo_plus_fetch, exa_mcp"
        ),
    ):
        build_runtime_config_for_supervisor(
            args=_base_args(
                upstream="http://127.0.0.1:8457",
                web_search_profile="missing_profile",
            ),
            env=EnvSource(environ={}),
        )


def test_run_serve_bootstraps_builtin_registries_before_runtime_config_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_stack.entrypoints.serve as serve_entrypoint
    import agentic_stack.tools as tools_mod

    monkeypatch.setattr(tools_mod, "TOOLS", {})
    monkeypatch.setattr(
        serve_entrypoint, "build_serve_spec", lambda runtime_config: runtime_config
    )
    monkeypatch.setattr(serve_entrypoint, "run_serve_spec", lambda spec: 0)

    exit_code = serve_entrypoint._run_serve(
        _base_args(
            upstream="http://127.0.0.1:8457",
            web_search_profile="exa_mcp",
        )
    )

    assert exit_code == 0


def test_build_serve_spec_code_interpreter_prefers_bundled_binary(
    tmp_path: Path, monkeypatch
) -> None:
    import agentic_stack.entrypoints._helper_runtime as helper_runtime_mod

    class _FakeSpec:
        def __init__(self, path: Path) -> None:
            self.submodule_search_locations = [str(path)]

    monkeypatch.setattr(
        helper_runtime_mod.importlib.util, "find_spec", lambda name: _FakeSpec(tmp_path)
    )

    bundled = tmp_path / "bin" / "linux" / "x86_64" / "code-interpreter-server"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("stub", encoding="utf-8")

    runtime_config = build_runtime_config_for_supervisor(
        args=_base_args(
            upstream="http://127.0.0.1:8457",
            code_interpreter="spawn",
            code_interpreter_port=5971,
            code_interpreter_workers=2,
        ),
        env=EnvSource(environ={"VR_PYODIDE_CACHE_DIR": str(tmp_path / "cache")}),
    )
    spec = build_serve_spec(runtime_config)
    assert isinstance(spec.code_interpreter, SpawnCodeInterpreterSpec)
    assert spec.code_interpreter.cmd[0] == str(bundled)
    assert spec.code_interpreter.cmd[-2:] == ["--workers", "2"]
    assert spec.code_interpreter.cwd == tmp_path


def test_build_serve_spec_code_interpreter_uses_bun_fallback(tmp_path: Path, monkeypatch) -> None:
    import agentic_stack.entrypoints._helper_runtime as helper_runtime_mod

    class _FakeSpec:
        def __init__(self, path: Path) -> None:
            self.submodule_search_locations = [str(path)]

    monkeypatch.setattr(
        helper_runtime_mod.importlib.util, "find_spec", lambda name: _FakeSpec(tmp_path)
    )
    monkeypatch.setattr(helper_runtime_mod.shutil, "which", lambda name: "/usr/bin/bun")

    src = tmp_path / "src" / "index.ts"
    src.parent.mkdir(parents=True)
    src.write_text("console.log('hi')", encoding="utf-8")

    runtime_config = build_runtime_config_for_supervisor(
        args=_base_args(
            upstream="http://127.0.0.1:8457",
            code_interpreter="spawn",
            code_interpreter_port=5971,
            code_interpreter_workers=0,
        ),
        env=EnvSource(
            environ={
                "VR_PYODIDE_CACHE_DIR": str(tmp_path / "cache"),
                "VR_CODE_INTERPRETER_DEV_BUN_FALLBACK": "1",
            }
        ),
    )
    spec = build_serve_spec(runtime_config)
    assert isinstance(spec.code_interpreter, SpawnCodeInterpreterSpec)
    assert spec.code_interpreter.cmd[:2] == ["/usr/bin/bun", "src/index.ts"]
    assert spec.code_interpreter.cwd == tmp_path


def test_build_serve_spec_code_interpreter_errors_without_binary_or_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    import agentic_stack.entrypoints._helper_runtime as helper_runtime_mod

    class _FakeSpec:
        def __init__(self, path: Path) -> None:
            self.submodule_search_locations = [str(path)]

    monkeypatch.setattr(
        helper_runtime_mod.importlib.util, "find_spec", lambda name: _FakeSpec(tmp_path)
    )

    runtime_config = build_runtime_config_for_supervisor(
        args=_base_args(
            upstream="http://127.0.0.1:8457",
            code_interpreter="spawn",
            code_interpreter_port=5971,
            code_interpreter_workers=0,
        ),
        env=EnvSource(environ={"VR_PYODIDE_CACHE_DIR": str(tmp_path / "cache")}),
    )
    with pytest.raises(ServeSpecError, match=r"no bundled code-interpreter binary"):
        build_serve_spec(runtime_config)


def test_build_runtime_config_for_supervisor_ignores_env_gateway_settings() -> None:
    runtime_config = build_runtime_config_for_supervisor(
        args=_base_args(upstream="http://127.0.0.1:8457"),
        env=EnvSource(environ={"VR_HOST": "1.2.3.4", "VR_PORT": "7777", "VR_WORKERS": "9"}),
    )
    assert runtime_config.gateway_host == "0.0.0.0"
    assert runtime_config.gateway_port == 5969
    assert runtime_config.gateway_workers == 1


def test_build_runtime_config_for_supervisor_ignores_env_code_interpreter_settings() -> None:
    runtime_config = build_runtime_config_for_supervisor(
        args=_base_args(upstream="http://127.0.0.1:8457"),
        env=EnvSource(
            environ={
                "VR_CODE_INTERPRETER_MODE": "external",
                "VR_CODE_INTERPRETER_PORT": "6111",
                "VR_CODE_INTERPRETER_WORKERS": "3",
                "VR_CODE_INTERPRETER_STARTUP_TIMEOUT": "15.0",
            }
        ),
    )
    assert runtime_config.code_interpreter_mode == "disabled"
    assert runtime_config.code_interpreter_port is None
    assert runtime_config.code_interpreter_workers is None
    assert runtime_config.code_interpreter_startup_timeout_s == 600.0


def test_build_runtime_config_for_supervisor_cli_zero_values_are_preserved() -> None:
    runtime_config = build_runtime_config_for_supervisor(
        args=_base_args(
            upstream="http://127.0.0.1:8457",
            gateway_port=0,
            code_interpreter="external",
            code_interpreter_workers=0,
        ),
        env=EnvSource(environ={}),
    )
    assert runtime_config.gateway_port == 0
    assert runtime_config.code_interpreter_workers == 0


def test_build_runtime_config_for_supervisor_reads_upstream_ready_controls() -> None:
    runtime_config = build_runtime_config_for_supervisor(
        args=_base_args(
            upstream="http://127.0.0.1:8457",
            upstream_ready_timeout="45.0",
            upstream_ready_interval="1.5",
        ),
        env=EnvSource(environ={}),
    )
    assert runtime_config.upstream_ready_timeout_s == 45.0
    assert runtime_config.upstream_ready_interval_s == 1.5


def test_build_serve_spec_enables_builtin_mcp_runtime_only_with_config_path() -> None:
    runtime_config = build_runtime_config_for_supervisor(
        args=_base_args(upstream="http://127.0.0.1:8457"),
        env=EnvSource(environ={}),
    )
    spec_disabled = build_serve_spec(runtime_config)
    assert spec_disabled.mcp_runtime is None

    runtime_config = build_runtime_config_for_supervisor(
        args=_base_args(
            upstream="http://127.0.0.1:8457",
            mcp_config="/tmp/mcp.json",
            mcp_port=6101,
        ),
        env=EnvSource(environ={}),
    )
    spec_enabled = build_serve_spec(runtime_config)
    assert spec_enabled.mcp_runtime is not None
    assert spec_enabled.mcp_runtime.host == "127.0.0.1"
    assert spec_enabled.mcp_runtime.port == 6101
    assert spec_enabled.mcp_runtime.ready_url == "http://127.0.0.1:6101/health"


def test_build_serve_spec_builtin_mcp_runtime_uses_default_url_when_unset() -> None:
    runtime_config = build_runtime_config_for_supervisor(
        args=_base_args(upstream="http://127.0.0.1:8457", mcp_config="/tmp/mcp.json"),
        env=EnvSource(environ={}),
    )
    spec = build_serve_spec(runtime_config)

    assert spec.mcp_runtime is not None
    assert spec.mcp_runtime.host == "127.0.0.1"
    assert spec.mcp_runtime.port == 5981
    assert spec.mcp_runtime.ready_url == "http://127.0.0.1:5981/health"


def test_build_serve_spec_web_search_profile_can_enable_builtin_mcp_runtime_without_config() -> (
    None
):
    runtime_config = build_runtime_config_for_supervisor(
        args=_base_args(upstream="http://127.0.0.1:8457", web_search_profile="exa_mcp"),
        env=EnvSource(environ={}),
    )
    spec = build_serve_spec(runtime_config)

    assert runtime_config.mcp_config_path is None
    assert runtime_config.mcp_builtin_runtime_url == "http://127.0.0.1:5981"
    assert spec.mcp_runtime is not None
    assert spec.mcp_runtime.port == 5981


def test_build_runtime_config_for_supervisor_cli_mcp_port_overrides_env_url() -> None:
    runtime_config = build_runtime_config_for_supervisor(
        args=_base_args(
            upstream="http://127.0.0.1:8457",
            mcp_config="/tmp/mcp.json",
            mcp_port=6201,
        ),
        env=EnvSource(environ={}),
    )
    spec = build_serve_spec(runtime_config)

    assert spec.mcp_runtime is not None
    assert spec.mcp_runtime.host == "127.0.0.1"
    assert spec.mcp_runtime.port == 6201
    assert runtime_config.mcp_builtin_runtime_url == "http://127.0.0.1:6201"


def test_build_runtime_config_for_supervisor_rejects_mcp_port_without_config() -> None:
    with pytest.raises(
        RuntimeConfigError,
        match=r"--mcp-port requires --mcp-config or a web_search profile",
    ):
        build_runtime_config_for_supervisor(
            args=_base_args(upstream="http://127.0.0.1:8457", mcp_port=6201),
            env=EnvSource(environ={}),
        )


def test_build_runtime_config_for_supervisor_allows_mcp_port_without_config_for_web_search() -> (
    None
):
    runtime_config = build_runtime_config_for_supervisor(
        args=_base_args(
            upstream="http://127.0.0.1:8457",
            web_search_profile="duckduckgo_plus_fetch",
            mcp_port=6201,
        ),
        env=EnvSource(environ={}),
    )

    assert runtime_config.mcp_config_path is None
    assert runtime_config.mcp_builtin_runtime_url == "http://127.0.0.1:6201"


def test_build_serve_spec_remote_upstream_has_no_spawned_vllm_plan() -> None:
    runtime_config = build_runtime_config_for_supervisor(
        args=_base_args(
            upstream="http://127.0.0.1:8457",
            code_interpreter="external",
            code_interpreter_port=6111,
        ),
        env=EnvSource(environ={}),
    )
    spec = build_serve_spec(runtime_config)
    assert isinstance(spec.upstream, ExternalUpstreamSpec)
    assert isinstance(spec.code_interpreter, ExternalCodeInterpreterSpec)
