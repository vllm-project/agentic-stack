from __future__ import annotations

import pytest
from pydantic_ai import BuiltinToolCallPart, BuiltinToolReturnPart, ModelResponse

from agentic_stack.configs.builders import (
    RuntimeConfigError,
    build_runtime_config_for_standalone,
)
from agentic_stack.configs.sources import EnvSource
from agentic_stack.mcp.types import McpExecutionResult
from agentic_stack.tools.base.runtime import bind_runtime_requirements
from agentic_stack.tools.profile_resolution import resolve_profiled_builtin_tool
from agentic_stack.tools.web_search.adapters import WEB_SEARCH_ADAPTER_SPECS
from agentic_stack.tools.web_search.config import ResolvedWebSearchRequestConfig
from agentic_stack.tools.web_search.executor import WebSearchExecutor
from agentic_stack.tools.web_search.page_cache import WebSearchPageCache
from agentic_stack.tools.web_search.runtime import build_web_search_tool_runtime
from agentic_stack.tools.web_search.types import WebSearchActionRequest
from agentic_stack.types.openai import vLLMResponsesRequest
from agentic_stack.utils.exceptions import BadInputError


class _FakeBuiltinMcpRuntimeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    async def call_tool(
        self,
        *,
        server_label: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> McpExecutionResult:
        self.calls.append((server_label, tool_name, arguments))
        if tool_name == "web_search_exa":
            return McpExecutionResult(
                ok=True,
                output_text=(
                    "Title: Example Result\n"
                    "URL: https://example.com/a?utm=1\n"
                    "Text: Example snippet\n\n"
                    "Title: Other Result\n"
                    "URL: https://other.com/b\n"
                    "Text: Other snippet"
                ),
                error_text=None,
            )
        if tool_name == "crawling_exa":
            return McpExecutionResult(
                ok=True,
                output_text="Needle in a haystack. Another needle appears.",
                error_text=None,
            )
        if tool_name == "fetch":
            if arguments["url"] == "https://example.com/empty":
                return McpExecutionResult(
                    ok=True,
                    output_text=f"Contents of {arguments['url']}:\n",
                    error_text=None,
                )
            return McpExecutionResult(
                ok=True,
                output_text=(
                    f"Contents of {arguments['url']}:\n"
                    "Needle in a haystack. Another needle appears."
                ),
                error_text=None,
            )
        raise AssertionError(f"unexpected tool_name: {tool_name}")


def _build_adapter_by_action(resolved_tool) -> dict[str, object]:
    return {
        binding.action_name: WEB_SEARCH_ADAPTER_SPECS[binding.adapter_id].build_adapter()
        for binding in resolved_tool.action_bindings
    }


def _build_executor(
    *,
    runtime_client: _FakeBuiltinMcpRuntimeClient | None,
    profile_id: str = "exa_mcp",
    allowed_domains: tuple[str, ...] = (),
) -> WebSearchExecutor:
    resolved_tool = resolve_profiled_builtin_tool(tool_type="web_search", profile_id=profile_id)
    bound_requirements = bind_runtime_requirements(
        resolved_tool=resolved_tool,
        builtin_mcp_runtime_client=runtime_client,
    )
    return WebSearchExecutor(
        request_config=ResolvedWebSearchRequestConfig(
            profile_id=profile_id,
            allowed_domains=allowed_domains,
            search_context_size="medium",
            user_location=None,
        ),
        resolved_tool=resolved_tool,
        bound_requirements=bound_requirements,
        adapter_by_action=_build_adapter_by_action(resolved_tool),
        page_cache=WebSearchPageCache(),
    )


def test_resolve_profiled_builtin_tool_is_import_safe_without_runtime_registry_bootstrap() -> None:
    resolved_tool = resolve_profiled_builtin_tool(
        tool_type="web_search",
        profile_id="duckduckgo_plus_fetch",
    )

    assert resolved_tool.tool_type == "web_search"
    assert resolved_tool.profile_id == "duckduckgo_plus_fetch"
    assert {binding.action_name for binding in resolved_tool.action_bindings} == {
        "search",
        "open_page",
    }


def test_resolve_profiled_builtin_tool_does_not_mutate_runtime_registry_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_stack.tools as tools_mod

    tools: dict[str, object] = {}
    monkeypatch.setattr(tools_mod, "TOOLS", tools)

    resolved_tool = resolve_profiled_builtin_tool(tool_type="web_search", profile_id="exa_mcp")

    assert resolved_tool.profile_id == "exa_mcp"
    assert tools == {}


def test_web_search_planning_validates_all_shipped_profiles_before_first_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import dataclass

    import agentic_stack.tools.web_search.profiles as profiles_mod
    from agentic_stack.tools.base.types import ActionBindingSpec

    profiles_mod.validate_web_search_planning_descriptors.cache_clear()
    original_profiles = profiles_mod._WEB_SEARCH_PROFILES.copy()

    @dataclass(frozen=True, slots=True)
    class _BrokenProfile:
        profile_id: str = "broken_profile"
        action_bindings: tuple[ActionBindingSpec, ...] = (
            ActionBindingSpec(action_name="search", adapter_id="missing_adapter"),
        )

    monkeypatch.setattr(
        profiles_mod,
        "_WEB_SEARCH_PROFILES",
        {**original_profiles, "broken_profile": _BrokenProfile()},
    )

    with pytest.raises(RuntimeError, match="missing_adapter"):
        resolve_profiled_builtin_tool(tool_type="web_search", profile_id="exa_mcp")


@pytest.mark.anyio
@pytest.mark.parametrize("tool_type", ["web_search_preview", "web_search_2025_08_26"])
async def test_as_run_settings_admits_web_search_tool_alias(tool_type: str) -> None:
    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": tool_type}],
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": tool_type}],
            },
        }
    )

    _run_settings, builtin_tools, _mcp_map = await req.as_run_settings(
        builtin_mcp_runtime_client=None,
        request_remote_enabled=True,
        request_remote_url_checks_enabled=True,
    )

    assert len(builtin_tools) == 1
    assert builtin_tools[0].name == "web_search"


@pytest.mark.anyio
async def test_allowed_tools_filters_builtin_tools() -> None:
    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "code_interpreter"}, {"type": "web_search"}],
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "web_search"}],
            },
        }
    )

    _run_settings, builtin_tools, _mcp_map = await req.as_run_settings(
        builtin_mcp_runtime_client=None,
        request_remote_enabled=True,
        request_remote_url_checks_enabled=True,
    )

    assert [tool.name for tool in builtin_tools] == ["web_search"]


@pytest.mark.anyio
async def test_allowed_tools_required_adds_internal_instruction() -> None:
    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "instructions": "Answer carefully.",
            "tools": [{"type": "code_interpreter"}, {"type": "web_search"}],
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "web_search"}],
            },
        }
    )

    run_settings, builtin_tools, _mcp_map = await req.as_run_settings(
        builtin_mcp_runtime_client=None,
        request_remote_enabled=True,
        request_remote_url_checks_enabled=True,
    )

    assert [tool.name for tool in builtin_tools] == ["web_search"]
    assert run_settings["instructions"] == (
        "Answer carefully.\n\n"
        "You must call at least one of the allowed tools before producing the final answer. "
        "Do not answer directly without a tool call."
    )


@pytest.mark.anyio
async def test_hosted_tool_choice_requires_effective_builtin() -> None:
    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "code_interpreter"}],
            "tool_choice": {"type": "web_search"},
        }
    )

    with pytest.raises(BadInputError, match="not present in effective tools"):
        await req.as_run_settings(
            builtin_mcp_runtime_client=None,
            request_remote_enabled=True,
            request_remote_url_checks_enabled=True,
        )


@pytest.mark.anyio
@pytest.mark.parametrize("tool_choice_type", ["web_search_preview", "web_search_2025_08_26"])
async def test_hosted_tool_choice_adds_internal_instruction(tool_choice_type: str) -> None:
    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "web_search"}],
            "tool_choice": {"type": tool_choice_type},
        }
    )

    run_settings, builtin_tools, _mcp_map = await req.as_run_settings(
        builtin_mcp_runtime_client=None,
        request_remote_enabled=True,
        request_remote_url_checks_enabled=True,
    )

    assert [tool.name for tool in builtin_tools] == ["web_search"]
    assert run_settings["instructions"] == (
        "You must call the `web_search` tool before producing the final answer. "
        "Do not answer directly without calling it."
    )


@pytest.mark.anyio
async def test_function_tool_choice_requires_effective_function() -> None:
    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "web_search"}],
            "tool_choice": {"type": "function", "name": "get_weather"},
        }
    )

    with pytest.raises(BadInputError, match="not present in effective tools"):
        await req.as_run_settings(
            builtin_mcp_runtime_client=None,
            request_remote_enabled=True,
            request_remote_url_checks_enabled=True,
        )


@pytest.mark.anyio
async def test_as_run_settings_replays_completed_web_search_call() -> None:
    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [
                {
                    "id": "ws_123",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "example query",
                        "sources": [{"type": "url", "url": "https://example.com/a"}],
                    },
                }
            ],
            "tool_choice": "none",
        }
    )

    run_settings, _builtin_tools, _mcp_map = await req.as_run_settings(
        builtin_mcp_runtime_client=None,
        request_remote_enabled=True,
        request_remote_url_checks_enabled=True,
    )

    tool_call_part: BuiltinToolCallPart | None = None
    tool_return_part: BuiltinToolReturnPart | None = None
    for message in run_settings["message_history"] or []:
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if isinstance(part, BuiltinToolCallPart):
                tool_call_part = part
            elif isinstance(part, BuiltinToolReturnPart):
                tool_return_part = part

    assert tool_call_part is not None
    assert tool_return_part is not None
    assert tool_call_part.tool_name == "web_search"
    assert tool_call_part.tool_call_id == "ws_123"
    assert tool_call_part.args == {
        "type": "search",
        "query": "example query",
        "sources": [{"type": "url", "url": "https://example.com/a"}],
    }
    assert tool_return_part.content == {
        "status": "completed",
        "action": {
            "type": "search",
            "query": "example query",
            "sources": [{"type": "url", "url": "https://example.com/a"}],
        },
    }


def test_build_request_runtime_requires_enabled_profile() -> None:
    runtime_config = build_runtime_config_for_standalone(env=EnvSource(environ={}))
    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "web_search"}],
        }
    )

    with pytest.raises(BadInputError, match="web_search"):
        build_web_search_tool_runtime(
            request=req,
            enabled_builtin_tool_names={"web_search"},
            runtime_config=runtime_config,
            builtin_mcp_runtime_client=None,
        )


def test_build_request_runtime_skips_web_search_when_not_effective() -> None:
    runtime_config = build_runtime_config_for_standalone(env=EnvSource(environ={}))
    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "web_search"}],
            "tool_choice": "none",
        }
    )

    runtime = build_web_search_tool_runtime(
        request=req,
        enabled_builtin_tool_names=set(),
        runtime_config=runtime_config,
        builtin_mcp_runtime_client=None,
    )

    assert runtime is None


def test_build_request_runtime_rejects_unknown_profile_early() -> None:
    with pytest.raises(
        RuntimeConfigError,
        match=(
            "unknown --web-search-profile='missing_profile'; expected one of: "
            "duckduckgo_plus_fetch, exa_mcp"
        ),
    ):
        build_runtime_config_for_standalone(
            env=EnvSource(environ={"VR_WEB_SEARCH_PROFILE": "missing_profile"})
        )


def test_build_request_runtime_requires_bound_mcp_runtime_for_exa_profile() -> None:
    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(environ={"VR_WEB_SEARCH_PROFILE": "exa_mcp"})
    )
    req = vLLMResponsesRequest.model_validate(
        {
            "model": "some-model",
            "stream": False,
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "web_search"}],
        }
    )

    with pytest.raises(BadInputError, match="Built-in MCP runtime is required"):
        build_web_search_tool_runtime(
            request=req,
            enabled_builtin_tool_names={"web_search"},
            runtime_config=runtime_config,
            builtin_mcp_runtime_client=None,
        )


@pytest.mark.anyio
async def test_duckduckgo_plus_fetch_profile_search_open_page_and_find_in_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_stack.tools.web_search.adapters.duckduckgo_common as duckduckgo_adapter

    async def _fake_run_duckduckgo_search(*, query: str, max_results: int) -> list[dict[str, str]]:
        assert query == "example query"
        assert max_results == 8
        return [
            {
                "href": "https://example.com/a?utm=1",
                "title": "Example Result",
                "body": "Example snippet",
            },
            {
                "href": "https://evil.com/b",
                "title": "Filtered Result",
                "body": "Filtered snippet",
            },
        ]

    monkeypatch.setattr(duckduckgo_adapter, "_run_duckduckgo_search", _fake_run_duckduckgo_search)
    runtime_client = _FakeBuiltinMcpRuntimeClient()
    executor = _build_executor(
        runtime_client=runtime_client,
        profile_id="duckduckgo_plus_fetch",
        allowed_domains=("example.com",),
    )

    search_result = await executor.execute(
        WebSearchActionRequest(action="search", query="example query")
    )
    assert search_result.action.type == "search"
    assert search_result.action.query == "example query"
    assert [source.url for source in search_result.action.sources] == [
        "https://example.com/a?utm=1"
    ]
    assert [result.url for result in search_result.results] == ["https://example.com/a?utm=1"]

    open_page_result = await executor.execute(
        WebSearchActionRequest(action="open_page", url="https://example.com/a?utm=1")
    )
    assert open_page_result.action.type == "open_page"
    assert open_page_result.page.url == "https://example.com/a?utm=1"
    assert open_page_result.page.text == "Needle in a haystack. Another needle appears."

    find_in_page_result = await executor.execute(
        WebSearchActionRequest(
            action="find_in_page",
            url="https://example.com/a?utm=1",
            pattern="needle",
        )
    )
    assert len(find_in_page_result.matches) == 2
    assert [tool_name for _, tool_name, _ in runtime_client.calls] == ["fetch"]


@pytest.mark.anyio
async def test_web_search_executor_warns_only_for_ignored_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_stack.tools.web_search.adapters.duckduckgo_common as duckduckgo_adapter
    import agentic_stack.tools.web_search.executor as executor_mod

    warnings: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        executor_mod.logger, "warning", lambda *args, **kwargs: warnings.append(args)
    )

    async def _fake_run_duckduckgo_search(*, query: str, max_results: int) -> list[dict[str, str]]:
        assert query == "example query"
        assert max_results == 12
        return [{"href": "https://example.com/a", "title": "Example", "body": "snippet"}]

    monkeypatch.setattr(duckduckgo_adapter, "_run_duckduckgo_search", _fake_run_duckduckgo_search)
    resolved_tool = resolve_profiled_builtin_tool(
        tool_type="web_search",
        profile_id="duckduckgo_plus_fetch",
    )
    executor = WebSearchExecutor(
        request_config=ResolvedWebSearchRequestConfig(
            profile_id="duckduckgo_plus_fetch",
            allowed_domains=(),
            search_context_size="high",
            user_location={"country": "US", "city": None, "region": None, "timezone": None},
        ),
        resolved_tool=resolved_tool,
        bound_requirements=bind_runtime_requirements(
            resolved_tool=resolved_tool,
            builtin_mcp_runtime_client=_FakeBuiltinMcpRuntimeClient(),
        ),
        adapter_by_action=_build_adapter_by_action(resolved_tool),
        page_cache=WebSearchPageCache(),
    )

    await executor.execute(WebSearchActionRequest(action="search", query="example query"))

    assert len(warnings) == 1
    assert "user_location is ignored" in str(warnings[0][0])


@pytest.mark.anyio
async def test_web_search_executor_warns_once_when_adapter_ignores_all_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentic_stack.tools.web_search.executor as executor_mod

    warnings: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        executor_mod.logger, "warning", lambda *args, **kwargs: warnings.append(args)
    )

    runtime_client = _FakeBuiltinMcpRuntimeClient()
    resolved_tool = resolve_profiled_builtin_tool(tool_type="web_search", profile_id="exa_mcp")
    executor = WebSearchExecutor(
        request_config=ResolvedWebSearchRequestConfig(
            profile_id="exa_mcp",
            allowed_domains=(),
            search_context_size="high",
            user_location={"country": "US", "city": None, "region": None, "timezone": None},
        ),
        resolved_tool=resolved_tool,
        bound_requirements=bind_runtime_requirements(
            resolved_tool=resolved_tool,
            builtin_mcp_runtime_client=runtime_client,
        ),
        adapter_by_action=_build_adapter_by_action(resolved_tool),
        page_cache=WebSearchPageCache(),
    )

    await executor.execute(WebSearchActionRequest(action="search", query="example query"))
    await executor.execute(WebSearchActionRequest(action="search", query="example query 2"))

    assert len(warnings) == 2
    assert "user_location is ignored" in str(warnings[0][0])
    assert "search_context_size=" in str(warnings[1][0])


@pytest.mark.anyio
async def test_web_search_executor_search_open_page_and_find_in_page() -> None:
    runtime_client = _FakeBuiltinMcpRuntimeClient()
    executor = _build_executor(runtime_client=runtime_client, allowed_domains=("example.com",))

    search_result = await executor.execute(
        WebSearchActionRequest(action="search", query="example query")
    )
    assert search_result.action.type == "search"
    assert search_result.action.query == "example query"
    assert search_result.action.sources is not None
    assert [source.url for source in search_result.action.sources] == [
        "https://example.com/a?utm=1"
    ]
    assert [result.url for result in search_result.results] == ["https://example.com/a?utm=1"]

    open_page_result = await executor.execute(
        WebSearchActionRequest(action="open_page", url="https://example.com/a?utm=1")
    )
    assert open_page_result.action.type == "open_page"
    assert open_page_result.page.url == "https://example.com/a?utm=1"
    assert open_page_result.page.text == "Needle in a haystack. Another needle appears."

    find_in_page_result = await executor.execute(
        WebSearchActionRequest(
            action="find_in_page",
            url="https://example.com/a?utm=1",
            pattern="needle",
        )
    )
    assert find_in_page_result.action.type == "find_in_page"
    assert find_in_page_result.action.url == "https://example.com/a?utm=1"
    assert len(find_in_page_result.matches) == 2
    assert all("needle" in match.text.casefold() for match in find_in_page_result.matches)

    assert [tool_name for _, tool_name, _ in runtime_client.calls] == [
        "web_search_exa",
        "crawling_exa",
    ]


@pytest.mark.anyio
async def test_web_search_executor_blocks_open_page_outside_allowlist() -> None:
    runtime_client = _FakeBuiltinMcpRuntimeClient()
    executor = _build_executor(runtime_client=runtime_client, allowed_domains=("example.com",))

    result = await executor.execute(
        WebSearchActionRequest(action="open_page", url="https://evil.com/article")
    )

    assert result.action.url is None
    assert result.page.url is None
    assert result.error == "URL is not allowed by `filters.allowed_domains`."
    assert runtime_client.calls == []


@pytest.mark.anyio
async def test_web_search_executor_find_in_page_uses_request_local_cache_only() -> None:
    runtime_client = _FakeBuiltinMcpRuntimeClient()
    executor = _build_executor(runtime_client=runtime_client, allowed_domains=("example.com",))

    result = await executor.execute(
        WebSearchActionRequest(
            action="find_in_page",
            url="https://example.com/a?utm=1",
            pattern="needle",
        )
    )

    assert result.matches == []
    assert result.error == "Page is not available in the request-local web_search page cache."
    assert runtime_client.calls == []


@pytest.mark.anyio
async def test_web_search_executor_find_in_page_accepts_cached_empty_page_text() -> None:
    runtime_client = _FakeBuiltinMcpRuntimeClient()
    executor = _build_executor(
        runtime_client=runtime_client,
        profile_id="duckduckgo_plus_fetch",
        allowed_domains=("example.com",),
    )

    open_page_result = await executor.execute(
        WebSearchActionRequest(action="open_page", url="https://example.com/empty")
    )

    assert open_page_result.action.type == "open_page"
    assert open_page_result.page.url == "https://example.com/empty"
    assert open_page_result.page.text == ""
    assert open_page_result.error is None

    find_in_page_result = await executor.execute(
        WebSearchActionRequest(
            action="find_in_page",
            url="https://example.com/empty",
            pattern="needle",
        )
    )

    assert find_in_page_result.action.type == "find_in_page"
    assert find_in_page_result.action.url == "https://example.com/empty"
    assert find_in_page_result.matches == []
    assert find_in_page_result.error is None
    assert [tool_name for _, tool_name, _ in runtime_client.calls] == ["fetch"]
