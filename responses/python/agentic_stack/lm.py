from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncGenerator

from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    ModelHTTPError,
    UnexpectedModelBehavior,
    capture_run_messages,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import RetryConfig

from agentic_stack.configs.runtime import INTERNAL_UPSTREAM_HEADER_NAME, RuntimeConfig
from agentic_stack.lm_failures import (
    FailureCounters as _FailureCounters,
)
from agentic_stack.lm_failures import (
    classify_failure_log_level as _classify_failure_log_level,
)
from agentic_stack.lm_failures import (
    extract_failure_details as _extract_failure_details,
)
from agentic_stack.lm_failures import (
    log_failure_summary as _log_failure_summary,
)
from agentic_stack.mcp.runtime_client import BuiltinMcpRuntimeClient
from agentic_stack.mcp.types import McpToolRef
from agentic_stack.responses_core.composer import ResponseComposer
from agentic_stack.responses_core.normalizer import PydanticAINormalizer
from agentic_stack.responses_core.sse import stream_responses_sse
from agentic_stack.responses_core.store import ResponseStore, get_default_response_store
from agentic_stack.tools.ids import CODE_INTERPRETER_TOOL
from agentic_stack.tools.runtime import ToolRuntimeContext, bind_tool_runtime_context
from agentic_stack.tools.web_search.runtime import build_web_search_tool_runtime
from agentic_stack.types.openai import (
    AgentRunSettings,
    OpenAIResponsesError,
    OpenAIResponsesResponse,
    OpenAIResponsesResponseError,
    OpenAIResponsesStream,
    OpenAIResponsesStreamOutput,
    OpenAIResponsesStreamPart,
    OpenAIResponsesStreamText,
    vLLMResponsesRequest,
)
from agentic_stack.utils.exceptions import BadInputError
from agentic_stack.utils.io import get_async_client

LM_CLIENT = get_async_client()
INTEGRATED_LM_CLIENT = get_async_client()
INTEGRATED_LM_CLIENT.headers[INTERNAL_UPSTREAM_HEADER_NAME] = "1"


@dataclass(slots=True)
class ResponseRunContext:
    response: OpenAIResponsesResponse
    hydrated_request: vLLMResponsesRequest
    run_settings: AgentRunSettings
    mcp_tool_name_map: dict[str, McpToolRef]
    tool_runtime_context: ToolRuntimeContext
    normalizer: PydanticAINormalizer
    composer: ResponseComposer


def get_openai_provider(
    runtime_config: RuntimeConfig,
    base_url: str | None = None,
    *,
    api_key: str | None = None,
) -> OpenAIProvider:
    return OpenAIProvider(
        api_key=(runtime_config.openai_api_key or "") if api_key is None else api_key,
        base_url=runtime_config.llm_api_base if base_url is None else base_url,
        http_client=INTEGRATED_LM_CLIENT
        if runtime_config.runtime_mode == "integrated"
        else LM_CLIENT,
    )


class LMEngine:
    """Orchestrate one Responses request using vLLM Chat Completions via Pydantic AI.

    MVP staging notes:
    - keep the alpha tool execution model (Option A: code interpreter executed via `pydantic_ai` tool registration)
    - `previous_response_id` is supported via a shared ResponseStore (Stage 2)
    - move Responses contract correctness into `agentic_stack.responses_core` (Normalizer → Composer)
    """

    def __init__(
        self,
        body: vLLMResponsesRequest,
        *,
        retry_config: RetryConfig | None = None,
        store: ResponseStore | None = None,
        builtin_mcp_runtime_client: BuiltinMcpRuntimeClient | None = None,
        runtime_config: RuntimeConfig,
    ) -> None:
        self._body = body
        self._store = store or get_default_response_store()
        self._builtin_mcp_runtime_client = builtin_mcp_runtime_client
        self._runtime_config = runtime_config
        self._hydrated_body: vLLMResponsesRequest | None = None
        # NOTE: the installed `pydantic_ai` version in this repo does not accept `retry_config`
        # on `OpenAIProvider.__init__`. Keep the parameter for future use, but do not pass it.
        self._agent = Agent(
            OpenAIChatModel(
                model_name=body.model,
                provider=get_openai_provider(runtime_config),
            ),
            model_settings=body.as_openai_chat_settings(),
        )
        self._response: OpenAIResponsesResponse | None = None

    async def run(
        self,
    ) -> (
        AsyncGenerator[
            str,
            None,
        ]
        | OpenAIResponsesResponse
    ):
        if self._body.stream:
            return self._run_stream()
        return await self._run()

    async def _run_stream(
        self,
    ) -> AsyncGenerator[str, None]:
        async for frame in stream_responses_sse(
            self._tap_events(self._iter_responses_events_stream())
        ):
            yield frame

    async def _run(self) -> OpenAIResponsesResponse:
        async for chunk in self._iter_responses_events_non_stream():
            if isinstance(chunk, OpenAIResponsesStream) and chunk.type in {
                "response.completed",
                "response.incomplete",
            }:
                self._response = chunk.response
                if self._hydrated_body is not None:
                    await self._store.put_completed(
                        request=self._body,
                        hydrated_request=self._hydrated_body,
                        response=chunk.response,
                    )
        if self._response is None:
            raise BadInputError("No response generated from LMEngine.")
        return self._response

    async def _tap_events(
        self,
        events: AsyncGenerator[
            OpenAIResponsesStream
            | OpenAIResponsesStreamOutput
            | OpenAIResponsesStreamPart
            | OpenAIResponsesStreamText,
            None,
        ],
    ) -> AsyncGenerator[
        OpenAIResponsesStream
        | OpenAIResponsesStreamOutput
        | OpenAIResponsesStreamPart
        | OpenAIResponsesStreamText,
        None,
    ]:
        async for event in events:
            if isinstance(event, OpenAIResponsesStream) and event.type in {
                "response.completed",
                "response.incomplete",
            }:
                self._response = event.response
                if self._hydrated_body is not None:
                    await self._store.put_completed(
                        request=self._body,
                        hydrated_request=self._hydrated_body,
                        response=event.response,
                    )
            yield event

    async def _build_response_pipeline(
        self,
    ) -> ResponseRunContext:
        # Seed the response from request fields, but do not allow `None` request values
        # to clobber schema-required response defaults (e.g. `tools: []`, `truncation: "disabled"`).
        response = OpenAIResponsesResponse.model_validate(self._body.model_dump(exclude_none=True))

        hydrated_body = await self._store.rehydrate_request(request=self._body)
        self._hydrated_body = hydrated_body
        run_settings, builtin_tools, mcp_tool_name_map = await hydrated_body.as_run_settings(
            builtin_mcp_runtime_client=self._builtin_mcp_runtime_client,
            request_remote_enabled=self._runtime_config.mcp_request_remote_enabled,
            request_remote_url_checks_enabled=self._runtime_config.mcp_request_remote_url_checks,
        )
        builtin_tool_names = {t.name for t in builtin_tools}
        tool_runtime_context = ToolRuntimeContext(
            runtime_config=self._runtime_config,
            web_search=build_web_search_tool_runtime(
                request=hydrated_body,
                enabled_builtin_tool_names=builtin_tool_names,
                runtime_config=self._runtime_config,
                builtin_mcp_runtime_client=self._builtin_mcp_runtime_client,
            ),
        )

        normalizer = PydanticAINormalizer(
            builtin_tool_names=builtin_tool_names,
            code_interpreter_tool_name=CODE_INTERPRETER_TOOL,
            mcp_tool_name_map=mcp_tool_name_map,
        )
        include_set = set(hydrated_body.include or [])
        composer = ResponseComposer(response=response, include=include_set)
        return ResponseRunContext(
            response=response,
            hydrated_request=hydrated_body,
            run_settings=run_settings,
            mcp_tool_name_map=mcp_tool_name_map,
            tool_runtime_context=tool_runtime_context,
            normalizer=normalizer,
            composer=composer,
        )

    async def _iter_responses_events_non_stream(
        self,
    ) -> AsyncGenerator[
        OpenAIResponsesStream
        | OpenAIResponsesStreamOutput
        | OpenAIResponsesStreamPart
        | OpenAIResponsesStreamText,
        None,
    ]:
        # Non-stream mode: exceptions should propagate so the HTTP layer can return a non-200 response.
        # (There is no SSE stream to carry error events.)
        run_context = await self._build_response_pipeline()
        failure_counters = _FailureCounters()

        # Emit created/in_progress even if the upstream call fails immediately, to match the streaming contract.
        for chunk in run_context.composer.start():
            yield chunk

        with capture_run_messages() as messages:
            try:
                with bind_tool_runtime_context(run_context.tool_runtime_context):
                    async for event in self._agent.run_stream_events(
                        output_type=[self._agent.output_type, DeferredToolRequests],
                        message_history=run_context.run_settings["message_history"],
                        instructions=run_context.run_settings["instructions"],
                        toolsets=run_context.run_settings["toolsets"],
                        usage_limits=run_context.run_settings["usage_limits"],
                    ):
                        for normalized in run_context.normalizer.on_event(event):
                            failure_counters.observe(normalized)
                            for out in run_context.composer.feed(normalized):
                                yield out
            except (ModelHTTPError, UnexpectedModelBehavior) as e:
                details = _extract_failure_details(e)
                _log_failure_summary(
                    response_id=run_context.composer.response.id,
                    failure_phase="non_stream",
                    error_class=details.error_class,
                    log_level=_classify_failure_log_level(
                        error_class=details.error_class,
                        upstream_status_code=details.upstream_status_code,
                    ),
                    upstream_status_code=details.upstream_status_code,
                    error_message=details.message,
                    messages=messages,
                    counters=failure_counters,
                    upstream_error_raw=details.upstream_error_raw,
                    log_model_messages=self._runtime_config.log_model_messages,
                )
                raise

    async def _iter_responses_events_stream(
        self,
    ) -> AsyncGenerator[
        OpenAIResponsesStream
        | OpenAIResponsesStreamOutput
        | OpenAIResponsesStreamPart
        | OpenAIResponsesStreamText,
        None,
    ]:
        # Stream mode: convert upstream failures into Responses stream error ordering.
        run_context = await self._build_response_pipeline()
        failure_counters = _FailureCounters()

        # Emit created/in_progress even if the upstream call fails immediately, to match the streaming contract.
        for chunk in run_context.composer.start():
            yield chunk

        with capture_run_messages() as messages:
            try:
                with bind_tool_runtime_context(run_context.tool_runtime_context):
                    async for event in self._agent.run_stream_events(
                        output_type=[self._agent.output_type, DeferredToolRequests],
                        message_history=run_context.run_settings["message_history"],
                        instructions=run_context.run_settings["instructions"],
                        toolsets=run_context.run_settings["toolsets"],
                        usage_limits=run_context.run_settings["usage_limits"],
                    ):
                        for normalized in run_context.normalizer.on_event(event):
                            failure_counters.observe(normalized)
                            for out in run_context.composer.feed(normalized):
                                yield out
            except (ModelHTTPError, UnexpectedModelBehavior) as e:
                details = _extract_failure_details(e)
                _log_failure_summary(
                    response_id=run_context.response.id,
                    failure_phase="stream",
                    error_class=details.error_class,
                    log_level=_classify_failure_log_level(
                        error_class=details.error_class,
                        upstream_status_code=details.upstream_status_code,
                    ),
                    upstream_status_code=details.upstream_status_code,
                    error_message=details.message,
                    messages=messages,
                    counters=failure_counters,
                    upstream_error_raw=details.upstream_error_raw,
                    log_model_messages=self._runtime_config.log_model_messages,
                )
                yield OpenAIResponsesError(
                    code=details.code,
                    message=details.message,
                    param=details.param,
                    sequence_number=run_context.composer.alloc_sequence_number(),
                )
                run_context.response.error = OpenAIResponsesResponseError(
                    code=details.code,
                    message=details.message,
                )
                run_context.response.status = "failed"
                yield OpenAIResponsesStream(
                    type="response.failed",
                    response=run_context.response,
                    sequence_number=run_context.composer.alloc_sequence_number(),
                )
                return
