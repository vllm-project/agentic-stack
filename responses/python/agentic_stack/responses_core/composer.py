from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import time

from agentic_stack.responses_core.models import (
    CodeInterpreterCallCodeDelta,
    CodeInterpreterCallCodeDone,
    CodeInterpreterCallCompleted,
    CodeInterpreterCallInterpreting,
    CodeInterpreterCallStarted,
    FunctionCallArgumentsDelta,
    FunctionCallDone,
    FunctionCallStarted,
    McpCallArgumentsDelta,
    McpCallArgumentsDone,
    McpCallCompleted,
    McpCallFailed,
    McpCallStarted,
    MessageDelta,
    MessageDone,
    MessageStarted,
    NormalizedEvent,
    ReasoningDelta,
    ReasoningDone,
    ReasoningStarted,
    UsageFinal,
    WebSearchCallCompleted,
    WebSearchCallSearching,
    WebSearchCallStarted,
)
from agentic_stack.types.openai import (
    OpenAICodeOutputLog,
    OpenAICodeToolCall,
    OpenAIFunctionToolCall,
    OpenAIInputTokenDetails,
    OpenAIMcpToolCall,
    OpenAIOutputItem,
    OpenAIOutputTextContent,
    OpenAIOutputTokenDetails,
    OpenAIReasoningContent,
    OpenAIReasoningItem,
    OpenAIResponsesIncompleteDetails,
    OpenAIResponsesResponse,
    OpenAIResponsesStream,
    OpenAIResponsesStreamOutput,
    OpenAIResponsesStreamPart,
    OpenAIResponsesStreamText,
    OpenAIResponsesUsage,
    OpenAIWebSearchActionFindInPage,
    OpenAIWebSearchActionOpenPage,
    OpenAIWebSearchActionSearch,
    OpenAIWebSearchSource,
    OpenAIWebSearchToolCall,
    vLLMOutput,
)
from agentic_stack.utils import uuid7_str


@dataclass(slots=True)
class _ItemState:
    item_id: str
    output_index: int
    kind: str
    text: str = ""
    reasoning: str = ""
    function_name: str | None = None
    function_call_id: str | None = None
    function_args_json: str = ""
    code: str | None = None
    code_stdout: str | None = None
    code_stderr: str | None = None
    code_result: str | None = None
    container_id: str | None = None
    mcp_server_label: str | None = None
    mcp_name: str | None = None
    mcp_arguments_json: str = ""
    mcp_output: str | None = None
    mcp_error: str | None = None
    web_search_action_type: str | None = None
    web_search_query: str | None = None
    web_search_queries: tuple[str, ...] = ()
    web_search_sources: tuple[dict[str, str], ...] = ()
    web_search_url: str | None = None
    web_search_pattern: str | None = None


class ResponseComposer:
    """Compose a Responses SSE/JSON contract from NormalizedEvents.

    Stage-1 scope:
    - single-request composition (no `previous_response_id` store yet)
    - stable identity/ordering guarantees for stream reconstruction
    """

    def __init__(
        self, *, response: OpenAIResponsesResponse, include: set[str] | None = None
    ) -> None:
        self._response = response
        self._started = False
        self._sequence_number = 0
        self._next_output_index = 0
        self._items: dict[str, _ItemState] = {}
        self._output_items: list[vLLMOutput] = []
        self._reasoning_item: OpenAIReasoningItem | None = None
        self._reasoning_state: _ItemState | None = None
        include_set = include or set()
        self._include_code_interpreter_outputs = "code_interpreter_call.outputs" in include_set
        self._include_web_search_action_sources = "web_search_call.action.sources" in include_set

    @property
    def response(self) -> OpenAIResponsesResponse:
        return self._response

    def feed(
        self, event: NormalizedEvent
    ) -> Iterable[
        OpenAIResponsesStream
        | OpenAIResponsesStreamOutput
        | OpenAIResponsesStreamPart
        | OpenAIResponsesStreamText
    ]:
        if not self._started:
            raise RuntimeError("ResponseComposer.start() must be called before feed().")

        if isinstance(event, MessageStarted):
            yield from self._start_message(event)
        elif isinstance(event, MessageDelta):
            yield from self._message_delta(event)
        elif isinstance(event, MessageDone):
            yield from self._message_done(event)
        elif isinstance(event, ReasoningStarted):
            yield from self._start_reasoning(event)
        elif isinstance(event, ReasoningDelta):
            yield from self._reasoning_delta(event)
        elif isinstance(event, ReasoningDone):
            yield from self._reasoning_done(event)
        elif isinstance(event, FunctionCallStarted):
            yield from self._start_function_call(event)
        elif isinstance(event, FunctionCallArgumentsDelta):
            yield from self._function_args_delta(event)
        elif isinstance(event, FunctionCallDone):
            yield from self._function_done(event)
        elif isinstance(event, McpCallStarted):
            yield from self._start_mcp_call(event)
        elif isinstance(event, McpCallArgumentsDelta):
            yield from self._mcp_args_delta(event)
        elif isinstance(event, McpCallArgumentsDone):
            yield from self._mcp_args_done(event)
        elif isinstance(event, McpCallCompleted):
            yield from self._mcp_completed(event)
        elif isinstance(event, McpCallFailed):
            yield from self._mcp_failed(event)
        elif isinstance(event, CodeInterpreterCallStarted):
            yield from self._start_code_interpreter_call(event)
        elif isinstance(event, WebSearchCallStarted):
            yield from self._start_web_search_call(event)
        elif isinstance(event, WebSearchCallSearching):
            yield from self._web_search_searching(event)
        elif isinstance(event, WebSearchCallCompleted):
            yield from self._web_search_completed(event)
        elif isinstance(event, CodeInterpreterCallCodeDelta):
            yield from self._code_delta(event)
        elif isinstance(event, CodeInterpreterCallCodeDone):
            yield from self._code_done(event)
        elif isinstance(event, CodeInterpreterCallInterpreting):
            yield from self._code_interpreting(event)
        elif isinstance(event, CodeInterpreterCallCompleted):
            yield from self._code_completed(event)
        elif isinstance(event, UsageFinal):
            yield from self._complete_response(event)

    def start(
        self,
    ) -> Iterable[OpenAIResponsesStream | OpenAIResponsesStreamOutput]:
        if self._started:
            return []
        self._started = True
        # Match observed Responses streams: the created/in_progress lifecycle events contain a response whose
        # status is already `in_progress`.
        self._response.status = "in_progress"
        return [
            OpenAIResponsesStream(
                type="response.created",
                sequence_number=self.alloc_sequence_number(),
                response=self._response,
            ),
            OpenAIResponsesStream(
                type="response.in_progress",
                sequence_number=self.alloc_sequence_number(),
                response=self._response,
            ),
        ]

    def alloc_sequence_number(self) -> int:
        return self._incr_seq()

    def _incr_seq(self) -> int:
        current = self._sequence_number
        self._sequence_number += 1
        return current

    def _alloc_output_index(self) -> int:
        current = self._next_output_index
        self._next_output_index += 1
        return current

    def _start_message(self, event: MessageStarted) -> Iterable:
        item_id = uuid7_str("msg_")
        out_index = self._alloc_output_index()
        state = _ItemState(item_id=item_id, output_index=out_index, kind="message")
        self._items[event.item_key] = state

        yield OpenAIResponsesStreamOutput(
            type="response.output_item.added",
            sequence_number=self._incr_seq(),
            output_index=out_index,
            item=OpenAIOutputItem(content=[], id=item_id, status="in_progress"),
        )
        yield OpenAIResponsesStreamPart(
            type="response.content_part.added",
            item_id=item_id,
            sequence_number=self._incr_seq(),
            output_index=out_index,
            content_index=0,
            part=OpenAIOutputTextContent(text=""),
        )

    def _message_delta(self, event: MessageDelta) -> Iterable:
        state = self._items[event.item_key]
        state.text += event.delta
        yield OpenAIResponsesStreamText(
            type="response.output_text.delta",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            content_index=0,
            delta=event.delta,
            # OpenResponses schema requires `logprobs` to be present for output_text delta events.
            logprobs=[],
        )

    def _message_done(self, event: MessageDone) -> Iterable:
        state = self._items[event.item_key]
        state.text = event.text
        yield OpenAIResponsesStreamText(
            type="response.output_text.done",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            content_index=0,
            text=event.text,
            # OpenResponses schema requires `logprobs` to be present for output_text done events.
            logprobs=[],
        )
        yield OpenAIResponsesStreamPart(
            type="response.content_part.done",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            content_index=0,
            part=OpenAIOutputTextContent(text=event.text),
        )
        item = OpenAIOutputItem(
            content=[OpenAIOutputTextContent(text=event.text)],
            id=state.item_id,
            status="completed",
        )
        yield OpenAIResponsesStreamOutput(
            type="response.output_item.done",
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            item=item,
        )
        self._output_items.append(item)

    def _start_reasoning(self, event: ReasoningStarted) -> Iterable:
        # OpenAI parity: some models emit no reasoning at all. In that case OpenAI omits the `reasoning`
        # output item entirely (output only contains the assistant `message`). We therefore create the
        # `reasoning` output item only if the upstream produces a ThinkingPart.
        if self._reasoning_item is not None:
            # Upstream may emit multiple ThinkingPart segments in a single response. We map all of them
            # onto the single reasoning output item.
            if self._reasoning_state is None:
                raise RuntimeError("reasoning output item exists but reasoning state is missing")
            self._items[event.item_key] = self._reasoning_state
            return []

        item_id = uuid7_str("rs_")
        out_index = self._alloc_output_index()
        self._reasoning_item = OpenAIReasoningItem(id=item_id)
        self._output_items.append(self._reasoning_item)

        state = _ItemState(item_id=item_id, output_index=out_index, kind="reasoning")
        self._reasoning_state = state
        self._items[event.item_key] = state

        return [
            OpenAIResponsesStreamOutput(
                type="response.output_item.added",
                sequence_number=self._incr_seq(),
                output_index=out_index,
                item=self._reasoning_item,
            ),
            # OpenAI emits `output_item.done` for reasoning early (and streams reasoning deltas separately).
            OpenAIResponsesStreamOutput(
                type="response.output_item.done",
                sequence_number=self._incr_seq(),
                output_index=out_index,
                item=self._reasoning_item,
            ),
        ]

    def _reasoning_delta(self, event: ReasoningDelta) -> Iterable:
        state = self._items[event.item_key]
        state.reasoning += event.delta
        yield OpenAIResponsesStreamText(
            type="response.reasoning.delta",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            content_index=0,
            delta=event.delta,
        )

    def _reasoning_done(self, event: ReasoningDone) -> Iterable:
        state = self._items[event.item_key]
        state.reasoning = event.text
        yield OpenAIResponsesStreamText(
            type="response.reasoning.done",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            content_index=0,
            text=event.text,
        )
        # Keep `summary` empty for vLLM raw reasoning; `summary_text` is meant to be an actual summary.
        # Preserve raw reasoning in `content` as `reasoning_text`.
        for item in self._output_items:
            if isinstance(item, OpenAIReasoningItem) and item.id == state.item_id:
                item.summary = []
                if event.text:
                    item.content.append(OpenAIReasoningContent(text=event.text))
                break

    def _start_function_call(self, event: FunctionCallStarted) -> Iterable:
        item_id = uuid7_str("fc_")
        out_index = self._alloc_output_index()
        state = _ItemState(
            item_id=item_id,
            output_index=out_index,
            kind="function_call",
            function_name=event.name,
            function_call_id=event.call_id,
            function_args_json=event.initial_arguments_json,
        )
        self._items[event.item_key] = state

        yield OpenAIResponsesStreamOutput(
            type="response.output_item.added",
            sequence_number=self._incr_seq(),
            output_index=out_index,
            item=OpenAIFunctionToolCall(
                arguments=event.initial_arguments_json,
                call_id=event.call_id,
                name=event.name,
                id=item_id,
                status="in_progress",
            ),
        )

    def _function_args_delta(self, event: FunctionCallArgumentsDelta) -> Iterable:
        state = self._items[event.item_key]
        state.function_args_json += event.delta
        yield OpenAIResponsesStreamText(
            type="response.function_call_arguments.delta",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            delta=event.delta,
        )

    def _function_done(self, event: FunctionCallDone) -> Iterable:
        state = self._items[event.item_key]
        state.function_args_json = event.arguments_json
        yield OpenAIResponsesStreamText(
            type="response.function_call_arguments.done",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            arguments=event.arguments_json,
        )
        item = OpenAIFunctionToolCall(
            arguments=event.arguments_json,
            call_id=state.function_call_id or "",
            name=state.function_name or "",
            id=state.item_id,
            status="completed",
        )
        yield OpenAIResponsesStreamOutput(
            type="response.output_item.done",
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            item=item,
        )
        self._output_items.append(item)

    def _start_code_interpreter_call(self, event: CodeInterpreterCallStarted) -> Iterable:
        item_id = uuid7_str("ci_")
        out_index = self._alloc_output_index()
        container_id = uuid7_str("cntr_")
        state = _ItemState(
            item_id=item_id,
            output_index=out_index,
            kind="code_interpreter_call",
            code=event.initial_code,
            container_id=container_id,
        )
        self._items[event.item_key] = state

        yield OpenAIResponsesStreamOutput(
            type="response.output_item.added",
            sequence_number=self._incr_seq(),
            output_index=out_index,
            item=OpenAICodeToolCall(
                code=event.initial_code,
                container_id=container_id,
                id=item_id,
                status="in_progress",
            ),
        )
        yield OpenAIResponsesStreamText(
            type="response.code_interpreter_call.in_progress",
            item_id=item_id,
            sequence_number=self._incr_seq(),
            output_index=out_index,
        )

    def _code_delta(self, event: CodeInterpreterCallCodeDelta) -> Iterable:
        state = self._items[event.item_key]
        yield OpenAIResponsesStreamText(
            type="response.code_interpreter_call_code.delta",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            delta=event.delta,
        )

    def _code_done(self, event: CodeInterpreterCallCodeDone) -> Iterable:
        state = self._items[event.item_key]
        state.code = event.code
        yield OpenAIResponsesStreamText(
            type="response.code_interpreter_call_code.done",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            code=event.code,
        )

    def _code_interpreting(self, event: CodeInterpreterCallInterpreting) -> Iterable:
        state = self._items[event.item_key]
        yield OpenAIResponsesStreamText(
            type="response.code_interpreter_call.interpreting",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
        )

    def _code_completed(self, event: CodeInterpreterCallCompleted) -> Iterable:
        state = self._items[event.item_key]
        state.code_stdout = event.stdout
        state.code_stderr = event.stderr
        state.code_result = event.result
        yield OpenAIResponsesStreamText(
            type="response.code_interpreter_call.completed",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
        )
        outputs = None
        if self._include_code_interpreter_outputs:
            outs: list[OpenAICodeOutputLog] = []
            stdio = (event.stdout or "") + (event.stderr or "")
            if stdio:
                outs.append(OpenAICodeOutputLog(logs=stdio))
            if event.result:
                outs.append(OpenAICodeOutputLog(logs=event.result))
            outputs = outs or None
        item = OpenAICodeToolCall(
            code=state.code,
            container_id=state.container_id or uuid7_str("cntr_"),
            id=state.item_id,
            status="completed",
            outputs=outputs,
        )
        yield OpenAIResponsesStreamOutput(
            type="response.output_item.done",
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            item=item,
        )
        self._output_items.append(item)

    def _start_web_search_call(self, event: WebSearchCallStarted) -> Iterable:
        item_id = uuid7_str("ws_")
        out_index = self._alloc_output_index()
        state = _ItemState(
            item_id=item_id,
            output_index=out_index,
            kind="web_search_call",
        )
        self._items[event.item_key] = state

        yield OpenAIResponsesStreamOutput(
            type="response.output_item.added",
            sequence_number=self._incr_seq(),
            output_index=out_index,
            item=OpenAIWebSearchToolCall(
                id=item_id,
                status="in_progress",
            ),
        )
        yield OpenAIResponsesStreamText(
            type="response.web_search_call.in_progress",
            item_id=item_id,
            sequence_number=self._incr_seq(),
            output_index=out_index,
        )

    def _web_search_searching(self, event: WebSearchCallSearching) -> Iterable:
        state = self._items[event.item_key]
        yield OpenAIResponsesStreamText(
            type="response.web_search_call.searching",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
        )

    def _web_search_completed(self, event: WebSearchCallCompleted) -> Iterable:
        state = self._items[event.item_key]
        state.web_search_action_type = event.action_type
        state.web_search_query = event.query
        state.web_search_queries = event.queries
        state.web_search_sources = event.sources
        state.web_search_url = event.url
        state.web_search_pattern = event.pattern
        yield OpenAIResponsesStreamText(
            type="response.web_search_call.completed",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
        )
        if event.action_type == "search":
            item_action = OpenAIWebSearchActionSearch(
                query=event.query or "",
                queries=list(event.queries) or None,
                sources=(
                    [OpenAIWebSearchSource.model_validate(source) for source in event.sources]
                    if self._include_web_search_action_sources and event.sources
                    else None
                ),
            )
        elif event.action_type == "open_page":
            item_action = OpenAIWebSearchActionOpenPage(url=event.url)
        else:
            item_action = OpenAIWebSearchActionFindInPage(
                url=event.url,
                pattern=event.pattern or "",
            )
        item = OpenAIWebSearchToolCall(
            id=state.item_id,
            status="completed",
            action=item_action,
        )
        yield OpenAIResponsesStreamOutput(
            type="response.output_item.done",
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            item=item,
        )
        self._output_items.append(item)

    def _start_mcp_call(self, event: McpCallStarted) -> Iterable:
        item_id = uuid7_str("mcp_")
        out_index = self._alloc_output_index()
        state = _ItemState(
            item_id=item_id,
            output_index=out_index,
            kind="mcp_call",
            mcp_server_label=event.server_label,
            mcp_name=event.name,
            mcp_arguments_json=event.initial_arguments_json,
        )
        self._items[event.item_key] = state

        yield OpenAIResponsesStreamOutput(
            type="response.output_item.added",
            sequence_number=self._incr_seq(),
            output_index=out_index,
            item=OpenAIMcpToolCall(
                id=item_id,
                server_label=event.server_label,
                name=event.name,
                arguments=event.initial_arguments_json,
                status="in_progress",
            ),
        )
        yield OpenAIResponsesStreamText(
            type="response.mcp_call.in_progress",
            item_id=item_id,
            sequence_number=self._incr_seq(),
            output_index=out_index,
        )

    def _mcp_args_delta(self, event: McpCallArgumentsDelta) -> Iterable:
        state = self._items[event.item_key]
        state.mcp_arguments_json += event.delta
        yield OpenAIResponsesStreamText(
            type="response.mcp_call_arguments.delta",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            delta=event.delta,
        )

    def _mcp_args_done(self, event: McpCallArgumentsDone) -> Iterable:
        state = self._items[event.item_key]
        state.mcp_arguments_json = event.arguments_json
        yield OpenAIResponsesStreamText(
            type="response.mcp_call_arguments.done",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            arguments=event.arguments_json,
        )

    def _mcp_completed(self, event: McpCallCompleted) -> Iterable:
        state = self._items[event.item_key]
        state.mcp_output = event.output_text
        state.mcp_error = None
        yield OpenAIResponsesStreamText(
            type="response.mcp_call.completed",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            output=event.output_text,
        )
        item = OpenAIMcpToolCall(
            id=state.item_id,
            server_label=state.mcp_server_label or "",
            name=state.mcp_name or "",
            arguments=state.mcp_arguments_json,
            status="completed",
            output=event.output_text,
        )
        yield OpenAIResponsesStreamOutput(
            type="response.output_item.done",
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            item=item,
        )
        self._output_items.append(item)

    def _mcp_failed(self, event: McpCallFailed) -> Iterable:
        state = self._items[event.item_key]
        state.mcp_output = None
        state.mcp_error = event.error_text
        # OpenAI MCP stream parity + SDK compatibility:
        # keep `response.mcp_call.failed` as metadata-only and carry failure text on the final
        # `mcp_call` output item (`item.error`). A top-level `error` here is treated by the
        # OpenAI SDK stream parser as a stream-fatal API error.
        yield OpenAIResponsesStreamText(
            type="response.mcp_call.failed",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
        )
        item = OpenAIMcpToolCall(
            id=state.item_id,
            server_label=state.mcp_server_label or "",
            name=state.mcp_name or "",
            arguments=state.mcp_arguments_json,
            status="failed",
            error=event.error_text,
        )
        yield OpenAIResponsesStreamOutput(
            type="response.output_item.done",
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            item=item,
        )
        self._output_items.append(item)

    def _complete_response(self, event: UsageFinal) -> Iterable:
        self._response.usage = OpenAIResponsesUsage(
            input_tokens=event.input_tokens,
            input_tokens_details=OpenAIInputTokenDetails(cached_tokens=event.cache_read_tokens),
            output_tokens=event.output_tokens,
            output_tokens_details=OpenAIOutputTokenDetails(
                reasoning_tokens=event.reasoning_tokens
            ),
            total_tokens=event.total_tokens,
        )
        self._response.output = list(self._output_items)
        if event.incomplete_reason is not None:
            self._response.status = "incomplete"
            self._response.incomplete_details = OpenAIResponsesIncompleteDetails(
                reason=event.incomplete_reason
            )
            self._response.completed_at = None
            yield OpenAIResponsesStream(
                type="response.incomplete",
                sequence_number=self._incr_seq(),
                response=self._response,
            )
            return

        self._response.status = "completed"
        self._response.incomplete_details = None
        self._response.completed_at = int(time())
        yield OpenAIResponsesStream(
            type="response.completed",
            sequence_number=self._incr_seq(),
            response=self._response,
        )
