from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import (
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RunUsage,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)

from agentic_stack.mcp.types import McpToolRef
from agentic_stack.mcp.utils import parse_mcp_tool_result_payload, truncate_error_text
from agentic_stack.observability.metrics import record_tool_call_requested
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
from agentic_stack.tools.ids import WEB_SEARCH_TOOL
from agentic_stack.tools.web_search.types import (
    FindInPageActionPublic,
    OpenPageActionPublic,
    SearchActionPublic,
    parse_web_search_tool_result,
)


class PydanticAINormalizer:
    """Convert `pydantic_ai` stream events into internal NormalizedEvents.

    Stage-1 scope:
    - No `previous_response_id` rehydration (handled later in Layer 4).
    - Preserve the alpha execution model for built-ins (Option A): code interpreter executed via `pydantic_ai` tools.
    """

    def __init__(
        self,
        *,
        builtin_tool_names: set[str],
        code_interpreter_tool_name: str,
        mcp_tool_name_map: dict[str, McpToolRef] | None = None,
    ) -> None:
        self._builtin_tool_names = builtin_tool_names
        self._code_interpreter_tool_name = code_interpreter_tool_name
        self._mcp_tool_name_map = mcp_tool_name_map or {}

        self._index_to_item_key: dict[int, str] = {}
        self._tool_call_id_to_item_key: dict[str, str] = {}
        self._item_kind: dict[str, str] = {}
        self._code_arg_extractors: dict[str, _CodeJsonArgsExtractor] = {}

    def on_event(self, event) -> Iterable[NormalizedEvent]:
        if isinstance(event, PartStartEvent):
            return list(self._on_part_start(event))
        if isinstance(event, PartDeltaEvent):
            return list(self._on_part_delta(event))
        if isinstance(event, PartEndEvent):
            return list(self._on_part_end(event))
        if isinstance(event, FunctionToolCallEvent):
            return list(self._on_function_tool_call(event))
        if isinstance(event, FunctionToolResultEvent):
            return list(self._on_function_tool_result(event))
        if isinstance(event, AgentRunResultEvent):
            return [self._usage_final(event)]
        return []

    def _on_part_start(self, event: PartStartEvent) -> Iterable[NormalizedEvent]:
        part = event.part
        item_key = f"part:{event.index}"
        self._index_to_item_key[event.index] = item_key

        if isinstance(part, TextPart):
            self._item_kind[item_key] = "message"
            yield MessageStarted(item_key=item_key)
            if part.content:
                yield MessageDelta(item_key=item_key, delta=part.content)
            return

        if isinstance(part, ThinkingPart):
            self._item_kind[item_key] = "reasoning"
            yield ReasoningStarted(item_key=item_key)
            if part.content:
                yield ReasoningDelta(item_key=item_key, delta=part.content)
            return

        if isinstance(part, ToolCallPart):
            tool_name = part.tool_name
            self._tool_call_id_to_item_key[part.tool_call_id] = item_key
            if tool_name in self._mcp_tool_name_map:
                ref = self._mcp_tool_name_map[tool_name]
                record_tool_call_requested("mcp")
                self._item_kind[item_key] = "mcp_call"
                yield McpCallStarted(
                    item_key=item_key,
                    server_label=ref.server_label,
                    name=ref.tool_name,
                    initial_arguments_json="",
                    mode=ref.mode,
                )
                if part.args_as_json_str():
                    yield McpCallArgumentsDelta(item_key=item_key, delta=part.args_as_json_str())
                return

            if tool_name in self._builtin_tool_names:
                if tool_name == WEB_SEARCH_TOOL:
                    record_tool_call_requested("web_search")
                    self._item_kind[item_key] = "web_search_call"
                    yield WebSearchCallStarted(item_key=item_key)
                    return
                if tool_name != self._code_interpreter_tool_name:
                    return
                record_tool_call_requested("code_interpreter")
                self._item_kind[item_key] = "code_interpreter_call"
                # Parity note: OpenAI Responses streams code deltas as raw code text. In our vLLM+ChatCompletions path,
                # the model emits tool-call argument JSON fragments (e.g. `{"code":"print(1)"}`) via OpenAI Chat
                # Completions streaming. We extract `"code"` string content incrementally and emit it as code deltas.
                yield CodeInterpreterCallStarted(item_key=item_key, initial_code="")
                self._code_arg_extractors[item_key] = _CodeJsonArgsExtractor(target_key="code")
                initial_args = part.args_as_json_str()
                if initial_args and initial_args != "{}":
                    if code_delta := self._code_arg_extractors[item_key].feed(initial_args):
                        yield CodeInterpreterCallCodeDelta(item_key=item_key, delta=code_delta)
                return

            record_tool_call_requested("function")
            self._item_kind[item_key] = "function_call"
            yield FunctionCallStarted(
                item_key=item_key,
                call_id=part.tool_call_id,
                name=tool_name,
                initial_arguments_json="",
            )
            if part.args_as_json_str():
                yield FunctionCallArgumentsDelta(item_key=item_key, delta=part.args_as_json_str())
            return

    def _on_part_delta(self, event: PartDeltaEvent) -> Iterable[NormalizedEvent]:
        item_key = self._index_to_item_key.get(event.index)
        if item_key is None:
            return

        delta = event.delta
        if isinstance(delta, TextPartDelta):
            if delta.content_delta:
                yield MessageDelta(item_key=item_key, delta=delta.content_delta)
            return

        if isinstance(delta, ThinkingPartDelta):
            if delta.content_delta:
                yield ReasoningDelta(item_key=item_key, delta=delta.content_delta)
            return

        if isinstance(delta, ToolCallPartDelta):
            tool_item_key = (
                self._tool_call_id_to_item_key.get(delta.tool_call_id, item_key)
                if delta.tool_call_id is not None
                else item_key
            )
            kind = self._item_kind.get(tool_item_key, "")
            if kind == "code_interpreter_call":
                if delta.args_delta is None:
                    return
                extractor = self._code_arg_extractors.setdefault(
                    tool_item_key, _CodeJsonArgsExtractor(target_key="code")
                )
                args_delta = delta.args_delta
                raw = args_delta if isinstance(args_delta, str) else _json_dumps(args_delta)
                if code_delta := extractor.feed(raw):
                    yield CodeInterpreterCallCodeDelta(item_key=tool_item_key, delta=code_delta)
                return

            if kind == "mcp_call":
                if delta.args_delta is None:
                    return
                yield McpCallArgumentsDelta(
                    item_key=tool_item_key,
                    delta=delta.args_delta
                    if isinstance(delta.args_delta, str)
                    else _json_dumps(delta.args_delta),
                )
                return

            if kind == "web_search_call":
                return

            if delta.args_delta is None:
                return
            yield FunctionCallArgumentsDelta(
                item_key=tool_item_key,
                delta=delta.args_delta
                if isinstance(delta.args_delta, str)
                else _json_dumps(delta.args_delta),
            )
            return

    def _on_part_end(self, event: PartEndEvent) -> Iterable[NormalizedEvent]:
        part = event.part
        item_key = self._index_to_item_key.get(event.index, f"part:{event.index}")

        if isinstance(part, TextPart):
            yield MessageDone(item_key=item_key, text=part.content)
            return

        if isinstance(part, ThinkingPart):
            yield ReasoningDone(item_key=item_key, text=part.content)
            return

        if isinstance(part, ToolCallPart):
            tool_item_key = self._tool_call_id_to_item_key.get(part.tool_call_id, item_key)
            kind = self._item_kind.get(tool_item_key, "")
            if kind == "code_interpreter_call":
                code = part.args_as_dict().get("code")
                yield CodeInterpreterCallCodeDone(item_key=tool_item_key, code=code)
            elif kind == "web_search_call":
                return
            elif kind == "mcp_call":
                yield McpCallArgumentsDone(
                    item_key=tool_item_key,
                    arguments_json=part.args_as_json_str(),
                )
            else:
                yield FunctionCallDone(
                    item_key=tool_item_key, arguments_json=part.args_as_json_str()
                )
            return

    def _on_function_tool_call(self, event: FunctionToolCallEvent) -> Iterable[NormalizedEvent]:
        if event.part.tool_name != self._code_interpreter_tool_name:
            if event.part.tool_name == WEB_SEARCH_TOOL:
                item_key = self._tool_call_id_to_item_key.get(event.part.tool_call_id)
                if item_key:
                    yield WebSearchCallSearching(item_key=item_key)
            return
        item_key = self._tool_call_id_to_item_key.get(event.tool_call_id)
        if item_key:
            yield CodeInterpreterCallInterpreting(item_key=item_key)

    def _on_function_tool_result(
        self, event: FunctionToolResultEvent
    ) -> Iterable[NormalizedEvent]:
        item_key = self._tool_call_id_to_item_key.get(event.tool_call_id)
        if item_key is None:
            return

        if self._item_kind.get(item_key) == "mcp_call":
            yield from self._on_mcp_tool_result(event=event, item_key=item_key)
            return

        if event.result.tool_name == WEB_SEARCH_TOOL:
            yield from self._on_web_search_tool_result(event=event, item_key=item_key)
            return

        if event.result.tool_name != self._code_interpreter_tool_name:
            return
        yield from self._on_code_interpreter_tool_result(event=event, item_key=item_key)

    def _on_web_search_tool_result(
        self, *, event: FunctionToolResultEvent, item_key: str
    ) -> Iterable[NormalizedEvent]:
        raw = self._result_raw_text(event.result)
        if raw is None:
            return
        payload = parse_web_search_tool_result(raw)
        action = payload.action
        action_type = action.type
        source_items: tuple[dict[str, str], ...] = ()
        query: str | None = None
        queries: tuple[str, ...] = ()
        url: str | None = None
        pattern: str | None = None

        if isinstance(action, SearchActionPublic):
            query = action.query
            queries = tuple(action.queries or ())
            source_items = tuple(
                {"type": source.type, "url": source.url} for source in (action.sources or [])
            )
        elif isinstance(action, OpenPageActionPublic):
            url = action.url
        elif isinstance(action, FindInPageActionPublic):
            url = action.url
            pattern = action.pattern

        yield WebSearchCallCompleted(
            item_key=item_key,
            action_type=action_type,
            query=query,
            queries=queries,
            sources=source_items,
            url=url,
            pattern=pattern,
        )

    def _on_mcp_tool_result(
        self, *, event: FunctionToolResultEvent, item_key: str
    ) -> Iterable[NormalizedEvent]:
        if not isinstance(event.result, ToolReturnPart):
            yield McpCallFailed(
                item_key=item_key,
                error_text="MCP tool returned empty result.",
            )
            return
        try:
            ref, result = parse_mcp_tool_result_payload(event.result.content)
        except ValueError as exc:
            yield McpCallFailed(item_key=item_key, error_text=truncate_error_text(str(exc)))
            return

        expected_ref = self._mcp_tool_name_map.get(event.result.tool_name)
        if expected_ref is None or expected_ref != ref:
            yield McpCallFailed(
                item_key=item_key,
                error_text=truncate_error_text("MCP tool returned mismatched tool ref."),
            )
            return

        if result.ok:
            yield McpCallCompleted(item_key=item_key, output_text=result.output_text or "")
            return

        yield McpCallFailed(
            item_key=item_key,
            error_text=truncate_error_text(result.error_text),
        )

    def _on_code_interpreter_tool_result(
        self, *, event: FunctionToolResultEvent, item_key: str
    ) -> Iterable[NormalizedEvent]:
        raw = self._result_raw_text(event.result)
        stdout, stderr, result = _parse_code_interpreter_tool_output(raw)
        yield CodeInterpreterCallCompleted(
            item_key=item_key,
            stdout=stdout,
            stderr=stderr,
            result=result,
        )

    @staticmethod
    def _result_raw_text(result: Any) -> str | None:
        if not isinstance(result, ToolReturnPart):
            return None
        return result.model_response_str()

    @staticmethod
    def _usage_final(event: AgentRunResultEvent) -> UsageFinal:
        run_usage: RunUsage = event.result.usage()
        incomplete_reason = _incomplete_reason_from_model_messages(event.result.all_messages())
        return UsageFinal(
            input_tokens=run_usage.input_tokens,
            output_tokens=run_usage.output_tokens,
            total_tokens=run_usage.total_tokens,
            cache_read_tokens=run_usage.cache_read_tokens,
            cache_write_tokens=run_usage.cache_write_tokens,
            reasoning_tokens=run_usage.details.get("reasoning_tokens", 0),
            incomplete_reason=incomplete_reason,
        )


def _incomplete_reason_from_model_messages(
    messages: list[ModelMessage],
) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, ModelResponse):
            continue
        finish_reason = message.finish_reason
        if finish_reason == "length":
            return "max_output_tokens"
        if finish_reason == "content_filter":
            return "content_filter"
        return None
    return None


def _json_dumps(value: dict[str, Any]) -> str:
    # Used only for rare providers that send `ToolCallPartDelta.args_delta` as a dict.
    # Keep dependencies minimal and produce stable JSON text.
    import json

    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _parse_code_interpreter_tool_output(
    raw: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Parse the code interpreter tool return payload.

    The Bun+Pyodide server returns a JSON object (stringified) with fields like:
    - status: "success" | "exception"
    - stdout: captured stdout (print output)
    - stderr: captured stderr
    - result: final expression display value (string) or null

    Backwards/defensive behavior:
    - If `raw` isn't JSON, treat it as the display `result` and leave stdout/stderr empty.
    """
    if not raw:
        return None, None, None
    try:
        import json

        obj = json.loads(raw)
    except Exception:
        return None, None, raw
    if not isinstance(obj, dict):
        return None, None, raw

    stdout_val = obj.get("stdout")
    stdout = stdout_val if isinstance(stdout_val, str) and stdout_val else None
    stderr_val = obj.get("stderr")
    stderr = stderr_val if isinstance(stderr_val, str) and stderr_val else None

    result_val = obj.get("result")
    if result_val is None:
        result = None
    elif isinstance(result_val, str):
        result = result_val
    else:
        result = str(result_val)

    return stdout, stderr, result


@dataclass(slots=True)
class _CodeJsonArgsExtractor:
    """Incrementally extract the JSON string value for a single key (default: `"code"`).

    The upstream stream delivers tool-call argument JSON fragments (usually `str` chunks). We need to emit the *code
    string* as it is being produced, not the JSON fragments themselves.
    """

    target_key: str = "code"

    _buf: str = ""
    _search_from: int = 0
    _in_value: bool = False
    _done: bool = False
    _pos: int = 0
    _escape: bool = False
    _unicode_remaining: int = 0
    _unicode_buf: str = ""

    def feed(self, chunk: str) -> str:
        if self._done or not chunk:
            return ""
        self._buf += chunk
        out: list[str] = []

        while True:
            if self._done:
                break
            if not self._in_value:
                start = self._find_value_start()
                if start is None:
                    break
                self._in_value = True
                self._pos = start

            # Parse JSON string content until we run out of buffer or hit the closing quote.
            while self._pos < len(self._buf):
                ch = self._buf[self._pos]
                self._pos += 1

                if self._unicode_remaining:
                    if ch.lower() in "0123456789abcdef":
                        self._unicode_buf += ch
                        self._unicode_remaining -= 1
                        if self._unicode_remaining == 0:
                            try:
                                out.append(chr(int(self._unicode_buf, 16)))
                            finally:
                                self._unicode_buf = ""
                    else:
                        # Invalid escape; abandon unicode decoding.
                        self._unicode_remaining = 0
                        self._unicode_buf = ""
                    continue

                if self._escape:
                    self._escape = False
                    if ch == "n":
                        out.append("\n")
                    elif ch == "r":
                        out.append("\r")
                    elif ch == "t":
                        out.append("\t")
                    elif ch == "b":
                        out.append("\b")
                    elif ch == "f":
                        out.append("\f")
                    elif ch == '"':
                        out.append('"')
                    elif ch == "\\":
                        out.append("\\")
                    elif ch == "/":
                        out.append("/")
                    elif ch == "u":
                        self._unicode_remaining = 4
                        self._unicode_buf = ""
                    else:
                        out.append(ch)
                    continue

                if ch == "\\":
                    self._escape = True
                    continue
                if ch == '"':
                    self._done = True
                    self._in_value = False
                    break
                out.append(ch)

            # Drop consumed prefix to avoid unbounded growth.
            if self._pos > 4096:
                self._buf = self._buf[self._pos :]
                self._search_from = 0
                self._pos = 0
            else:
                break

        # Keep a small tail while searching to handle boundary splits.
        if not self._in_value and not self._done and len(self._buf) > 256:
            self._buf = self._buf[-256:]
            self._search_from = max(0, self._search_from - max(0, len(self._buf) - 256))

        return "".join(out)

    def _find_value_start(self) -> int | None:
        key_pat = f'"{self.target_key}"'
        while True:
            idx = self._buf.find(key_pat, self._search_from)
            if idx == -1:
                self._search_from = max(0, len(self._buf) - 16)
                return None
            j = idx + len(key_pat)
            while j < len(self._buf) and self._buf[j] in " \t\r\n":
                j += 1
            if j >= len(self._buf):
                self._search_from = idx
                return None
            if self._buf[j] != ":":
                self._search_from = idx + 1
                continue
            j += 1
            while j < len(self._buf) and self._buf[j] in " \t\r\n":
                j += 1
            if j >= len(self._buf):
                self._search_from = idx
                return None
            if self._buf[j] != '"':
                self._search_from = idx + 1
                continue
            self._search_from = j + 1
            return j + 1
