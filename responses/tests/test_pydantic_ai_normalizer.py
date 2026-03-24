from __future__ import annotations

from pydantic_ai import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)

from agentic_stack.responses_core.models import (
    CodeInterpreterCallCodeDelta,
    CodeInterpreterCallCodeDone,
    CodeInterpreterCallStarted,
    FunctionCallArgumentsDelta,
    FunctionCallDone,
    WebSearchCallCompleted,
    WebSearchCallSearching,
    WebSearchCallStarted,
)
from agentic_stack.responses_core.normalizer import PydanticAINormalizer


def test_code_interpreter_code_deltas_emitted_from_tool_call_args_json_fragments():
    normalizer = PydanticAINormalizer(
        builtin_tool_names={"code_interpreter"},
        code_interpreter_tool_name="code_interpreter",
    )

    events = [
        PartStartEvent(
            index=0,
            part=ToolCallPart(tool_name="code_interpreter", args=None, tool_call_id="call_1"),
        ),
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(tool_call_id="call_1", args_delta='{"code":"print('),
        ),
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(tool_call_id="call_1", args_delta="1"),
        ),
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(tool_call_id="call_1", args_delta=')"}'),
        ),
        PartEndEvent(
            index=0,
            part=ToolCallPart(
                tool_name="code_interpreter",
                args='{"code":"print(1)"}',
                tool_call_id="call_1",
            ),
        ),
    ]

    out = []
    for e in events:
        out.extend(list(normalizer.on_event(e)))

    assert any(isinstance(e, CodeInterpreterCallStarted) for e in out)

    deltas = [e for e in out if isinstance(e, CodeInterpreterCallCodeDelta)]
    assert [d.delta for d in deltas] == ["print(", "1", ")"]

    done = [e for e in out if isinstance(e, CodeInterpreterCallCodeDone)]
    assert len(done) == 1
    assert done[0].code == "print(1)"


def test_web_search_normalizer_emits_web_search_events_without_function_arg_events():
    normalizer = PydanticAINormalizer(
        builtin_tool_names={"web_search"},
        code_interpreter_tool_name="code_interpreter",
    )

    events = [
        PartStartEvent(
            index=0,
            part=ToolCallPart(tool_name="web_search", args=None, tool_call_id="call_ws_1"),
        ),
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(
                tool_call_id="call_ws_1",
                args_delta='{"action":"search","query":"example query"}',
            ),
        ),
        PartEndEvent(
            index=0,
            part=ToolCallPart(
                tool_name="web_search",
                args='{"action":"search","query":"example query"}',
                tool_call_id="call_ws_1",
            ),
        ),
        FunctionToolCallEvent(
            part=ToolCallPart(
                tool_name="web_search",
                args='{"action":"search","query":"example query"}',
                tool_call_id="call_ws_1",
            )
        ),
        FunctionToolResultEvent(
            result=ToolReturnPart(
                tool_name="web_search",
                content=(
                    '{"action":{"type":"search","query":"example query","sources":'
                    '[{"type":"url","url":"https://example.com/a"}]}}'
                ),
                tool_call_id="call_ws_1",
            )
        ),
    ]

    out = []
    for event in events:
        out.extend(list(normalizer.on_event(event)))

    assert any(isinstance(event, WebSearchCallStarted) for event in out)
    assert any(isinstance(event, WebSearchCallSearching) for event in out)
    completed = [event for event in out if isinstance(event, WebSearchCallCompleted)]
    assert len(completed) == 1
    assert completed[0].action_type == "search"
    assert completed[0].query == "example query"
    assert completed[0].sources == ({"type": "url", "url": "https://example.com/a"},)
    assert not any(isinstance(event, FunctionCallArgumentsDelta) for event in out)
    assert not any(isinstance(event, FunctionCallDone) for event in out)
