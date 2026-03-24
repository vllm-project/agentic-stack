from __future__ import annotations

from collections.abc import Iterable

from agentic_stack.responses_core.composer import ResponseComposer
from agentic_stack.responses_core.models import (
    CodeInterpreterCallCodeDone,
    CodeInterpreterCallCompleted,
    CodeInterpreterCallInterpreting,
    CodeInterpreterCallStarted,
    FunctionCallArgumentsDelta,
    FunctionCallDone,
    FunctionCallStarted,
    MessageDelta,
    MessageDone,
    MessageStarted,
    UsageFinal,
    WebSearchCallCompleted,
    WebSearchCallSearching,
    WebSearchCallStarted,
)
from agentic_stack.types.openai import OpenAIResponsesResponse


def _drain(composer: ResponseComposer, events: Iterable[object]):
    out = []
    out.extend(composer.start())
    for e in events:
        out.extend(list(composer.feed(e)))  # type: ignore[arg-type]
    return out


def test_message_output_index_and_item_id_stable():
    composer = ResponseComposer(response=OpenAIResponsesResponse(model="test-model"))

    out = _drain(
        composer,
        [
            MessageStarted(item_key="m1"),
            MessageDelta(item_key="m1", delta="Hello"),
            MessageDelta(item_key="m1", delta=", world"),
            MessageDone(item_key="m1", text="Hello, world"),
            UsageFinal(
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            ),
        ],
    )

    added = [e for e in out if e.type == "response.output_item.added" and e.item.type == "message"]
    assert len(added) == 1
    out_index = added[0].output_index
    item_id = added[0].item.id

    deltas = [e for e in out if e.type == "response.output_text.delta"]
    assert deltas
    assert {d.output_index for d in deltas} == {out_index}
    assert {d.item_id for d in deltas} == {item_id}

    done = [e for e in out if e.type == "response.output_text.done"]
    assert len(done) == 1
    assert done[0].output_index == out_index
    assert done[0].item_id == item_id


def test_completed_response_omits_reasoning_item_when_no_thinking_part():
    composer = ResponseComposer(response=OpenAIResponsesResponse(model="test-model"))

    out = _drain(
        composer,
        [
            MessageStarted(item_key="m1"),
            MessageDelta(item_key="m1", delta="Hello"),
            MessageDone(item_key="m1", text="Hello"),
            UsageFinal(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            ),
        ],
    )

    completed = [e for e in out if e.type == "response.completed"]
    assert len(completed) == 1
    resp = completed[0].response
    assert resp.output is not None
    assert [o.type for o in resp.output] == ["message"]


def test_incomplete_response_sets_incomplete_details_for_max_output_tokens():
    composer = ResponseComposer(response=OpenAIResponsesResponse(model="test-model"))

    out = _drain(
        composer,
        [
            MessageStarted(item_key="m1"),
            MessageDone(item_key="m1", text="Partial"),
            UsageFinal(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
                incomplete_reason="max_output_tokens",
            ),
        ],
    )

    incomplete = [e for e in out if e.type == "response.incomplete"]
    completed = [e for e in out if e.type == "response.completed"]
    assert len(incomplete) == 1
    assert not completed
    resp = incomplete[0].response
    assert resp.status == "incomplete"
    assert resp.incomplete_details is not None
    assert resp.incomplete_details.reason == "max_output_tokens"
    assert resp.completed_at is None


def test_function_call_arguments_deltas_attributed_to_function_item():
    composer = ResponseComposer(response=OpenAIResponsesResponse(model="test-model"))

    out = _drain(
        composer,
        [
            FunctionCallStarted(
                item_key="fc1",
                call_id="call_123",
                name="get_weather",
                initial_arguments_json="",
            ),
            MessageStarted(item_key="m1"),
            MessageDelta(item_key="m1", delta="(ignored)"),
            FunctionCallArgumentsDelta(item_key="fc1", delta="{"),
            FunctionCallArgumentsDelta(item_key="fc1", delta='"x":1'),
            FunctionCallArgumentsDelta(item_key="fc1", delta="}"),
            FunctionCallDone(item_key="fc1", arguments_json='{"x":1}'),
            UsageFinal(
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            ),
        ],
    )

    fc_added = [
        e for e in out if e.type == "response.output_item.added" and e.item.type == "function_call"
    ]
    assert len(fc_added) == 1
    fc_item_id = fc_added[0].item.id
    fc_out_index = fc_added[0].output_index

    arg_deltas = [e for e in out if e.type == "response.function_call_arguments.delta"]
    assert arg_deltas
    assert {d.item_id for d in arg_deltas} == {fc_item_id}
    assert {d.output_index for d in arg_deltas} == {fc_out_index}


def test_code_interpreter_outputs_populated_on_completion():
    composer = ResponseComposer(
        response=OpenAIResponsesResponse(model="test-model"),
        include={"code_interpreter_call.outputs"},
    )

    out = _drain(
        composer,
        [
            CodeInterpreterCallStarted(item_key="ci1", initial_code=""),
            CodeInterpreterCallCodeDone(item_key="ci1", code="print(1)"),
            CodeInterpreterCallInterpreting(item_key="ci1"),
            CodeInterpreterCallCompleted(item_key="ci1", stdout="1\n", stderr=None, result=None),
            UsageFinal(
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
                cache_read_tokens=4,
                cache_write_tokens=5,
                reasoning_tokens=6,
            ),
        ],
    )

    ci_done = [
        e
        for e in out
        if e.type == "response.output_item.done" and e.item.type == "code_interpreter_call"
    ]
    assert len(ci_done) == 1
    item = ci_done[0].item
    assert item.status == "completed"
    assert item.code == "print(1)"
    assert item.outputs and item.outputs[0].type == "logs"
    assert item.outputs[0].logs == "1\n"

    completed = [e for e in out if e.type == "response.completed"]
    assert len(completed) == 1
    resp = completed[0].response
    assert resp.usage is not None
    assert resp.usage.input_tokens_details.cached_tokens == 4
    assert resp.usage.output_tokens_details.reasoning_tokens == 6


def test_code_interpreter_outputs_omitted_without_include():
    composer = ResponseComposer(response=OpenAIResponsesResponse(model="test-model"))

    out = _drain(
        composer,
        [
            CodeInterpreterCallStarted(item_key="ci1", initial_code=""),
            CodeInterpreterCallCodeDone(item_key="ci1", code="print(1)"),
            CodeInterpreterCallInterpreting(item_key="ci1"),
            CodeInterpreterCallCompleted(item_key="ci1", stdout="1\n", stderr=None, result=None),
            UsageFinal(
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            ),
        ],
    )

    ci_done = [
        e
        for e in out
        if e.type == "response.output_item.done" and e.item.type == "code_interpreter_call"
    ]
    assert len(ci_done) == 1
    assert ci_done[0].item.outputs is None


def test_code_interpreter_outputs_include_stdout_and_final_expression():
    composer = ResponseComposer(
        response=OpenAIResponsesResponse(model="test-model"),
        include={"code_interpreter_call.outputs"},
    )

    out = _drain(
        composer,
        [
            CodeInterpreterCallStarted(item_key="ci1", initial_code=""),
            CodeInterpreterCallCodeDone(item_key="ci1", code='print("P1"); print("P2"); 2+2'),
            CodeInterpreterCallInterpreting(item_key="ci1"),
            CodeInterpreterCallCompleted(item_key="ci1", stdout="P1\nP2\n", stderr="", result="4"),
            UsageFinal(
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            ),
        ],
    )

    ci_done = [
        e
        for e in out
        if e.type == "response.output_item.done" and e.item.type == "code_interpreter_call"
    ]
    assert len(ci_done) == 1
    outputs = ci_done[0].item.outputs
    assert outputs is not None
    assert [o.logs for o in outputs if o.type == "logs"] == ["P1\nP2\n", "4"]


def test_web_search_output_uses_generic_event_family() -> None:
    composer = ResponseComposer(response=OpenAIResponsesResponse(model="test-model"))

    out = _drain(
        composer,
        [
            WebSearchCallStarted(item_key="ws1"),
            WebSearchCallSearching(item_key="ws1"),
            WebSearchCallCompleted(
                item_key="ws1",
                action_type="search",
                query="example query",
                queries=("example query", "rewritten query"),
                sources=({"type": "url", "url": "https://example.com/a"},),
            ),
            UsageFinal(
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            ),
        ],
    )

    assert [event.type for event in out if event.type.startswith("response.web_search_call.")] == [
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
        "response.web_search_call.completed",
    ]
    assert not any(event.type.startswith("response.function_call_arguments.") for event in out)

    added_item = next(
        event.item
        for event in out
        if event.type == "response.output_item.added" and event.item.type == "web_search_call"
    )
    assert added_item.status == "in_progress"
    assert added_item.action is None


def test_web_search_sources_are_include_gated() -> None:
    composer = ResponseComposer(
        response=OpenAIResponsesResponse(model="test-model"),
        include={"web_search_call.action.sources"},
    )

    out = _drain(
        composer,
        [
            WebSearchCallStarted(item_key="ws1"),
            WebSearchCallCompleted(
                item_key="ws1",
                action_type="search",
                query="example query",
                sources=({"type": "url", "url": "https://example.com/a"},),
            ),
            UsageFinal(
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            ),
        ],
    )

    done_item = next(
        event.item
        for event in out
        if event.type == "response.output_item.done" and event.item.type == "web_search_call"
    )
    assert done_item.action is not None
    assert done_item.action.type == "search"
    assert done_item.action.sources is not None
    assert done_item.action.sources[0].url == "https://example.com/a"

    composer_without_sources = ResponseComposer(
        response=OpenAIResponsesResponse(model="test-model")
    )
    out_without_sources = _drain(
        composer_without_sources,
        [
            WebSearchCallStarted(item_key="ws2"),
            WebSearchCallCompleted(
                item_key="ws2",
                action_type="search",
                query="example query",
                sources=({"type": "url", "url": "https://example.com/a"},),
            ),
            UsageFinal(
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            ),
        ],
    )
    done_item_without_sources = next(
        event.item
        for event in out_without_sources
        if event.type == "response.output_item.done" and event.item.type == "web_search_call"
    )
    assert done_item_without_sources.action is not None
    assert done_item_without_sources.action.sources is None
