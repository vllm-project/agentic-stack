//! Cassette-driven integration tests: feed real vLLM SSE recordings through
//! the full accumulator pipeline (normalize → `process_event` → finalize) and
//! verify the resulting output items match expected values.
//!
//! Tests cover both the legacy `events/` cassettes (flat SSE list) and the
//! newer `tool_calls/` cassettes from PR #60 (multi-turn `turns` format).

use serde::Deserialize;

use agentic_core::executor::accumulator::ResponseAccumulator;
use agentic_core::types::event::MessageStatus;
use agentic_core::types::io::{CustomToolCall, FunctionToolCall, OutputItem, WebSearchCall};

const CASSETTE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/events");
const TOOL_CALLS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/tool_calls");
const REASONING_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/reasoning/responses");
const CODEX_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/codex");
const WEB_SEARCH_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/web_search");
const CUSTOM_TOOL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/custom_tool");
const WEB_SEARCH_GATEWAY_MODEL: &str = "Qwen/Qwen3.5-35B-A3B-FP8";
const WEB_SEARCH_GATEWAY_MODEL_SLUG: &str = "Qwen-Qwen3.5-35B-A3B-FP8";
const WEB_SEARCH_OPENAI_MODEL: &str = "gpt-5.6";
const WEB_SEARCH_OPENAI_MODEL_SLUG: &str = "gpt-5.6";

fn from_sse_lines(lines: impl IntoIterator<Item = String>, conversation_id: Option<&str>) -> ResponseAccumulator {
    ResponseAccumulator::from_sse_lines(lines, conversation_id).expect("valid cassette SSE stream")
}

// --- Legacy event cassette format ---

#[derive(Deserialize)]
struct EventCassette {
    sse: Vec<String>,
    expected_function_call: Option<ExpectedFunctionCall>,
    #[allow(dead_code)]
    expected_text: Option<String>,
}

#[derive(Deserialize)]
struct ExpectedFunctionCall {
    name: String,
    arguments: String,
}

fn load_cassette(filename: &str) -> EventCassette {
    let path = format!("{CASSETTE_DIR}/{filename}");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_yml::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

// --- New multi-turn cassette format (PR #60) ---

#[derive(Deserialize)]
struct TurnCassette {
    turns: Vec<Turn>,
}

#[derive(Deserialize)]
struct Turn {
    #[allow(dead_code)]
    filename: String,
    #[allow(dead_code)]
    request: serde_yml::Value,
    response: TurnResponse,
}

#[derive(Deserialize)]
struct TurnResponse {
    #[allow(dead_code)]
    headers: serde_yml::Value,
    #[allow(dead_code)]
    status_code: Option<u16>,
    #[serde(default)]
    sse: Vec<String>,
    body: Option<serde_json::Value>,
}

fn load_turn_cassette_from(dir: &str, filename: &str) -> TurnCassette {
    let path = format!("{dir}/{filename}");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_yml::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

fn load_turn_cassette(filename: &str) -> TurnCassette {
    load_turn_cassette_from(TOOL_CALLS_DIR, filename)
}

fn load_reasoning_cassette(filename: &str) -> TurnCassette {
    load_turn_cassette_from(REASONING_DIR, filename)
}

fn load_codex_cassette(filename: &str) -> TurnCassette {
    load_turn_cassette_from(CODEX_DIR, filename)
}

fn load_web_search_cassette(filename: &str) -> TurnCassette {
    load_turn_cassette_from(WEB_SEARCH_DIR, filename)
}

fn load_custom_tool_cassette(filename: &str) -> TurnCassette {
    load_turn_cassette_from(CUSTOM_TOOL_DIR, filename)
}

fn load_web_search_cassette_pair(streaming: bool) -> (TurnCassette, TurnCassette) {
    let mode = if streaming { "streaming" } else { "nonstreaming" };
    let openai = load_web_search_cassette(&format!(
        "web-search-openai-reference-{WEB_SEARCH_OPENAI_MODEL_SLUG}-{mode}.yaml"
    ));
    let gateway = load_web_search_cassette(&format!(
        "web-search-gateway-{WEB_SEARCH_GATEWAY_MODEL_SLUG}-{mode}.yaml"
    ));
    (openai, gateway)
}

/// Extracts `data: ...` lines from raw SSE entries (which may include
/// `event:` lines and blank separators).
fn extract_data_lines(sse_entries: &[String]) -> Vec<String> {
    sse_entries
        .iter()
        .flat_map(|entry| entry.lines())
        .filter(|line| line.starts_with("data: "))
        .map(ToString::to_string)
        .collect()
}

fn response_body_from_data_line(line: &str) -> Option<serde_json::Value> {
    let data = line.strip_prefix("data: ")?;
    if data == "[DONE]" {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(data).ok()?;
    if let Some(response) = value.get("response").and_then(serde_json::Value::as_object) {
        return Some(serde_json::Value::Object(response.clone()));
    }
    if value.get("object").and_then(serde_json::Value::as_str) == Some("response") {
        return Some(value);
    }
    None
}

fn process_completed_response_object_from_sse(
    cassette: &TurnCassette,
    turn_idx: usize,
    model: &str,
) -> Vec<OutputItem> {
    let data_lines = extract_data_lines(&cassette.turns[turn_idx].response.sse);
    let response = data_lines
        .iter()
        .filter_map(|line| response_body_from_data_line(line))
        .find(|value| value.get("status").and_then(serde_json::Value::as_str) == Some("completed"))
        .unwrap_or_else(|| panic!("turn {} must contain a completed response object", turn_idx + 1));
    let body = serde_json::to_string(&response).unwrap();
    let acc = ResponseAccumulator::from_json(&body, None).unwrap();
    let payload = acc.finalize(model, None, None);
    assert_eq!(payload.status, "completed");
    payload.output
}

fn process_codex_streaming_turn(cassette: &TurnCassette, turn_idx: usize, model: &str) -> Vec<OutputItem> {
    let data_lines = extract_data_lines(&cassette.turns[turn_idx].response.sse);
    assert!(
        !data_lines.is_empty(),
        "Codex cassette turn {} must have SSE data lines",
        turn_idx + 1
    );
    let acc = from_sse_lines(data_lines, None);
    let payload = acc.finalize(model, None, None);
    assert_eq!(payload.status, "completed");
    payload.output
}

fn first_function_call(output: &[OutputItem]) -> &FunctionToolCall {
    output
        .iter()
        .find_map(|item| {
            if let OutputItem::FunctionCall(call) = item {
                Some(call)
            } else {
                None
            }
        })
        .expect("output must contain a function call")
}

fn first_custom_tool_call(output: &[OutputItem]) -> &CustomToolCall {
    output
        .iter()
        .find_map(|item| {
            if let OutputItem::CustomToolCall(call) = item {
                Some(call)
            } else {
                None
            }
        })
        .expect("output must contain a custom tool call")
}

fn turn_request_body(turn: &Turn) -> serde_json::Value {
    let body = turn.request.get("body").expect("turn request must have body");
    serde_json::to_value(body).expect("request body must convert to JSON")
}

// === Legacy cassette tests ===

/// Feeds a real vLLM `function_call` SSE recording through the accumulator and
/// verifies the output contains the correct `FunctionCall` item.
#[test]
fn test_accumulator_cassette_function_call_vllm_gemma4() {
    let cassette = load_cassette("function-call-vllm-gemma4.yaml");
    let expected_fc = cassette
        .expected_function_call
        .expect("cassette must have expected_function_call");

    let acc = from_sse_lines(cassette.sse, None);
    let payload = acc.finalize("google/gemma-4-26B-A4B-it", None, None);

    assert_eq!(payload.status, "completed");
    assert_eq!(payload.output.len(), 1, "expected exactly one output item");

    if let OutputItem::FunctionCall(fc) = &payload.output[0] {
        assert_eq!(fc.name, expected_fc.name);
        assert_eq!(fc.arguments, expected_fc.arguments);
        assert_eq!(fc.status, MessageStatus::Completed);
        assert!(!fc.call_id.is_empty(), "call_id should be populated");
        assert!(!fc.id.is_empty(), "id should be populated");
    } else {
        panic!("expected OutputItem::FunctionCall, got {:?}", payload.output[0]);
    }

    assert!(payload.usage.is_some(), "usage should be present");
    let usage = payload.usage.unwrap();
    assert_eq!(usage.input_tokens, 66);
    assert_eq!(usage.output_tokens, 21);
    assert_eq!(usage.total_tokens, 87);
}

/// Feeds the text-only cassette through the accumulator and verifies no
/// `function_call` items leak in — regression guard for type-aware branching.
#[test]
fn test_accumulator_cassette_text_only_no_function_calls() {
    let cassette = load_cassette("text-only-vllm-gemma4.yaml");

    let acc = from_sse_lines(cassette.sse, None);
    let payload = acc.finalize("google/gemma-4-26B-A4B-it", None, None);

    assert_eq!(payload.status, "completed");
    for item in &payload.output {
        assert!(
            matches!(item, OutputItem::Message(_)),
            "text-only cassette should only produce Message items, got {item:?}"
        );
    }
}

// === PR #60 tool_calls cassette tests ===

/// `tool_choice=auto` streaming: model decides to call multiple tools (parallel tool use).
#[test]
fn test_tool_calls_cassette_auto_streaming() {
    let cassette = load_turn_cassette("tool-call-auto-Qwen-Qwen3-30B-A3B-FP8-streaming.yaml");
    let turn = &cassette.turns[0];
    let data_lines = extract_data_lines(&turn.response.sse);

    let acc = from_sse_lines(data_lines, None);
    let payload = acc.finalize("Qwen/Qwen3-30B-A3B-FP8", None, None);

    assert_eq!(payload.status, "completed");

    let function_calls: Vec<_> = payload
        .output
        .iter()
        .filter(|item| matches!(item, OutputItem::FunctionCall(_)))
        .collect();

    assert!(
        !function_calls.is_empty(),
        "auto mode should produce at least one function call"
    );

    for item in &function_calls {
        if let OutputItem::FunctionCall(fc) = item {
            assert!(!fc.name.is_empty(), "function call name must not be empty");
            assert!(!fc.arguments.is_empty(), "function call arguments must not be empty");
            assert_eq!(fc.status, MessageStatus::Completed);
            assert!(!fc.call_id.is_empty(), "call_id must be populated");
        }
    }

    assert!(payload.usage.is_some());
}

/// `tool_choice=required` streaming: model is forced to call a tool.
#[test]
fn test_tool_calls_cassette_required_streaming() {
    let cassette = load_turn_cassette("tool-call-required-Qwen-Qwen3-30B-A3B-FP8-streaming.yaml");
    let turn = &cassette.turns[0];
    let data_lines = extract_data_lines(&turn.response.sse);

    let acc = from_sse_lines(data_lines, None);
    let payload = acc.finalize("Qwen/Qwen3-30B-A3B-FP8", None, None);

    assert_eq!(payload.status, "completed");

    let function_calls: Vec<_> = payload
        .output
        .iter()
        .filter(|item| matches!(item, OutputItem::FunctionCall(_)))
        .collect();

    assert!(
        !function_calls.is_empty(),
        "required mode must produce at least one function call"
    );

    for item in &function_calls {
        if let OutputItem::FunctionCall(fc) = item {
            assert_eq!(fc.status, MessageStatus::Completed);
        }
    }
}

/// `tool_choice=named` streaming: model calls a specific named tool.
#[test]
fn test_tool_calls_cassette_named_streaming() {
    let cassette = load_turn_cassette("tool-call-named-Qwen-Qwen3-30B-A3B-FP8-streaming.yaml");
    let turn = &cassette.turns[0];
    let data_lines = extract_data_lines(&turn.response.sse);

    let acc = from_sse_lines(data_lines, None);
    let payload = acc.finalize("Qwen/Qwen3-30B-A3B-FP8", None, None);

    assert_eq!(payload.status, "completed");

    let function_calls: Vec<_> = payload
        .output
        .iter()
        .filter(|item| matches!(item, OutputItem::FunctionCall(_)))
        .collect();

    assert!(
        !function_calls.is_empty(),
        "named mode must produce at least one function call"
    );
}

/// `tool_choice=none` streaming: model should NOT call any tools.
#[test]
fn test_tool_calls_cassette_none_streaming() {
    let cassette = load_turn_cassette("tool-call-none-Qwen-Qwen3-30B-A3B-FP8-streaming.yaml");
    let turn = &cassette.turns[0];
    let data_lines = extract_data_lines(&turn.response.sse);

    let acc = from_sse_lines(data_lines, None);
    let payload = acc.finalize("Qwen/Qwen3-30B-A3B-FP8", None, None);

    assert_eq!(payload.status, "completed");

    let function_calls: Vec<_> = payload
        .output
        .iter()
        .filter(|item| matches!(item, OutputItem::FunctionCall(_)))
        .collect();

    assert!(
        function_calls.is_empty(),
        "none mode should produce zero function calls, got {}",
        function_calls.len()
    );

    assert!(
        !payload.output.is_empty(),
        "none mode should still produce message output"
    );
}

// === Non-streaming tool_calls cassette tests (exercises `from_json` path) ===

/// `tool_choice=auto` non-streaming: JSON response with parallel function calls.
#[test]
fn test_tool_calls_cassette_auto_nonstreaming() {
    let cassette = load_turn_cassette("tool-call-auto-Qwen-Qwen3-30B-A3B-FP8-nonstreaming.yaml");
    let body = cassette.turns[0]
        .response
        .body
        .as_ref()
        .expect("non-streaming cassette must have body");
    let body_str = serde_json::to_string(body).unwrap();

    let acc = ResponseAccumulator::from_json(&body_str, None).unwrap();
    let payload = acc.finalize("Qwen/Qwen3-30B-A3B-FP8", None, None);

    assert_eq!(payload.status, "completed");

    let function_calls: Vec<_> = payload
        .output
        .iter()
        .filter(|item| matches!(item, OutputItem::FunctionCall(_)))
        .collect();

    assert!(
        !function_calls.is_empty(),
        "auto mode should produce at least one function call"
    );

    for item in &function_calls {
        if let OutputItem::FunctionCall(fc) = item {
            assert!(!fc.name.is_empty());
            assert!(!fc.arguments.is_empty());
            assert_eq!(fc.status, MessageStatus::Completed);
            assert!(!fc.call_id.is_empty());
        }
    }
}

/// `tool_choice=required` non-streaming: forced tool call in JSON response.
#[test]
fn test_tool_calls_cassette_required_nonstreaming() {
    let cassette = load_turn_cassette("tool-call-required-Qwen-Qwen3-30B-A3B-FP8-nonstreaming.yaml");
    let body = cassette.turns[0]
        .response
        .body
        .as_ref()
        .expect("non-streaming cassette must have body");
    let body_str = serde_json::to_string(body).unwrap();

    let acc = ResponseAccumulator::from_json(&body_str, None).unwrap();
    let payload = acc.finalize("Qwen/Qwen3-30B-A3B-FP8", None, None);

    assert_eq!(payload.status, "completed");

    let function_calls: Vec<_> = payload
        .output
        .iter()
        .filter(|item| matches!(item, OutputItem::FunctionCall(_)))
        .collect();

    assert!(
        !function_calls.is_empty(),
        "required mode must produce at least one function call"
    );
}

/// `tool_choice=named` non-streaming: specific named tool in JSON response.
#[test]
fn test_tool_calls_cassette_named_nonstreaming() {
    let cassette = load_turn_cassette("tool-call-named-Qwen-Qwen3-30B-A3B-FP8-nonstreaming.yaml");
    let body = cassette.turns[0]
        .response
        .body
        .as_ref()
        .expect("non-streaming cassette must have body");
    let body_str = serde_json::to_string(body).unwrap();

    let acc = ResponseAccumulator::from_json(&body_str, None).unwrap();
    let payload = acc.finalize("Qwen/Qwen3-30B-A3B-FP8", None, None);

    assert_eq!(payload.status, "completed");

    let function_calls: Vec<_> = payload
        .output
        .iter()
        .filter(|item| matches!(item, OutputItem::FunctionCall(_)))
        .collect();

    assert!(
        !function_calls.is_empty(),
        "named mode must produce at least one function call"
    );
}

/// `tool_choice=none` non-streaming: no function calls in JSON response.
#[test]
fn test_tool_calls_cassette_none_nonstreaming() {
    let cassette = load_turn_cassette("tool-call-none-Qwen-Qwen3-30B-A3B-FP8-nonstreaming.yaml");
    let body = cassette.turns[0]
        .response
        .body
        .as_ref()
        .expect("non-streaming cassette must have body");
    let body_str = serde_json::to_string(body).unwrap();

    let acc = ResponseAccumulator::from_json(&body_str, None).unwrap();
    let payload = acc.finalize("Qwen/Qwen3-30B-A3B-FP8", None, None);

    assert_eq!(payload.status, "completed");

    let function_calls: Vec<_> = payload
        .output
        .iter()
        .filter(|item| matches!(item, OutputItem::FunctionCall(_)))
        .collect();

    assert!(
        function_calls.is_empty(),
        "none mode should produce zero function calls, got {}",
        function_calls.len()
    );
}

// === Reasoning cassette tests (regression guard for reasoning + function_call coexistence) ===

/// Reasoning streaming (Qwen3): accumulator produces `Reasoning` + `Message` items.
#[test]
fn test_reasoning_cassette_qwen3_streaming() {
    let cassette = load_reasoning_cassette("reasoning-single-Qwen-Qwen3-30B-A3B-FP8-streaming.yaml");
    let turn = &cassette.turns[0];
    let data_lines = extract_data_lines(&turn.response.sse);

    let acc = from_sse_lines(data_lines, None);
    let payload = acc.finalize("Qwen/Qwen3-30B-A3B-FP8", None, None);

    assert_eq!(payload.status, "completed");

    let reasoning_items: Vec<_> = payload
        .output
        .iter()
        .filter(|item| matches!(item, OutputItem::Reasoning(_)))
        .collect();

    let message_items: Vec<_> = payload
        .output
        .iter()
        .filter(|item| matches!(item, OutputItem::Message(_)))
        .collect();

    assert!(
        !reasoning_items.is_empty(),
        "reasoning cassette must produce at least one Reasoning item"
    );
    assert!(
        !message_items.is_empty(),
        "reasoning cassette should also produce a Message item"
    );

    // No function calls should leak in
    let function_calls: Vec<_> = payload
        .output
        .iter()
        .filter(|item| matches!(item, OutputItem::FunctionCall(_)))
        .collect();
    assert!(
        function_calls.is_empty(),
        "reasoning-only cassette should not produce function calls"
    );
}

/// Reasoning streaming (GPT-oss): validates accumulator handles different model's reasoning format.
/// Note: GPT-oss emits `output_text.done` without a preceding `output_item.added` for the
/// message, so the accumulator only captures the reasoning item from the streaming path.
/// The message content is available in the `response.completed` payload's output array.
#[test]
fn test_reasoning_cassette_gpt_oss_streaming() {
    let cassette = load_reasoning_cassette("reasoning-single-openai-gpt-oss-20b-streaming.yaml");
    let turn = &cassette.turns[0];
    let data_lines = extract_data_lines(&turn.response.sse);

    let acc = from_sse_lines(data_lines, None);
    let payload = acc.finalize("openai/gpt-oss-20b", None, None);

    assert_eq!(payload.status, "completed");

    let reasoning_items: Vec<_> = payload
        .output
        .iter()
        .filter(|item| matches!(item, OutputItem::Reasoning(_)))
        .collect();

    assert!(
        !reasoning_items.is_empty(),
        "GPT-oss reasoning cassette must produce at least one Reasoning item"
    );

    // No function calls should leak in
    let function_calls: Vec<_> = payload
        .output
        .iter()
        .filter(|item| matches!(item, OutputItem::FunctionCall(_)))
        .collect();
    assert!(
        function_calls.is_empty(),
        "reasoning-only cassette should not produce function calls"
    );
}

// === Codex integration cassettes ===

#[test]
fn test_codex_gateway_http_cassettes_parse_completed_response_objects() {
    let cases = [
        (
            "codex-gateway-http-function-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
            None,
            "agentic_plain_echo",
        ),
        (
            "codex-gateway-http-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
            Some("mcp__agentic_fixture"),
            "add_numbers",
        ),
    ];

    for (filename, expected_namespace, expected_name) in cases {
        let cassette = load_codex_cassette(filename);
        assert_eq!(cassette.turns.len(), 2, "{filename} should have two turns");

        let output = process_completed_response_object_from_sse(&cassette, 0, "Qwen/Qwen3.6-35B-A3B");
        let call = first_function_call(&output);
        assert_eq!(call.namespace.as_deref(), expected_namespace, "{filename} namespace");
        assert_eq!(call.name, expected_name, "{filename} function name");
        assert_eq!(call.status, MessageStatus::Completed);
        assert!(!call.call_id.is_empty(), "{filename} call_id must be populated");
        assert!(!call.arguments.is_empty(), "{filename} arguments must be populated");
    }
}

#[test]
fn test_codex_gateway_websocket_cassettes_preserve_function_and_namespace_calls() {
    let cases = [
        (
            "codex-gateway-websocket-function-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
            None,
            "agentic_plain_echo",
        ),
        (
            "codex-gateway-websocket-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
            Some("mcp__agentic_fixture"),
            "add_numbers",
        ),
    ];

    for (filename, expected_namespace, expected_name) in cases {
        let cassette = load_codex_cassette(filename);
        assert_eq!(cassette.turns.len(), 2, "{filename} should have two turns");

        let output = process_codex_streaming_turn(&cassette, 0, "Qwen/Qwen3.6-35B-A3B");
        let call = first_function_call(&output);
        assert_eq!(call.namespace.as_deref(), expected_namespace, "{filename} namespace");
        assert_eq!(call.name, expected_name, "{filename} function name");
        assert_eq!(call.status, MessageStatus::Completed);
    }
}

#[test]
fn test_codex_custom_tool_cassettes_preserve_raw_input() {
    let gateway_http = load_codex_cassette("codex-gateway-http-custom-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml");
    let gateway_http_output = process_completed_response_object_from_sse(&gateway_http, 0, "Qwen/Qwen3.6-35B-A3B");

    let gateway_websocket =
        load_codex_cassette("codex-gateway-websocket-custom-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml");
    let gateway_websocket_output = process_codex_streaming_turn(&gateway_websocket, 0, "Qwen/Qwen3.6-35B-A3B");

    let direct_vllm = load_codex_cassette("codex-direct-vllm-http-custom-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml");
    let direct_vllm_output = process_codex_streaming_turn(&direct_vllm, 0, "Qwen/Qwen3.6-35B-A3B");

    let openai_https = load_codex_cassette("codex-openai-https-custom-tool-gpt-5.6-streaming.yaml");
    let openai_https_output = process_codex_streaming_turn(&openai_https, 0, "gpt-5.6");

    let openai_websocket = load_codex_cassette("codex-openai-websocket-custom-tool-gpt-5.6-streaming.yaml");
    let openai_websocket_output = process_codex_streaming_turn(&openai_websocket, 0, "gpt-5.6");

    for (label, output) in [
        ("gateway HTTP", gateway_http_output),
        ("gateway WebSocket", gateway_websocket_output),
        ("direct vLLM HTTP", direct_vllm_output),
        ("OpenAI HTTPS", openai_https_output),
        ("OpenAI WebSocket", openai_websocket_output),
    ] {
        let call = first_custom_tool_call(&output);
        assert_eq!(call.name, "agentic_raw_echo", "{label} custom tool name");
        assert_eq!(call.input, "CUSTOM_CASSETTE_OK", "{label} raw custom input");
        assert_eq!(call.status, Some(MessageStatus::Completed), "{label} custom status");
        assert!(!call.call_id.is_empty(), "{label} call_id must be populated");
    }
}

#[test]
fn test_codex_direct_vllm_http_cassettes_capture_upstream_tool_shapes() {
    let cases = [
        (
            "codex-direct-vllm-http-function-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
            None,
            "agentic_plain_echo",
        ),
        (
            "codex-direct-vllm-http-flat-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
            None,
            "agentic_ns__mcp__agentic_fixture__add_numbers",
        ),
    ];

    for (filename, expected_namespace, expected_name) in cases {
        let cassette = load_codex_cassette(filename);
        assert_eq!(cassette.turns.len(), 2, "{filename} should have two turns");

        let output = process_codex_streaming_turn(&cassette, 0, "Qwen/Qwen3.6-35B-A3B");
        let call = first_function_call(&output);
        assert_eq!(call.namespace.as_deref(), expected_namespace, "{filename} namespace");
        assert_eq!(call.name, expected_name, "{filename} function name");
        assert_eq!(call.status, MessageStatus::Completed);
    }
}

#[test]
fn test_codex_openai_baseline_cassettes_accept_namespace_on_http_and_websocket() {
    let cases = [
        (
            "codex-openai-https-function-tool-gpt-4o-streaming.yaml",
            None,
            "agentic_plain_echo",
        ),
        (
            "codex-openai-https-namespace-tool-gpt-4o-streaming.yaml",
            Some("mcp__agentic_fixture"),
            "add_numbers",
        ),
        (
            "codex-openai-websocket-function-tool-gpt-4o-streaming.yaml",
            None,
            "agentic_plain_echo",
        ),
        (
            "codex-openai-websocket-namespace-tool-gpt-4o-streaming.yaml",
            Some("mcp__agentic_fixture"),
            "add_numbers",
        ),
    ];

    for (filename, expected_namespace, expected_name) in cases {
        let cassette = load_codex_cassette(filename);
        assert_eq!(cassette.turns.len(), 2, "{filename} should have two turns");

        let output = process_codex_streaming_turn(&cassette, 0, "gpt-4o");
        let call = first_function_call(&output);
        assert_eq!(call.namespace.as_deref(), expected_namespace, "{filename} namespace");
        assert_eq!(call.name, expected_name, "{filename} function name");
        assert_eq!(call.status, MessageStatus::Completed);
    }
}

#[test]
fn test_codex_cassette_second_turns_are_tool_output_continuations() {
    let cassettes = [
        (
            "codex-direct-vllm-http-custom-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
            "custom_tool_call_output",
        ),
        (
            "codex-direct-vllm-http-flat-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
            "function_call_output",
        ),
        (
            "codex-direct-vllm-http-function-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
            "function_call_output",
        ),
        (
            "codex-gateway-http-custom-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
            "custom_tool_call_output",
        ),
        (
            "codex-gateway-http-function-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
            "function_call_output",
        ),
        (
            "codex-gateway-http-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
            "function_call_output",
        ),
        (
            "codex-gateway-websocket-custom-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
            "custom_tool_call_output",
        ),
        (
            "codex-gateway-websocket-function-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
            "function_call_output",
        ),
        (
            "codex-gateway-websocket-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
            "function_call_output",
        ),
        (
            "codex-openai-https-function-tool-gpt-4o-streaming.yaml",
            "function_call_output",
        ),
        (
            "codex-openai-https-custom-tool-gpt-5.6-streaming.yaml",
            "custom_tool_call_output",
        ),
        (
            "codex-openai-https-namespace-tool-gpt-4o-streaming.yaml",
            "function_call_output",
        ),
        (
            "codex-openai-websocket-function-tool-gpt-4o-streaming.yaml",
            "function_call_output",
        ),
        (
            "codex-openai-websocket-custom-tool-gpt-5.6-streaming.yaml",
            "custom_tool_call_output",
        ),
        (
            "codex-openai-websocket-namespace-tool-gpt-4o-streaming.yaml",
            "function_call_output",
        ),
    ];

    for (filename, expected_output_type) in cassettes {
        let cassette = load_codex_cassette(filename);
        assert_eq!(cassette.turns.len(), 2, "{filename} should have two turns");
        let turn2_body = turn_request_body(&cassette.turns[1]);
        assert!(
            turn2_body
                .get("previous_response_id")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|id| !id.is_empty()),
            "{filename} turn 2 must include previous_response_id"
        );

        let input = turn2_body
            .get("input")
            .and_then(serde_json::Value::as_array)
            .unwrap_or_else(|| panic!("{filename} turn 2 input must be an array"));
        assert!(
            input
                .iter()
                .any(|item| item.get("type").and_then(serde_json::Value::as_str) == Some(expected_output_type)),
            "{filename} turn 2 must include a {expected_output_type}"
        );
    }
}

// === Stateful multi-turn cassette tests (previous_response_id chaining) ===
//
// These cassettes are recorded against gpt-oss-20b with `store=true` and
// `previous_response_id` chaining (via record_cassette.py --mode responses).
// They exercise realistic multi-turn conversations where the server maintains
// conversation state — the key pattern our accumulator must handle for PR #67.
//
// Scenario: SRE debugging a failed ETL pipeline job-382.
// Tools: get_job_status, get_error_logs, search_runbook, run_analysis,
// restart_job, web_search.

const MULTI_TURN_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/tool_calls/multi_turn");

// --- Helpers ---

fn process_nonstreaming_turn(cassette: &TurnCassette, turn_idx: usize, model: &str) -> Vec<OutputItem> {
    let body = cassette.turns[turn_idx]
        .response
        .body
        .as_ref()
        .unwrap_or_else(|| panic!("turn {} must have response body", turn_idx + 1));
    let body_str = serde_json::to_string(body).unwrap();
    let acc = ResponseAccumulator::from_json(&body_str, None).unwrap();
    let payload = acc.finalize(model, None, None);
    assert_eq!(payload.status, "completed");
    payload.output
}

fn process_streaming_turn(cassette: &TurnCassette, turn_idx: usize, model: &str) -> Vec<OutputItem> {
    let data_lines = extract_data_lines(&cassette.turns[turn_idx].response.sse);
    assert!(
        !data_lines.is_empty(),
        "streaming turn {} must have SSE data lines",
        turn_idx + 1
    );
    let acc = from_sse_lines(data_lines, None);
    let payload = acc.finalize(model, None, None);
    assert_eq!(payload.status, "completed");
    payload.output
}

fn count_function_calls(output: &[OutputItem]) -> usize {
    output
        .iter()
        .filter(|item| matches!(item, OutputItem::FunctionCall(_)))
        .count()
}

fn get_function_call_names(output: &[OutputItem]) -> Vec<String> {
    output
        .iter()
        .filter_map(|item| {
            if let OutputItem::FunctionCall(fc) = item {
                Some(fc.name.clone())
            } else {
                None
            }
        })
        .collect()
}

fn has_reasoning(output: &[OutputItem]) -> bool {
    output.iter().any(|item| matches!(item, OutputItem::Reasoning(_)))
}

fn assert_completed_potato_web_search(output: &[OutputItem]) -> &WebSearchCall {
    assert_eq!(
        count_function_calls(output),
        0,
        "raw web_search function call should not leak"
    );
    let web_search = output
        .iter()
        .find_map(|item| match item {
            OutputItem::WebSearchCall(call) => Some(call),
            _ => None,
        })
        .expect("cassette output should include a web_search_call item");
    assert_eq!(web_search.status.as_str(), "completed");
    assert_eq!(web_search.action.as_search().unwrap().query, "potato");
    assert_eq!(web_search.action.type_str(), "search");
    assert!(web_search.id.starts_with("ws_"));
    assert!(
        output.iter().any(|item| matches!(item, OutputItem::Message(_))),
        "cassette output should include a final assistant message"
    );
    web_search
}

fn assert_matching_web_search_output(openai: &[OutputItem], gateway: &[OutputItem]) {
    for (provider, output) in [("OpenAI", openai), ("gateway", gateway)] {
        let public_types: Vec<&str> = output
            .iter()
            .filter_map(|item| match item {
                OutputItem::Reasoning(_) => None,
                OutputItem::WebSearchCall(_) => Some("web_search_call"),
                OutputItem::Message(_) => Some("message"),
                _ => Some("unexpected"),
            })
            .collect();
        assert_eq!(
            public_types,
            ["web_search_call", "message"],
            "{provider} output should preserve web_search_call before message"
        );
    }

    let openai_call = assert_completed_potato_web_search(openai);
    let gateway_call = assert_completed_potato_web_search(gateway);
    assert_eq!(gateway_call.status, openai_call.status);
    assert_eq!(gateway_call.action.type_str(), openai_call.action.type_str());

    let openai_action = openai_call.action.as_search().unwrap();
    let gateway_action = gateway_call.action.as_search().unwrap();
    assert_eq!(gateway_action.query, openai_action.query);
    assert_eq!(openai_action.queries, ["potato"]);
    assert!(
        gateway_action
            .sources
            .iter()
            .any(|source| source.url == "https://en.wikipedia.org/wiki/Potato"),
        "gateway web_search_call should retain its recorded sources"
    );
}

/// Extracts the `arguments` JSON string from the first function call in output items.
fn get_first_fc_arguments(output: &[OutputItem]) -> String {
    output
        .iter()
        .find_map(|item| {
            if let OutputItem::FunctionCall(fc) = item {
                Some(fc.arguments.clone())
            } else {
                None
            }
        })
        .expect("output must contain at least one function call")
}

#[test]
fn test_web_search_accumulator_nonstreaming_matches_openai() {
    let (openai, gateway) = load_web_search_cassette_pair(false);
    let openai_output = process_nonstreaming_turn(&openai, 0, WEB_SEARCH_OPENAI_MODEL);
    let gateway_output = process_nonstreaming_turn(&gateway, 0, WEB_SEARCH_GATEWAY_MODEL);

    assert!(
        has_reasoning(&openai_output),
        "reasoning-capable OpenAI reference should include reasoning"
    );
    assert!(
        has_reasoning(&gateway_output),
        "gateway output should include reasoning"
    );
    assert_matching_web_search_output(&openai_output, &gateway_output);
}

#[test]
fn test_web_search_accumulator_streaming_matches_openai() {
    let (openai, gateway) = load_web_search_cassette_pair(true);
    let openai_output = process_streaming_turn(&openai, 0, WEB_SEARCH_OPENAI_MODEL);
    let gateway_output = process_streaming_turn(&gateway, 0, WEB_SEARCH_GATEWAY_MODEL);

    assert!(
        has_reasoning(&gateway_output),
        "gateway output should preserve reasoning around web_search_call"
    );
    assert_matching_web_search_output(&openai_output, &gateway_output);
}

#[test]
fn test_custom_tool_cassettes_accumulate_public_streaming_shape() {
    let cases = [
        (
            "gateway",
            "custom-tool-gateway-Qwen-Qwen3.5-35B-A3B-FP8-streaming.yaml",
            "Qwen/Qwen3.5-35B-A3B-FP8",
        ),
        (
            "OpenAI",
            "custom-tool-openai-reference-gpt-5.6-streaming.yaml",
            "gpt-5.6",
        ),
    ];

    for (provider, filename, model) in cases {
        let cassette = load_custom_tool_cassette(filename);
        assert_eq!(cassette.turns.len(), 2, "{provider} cassette should contain two turns");

        let output = process_codex_streaming_turn(&cassette, 0, model);
        assert!(
            !output.iter().any(|item| matches!(item, OutputItem::FunctionCall(_))),
            "{provider} must not expose the normalized function call"
        );
        let call = first_custom_tool_call(&output);
        assert!(call.id.starts_with("ctc_"), "{provider} custom item ID");
        assert!(!call.call_id.is_empty(), "{provider} custom call ID");
        assert_eq!(call.name, "agentic_raw_echo", "{provider} custom tool name");
        assert_eq!(call.input.trim(), "CUSTOM_CASSETTE_OK", "{provider} custom input");
        assert_eq!(call.status, Some(MessageStatus::Completed), "{provider} custom status");

        let continuation = process_codex_streaming_turn(&cassette, 1, model);
        assert!(continuation.iter().any(|item| {
            matches!(
                item,
                OutputItem::Message(message)
                    if message.content.iter().any(|content| content.text.contains("CUSTOM_CASSETTE_OUTPUT_OK"))
            )
        }));
    }
}

#[test]
fn test_custom_tool_cassettes_accumulate_public_nonstreaming_shape() {
    let cases = [
        (
            "gateway",
            "custom-tool-gateway-Qwen-Qwen3.5-35B-A3B-FP8-nonstreaming.yaml",
            "Qwen/Qwen3.5-35B-A3B-FP8",
        ),
        (
            "OpenAI",
            "custom-tool-openai-reference-gpt-5.6-nonstreaming.yaml",
            "gpt-5.6",
        ),
    ];

    for (provider, filename, model) in cases {
        let cassette = load_custom_tool_cassette(filename);
        assert_eq!(cassette.turns.len(), 2, "{provider} cassette should contain two turns");

        let output = process_nonstreaming_turn(&cassette, 0, model);
        assert!(
            !output.iter().any(|item| matches!(item, OutputItem::FunctionCall(_))),
            "{provider} must not expose the normalized function call"
        );
        let call = first_custom_tool_call(&output);
        assert!(call.id.starts_with("ctc_"), "{provider} custom item ID");
        assert!(!call.call_id.is_empty(), "{provider} custom call ID");
        assert_eq!(call.name, "agentic_raw_echo", "{provider} custom tool name");
        assert_eq!(call.input.trim(), "CUSTOM_CASSETTE_OK", "{provider} custom input");
        assert_eq!(call.status, Some(MessageStatus::Completed), "{provider} custom status");

        let continuation = process_nonstreaming_turn(&cassette, 1, model);
        assert!(continuation.iter().any(|item| {
            matches!(
                item,
                OutputItem::Message(message)
                    if message.content.iter().any(|content| content.text.contains("CUSTOM_CASSETTE_OUTPUT_OK"))
            )
        }));
    }
}

// ═══════════════════════════════════════════════════════════════════
// Stateful 3-turn: get_job_status → get_error_logs → search_runbook
// Non-streaming, store=true, previous_response_id chain
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_stateful_responses_3turn_tool_calls() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "responses_tool_calls_3turn.yaml");

    let t1 = process_nonstreaming_turn(&cassette, 0, "openai/gpt-oss-20b");
    let t1_names = get_function_call_names(&t1);
    assert_eq!(count_function_calls(&t1), 1);
    assert_eq!(t1_names, vec!["get_job_status"]);
    assert!(has_reasoning(&t1));

    let t2 = process_nonstreaming_turn(&cassette, 1, "openai/gpt-oss-20b");
    let t2_names = get_function_call_names(&t2);
    assert_eq!(count_function_calls(&t2), 1);
    assert_eq!(t2_names, vec!["get_error_logs"]);

    let t3 = process_nonstreaming_turn(&cassette, 2, "openai/gpt-oss-20b");
    let t3_names = get_function_call_names(&t3);
    assert_eq!(count_function_calls(&t3), 1);
    assert_eq!(t3_names, vec!["search_runbook"]);
}

/// Context retention proof: turn 2 prompt says "that job" (no explicit job ID),
/// but the model resolves it to "job-382" because `previous_response_id` gives
/// it access to turn 1's conversation state.
#[test]
fn test_stateful_responses_3turn_context_retention() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "responses_tool_calls_3turn.yaml");

    // Turn 2 prompt says "that job" — model must resolve from turn 1 context
    let t2 = process_nonstreaming_turn(&cassette, 1, "openai/gpt-oss-20b");
    let t2_args = get_first_fc_arguments(&t2);
    assert!(
        t2_args.contains("job-382"),
        "turn 2 must resolve 'that job' to 'job-382' via retained context, got: {t2_args}"
    );

    // Turn 3 prompt says "those errors" — model must recall turn 2's investigation
    let t3 = process_nonstreaming_turn(&cassette, 2, "openai/gpt-oss-20b");
    let t3_args = get_first_fc_arguments(&t3);
    assert!(
        t3_args.contains("job-382") || t3_args.contains("error") || t3_args.contains("ETL"),
        "turn 3 must reference context from earlier turns, got: {t3_args}"
    );
}

#[test]
fn test_stateful_responses_3turn_null_status_deserialization() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "responses_tool_calls_3turn.yaml");
    for i in 0..3 {
        let output = process_nonstreaming_turn(&cassette, i, "openai/gpt-oss-20b");
        for item in &output {
            if let OutputItem::FunctionCall(fc) = item {
                assert_eq!(
                    fc.status,
                    MessageStatus::Completed,
                    "turn {} function_call status must default to Completed (gpt-oss emits null)",
                    i + 1
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Stateful 5-turn: full investigation pipeline
// get_job_status → get_error_logs → search_runbook → run_analysis → restart_job
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_stateful_responses_5turn_tool_sequence() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "responses_tool_calls_5turn.yaml");

    let expected_tools = [
        "get_job_status",
        "get_error_logs",
        "search_runbook",
        "run_analysis",
        "restart_job",
    ];
    for (i, expected) in expected_tools.iter().enumerate() {
        let output = process_nonstreaming_turn(&cassette, i, "openai/gpt-oss-20b");
        let names = get_function_call_names(&output);
        assert_eq!(names.len(), 1, "turn {} should call exactly 1 tool", i + 1);
        assert_eq!(&names[0], expected, "turn {} should call {expected}", i + 1);
        assert!(has_reasoning(&output), "turn {} should have reasoning", i + 1);
    }
}

/// Context retention proof for 5-turn: turn 5 says "restart it" without naming
/// job-382, but the model resolves correctly because all prior context is retained.
#[test]
fn test_stateful_responses_5turn_context_retention() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "responses_tool_calls_5turn.yaml");

    // Turn 2: "that failed job" → must resolve to job-382
    let t2 = process_nonstreaming_turn(&cassette, 1, "openai/gpt-oss-20b");
    let t2_args = get_first_fc_arguments(&t2);
    assert!(
        t2_args.contains("job-382"),
        "turn 2 'that failed job' must resolve to job-382, got: {t2_args}"
    );

    // Turn 5: "restart it" → must resolve to job-382 with correct params
    let t5 = process_nonstreaming_turn(&cassette, 4, "openai/gpt-oss-20b");
    let t5_args = get_first_fc_arguments(&t5);
    assert!(
        t5_args.contains("job-382"),
        "turn 5 'restart it' must resolve to job-382, got: {t5_args}"
    );
    assert!(
        t5_args.contains("64"),
        "turn 5 must include memory_override_gb=64, got: {t5_args}"
    );
}

#[test]
fn test_stateful_responses_5turn_function_call_fields() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "responses_tool_calls_5turn.yaml");
    for i in 0..5 {
        let output = process_nonstreaming_turn(&cassette, i, "openai/gpt-oss-20b");
        for item in &output {
            if let OutputItem::FunctionCall(fc) = item {
                assert!(!fc.id.is_empty(), "turn {} fc.id must not be empty", i + 1);
                assert!(!fc.call_id.is_empty(), "turn {} fc.call_id must not be empty", i + 1);
                assert!(!fc.name.is_empty(), "turn {} fc.name must not be empty", i + 1);
                assert!(
                    !fc.arguments.is_empty(),
                    "turn {} fc.arguments must not be empty",
                    i + 1
                );
                assert_eq!(fc.status, MessageStatus::Completed);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Stateful 3-turn streaming: SSE events with previous_response_id
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_stateful_responses_streaming_3turn() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "responses_tool_calls_3turn_streaming.yaml");
    assert_eq!(cassette.turns.len(), 3);

    for i in 0..3 {
        let output = process_streaming_turn(&cassette, i, "openai/gpt-oss-20b");
        assert!(
            count_function_calls(&output) >= 1,
            "streaming turn {} must produce at least one function_call",
            i + 1
        );
        for item in &output {
            if let OutputItem::FunctionCall(fc) = item {
                assert!(!fc.call_id.is_empty(), "streaming fc must have call_id");
                assert!(!fc.name.is_empty(), "streaming fc must have name");
                assert!(!fc.arguments.is_empty(), "streaming fc must have arguments");
                assert_eq!(fc.status, MessageStatus::Completed);
            }
        }
    }
}

/// Context retention in streaming mode: turn 2 says "that job" and the model
/// resolves it to "job-382" even in streaming (SSE) delivery.
#[test]
fn test_stateful_responses_streaming_context_retention() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "responses_tool_calls_3turn_streaming.yaml");

    // Turn 2: "that job" → must resolve to job-382 in streaming mode
    let t2 = process_streaming_turn(&cassette, 1, "openai/gpt-oss-20b");
    let t2_args = get_first_fc_arguments(&t2);
    assert!(
        t2_args.contains("job-382"),
        "streaming turn 2 must resolve 'that job' to job-382, got: {t2_args}"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Branching: turn 3 diverges from turn 1 (not turn 2)
// Tests previous_response_id pointing back to an earlier response
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_stateful_responses_branch_divergence() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "responses_tool_calls_branch.yaml");
    assert_eq!(cassette.turns.len(), 3);

    // Turn 1: no prev_id
    let body1 = cassette.turns[0].request.as_mapping().unwrap();
    let req1 = body1
        .get(serde_yml::Value::String("body".into()))
        .and_then(serde_yml::Value::as_mapping)
        .unwrap();
    let prev1 = req1.get(serde_yml::Value::String("previous_response_id".into()));
    assert!(prev1.is_none() || prev1.unwrap().is_null());

    // Turn 2: prev_id = turn 1's response id
    let body2 = cassette.turns[1].request.as_mapping().unwrap();
    let req2 = body2
        .get(serde_yml::Value::String("body".into()))
        .and_then(serde_yml::Value::as_mapping)
        .unwrap();
    let prev2 = req2
        .get(serde_yml::Value::String("previous_response_id".into()))
        .and_then(serde_yml::Value::as_str)
        .expect("turn 2 must have prev_id");

    // Turn 3: prev_id = turn 1's response id (branches back, NOT from turn 2)
    let body3 = cassette.turns[2].request.as_mapping().unwrap();
    let req3 = body3
        .get(serde_yml::Value::String("body".into()))
        .and_then(serde_yml::Value::as_mapping)
        .unwrap();
    let prev3 = req3
        .get(serde_yml::Value::String("previous_response_id".into()))
        .and_then(serde_yml::Value::as_str)
        .expect("turn 3 must have prev_id");

    // Turn 2 and Turn 3 both point to the same response (turn 1)
    assert_eq!(
        prev2, prev3,
        "branch: turn 3 should reference same prev_id as turn 2 (turn 1's response)"
    );
}

#[test]
fn test_stateful_responses_branch_all_turns_parse() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "responses_tool_calls_branch.yaml");
    for i in 0..3 {
        let output = process_nonstreaming_turn(&cassette, i, "openai/gpt-oss-20b");
        assert!(
            count_function_calls(&output) >= 1,
            "branch turn {} must produce a function_call",
            i + 1
        );
        assert!(has_reasoning(&output), "branch turn {} should have reasoning", i + 1);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Cross-cassette: all stateful cassettes parse without error
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// Tool-output-only turn: model responds autonomously with text
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_stateful_responses_tool_output_only_produces_text() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "responses_tool_calls_tool_output_only.yaml");
    assert_eq!(cassette.turns.len(), 3);

    // Turn 2 has tool output only (no user message) → model should produce text
    let t2 = process_nonstreaming_turn(&cassette, 1, "openai/gpt-oss-20b");
    let has_text = t2.iter().any(|item| matches!(item, OutputItem::Message(_)));
    assert!(has_text, "tool-output-only turn should produce a text response");
}

// ═══════════════════════════════════════════════════════════════════
// Parallel tool calls (OpenAI only — gpt-4o reliably produces these)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_openai_parallel_tool_calls() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "openai_responses_tool_calls_parallel.yaml");
    assert_eq!(cassette.turns.len(), 3);

    // Turn 1 should have 2 parallel function calls
    let t1 = process_nonstreaming_turn(&cassette, 0, "gpt-4o");
    let t1_names = get_function_call_names(&t1);
    assert!(
        t1_names.len() >= 2,
        "parallel cassette turn 1 must have 2+ function calls, got: {t1_names:?}"
    );
    assert!(t1_names.contains(&"get_job_status".to_string()));
    assert!(t1_names.contains(&"web_search".to_string()));
}

/// Verifies that the request input for turn 2 contains multiple `function_call_output`
/// items (one per parallel call from turn 1).
#[test]
fn test_openai_parallel_tool_outputs_in_request() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "openai_responses_tool_calls_parallel.yaml");

    let body2 = cassette.turns[1].request.as_mapping().unwrap();
    let req2 = body2
        .get(serde_yml::Value::String("body".into()))
        .and_then(serde_yml::Value::as_mapping)
        .unwrap();
    let input2 = req2
        .get(serde_yml::Value::String("input".into()))
        .expect("turn 2 must have input");
    let input_seq = input2.as_sequence().expect("turn 2 input must be a list");

    let tool_outputs: Vec<_> = input_seq
        .iter()
        .filter(|item| {
            item.as_mapping()
                .and_then(|m| m.get(serde_yml::Value::String("type".into())))
                .and_then(serde_yml::Value::as_str)
                == Some("function_call_output")
        })
        .collect();

    assert!(
        tool_outputs.len() >= 2,
        "turn 2 input must contain 2+ function_call_output items for parallel calls, got {}",
        tool_outputs.len()
    );
}

// ═══════════════════════════════════════════════════════════════════
// OpenAI cassettes: verify they parse identically to vLLM
// (status is "completed" string, not null)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_openai_3turn_parses_and_retains_context() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "openai_responses_tool_calls_3turn.yaml");
    assert_eq!(cassette.turns.len(), 3);

    let t1 = process_nonstreaming_turn(&cassette, 0, "gpt-4o");
    assert_eq!(get_function_call_names(&t1), vec!["get_job_status"]);

    // Context retention: turn 2 says "that job"
    let t2 = process_nonstreaming_turn(&cassette, 1, "gpt-4o");
    let t2_args = get_first_fc_arguments(&t2);
    assert!(
        t2_args.contains("job-382"),
        "OpenAI turn 2 must resolve 'that job' to job-382, got: {t2_args}"
    );
}

#[test]
fn test_openai_5turn_full_sequence() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "openai_responses_tool_calls_5turn.yaml");
    assert_eq!(cassette.turns.len(), 5);

    let expected_tools = [
        "get_job_status",
        "get_error_logs",
        "search_runbook",
        "run_analysis",
        "restart_job",
    ];
    for (i, expected) in expected_tools.iter().enumerate() {
        let output = process_nonstreaming_turn(&cassette, i, "gpt-4o");
        let names = get_function_call_names(&output);
        assert_eq!(names.len(), 1, "OpenAI turn {} should call 1 tool", i + 1);
        assert_eq!(&names[0], expected, "OpenAI turn {} should call {expected}", i + 1);
    }
}

#[test]
fn test_openai_streaming_3turn() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "openai_responses_tool_calls_3turn_streaming.yaml");
    assert_eq!(cassette.turns.len(), 3);

    for i in 0..3 {
        let output = process_streaming_turn(&cassette, i, "gpt-4o");
        assert!(
            count_function_calls(&output) >= 1,
            "OpenAI streaming turn {} must produce a function_call",
            i + 1
        );
    }
}

#[test]
fn test_openai_branch_divergence() {
    let cassette = load_turn_cassette_from(MULTI_TURN_DIR, "openai_responses_tool_calls_branch.yaml");
    assert_eq!(cassette.turns.len(), 3);

    let body2 = cassette.turns[1].request.as_mapping().unwrap();
    let req2 = body2
        .get(serde_yml::Value::String("body".into()))
        .and_then(serde_yml::Value::as_mapping)
        .unwrap();
    let prev2 = req2
        .get(serde_yml::Value::String("previous_response_id".into()))
        .and_then(serde_yml::Value::as_str)
        .expect("turn 2 must have prev_id");

    let body3 = cassette.turns[2].request.as_mapping().unwrap();
    let req3 = body3
        .get(serde_yml::Value::String("body".into()))
        .and_then(serde_yml::Value::as_mapping)
        .unwrap();
    let prev3 = req3
        .get(serde_yml::Value::String("previous_response_id".into()))
        .and_then(serde_yml::Value::as_str)
        .expect("turn 3 must have prev_id");

    assert_eq!(
        prev2, prev3,
        "OpenAI branch: turn 3 must branch from turn 1 (same prev_id as turn 2)"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Cross-cassette: ALL stateful cassettes parse without error
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_all_stateful_cassettes_parse_without_error() {
    let nonstreaming = [
        "responses_tool_calls_3turn.yaml",
        "responses_tool_calls_5turn.yaml",
        "responses_tool_calls_branch.yaml",
        "responses_tool_calls_parallel.yaml",
        "responses_tool_calls_tool_output_only.yaml",
        "openai_responses_tool_calls_3turn.yaml",
        "openai_responses_tool_calls_5turn.yaml",
        "openai_responses_tool_calls_branch.yaml",
        "openai_responses_tool_calls_parallel.yaml",
        "openai_responses_tool_calls_tool_output_only.yaml",
    ];

    for filename in &nonstreaming {
        let cassette = load_turn_cassette_from(MULTI_TURN_DIR, filename);
        for i in 0..cassette.turns.len() {
            let body = cassette.turns[i]
                .response
                .body
                .as_ref()
                .unwrap_or_else(|| panic!("{filename} turn {i} must have body"));
            let body_str = serde_json::to_string(body).unwrap();
            let result = ResponseAccumulator::from_json(&body_str, None);
            assert!(
                result.is_ok(),
                "{filename} turn {} failed to parse: {:?}",
                i + 1,
                result.err()
            );
            let payload = result.unwrap().finalize("gpt-4o", None, None);
            assert_eq!(
                payload.status,
                "completed",
                "{filename} turn {} status != completed",
                i + 1
            );
        }
    }

    let streaming = [
        "responses_tool_calls_3turn_streaming.yaml",
        "openai_responses_tool_calls_3turn_streaming.yaml",
    ];
    for filename in &streaming {
        let cassette = load_turn_cassette_from(MULTI_TURN_DIR, filename);
        for i in 0..cassette.turns.len() {
            let data_lines = extract_data_lines(&cassette.turns[i].response.sse);
            assert!(
                !data_lines.is_empty(),
                "{filename} turn {} has no SSE data lines",
                i + 1
            );
            let acc = from_sse_lines(data_lines, None);
            let payload = acc.finalize("gpt-4o", None, None);
            assert_eq!(
                payload.status,
                "completed",
                "{filename} turn {} status != completed",
                i + 1
            );
        }
    }
}
