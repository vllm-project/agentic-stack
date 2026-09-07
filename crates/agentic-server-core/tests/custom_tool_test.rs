use std::collections::HashSet;

use agentic_core::executor::accumulator::ResponseAccumulator;
use agentic_core::tool::{GatewayExecutors, ToolOwnership, ToolRegistry, ToolType};
use agentic_core::types::event::MessageStatus;
use agentic_core::types::io::{CustomToolCall, OutputItem};
use agentic_core::types::tools::ResponsesTool;
use serde_json::{Value, json};

mod support;

const MODEL: &str = "Qwen/Qwen3.5-35B-A3B-FP8";
const CUSTOM_TOOL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/custom_tool");
const GATEWAY_MODEL_SLUG: &str = "Qwen-Qwen3.5-35B-A3B-FP8";
const OPENAI_MODEL_SLUG: &str = "gpt-5.6";

fn load_custom_tool_cassette(filename: &str) -> support::Cassette {
    support::load_cassette(&format!("{CUSTOM_TOOL_DIR}/{filename}"))
}

fn load_pair(streaming: bool) -> (support::Cassette, support::Cassette) {
    let mode = if streaming { "streaming" } else { "nonstreaming" };
    let openai = load_custom_tool_cassette(&format!("custom-tool-openai-reference-{OPENAI_MODEL_SLUG}-{mode}.yaml"));
    let gateway = load_custom_tool_cassette(&format!("custom-tool-gateway-{GATEWAY_MODEL_SLUG}-{mode}.yaml"));
    (openai, gateway)
}

fn streaming_events(turn: &support::Turn) -> Vec<Value> {
    support::recorded_named_sse_events(turn)
}

fn response_output(turn: &support::Turn) -> Vec<OutputItem> {
    let response = if let Some(body) = &turn.response.body {
        body.clone()
    } else {
        streaming_events(turn)
            .into_iter()
            .rev()
            .filter_map(|event| event.get("response").cloned())
            .find(|response| response["status"] == "completed" && response["output"].is_array())
            .expect("completed streaming response payload")
    };
    let accumulator = ResponseAccumulator::from_json(&response.to_string(), None).expect("valid completed response");
    let payload = accumulator.finalize(MODEL, None, None);
    assert_eq!(payload.status, "completed");
    payload.output
}

fn custom_call(output: &[OutputItem]) -> &CustomToolCall {
    assert!(
        !output.iter().any(|item| matches!(item, OutputItem::FunctionCall(_))),
        "normalized function calls must not leak through the public response"
    );
    let calls = output
        .iter()
        .filter_map(|item| match item {
            OutputItem::CustomToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 1, "expected exactly one custom tool call");
    calls[0]
}

fn output_text(output: &[OutputItem]) -> String {
    output
        .iter()
        .filter_map(|item| match item {
            OutputItem::Message(message) => Some(
                message
                    .content
                    .iter()
                    .map(|content| content.text.as_str())
                    .collect::<String>(),
            ),
            _ => None,
        })
        .collect::<String>()
        .trim()
        .to_owned()
}

fn assert_request_contract(cassette: &support::Cassette, streaming: bool) {
    assert_eq!(cassette.turns.len(), 2);

    for turn in &cassette.turns {
        assert_eq!(turn.request.path, "/v1/responses");
        assert_eq!(turn.request.body.stream, streaming);
        assert_eq!(turn.request.body.tools.len(), 1);
        let tool = &turn.request.body.tools[0];
        assert_eq!(tool["type"], "custom");
        assert_eq!(tool["name"], "agentic_raw_echo");
        assert!(tool.get("format").is_none());
    }

    let continuation = cassette.turns[1]
        .request
        .body
        .input
        .as_array()
        .expect("continuation input array");
    assert_eq!(continuation[0]["type"], "custom_tool_call_output");
    assert_eq!(continuation[0]["output"], "CUSTOM_CASSETTE_OUTPUT_OK");
    assert_eq!(continuation[1]["type"], "message");
    assert!(
        continuation[1]["content"]
            .as_str()
            .is_some_and(|content| content.contains("CUSTOM_CASSETTE_OUTPUT_OK"))
    );
}

fn assert_public_calls_match(openai: &support::Cassette, gateway: &support::Cassette) {
    let expected_output = response_output(&openai.turns[0]);
    let actual_output = response_output(&gateway.turns[0]);
    let expected = custom_call(&expected_output);
    let actual = custom_call(&actual_output);

    for call in [expected, actual] {
        assert!(call.id.starts_with("ctc_"));
        assert!(!call.call_id.is_empty());
        assert_eq!(call.name, "agentic_raw_echo");
        assert_eq!(call.input.trim(), "CUSTOM_CASSETTE_OK");
        assert_eq!(call.status, Some(MessageStatus::Completed));
    }
    assert_eq!(actual.name, expected.name);
    assert_eq!(actual.input.trim(), expected.input.trim());
    assert_eq!(actual.status, expected.status);

    assert_eq!(
        gateway.turns[1].request.body.input[0]["call_id"].as_str(),
        Some(actual.call_id.as_str())
    );
    assert_eq!(
        openai.turns[1].request.body.input[0]["call_id"].as_str(),
        Some(expected.call_id.as_str())
    );
    assert_eq!(
        output_text(&response_output(&gateway.turns[1])),
        output_text(&response_output(&openai.turns[1]))
    );
}

fn normalized_custom_lifecycle(events: &[Value]) -> Value {
    let added = events
        .iter()
        .find(|event| event["type"] == "response.output_item.added" && event["item"]["type"] == "custom_tool_call")
        .expect("custom output item added");
    let item_id = added["item"]["id"].as_str().expect("custom item ID");
    let mut lifecycle = Vec::new();
    let mut deltas = Vec::new();
    let mut input = String::new();
    let mut lifecycle_item_ids = HashSet::new();
    let mut done_item = None;

    for event in events {
        let event_type = event["type"].as_str().unwrap_or_default();
        let event_item_id = event["item_id"].as_str().or_else(|| event["item"]["id"].as_str());
        if event_item_id != Some(item_id) {
            continue;
        }
        lifecycle_item_ids.insert(event_item_id.unwrap().to_owned());
        match event_type {
            "response.output_item.added" if event["item"]["type"] == "custom_tool_call" => {
                lifecycle.push(event_type);
            }
            "response.custom_tool_call_input.delta" => {
                let delta = event["delta"].as_str().unwrap_or_default();
                deltas.push(delta);
                input.push_str(delta);
                if lifecycle.last().copied() != Some(event_type) {
                    lifecycle.push(event_type);
                }
            }
            "response.custom_tool_call_input.done" => lifecycle.push(event_type),
            "response.output_item.done" if event["item"]["type"] == "custom_tool_call" => {
                lifecycle.push(event_type);
                done_item = Some(&event["item"]);
            }
            _ => {}
        }
    }

    let done_item = done_item.expect("custom output item done");
    assert_eq!(lifecycle_item_ids.len(), 1, "one public ID must span the lifecycle");
    json!({
        "lifecycle": lifecycle,
        "deltas": deltas,
        "delta_input": input,
        "added": {
            "type": added["item"]["type"],
            "name": added["item"]["name"],
            "status": added["item"]["status"],
            "input": added["item"]["input"],
        },
        "done": {
            "type": done_item["type"],
            "name": done_item["name"],
            "status": done_item["status"],
            "input": done_item["input"],
        }
    })
}

fn assert_contiguous_sequence_numbers(events: &[Value]) {
    let sequence_numbers = events
        .iter()
        .filter_map(|event| event["sequence_number"].as_u64())
        .collect::<Vec<_>>();
    assert!(
        sequence_numbers.windows(2).all(|pair| pair[1] == pair[0] + 1),
        "stream sequence numbers must be contiguous: {sequence_numbers:?}"
    );
}

#[tokio::test]
async fn custom_tool_type_normalizes_for_the_model_but_remains_client_owned() {
    let mut tools = vec![
        serde_json::from_value::<ResponsesTool>(serde_json::json!({
            "type": "custom",
            "name": "agentic_raw_echo",
            "description": "Emit raw text."
        }))
        .expect("custom declaration"),
    ];
    let registry = ToolRegistry::build_with_handlers(&mut tools, &mut GatewayExecutors::default())
        .await
        .expect("custom registry");
    let entry = registry.lookup("agentic_raw_echo").expect("custom entry");
    assert_eq!(entry.tool_type, ToolType::Custom);
    assert!(!entry.tool_type.is_gateway_owned());
    assert!(matches!(entry.ownership, ToolOwnership::Client));

    let normalized = tools[0].to_function_tools();
    assert_eq!(normalized.len(), 1);
    assert_eq!(normalized[0].type_, "function");
    assert_eq!(normalized[0].name, "agentic_raw_echo");
    assert_eq!(
        normalized[0].parameters.as_ref().unwrap()["properties"]["input"]["type"],
        "string"
    );
}

#[test]
fn custom_tool_grammar_format_is_rejected() {
    let tool = serde_json::from_value::<ResponsesTool>(serde_json::json!({
        "type": "custom",
        "name": "constrained_input",
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": "start: value"
        }
    }))
    .expect("custom declaration");

    let error = tool.validate().expect_err("unsupported grammar must fail closed");
    assert!(error.to_string().contains("cannot preserve constrained decoding"));
}

#[test]
fn streaming_custom_tool_contract_matches_openai() {
    let (openai, gateway) = load_pair(true);
    assert_request_contract(&openai, true);
    assert_request_contract(&gateway, true);
    assert_public_calls_match(&openai, &gateway);

    let expected_events = streaming_events(&openai.turns[0]);
    let actual_events = streaming_events(&gateway.turns[0]);
    assert_contiguous_sequence_numbers(&expected_events);
    assert_contiguous_sequence_numbers(&actual_events);
    assert_eq!(
        normalized_custom_lifecycle(&actual_events),
        normalized_custom_lifecycle(&expected_events)
    );
}

#[test]
fn nonstreaming_custom_tool_contract_matches_openai() {
    let (openai, gateway) = load_pair(false);
    assert_request_contract(&openai, false);
    assert_request_contract(&gateway, false);
    assert_public_calls_match(&openai, &gateway);
}
