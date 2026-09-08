mod support;

use std::sync::Arc;

use agentic_core::executor::{compact_response, create_conversation, execute};
use agentic_core::{CompactRequest, InputItem, RequestPayload};
use serde_json::{Value, json};
use support::{
    MockResponse, TestFixture, function_call_response, load_cassette, output_text, text_response, tool_search_request,
    unwrap_blocking,
};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/compaction");
const COMPACTION_PROMPT: &str = "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a concise handoff summary that preserves current progress, decisions, constraints, unresolved work, and critical references for the next model. Return only the summary.";

fn basic_source_input() -> Value {
    json!([
        {
            "type": "message",
            "role": "user",
            "content": "Remember the stable marker BANANA-CASSETTE for the next model."
        },
        {
            "type": "message",
            "role": "assistant",
            "content": "I will preserve BANANA-CASSETTE in the checkpoint."
        },
        {
            "type": "message",
            "role": "user",
            "content": "The implementation uses Rust and must keep offline replay tests."
        },
        {
            "type": "message",
            "role": "assistant",
            "content": "The constraints are OpenAI-only recordings and no network access in CI."
        }
    ])
}

fn tool_prior_source_input() -> Value {
    json!([
        {
            "type": "message",
            "id": "msg_retained",
            "role": "user",
            "status": "completed",
            "content": "Keep the stable marker BANANA-CASSETTE."
        },
        {
            "type": "compaction",
            "id": "cmp_prior",
            "encrypted_content": "Prior checkpoint: preserve BANANA-CASSETTE and continue the Rust replay work."
        },
        {
            "type": "message",
            "role": "user",
            "content": "Use the documentation lookup result when updating the tests."
        },
        {
            "type": "function_call",
            "id": "fc_lookup",
            "call_id": "call_lookup",
            "name": "lookup_docs",
            "arguments": "{\"topic\":\"compaction\"}",
            "status": "completed"
        },
        {
            "type": "function_call_output",
            "call_id": "call_lookup",
            "output": "{\"requirement\":\"record OpenAI replay cassettes\"}"
        }
    ])
}

fn compact_request(input: &Value) -> CompactRequest {
    serde_json::from_value(json!({
        "model": "gpt-4o",
        "input": input
    }))
    .expect("valid compact request")
}

fn response_request(input: &Value, context_management: Option<&Value>) -> RequestPayload {
    serde_json::from_value(json!({
        "model": "gpt-4o",
        "input": input,
        "stream": false,
        "store": false,
        "context_management": context_management
    }))
    .expect("valid Responses request")
}

fn tool_search_declarations() -> Value {
    json!([
        {
            "type": "tool_search",
            "execution": "client",
            "description": "Find a tool",
            "parameters": {"type": "object"}
        },
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object"},
            "defer_loading": true
        },
        {
            "type": "function",
            "name": "get_time",
            "description": "Get time",
            "parameters": {"type": "object"},
            "defer_loading": true
        }
    ])
}

fn completed_tool_search_history() -> Value {
    json!([
        {
            "type": "message",
            "role": "user",
            "content": "Find the weather tool"
        },
        {
            "type": "tool_search_call",
            "id": "tsc_search",
            "call_id": "call_search",
            "arguments": {"query": "weather"}
        },
        {
            "type": "tool_search_output",
            "call_id": "call_search",
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object"},
                "defer_loading": true
            }]
        }
    ])
}

fn cassette(name: &str) -> support::Cassette {
    load_cassette(&format!("{DIR}/{name}"))
}

fn cassette_response(turn: &support::Turn) -> MockResponse {
    MockResponse::from_turn(turn)
}

fn recorded_total_tokens(turn: &support::Turn) -> i64 {
    turn.response
        .body
        .as_ref()
        .and_then(|body| body["usage"]["total_tokens"].as_i64())
        .expect("recorded response usage")
}

#[tokio::test]
async fn standalone_compaction_replays_openai_summary() {
    let recording = cassette("compact-basic-gpt-4o-nonstreaming.yaml");
    let turn = &recording.turns[0];
    let fixture = TestFixture::new(&[turn]).await;

    let compacted = compact_response(compact_request(&basic_source_input()), &fixture.exec_ctx, None)
        .await
        .expect("compact recorded history");

    assert_eq!(compacted.object, "response.compaction");
    assert_eq!(
        compacted
            .output
            .iter()
            .filter(|item| matches!(item, InputItem::Message(_)))
            .count(),
        2
    );
    assert!(matches!(compacted.output.last(), Some(InputItem::Compaction(_))));

    let requests = fixture.request_bodies().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["input"], turn.request.body.input);
    assert_eq!(
        requests[0]["input"]
            .as_array()
            .and_then(|items| items.last())
            .expect("summary prompt item")["content"],
        COMPACTION_PROMPT
    );
}

#[tokio::test]
async fn repeated_compaction_drops_tool_pair_and_replaces_prior_checkpoint() {
    let recording = cassette("compact-tool-prior-gpt-4o-nonstreaming.yaml");
    let turn = &recording.turns[0];
    let fixture = TestFixture::new(&[turn]).await;

    let compacted = compact_response(compact_request(&tool_prior_source_input()), &fixture.exec_ctx, None)
        .await
        .expect("compact recorded tool history");

    assert_eq!(
        compacted
            .output
            .iter()
            .filter(|item| matches!(item, InputItem::Message(_)))
            .count(),
        2
    );
    assert_eq!(
        compacted
            .output
            .iter()
            .filter(|item| matches!(item, InputItem::Compaction(_)))
            .count(),
        1
    );
    assert!(
        compacted
            .output
            .iter()
            .all(|item| !matches!(item, InputItem::FunctionCall(_) | InputItem::FunctionCallOutput(_)))
    );

    let requests = fixture.request_bodies().await;
    assert_eq!(requests[0]["input"], turn.request.body.input);
}

#[tokio::test]
async fn compacted_window_round_trips_with_recorded_followup() {
    let summary = cassette("compact-basic-gpt-4o-nonstreaming.yaml");
    let followup = cassette("compact-followup-gpt-4o-nonstreaming.yaml");
    let fixture = TestFixture::new_with_responses(vec![
        cassette_response(&summary.turns[0]),
        cassette_response(&followup.turns[0]),
    ])
    .await;

    let compacted = compact_response(compact_request(&basic_source_input()), &fixture.exec_ctx, None)
        .await
        .expect("compact recorded history");
    let mut input = compacted.output;
    input.push(
        serde_json::from_value(json!({
            "type": "message",
            "role": "user",
            "content": "Reply with only the stable marker from the checkpoint."
        }))
        .expect("valid follow-up item"),
    );
    let response = unwrap_blocking(
        execute(
            response_request(&serde_json::to_value(input).expect("serialize compacted input"), None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("execute recorded follow-up"),
    );

    assert_eq!(output_text(&response).trim(), "BANANA-CASSETTE");
}

#[tokio::test]
async fn automatic_compaction_replays_both_rounds_and_accumulates_usage() {
    let summary = cassette("compact-basic-gpt-4o-nonstreaming.yaml");
    let followup = cassette("compact-followup-gpt-4o-nonstreaming.yaml");
    let expected_usage = recorded_total_tokens(&summary.turns[0]) + recorded_total_tokens(&followup.turns[0]);
    let fixture = TestFixture::new_with_responses(vec![
        cassette_response(&summary.turns[0]),
        cassette_response(&followup.turns[0]),
    ])
    .await;

    let response = unwrap_blocking(
        execute(
            response_request(
                &basic_source_input(),
                Some(&json!([{"type": "compaction", "compact_threshold": 1}])),
            ),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("execute automatic compaction"),
    );

    assert_eq!(
        response.usage.as_ref().map(|usage| usage.total_tokens),
        Some(expected_usage)
    );
    let requests = fixture.request_bodies().await;
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0]["input"], summary.turns[0].request.body.input);
    assert!(
        requests
            .iter()
            .all(|request| request.get("context_management").is_none())
    );
}

#[tokio::test]
async fn tool_search_standalone_compaction_lowers_summary_history_and_preserves_loaded_state() {
    let fixture = TestFixture::new_with_responses(vec![
        function_call_response("fc_search", "call_search", "tool_search", r#"{"query":"weather"}"#),
        text_response("tool loaded"),
        text_response("durable tool summary"),
        function_call_response("fc_weather", "call_weather", "get_weather", r#"{"city":"Paris"}"#),
    ])
    .await;

    let first = unwrap_blocking(
        execute(
            tool_search_request("find weather", Some(tool_search_declarations()), true, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("stored search call"),
    );
    let second = unwrap_blocking(
        execute(
            tool_search_request(
                json!([{
                    "type": "tool_search_output",
                    "call_id": "call_search",
                    "tools": [tool_search_declarations()[1].clone()]
                }]),
                None,
                true,
                Some(first.id),
                None,
            ),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("persist loaded state"),
    );
    let compact_request: CompactRequest = serde_json::from_value(json!({
        "model": "test-model",
        "previous_response_id": second.id
    }))
    .expect("valid stored compaction request");
    let compacted = compact_response(compact_request, &fixture.exec_ctx, None)
        .await
        .expect("standalone tool-search compaction");

    assert!(
        compacted
            .output
            .iter()
            .all(|item| !matches!(item, InputItem::ToolSearchCall(_) | InputItem::ToolSearchOutput(_)))
    );
    let followup = unwrap_blocking(
        execute(
            tool_search_request("use the loaded weather tool", None, false, Some(compacted.id), None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("compacted continuation"),
    );
    assert_eq!(
        serde_json::to_value(&followup.output[0]).unwrap()["name"],
        "get_weather"
    );

    let requests = fixture.request_bodies().await;
    assert_eq!(requests.len(), 4);
    let summary_request = &requests[2];
    assert!(
        summary_request["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| { item["type"] == "function_call" && item["name"] == "tool_search" })
    );
    assert!(!summary_request.to_string().contains("tool_search_call"));
    assert!(!summary_request.to_string().contains("tool_search_output"));
    let followup_tools = requests[3]["tools"].as_array().expect("compacted effective tools");
    assert!(followup_tools.iter().any(|tool| tool["name"] == "get_weather"));
    assert!(!followup_tools.iter().any(|tool| tool["name"] == "get_time"));
}

#[tokio::test]
async fn tool_search_automatic_compaction_restores_only_loaded_definition_on_continuation() {
    let fixture = TestFixture::new_with_responses(vec![
        text_response("durable automatic tool summary"),
        function_call_response("fc_weather", "call_weather", "get_weather", r#"{"city":"Paris"}"#),
        text_response("automatic compaction complete"),
    ])
    .await;
    let conversation_id = create_conversation(&fixture.exec_ctx)
        .await
        .expect("create compacted conversation")
        .conversation_id;
    let mut request = tool_search_request(
        completed_tool_search_history(),
        Some(tool_search_declarations()),
        true,
        None,
        Some(conversation_id.clone()),
    );
    request.context_management = serde_json::from_value(json!([{
        "type": "compaction",
        "compact_threshold": 1
    }]))
    .expect("valid automatic compaction policy");

    let first = unwrap_blocking(
        execute(request, Arc::clone(&fixture.exec_ctx))
            .await
            .expect("automatic tool-search compaction"),
    );
    assert_eq!(serde_json::to_value(&first.output[0]).unwrap()["name"], "get_weather");
    let final_response = unwrap_blocking(
        execute(
            tool_search_request(
                json!([{
                    "type": "function_call_output",
                    "call_id": "call_weather",
                    "output": "sunny"
                }]),
                None,
                true,
                None,
                Some(conversation_id),
            ),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("continue automatically compacted state"),
    );
    assert_eq!(output_text(&final_response), "automatic compaction complete");

    let requests = fixture.request_bodies().await;
    assert_eq!(requests.len(), 3);
    assert!(!requests[0].to_string().contains("tool_search_call"));
    assert!(!requests[0].to_string().contains("tool_search_output"));
    assert!(
        requests[0]["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| { item["type"] == "function_call" && item["name"] == "tool_search" })
    );
    assert!(requests[1]["input"].as_array().unwrap().iter().all(|item| {
        !(item["type"] == "function_call" && item["name"] == "tool_search"
            || item["type"] == "function_call_output" && item["call_id"] == "call_search")
    }));
    for request in &requests[1..] {
        let tools = request["tools"].as_array().expect("effective private tools");
        assert!(tools.iter().any(|tool| tool["name"] == "get_weather"));
        assert!(!tools.iter().any(|tool| tool["name"] == "get_time"));
    }
}
