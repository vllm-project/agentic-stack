//! Cassette-based integration tests for the Responses API (cases 1–5).
//!
//! Mirrors `test_responses_api.py`. Each test replays a YAML cassette
//! against a mock HTTP server and verifies `execute()` output.

mod support;

use agentic_core::executor::execute;
use agentic_core::executor::request::RequestContext;
use agentic_core::storage::InOutItem;
use agentic_core::types::request_response::RequestPayload;
use agentic_core::types::tools::{FunctionToolParam, NonEmptyToolName};
use agentic_core::{
    FunctionToolResultMessage, InputItem, OutputItem, ReasoningOutput, ResponsesInput, ResponsesTool, ToolChoice,
};
use either::Either;
use futures::StreamExt;
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::sync::Arc;
use support::{
    MockResponse, TestFixture, collect_stream, expected_text, function_call_response, load_cassette, make_request,
    output_text, request_input_texts, text_response, tool_search_function_declarations, tool_search_output,
    tool_search_request, unwrap_blocking,
};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/text_only/responses");

/// Case 1 — single turn, non-streaming.
#[tokio::test]
async fn test_single_turn_nonstreaming() {
    // Arrange
    let cassette = load_cassette(&format!("{DIR}/resp-single-gpt-4o-nonstreaming.yaml"));
    let t1 = &cassette.turns[0];
    let fixture = TestFixture::new(&[t1]).await;

    // Act
    let payload = unwrap_blocking(
        execute(
            make_request(&t1.request.body.input, t1.request.body.store, false, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("execute"),
    );

    // Assert
    assert!(payload.id.starts_with("resp_"), "id={}", payload.id);
    assert_eq!(payload.status, "completed");
    assert_eq!(output_text(&payload), expected_text(t1));
}

/// Case 2 — single turn, streaming.
#[tokio::test]
async fn test_single_turn_streaming() {
    // Arrange
    let cassette = load_cassette(&format!("{DIR}/resp-single-gpt-4o-streaming.yaml"));
    let t1 = &cassette.turns[0];
    let fixture = TestFixture::new(&[t1]).await;

    // Act
    let payload = collect_stream(
        execute(
            make_request(&t1.request.body.input, t1.request.body.store, true, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("execute"),
    )
    .await;

    // Assert
    assert!(payload.id.starts_with("resp_"), "id={}", payload.id);
    assert_eq!(payload.status, "completed");
    assert_eq!(output_text(&payload), expected_text(t1));
}

#[tokio::test]
async fn test_single_turn_streaming_emits_response_completed_event() {
    let cassette = load_cassette(&format!("{DIR}/resp-single-gpt-4o-streaming.yaml"));
    let t1 = &cassette.turns[0];
    let fixture = TestFixture::new(&[t1]).await;

    let result = execute(
        make_request(&t1.request.body.input, t1.request.body.store, true, None, None),
        Arc::clone(&fixture.exec_ctx),
    )
    .await
    .expect("execute");
    let Either::Right(stream) = result else {
        panic!("expected streaming response");
    };
    let chunks = stream.collect::<Vec<_>>().await;
    let events = support::streamed_sse_events(&chunks);

    assert!(
        !events
            .iter()
            .any(|event| event.get("object").and_then(Value::as_str) == Some("response")),
        "executor stream should not emit a bare ResponsePayload"
    );
    let event_types = events
        .iter()
        .filter_map(|event| event["type"].as_str())
        .collect::<Vec<_>>();
    assert!(event_types.contains(&"response.created"));
    assert!(event_types.contains(&"response.output_text.delta"));
    let completed = events.last().expect("stream should include events");
    assert_eq!(completed["type"], "response.completed");
    assert_eq!(completed["response"]["status"], "completed");
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["text"],
        expected_text(t1)
    );
}

#[tokio::test]
async fn test_stream_persists_when_client_disconnects_after_completion_event() {
    let cassette = load_cassette(&format!("{DIR}/resp-single-gpt-4o-streaming.yaml"));
    let t1 = &cassette.turns[0];
    let responses = vec![t1, t1];
    let fixture = TestFixture::new(&responses).await;

    let result = execute(
        make_request(&t1.request.body.input, true, true, None, None),
        Arc::clone(&fixture.exec_ctx),
    )
    .await
    .expect("execute");
    let Either::Right(stream) = result else {
        panic!("expected streaming response");
    };
    let mut stream = Box::pin(stream);
    let response_id = loop {
        let Some(chunk) = stream.next().await else {
            panic!("stream ended before response.completed");
        };
        let Some(event) = support::streamed_sse_event(&chunk) else {
            continue;
        };
        if event["type"] == "response.completed" {
            break event["response"]["id"].as_str().expect("response id").to_owned();
        }
    };

    // Dropping before requesting the next chunk simulates a client disconnect
    // immediately after receiving the terminal response event.
    drop(stream);

    let follow_up = execute(
        make_request("follow up", true, true, Some(response_id), None),
        Arc::clone(&fixture.exec_ctx),
    )
    .await
    .expect("response must be persisted before response.completed is emitted");
    assert!(matches!(follow_up, Either::Right(_)));
}

/// Case 3 — two turns, non-streaming, chained via `previous_response_id`.
#[tokio::test]
async fn test_two_turn_nonstreaming_previous_response_id() {
    // Arrange
    let cassette = load_cassette(&format!("{DIR}/resp-two-turn-gpt-4o-nonstreaming.yaml"));
    let (t1, t2) = (&cassette.turns[0], &cassette.turns[1]);
    let fixture = TestFixture::new(&[t1, t2]).await;

    // Act
    let p1 = unwrap_blocking(
        execute(
            make_request(&t1.request.body.input, true, false, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("t1"),
    );
    let p2 = unwrap_blocking(
        execute(
            make_request(&t2.request.body.input, true, false, Some(p1.id.clone()), None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("t2"),
    );

    // Assert
    assert!(p1.id.starts_with("resp_"));
    assert_eq!(p1.status, "completed");
    assert_eq!(output_text(&p1), expected_text(t1));
    assert_ne!(p2.id, p1.id);
    assert_eq!(p2.status, "completed");
    assert_eq!(p2.previous_response_id.as_deref(), Some(p1.id.as_str()));
    assert_eq!(output_text(&p2), expected_text(t2));
}

/// Case 4 — two turns, streaming, chained via `previous_response_id`.
#[tokio::test]
async fn test_two_turn_streaming_previous_response_id() {
    // Arrange
    let cassette = load_cassette(&format!("{DIR}/resp-two-turn-gpt-4o-streaming.yaml"));
    let (t1, t2) = (&cassette.turns[0], &cassette.turns[1]);
    let fixture = TestFixture::new(&[t1, t2]).await;

    // Act
    let p1 = collect_stream(
        execute(
            make_request(&t1.request.body.input, true, true, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("t1"),
    )
    .await;
    let p2 = collect_stream(
        execute(
            make_request(&t2.request.body.input, true, true, Some(p1.id.clone()), None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("t2"),
    )
    .await;

    // Assert
    assert!(p1.id.starts_with("resp_"));
    assert_eq!(p1.status, "completed");
    assert_eq!(output_text(&p1), expected_text(t1));
    assert_ne!(p2.id, p1.id);
    assert_eq!(p2.status, "completed");
    assert_eq!(output_text(&p2), expected_text(t2));
}

/// Case 5 — `store=false` response cannot be used as `previous_response_id`.
#[tokio::test]
async fn test_store_disabled_not_reusable_as_previous_response_id() {
    // Arrange — only one mock needed; follow-up errors before hitting the LLM
    let cassette = load_cassette(&format!("{DIR}/resp-no-store-gpt-4o-nonstreaming.yaml"));
    let t1 = &cassette.turns[0];
    let fixture = TestFixture::new(&[t1]).await;

    // Act — turn 1, store=false
    let p1 = unwrap_blocking(
        execute(
            make_request(&t1.request.body.input, false, false, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("t1"),
    );
    assert_eq!(p1.status, "completed");

    // Act — follow-up with the unstored id
    let result = execute(
        make_request("follow up", false, false, Some(p1.id.clone()), None),
        Arc::clone(&fixture.exec_ctx),
    )
    .await;

    // Assert — executor errors at rehydrate, before calling the LLM
    assert!(result.is_err(), "expected error for unstored previous_response_id");
}

#[tokio::test]
async fn test_previous_response_id_rehydrates_full_checkpoint_history() {
    let fixture = TestFixture::new_with_responses(vec![
        text_response("first answer"),
        text_response("second answer"),
        text_response("third answer"),
    ])
    .await;

    let p1 = unwrap_blocking(
        execute(
            make_request("turn 1", true, false, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("t1"),
    );
    let p2 = unwrap_blocking(
        execute(
            make_request("turn 2", true, false, Some(p1.id.clone()), None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("t2"),
    );
    let p3 = unwrap_blocking(
        execute(
            make_request("turn 3", true, false, Some(p2.id), None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("t3"),
    );

    assert_eq!(output_text(&p3), "third answer");
    let requests = fixture.request_bodies().await;
    assert_eq!(requests.len(), 3);
    assert_eq!(
        request_input_texts(&requests[2]),
        vec!["turn 1", "first answer", "turn 2", "second answer", "turn 3"]
    );
}

#[tokio::test]
async fn test_codex_namespace_tool_shape_rehydrates_from_previous_response_metadata() {
    let fixture = TestFixture::new_with_responses(vec![
        text_response("seed answer"),
        text_response("next answer"),
        text_response("third answer"),
    ])
    .await;
    let tool_json = serde_json::json!([
        {
            "type": "namespace",
            "name": "mcp__shell",
            "tools": [{"type": "function", "name": "run", "parameters": {"type": "object"}}]
        }
    ]);
    let tools: Vec<ResponsesTool> = serde_json::from_value(tool_json.clone()).unwrap();

    let mut first = make_request("seed", true, false, None, None);
    first.tools = Some(tools);
    let p1 = unwrap_blocking(execute(first, Arc::clone(&fixture.exec_ctx)).await.expect("first turn"));

    let second = make_request("next", true, false, Some(p1.id), None);
    let p2 = unwrap_blocking(
        execute(second, Arc::clone(&fixture.exec_ctx))
            .await
            .expect("second turn"),
    );
    let third = make_request("third", true, false, Some(p2.id), None);
    let _p3 = unwrap_blocking(execute(third, Arc::clone(&fixture.exec_ctx)).await.expect("third turn"));

    let requests = fixture.request_bodies().await;
    for request in &requests {
        let tools = request["tools"].as_array().expect("typed upstream tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "agentic_ns__mcp__shell__run");
        assert_eq!(tools[0]["parameters"], tool_json[0]["tools"][0]["parameters"]);
    }
}

#[tokio::test]
async fn test_codex_namespace_collision_with_top_level_function_is_rejected() {
    let fixture = TestFixture::new_with_responses(vec![text_response("should not be called")]).await;
    let tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
        {"type": "function", "name": "agentic_ns__mcp__shell__run"},
        {
            "type": "namespace",
            "name": "mcp__shell",
            "tools": [{"type": "function", "name": "run"}]
        }
    ]))
    .unwrap();

    let mut request = make_request("run pwd", true, false, None, None);
    request.tools = Some(tools);

    let Err(err) = execute(request, Arc::clone(&fixture.exec_ctx)).await else {
        panic!("colliding namespace member should be rejected");
    };

    assert!(
        err.to_string().contains("collides with a declared function tool"),
        "unexpected error: {err}"
    );
    assert!(
        fixture.request_bodies().await.is_empty(),
        "invalid request must fail before calling upstream"
    );
}

#[tokio::test]
async fn test_previous_response_id_explicit_tool_choice_overrides_stored_choice() {
    let fixture =
        TestFixture::new_with_responses(vec![text_response("seed answer"), text_response("next answer")]).await;

    let mut first = make_request("seed", true, false, None, None);
    first.tool_choice = Some(ToolChoice::Required);
    let p1 = unwrap_blocking(execute(first, Arc::clone(&fixture.exec_ctx)).await.expect("first turn"));

    let mut second = make_request("next", true, false, Some(p1.id), None);
    second.tool_choice = Some(ToolChoice::None);
    let _p2 = unwrap_blocking(
        execute(second, Arc::clone(&fixture.exec_ctx))
            .await
            .expect("second turn"),
    );

    let requests = fixture.request_bodies().await;
    assert_eq!(requests[0]["tool_choice"], "required");
    assert_eq!(requests[1]["tool_choice"], "none");
}

#[tokio::test]
async fn test_previous_response_id_rehydrates_function_call_before_tool_output() {
    let tool_call_response = MockResponse::Json(
        serde_json::json!({
            "id": "resp_tool",
            "object": "response",
            "created_at": 0,
            "model": "test-model",
            "status": "completed",
            "output": [{
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "run",
                "namespace": "mcp__shell",
                "arguments": "{\"cmd\":\"pwd\"}",
                "status": "completed"
            }],
            "usage": null,
            "incomplete_details": null,
            "error": null,
            "previous_response_id": null,
            "conversation_id": null,
            "instructions": null
        })
        .to_string(),
    );
    let fixture = TestFixture::new_with_responses(vec![tool_call_response, text_response("tool result handled")]).await;

    let first = make_request("run pwd", true, false, None, None);
    let p1 = unwrap_blocking(execute(first, Arc::clone(&fixture.exec_ctx)).await.expect("first turn"));

    let mut second = make_request("ignored", true, false, Some(p1.id), None);
    second.input = ResponsesInput::Items(vec![InputItem::FunctionCallOutput(FunctionToolResultMessage {
        call_id: "call_1".to_string(),
        output: "{\"stdout\":\"/workspace\"}".into(),
    })]);
    let _p2 = unwrap_blocking(
        execute(second, Arc::clone(&fixture.exec_ctx))
            .await
            .expect("second turn"),
    );

    let requests = fixture.request_bodies().await;
    let input = requests[1]["input"].as_array().expect("input array");
    assert_eq!(input[1]["type"], "function_call");
    assert_eq!(input[1]["namespace"], "mcp__shell");
    assert_eq!(input[1]["name"], "run");
    assert_eq!(input[2]["type"], "function_call_output");
    assert_eq!(input[2]["call_id"], "call_1");
}

#[tokio::test]
async fn test_previous_response_id_replays_plaintext_reasoning_without_opaque_state() {
    Box::pin(assert_plaintext_reasoning_replay(false, false, true)).await;
}

#[tokio::test]
async fn test_streaming_previous_response_id_replays_plaintext_reasoning_without_opaque_state() {
    Box::pin(assert_plaintext_reasoning_replay(true, false, true)).await;
}

#[tokio::test]
async fn test_conversation_replays_plaintext_reasoning_without_opaque_state() {
    Box::pin(assert_plaintext_reasoning_replay(false, true, true)).await;
}

#[tokio::test]
async fn test_streaming_conversation_replays_plaintext_reasoning_with_null_state() {
    Box::pin(assert_plaintext_reasoning_replay(true, true, false)).await;
}

#[tokio::test]
async fn test_summary_only_reasoning_is_not_replayed() {
    for stream in [false, true] {
        for conversation in [false, true] {
            Box::pin(assert_summary_only_reasoning_not_replayed(stream, conversation)).await;
        }
    }
}

#[tokio::test]
async fn test_encrypted_only_persisted_reasoning_fails_before_upstream() {
    let fixture = TestFixture::new_with_responses(vec![reasoning_response(false, &[], true)]).await;
    let first = unwrap_blocking(
        execute(
            make_request("historical user", true, false, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("persist encrypted-only reasoning"),
    );

    for stream in [false, true] {
        let mut continuation = make_request("continue", true, stream, Some(first.id.clone()), None);
        continuation.input = ResponsesInput::Items(vec![InputItem::FunctionCallOutput(FunctionToolResultMessage {
            call_id: "call_prior".to_owned(),
            output: "tool output".into(),
        })]);
        let result = execute(continuation, Arc::clone(&fixture.exec_ctx)).await;
        let Err(error) = result else {
            panic!("encrypted-only reasoning must be rejected before inference");
        };
        assert_eq!(error.http_status(), http::StatusCode::BAD_REQUEST);
        assert!(
            error
                .to_string()
                .contains("encrypted state without plaintext reasoning content")
        );
        assert!(!error.to_string().contains("opaque-provider-state"));
    }

    assert_eq!(
        fixture.request_bodies().await.len(),
        1,
        "invalid continuations must not call the upstream"
    );
}

#[tokio::test]
async fn test_mcp_namespace_showcase_round_trip_rehydrates_calls_tools_and_outputs() {
    let tool_json = mcp_showcase_tools_json();
    let tools: Vec<ResponsesTool> = serde_json::from_value(tool_json.clone()).expect("tool fixture parses");
    let tool_call_response = MockResponse::Json(
        serde_json::json!({
            "id": "resp_mcp_showcase",
            "object": "response",
            "created_at": 0,
            "model": "test-model",
            "status": "completed",
            "output": [
                upstream_mcp_fixture_call("fc_echo", "call_echo", "echo_text", r#"{"text":"namespace showcase","uppercase":true}"#),
                upstream_mcp_fixture_call("fc_sum", "call_sum", "add_numbers", r#"{"numbers":[2,3,5]}"#),
                upstream_mcp_fixture_call("fc_slug", "call_slug", "make_slug", r#"{"text":"Codex MCP Showcase"}"#),
                upstream_mcp_fixture_call("fc_head", "call_head", "repo_file_head", r#"{"path":"README.md","lines":2}"#),
                upstream_mcp_fixture_call("fc_search", "call_search", "search_repo", r#"{"query":"codex","path_prefix":"scripts","max_results":3}"#)
            ],
            "usage": null,
            "incomplete_details": null,
            "error": null,
            "previous_response_id": null,
            "conversation_id": null,
            "instructions": null
        })
        .to_string(),
    );
    let fixture = TestFixture::new_with_responses(vec![tool_call_response, text_response("showcase complete")]).await;

    let mut first = make_request("use the agentic_fixture MCP toolbox", true, false, None, None);
    first.tools = Some(tools);
    let p1 = unwrap_blocking(execute(first, Arc::clone(&fixture.exec_ctx)).await.expect("first turn"));

    let output = serde_json::to_value(&p1.output).expect("output serializes");
    assert_namespaced_calls(
        output.as_array().expect("output array"),
        &["echo_text", "add_numbers", "make_slug", "repo_file_head", "search_repo"],
    );

    let mut second = make_request("ignored", true, false, Some(p1.id), None);
    second.input = ResponsesInput::Items(vec![
        tool_output(
            "call_echo",
            r#"{"echo":"NAMESPACE SHOWCASE","characters":18,"words":2}"#,
        ),
        tool_output("call_sum", r#"{"count":3,"sum":10}"#),
        tool_output("call_slug", r#"{"slug":"codex-mcp-showcase"}"#),
        tool_output(
            "call_head",
            "README.md first 2 lines:\n1: # agentic-api\n2: Stateful API logic",
        ),
        tool_output(
            "call_search",
            r#"{"query":"codex","matches":[{"path":"scripts/codex-run.sh","line":16}]}"#,
        ),
    ]);
    let p2 = unwrap_blocking(
        execute(second, Arc::clone(&fixture.exec_ctx))
            .await
            .expect("second turn"),
    );

    assert_eq!(output_text(&p2), "showcase complete");
    let requests = fixture.request_bodies().await;
    assert_eq!(requests.len(), 2);
    assert_flat_mcp_showcase_tools(&requests[0]["tools"]);
    assert_flat_mcp_showcase_tools(&requests[1]["tools"]);

    let input = requests[1]["input"].as_array().expect("rehydrated input array");
    assert_flat_namespaced_calls(
        input,
        &["echo_text", "add_numbers", "make_slug", "repo_file_head", "search_repo"],
    );
    assert_tool_outputs(
        input,
        &["call_echo", "call_sum", "call_slug", "call_head", "call_search"],
    );
    assert!(
        !contains_key(&requests[1], "_agentic_item_kind"),
        "storage marker must not leak into rehydrated upstream request"
    );
}

#[tokio::test]
async fn test_store_false_with_previous_response_id_hydrates_but_does_not_persist() {
    let fixture =
        TestFixture::new_with_responses(vec![text_response("stored answer"), text_response("stateless answer")]).await;

    let p1 = unwrap_blocking(
        execute(
            make_request("seed", true, false, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("stored turn"),
    );
    let p2 = unwrap_blocking(
        execute(
            make_request("follow up", false, false, Some(p1.id), None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("store=false follow-up"),
    );

    assert_eq!(output_text(&p2), "stateless answer");
    let requests = fixture.request_bodies().await;
    assert_eq!(requests.len(), 2);
    assert_eq!(
        request_input_texts(&requests[1]),
        vec!["seed", "stored answer", "follow up"]
    );

    let result = execute(
        make_request("should not find stateless response", true, false, Some(p2.id), None),
        Arc::clone(&fixture.exec_ctx),
    )
    .await;
    assert!(result.is_err(), "store=false response should not be persisted");
}

#[tokio::test]
async fn tool_search_previous_response_continuation_omits_tools_after_first_turn() {
    let fixture = TestFixture::new_with_responses(vec![
        function_call_response("fc_search", "call_search", "tool_search", r#"{"query":"weather"}"#),
        function_call_response("fc_weather", "call_weather", "get_weather", r#"{"city":"Paris"}"#),
        text_response("weather complete"),
    ])
    .await;
    let declarations = tool_search_function_declarations("get_weather", "Get weather");

    let first = unwrap_blocking(
        execute(
            tool_search_request("find weather", Some(declarations), true, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("stored search turn"),
    );
    let search_call = serde_json::to_value(&first.output[0]).expect("public search call");
    assert_eq!(search_call["type"], "tool_search_call");

    let second = unwrap_blocking(
        execute(
            tool_search_request(
                json!([tool_search_output("call_search", "get_weather", "Get weather")]),
                None,
                true,
                Some(first.id.clone()),
                None,
            ),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("search-output continuation"),
    );
    let weather_call = serde_json::to_value(&second.output[0]).expect("public weather call");
    assert_eq!(weather_call["name"], "get_weather");

    let third = unwrap_blocking(
        execute(
            tool_search_request(
                json!([{
                    "type": "function_call_output",
                    "call_id": "call_weather",
                    "output": "sunny"
                }]),
                None,
                true,
                Some(second.id.clone()),
                None,
            ),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("loaded-function continuation"),
    );
    assert_eq!(output_text(&third), "weather complete");

    let requests = fixture.request_bodies().await;
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[1]["input"][1]["type"], "function_call");
    assert_eq!(requests[1]["input"][1]["name"], "tool_search");
    assert!(
        requests[1]["tools"]
            .as_array()
            .expect("inherited tools")
            .iter()
            .any(|tool| { tool["name"] == "get_weather" && tool.get("defer_loading").is_none() })
    );
    assert!(
        requests[1]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "tool_search")
    );
    assert!(
        requests[2]["tools"]
            .as_array()
            .expect("replayed tools")
            .iter()
            .any(|tool| { tool["name"] == "get_weather" && tool.get("defer_loading").is_none() })
    );
    assert!(
        !requests
            .iter()
            .any(|body| body.to_string().contains("tool_search_call"))
    );
    assert!(
        !requests
            .iter()
            .any(|body| body.to_string().contains("tool_search_output"))
    );
}

#[tokio::test]
async fn tool_search_branch_from_earlier_response_does_not_inherit_later_loaded_definition() {
    let fixture = TestFixture::new_with_responses(vec![
        function_call_response("fc_search", "call_search", "tool_search", r#"{"query":"weather"}"#),
        text_response("loaded branch"),
        text_response("empty branch"),
    ])
    .await;
    let declarations = json!([
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
        }
    ]);
    let first = unwrap_blocking(
        execute(
            tool_search_request("find weather", Some(declarations), true, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("search turn"),
    );

    let loaded_branch = unwrap_blocking(
        execute(
            tool_search_request(
                json!([{
                    "type": "tool_search_output",
                    "call_id": "call_search",
                    "tools": [{
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                        "defer_loading": true
                    }]
                }]),
                None,
                true,
                Some(first.id.clone()),
                None,
            ),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("loaded branch"),
    );
    assert_eq!(output_text(&loaded_branch), "loaded branch");

    let empty_branch = unwrap_blocking(
        execute(
            tool_search_request(
                json!([{
                    "type": "tool_search_output",
                    "call_id": "call_search",
                    "tools": []
                }]),
                None,
                true,
                Some(first.id),
                None,
            ),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("independent empty branch"),
    );
    assert_eq!(output_text(&empty_branch), "empty branch");

    let requests = fixture.request_bodies().await;
    let loaded_tools = requests[1]["tools"].as_array().expect("loaded branch tools");
    let empty_tools = requests[2]["tools"].as_array().expect("empty branch tools");
    assert!(loaded_tools.iter().any(|tool| tool["name"] == "get_weather"));
    assert!(!empty_tools.iter().any(|tool| tool["name"] == "get_weather"));
}

#[tokio::test]
async fn tool_search_store_false_manual_replay_completes_without_reusable_response() {
    let fixture = TestFixture::new_with_responses(vec![
        function_call_response("fc_search", "call_search", "tool_search", r#"{"query":"weather"}"#),
        function_call_response("fc_weather", "call_weather", "get_weather", r#"{"city":"Paris"}"#),
        text_response("manual replay complete"),
    ])
    .await;
    let declarations = json!([
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
        }
    ]);
    let user = json!({"type": "message", "role": "user", "content": "find weather"});
    let first = unwrap_blocking(
        execute(
            tool_search_request(json!([user.clone()]), Some(declarations), false, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("stateless search call"),
    );
    let search_call = serde_json::to_value(&first.output[0]).unwrap();
    let search_output = json!({
        "type": "tool_search_output",
        "call_id": "call_search",
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object"},
            "defer_loading": true
        }]
    });
    let second_input = json!([user.clone(), search_call, search_output]);
    let second = unwrap_blocking(
        execute(
            tool_search_request(second_input.clone(), None, false, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("stateless loaded call"),
    );
    let mut final_input = second_input.as_array().unwrap().clone();
    final_input.push(serde_json::to_value(&second.output[0]).unwrap());
    final_input.push(json!({
        "type": "function_call_output",
        "call_id": "call_weather",
        "output": "sunny"
    }));
    let final_response = unwrap_blocking(
        execute(
            tool_search_request(Value::Array(final_input), None, false, None, None),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("stateless final response"),
    );
    assert_eq!(output_text(&final_response), "manual replay complete");

    let lookup_ctx = RequestContext {
        original_request: make_request("lookup", true, false, Some(final_response.id.clone()), None),
        enriched_request: make_request("lookup", true, false, Some(final_response.id), None),
        new_input_items: Vec::new(),
        response_id: "resp_lookup".to_owned(),
        conversation_id: None,
        conversation_version: None,
        continuation: None,
    };
    let error = fixture
        .exec_ctx
        .resp_handler
        .get(&lookup_ctx)
        .await
        .expect_err("store:false response ID must not be reusable");
    assert!(matches!(error, agentic_core::executor::ExecutorError::Storage(source) if source.is_not_found()));
}

#[tokio::test]
async fn test_previous_response_id_persists_inherited_tools_and_choice() {
    let fixture =
        TestFixture::new_with_responses(vec![text_response("seed answer"), text_response("follow up answer")]).await;

    let tool = ResponsesTool::Function(FunctionToolParam {
        name: NonEmptyToolName::try_from("lookup_weather").expect("valid tool name"),
        description: Some("Look up weather".to_string()),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            }
        })),
        strict: Some(true),
        defer_loading: None,
        extra: std::collections::HashMap::new(),
    });

    let mut first_request = make_request("seed", true, false, None, None);
    first_request.tools = Some(vec![tool]);
    first_request.tool_choice = Some(ToolChoice::Required);

    let p1 = unwrap_blocking(
        execute(first_request, Arc::clone(&fixture.exec_ctx))
            .await
            .expect("seed turn"),
    );

    let mut second_request = make_request("follow up", true, false, Some(p1.id.clone()), None);
    second_request.tools = None;
    second_request.tool_choice = None;

    let p2 = unwrap_blocking(
        execute(second_request.clone(), Arc::clone(&fixture.exec_ctx))
            .await
            .expect("follow-up turn"),
    );

    assert_eq!(output_text(&p2), "follow up answer");

    let lookup_ctx = RequestContext {
        original_request: RequestPayload {
            previous_response_id: Some(p2.id.clone()),
            ..second_request
        },
        enriched_request: RequestPayload {
            previous_response_id: Some(p2.id.clone()),
            ..make_request("lookup", true, false, None, None)
        },
        new_input_items: vec![],
        response_id: "resp_lookup".into(),
        conversation_id: None,
        conversation_version: None,
        continuation: None,
    };

    let stored = fixture
        .exec_ctx
        .resp_handler
        .get(&lookup_ctx)
        .await
        .expect("fetch persisted response");

    assert_eq!(stored.metadata.model, "test-model");
    assert!(matches!(stored.metadata.effective_tool_choice, ToolChoice::Required));

    let tools = stored.metadata.effective_tools.expect("expected persisted tools");
    assert_eq!(tools.len(), 1);
    match &tools[0] {
        ResponsesTool::Function(p) => {
            assert_eq!(p.name.as_str(), "lookup_weather");
            assert_eq!(p.description.as_deref(), Some("Look up weather"));
            assert_eq!(p.strict, Some(true));
            assert_eq!(p.parameters.as_ref().and_then(|v| v["type"].as_str()), Some("object"));
        }
        _ => panic!("expected function tool"),
    }
}

#[tokio::test]
async fn test_conversation_id_and_previous_response_id_are_rejected_together() {
    let fixture = TestFixture::new_with_responses(vec![]).await;

    let result = execute(
        make_request(
            "ambiguous",
            true,
            false,
            Some("resp_ambiguous".to_string()),
            Some("conv_ambiguous".to_string()),
        ),
        Arc::clone(&fixture.exec_ctx),
    )
    .await;

    assert!(result.is_err(), "expected ambiguous state IDs to be rejected");
    assert!(fixture.request_bodies().await.is_empty());
}

fn mcp_showcase_tools_json() -> Value {
    serde_json::json!([
        {
            "type": "namespace",
            "name": "mcp__agentic_fixture",
            "description": "Fixture namespace tool for Codex MCP round-trip tests.",
            "tools": [
                {
                    "type": "function",
                    "name": "run",
                    "description": "Echo a command string for namespace round-trip validation.",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                        "additionalProperties": false
                    },
                    "strict": true
                },
                {
                    "type": "function",
                    "name": "echo_text",
                    "description": "Echo text with basic metadata.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "uppercase": {"type": "boolean"}
                        },
                        "required": ["text"],
                        "additionalProperties": false
                    },
                    "strict": true
                },
                {
                    "type": "function",
                    "name": "add_numbers",
                    "description": "Add a list of numbers and return the total.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "numbers": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 1
                            }
                        },
                        "required": ["numbers"],
                        "additionalProperties": false
                    },
                    "strict": true
                },
                {
                    "type": "function",
                    "name": "make_slug",
                    "description": "Turn text into a lowercase URL/file-name friendly slug.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "separator": {"type": "string"}
                        },
                        "required": ["text"],
                        "additionalProperties": false
                    },
                    "strict": true
                },
                {
                    "type": "function",
                    "name": "repo_file_head",
                    "description": "Read the first lines of a repository file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "lines": {"type": "integer", "minimum": 1, "maximum": 80}
                        },
                        "required": ["path"],
                        "additionalProperties": false
                    },
                    "strict": true
                },
                {
                    "type": "function",
                    "name": "search_repo",
                    "description": "Literal text search across repository files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "path_prefix": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 30}
                        },
                        "required": ["query"],
                        "additionalProperties": false
                    },
                    "strict": true
                }
            ]
        }
    ])
}

fn upstream_mcp_fixture_call(id: &str, call_id: &str, name: &str, arguments: &str) -> Value {
    serde_json::json!({
        "id": id,
        "type": "function_call",
        "call_id": call_id,
        "name": format!("agentic_ns__mcp__agentic_fixture__{name}"),
        "arguments": arguments,
        "status": "completed"
    })
}

fn tool_output(call_id: &str, output: &str) -> InputItem {
    InputItem::FunctionCallOutput(FunctionToolResultMessage {
        call_id: call_id.to_string(),
        output: output.into(),
    })
}

fn assert_namespaced_calls(items: &[Value], expected_names: &[&str]) {
    for expected_name in expected_names {
        assert!(
            items.iter().any(|item| {
                item.get("type").and_then(Value::as_str) == Some("function_call")
                    && item.get("namespace").and_then(Value::as_str) == Some("mcp__agentic_fixture")
                    && item.get("name").and_then(Value::as_str) == Some(expected_name)
            }),
            "missing namespaced function call mcp__agentic_fixture.{expected_name}"
        );
    }
}

fn assert_flat_namespaced_calls(items: &[Value], expected_names: &[&str]) {
    for expected_name in expected_names {
        let flat_name = format!("agentic_ns__mcp__agentic_fixture__{expected_name}");
        assert!(
            items.iter().any(|item| {
                item.get("type").and_then(Value::as_str) == Some("function_call")
                    && item.get("name").and_then(Value::as_str) == Some(&flat_name)
                    && item.get("namespace").is_none()
            }),
            "missing private flat function call {flat_name}"
        );
    }
}

fn assert_tool_outputs(items: &[Value], expected_call_ids: &[&str]) {
    for expected_call_id in expected_call_ids {
        assert!(
            items.iter().any(|item| {
                item.get("type").and_then(Value::as_str) == Some("function_call_output")
                    && item.get("call_id").and_then(Value::as_str) == Some(expected_call_id)
            }),
            "missing function_call_output for {expected_call_id}"
        );
    }
}

fn assert_flat_mcp_showcase_tools(tools: &Value) {
    let tools = tools.as_array().expect("tools array");
    assert_eq!(tools.len(), 6);
    for name in [
        "run",
        "echo_text",
        "add_numbers",
        "make_slug",
        "repo_file_head",
        "search_repo",
    ] {
        let flat_name = format!("agentic_ns__mcp__agentic_fixture__{name}");
        assert!(
            tools.iter().any(|tool| {
                tool.get("type").and_then(Value::as_str) == Some("function")
                    && tool.get("name").and_then(Value::as_str) == Some(flat_name.as_str())
            }),
            "missing flat upstream tool {flat_name}"
        );
    }
}

fn contains_key(value: &Value, key: &str) -> bool {
    match value {
        Value::Object(object) => object.contains_key(key) || object.values().any(|nested| contains_key(nested, key)),
        Value::Array(values) => values.iter().any(|nested| contains_key(nested, key)),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => false,
    }
}

async fn assert_plaintext_reasoning_replay(stream: bool, conversation: bool, opaque_state: bool) {
    let follow_up = if stream {
        let cassette = load_cassette(&format!("{DIR}/resp-single-gpt-4o-streaming.yaml"));
        MockResponse::from_turn(&cassette.turns[0])
    } else {
        text_response("follow-up answer")
    };
    let fixture = TestFixture::new_with_responses(vec![
        reasoning_response(stream, &["", "plaintext continuation"], opaque_state),
        follow_up,
    ])
    .await;
    let conversation_id = conversation.then(|| "conv_reasoning_replay".to_owned());
    let first = run_response(
        make_request("historical user", true, stream, None, conversation_id.clone()),
        Arc::clone(&fixture.exec_ctx),
    )
    .await;
    let previous_response_id = (!conversation).then(|| first.id.clone());

    let mut second = make_request(
        "ignored",
        true,
        stream,
        previous_response_id.clone(),
        conversation_id.clone(),
    );
    second.input = serde_json::from_value(serde_json::json!([
        {
            "type": "function_call_output",
            "call_id": "call_prior",
            "output": "tool output"
        },
        {"role": "user", "content": "new user input"}
    ]))
    .expect("valid follow-up input");
    let result = execute(second, Arc::clone(&fixture.exec_ctx))
        .await
        .expect("execute continuation");
    if stream {
        collect_stream(result).await;
    } else {
        unwrap_blocking(result);
    }

    let requests = fixture.request_bodies().await;
    assert_eq!(requests.len(), 2);
    let input = requests[1]["input"].as_array().expect("rehydrated input array");
    let item_types = input
        .iter()
        .map(|item| item["type"].as_str().expect("typed input item"))
        .collect::<Vec<_>>();
    assert_eq!(
        item_types,
        [
            "message",
            "reasoning",
            "message",
            "function_call",
            "function_call_output",
            "message"
        ]
    );

    let reasoning = &input[1];
    assert_eq!(reasoning["id"], "rs_prior");
    assert_eq!(reasoning["content"].as_array().map(Vec::len), Some(1));
    assert_eq!(reasoning["content"][0]["text"], "\nplaintext continuation");
    assert_eq!(reasoning["summary"], serde_json::json!([]));
    assert_eq!(reasoning["status"], "completed");
    assert!(
        reasoning["encrypted_content"].is_null(),
        "opaque provider state must not be forwarded to vLLM"
    );
    assert!(!contains_key(&requests[1], "_agentic_item_kind"));

    let lookup = lookup_context(previous_response_id, conversation_id);
    let history = if conversation {
        fixture
            .exec_ctx
            .conv_handler
            .rehydrate(&lookup)
            .await
            .expect("rehydrate conversation")
    } else {
        fixture
            .exec_ctx
            .resp_handler
            .rehydrate(&lookup)
            .await
            .expect("rehydrate response")
    };
    let stored = persisted_reasoning(&history);
    assert_eq!(stored.content.len(), 2);
    assert_eq!(stored.content[0].text, "");
    assert_eq!(stored.content[1].text, "plaintext continuation");
    assert_eq!(stored.summary[0]["text"], "public summary");
    let expected_state = opaque_state.then(|| serde_json::json!("opaque-provider-state"));
    assert_eq!(stored.encrypted_content, expected_state);
    assert_eq!(stored.status.as_deref(), Some("completed"));
}

async fn assert_summary_only_reasoning_not_replayed(stream: bool, conversation: bool) {
    let follow_up = if stream {
        let cassette = load_cassette(&format!("{DIR}/resp-single-gpt-4o-streaming.yaml"));
        MockResponse::from_turn(&cassette.turns[0])
    } else {
        text_response("follow-up answer")
    };
    let fixture = TestFixture::new_with_responses(vec![reasoning_response(stream, &[], false), follow_up]).await;
    let conversation_id = conversation.then(|| format!("conv_summary_only_{stream}"));
    let first = run_response(
        make_request("historical user", true, stream, None, conversation_id.clone()),
        Arc::clone(&fixture.exec_ctx),
    )
    .await;
    let previous_response_id = (!conversation).then(|| first.id.clone());

    let mut second = make_request(
        "ignored",
        true,
        stream,
        previous_response_id.clone(),
        conversation_id.clone(),
    );
    second.input = serde_json::from_value(serde_json::json!([
        {
            "type": "function_call_output",
            "call_id": "call_prior",
            "output": "tool output"
        },
        {"role": "user", "content": "new user input"}
    ]))
    .expect("valid follow-up input");
    run_response(second, Arc::clone(&fixture.exec_ctx)).await;

    let requests = fixture.request_bodies().await;
    assert_eq!(requests.len(), 2);
    let input = requests[1]["input"].as_array().expect("rehydrated input array");
    let item_types = input
        .iter()
        .map(|item| item["type"].as_str().expect("typed input item"))
        .collect::<Vec<_>>();
    assert_eq!(
        item_types,
        ["message", "message", "function_call", "function_call_output", "message"]
    );
    assert!(
        input.iter().all(|item| item["type"] != "reasoning"),
        "summary-only reasoning must not reach vLLM"
    );
    assert!(
        !requests[1].to_string().contains("public summary"),
        "a reasoning summary must never be promoted into the vLLM-bound copy"
    );

    let lookup = lookup_context(previous_response_id, conversation_id);
    let history = if conversation {
        fixture
            .exec_ctx
            .conv_handler
            .rehydrate(&lookup)
            .await
            .expect("rehydrate conversation")
    } else {
        fixture
            .exec_ctx
            .resp_handler
            .rehydrate(&lookup)
            .await
            .expect("rehydrate response")
    };
    let stored = persisted_reasoning(&history);
    assert!(stored.content.is_empty());
    assert_eq!(stored.summary[0]["text"], "public summary");
    assert_eq!(stored.encrypted_content, None);
    assert_eq!(stored.status.as_deref(), Some("completed"));
}

async fn run_response(
    request: RequestPayload,
    exec_ctx: Arc<agentic_core::executor::ExecutionContext>,
) -> agentic_core::ResponsePayload {
    let stream = request.stream;
    let result = execute(request, exec_ctx).await.expect("execute response");
    if stream {
        collect_stream(result).await
    } else {
        unwrap_blocking(result)
    }
}

fn lookup_context(previous_response_id: Option<String>, conversation_id: Option<String>) -> RequestContext {
    let request = make_request("lookup", true, false, previous_response_id, conversation_id);
    RequestContext {
        enriched_request: request.clone(),
        original_request: request,
        new_input_items: Vec::new(),
        response_id: "resp_lookup".to_owned(),
        conversation_id: None,
        conversation_version: None,
        continuation: None,
    }
}

fn persisted_reasoning(history: &[InOutItem]) -> &ReasoningOutput {
    history
        .iter()
        .find_map(|item| match item {
            InOutItem::Output(OutputItem::Reasoning(reasoning)) => Some(reasoning),
            InOutItem::Input(_) | InOutItem::Output(_) => None,
        })
        .expect("persisted reasoning item")
}

fn reasoning_response(stream: bool, plaintext: &[&str], opaque_state: bool) -> MockResponse {
    let content = plaintext
        .iter()
        .map(|text| serde_json::json!({"type": "reasoning_text", "text": text}))
        .collect::<Vec<_>>();
    let reasoning = serde_json::json!({
        "type": "reasoning",
        "id": "rs_prior",
        "content": content,
        "summary": [{"type": "summary_text", "text": "public summary"}],
        "encrypted_content": opaque_state.then_some("opaque-provider-state"),
        "status": "completed"
    });
    let message = serde_json::json!({
        "type": "message",
        "id": "msg_prior",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "assistant context", "annotations": []}]
    });
    let function_call = serde_json::json!({
        "type": "function_call",
        "id": "fc_prior",
        "call_id": "call_prior",
        "name": "client_tool",
        "arguments": "{}",
        "status": "completed"
    });
    let response = serde_json::json!({
        "id": "resp_reasoning",
        "object": "response",
        "created_at": 0,
        "model": "test-model",
        "status": "completed",
        "output": [reasoning.clone(), message.clone(), function_call.clone()],
        "usage": null,
        "incomplete_details": null,
        "error": null,
        "previous_response_id": null,
        "conversation_id": null,
        "instructions": null
    });
    if !stream {
        return MockResponse::Json(response.to_string());
    }

    let events = vec![
        serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": reasoning
        }),
        serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {"type": "message", "id": "msg_prior", "role": "assistant", "status": "in_progress", "content": []}
        }),
        serde_json::json!({
            "type": "response.output_text.delta",
            "item_id": "msg_prior",
            "output_index": 1,
            "content_index": 0,
            "delta": "assistant context"
        }),
        serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 1,
            "item": message
        }),
        serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 2,
            "item": function_call
        }),
        serde_json::json!({"type": "response.completed", "response": response}),
    ];
    let mut body = String::new();
    for event in events {
        let event_type = event["type"].as_str().expect("event type");
        writeln!(body, "event: {event_type}\ndata: {event}\n").expect("write SSE fixture");
    }
    body.push_str("data: [DONE]\n\n");
    MockResponse::Sse(body)
}
