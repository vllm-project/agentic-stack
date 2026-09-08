//! Cassette-based integration tests for the Conversation API (cases 6–10).
//!
//! Mirrors `test_conversation_api.py`. Each conversation cassette includes a
//! `/v1/conversations` creation turn — mirrored here via `create_conversation()`.
//! `TestFixture` serves only `/v1/responses` turns on the mock HTTP server.

mod support;

use agentic_core::executor::{create_conversation, execute};
use serde_json::json;
use std::sync::Arc;
use support::{
    TestFixture, collect_stream, expected_text, function_call_response, load_cassette, make_request, output_text,
    request_input_texts, responses_turns, text_response, tool_search_function_declarations, tool_search_output,
    tool_search_request, unwrap_blocking,
};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/text_only/conversation");

/// Case 6 — two turns, non-streaming, via `conversation_id`.
#[tokio::test]
async fn test_two_turn_nonstreaming_conversation() {
    // Arrange
    let cassette = load_cassette(&format!("{DIR}/conv-two-turn-gpt-4o-nonstreaming.yaml"));
    let all: Vec<_> = cassette.turns.iter().collect();
    let fixture = TestFixture::new(&all).await;
    let ctx = &fixture.exec_ctx;
    let resp = responses_turns(&cassette);
    let (t1, t2) = (resp[0], resp[1]);

    // Mirrors /v1/conversations creation turn
    let conv_id = create_conversation(ctx).await.expect("create conv").conversation_id;

    // Act
    let p1 = unwrap_blocking(
        execute(
            make_request(&t1.request.body.input, true, false, None, Some(conv_id.clone())),
            Arc::clone(ctx),
        )
        .await
        .expect("t1"),
    );
    let p2 = unwrap_blocking(
        execute(
            make_request(&t2.request.body.input, true, false, None, Some(conv_id)),
            Arc::clone(ctx),
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
    assert_eq!(output_text(&p2), expected_text(t2));
}

#[tokio::test]
async fn tool_search_conversation_continuation_rebuilds_loaded_tools_from_history() {
    let fixture = TestFixture::new_with_responses(vec![
        function_call_response("fc_search", "call_search", "tool_search", r#"{"query":"weather"}"#),
        function_call_response("fc_weather", "call_weather", "get_weather", r#"{"city":"Paris"}"#),
        text_response("conversation weather complete"),
    ])
    .await;
    let conversation_id = create_conversation(&fixture.exec_ctx)
        .await
        .expect("create conversation")
        .conversation_id;
    let declarations = tool_search_function_declarations("get_weather", "Get weather");

    let first = unwrap_blocking(
        execute(
            tool_search_request(
                "find weather",
                Some(declarations),
                true,
                None,
                Some(conversation_id.clone()),
            ),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("conversation search turn"),
    );
    assert_eq!(
        serde_json::to_value(&first.output[0]).unwrap()["type"],
        "tool_search_call"
    );

    let second = unwrap_blocking(
        execute(
            tool_search_request(
                json!([tool_search_output("call_search", "get_weather", "Get weather")]),
                None,
                true,
                None,
                Some(conversation_id.clone()),
            ),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("conversation search output"),
    );
    assert_eq!(serde_json::to_value(&second.output[0]).unwrap()["name"], "get_weather");

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
                None,
                Some(conversation_id),
            ),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("conversation loaded-function output"),
    );
    assert_eq!(output_text(&third), "conversation weather complete");

    let requests = fixture.request_bodies().await;
    assert_eq!(requests.len(), 3);
    for request in &requests[1..] {
        assert!(
            request["tools"]
                .as_array()
                .expect("inherited tools")
                .iter()
                .any(|tool| { tool["name"] == "get_weather" && tool.get("defer_loading").is_none() })
        );
        assert!(
            request["tools"]
                .as_array()
                .unwrap()
                .iter()
                .all(|tool| tool["name"] != "tool_search")
        );
        assert!(!request.to_string().contains("tool_search_call"));
        assert!(!request.to_string().contains("tool_search_output"));
    }
}

#[tokio::test]
async fn tool_search_previous_response_branch_cannot_replace_conversation_history() {
    let fixture = TestFixture::new_with_responses(vec![
        function_call_response("fc_winner", "call_winner", "tool_search", r#"{"query":"winner"}"#),
        text_response("winner loaded"),
        text_response("winner checkpoint"),
        function_call_response("fc_branch", "call_branch", "tool_search", r#"{"query":"branch"}"#),
        text_response("branch loaded"),
        text_response("conversation resumed"),
    ])
    .await;
    let conversation_id = create_conversation(&fixture.exec_ctx)
        .await
        .expect("create conversation")
        .conversation_id;

    let winner_search = unwrap_blocking(
        execute(
            tool_search_request(
                "find winner",
                Some(tool_search_function_declarations("winner_tool", "Winner tool")),
                true,
                None,
                Some(conversation_id.clone()),
            ),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("conversation search call"),
    );
    assert_eq!(
        serde_json::to_value(&winner_search.output[0]).unwrap()["type"],
        "tool_search_call"
    );
    let winner = unwrap_blocking(
        execute(
            tool_search_request(
                json!([tool_search_output("call_winner", "winner_tool", "Winner tool")]),
                None,
                true,
                None,
                Some(conversation_id.clone()),
            ),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("conversation loaded state"),
    );
    assert_eq!(output_text(&winner), "winner loaded");

    let checkpoint = unwrap_blocking(
        execute(
            tool_search_request("checkpoint winner", None, true, None, Some(conversation_id.clone())),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("conversation checkpoint"),
    );
    assert_eq!(output_text(&checkpoint), "winner checkpoint");

    let branch_search = unwrap_blocking(
        execute(
            tool_search_request(
                "find branch",
                Some(tool_search_function_declarations("branch_tool", "Branch tool")),
                true,
                Some(checkpoint.id),
                None,
            ),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("response branch search call"),
    );
    let branch = unwrap_blocking(
        execute(
            tool_search_request(
                json!([tool_search_output("call_branch", "branch_tool", "Branch tool")]),
                None,
                true,
                Some(branch_search.id),
                None,
            ),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("response branch loaded state"),
    );
    assert_eq!(output_text(&branch), "branch loaded");

    let resumed = unwrap_blocking(
        execute(
            tool_search_request("resume conversation", None, true, None, Some(conversation_id)),
            Arc::clone(&fixture.exec_ctx),
        )
        .await
        .expect("conversation continuation"),
    );
    assert_eq!(output_text(&resumed), "conversation resumed");

    let requests = fixture.request_bodies().await;
    let resumed_tools = requests[5]["tools"].as_array().expect("conversation tools");
    assert!(resumed_tools.iter().any(|tool| tool["name"] == "winner_tool"));
    assert!(!resumed_tools.iter().any(|tool| tool["name"] == "branch_tool"));
    assert!(!requests[5].to_string().contains("branch loaded"));
}

/// Case 7 — two turns, streaming, via `conversation_id`.
#[tokio::test]
async fn test_two_turn_streaming_conversation() {
    // Arrange
    let cassette = load_cassette(&format!("{DIR}/conv-two-turn-gpt-4o-streaming.yaml"));
    let all: Vec<_> = cassette.turns.iter().collect();
    let fixture = TestFixture::new(&all).await;
    let ctx = &fixture.exec_ctx;
    let resp = responses_turns(&cassette);
    let (t1, t2) = (resp[0], resp[1]);

    let conv_id = create_conversation(ctx).await.expect("create conv").conversation_id;

    // Act
    let p1 = collect_stream(
        execute(
            make_request(&t1.request.body.input, true, true, None, Some(conv_id.clone())),
            Arc::clone(ctx),
        )
        .await
        .expect("t1"),
    )
    .await;
    let p2 = collect_stream(
        execute(
            make_request(&t2.request.body.input, true, true, None, Some(conv_id)),
            Arc::clone(ctx),
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

/// Case 8 — two independent conversations must not share context.
#[tokio::test]
async fn test_conversation_isolation() {
    // Arrange
    let cassette = load_cassette(&format!("{DIR}/conv-isolation-gpt-4o-nonstreaming.yaml"));
    let all: Vec<_> = cassette.turns.iter().collect();
    let fixture = TestFixture::new(&all).await;
    let ctx = &fixture.exec_ctx;
    let resp = responses_turns(&cassette);
    let (ta1, ta2, ta3, tb1, tb2, tb3) = (resp[0], resp[1], resp[2], resp[3], resp[4], resp[5]);

    // Conv A
    let conv_a = create_conversation(ctx).await.expect("create conv A").conversation_id;
    let pa1 = unwrap_blocking(
        execute(
            make_request(&ta1.request.body.input, true, false, None, Some(conv_a.clone())),
            Arc::clone(ctx),
        )
        .await
        .expect("a1"),
    );
    assert_eq!(output_text(&pa1), expected_text(ta1));
    let pa2 = unwrap_blocking(
        execute(
            make_request(&ta2.request.body.input, true, false, None, Some(conv_a.clone())),
            Arc::clone(ctx),
        )
        .await
        .expect("a2"),
    );
    assert_eq!(output_text(&pa2), expected_text(ta2));
    let pa3 = unwrap_blocking(
        execute(
            make_request(&ta3.request.body.input, true, false, None, Some(conv_a.clone())),
            Arc::clone(ctx),
        )
        .await
        .expect("a3"),
    );
    assert_eq!(output_text(&pa3), expected_text(ta3));

    // Conv B
    let conv_b = create_conversation(ctx).await.expect("create conv B").conversation_id;
    let pb1 = unwrap_blocking(
        execute(
            make_request(&tb1.request.body.input, true, false, None, Some(conv_b.clone())),
            Arc::clone(ctx),
        )
        .await
        .expect("b1"),
    );
    assert_eq!(output_text(&pb1), expected_text(tb1));
    let pb2 = unwrap_blocking(
        execute(
            make_request(&tb2.request.body.input, true, false, None, Some(conv_b.clone())),
            Arc::clone(ctx),
        )
        .await
        .expect("b2"),
    );
    assert_eq!(output_text(&pb2), expected_text(tb2));
    let pb3 = unwrap_blocking(
        execute(
            make_request(&tb3.request.body.input, true, false, None, Some(conv_b.clone())),
            Arc::clone(ctx),
        )
        .await
        .expect("b3"),
    );
    assert_eq!(output_text(&pb3), expected_text(tb3));

    // Assert — conversations are isolated
    assert_ne!(conv_a, conv_b, "conversations must not share an id");
}

/// Case 9 — 3-turn chain then branch off turn 1 via `previous_response_id`.
#[tokio::test]
async fn test_branch_off_turn_1() {
    // Arrange
    let cassette = load_cassette(&format!("{DIR}/conv-multi-turn-single-branch-gpt-4o-nonstreaming.yaml"));
    let all: Vec<_> = cassette.turns.iter().collect();
    let fixture = TestFixture::new(&all).await;
    let ctx = &fixture.exec_ctx;
    let resp = responses_turns(&cassette);
    let (t1, t2, t3, t4) = (resp[0], resp[1], resp[2], resp[3]);

    let conv_id = create_conversation(ctx).await.expect("create conv").conversation_id;

    // Main chain
    let p1 = unwrap_blocking(
        execute(
            make_request(&t1.request.body.input, true, false, None, Some(conv_id.clone())),
            Arc::clone(ctx),
        )
        .await
        .expect("t1"),
    );
    assert_eq!(output_text(&p1), expected_text(t1));
    let r1_id = p1.id.clone();

    let p2 = unwrap_blocking(
        execute(
            make_request(&t2.request.body.input, true, false, None, Some(conv_id.clone())),
            Arc::clone(ctx),
        )
        .await
        .expect("t2"),
    );
    assert_eq!(output_text(&p2), expected_text(t2));

    let p3 = unwrap_blocking(
        execute(
            make_request(&t3.request.body.input, true, false, None, Some(conv_id)),
            Arc::clone(ctx),
        )
        .await
        .expect("t3"),
    );
    assert_eq!(output_text(&p3), expected_text(t3));

    // Branch off turn 1 — only turn 1 context visible
    let p4 = unwrap_blocking(
        execute(
            make_request(&t4.request.body.input, true, false, Some(r1_id), None),
            Arc::clone(ctx),
        )
        .await
        .expect("t4"),
    );
    assert_eq!(p4.status, "completed");
    assert_eq!(output_text(&p4), expected_text(t4));
}

/// Case 10 — 5-turn chain with 2 inline branches.
#[tokio::test]
async fn test_multi_branch() {
    // Arrange
    let cassette = load_cassette(&format!("{DIR}/conv-multi-branch-multi-turn-gpt-4o-nonstreaming.yaml"));
    let all: Vec<_> = cassette.turns.iter().collect();
    let fixture = TestFixture::new(&all).await;
    let ctx = &fixture.exec_ctx;
    let resp = responses_turns(&cassette);
    let (t1, t2, t3, t4, t5) = (resp[0], resp[1], resp[2], resp[3], resp[4]);

    let conv_id = create_conversation(ctx).await.expect("create conv").conversation_id;

    // Turn 1
    let p1 = unwrap_blocking(
        execute(
            make_request(&t1.request.body.input, true, false, None, Some(conv_id.clone())),
            Arc::clone(ctx),
        )
        .await
        .expect("t1"),
    );
    assert_eq!(output_text(&p1), expected_text(t1));
    let r1_id = p1.id.clone();

    // Turn 2 (main branch)
    let p2 = unwrap_blocking(
        execute(
            make_request(&t2.request.body.input, true, false, None, Some(conv_id.clone())),
            Arc::clone(ctx),
        )
        .await
        .expect("t2"),
    );
    assert_eq!(output_text(&p2), expected_text(t2));
    let r2_id = p2.id.clone();

    // Branch 1 — off turn 1
    let p3 = unwrap_blocking(
        execute(
            make_request(&t3.request.body.input, true, false, Some(r1_id), None),
            Arc::clone(ctx),
        )
        .await
        .expect("t3"),
    );
    assert_eq!(p3.status, "completed");
    assert_eq!(output_text(&p3), expected_text(t3));

    let p4 = unwrap_blocking(
        execute(
            make_request(&t4.request.body.input, true, false, Some(p3.id.clone()), None),
            Arc::clone(ctx),
        )
        .await
        .expect("t4"),
    );
    assert_eq!(p4.status, "completed");
    assert_eq!(output_text(&p4), expected_text(t4));
    assert_eq!(p4.conversation_id.as_deref(), Some(conv_id.as_str()));

    // Branch 2 — off turn 2
    let p5 = unwrap_blocking(
        execute(
            make_request(&t5.request.body.input, true, false, Some(r2_id), None),
            Arc::clone(ctx),
        )
        .await
        .expect("t5"),
    );
    assert_eq!(p5.status, "completed");
    assert_eq!(output_text(&p5), expected_text(t5));
}

#[tokio::test]
async fn test_store_false_with_conversation_id_hydrates_and_persists() {
    let cassette = load_cassette(&format!("{DIR}/conv-store-false-followup-gpt-4o-nonstreaming.yaml"));
    let all: Vec<_> = cassette.turns.iter().collect();
    let fixture = TestFixture::new(&all).await;
    let ctx = &fixture.exec_ctx;
    let resp = responses_turns(&cassette);
    let (t1, t2) = (resp[0], resp[1]);

    let conv_id = create_conversation(ctx).await.expect("create conv").conversation_id;

    // Turn 1: store=true
    let p1 = unwrap_blocking(
        execute(
            make_request(
                &t1.request.body.input,
                t1.request.body.store,
                false,
                None,
                Some(conv_id.clone()),
            ),
            Arc::clone(ctx),
        )
        .await
        .expect("t1"),
    );
    assert_eq!(p1.status, "completed");
    assert_eq!(output_text(&p1), expected_text(t1));

    // Turn 2: store=false but conversation_id passed — must rehydrate and persist locally
    let p2 = unwrap_blocking(
        execute(
            make_request(
                &t2.request.body.input,
                t2.request.body.store,
                false,
                None,
                Some(conv_id.clone()),
            ),
            Arc::clone(ctx),
        )
        .await
        .expect("t2"),
    );
    assert_eq!(p2.status, "completed");
    assert_eq!(output_text(&p2), expected_text(t2));

    // Turn 2 was sent the full history from turn 1 (rehydrated correctly)
    let requests = fixture.request_bodies().await;
    assert_eq!(requests.len(), 2);
    assert_eq!(
        request_input_texts(&requests[1]),
        vec![
            t1.request.body.input.as_str().expect("text cassette input"),
            expected_text(t1).as_str(),
            t2.request.body.input.as_str().expect("text cassette input")
        ]
    );
}
