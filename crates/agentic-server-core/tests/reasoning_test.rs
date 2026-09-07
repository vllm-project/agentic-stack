use agentic_core::executor::ExecuteRequest;
use agentic_core::types::io::OutputItem;
use agentic_core::types::request_response::{RequestPayload, ResponsePayload};
use serde_json::{Value, json};

mod support;

const CASSETTE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/reasoning/responses");
const GATEWAY_MODEL: &str = "gpt-5.6";
const GATEWAY_MODEL_SLUG: &str = "gpt-5.6";
const OPENAI_MODEL: &str = "gpt-5.6";
const OPENAI_MODEL_SLUG: &str = "gpt-5.6";
const PROMPT: &str = "Determine whether 47 is the unique two-digit positive integer whose digits sum to 11 and whose reversal is 27 larger. Analyze the constraints, then reply with exactly one word: VALID or INVALID.";
const EXPECTED_STREAMING_LIFECYCLE: &[&str] = &[
    "response.created",
    "response.in_progress",
    "response.output_item.added:reasoning",
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary_text.done",
    "response.reasoning_summary_part.done",
    "response.output_item.done:reasoning",
    "response.output_item.added:message",
    "response.content_part.added",
    "response.output_text.delta",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done:message",
    "response.completed",
];

fn expected_reasoning() -> Value {
    json!({"effort": "high", "summary": "detailed"})
}

fn load_recorded_pair(streaming: bool) -> (support::Cassette, support::Cassette) {
    let mode = if streaming { "streaming" } else { "nonstreaming" };
    let openai = support::load_cassette(&format!(
        "{CASSETTE_DIR}/reasoning-openai-reference-{OPENAI_MODEL_SLUG}-{mode}.yaml"
    ));
    let gateway = support::load_cassette(&format!(
        "{CASSETTE_DIR}/reasoning-gateway-{GATEWAY_MODEL_SLUG}-{mode}.yaml"
    ));
    (openai, gateway)
}

fn terminal_response(turn: &support::Turn) -> Value {
    if let Some(body) = &turn.response.body {
        return body.clone();
    }

    support::recorded_named_sse_events(turn)
        .into_iter()
        .rev()
        .find_map(|event| {
            (event["type"] == "response.completed")
                .then(|| event.get("response").cloned())
                .flatten()
        })
        .expect("streaming cassette should contain response.completed")
}

fn assert_request_contract(openai: &support::Cassette, gateway: &support::Cassette, streaming: bool) {
    assert_eq!(openai.turns.len(), 1);
    assert_eq!(gateway.turns.len(), 1);
    let openai = &openai.turns[0].request;
    let gateway = &gateway.turns[0].request;

    assert_eq!(openai.path, "/v1/responses");
    assert_eq!(gateway.path, openai.path);
    assert_eq!(
        gateway.body, openai.body,
        "OpenAI and gateway recordings must contain the same complete request body"
    );
    assert_eq!(openai.body.model.as_deref(), Some(OPENAI_MODEL));
    assert_eq!(gateway.body.model.as_deref(), Some(GATEWAY_MODEL));
    assert_eq!(openai.body.input, PROMPT);
    assert_eq!(gateway.body.input, openai.body.input);
    assert!(openai.body.store);
    assert_eq!(gateway.body.store, openai.body.store);
    assert_eq!(openai.body.stream, streaming);
    assert_eq!(gateway.body.stream, openai.body.stream);
    assert_eq!(openai.body.max_output_tokens, Some(2048));
    assert_eq!(gateway.body.max_output_tokens, openai.body.max_output_tokens);
    assert_eq!(openai.body.reasoning, Some(expected_reasoning()));
    assert_eq!(gateway.body.reasoning, openai.body.reasoning);
}

fn lifecycle_event_name(event: &Value) -> String {
    let event_type = event["type"].as_str().expect("stream event should contain a type");
    if matches!(event_type, "response.output_item.added" | "response.output_item.done") {
        let item_type = event["item"]["type"]
            .as_str()
            .expect("output-item lifecycle event should contain an item type");
        format!("{event_type}:{item_type}")
    } else {
        event_type.to_owned()
    }
}

fn normalized_streaming_lifecycle(events: &[Value]) -> Vec<String> {
    let mut lifecycle = Vec::new();
    for event in events {
        let event_name = lifecycle_event_name(event);
        let is_text_delta = matches!(
            event_name.as_str(),
            "response.reasoning_summary_text.delta" | "response.output_text.delta"
        );
        if is_text_delta && lifecycle.last() == Some(&event_name) {
            continue;
        }
        lifecycle.push(event_name);
    }
    lifecycle
}

fn assert_item_lifecycle(events: &[Value], item_type: &str, expected_output_index: u64) {
    let added = events
        .iter()
        .find(|event| event["type"] == "response.output_item.added" && event["item"]["type"] == item_type)
        .unwrap_or_else(|| panic!("stream should add a {item_type} output item"));
    let item_id = added["item"]["id"]
        .as_str()
        .unwrap_or_else(|| panic!("{item_type} output item should contain an id"));
    let mut lifecycle_event_count = 0;
    let mut saw_done = false;

    for event in events {
        let event_item_id = event["item_id"].as_str().or_else(|| event["item"]["id"].as_str());
        if event_item_id != Some(item_id) {
            continue;
        }
        lifecycle_event_count += 1;
        assert_eq!(
            event["output_index"].as_u64(),
            Some(expected_output_index),
            "every {item_type} lifecycle event should retain one output index"
        );
        if event["type"] == "response.output_item.done" {
            assert_eq!(event["item"]["type"], item_type);
            saw_done = true;
        }
    }

    assert!(
        lifecycle_event_count >= 2,
        "{item_type} should have a multi-event lifecycle"
    );
    assert!(saw_done, "{item_type} lifecycle should end with output_item.done");
}

fn assert_streamed_text_reconciles(events: &[Value], delta_type: &str, done_type: &str) -> String {
    let delta_text = events
        .iter()
        .filter(|event| event["type"] == delta_type)
        .map(|event| event["delta"].as_str().expect("text delta should contain delta text"))
        .collect::<String>();
    assert!(!delta_text.is_empty(), "{delta_type} events should contain text");

    let done_events = events
        .iter()
        .filter(|event| event["type"] == done_type)
        .collect::<Vec<_>>();
    assert_eq!(
        done_events.len(),
        1,
        "stream should contain exactly one {done_type} event"
    );
    assert_eq!(
        done_events[0]["text"].as_str(),
        Some(delta_text.as_str()),
        "the authoritative {done_type} text should equal the accumulated deltas"
    );
    delta_text
}

fn assert_streaming_contract(turn: &support::Turn) -> Vec<String> {
    let events = support::recorded_named_sse_events(turn);
    let sequence_numbers = events
        .iter()
        .map(|event| {
            event["sequence_number"]
                .as_u64()
                .expect("every recorded stream event should contain a sequence number")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sequence_numbers,
        (0..u64::try_from(events.len()).expect("stream length should fit in u64")).collect::<Vec<_>>(),
        "stream sequence numbers should be unique and contiguous"
    );

    let lifecycle = normalized_streaming_lifecycle(&events);
    assert_eq!(lifecycle, EXPECTED_STREAMING_LIFECYCLE);
    assert_item_lifecycle(&events, "reasoning", 0);
    assert_item_lifecycle(&events, "message", 1);

    let reasoning_text = assert_streamed_text_reconciles(
        &events,
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
    );
    let output_text =
        assert_streamed_text_reconciles(&events, "response.output_text.delta", "response.output_text.done");
    let terminal = terminal_response(turn);
    let terminal_reasoning_text = terminal["output"]
        .as_array()
        .expect("completed response should contain output")
        .iter()
        .find(|item| item["type"] == "reasoning")
        .and_then(|item| item["summary"].as_array())
        .into_iter()
        .flatten()
        .filter_map(|part| part["text"].as_str())
        .collect::<String>();
    let terminal_output_text = terminal["output"]
        .as_array()
        .expect("completed response should contain output")
        .iter()
        .find(|item| item["type"] == "message")
        .and_then(|item| item["content"].as_array())
        .into_iter()
        .flatten()
        .filter_map(|part| part["text"].as_str())
        .collect::<String>();
    assert_eq!(terminal_reasoning_text, reasoning_text);
    assert_eq!(terminal_output_text, output_text);

    lifecycle
}

fn assert_terminal_contract(turn: &support::Turn) {
    let response = terminal_response(turn);
    assert_eq!(response["status"], "completed");
    let output = response["output"]
        .as_array()
        .expect("completed response should contain output");
    let reasoning = output
        .iter()
        .find(|item| item["type"] == "reasoning")
        .expect("explicit reasoning request should produce a reasoning item");
    let has_reasoning_text = ["content", "summary"].into_iter().any(|field| {
        reasoning[field].as_array().is_some_and(|parts| {
            parts
                .iter()
                .any(|part| part["text"].as_str().is_some_and(|text| !text.is_empty()))
        })
    });
    let has_encrypted_reasoning = reasoning["encrypted_content"]
        .as_str()
        .is_some_and(|content| !content.is_empty());
    assert!(
        has_reasoning_text || has_encrypted_reasoning,
        "reasoning item should contain recorded reasoning content"
    );
    let message = output
        .iter()
        .find(|item| item["type"] == "message")
        .expect("completed response should contain a message");
    assert_eq!(message["status"], "completed");
    let text = message["content"]
        .as_array()
        .expect("message should contain content")
        .iter()
        .filter_map(|part| part["text"].as_str())
        .collect::<String>();
    assert_eq!(text.trim(), "VALID");
}

async fn replay_through_gateway(turn: &support::Turn) -> ResponsePayload {
    let fixture = support::TestFixture::new(&[turn]).await;
    let payload: RequestPayload = serde_json::from_value(json!({
        "model": turn.request.body.model,
        "input": turn.request.body.input,
        "store": turn.request.body.store,
        "stream": turn.request.body.stream,
        "max_output_tokens": turn.request.body.max_output_tokens,
        "reasoning": turn.request.body.reasoning,
    }))
    .expect("recorded request should satisfy the gateway request schema");
    let streaming = payload.stream;

    let result = ExecuteRequest::new(payload, fixture.exec_ctx.clone())
        .run()
        .await
        .expect("recorded response should replay through the gateway");
    let response = if streaming {
        support::collect_stream(result).await
    } else {
        support::unwrap_blocking(result)
    };

    let requests = fixture.request_bodies().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["reasoning"], expected_reasoning());
    assert_eq!(requests[0]["stream"], streaming);
    response
}

fn assert_replayed_output(response: &ResponsePayload) {
    assert_eq!(response.status, "completed");
    assert!(
        response
            .output
            .iter()
            .any(|item| matches!(item, OutputItem::Reasoning(_))),
        "gateway replay should retain the reasoning item"
    );
    assert_eq!(support::output_text(response).trim(), "VALID");
}

#[tokio::test]
async fn recorded_nonstreaming_reasoning_matches_openai_contract() {
    let (openai, gateway) = load_recorded_pair(false);
    assert_request_contract(&openai, &gateway, false);
    assert_terminal_contract(&openai.turns[0]);
    assert_terminal_contract(&gateway.turns[0]);

    assert_replayed_output(&replay_through_gateway(&openai.turns[0]).await);
    assert_replayed_output(&replay_through_gateway(&gateway.turns[0]).await);
}

#[tokio::test]
async fn recorded_streaming_reasoning_matches_openai_contract() {
    let (openai, gateway) = load_recorded_pair(true);
    assert_request_contract(&openai, &gateway, true);
    assert_terminal_contract(&openai.turns[0]);
    assert_terminal_contract(&gateway.turns[0]);
    let openai_lifecycle = assert_streaming_contract(&openai.turns[0]);
    let gateway_lifecycle = assert_streaming_contract(&gateway.turns[0]);
    assert_eq!(gateway_lifecycle, openai_lifecycle);

    assert_replayed_output(&replay_through_gateway(&openai.turns[0]).await);
    assert_replayed_output(&replay_through_gateway(&gateway.turns[0]).await);
}
