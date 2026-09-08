use std::fs;
use std::path::{Path, PathBuf};

use agentic_core::executor::request::RequestContext;
use agentic_core::executor::{UpstreamBody, decode_upstream};
use agentic_core::types::io::OutputItem;
use agentic_core::types::request_response::RequestPayload;
use serde_json::json;

fn request_context() -> RequestContext {
    let request: RequestPayload = serde_json::from_value(json!({
        "model": "test-model",
        "input": "hi",
        "store": true,
        "stream": true
    }))
    .expect("valid request");
    RequestContext {
        original_request: request.clone(),
        enriched_request: request,
        new_input_items: Vec::new(),
        response_id: "resp_reserved".to_owned(),
        conversation_id: None,
        conversation_version: None,
    }
}

fn yaml_files(root: &Path) -> Vec<PathBuf> {
    let mut pending = vec![root.to_owned()];
    let mut files = Vec::new();
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display())) {
            let path = entry.expect("cassette directory entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "yaml") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn response_streams(path: &Path) -> Vec<Vec<String>> {
    let text = fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let document: serde_json::Value =
        serde_yaml::from_str(&text).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()));

    if let Some(sse) = document["sse"].as_array() {
        return vec![
            sse.iter()
                .map(|chunk| chunk.as_str().expect("SSE chunk string").to_owned())
                .collect(),
        ];
    }

    document["turns"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|turn| turn["request"]["path"] == "/v1/responses")
        .filter_map(|turn| turn["response"]["sse"].as_array())
        .map(|sse| {
            sse.iter()
                .map(|chunk| chunk.as_str().expect("SSE chunk string").to_owned())
                .collect()
        })
        .collect()
}

fn is_responses_event_stream(stream: &str) -> bool {
    stream.lines().any(|line| {
        line.strip_prefix("data: ")
            .and_then(|data| serde_json::from_str::<serde_json::Value>(data).ok())
            .is_some_and(|event| event["type"] == "response.created")
    })
}

fn message_stream(terminal_ids: [&str; 2]) -> String {
    [
        json!({
            "type": "response.created",
            "response": {"id": "resp_upstream", "status": "in_progress"}
        }),
        json!({
            "type": "response.in_progress",
            "response": {"id": "resp_upstream", "status": "in_progress"}
        }),
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "msg_streamed_0", "type": "message", "role": "assistant", "status": "in_progress"}
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"id": "msg_streamed_0", "type": "message", "role": "assistant", "status": "completed"}
        }),
        json!({
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {"id": "msg_streamed_1", "type": "message", "role": "assistant", "status": "in_progress"}
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {"id": "msg_streamed_1", "type": "message", "role": "assistant", "status": "completed"}
        }),
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp_upstream",
                "status": "completed",
                "output": [
                    {"id": terminal_ids[0], "type": "message", "role": "assistant", "status": "completed"},
                    {"id": terminal_ids[1], "type": "message", "role": "assistant", "status": "completed"}
                ]
            }
        }),
    ]
    .map(|event| format!("data: {event}"))
    .join("\n")
}

#[test]
fn strict_relay_decoder_accepts_data_lines_without_a_space() {
    let stream = message_stream(["msg_terminal_0", "msg_terminal_1"]).replace("data: ", "data:");

    decode_upstream(&request_context(), UpstreamBody::Sse(&stream))
        .expect("SSE data fields may omit the optional space after the colon");
}

#[test]
fn strict_relay_decoder_rejects_duplicate_terminal_item_ids() {
    let stream = message_stream(["msg_terminal", "msg_terminal"]);

    let error = decode_upstream(&request_context(), UpstreamBody::Sse(&stream))
        .expect_err("duplicate terminal item ids must be rejected");
    assert!(error.to_string().contains("repeats output item 'msg_terminal'"));
}

#[test]
fn strict_relay_decoder_rejects_repeated_item_done() {
    let stream = message_stream(["msg_terminal_0", "msg_terminal_1"]);
    let mut lines: Vec<_> = stream.lines().collect();
    lines.insert(4, lines[3]);
    let stream = lines.join("\n");

    let error = decode_upstream(&request_context(), UpstreamBody::Sse(&stream))
        .expect_err("strict validation must reject a second completion for the same item");

    assert!(error.to_string().contains("has no active output item"));
}

#[test]
fn strict_relay_decoder_rejects_item_id_with_the_wrong_output_index() {
    let stream = message_stream(["msg_terminal_0", "msg_terminal_1"]);
    let stream = stream.replace(
        r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_streamed_0""#,
        r#"{"type":"response.output_item.done","output_index":1,"item":{"id":"msg_streamed_0""#,
    );

    let error = decode_upstream(&request_context(), UpstreamBody::Sse(&stream))
        .expect_err("an item id must not resolve through a different explicit output index");

    assert!(error.to_string().contains("does not match its active output item"));
}

#[test]
fn strict_relay_decoder_rejects_changed_call_id_without_terminal_output() {
    let stream = [
        json!({"type": "response.created", "response": {"id": "resp_upstream", "status": "in_progress"}}),
        json!({"type": "response.in_progress", "response": {"id": "resp_upstream", "status": "in_progress"}}),
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"id": "fc_1", "type": "function_call", "status": "in_progress", "call_id": "call_1", "name": "lookup", "arguments": ""}}),
        json!({"type": "response.function_call_arguments.done", "output_index": 0, "item_id": "fc_1", "call_id": "call_2", "arguments": "{}"}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": {"id": "fc_1", "type": "function_call", "status": "completed", "call_id": "call_2", "name": "lookup", "arguments": "{}"}}),
        json!({"type": "response.completed", "response": {"id": "resp_upstream", "status": "completed"}}),
    ]
    .map(|event| format!("data: {event}"))
    .join("\n");

    let error = decode_upstream(&request_context(), UpstreamBody::Sse(&stream))
        .expect_err("call-id stability must not depend on terminal output being present");

    assert!(error.to_string().contains("changes 'call_id' for output[0]"));
}

#[test]
fn strict_relay_decoder_rejects_unsupported_item_type() {
    let stream = message_stream(["msg_terminal_0", "msg_terminal_1"]).replace("\"message\"", "\"unsupported_item\"");

    let error = decode_upstream(&request_context(), UpstreamBody::Sse(&stream))
        .expect_err("unsupported item types must be rejected before lifecycle validation");

    assert!(
        error
            .to_string()
            .contains("output item type 'unsupported_item' is unsupported")
    );
}

#[test]
fn strict_relay_decoder_preserves_web_search_item_id_fallback() {
    for id in [Some(json!("ws_1")), None, Some(json!(null)), Some(json!(""))] {
        for include_terminal_output in [false, true] {
            let mut item = json!({
                "type": "web_search_call",
                "item_id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "rust", "sources": []}
            });
            if let Some(id) = &id {
                item["id"] = id.clone();
            }
            let mut terminal = json!({"id": "resp_upstream", "status": "completed"});
            if include_terminal_output {
                terminal["output"] = json!([item.clone()]);
            }
            let stream = [
                json!({"type": "response.created", "response": {"id": "resp_upstream", "status": "in_progress"}}),
                json!({"type": "response.in_progress", "response": {"id": "resp_upstream", "status": "in_progress"}}),
                json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"}}),
                json!({"type": "response.output_item.done", "output_index": 0, "item": item}),
                json!({"type": "response.completed", "response": terminal}),
            ]
            .map(|event| format!("data: {event}"))
            .join("\n");

            let payload = decode_upstream(&request_context(), UpstreamBody::Sse(&stream))
                .unwrap_or_else(|error| panic!("id={id:?}, terminal output={include_terminal_output}: {error}"));

            assert_eq!(
                payload.output.len(),
                1,
                "id={id:?}, terminal output={include_terminal_output}"
            );
            let OutputItem::WebSearchCall(call) = &payload.output[0] else {
                panic!("expected web-search call");
            };
            assert_eq!(call.id, "ws_1");
            assert_eq!(serde_json::to_value(&call.action).unwrap()["query"], "rust");
        }
    }
}

#[test]
fn strict_relay_decoder_accepts_compatible_recorded_responses_streams() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cassettes");
    let mut decoded = 0;
    let mut rejected = Vec::new();

    for path in yaml_files(&root) {
        for (turn_index, chunks) in response_streams(&path).into_iter().enumerate() {
            let stream = chunks.join("\n");
            if !is_responses_event_stream(&stream) {
                continue;
            }
            match decode_upstream(&request_context(), UpstreamBody::Sse(&stream)) {
                Ok(_) => decoded += 1,
                Err(error) => rejected.push(format!(
                    "{} turn {}: {error}",
                    path.strip_prefix(&root).expect("cassette below root").display(),
                    turn_index + 1
                )),
            }
        }
    }

    let unexpected: Vec<_> = rejected
        .iter()
        .filter(|failure| !failure.contains("changes 'call_id'"))
        .collect();
    assert!(
        unexpected.is_empty(),
        "recorded streams failed for reasons other than the call-ID stability rule:\n{}",
        unexpected
            .iter()
            .map(|failure| failure.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert_eq!(
        rejected.len(),
        10,
        "the recorded call-ID incompatibility baseline changed:\n{}",
        rejected.join("\n")
    );
    assert!(decoded >= 50, "decoded only {decoded} recorded Responses SSE streams");
}
