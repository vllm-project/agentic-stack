//! Constructed regression fixtures, not live OpenAI/vLLM recordings.
use std::fmt::Write;
use std::sync::Arc;

use agentic_core::executor::request::RequestContext;
use agentic_core::executor::{ExecuteRequest, UpstreamBody, decode_upstream};
use agentic_core::types::io::{InputItem, OutputItem, ResponsesInput, ShellCall, ShellCallStatus};
use agentic_core::types::request_response::{RequestPayload, ResponsePayload};
use either::Either;
use futures::StreamExt;
use serde_json::{Value, json};

mod support;

fn request(stream: bool) -> RequestPayload {
    serde_json::from_value(json!({
        "model": "test-model", "input": "Inspect the sandbox", "store": true, "stream": stream,
        "tools": [{"type": "shell", "environment": {"type": "local"}}],
        "tool_choice": {"type": "shell"}
    }))
    .unwrap()
}

fn shell_item(status: &str) -> Value {
    json!({"type": "shell_call", "id": "sh_1", "call_id": "call_1", "status": status,
        "action": {"commands": ["pwd"], "timeout_ms": 1000, "max_output_length": 128}})
}

fn shell_output() -> Value {
    json!({"type": "shell_call_output", "call_id": "call_1", "output": [
        {"stdout": "/sandbox\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}
    ]})
}

fn context() -> RequestContext {
    let request = request(true);
    RequestContext {
        original_request: request.clone(),
        enriched_request: request,
        new_input_items: Vec::new(),
        response_id: "resp_reserved".to_owned(),
        conversation_id: None,
        conversation_version: None,
    }
}

fn sse(events: &[Value]) -> String {
    let mut stream = String::new();
    for event in events {
        write!(&mut stream, "data: {event}\n\n").unwrap();
    }
    stream.push_str("data: [DONE]\n\n");
    stream
}

fn lifecycle(item: &Value) -> Vec<Value> {
    let mut added = item.clone();
    added["status"] = json!("in_progress");
    vec![
        json!({"type": "response.created", "response": {"id": "resp_upstream", "status": "in_progress"}}),
        json!({"type": "response.in_progress", "response": {"id": "resp_upstream", "status": "in_progress"}}),
        json!({"type": "response.output_item.added", "output_index": 0, "item": added}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": item}),
        json!({"type": "response.completed", "response": {"id": "resp_upstream", "status": "completed", "output": [item]}}),
    ]
}

#[test]
fn shell_input_bytes_have_one_discriminator_and_keep_extensions() {
    for mut wire in [shell_item("completed"), shell_output()] {
        wire["future_field"] = json!(true);
        let item: InputItem = serde_json::from_value(wire.clone()).unwrap();
        let bytes = serde_json::to_string(&item).unwrap();
        let tag = format!("\"type\":\"{}\"", wire["type"].as_str().unwrap());
        assert_eq!(
            bytes.matches(&tag).count(),
            1,
            "duplicate tag in actual wire bytes: {bytes}"
        );
        assert_eq!(serde_json::from_str::<Value>(&bytes).unwrap(), wire);
    }
}

#[test]
fn prepared_shell_history_and_choice_match_upstream_function_tools() {
    let mut request = request(false);
    request.input = serde_json::from_value(json!([shell_item("completed"), shell_output()])).unwrap();
    let mut prepared = request.clone();
    prepared.input = ResponsesInput::Items(Vec::from(&request.input));
    let upstream = serde_json::to_value(prepared.to_upstream_request(false).unwrap()).unwrap();
    assert_eq!(upstream["tool_choice"], json!({"type": "function", "name": "shell"}));
    assert_eq!(upstream["tools"][0]["name"], "shell");
    assert_eq!(upstream["input"][0]["type"], "function_call");
    assert_eq!(upstream["input"][0]["call_id"], upstream["input"][1]["call_id"]);
    assert_eq!(upstream["input"][1]["type"], "function_call_output");
    let action: Value = serde_json::from_str(upstream["input"][0]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(action, shell_item("completed")["action"]);
    let output: Value = serde_json::from_str(upstream["input"][1]["output"].as_str().unwrap()).unwrap();
    assert_eq!(output, shell_output()["output"]);
    assert_eq!(
        serde_json::to_value(&request.input).unwrap(),
        json!([shell_item("completed"), shell_output()])
    );
    assert_eq!(
        serde_json::to_value(&request.tool_choice).unwrap(),
        json!({"type": "shell"})
    );
}

#[test]
fn native_shell_stream_obeys_strict_lifecycle() {
    for status in ["completed", "incomplete"] {
        let events = lifecycle(&shell_item(status));
        let response = decode_upstream(&context(), UpstreamBody::Sse(&sse(&events))).expect("native shell lifecycle");
        assert_eq!(serde_json::to_value(&response.output[0]).unwrap(), shell_item(status));
        let frame = agentic_core::events::normalize_sse_line(&format!("data: {}", events[2])).unwrap();
        let added = ShellCall::try_from(&frame.payload).unwrap();
        assert_eq!(added.action.commands, ["pwd"]);
        assert!(!added.extra.contains_key("type"));
        let mut changed_id = events.clone();
        changed_id[3]["item"]["id"] = json!("sh_wrong");
        assert!(decode_upstream(&context(), UpstreamBody::Sse(&sse(&changed_id))).is_err());
        let mut changed_call = events.clone();
        changed_call[3]["item"]["call_id"] = json!("call_wrong");
        assert!(decode_upstream(&context(), UpstreamBody::Sse(&sse(&changed_call))).is_err());
        let mut changed_index = events.clone();
        changed_index[3]["output_index"] = json!(1);
        assert!(decode_upstream(&context(), UpstreamBody::Sse(&sse(&changed_index))).is_err());
        let mut repeated = events.clone();
        repeated.insert(4, events[3].clone());
        assert!(decode_upstream(&context(), UpstreamBody::Sse(&sse(&repeated))).is_err());
    }
}

fn model_response(stream: bool, shell: bool) -> support::MockResponse {
    let item = if shell {
        json!({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "shell", "status": "completed",
            "arguments": shell_item("completed")["action"].to_string()})
    } else {
        json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "sandbox checked", "annotations": []}]})
    };
    if stream {
        support::MockResponse::Sse(sse(&lifecycle(&item)))
    } else {
        support::MockResponse::Json(
            json!({"id": "resp_upstream", "object": "response", "model": "test-model",
            "status": "completed", "output": [item]})
            .to_string(),
        )
    }
}

async fn run(request: RequestPayload, ctx: Arc<agentic_core::executor::ExecutionContext>) -> ResponsePayload {
    match ExecuteRequest::new(request, ctx).run().await.unwrap() {
        Either::Left(response) => response,
        Either::Right(stream) => {
            let chunks = stream.collect::<Vec<_>>().await;
            let events = support::streamed_sse_events(&chunks);
            let response = events
                .iter()
                .find(|event| event["type"] == "response.completed")
                .expect("completed response");
            for event in events
                .iter()
                .filter(|event| event["type"] == "response.output_item.done" && event["item"]["type"] == "shell_call")
            {
                assert_eq!(event["item"]["status"], "completed");
            }
            // A consumer outside the gateway must be able to strictly replay its stream.
            decode_upstream(&context(), UpstreamBody::Sse(&chunks.join(""))).expect("public strict replay");
            serde_json::from_value(response["response"].clone()).unwrap()
        }
    }
}

#[tokio::test]
async fn client_shell_continuation_blocking_and_streaming() {
    for stream in [false, true] {
        let fixture =
            support::TestFixture::new_with_responses(vec![model_response(stream, true), model_response(stream, false)])
                .await;
        let first = run(request(stream), fixture.exec_ctx.clone()).await;
        let OutputItem::ShellCall(call) = &first.output[0] else {
            panic!("public shell call")
        };
        assert_eq!(call.status, Some(ShellCallStatus::Completed));
        assert_eq!(
            fixture.request_bodies().await.len(),
            1,
            "default shell must wait for client execution"
        );
        let mut continuation = request(stream);
        continuation.previous_response_id = Some(first.id);
        continuation.input = serde_json::from_value(json!([shell_output()])).unwrap();
        let final_response = run(continuation, fixture.exec_ctx.clone()).await;
        assert_eq!(support::output_text(&final_response), "sandbox checked");
        let requests = fixture.request_bodies().await;
        let history = requests[1]["input"].as_array().unwrap();
        assert_eq!(history.iter().filter(|item| item["type"] == "function_call").count(), 1);
        assert_eq!(
            history
                .iter()
                .filter(|item| item["type"] == "function_call_output")
                .count(),
            1
        );
        assert!(
            !history
                .iter()
                .any(|item| item["type"] == "shell_call" || item["type"] == "shell_call_output")
        );
    }
}

#[test]
fn recorded_shell_streams_replay_strictly() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cassettes/shell");
    for (provider, model) in [("openai-reference", "gpt-5.6"), ("gateway", "Qwen-Qwen3.5-35B-A3B-FP8")] {
        for scenario in ["success", "nonzero-exit", "timeout", "multiple-commands"] {
            let path = root.join(format!("shell-{provider}-{scenario}-{model}-streaming.yaml"));
            let cassette = support::load_cassette(path.to_str().unwrap());
            assert_eq!(cassette.turns.len(), 2);
            for (index, turn) in cassette.turns.iter().enumerate() {
                let wire = turn.response.sse.as_ref().unwrap().join("");
                let response = decode_upstream(&context(), UpstreamBody::Sse(&wire))
                    .unwrap_or_else(|error| panic!("{} turn {index}: {error}", path.display()));
                if index == 0 {
                    let call = response
                        .output
                        .iter()
                        .find_map(|item| match item {
                            OutputItem::ShellCall(call) => Some(call),
                            _ => None,
                        })
                        .expect("shell call");
                    assert_eq!(call.status, Some(ShellCallStatus::Completed));
                    let expected_commands = match scenario {
                        "multiple-commands" => 3..=4,
                        "nonzero-exit" => 1..=2,
                        _ => 1..=1,
                    };
                    assert!(expected_commands.contains(&call.action.commands.len()));
                } else {
                    assert!(!support::output_text(&response).is_empty());
                }
            }
        }
    }
}

fn command_lifecycle() -> Vec<Value> {
    let item = shell_item("completed");
    let mut events = lifecycle(&item);
    events[2]["item"]["action"] = json!({"commands": [], "timeout_ms": null, "max_output_length": null});
    events.splice(3..3, [
        json!({"type": "response.shell_call_command.added", "output_index": 0, "command_index": 0, "command": ""}),
        json!({"type": "response.shell_call_command.delta", "output_index": 0, "command_index": 0, "delta": "pw"}),
        json!({"type": "response.shell_call_command.delta", "output_index": 0, "command_index": 0, "delta": "d"}),
        json!({"type": "response.shell_call_command.done", "output_index": 0, "command_index": 0, "command": "pwd"}),
    ]);
    events
}

#[test]
fn shell_command_stream_validates_indices_order_and_final_commands() {
    let events = command_lifecycle();
    decode_upstream(&context(), UpstreamBody::Sse(&sse(&events))).unwrap();
    for failure in [
        "index",
        "item-id",
        "command-index",
        "missing-index",
        "negative-index",
        "string-index",
        "missing-delta",
        "before-added",
        "duplicate-added",
        "duplicate-done",
        "after-done",
        "unfinished",
        "contradict-done",
        "contradict-item",
        "wrong-kind",
    ] {
        let mut bad = events.clone();
        match failure {
            "index" => bad[4]["output_index"] = json!(1),
            "item-id" => bad[4]["item_id"] = json!("wrong"),
            "command-index" => bad[4]["command_index"] = json!(1),
            "missing-index" => {
                bad[4].as_object_mut().unwrap().remove("command_index");
            }
            "negative-index" => bad[4]["command_index"] = json!(-1),
            "string-index" => bad[4]["command_index"] = json!("0"),
            "missing-delta" => {
                bad[4].as_object_mut().unwrap().remove("delta");
            }
            "before-added" => {
                bad.remove(3);
            }
            "duplicate-added" => bad.insert(4, events[3].clone()),
            "duplicate-done" => bad.insert(7, events[6].clone()),
            "after-done" => bad.insert(7, events[4].clone()),
            "unfinished" => {
                bad.remove(6);
            }
            "contradict-done" => bad[6]["command"] = json!("other"),
            "contradict-item" => bad[7]["item"]["action"]["commands"] = json!(["other"]),
            "wrong-kind" => bad[2]["item"] = json!({"type":"message","id":"sh_1","role":"assistant","content":[]}),
            _ => unreachable!(),
        }
        assert!(
            decode_upstream(&context(), UpstreamBody::Sse(&sse(&bad))).is_err(),
            "accepted {failure}"
        );
    }
}

// Compare semantic shell events; IDs, sequence numbers, and delta boundaries
// belong to individual responses and are not provider compatibility requirements.
fn recorded_shell_lifecycle(events: &[Value], call: &Value) -> Vec<Value> {
    let added = events
        .iter()
        .find(|event| event["type"] == "response.output_item.added" && event["item"]["id"] == call["id"])
        .expect("shell item added");
    assert_eq!(added["item"]["status"], "in_progress");
    assert_eq!(added["item"]["action"]["commands"], json!([]));
    assert!(added["item"]["action"]["timeout_ms"].is_null());
    assert!(added["item"]["action"]["max_output_length"].is_null());
    let mut commands = Vec::<String>::new();
    let mut trace = Vec::new();
    for event in events {
        let kind = event["type"].as_str().unwrap();
        if kind.starts_with("response.shell_call_command.") {
            assert_eq!(event["output_index"], added["output_index"]);
            let index = usize::try_from(event["command_index"].as_u64().unwrap()).unwrap();
            match kind {
                "response.shell_call_command.added" => {
                    assert_eq!(index, commands.len());
                    assert_eq!(event["command"], "");
                    commands.push(String::new());
                }
                "response.shell_call_command.delta" => {
                    commands[index].push_str(event["delta"].as_str().unwrap());
                    continue;
                }
                "response.shell_call_command.done" => assert_eq!(event["command"], commands[index]),
                _ => panic!("unexpected shell event: {kind}"),
            }
            trace.push(json!({"type": kind, "command_index": index, "command": event["command"]}));
        } else if event["item"]["id"] == call["id"] {
            assert_eq!(event["output_index"], added["output_index"]);
            assert_eq!(event["item"]["call_id"], call["call_id"]);
            if kind == "response.output_item.done" {
                assert_eq!(event["item"]["action"], call["action"]);
                assert_eq!(event["item"]["status"], "completed");
            }
            trace.push(json!({"type": kind, "status": event["item"]["status"], "action": event["item"]["action"]}));
        }
    }
    assert_eq!(json!(commands), call["action"]["commands"]);
    assert_eq!(trace.first().unwrap()["type"], "response.output_item.added");
    assert_eq!(trace.last().unwrap()["type"], "response.output_item.done");
    trace
}

fn recorded_shell_contract(provider: &str, model: &str, scenario: &str, streaming: bool) -> Value {
    let mode = if streaming { "streaming" } else { "nonstreaming" };
    let path = format!(
        "{}/tests/cassettes/shell/shell-{provider}-{scenario}-{model}-{mode}.yaml",
        env!("CARGO_MANIFEST_DIR")
    );
    let cassette = support::load_cassette(&path);
    assert_eq!(cassette.turns.len(), 2, "{path}");
    let mut responses = Vec::new();
    let mut first_events = Vec::new();
    for (index, turn) in cassette.turns.iter().enumerate() {
        assert_eq!(turn.request.path, "/v1/responses");
        assert_eq!(turn.request.body.stream, streaming);
        assert!(turn.request.body.store);
        assert_eq!(
            turn.request.body.tools,
            vec![json!({"type":"shell", "environment":{"type":"local"}})]
        );
        let body = if streaming {
            let chunks = turn.response.sse.as_ref().unwrap();
            let wire = chunks.join("");
            decode_upstream(&context(), UpstreamBody::Sse(&wire)).expect("strict shell stream replay");
            let events = support::streamed_sse_events(chunks);
            let completed = events
                .iter()
                .filter(|event| event["type"] == "response.completed")
                .collect::<Vec<_>>();
            assert_eq!(completed.len(), 1);
            let body = completed[0]["response"].clone();
            if index == 0 {
                first_events = events;
            }
            body
        } else {
            turn.response.body.clone().unwrap()
        };
        assert_eq!(body["status"], "completed", "{path} turn {index}");
        responses.push(body);
    }
    let calls = responses[0]["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "shell_call")
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    let call = calls[0];
    assert!(call["id"].as_str().unwrap().starts_with("sh_"));
    assert!(!call["call_id"].as_str().unwrap().is_empty());
    assert_eq!(call["status"], "completed");
    assert_eq!(call["action"]["timeout_ms"], 1000);
    assert_eq!(call["action"]["max_output_length"], 4096);
    let continuation = &cassette.turns[1].request.body;
    assert_eq!(
        continuation.previous_response_id.as_deref(),
        responses[0]["id"].as_str()
    );
    let input = continuation.input.as_array().unwrap();
    assert_eq!(input.len(), 2);
    assert_eq!(input[0]["type"], "shell_call_output");
    assert_eq!(input[0]["call_id"], call["call_id"]);
    assert_eq!(input[0]["max_output_length"], call["action"]["max_output_length"]);
    assert_eq!(
        input[0]["output"].as_array().unwrap().len(),
        call["action"]["commands"].as_array().unwrap().len()
    );
    assert_eq!(input[1]["role"], "user");
    let final_response: ResponsePayload = serde_json::from_value(responses[1].clone()).unwrap();
    assert!(!support::output_text(&final_response).trim().is_empty());
    assert!(!final_response.output.iter().any(|item| matches!(
        item,
        OutputItem::ShellCall(_) | OutputItem::FunctionCall(_) | OutputItem::CustomToolCall(_)
    )));
    json!({
        "action": call["action"], "status": call["status"],
        "output": input[0]["output"], "follow_up": input[1],
        "lifecycle": if streaming { recorded_shell_lifecycle(&first_events, call) } else { Vec::new() }
    })
}

#[test]
fn recorded_gateway_shell_contract_matches_openai() {
    for scenario in ["success", "nonzero-exit", "timeout", "multiple-commands"] {
        for streaming in [false, true] {
            let reference = recorded_shell_contract("openai-reference", "gpt-5.6", scenario, streaming);
            let gateway = recorded_shell_contract("gateway", "Qwen-Qwen3.5-35B-A3B-FP8", scenario, streaming);
            assert_eq!(gateway, reference, "{scenario}, streaming={streaming}");
        }
    }
}
