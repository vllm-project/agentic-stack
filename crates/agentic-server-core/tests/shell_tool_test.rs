//! Constructed regression fixtures, not live OpenAI/vLLM recordings.
use std::fmt::Write;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use agentic_core::executor::request::RequestContext;
use agentic_core::executor::{ExecuteRequest, UpstreamBody, decode_upstream};
use agentic_core::tool::shell::CancellationToken;
use agentic_core::tool::{GatewayExecutorRegistration, ShellExecutor, ToolError};
use agentic_core::types::io::{InputItem, OutputItem, ShellCall, ShellCallOutputContent, ShellCallStatus};
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
fn shell_history_and_choice_lower_only_at_inference_boundary() {
    let mut request = request(false);
    request.input = serde_json::from_value(json!([shell_item("completed"), shell_output()])).unwrap();
    let upstream = serde_json::to_value(request.to_upstream_request(false).unwrap()).unwrap();
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

#[derive(Default)]
struct SandboxAdapter {
    calls: Mutex<Vec<ShellCall>>,
}

impl ShellExecutor for SandboxAdapter {
    fn execute(
        &self,
        call: ShellCall,
        cancellation: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ShellCallOutputContent>, ToolError>> + Send + '_>> {
        Box::pin(async move {
            assert!(!cancellation.is_cancelled());
            assert_eq!(call.action.commands, ["pwd"]);
            assert_eq!(call.action.timeout_ms, Some(1000));
            assert_eq!(call.action.max_output_length, Some(128));
            self.calls.lock().unwrap().push(call);
            Ok(serde_json::from_value(shell_output()["output"].clone()).unwrap())
        })
    }
}

#[tokio::test]
async fn external_shell_executor_completes_two_rounds_and_rehydrates() {
    for stream in [false, true] {
        let fixture = support::TestFixture::new_with_responses(vec![
            model_response(stream, true),
            model_response(stream, false),
            model_response(stream, false),
        ])
        .await;
        let adapter = Arc::new(SandboxAdapter::default());
        let ctx = agentic_core::executor::ExecutionContext::new(
            fixture.exec_ctx.conv_handler.clone(),
            fixture.exec_ctx.resp_handler.clone(),
            fixture.exec_ctx.client.clone(),
            fixture.exec_ctx.llm_base_url.clone(),
        )
        .with_gateway_executor(GatewayExecutorRegistration::Shell(adapter.clone()));
        let ctx = Arc::new(ctx);
        let first = run(request(stream), ctx.clone()).await;
        assert_eq!(support::output_text(&first), "sandbox checked");
        assert_eq!(adapter.calls.lock().unwrap().len(), 1);
        assert!(
            matches!(&first.output[0], OutputItem::ShellCall(call) if call.status == Some(ShellCallStatus::Completed))
        );
        let requests = fixture.request_bodies().await;
        assert_eq!(requests.len(), 2);
        assert!(
            requests[1].get("tool_choice").is_none(),
            "auto is omitted on the upstream wire"
        );
        assert!(
            requests[1]["input"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["type"] == "function_call_output")
        );
        let mut continuation = request(stream);
        continuation.previous_response_id = Some(first.id);
        run(continuation, ctx).await;
        let requests = fixture.request_bodies().await;
        let history = requests[2]["input"].as_array().unwrap();
        assert_eq!(
            history.iter().filter(|item| item["type"] == "function_call").count(),
            1,
            "don't replay the public shell projection twice"
        );
    }
}
