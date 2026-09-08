#[allow(dead_code)]
mod common;

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::body::Bytes;
use axum::routing::post;
use tokio::net::TcpListener;
use tokio::sync::Mutex;

use agentic_core::tool::{GatewayExecutor, ToolError, ToolHandler, ToolOutput, ToolType};
use agentic_core::types::io::FunctionTool;
use agentic_core::types::tools::WebSearchToolParam;

use common::{spawn_gateway, test_config, test_state};

#[derive(Clone)]
struct TestWebSearchExecutor {
    calls: Arc<AtomicUsize>,
}

impl ToolHandler for TestWebSearchExecutor {
    type ToolParams = WebSearchToolParam;

    fn tool_type(&self) -> ToolType {
        ToolType::WebSearch
    }

    fn validate(&self, _params: &WebSearchToolParam) -> Result<(), ToolError> {
        Ok(())
    }

    fn normalize(&self, _params: &WebSearchToolParam) -> Vec<FunctionTool> {
        vec![FunctionTool {
            type_: "function".to_owned(),
            name: "web_search".to_owned(),
            description: Some("Search the web.".to_owned()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            })),
            strict: Some(false),
        }]
    }
}

impl GatewayExecutor for TestWebSearchExecutor {
    type ExecutionParams = WebSearchToolParam;

    fn execute(
        &self,
        call_id: &str,
        _tool_name: &str,
        _arguments: &str,
        _params: &WebSearchToolParam,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let call_id = call_id.to_owned();
        Box::pin(async move {
            Ok(ToolOutput {
                call_id,
                output: serde_json::json!({
                    "query": "rust compaction",
                    "results": {"web": [], "news": []},
                    "metadata": {}
                })
                .to_string(),
            })
        })
    }
}

async fn spawn_compaction_model() -> (String, Arc<Mutex<Vec<serde_json::Value>>>, tokio::task::JoinHandle<()>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&requests);
    let app = Router::new().route(
        "/v1/responses",
        post(move |body: Bytes| {
            let captured = Arc::clone(&captured);
            async move {
                captured
                    .lock()
                    .await
                    .push(serde_json::from_slice(&body).expect("model request is JSON"));
                axum::Json(serde_json::json!({
                    "id": "resp_upstream",
                    "object": "response",
                    "created_at": 0,
                    "model": "test-model",
                    "status": "completed",
                    "output": [{
                        "id": "msg_upstream",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{
                            "type": "output_text",
                            "text": "preserved checkpoint summary",
                            "annotations": []
                        }]
                    }],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 5,
                        "total_tokens": 25
                    }
                }))
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind model");
    let address = listener.local_addr().expect("model address");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve model");
    });
    (format!("http://{address}"), requests, handle)
}

async fn spawn_sequential_model(
    responses: Vec<(&str, i64)>,
) -> (String, Arc<Mutex<Vec<serde_json::Value>>>, tokio::task::JoinHandle<()>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&requests);
    let responses = Arc::new(Mutex::new(VecDeque::from(
        responses
            .into_iter()
            .map(|(text, tokens)| (text.to_owned(), tokens))
            .collect::<Vec<_>>(),
    )));
    let app = Router::new().route(
        "/v1/responses",
        post(move |body: Bytes| {
            let captured = Arc::clone(&captured);
            let responses = Arc::clone(&responses);
            async move {
                captured
                    .lock()
                    .await
                    .push(serde_json::from_slice(&body).expect("model request is JSON"));
                let (text, tokens) = responses.lock().await.pop_front().expect("model response available");
                axum::Json(serde_json::json!({
                    "id": "resp_upstream",
                    "object": "response",
                    "created_at": 0,
                    "model": "test-model",
                    "status": "completed",
                    "output": [{
                        "id": "msg_upstream",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]
                    }],
                    "usage": {
                        "input_tokens": tokens - 1,
                        "output_tokens": 1,
                        "total_tokens": tokens
                    }
                }))
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind model");
    let address = listener.local_addr().expect("model address");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve model");
    });
    (format!("http://{address}"), requests, handle)
}

async fn spawn_compaction_tool_loop_model() -> (String, Arc<Mutex<Vec<serde_json::Value>>>, tokio::task::JoinHandle<()>)
{
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&requests);
    let responses = Arc::new(Mutex::new(VecDeque::from([
        serde_json::json!({
            "id": "resp_tool",
            "object": "response",
            "created_at": 0,
            "model": "test-model",
            "status": "completed",
            "output": [{
                "id": "fc_search",
                "type": "function_call",
                "call_id": "call_search",
                "name": "web_search",
                "arguments": "{\"query\":\"rust compaction\"}",
                "status": "completed"
            }],
            "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        }),
        serde_json::json!({
            "id": "resp_summary",
            "object": "response",
            "created_at": 0,
            "model": "test-model",
            "status": "completed",
            "output": [{
                "id": "msg_summary",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "automatic summary", "annotations": []}]
            }],
            "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}
        }),
        serde_json::json!({
            "id": "resp_answer",
            "object": "response",
            "created_at": 0,
            "model": "test-model",
            "status": "completed",
            "output": [{
                "id": "msg_answer",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "final answer", "annotations": []}]
            }],
            "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5}
        }),
    ])));
    let app = Router::new().route(
        "/v1/responses",
        post(move |body: Bytes| {
            let captured = Arc::clone(&captured);
            let responses = Arc::clone(&responses);
            async move {
                captured
                    .lock()
                    .await
                    .push(serde_json::from_slice(&body).expect("model request is JSON"));
                axum::Json(responses.lock().await.pop_front().expect("model response available"))
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind model");
    let address = listener.local_addr().expect("model address");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve model");
    });
    (format!("http://{address}"), requests, handle)
}

#[tokio::test]
async fn compact_endpoint_returns_reusable_canonical_window() {
    let (model_url, model_requests, _model) = spawn_compaction_model().await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&model_url))).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses/compact"))
        .json(&serde_json::json!({
            "model": "test-model",
            "input": [
                {"role": "user", "content": "retain this request"},
                {"type": "function_call_output", "call_id": "call_1", "output": "large result"}
            ],
            "tools": [],
            "parallel_tool_calls": true,
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "low"}
        }))
        .send()
        .await
        .expect("compact request");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = response.json().await.expect("compact response JSON");
    assert_eq!(body["object"], "response.compaction");
    assert_eq!(body["usage"]["total_tokens"], 25);
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(body["output"][0]["status"], "completed");
    assert_eq!(body["output"][1]["type"], "compaction");
    assert_eq!(body["output"][1]["encrypted_content"], "preserved checkpoint summary");

    let reused = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test-model",
            "input": body["output"].clone(),
            "store": false,
            "stream": false
        }))
        .send()
        .await
        .expect("reuse compacted output");
    assert_eq!(reused.status(), reqwest::StatusCode::OK);
    let reused_body: serde_json::Value = reused.json().await.expect("reuse response JSON");
    assert!(reused_body["id"].as_str().is_some_and(|id| id.starts_with("resp_")));

    let requests = model_requests.lock().await;
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0]["stream"], false);
    let final_input = requests[0]["input"]
        .as_array()
        .expect("model input array")
        .last()
        .unwrap();
    assert!(
        final_input["content"]
            .as_str()
            .expect("summary prompt")
            .contains("CONTEXT CHECKPOINT COMPACTION")
    );
    assert_eq!(requests[1]["input"].as_array().map(Vec::len), Some(2));
    assert_eq!(requests[1]["input"][0]["role"], "user");
    assert_eq!(requests[1]["input"][1]["role"], "assistant");
    assert_eq!(requests[1]["input"][1]["content"][0]["type"], "output_text");
    assert_eq!(
        requests[1]["input"][1]["content"][0]["text"],
        "preserved checkpoint summary"
    );
}

#[tokio::test]
async fn automatic_compaction_runs_above_threshold_and_accumulates_usage() {
    let (model_url, requests, _model) =
        spawn_sequential_model(vec![("automatic summary", 25), ("final answer", 10)]).await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&model_url))).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test-model",
            "input": [{"role": "user", "content": "x".repeat(200)}],
            "store": false,
            "stream": false,
            "reasoning": {"effort": "high"},
            "context_management": [{"type": "compaction", "compact_threshold": 10}]
        }))
        .send()
        .await
        .expect("automatic compaction request");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = response.json().await.expect("automatic response JSON");
    assert_eq!(body["usage"]["total_tokens"], 35);
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| request.get("context_management").is_none())
    );
    assert_eq!(requests[1]["input"].as_array().map(Vec::len), Some(2));
    assert_eq!(requests[1]["input"][0]["role"], "user");
    assert_eq!(requests[1]["input"][1]["role"], "assistant");
    assert_eq!(requests[1]["input"][1]["content"][0]["text"], "automatic summary");
    assert_eq!(requests[1]["reasoning"], serde_json::json!({"effort": "high"}));
}

#[tokio::test]
async fn automatic_compaction_accumulates_usage_across_gateway_tool_loop() {
    let (model_url, requests, _model) = spawn_compaction_tool_loop_model().await;
    let tool_calls = Arc::new(AtomicUsize::new(0));
    let mut state = test_state(&test_config(&model_url));
    state.exec_ctx = Arc::new(
        state
            .exec_ctx
            .as_ref()
            .clone()
            .with_gateway_executor(Arc::new(TestWebSearchExecutor {
                calls: Arc::clone(&tool_calls),
            })),
    );
    let (gateway_url, _gateway) = spawn_gateway(state).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test-model",
            "input": [{"role": "user", "content": "short"}],
            "tools": [{"type": "web_search_preview"}],
            "store": false,
            "stream": false,
            "context_management": [{"type": "compaction", "compact_threshold": 50}]
        }))
        .send()
        .await
        .expect("automatic compaction tool-loop request");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = response.json().await.expect("automatic response JSON");
    assert_eq!(body["usage"]["total_tokens"], 40);
    assert_eq!(tool_calls.load(Ordering::Relaxed), 1);

    let requests = requests.lock().await;
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0]["input"].as_array().map(Vec::len), Some(1));
    assert!(
        requests[1]["input"]
            .as_array()
            .and_then(|input| input.last())
            .and_then(|item| item["content"].as_str())
            .is_some_and(|content| content.contains("CONTEXT CHECKPOINT COMPACTION"))
    );
    assert_eq!(requests[2]["input"].as_array().map(Vec::len), Some(2));
    assert_eq!(requests[2]["input"][0]["role"], "user");
    assert_eq!(requests[2]["input"][1]["role"], "assistant");
    assert_eq!(requests[2]["input"][1]["content"][0]["text"], "automatic summary");
}

#[tokio::test]
async fn automatic_compaction_skips_below_threshold() {
    let (model_url, requests, _model) = spawn_sequential_model(vec![("final answer", 10)]).await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&model_url))).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test-model",
            "input": [{"role": "user", "content": "short"}],
            "store": false,
            "stream": false,
            "context_management": [{"type": "compaction", "compact_threshold": 100_000}]
        }))
        .send()
        .await
        .expect("below-threshold request");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["input"][0]["content"], "short");
}

#[tokio::test]
async fn compact_endpoint_rejects_missing_context() {
    let (model_url, requests, _model) = spawn_sequential_model(Vec::new()).await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&model_url))).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses/compact"))
        .json(&serde_json::json!({"model": "test-model"}))
        .send()
        .await
        .expect("invalid compact request");

    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    assert!(requests.lock().await.is_empty());
}

// --- Image preservation across compaction (issue #253) ---
//
// Token estimation for image-bearing messages is issue #255; these tests only
// assert that compaction keeps a retained image-bearing user message intact.
const RED_PIXEL_PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP4z8AAAAMBAQD3A0FDAAAAAElFTkSuQmCC";

fn image_message_content() -> serde_json::Value {
    serde_json::json!([
        {"type": "input_text", "text": "retained"},
        {"type": "input_image", "image_url": RED_PIXEL_PNG, "detail": "low"}
    ])
}

#[tokio::test]
async fn compaction_window_retains_image_bearing_user_message() {
    let (model_url, model_requests, _model) = spawn_compaction_model().await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&model_url))).await;
    let content = image_message_content();

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test-model",
            "store": false,
            "stream": false,
            "input": [
                {"type": "message", "role": "user", "content": "superseded by the checkpoint"},
                {
                    "type": "message",
                    "id": "msg_keep",
                    "role": "user",
                    "status": "completed",
                    "content": content
                },
                {"type": "compaction", "encrypted_content": "summary so far"},
                {"type": "message", "role": "user", "content": "after the checkpoint"}
            ]
        }))
        .send()
        .await
        .expect("compacted continuation request");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let requests = model_requests.lock().await;
    assert_eq!(requests.len(), 1);
    let model_input = requests[0]["input"].as_array().expect("model input");
    assert_eq!(model_input.len(), 3, "only the retained window reaches the model");
    assert_eq!(
        model_input[0]["content"], content,
        "a retained user message must keep its image parts"
    );
    assert_eq!(model_input[1]["role"], "assistant");
    assert_eq!(model_input[1]["content"][0]["text"], "summary so far");
    assert_eq!(model_input[2]["content"], "after the checkpoint");
}

#[tokio::test]
async fn compact_endpoint_preserves_retained_image_message() {
    let (model_url, model_requests, _model) = spawn_compaction_model().await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&model_url))).await;
    let content = image_message_content();

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses/compact"))
        .json(&serde_json::json!({
            "model": "test-model",
            "input": [
                {"type": "message", "role": "user", "content": content},
                {"type": "function_call_output", "call_id": "call_1", "output": "large result"}
            ],
            "tools": []
        }))
        .send()
        .await
        .expect("compact request");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = response.json().await.expect("compact response JSON");
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(
        body["output"][0]["content"], content,
        "the compacted window must carry the image-bearing user message forward"
    );
    assert_eq!(body["output"][1]["type"], "compaction");

    // Reusing the compacted window must still send the image to the model.
    let reused = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test-model",
            "input": body["output"].clone(),
            "store": false,
            "stream": false
        }))
        .send()
        .await
        .expect("reuse compacted output");
    assert_eq!(reused.status(), reqwest::StatusCode::OK);

    let requests = model_requests.lock().await;
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1]["input"][0]["content"], content);
    assert_eq!(requests[1]["input"][1]["role"], "assistant");
}
