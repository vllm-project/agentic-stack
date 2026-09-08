//! Acceptance test for the streaming Messages-native gateway tool loop (#115).
//!
//! Drives `run_messages_stream` against a mock vLLM `/v1/messages` that replays
//! the recorded #123 **streaming** cassette (turn 0 streams a `web_search`
//! `tool_use`; turn 1 streams the final text) as `text/event-stream`, plus a
//! mock You.com backend. Asserts the client sees ONE logical message
//! (`message_start`/`message_stop` once), the gateway `tool_use` is suppressed,
//! block indices stay contiguous across rounds, and no raw per-round terminal
//! leaks.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use agentic_core::executor::{
    BoxStream, ConversationHandler, ExecutionContext, MessagesRequestContext, MessagesUpstream, ResponseHandler,
    run_messages_stream,
};
use agentic_core::storage::{ConversationStore, ResponseStore};
use agentic_core::tool::{ToolRegistry, WebSearchHandler};
use agentic_core::types::messages::{GatewayToolMap, ToolParam, registry_tools};
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures::StreamExt;
use http::StatusCode;
use serde_json::Value;
use tokio::net::TcpListener;

mod support;

const CASSETTE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/cassettes/messages/messages-web-search-Qwen-Qwen3-30B-A3B-FP8-streaming.yaml"
);

const MULTIROUND: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/cassettes/messages_multiround/multiround-web-search-qwen3-streaming.yaml"
);
const CLAUDE_CODE_CACHE_CONTROL_REQUEST: &str = include_str!("fixtures/claude-code-cache-control-request.json");

/// Load each streaming turn's SSE body (the raw event-stream text) from the cassette.
fn cassette_turn_streams() -> Vec<String> {
    streams_at(CASSETTE)
}

fn streams_at(path: &str) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let doc: Value = serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"));
    doc["turns"]
        .as_array()
        .expect("turns")
        .iter()
        .map(|t| {
            let mut body = t["response"]["sse"]
                .as_array()
                .expect("sse array")
                .iter()
                .map(|l| l.as_str().unwrap_or_default())
                .collect::<String>();
            if !body.contains("data: [DONE]") {
                body.push_str("data: [DONE]\n\n");
            }
            body
        })
        .collect()
}

#[derive(Clone)]
struct UpstreamState {
    streams: Arc<Vec<String>>,
    calls: Arc<AtomicUsize>,
    requests: Arc<tokio::sync::Mutex<Vec<Value>>>,
}

async fn spawn_mock_vllm_stream(streams: Vec<String>) -> (String, UpstreamState, tokio::task::JoinHandle<()>) {
    let state = UpstreamState {
        streams: Arc::new(streams),
        calls: Arc::new(AtomicUsize::new(0)),
        requests: Arc::new(tokio::sync::Mutex::new(Vec::new())),
    };
    let app = Router::new()
        .route(
            "/v1/messages",
            post(
                |State(st): State<UpstreamState>, Json(request): Json<Value>| async move {
                    let n = st.calls.fetch_add(1, Ordering::SeqCst);
                    st.requests.lock().await.push(request);
                    let body = st.streams.get(n).cloned().unwrap_or_default();
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(body))
                        .unwrap()
                        .into_response()
                },
            ),
        )
        .with_state(state.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    (format!("http://{addr}"), state, handle)
}

async fn spawn_mock_vllm_stream_then_error(
    first_stream: String,
    error_body: &'static str,
) -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let route_calls = Arc::clone(&calls);
    let app = Router::new().route(
        "/v1/messages",
        post(move |_body: axum::body::Bytes| {
            let n = route_calls.fetch_add(1, Ordering::SeqCst);
            let first_stream = first_stream.clone();
            async move {
                if n == 0 {
                    Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(first_stream))
                        .unwrap()
                        .into_response()
                } else {
                    Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(error_body))
                        .unwrap()
                        .into_response()
                }
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    (format!("http://{addr}"), calls, handle)
}

async fn spawn_mock_search() -> (String, tokio::task::JoinHandle<()>) {
    let app = Router::new().route(
        "/v1/search",
        post(|Json(_body): Json<Value>| async move {
            Json(serde_json::json!({
                "results": {"web": [{"url": "https://www.rust-lang.org/", "title": "Rust",
                    "description": "d", "snippets": ["Rust 1.89.0 is the latest stable release."]}], "news": []},
                "metadata": {"query": "q", "search_uuid": "s1", "latency": 0.1}
            }))
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    (format!("http://{addr}"), handle)
}

async fn build_exec_ctx(vllm_url: &str, search_url: &str) -> Arc<ExecutionContext> {
    let pool = support::setup_pool().await;
    let conv = ConversationHandler::new(ConversationStore::new(Arc::clone(&pool)));
    let resp = ResponseHandler::new(ResponseStore::new(pool));
    let client = Arc::new(reqwest::Client::new());
    Arc::new(
        ExecutionContext::new(conv, resp, Arc::clone(&client), vllm_url.to_owned()).with_gateway_executor(Arc::new(
            WebSearchHandler::with_api_key(client, "test-key".to_owned(), search_url),
        )),
    )
}

async fn run_test_messages_stream(
    request: Value,
    registry: Arc<ToolRegistry>,
    exec_ctx: Arc<ExecutionContext>,
) -> BoxStream {
    let upstream = MessagesUpstream::new(&exec_ctx.llm_base_url, None, reqwest::header::HeaderMap::new());
    let ctx = MessagesRequestContext::from_value(request).expect("request context");
    run_messages_stream(ctx, registry, exec_ctx, upstream)
        .await
        .map(|response| response.body)
        .unwrap()
}

#[tokio::test]
async fn messages_stream_presents_one_message_and_hides_gateway_tool() {
    let (vllm_url, upstream, _v) = spawn_mock_vllm_stream(cassette_turn_streams()).await;
    let (search_url, _s) = spawn_mock_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;

    let request = serde_json::json!({
        "model": "qwen3", "max_tokens": 1024, "stream": true,
        "messages": [{"role": "user", "content": "What is the latest stable Rust release? Use web_search."}],
        "tools": [{"name": "web_search", "description": "s",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]
    });
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let mut registry_tool_params = registry_tools(Some(&tools), &GatewayToolMap::default());
    let mut gateway_executors = exec_ctx.gateway_executors.clone();
    let registry = Arc::new(
        ToolRegistry::build_with_handlers(&mut registry_tool_params, &mut gateway_executors)
            .await
            .unwrap(),
    );

    let stream = run_test_messages_stream(request, registry, Arc::clone(&exec_ctx)).await;
    let chunks: Vec<String> = stream.collect().await;
    let sse = chunks.join("");

    // Two upstream rounds ran (tool round + final).
    assert_eq!(
        upstream.calls.load(Ordering::SeqCst),
        2,
        "one tool round + one final round"
    );

    // Exactly one logical message lifecycle.
    assert_eq!(
        sse.matches("event: message_start").count(),
        1,
        "one message_start: {sse:?}"
    );
    assert_eq!(sse.matches("event: message_stop").count(), 1, "one message_stop");
    assert_eq!(
        sse.matches("event: message_delta").count(),
        1,
        "one terminal message_delta (intermediate suppressed)"
    );

    // Gateway tool_use suppressed — no tool_use content block surfaces.
    assert!(
        !sse.contains(r#""type":"tool_use""#),
        "gateway tool_use must be hidden from the client stream"
    );

    // Final terminal is end_turn (not the intermediate tool_use).
    assert!(sse.contains(r#""stop_reason":"end_turn""#), "terminal is end_turn");

    // Block indices contiguous across rounds (no reset/collision): parse every
    // content_block_start index and assert 0..N with no dup.
    let mut indices: Vec<u64> = Vec::new();
    for line in sse.lines() {
        if let Some(d) = line.strip_prefix("data: ") {
            if let Ok(ev) = serde_json::from_str::<Value>(d) {
                if ev["type"] == "content_block_start" {
                    indices.push(ev["index"].as_u64().expect("index"));
                }
            }
        }
    }
    assert!(!indices.is_empty(), "some blocks surfaced");
    assert_eq!(
        indices,
        (0..indices.len() as u64).collect::<Vec<_>>(),
        "surfaced block indices contiguous across rounds: {indices:?}"
    );
}

#[tokio::test]
async fn messages_stream_preserves_error_event_after_gateway_tool_round() {
    let error_body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad second round"}}"#;
    let first_stream = cassette_turn_streams().remove(0);
    let (vllm_url, calls, _v) = spawn_mock_vllm_stream_then_error(first_stream, error_body).await;
    let (search_url, _s) = spawn_mock_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;
    let request = serde_json::json!({
        "model": "qwen3", "max_tokens": 1024, "stream": true,
        "messages": [{"role": "user", "content": "Use web_search."}],
        "tools": [{"name": "web_search", "description": "s", "input_schema": {"type": "object"}}]
    });
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let mut registry_tool_params = registry_tools(Some(&tools), &GatewayToolMap::default());
    let mut gateway_executors = exec_ctx.gateway_executors.clone();
    let registry = Arc::new(
        ToolRegistry::build_with_handlers(&mut registry_tool_params, &mut gateway_executors)
            .await
            .unwrap(),
    );

    let stream = run_test_messages_stream(request, registry, Arc::clone(&exec_ctx)).await;
    let sse = stream.collect::<Vec<_>>().await.join("");

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(
        sse.contains(&format!("event: error\ndata: {error_body}\n\n")),
        "the original Anthropic error object should survive as the SSE error event: {sse}"
    );
}

#[tokio::test]
async fn messages_stream_forwards_upstream_sse_error_and_stops() {
    let error = r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
    let upstream_stream = format!(
        "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"m\"}}}}\n\n\
         event: error\ndata: {error}\n\n\
         event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n\
         data: [DONE]\n\n"
    );
    let (vllm_url, upstream, _v) = spawn_mock_vllm_stream(vec![upstream_stream]).await;
    let (search_url, _s) = spawn_mock_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;
    let request = serde_json::json!({
        "model": "qwen3", "max_tokens": 1024, "stream": true,
        "messages": [{"role": "user", "content": "Use web_search."}],
        "tools": [{"name": "web_search", "description": "s", "input_schema": {"type": "object"}}]
    });
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let mut registry_tool_params = registry_tools(Some(&tools), &GatewayToolMap::default());
    let mut gateway_executors = exec_ctx.gateway_executors.clone();
    let registry = Arc::new(
        ToolRegistry::build_with_handlers(&mut registry_tool_params, &mut gateway_executors)
            .await
            .unwrap(),
    );

    let stream = run_test_messages_stream(request, registry, Arc::clone(&exec_ctx)).await;
    let sse = stream.collect::<Vec<_>>().await.join("");

    assert_eq!(upstream.calls.load(Ordering::SeqCst), 1);
    assert!(sse.contains("event: error"), "error event missing: {sse}");
    assert!(
        sse.contains(r#""type":"overloaded_error""#),
        "upstream error payload missing: {sse}"
    );
    assert!(
        !sse.contains("event: message_stop"),
        "an error must terminate without message_stop: {sse}"
    );
}

// Multi-round streaming: replay the live-recorded multi-round streaming cassette
// and assert the same single-lifecycle / contiguous-index / hidden-tool
// invariants hold across a tool round + a final round.
#[tokio::test]
async fn messages_stream_multiround_single_lifecycle() {
    let (vllm_url, upstream, _v) = spawn_mock_vllm_stream(streams_at(MULTIROUND)).await;
    let (search_url, _s) = spawn_mock_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;

    let request = serde_json::json!({
        "model": "qwen3", "max_tokens": 1024, "stream": true,
        "messages": [{"role": "user", "content": "Use web_search for the latest rust version, then its date."}],
        "tools": [{"name": "web_search", "description": "s",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]
    });
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let mut registry_tool_params = registry_tools(Some(&tools), &GatewayToolMap::default());
    let mut gateway_executors = exec_ctx.gateway_executors.clone();
    let registry = Arc::new(
        ToolRegistry::build_with_handlers(&mut registry_tool_params, &mut gateway_executors)
            .await
            .unwrap(),
    );

    let stream = run_test_messages_stream(request, registry, Arc::clone(&exec_ctx)).await;
    let sse = stream.collect::<Vec<_>>().await.join("");

    assert!(
        upstream.calls.load(Ordering::SeqCst) >= 2,
        "at least a tool round + a final round"
    );
    assert_eq!(
        sse.matches("event: message_start").count(),
        1,
        "one message_start across rounds"
    );
    assert_eq!(
        sse.matches("event: message_stop").count(),
        1,
        "one message_stop across rounds"
    );
    assert!(
        !sse.contains(r#""type":"tool_use""#),
        "gateway tool_use suppressed in the client stream"
    );
    assert!(
        !sse.contains("response.output_text.delta"),
        "no raw Responses SSE leaks"
    );
    // Contiguous surfaced indices across rounds.
    let mut idx = Vec::new();
    for line in sse.lines() {
        if let Some(d) = line.strip_prefix("data: ") {
            if let Ok(ev) = serde_json::from_str::<Value>(d) {
                if ev["type"] == "content_block_start" {
                    idx.push(ev["index"].as_u64().expect("index"));
                }
            }
        }
    }
    assert_eq!(
        idx,
        (0..idx.len() as u64).collect::<Vec<_>>(),
        "contiguous indices: {idx:?}"
    );
}

#[tokio::test]
async fn messages_stream_preserves_multi_block_system_across_rounds() {
    let (vllm_url, upstream, _v) = spawn_mock_vllm_stream(cassette_turn_streams()).await;
    let (search_url, _s) = spawn_mock_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;

    let system = serde_json::json!([
        {"type": "text", "text": "<attribution>session-1</attribution>"},
        {"type": "text", "text": "You are helpful."}
    ]);
    let request = serde_json::json!({
        "model": "qwen3",
        "max_tokens": 1024,
        "stream": true,
        "system": system,
        "messages": [{"role": "user", "content": "What is the latest stable Rust release? Use web_search."}],
        "tools": [{"name": "web_search", "description": "s",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]
    });
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let mut registry_tool_params = registry_tools(Some(&tools), &GatewayToolMap::default());
    let mut gateway_executors = exec_ctx.gateway_executors.clone();
    let registry = Arc::new(
        ToolRegistry::build_with_handlers(&mut registry_tool_params, &mut gateway_executors)
            .await
            .unwrap(),
    );

    let stream = run_test_messages_stream(request, registry, Arc::clone(&exec_ctx)).await;
    let _chunks: Vec<String> = stream.collect().await;

    assert_eq!(
        upstream.calls.load(Ordering::SeqCst),
        2,
        "one tool round + one final round"
    );
    let requests = upstream.requests.lock().await;
    assert_eq!(
        requests[0]["system"], system,
        "round 1 forwards the system blocks verbatim"
    );
    assert_eq!(
        requests[1]["system"], system,
        "round 2 still carries the system blocks unchanged"
    );
}

#[tokio::test]
async fn messages_stream_preserves_claude_code_cache_control_across_rounds() {
    let streams = cassette_turn_streams()
        .into_iter()
        .map(|stream| stream.replace(r#""web_search""#, r#""WebSearch""#))
        .collect();
    let (vllm_url, upstream, _v) = spawn_mock_vllm_stream(streams).await;
    let (search_url, _s) = spawn_mock_search().await;
    let mut exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;
    Arc::get_mut(&mut exec_ctx).unwrap().messages_gateway_tools =
        GatewayToolMap::from_pairs([("WebSearch", "web_search")]);

    let mut request: Value = serde_json::from_str(CLAUDE_CODE_CACHE_CONTROL_REQUEST).unwrap();
    request["stream"] = Value::Bool(true);
    let original_system = request["system"].clone();
    let original_user_message = request["messages"][0].clone();
    let original_tools = request["tools"].clone();
    let tools: Vec<ToolParam> = serde_json::from_value(original_tools.clone()).unwrap();
    let mut registry_tool_params = registry_tools(Some(&tools), &exec_ctx.messages_gateway_tools);
    let mut gateway_executors = exec_ctx.gateway_executors.clone();
    let registry = Arc::new(
        ToolRegistry::build_with_handlers(&mut registry_tool_params, &mut gateway_executors)
            .await
            .unwrap(),
    );

    let stream = run_test_messages_stream(request, registry, Arc::clone(&exec_ctx)).await;
    let _chunks: Vec<String> = stream.collect().await;

    let requests = upstream.requests.lock().await;
    assert_eq!(requests.len(), 2, "tool round + final round");
    for upstream_request in requests.iter() {
        assert_eq!(upstream_request["system"], original_system);
        assert_eq!(upstream_request["messages"][0], original_user_message);
        assert_eq!(upstream_request["tools"], original_tools);
    }
    assert_eq!(
        requests[1]["system"][0]["cache_control"],
        serde_json::json!({"type": "ephemeral", "ttl": "1h"})
    );
    assert_eq!(
        requests[1]["system"][1]["cache_control"],
        serde_json::json!({"type": "ephemeral", "ttl": "5m"})
    );
    assert_eq!(
        requests[1]["messages"][0]["content"][0]["cache_control"],
        serde_json::json!({"type": "ephemeral", "ttl": "5m"})
    );
    assert_eq!(
        requests[1]["tools"][1]["cache_control"],
        serde_json::json!({"type": "ephemeral", "ttl": "1h"})
    );
    assert!(requests[1]["tools"][0].get("cache_control").is_none());
}
