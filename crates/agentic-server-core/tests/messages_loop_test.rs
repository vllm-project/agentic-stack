//! Acceptance test for the Messages-native gateway tool loop (#115, non-streaming).
//!
//! Drives `run_messages_loop` against a mock vLLM `/v1/messages` upstream that
//! replays the recorded #123 cassette (turn 0: model emits a `web_search`
//! `tool_use`; turn 1: final text after the fed-back `tool_result`) and a mock
//! You.com search backend. Asserts the gateway tool is executed server-side,
//! hidden from the client, and only the final assistant message surfaces.
//!
//! The #123 cassette records real vLLM `/v1/messages` upstream turns — exactly
//! what this loop consumes — so replaying it is a faithful acceptance test, not
//! a hand-authored mock.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use agentic_core::executor::{
    ConversationHandler, ExecutionContext, ExecutorResult, MessagesRequestContext, MessagesUpstream, ResponseHandler,
    run_messages_loop,
};
use agentic_core::storage::{ConversationStore, ResponseStore};
use agentic_core::tool::{ToolRegistry, WebSearchHandler};
use agentic_core::types::messages::{GatewayToolMap, ToolParam, registry_tools};
use axum::extract::State;
use axum::http::Uri;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::Value;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

mod support;

const CASSETTE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/cassettes/messages/messages-web-search-Qwen-Qwen3-30B-A3B-FP8-nonstreaming.yaml"
);

/// Load the recorded assistant response bodies (one per turn) from a cassette.
fn cassette_turn_bodies() -> Vec<Value> {
    cassette_bodies_at(CASSETTE)
}

fn cassette_bodies_at(path: &str) -> Vec<Value> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let doc: Value = serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"));
    doc["turns"]
        .as_array()
        .expect("turns array")
        .iter()
        .map(|t| t["response"]["body"].clone())
        .collect()
}

const MULTIROUND_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/messages_multiround");
const CLAUDE_CODE_CACHE_CONTROL_REQUEST: &str = include_str!("fixtures/claude-code-cache-control-request.json");

/// Mock vLLM `/v1/messages` — serves the recorded response bodies in order and
/// records each request body it received (to assert the loop fed the
/// `tool_result` back on round 2).
#[derive(Clone)]
struct UpstreamState {
    bodies: Arc<Vec<Value>>,
    calls: Arc<AtomicUsize>,
    requests: Arc<tokio::sync::Mutex<Vec<Value>>>,
}

async fn spawn_mock_vllm_messages(bodies: Vec<Value>) -> (String, UpstreamState, tokio::task::JoinHandle<()>) {
    let state = UpstreamState {
        bodies: Arc::new(bodies),
        calls: Arc::new(AtomicUsize::new(0)),
        requests: Arc::new(tokio::sync::Mutex::new(Vec::new())),
    };
    let app = Router::new()
        .route(
            "/v1/messages",
            post(|State(st): State<UpstreamState>, Json(req): Json<Value>| async move {
                let n = st.calls.fetch_add(1, Ordering::SeqCst);
                st.requests.lock().await.push(req);
                let body = st.bodies.get(n).cloned().unwrap_or_else(|| {
                    serde_json::json!({"type": "error", "error": {"type": "api_error", "message": "mock exhausted"}})
                });
                Json(body)
            }),
        )
        .with_state(state.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    (format!("http://{addr}"), state, handle)
}

struct CapturedSearch {
    body: Value,
}

async fn recv_search(captured: &mut mpsc::UnboundedReceiver<CapturedSearch>, expectation: &str) -> CapturedSearch {
    tokio::time::timeout(std::time::Duration::from_secs(5), captured.recv())
        .await
        .unwrap_or_else(|_| panic!("{expectation}: timed out after 5 seconds"))
        .unwrap_or_else(|| panic!("{expectation}: mock server stopped before receiving a request"))
}

fn query_params_as_json(uri: &Uri) -> Value {
    let mut params = serde_json::Map::new();
    for (key, value) in url::form_urlencoded::parse(uri.query().unwrap_or_default().as_bytes()) {
        let value = if let Ok(number) = value.parse::<u64>() {
            Value::from(number)
        } else {
            Value::String(value.into_owned())
        };
        let key = key.into_owned();
        match params.remove(&key) {
            None => {
                params.insert(key, value);
            }
            Some(Value::Array(mut values)) => {
                values.push(value);
                params.insert(key, Value::Array(values));
            }
            Some(previous) => {
                params.insert(key, Value::Array(vec![previous, value]));
            }
        }
    }
    Value::Object(params)
}

/// Mock You.com search backend the `web_search` executor calls. Uses an
/// unbounded channel so a test that doesn't drain captures (e.g. the max-rounds
/// cap, which fires ~10 searches) never blocks the handler.
async fn spawn_mock_search() -> (
    String,
    mpsc::UnboundedReceiver<CapturedSearch>,
    tokio::task::JoinHandle<()>,
) {
    let (tx, rx) = mpsc::unbounded_channel();
    let app = Router::new()
        .route(
            "/v1/search",
            get(
                |State(tx): State<mpsc::UnboundedSender<CapturedSearch>>, uri: Uri| async move {
                    let body = query_params_as_json(&uri);
                    let _ = tx.send(CapturedSearch { body });
                    Json(serde_json::json!({
                        "results": {"web": [{"url": "https://www.rust-lang.org/", "title": "Rust",
                            "description": "d", "snippets": ["Rust 1.89.0 is the latest stable release."]}], "news": []},
                        "metadata": {"query": "q", "search_uuid": "s1", "latency": 0.1}
                    }))
                },
            ),
        )
        .with_state(tx);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    (format!("http://{addr}"), rx, handle)
}

/// Search backend that always 500s — drives a gateway tool dispatch failure (E5).
async fn spawn_failing_search() -> (String, mpsc::Receiver<()>, tokio::task::JoinHandle<()>) {
    let (tx, rx) = mpsc::channel(8);
    let app = Router::new()
        .route(
            "/v1/search",
            get(|State(tx): State<mpsc::Sender<()>>| async move {
                let _ = tx.send(()).await;
                (http::StatusCode::INTERNAL_SERVER_ERROR, "search backend down")
            }),
        )
        .with_state(tx);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    (format!("http://{addr}"), rx, handle)
}

#[allow(clippy::result_large_err)]
async fn build_exec_ctx(vllm_url: &str, search_url: &str) -> ExecutionContext {
    let pool = support::setup_pool().await;
    let conv = ConversationHandler::new(ConversationStore::new(Arc::clone(&pool)));
    let resp = ResponseHandler::new(ResponseStore::new(pool));
    let client = Arc::new(reqwest::Client::new());
    ExecutionContext::new(conv, resp, Arc::clone(&client), vllm_url.to_owned()).with_gateway_executor(Arc::new(
        WebSearchHandler::with_api_key(client, "test-key".to_owned(), search_url),
    ))
}

async fn build_tool_registry(tools: &Vec<ToolParam>, exec_ctx: &ExecutionContext) -> ToolRegistry {
    let mut registry_tool_params = registry_tools(Some(tools), &exec_ctx.messages_gateway_tools);
    let mut gateway_executors = exec_ctx.gateway_executors.clone();
    ToolRegistry::build_with_handlers(&mut registry_tool_params, &mut gateway_executors)
        .await
        .unwrap()
}

#[allow(clippy::result_large_err)]
async fn run_test_messages_loop(
    request: Value,
    registry: &ToolRegistry,
    exec_ctx: &ExecutionContext,
) -> ExecutorResult<Value> {
    let upstream = MessagesUpstream::new(&exec_ctx.llm_base_url, None, reqwest::header::HeaderMap::new());
    run_messages_loop(
        MessagesRequestContext::from_value(request)?,
        registry,
        exec_ctx,
        &upstream,
    )
    .await
    .map(|response| response.body)
}

fn web_search_request() -> Value {
    serde_json::json!({
        "model": "qwen3",
        "max_tokens": 1024,
        "stream": false,
        "messages": [{"role": "user", "content": "What is the latest stable Rust release? Use web_search."}],
        "tools": [{"name": "web_search", "description": "Search the web.",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]
    })
}

#[tokio::test]
async fn messages_loop_hides_gateway_tool_and_surfaces_final_text() {
    let (vllm_url, upstream, _v) = spawn_mock_vllm_messages(cassette_turn_bodies()).await;
    let (search_url, mut captured, _s) = spawn_mock_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;

    let request = web_search_request();
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let registry = build_tool_registry(&tools, &exec_ctx).await;

    let result = run_test_messages_loop(request, &registry, &exec_ctx)
        .await
        .expect("loop runs");

    // The gateway executed the web_search server-side.
    let search = recv_search(&mut captured, "search backend hit").await;
    assert!(search.body.get("query").is_some(), "web_search dispatched with a query");

    // Two upstream rounds, and round 2 carried the fed-back tool_result.
    assert_eq!(
        upstream.calls.load(Ordering::SeqCst),
        2,
        "one tool round + one final round"
    );
    let reqs = upstream.requests.lock().await;
    let round2_msgs = reqs[1]["messages"].as_array().expect("round-2 messages");
    let has_tool_result = round2_msgs.iter().any(|m| {
        m["content"]
            .as_array()
            .is_some_and(|blocks| blocks.iter().any(|b| b["type"] == "tool_result"))
    });
    assert!(has_tool_result, "round 2 fed the tool_result back to the model");

    // Hide-the-call: the returned message has NO tool_use block — only the final
    // thinking/text the client should see.
    let content = result["content"].as_array().expect("final content");
    assert!(
        !content.iter().any(|b| b["type"] == "tool_use"),
        "gateway tool_use must be hidden: {content:?}"
    );
    let text: String = content
        .iter()
        .filter(|b| b["type"] == "text")
        .map(|b| b["text"].as_str().unwrap_or_default())
        .collect();
    assert!(text.contains("1.89.0"), "final answer surfaces: {text}");
    assert_eq!(result["stop_reason"], "end_turn");
}

#[tokio::test]
async fn native_web_search_applies_domain_and_location_configuration() {
    let (vllm_url, _upstream, _v) = spawn_mock_vllm_messages(cassette_turn_bodies()).await;
    let (search_url, mut captured, _s) = spawn_mock_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;
    let request = serde_json::json!({
        "model": "qwen3", "max_tokens": 1024, "stream": false,
        "messages": [{"role": "user", "content": "Search Rust's official site."}],
        "tools": [{
            "type": "web_search_20250305",
            "name": "web_search",
            "allowed_domains": ["rust-lang.org"],
            "user_location": {"type": "approximate", "country": "ca"}
        }]
    });
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let registry = build_tool_registry(&tools, &exec_ctx).await;

    let transport = MessagesUpstream::new(&exec_ctx.llm_base_url, None, reqwest::header::HeaderMap::new());
    let ctx = MessagesRequestContext::from_value(request).expect("request context");
    run_messages_loop(ctx, &registry, &exec_ctx, &transport)
        .await
        .expect("loop runs");

    let search = recv_search(&mut captured, "search backend hit").await;
    assert_eq!(search.body["include_domains"], "rust-lang.org");
    assert_eq!(search.body["country"], "CA");
}

#[tokio::test]
async fn native_web_search_applies_blocked_domains() {
    let (vllm_url, _upstream, _v) = spawn_mock_vllm_messages(cassette_turn_bodies()).await;
    let (search_url, mut captured, _s) = spawn_mock_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;
    let request = serde_json::json!({
        "model": "qwen3", "max_tokens": 1024, "stream": false,
        "messages": [{"role": "user", "content": "Search outside an excluded site."}],
        "tools": [{
            "type": "web_search_20250305",
            "name": "web_search",
            "blocked_domains": ["example.com"]
        }]
    });
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let registry = build_tool_registry(&tools, &exec_ctx).await;

    let transport = MessagesUpstream::new(&exec_ctx.llm_base_url, None, reqwest::header::HeaderMap::new());
    let ctx = MessagesRequestContext::from_value(request).expect("request context");
    run_messages_loop(ctx, &registry, &exec_ctx, &transport)
        .await
        .expect("loop runs");

    let search = recv_search(&mut captured, "search backend hit").await;
    assert_eq!(search.body["exclude_domains"], "example.com");
}

#[tokio::test]
async fn native_web_search_enforces_max_uses() {
    let round0 = serde_json::json!({
        "id": "m", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [
            {"type": "tool_use", "id": "t1", "name": "web_search", "input": {"query": "rust one"}},
            {"type": "tool_use", "id": "t2", "name": "web_search", "input": {"query": "rust two"}}
        ],
        "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 3}
    });
    let round1 = serde_json::json!({
        "id": "m2", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [{"type": "text", "text": "Done."}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 3}
    });
    let (vllm_url, upstream, _v) = spawn_mock_vllm_messages(vec![round0, round1]).await;
    let (search_url, mut captured, _s) = spawn_mock_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;
    let request = serde_json::json!({
        "model": "qwen3", "max_tokens": 1024, "stream": false,
        "messages": [{"role": "user", "content": "Search twice."}],
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}]
    });
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let registry = build_tool_registry(&tools, &exec_ctx).await;

    let transport = MessagesUpstream::new(&exec_ctx.llm_base_url, None, reqwest::header::HeaderMap::new());
    let ctx = MessagesRequestContext::from_value(request).expect("request context");
    run_messages_loop(ctx, &registry, &exec_ctx, &transport)
        .await
        .expect("loop runs");

    recv_search(&mut captured, "first search runs").await;
    assert!(captured.try_recv().is_err(), "second search must not run");
    let requests = upstream.requests.lock().await;
    let results = requests[1]["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|message| message["content"].as_array())
        .expect("tool results fed back");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["is_error"], false);
    assert_eq!(results[1]["is_error"], true);
    assert!(results[1]["content"].as_str().unwrap_or_default().contains("max_uses"));
}

// ── Repro tests for Maral's #131 review (currently FAILING — proves each bug) ──

// F3: the assistant turn fed back on the next round must preserve preceding
// thinking/text/signature blocks, not just the gateway tool_use. Dropping them
// loses conversation state and breaks extended-thinking round-tripping.
#[tokio::test]
async fn repro_f3_next_round_preserves_thinking_and_text_blocks() {
    // Round 0: assistant emits thinking + text + a gateway tool_use.
    let round0 = serde_json::json!({
        "id": "m", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [
            {"type": "thinking", "thinking": "let me search", "signature": "sig_abc"},
            {"type": "text", "text": "I'll look that up."},
            {"type": "tool_use", "id": "t1", "name": "web_search", "input": {"query": "rust"}}
        ],
        "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 3}
    });
    let round1 = serde_json::json!({
        "id": "m2", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [{"type": "text", "text": "Rust 1.89.0."}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 3}
    });
    let (vllm_url, upstream, _v) = spawn_mock_vllm_messages(vec![round0, round1]).await;
    let (search_url, _rx, _s) = spawn_mock_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;
    let request = web_search_request();
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let registry = build_tool_registry(&tools, &exec_ctx).await;
    run_test_messages_loop(request, &registry, &exec_ctx).await.unwrap();

    // Inspect the assistant turn the loop appended for round 2.
    let reqs = upstream.requests.lock().await;
    let round2_msgs = reqs[1]["messages"].as_array().expect("round-2 messages");
    let assistant = round2_msgs
        .iter()
        .rev()
        .find(|m| m["role"] == "assistant")
        .expect("round-2 has a reconstructed assistant turn");
    let block_types: Vec<&str> = assistant["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|b| b["type"].as_str())
        .collect();
    // The model's thinking + text must be carried forward alongside the tool_use.
    assert!(
        block_types.contains(&"thinking"),
        "thinking preserved in history: {block_types:?}"
    );
    assert!(
        block_types.contains(&"text"),
        "text preserved in history: {block_types:?}"
    );
    assert!(block_types.contains(&"tool_use"), "tool_use present: {block_types:?}");
}

// F5: mixed gateway + client tool_use — the non-streaming path must NOT expose
// the gateway tool_use (hide-the-call), matching streaming. One consistent
// policy: surface the client tool_use, suppress the gateway one, stop the loop.
#[tokio::test]
async fn repro_f5_mixed_call_hides_gateway_tool_use() {
    let body = serde_json::json!({
        "id": "m", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [
            {"type": "tool_use", "id": "g1", "name": "web_search", "input": {"query": "x"}},
            {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "SF"}}
        ],
        "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 3}
    });
    let mut request = web_search_request();
    request["tools"] = serde_json::json!([
        {"name": "web_search", "description": "s", "input_schema": {"type": "object"}},
        {"name": "get_weather", "description": "w", "input_schema": {"type": "object"}}
    ]);
    let (result, _calls) = run_against(vec![body], request).await;
    let names: Vec<&str> = result["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|b| b["type"] == "tool_use")
        .filter_map(|b| b["name"].as_str())
        .collect();
    assert!(names.contains(&"get_weather"), "client tool_use surfaces: {names:?}");
    assert!(
        !names.contains(&"web_search"),
        "gateway tool_use must be hidden: {names:?}"
    );
}

/// Build a registry + `exec_ctx` for a canned single-turn upstream (no search
/// hit expected).
async fn run_against(bodies: Vec<Value>, request: Value) -> (Value, usize) {
    let (vllm_url, upstream, _v) = spawn_mock_vllm_messages(bodies).await;
    let (search_url, _rx, _s) = spawn_mock_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap_or_default();
    let registry = build_tool_registry(&tools, &exec_ctx).await;
    let result = run_test_messages_loop(request, &registry, &exec_ctx).await.unwrap();
    (result, upstream.calls.load(Ordering::SeqCst))
}

// E2: gateway tool declared but the model answers directly (end_turn) — one
// round, returned as-is, no loop.
#[tokio::test]
async fn messages_loop_returns_immediately_when_model_does_not_call_tool() {
    let body = serde_json::json!({
        "id": "m", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [{"type": "text", "text": "Rust 1.89.0."}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 3}
    });
    let (result, calls) = run_against(vec![body], web_search_request()).await;
    assert_eq!(calls, 1, "one round only");
    assert_eq!(result["stop_reason"], "end_turn");
    assert_eq!(result["content"][0]["text"], "Rust 1.89.0.");
}

// E7: a client-owned tool_use in the turn must be returned to the client (the
// loop can't execute it server-side), even if a gateway tool is also declared.
#[tokio::test]
async fn messages_loop_returns_client_owned_tool_use_to_client() {
    let body = serde_json::json!({
        "id": "m", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "SF"}}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 3}
    });
    // Declare BOTH a gateway tool and the client tool.
    let mut request = web_search_request();
    request["tools"] = serde_json::json!([
        {"name": "web_search", "description": "s", "input_schema": {"type": "object"}},
        {"name": "get_weather", "description": "w", "input_schema": {"type": "object"}}
    ]);
    let (result, calls) = run_against(vec![body], request).await;
    assert_eq!(calls, 1, "client tool_use ends the loop in one round");
    assert_eq!(result["stop_reason"], "tool_use");
    assert_eq!(
        result["content"][0]["name"], "get_weather",
        "client tool_use surfaces to the client"
    );
}

// Multi-round (3 rounds): replay the live-recorded sequential cassette
// (tool_use -> tool_use -> text). The loop must run three upstream rounds,
// hit the search backend twice, and surface only the final text.
#[tokio::test]
async fn messages_loop_multi_round_sequential() {
    let bodies = cassette_bodies_at(&format!(
        "{MULTIROUND_DIR}/sequential-web-search-qwen3-nonstreaming.yaml"
    ));
    assert_eq!(bodies.len(), 3, "cassette has 3 upstream rounds");
    let (vllm_url, upstream, _v) = spawn_mock_vllm_messages(bodies).await;
    let (search_url, mut captured, _s) = spawn_mock_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;
    let request = web_search_request();
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let registry = build_tool_registry(&tools, &exec_ctx).await;

    let result = run_test_messages_loop(request, &registry, &exec_ctx).await.unwrap();

    assert_eq!(upstream.calls.load(Ordering::SeqCst), 3, "three upstream rounds");
    // Two gateway searches executed (rounds 0 and 1).
    recv_search(&mut captured, "first search").await;
    recv_search(&mut captured, "second search").await;
    let content = result["content"].as_array().unwrap();
    assert!(!content.iter().any(|b| b["type"] == "tool_use"), "gateway tools hidden");
    assert_eq!(result["stop_reason"], "end_turn");
}

// Parallel: replay the live-recorded parallel cassette (two tool_use blocks in
// one turn). Both gateway calls execute; neither surfaces.
#[tokio::test]
async fn messages_loop_parallel_tool_use() {
    let bodies = cassette_bodies_at(&format!("{MULTIROUND_DIR}/parallel-web-search-qwen3-nonstreaming.yaml"));
    let (vllm_url, upstream, _v) = spawn_mock_vllm_messages(bodies).await;
    let (search_url, mut captured, _s) = spawn_mock_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;
    let request = web_search_request();
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let registry = build_tool_registry(&tools, &exec_ctx).await;

    let result = run_test_messages_loop(request, &registry, &exec_ctx).await.unwrap();

    // Two parallel gateway calls both executed against the backend.
    recv_search(&mut captured, "first parallel search").await;
    recv_search(&mut captured, "second parallel search").await;
    assert_eq!(upstream.calls.load(Ordering::SeqCst), 2, "tool round + final round");
    let content = result["content"].as_array().unwrap();
    assert!(!content.iter().any(|b| b["type"] == "tool_use"), "gateway tools hidden");
    // Round 2 fed BOTH tool_results back.
    let reqs = upstream.requests.lock().await;
    let round2 = reqs[1]["messages"].as_array().unwrap();
    let tool_results: usize = round2
        .iter()
        .filter_map(|m| m["content"].as_array())
        .flatten()
        .filter(|b| b["type"] == "tool_result")
        .count();
    assert_eq!(tool_results, 2, "both parallel tool_results fed back");
}

// E5: a gateway tool that fails to dispatch (search backend returns 500) becomes
// an error tool_result fed back to the model — never a whole-request failure.
#[tokio::test]
async fn messages_loop_tool_failure_becomes_error_tool_result() {
    // Round 0: web_search call; round 1: final text (model recovers from the error).
    let round0 = serde_json::json!({
        "id": "m", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [{"type": "tool_use", "id": "t1", "name": "web_search", "input": {"query": "x"}}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 3}
    });
    let round1 = serde_json::json!({
        "id": "m2", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [{"type": "text", "text": "Search failed, here's what I know."}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 3}
    });
    let (vllm_url, upstream, _v) = spawn_mock_vllm_messages(vec![round0, round1]).await;
    // Search backend that returns 500 → dispatch error.
    let (search_url, mut captured, _s) = spawn_failing_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;
    let request = web_search_request();
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let registry = build_tool_registry(&tools, &exec_ctx).await;

    let result = run_test_messages_loop(request, &registry, &exec_ctx).await.unwrap();

    tokio::time::timeout(std::time::Duration::from_secs(5), captured.recv())
        .await
        .expect("failing search backend should be reached within 5 seconds")
        .expect("failing search backend should receive a request");

    // The request did NOT fail — it looped to a final answer.
    assert_eq!(result["stop_reason"], "end_turn");
    assert_eq!(upstream.calls.load(Ordering::SeqCst), 2);
    // Round 2 fed back a tool_result (carrying the error text), not a hard failure.
    let reqs = upstream.requests.lock().await;
    let round2 = reqs[1]["messages"].as_array().unwrap();
    let has_tool_result = round2
        .iter()
        .filter_map(|m| m["content"].as_array())
        .flatten()
        .any(|b| b["type"] == "tool_result");
    assert!(has_tool_result, "tool failure fed back as an (error) tool_result");
}

// E14: an upstream error body mid-loop is surfaced (not swallowed or looped on).
#[tokio::test]
async fn messages_loop_surfaces_upstream_error_body() {
    let err = serde_json::json!({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}});
    let (result, calls) = run_against(vec![err], web_search_request()).await;
    assert_eq!(calls, 1, "error surfaced on the first round, no loop");
    assert_eq!(result["type"], "error");
    assert_eq!(result["error"]["type"], "overloaded_error");
}

// E4: the loop caps at MAX rounds. Feed an unbounded run of tool_use rounds and
// assert it terminates with an error rather than looping forever.
#[tokio::test]
async fn messages_loop_caps_at_max_rounds() {
    let tool_round = serde_json::json!({
        "id": "m", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [{"type": "tool_use", "id": "t", "name": "web_search", "input": {"query": "x"}}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    // 20 tool rounds available, but the loop must stop at its cap (10).
    let bodies = vec![tool_round; 20];
    let (vllm_url, upstream, _v) = spawn_mock_vllm_messages(bodies).await;
    let (search_url, _rx, _s) = spawn_mock_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;
    let request = web_search_request();
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let registry = build_tool_registry(&tools, &exec_ctx).await;

    let result = run_test_messages_loop(request, &registry, &exec_ctx).await.unwrap();

    let calls = upstream.calls.load(Ordering::SeqCst);
    assert!(calls <= 10, "loop must cap at MAX_GATEWAY_TOOL_ROUNDS (got {calls})");
    assert_eq!(result["type"], "error", "round-budget exhaustion surfaces an error");
    assert!(
        result["error"]["message"].as_str().unwrap().contains("rounds"),
        "error mentions the round cap"
    );
}

// E3: a malformed tool_use.input (not an object) must not panic — the arg
// stringification falls back and the call still dispatches.
#[tokio::test]
async fn messages_loop_handles_malformed_tool_input() {
    let round0 = serde_json::json!({
        "id": "m", "type": "message", "role": "assistant", "model": "qwen3",
        // input as a bare string instead of an object.
        "content": [{"type": "tool_use", "id": "t1", "name": "web_search", "input": "not-an-object"}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let round1 = serde_json::json!({
        "id": "m2", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}
    });
    let (result, calls) = run_against(vec![round0, round1], web_search_request()).await;
    assert_eq!(calls, 2, "loop still ran the tool round + final");
    assert_eq!(result["stop_reason"], "end_turn");
}

// #116 (Stage 3 parity): a multi-block `system` (attribution block + instructions,
// the shape Claude Code sends) must survive the gateway tool loop unchanged. The
// loop only appends the assistant turn + tool_result to `messages`
// (`append_round_to_history`) and must never touch `system` — so round 2's
// upstream body must carry the identical `system` blocks it started with.
#[tokio::test]
async fn messages_loop_preserves_multi_block_system_across_rounds() {
    let round0 = serde_json::json!({
        "id": "m", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [{"type": "tool_use", "id": "t1", "name": "web_search", "input": {"query": "x"}}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 3}
    });
    let round1 = serde_json::json!({
        "id": "m2", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [{"type": "text", "text": "Rust 1.89.0."}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 3}
    });
    let (vllm_url, upstream, _v) = spawn_mock_vllm_messages(vec![round0, round1]).await;
    let (search_url, _rx, _s) = spawn_mock_search().await;
    let exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;

    let system = serde_json::json!([
        {"type": "text", "text": "<attribution>session-1</attribution>"},
        {"type": "text", "text": "You are helpful."}
    ]);
    let mut request = web_search_request();
    request["system"] = system.clone();
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let registry = build_tool_registry(&tools, &exec_ctx).await;

    let result = run_test_messages_loop(request, &registry, &exec_ctx).await.unwrap();
    assert_eq!(result["stop_reason"], "end_turn");
    assert_eq!(upstream.calls.load(Ordering::SeqCst), 2, "tool round + final round");

    // Both upstream rounds must carry the identical multi-block `system`.
    let reqs = upstream.requests.lock().await;
    assert_eq!(reqs[0]["system"], system, "round 1 forwards the system blocks verbatim");
    assert_eq!(
        reqs[1]["system"], system,
        "round 2 (after append_round_to_history) still carries the system blocks unchanged"
    );
}

#[tokio::test]
async fn messages_loop_preserves_claude_code_cache_control_across_rounds() {
    let round0 = serde_json::json!({
        "id": "m", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [{"type": "tool_use", "id": "t1", "name": "WebSearch", "input": {"query": "rust"}}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 3}
    });
    let round1 = serde_json::json!({
        "id": "m2", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [{"type": "text", "text": "Rust is stable."}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 3}
    });
    let (vllm_url, upstream, _v) = spawn_mock_vllm_messages(vec![round0, round1]).await;
    let (search_url, mut search_requests, _s) = spawn_mock_search().await;
    let mut exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;
    exec_ctx.messages_gateway_tools = GatewayToolMap::from_pairs([("WebSearch", "web_search")]);

    let request: Value = serde_json::from_str(CLAUDE_CODE_CACHE_CONTROL_REQUEST).unwrap();
    let original_system = request["system"].clone();
    let original_user_message = request["messages"][0].clone();
    let original_tools = request["tools"].clone();
    let tools: Vec<ToolParam> = serde_json::from_value(original_tools.clone()).unwrap();
    let registry = build_tool_registry(&tools, &exec_ctx).await;

    let result = run_test_messages_loop(request, &registry, &exec_ctx).await.unwrap();
    assert_eq!(result["stop_reason"], "end_turn");
    let search_request = search_requests
        .try_recv()
        .expect("WebSearch alias dispatches to the gateway search backend");
    assert_eq!(search_request.body["query"], "rust");

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

/// The dual-view context exists so the upstream body stays the client's own
/// JSON. A typed round-trip through `MessagesRequest` would not: `ContentBlock`
/// models only the fields the gateway reads and catches everything else in
/// `#[serde(other)] Unknown`, which re-serializes as a literal
/// `{"type":"unknown"}` block. This locks in that the raw body — not the typed
/// view — is what reaches vLLM, across a gateway round.
#[tokio::test]
async fn messages_loop_preserves_unmodeled_blocks_and_tool_result_fields_across_rounds() {
    let round0 = serde_json::json!({
        "id": "m", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [{"type": "tool_use", "id": "t2", "name": "WebSearch", "input": {"query": "rust"}}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 3}
    });
    let round1 = serde_json::json!({
        "id": "m2", "type": "message", "role": "assistant", "model": "qwen3",
        "content": [{"type": "text", "text": "Done."}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 3}
    });
    let (vllm_url, upstream, _v) = spawn_mock_vllm_messages(vec![round0, round1]).await;
    let (search_url, _search_requests, _s) = spawn_mock_search().await;
    let mut exec_ctx = build_exec_ctx(&vllm_url, &search_url).await;
    exec_ctx.messages_gateway_tools = GatewayToolMap::from_pairs([("WebSearch", "web_search")]);

    // Every block here is either unmodeled by `ContentBlock` (`image`,
    // `redacted_thinking`) or carries fields it does not model (`citations`,
    // `is_error`, `cache_control`, a non-text `tool_result` part).
    let request = serde_json::json!({
        "model": "qwen3", "max_tokens": 1024, "stream": false,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "What is this?",
                 "citations": [{"type": "web_search_result_location", "url": "https://example.com"}]},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
            ]},
            {"role": "assistant", "content": [
                {"type": "redacted_thinking", "data": "encrypted-blob"},
                {"type": "tool_use", "id": "t1", "name": "WebSearch", "input": {"query": "prior"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "is_error": true,
                 "cache_control": {"type": "ephemeral", "ttl": "5m"},
                 "content": [
                     {"type": "text", "text": "search failed"},
                     {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BBBB"}}
                 ]}
            ]}
        ],
        "tools": [{"name": "WebSearch", "description": "Search the web",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}}]
    });
    let original_messages = request["messages"].clone();
    let tools: Vec<ToolParam> = serde_json::from_value(request["tools"].clone()).unwrap();
    let registry = build_tool_registry(&tools, &exec_ctx).await;

    let result = run_test_messages_loop(request, &registry, &exec_ctx).await.unwrap();
    assert_eq!(result["stop_reason"], "end_turn");

    let requests = upstream.requests.lock().await;
    assert_eq!(requests.len(), 2, "tool round + final round");
    for upstream_request in requests.iter() {
        let messages = upstream_request["messages"].as_array().expect("messages");
        assert_eq!(
            &messages[..3],
            original_messages.as_array().unwrap().as_slice(),
            "the client's own blocks reach upstream untouched"
        );
        // The failure mode a typed round-trip would introduce.
        let rendered = serde_json::to_string(upstream_request).unwrap();
        assert!(
            !rendered.contains(r#""type":"unknown""#),
            "no block collapsed to Unknown"
        );
    }
    // Round 2 carries the appended gateway turn on top of the untouched prefix.
    let round_two = requests[1]["messages"].as_array().expect("messages");
    assert_eq!(round_two.len(), 5, "3 client turns + assistant turn + tool_result turn");
    assert_eq!(round_two[3]["content"][0]["id"], "t2");
    assert_eq!(round_two[4]["content"][0]["type"], "tool_result");
}
