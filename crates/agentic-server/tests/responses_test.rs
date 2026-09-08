mod common;

use axum::Router;
use axum::body::Bytes;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::post;
use http::StatusCode;
use std::convert::Infallible;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, oneshot};
use tokio_util::sync::CancellationToken;

use agentic_core::executor::{ConversationHandler, ExecutionContext, ResponseHandler};
use agentic_core::proxy::ProxyState;
use agentic_core::storage::{
    ConversationStore, DbPool, InOutItem, ResponseMetadata, ResponseStore, create_pool_with_schema,
};
use agentic_core::types::io::{InputItem, ResponsesInput};
use agentic_server::app::{AppState, WebSocketTracker};

use common::{spawn_gateway, spawn_mock_llm, test_config, test_state};

const COMPETING_RESPONSE_ID: &str = "resp_competing";
const CONFLICT_MESSAGE: &str = "conversation changed while the response was being generated; retry the request";

enum MockResponse {
    GatedJson {
        body: String,
        arrived: oneshot::Sender<()>,
        release: oneshot::Receiver<()>,
    },
    GatedSse {
        first_chunk: String,
        terminal_chunk: String,
        arrived: oneshot::Sender<()>,
        release: oneshot::Receiver<()>,
    },
}

struct MockResponsesServer {
    url: String,
    handle: tokio::task::JoinHandle<()>,
}

struct GatedSse {
    first_chunk: Option<Bytes>,
    terminal_chunk: Option<Bytes>,
    release: oneshot::Receiver<()>,
}

impl futures::Stream for GatedSse {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(first_chunk) = self.first_chunk.take() {
            return Poll::Ready(Some(Ok(first_chunk)));
        }
        if self.terminal_chunk.is_none() {
            return Poll::Ready(None);
        }
        match Pin::new(&mut self.release).poll(cx) {
            Poll::Ready(_) => Poll::Ready(self.terminal_chunk.take().map(Ok)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl MockResponsesServer {
    async fn start_gated_json(body: String) -> (Self, oneshot::Receiver<()>, oneshot::Sender<()>) {
        let (arrived_tx, arrived_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let server = Self::start(MockResponse::GatedJson {
            body,
            arrived: arrived_tx,
            release: release_rx,
        })
        .await;
        (server, arrived_rx, release_tx)
    }

    async fn start_gated_sse(
        first_chunk: String,
        terminal_chunk: String,
    ) -> (Self, oneshot::Receiver<()>, oneshot::Sender<()>) {
        let (arrived_tx, arrived_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let server = Self::start(MockResponse::GatedSse {
            first_chunk,
            terminal_chunk,
            arrived: arrived_tx,
            release: release_rx,
        })
        .await;
        (server, arrived_rx, release_tx)
    }

    async fn start(response: MockResponse) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let response = Arc::new(Mutex::new(Some(response)));
        let route_response = Arc::clone(&response);
        let app = Router::new().route(
            "/v1/responses",
            post(move || {
                let response = Arc::clone(&route_response);
                async move {
                    let response = response.lock().await.take().expect("mock response already consumed");
                    match response {
                        MockResponse::GatedJson { body, arrived, release } => {
                            let _ = arrived.send(());
                            let _ = release.await;
                            axum::response::Response::builder()
                                .status(StatusCode::OK)
                                .header(header::CONTENT_TYPE, "application/json")
                                .body(axum::body::Body::from(body))
                                .unwrap()
                                .into_response()
                        }
                        MockResponse::GatedSse {
                            first_chunk,
                            terminal_chunk,
                            arrived,
                            release,
                        } => {
                            let _ = arrived.send(());
                            axum::response::Response::builder()
                                .status(StatusCode::OK)
                                .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
                                .body(axum::body::Body::from_stream(GatedSse {
                                    first_chunk: Some(Bytes::from(first_chunk)),
                                    terminal_chunk: Some(Bytes::from(terminal_chunk)),
                                    release,
                                }))
                                .unwrap()
                                .into_response()
                        }
                    }
                }
            }),
        );
        let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            url: format!("http://{addr}"),
            handle,
        }
    }
}

impl Drop for MockResponsesServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

struct TestDb {
    path: PathBuf,
}

impl TestDb {
    fn new() -> Self {
        Self {
            path: std::env::temp_dir().join(format!("agentic_http_test_{}.db", uuid::Uuid::now_v7())),
        }
    }

    fn url(&self) -> String {
        format!("sqlite://{}", self.path.display())
    }
}

impl Drop for TestDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_file(self.path.with_extension("db-shm"));
        let _ = std::fs::remove_file(self.path.with_extension("db-wal"));
    }
}

struct StorageBackedState {
    state: AppState,
    pool: Arc<DbPool>,
    _db: TestDb,
}

async fn storage_backed_state(llm_url: &str) -> StorageBackedState {
    let db = TestDb::new();
    let pool = create_pool_with_schema(Some(&db.url())).await.unwrap();
    let config = test_config(llm_url);
    let client = Arc::new(reqwest::Client::new());
    let exec_ctx = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::new(Arc::clone(&pool))),
        ResponseHandler::new(ResponseStore::new(Arc::clone(&pool))),
        client,
        config.llm_api_base.clone(),
    ));
    let proxy_state = ProxyState::new(config.clone()).expect("proxy state");
    let state = AppState {
        proxy_state,
        exec_ctx,
        llm_readiness_client: agentic_core::readiness::llm_readiness_client().expect("readiness client"),
        readiness_tracker: agentic_server::app::ReadinessTracker::default(),
        shutdown_token: CancellationToken::new(),
        websocket_tracker: WebSocketTracker::default(),
        llm_api_base: config.llm_api_base,
        skip_llm_ready_check: config.skip_llm_ready_check,
        openai_api_key: config.openai_api_key,
        model_capabilities: std::sync::Arc::default(),
    };
    StorageBackedState { state, pool, _db: db }
}

async fn create_conversation(client: &reqwest::Client, gateway_url: &str) -> String {
    let response = client
        .post(format!("{gateway_url}/v1/conversations"))
        .json(&serde_json::json!({"store": true}))
        .send()
        .await
        .expect("conversation request");
    assert_eq!(response.status(), StatusCode::OK);
    response
        .json::<serde_json::Value>()
        .await
        .expect("conversation response JSON")["id"]
        .as_str()
        .expect("conversation ID")
        .to_owned()
}

fn competing_turn_items() -> Vec<InOutItem> {
    Vec::<InputItem>::from(&ResponsesInput::Text("competing turn".to_owned()))
        .into_iter()
        .map(InOutItem::Input)
        .collect()
}

async fn persist_competing_turn(pool: &Arc<DbPool>, conversation_id: &str) {
    ConversationStore::new(Arc::clone(pool))
        .persist(
            conversation_id,
            COMPETING_RESPONSE_ID,
            None,
            competing_turn_items(),
            &ResponseMetadata {
                model: "competing-model".to_owned(),
                ..ResponseMetadata::default()
            },
        )
        .await
        .expect("competing turn should persist");
}

fn conflict_error() -> serde_json::Value {
    serde_json::json!({
        "message": CONFLICT_MESSAGE,
        "type": "invalid_request_error",
        "code": "conversation_locked",
        "param": "conversation"
    })
}

fn sse_events(body: &str) -> Vec<serde_json::Value> {
    body.split("\n\n")
        .filter(|frame| !frame.is_empty() && *frame != "data: [DONE]")
        .map(|frame| {
            let (event_line, data_line) = frame.split_once('\n').expect("named SSE event and data lines");
            let event_name = event_line.strip_prefix("event: ").expect("SSE event header");
            let data = data_line.strip_prefix("data: ").expect("SSE data");
            let event: serde_json::Value = serde_json::from_str(data).expect("SSE data should be JSON");
            assert_eq!(event["type"].as_str(), Some(event_name));
            event
        })
        .collect()
}

fn gated_sse_chunks() -> (String, String) {
    let created = serde_json::json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": {"id": "resp_upstream_stale_sse", "status": "in_progress"}
    });
    let added = serde_json::json!({
        "type": "response.output_item.added",
        "sequence_number": 1,
        "output_index": 0,
        "item": {"id": "msg_upstream_stale_sse", "type": "message"}
    });
    let delta = serde_json::json!({
        "type": "response.output_text.delta",
        "sequence_number": 2,
        "item_id": "msg_upstream_stale_sse",
        "output_index": 0,
        "content_index": 0,
        "delta": "partial"
    });
    let completed = serde_json::json!({
        "type": "response.completed",
        "sequence_number": 3,
        "response": {"id": "resp_upstream_stale_sse", "status": "completed", "usage": null}
    });
    (
        format!("data: {created}\n\ndata: {added}\n\ndata: {delta}\n\n"),
        format!("data: {completed}\n\ndata: [DONE]\n\n"),
    )
}

async fn assert_only_competing_turn_persisted(pool: &Arc<DbPool>, conversation_id: &str) {
    let conversation_store = ConversationStore::new(Arc::clone(pool));
    assert_eq!(
        conversation_store
            .rehydrate(conversation_id)
            .await
            .expect("conversation history"),
        competing_turn_items()
    );
    let response = ResponseStore::new(Arc::clone(pool))
        .get(COMPETING_RESPONSE_ID)
        .await
        .expect("competing response should remain");
    assert_eq!(response.conversation_id.as_deref(), Some(conversation_id));
    let response_count = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM responses")
        .fetch_one(pool.as_ref())
        .await
        .expect("response count");
    assert_eq!(response_count, 1, "only the competing response should be stored");
}

async fn assert_response_not_persisted(pool: &Arc<DbPool>, response_id: &str) {
    let error = ResponseStore::new(Arc::clone(pool))
        .get(response_id)
        .await
        .expect_err("rejected response must not be persisted");
    assert!(error.is_not_found(), "expected missing response, got {error}");
}

/// Spawn a mock vLLM that returns a minimal valid JSON response.
async fn spawn_mock_vllm_json() -> (String, tokio::task::JoinHandle<()>) {
    let app = Router::new().route(
        "/v1/responses",
        post(|| async {
            axum::response::Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .body(axum::body::Body::from(
                    r#"{"id":"mock_id","object":"response","status":"completed",
                        "model":"test","output":[],"created_at":0}"#,
                ))
                .unwrap()
                .into_response()
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), handle)
}

async fn spawn_mock_vllm_json_capture() -> (String, Arc<Mutex<Vec<serde_json::Value>>>, tokio::task::JoinHandle<()>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let route_requests = Arc::clone(&requests);
    let app = Router::new().route(
        "/v1/responses",
        post(move |body: Bytes| {
            let route_requests = Arc::clone(&route_requests);
            async move {
                let body = serde_json::from_slice::<serde_json::Value>(&body).unwrap();
                route_requests.lock().await.push(body);
                axum::response::Response::builder()
                    .status(200)
                    .header("Content-Type", "application/json")
                    .body(axum::body::Body::from(
                        r#"{"id":"mock_id","object":"response","status":"completed",
                            "model":"test","output":[],"created_at":0}"#,
                    ))
                    .unwrap()
                    .into_response()
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), requests, handle)
}

async fn spawn_mock_vllm_json_capture_body() -> (String, Arc<Mutex<Vec<Bytes>>>, tokio::task::JoinHandle<()>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let route_requests = Arc::clone(&requests);
    let app = Router::new().route(
        "/v1/responses",
        post(move |body: Bytes| {
            let route_requests = Arc::clone(&route_requests);
            async move {
                route_requests.lock().await.push(body);
                axum::response::Response::builder()
                    .status(200)
                    .header("Content-Type", "application/json")
                    .body(axum::body::Body::from(
                        r#"{"id":"mock_id","object":"response","status":"completed",
                            "model":"test","output":[],"created_at":0}"#,
                    ))
                    .unwrap()
                    .into_response()
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), requests, handle)
}

/// Spawn a mock vLLM that returns an SSE stream.
async fn spawn_mock_vllm_sse() -> (String, tokio::task::JoinHandle<()>) {
    let app = Router::new().route(
        "/v1/responses",
        post(|| async {
            axum::response::Response::builder()
                .status(200)
                .header("Content-Type", "text/event-stream; charset=utf-8")
                .body(axum::body::Body::from(
                    "data: {\"type\":\"response.done\"}\n\ndata: [DONE]\n\n",
                ))
                .unwrap()
                .into_response()
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), handle)
}

#[tokio::test]
async fn test_store_false_proxies_json_to_vllm() {
    // Arrange
    let (llm_url, _h1) = spawn_mock_vllm_json().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .json(&serde_json::json!({"model":"test","input":[{"type":"message","role":"user","content":"hi"}],"store":false,"stream":false}))
        .send()
        .await
        .unwrap();

    // Assert — proxy forwards vLLM response verbatim; mock_id is not resp_-prefixed
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["id"], "mock_id");
}

#[tokio::test]
async fn test_store_false_proxies_unknown_text_format_verbatim() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture_body().await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;
    let request_body = r#"{"model":"test","input":"hi","store":false,"stream":false,"text":{"format":{"type":"provider_format","provider_option":true}}}"#;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .header("Content-Type", "application/json")
        .body(request_body)
        .send()
        .await
        .expect("stateless response request");

    assert_eq!(response.status(), StatusCode::OK);
    let requests = requests.lock().await;
    assert_eq!(requests.as_slice(), [Bytes::from_static(request_body.as_bytes())]);
}

#[tokio::test]
async fn test_duplicate_routing_field_is_rejected_before_upstream() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture_body().await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;
    let request_body = r#"{"model":"test","input":"hi","store":true,"store":false}"#;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .header("Content-Type", "application/json")
        .body(request_body)
        .send()
        .await
        .expect("response request with a duplicate routing field");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        requests.lock().await.is_empty(),
        "invalid request must not reach upstream"
    );
}

#[tokio::test]
async fn test_invalid_json_is_rejected_before_upstream() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture_body().await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .header("Content-Type", "application/json")
        .body(r#"{"model":"test","store":false"#)
        .send()
        .await
        .expect("syntactically invalid response request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        requests.lock().await.is_empty(),
        "invalid request must not reach upstream"
    );
}

#[tokio::test]
async fn test_store_false_with_web_search_reaches_executor() {
    // Arrange
    let (llm_url, requests, _h1) = spawn_mock_vllm_json_capture().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "tools": [{"type": "web_search_preview"}],
            "store": false,
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    // Assert — gateway tools need executor normalization even when persistence is disabled.
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["id"].as_str().unwrap_or("").starts_with("resp_"));
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["tools"][0]["type"], "function");
    assert_eq!(requests[0]["tools"][0]["name"], "web_search");
}

#[tokio::test]
async fn test_stateful_request_forwards_reasoning_configuration() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture().await;
    let fixture = storage_backed_state(&llm_url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let reasoning = serde_json::json!({
        "context": "all_turns",
        "effort": "high",
        "generate_summary": "concise",
        "mode": "pro",
        "summary": "detailed"
    });

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": "hi",
            "reasoning": reasoning,
            "store": true,
            "stream": false
        }))
        .send()
        .await
        .expect("stateful response request");

    assert_eq!(response.status(), StatusCode::OK);
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["reasoning"], reasoning);
}

#[tokio::test]
async fn test_stateful_request_forwards_text_configuration() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture().await;
    let fixture = storage_backed_state(&llm_url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let text = serde_json::json!({
        "format": {
            "type": "json_schema",
            "name": "weather",
            "schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": false
            },
            "strict": true,
            "x-format-extension": "kept"
        },
        "verbosity": "low",
        "x-text-extension": {"enabled": true}
    });

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": "hi",
            "text": text,
            "store": true,
            "stream": false
        }))
        .send()
        .await
        .expect("stateful response request");

    assert_eq!(response.status(), StatusCode::OK);
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["text"], text);
}

#[tokio::test]
async fn test_stateful_request_preserves_json_schema_property_order() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture_body().await;
    let fixture = storage_backed_state(&llm_url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let request_body = r#"{"model":"test","input":"hi","store":true,"stream":false,"text":{"format":{"type":"json_schema","name":"ordered","schema":{"type":"object","properties":{"outer_z":{"type":"object","properties":{"inner_z":{"type":"string"},"inner_a":{"type":"string"}}},"outer_a":{"type":"string"}},"required":["outer_z","outer_a"],"additionalProperties":false},"strict":true}}}"#;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .header("Content-Type", "application/json")
        .body(request_body)
        .send()
        .await
        .expect("stateful response request");

    assert_eq!(response.status(), StatusCode::OK);
    let requests = requests.lock().await;
    let upstream = std::str::from_utf8(&requests[0]).expect("upstream request should be UTF-8 JSON");
    let outer_z = upstream
        .find("\"outer_z\"")
        .expect("outer_z property should be forwarded");
    let outer_a = upstream
        .find("\"outer_a\"")
        .expect("outer_a property should be forwarded");
    let inner_z = upstream
        .find("\"inner_z\"")
        .expect("inner_z property should be forwarded");
    let inner_a = upstream
        .find("\"inner_a\"")
        .expect("inner_a property should be forwarded");
    assert!(outer_z < outer_a, "outer schema property order changed: {upstream}");
    assert!(inner_z < inner_a, "nested schema property order changed: {upstream}");
}

#[tokio::test]
async fn test_executor_route_rejects_malformed_text_configuration_before_upstream() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture().await;
    let fixture = storage_backed_state(&llm_url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": "hi",
            "text": {"format": {"type": "json_schema", "name": "missing_schema"}},
            "tools": [{"type": "web_search_preview"}],
            "store": false,
            "stream": false
        }))
        .send()
        .await
        .expect("malformed executor-bound response request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        requests.lock().await.is_empty(),
        "invalid request must not reach upstream"
    );
}

#[tokio::test]
async fn test_gateway_normalization_preserves_parallel_tool_calls() {
    // Arrange
    let (llm_url, requests, _h1) = spawn_mock_vllm_json_capture().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "tools": [{"type": "web_search_preview"}],
            "parallel_tool_calls": false,
            "store": false,
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    // Assert
    assert_eq!(resp.status(), 200);
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["parallel_tool_calls"], false);
}

#[tokio::test]
async fn test_gateway_normalization_allows_parallel_tool_calls_true() {
    // Arrange
    let (llm_url, requests, _h1) = spawn_mock_vllm_json_capture().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "tools": [{"type": "web_search_preview"}],
            "parallel_tool_calls": true,
            "store": false,
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    // Assert
    assert_eq!(resp.status(), 200);
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["parallel_tool_calls"], true);
}

#[tokio::test]
async fn test_store_false_proxies_large_json_body_to_vllm() {
    // Arrange
    let (llm_url, requests, _h1) = spawn_mock_vllm_json_capture().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;
    let prompt = "x".repeat(100 * 1024);

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": [{"type": "message", "role": "user", "content": prompt}],
            "store": false,
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    // Assert — the gateway keeps this below-limit request on the proxy path.
    assert_eq!(resp.status(), 200);
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["store"], false);
    assert_eq!(requests[0]["stream"], false);
    assert_eq!(requests[0]["input"][0]["content"].as_str().unwrap().len(), 100 * 1024);
}

#[tokio::test]
async fn test_store_false_proxies_sse_to_vllm() {
    // Arrange
    let (llm_url, _h1) = spawn_mock_vllm_sse().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .json(&serde_json::json!({"model":"test","input":[{"type":"message","role":"user","content":"hi"}],"store":false,"stream":true}))
        .send()
        .await
        .unwrap();

    // Assert — SSE content-type forwarded from mock vLLM
    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()["content-type"]
            .to_str()
            .unwrap()
            .contains("event-stream")
    );
}

#[tokio::test]
async fn test_store_true_hides_internal_persistence_error_details() {
    // Arrange — mock vLLM returns 200, but the executor cannot persist into the disabled test store.
    let (llm_url, _h1) = spawn_mock_vllm_json().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .json(&serde_json::json!({"model":"test","input":[{"type":"message","role":"user","content":"hi"}],"store":true,"stream":false}))
        .send()
        .await
        .unwrap();

    // Assert — a stored request never reports success without durable state.
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = resp.text().await.unwrap();
    assert!(body.contains("failed to persist response"), "{body}");
    assert!(!body.contains("storage not configured or disabled"), "{body}");
}

#[tokio::test]
async fn test_streaming_store_true_hides_persistence_details_without_sequence_gap() {
    let (llm_url, _h1) = spawn_mock_vllm_sse().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "store": true,
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.text().await.unwrap();
    assert!(body.contains("\"type\":\"error\""), "{body}");
    assert!(body.contains("failed to persist response"), "{body}");
    assert!(body.contains("\"status\":500"), "{body}");
    assert!(body.contains("\"type\":\"server_error\""), "{body}");
    assert!(body.contains("\"code\":\"server_error\""), "{body}");
    assert!(!body.contains("storage not configured or disabled"), "{body}");
    assert!(body.contains("\"sequence_number\":0"), "{body}");
    assert!(body.contains("data: [DONE]"), "{body}");
    assert!(!body.contains("\"type\":\"response.completed\""), "{body}");
}

#[tokio::test]
async fn http_json_conversation_conflict_rejects_stale_turn_without_persisting_it() {
    // Arrange
    let upstream_body = serde_json::json!({
        "id": "resp_upstream_stale_json",
        "object": "response",
        "status": "completed",
        "model": "test-model",
        "output": [],
        "created_at": 0
    })
    .to_string();
    let (mock, arrived, release) = MockResponsesServer::start_gated_json(upstream_body).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let client = reqwest::Client::new();
    let conversation_id = create_conversation(&client, &gateway_url).await;

    // Act
    let response_task = {
        let client = client.clone();
        let gateway_url = gateway_url.clone();
        let conversation_id = conversation_id.clone();
        tokio::spawn(async move {
            client
                .post(format!("{gateway_url}/v1/responses"))
                .json(&serde_json::json!({
                    "model": "test-model",
                    "input": [{"type": "message", "role": "user", "content": "stale turn"}],
                    "conversation_id": conversation_id,
                    "store": true,
                    "stream": false
                }))
                .send()
                .await
                .expect("response request")
        })
    };
    arrived.await.expect("upstream request should arrive after rehydration");
    persist_competing_turn(&fixture.pool, &conversation_id).await;
    release.send(()).expect("release gated JSON response");
    let response = response_task.await.expect("response task");

    // Assert
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response.json::<serde_json::Value>().await.expect("error response JSON"),
        serde_json::json!({"error": conflict_error()})
    );
    assert_only_competing_turn_persisted(&fixture.pool, &conversation_id).await;
}

#[tokio::test]
async fn http_sse_conversation_conflict_terminates_after_observable_delta_without_persisting_stale_turn() {
    // Arrange
    let (first_chunk, terminal_chunk) = gated_sse_chunks();
    let (mock, arrived, release) = MockResponsesServer::start_gated_sse(first_chunk, terminal_chunk).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let client = reqwest::Client::new();
    let conversation_id = create_conversation(&client, &gateway_url).await;

    // Act
    let mut response = client
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "stale turn"}],
            "conversation_id": conversation_id,
            "store": true,
            "stream": true
        }))
        .send()
        .await
        .expect("streaming response request");
    assert_eq!(response.status(), StatusCode::OK);
    arrived.await.expect("upstream request should arrive after rehydration");

    let mut body = String::new();
    while !body.contains("\"type\":\"response.output_text.delta\"") {
        let chunk = response
            .chunk()
            .await
            .expect("stream chunk")
            .expect("stream should contain a delta before completion");
        body.push_str(std::str::from_utf8(&chunk).expect("SSE should be UTF-8"));
    }
    assert!(body.contains("\"delta\":\"partial\""), "{body}");

    persist_competing_turn(&fixture.pool, &conversation_id).await;
    release.send(()).expect("release gated SSE response");
    while let Some(chunk) = response.chunk().await.expect("stream chunk") {
        body.push_str(std::str::from_utf8(&chunk).expect("SSE should be UTF-8"));
    }

    // Assert
    let events = sse_events(&body);
    let stale_response_id = events
        .iter()
        .find(|event| event["type"] == "response.created")
        .and_then(|event| event["response"]["id"].as_str())
        .expect("gateway response ID from response.created");
    assert!(
        events
            .iter()
            .any(|event| event["type"] == "response.output_text.delta" && event["delta"] == "partial"),
        "{body}"
    );
    let errors = events
        .iter()
        .filter(|event| event["type"] == "error")
        .collect::<Vec<_>>();
    assert_eq!(errors.len(), 1, "{body}");
    assert_eq!(errors[0]["status"], StatusCode::BAD_REQUEST.as_u16());
    assert_eq!(errors[0]["error"], conflict_error());
    assert_eq!(events.last().expect("terminal SSE event")["type"], "error");
    assert!(body.contains("data: [DONE]"), "{body}");
    assert!(
        events.iter().all(|event| event["type"] != "response.completed"),
        "{body}"
    );
    assert_only_competing_turn_persisted(&fixture.pool, &conversation_id).await;
    assert_response_not_persisted(&fixture.pool, stale_response_id).await;
}

#[tokio::test]
async fn test_oversized_body_returns_413() {
    // Arrange — LLM is never reached (gateway rejects the body first)
    let (llm_url, _h1) = spawn_mock_llm().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    // Act — 11 MB body
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .header("Content-Type", "application/json")
        .body("x".repeat(11 * 1024 * 1024))
        .send()
        .await
        .unwrap();

    // Assert
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

// --- Image preservation through the Responses HTTP transport (issue #253) ---
//
// A 1x1 red PNG and a 1x1 blue PNG, inline as data URLs. They are real, valid
// PNGs with distinguishable pixels, so an ordering assertion cannot pass by
// accident, and no binary fixture has to ship with the tests. The gateway never
// decodes them: decoding and preprocessing stay in vLLM.
const RED_PIXEL_PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP4z8AAAAMBAQD3A0FDAAAAAElFTkSuQmCC";
const BLUE_PIXEL_PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mNgYPgPAAEDAQA2dBFAAAAAAElFTkSuQmCC";

fn image_part(image_url: &str, detail: Option<&str>) -> serde_json::Value {
    match detail {
        Some(detail) => serde_json::json!({"type": "input_image", "image_url": image_url, "detail": detail}),
        None => serde_json::json!({"type": "input_image", "image_url": image_url}),
    }
}

/// Spawn a mock vLLM that captures every request body and answers each one with a
/// distinct completed assistant message, so a stored turn rehydrates real history.
async fn spawn_mock_vllm_json_capture_answers()
-> (String, Arc<Mutex<Vec<serde_json::Value>>>, tokio::task::JoinHandle<()>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let route_requests = Arc::clone(&requests);
    let turn = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let app = Router::new().route(
        "/v1/responses",
        post(move |body: Bytes| {
            let route_requests = Arc::clone(&route_requests);
            let turn = Arc::clone(&turn);
            async move {
                let body = serde_json::from_slice::<serde_json::Value>(&body).unwrap();
                route_requests.lock().await.push(body);
                let index = turn.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                let payload = serde_json::json!({
                    "id": format!("resp_upstream_{index}"),
                    "object": "response",
                    "status": "completed",
                    "model": "test",
                    "created_at": 0,
                    "output": [{
                        "id": format!("msg_upstream_{index}"),
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": format!("ANSWER {index}")}]
                    }]
                });
                axum::response::Response::builder()
                    .status(200)
                    .header("Content-Type", "application/json")
                    .body(axum::body::Body::from(payload.to_string()))
                    .unwrap()
                    .into_response()
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), requests, handle)
}

async fn post_response(gateway_url: &str, body: &serde_json::Value) -> serde_json::Value {
    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .json(body)
        .send()
        .await
        .expect("response request");
    assert_eq!(response.status(), StatusCode::OK);
    response.json().await.expect("response JSON")
}

#[tokio::test]
async fn test_http_preserves_mixed_text_and_image_ordering() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture().await;
    let fixture = storage_backed_state(&llm_url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let content = serde_json::json!([
        {"type": "input_text", "text": "first"},
        image_part(RED_PIXEL_PNG, Some("low")),
        {"type": "input_text", "text": "between"},
        image_part(BLUE_PIXEL_PNG, Some("high")),
        {"type": "input_text", "text": "last"}
    ]);

    post_response(
        &gateway_url,
        &serde_json::json!({
            "model": "test",
            "input": [{"type": "message", "role": "user", "content": content}],
            "store": true,
            "stream": false
        }),
    )
    .await;

    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0]["input"][0]["content"], content,
        "mixed text and image parts must reach vLLM in the order the client sent them"
    );
}

#[tokio::test]
async fn test_http_preserves_multiple_images_across_messages() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture().await;
    let fixture = storage_backed_state(&llm_url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let input = serde_json::json!([
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "look at this"}, image_part(RED_PIXEL_PNG, None)]
        },
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "I see red."}]},
        {
            "type": "message",
            "role": "user",
            "content": [image_part(BLUE_PIXEL_PNG, None), {"type": "input_text", "text": "and this?"}]
        }
    ]);

    post_response(
        &gateway_url,
        &serde_json::json!({"model": "test", "input": input, "store": true, "stream": false}),
    )
    .await;

    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["input"], input);
    let images = requests[0]["input"]
        .as_array()
        .expect("input items")
        .iter()
        .filter_map(|item| item["content"].as_array())
        .flatten()
        .filter(|part| part["type"] == "input_image")
        .map(|part| part["image_url"].as_str().expect("image URL"))
        .collect::<Vec<_>>();
    assert_eq!(
        images,
        vec![RED_PIXEL_PNG, BLUE_PIXEL_PNG],
        "each turn's image must survive, in turn order"
    );
}

#[tokio::test]
async fn test_http_client_view_image_tool_output_reaches_next_round() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture().await;
    let fixture = storage_backed_state(&llm_url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let tool_output = serde_json::json!([
        {"type": "input_text", "text": "attached local image path: diagram.png"},
        image_part(RED_PIXEL_PNG, None)
    ]);

    post_response(
        &gateway_url,
        &serde_json::json!({
            "model": "test",
            "tools": [{
                "type": "function",
                "name": "view_image",
                "description": "Attach a local image to the conversation.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
            }],
            "input": [
                {"type": "message", "role": "user", "content": "look at diagram.png"},
                {
                    "type": "function_call",
                    "call_id": "call_view_image_1",
                    "name": "view_image",
                    "arguments": "{\"path\":\"diagram.png\"}"
                },
                {"type": "function_call_output", "call_id": "call_view_image_1", "output": tool_output}
            ],
            "store": true,
            "stream": false
        }),
    )
    .await;

    let requests = requests.lock().await;
    let output = requests[0]["input"]
        .as_array()
        .expect("input items")
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .map(|item| &item["output"])
        .expect("client tool output should reach the next inference round");
    assert!(
        output.is_array(),
        "structured tool output must stay an array, not an escaped JSON string: {output}"
    );
    assert_eq!(output, &tool_output);
}

#[tokio::test]
async fn test_http_custom_tool_image_output_normalizes_without_stringifying() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture().await;
    let fixture = storage_backed_state(&llm_url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let tool_output = serde_json::json!([
        {"type": "input_text", "text": "screenshot"},
        image_part(BLUE_PIXEL_PNG, Some("auto"))
    ]);

    post_response(
        &gateway_url,
        &serde_json::json!({
            "model": "test",
            "tools": [{"type": "custom", "name": "grab_screenshot", "description": "Capture the screen."}],
            "input": [
                {
                    "type": "custom_tool_call",
                    "id": "ctc_1",
                    "call_id": "call_custom_1",
                    "name": "grab_screenshot",
                    "input": "screen",
                    "status": "completed"
                },
                {"type": "custom_tool_call_output", "call_id": "call_custom_1", "output": tool_output}
            ],
            "store": true,
            "stream": false
        }),
    )
    .await;

    let requests = requests.lock().await;
    let output = requests[0]["input"]
        .as_array()
        .expect("input items")
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .map(|item| &item["output"])
        .expect("a custom-tool output must normalize to a function-tool output");
    assert!(
        output.is_array(),
        "normalization must not stringify the array: {output}"
    );
    assert_eq!(output, &tool_output);
}

#[tokio::test]
async fn test_http_previous_response_id_continuation_preserves_images() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture_answers().await;
    let fixture = storage_backed_state(&llm_url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let content = serde_json::json!([
        {"type": "input_text", "text": "describe this"},
        image_part(RED_PIXEL_PNG, Some("low"))
    ]);

    let first = post_response(
        &gateway_url,
        &serde_json::json!({
            "model": "test",
            "input": [{"type": "message", "role": "user", "content": content}],
            "store": true,
            "stream": false
        }),
    )
    .await;
    let previous_response_id = first["id"].as_str().expect("stored response ID");

    post_response(
        &gateway_url,
        &serde_json::json!({
            "model": "test",
            "previous_response_id": previous_response_id,
            "input": [{"type": "message", "role": "user", "content": "and now?"}],
            "store": true,
            "stream": false
        }),
    )
    .await;

    let requests = requests.lock().await;
    assert_eq!(requests.len(), 2);
    let history = requests[1]["input"].as_array().expect("rehydrated history");
    assert_eq!(
        history[0]["content"], content,
        "the stored image must survive the round trip through the response store"
    );
    assert_eq!(history[1]["role"], "assistant");
    assert_eq!(history[2]["content"], "and now?");
}

#[tokio::test]
async fn test_http_conversation_rehydration_preserves_images() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture_answers().await;
    let fixture = storage_backed_state(&llm_url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let client = reqwest::Client::new();
    let conversation_id = create_conversation(&client, &gateway_url).await;
    let content = serde_json::json!([
        {"type": "input_text", "text": "conversation image"},
        image_part(BLUE_PIXEL_PNG, None)
    ]);

    post_response(
        &gateway_url,
        &serde_json::json!({
            "model": "test",
            "conversation_id": conversation_id,
            "input": [{"type": "message", "role": "user", "content": content}],
            "store": true,
            "stream": false
        }),
    )
    .await;
    post_response(
        &gateway_url,
        &serde_json::json!({
            "model": "test",
            "conversation_id": conversation_id,
            "input": [{"type": "message", "role": "user", "content": "follow up"}],
            "store": true,
            "stream": false
        }),
    )
    .await;

    let requests = requests.lock().await;
    assert_eq!(requests.len(), 2);
    let history = requests[1]["input"].as_array().expect("rehydrated history");
    assert_eq!(history[0]["content"], content);
    assert_eq!(history.last().expect("newest turn")["content"], "follow up");
}

#[tokio::test]
async fn test_http_stored_tool_image_output_survives_continuation() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture_answers().await;
    let fixture = storage_backed_state(&llm_url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let tool_output = serde_json::json!([
        {"type": "input_text", "text": "attached local image path: diagram.png"},
        image_part(RED_PIXEL_PNG, None)
    ]);

    let first = post_response(
        &gateway_url,
        &serde_json::json!({
            "model": "test",
            "tools": [{
                "type": "function",
                "name": "view_image",
                "description": "Attach a local image to the conversation.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
            }],
            "input": [
                {"type": "message", "role": "user", "content": "look at diagram.png"},
                {
                    "type": "function_call",
                    "call_id": "call_view_image_1",
                    "name": "view_image",
                    "arguments": "{\"path\":\"diagram.png\"}"
                },
                {"type": "function_call_output", "call_id": "call_view_image_1", "output": tool_output}
            ],
            "store": true,
            "stream": false
        }),
    )
    .await;

    post_response(
        &gateway_url,
        &serde_json::json!({
            "model": "test",
            "previous_response_id": first["id"].as_str().expect("stored response ID"),
            "input": [{"type": "message", "role": "user", "content": "describe it"}],
            "store": true,
            "stream": false
        }),
    )
    .await;

    let requests = requests.lock().await;
    let stored_output = requests[1]["input"]
        .as_array()
        .expect("rehydrated history")
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .map(|item| &item["output"])
        .expect("the stored tool output must rehydrate");
    assert!(stored_output.is_array(), "persistence must not stringify the array");
    assert_eq!(stored_output, &tool_output);
}

#[tokio::test]
async fn test_store_false_proxies_image_content_verbatim() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture_body().await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;
    // An unmodeled field on the image part proves the raw proxy forwards bytes
    // rather than reserializing through the gateway's typed input model.
    let request_body = format!(
        r#"{{"model":"test","store":false,"stream":false,"input":[{{"type":"message","role":"user","content":[{{"type":"input_image","image_url":"{RED_PIXEL_PNG}","detail":"low","x_future_field":"kept"}},{{"type":"input_text","text":"raw proxy"}}]}}]}}"#
    );

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .header("Content-Type", "application/json")
        .body(request_body.clone())
        .send()
        .await
        .expect("raw proxy request");

    assert_eq!(response.status(), StatusCode::OK);
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(
        std::str::from_utf8(&requests[0]).expect("upstream body should be UTF-8"),
        request_body,
        "a stateless request must reach vLLM byte for byte"
    );
}

#[tokio::test]
async fn test_http_text_only_model_still_forwards_images_unchanged() {
    // The control half of the text-only catalog check. Codex decides whether to
    // send image content by reading its local catalog, so with a text-only model
    // a missing image must be attributable to client-side stripping. The gateway
    // itself never strips: modality resolution shapes `/v1/models` only, and
    // `models_test.rs` covers that side.
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture().await;
    let fixture = storage_backed_state(&llm_url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let content = serde_json::json!([
        {"type": "input_text", "text": "text-only catalog"},
        image_part(RED_PIXEL_PNG, None)
    ]);

    post_response(
        &gateway_url,
        &serde_json::json!({
            "model": "text-only-model",
            "input": [{"type": "message", "role": "user", "content": content}],
            "store": true,
            "stream": false
        }),
    )
    .await;

    let requests = requests.lock().await;
    assert_eq!(requests[0]["model"], "text-only-model");
    assert_eq!(requests[0]["input"][0]["content"], content);
}

#[tokio::test]
async fn test_http_unknown_content_part_is_dropped_instead_of_forwarded() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture().await;
    let fixture = storage_backed_state(&llm_url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;

    post_response(
        &gateway_url,
        &serde_json::json!({
            "model": "test",
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "before"},
                {"type": "input_audio", "audio_url": "https://example.com/clip.wav"},
                image_part(RED_PIXEL_PNG, None)
            ]}],
            "store": true,
            "stream": false
        }),
    )
    .await;

    let requests = requests.lock().await;
    let parts = requests[0]["input"][0]["content"]
        .as_array()
        .expect("forwarded content parts");
    assert_eq!(
        parts.iter().map(|part| &part["type"]).collect::<Vec<_>>(),
        vec!["input_text", "input_image"],
        "an unrepresentable part must be dropped, never rewritten as a synthetic part"
    );
    assert_eq!(parts[1]["image_url"], RED_PIXEL_PNG);
}
