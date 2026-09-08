#[allow(dead_code)]
mod common;

use std::collections::VecDeque;
use std::convert::Infallible;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::Router;
use axum::body::Bytes;
use axum::http::{Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures::{SinkExt, StreamExt};
use http::StatusCode;
use serde_json::{Value, json};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, oneshot};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};
use tokio_util::sync::CancellationToken;

use agentic_core::executor::{ConversationHandler, ExecutionContext, RequestContext, ResponseHandler};
use agentic_core::proxy::ProxyState;
use agentic_core::storage::{
    ConversationStore, DbPool, InOutItem, ResponseMetadata, ResponseStore, create_pool_with_schema,
};
use agentic_core::tool::{WebSearchHandler, model_visible_namespace_member_name};
use agentic_core::types::RequestPayload;
use agentic_core::types::io::{InputItem, ResponsesInput};
use agentic_core::types::tools::ResponsesTool;
use agentic_server::app::{AppState, WebSocketTracker};

use common::{spawn_gateway, test_config};

struct MockResponsesServer {
    url: String,
    requests: Arc<Mutex<Vec<Value>>>,
    handle: tokio::task::JoinHandle<()>,
}

struct MockYouSearchServer {
    url: String,
    requests: Arc<Mutex<Vec<Value>>>,
    handle: tokio::task::JoinHandle<()>,
}

impl MockYouSearchServer {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let route_requests = Arc::clone(&requests);

        let app = Router::new().route(
            "/v1/search",
            get(move |uri: Uri| {
                let requests = Arc::clone(&route_requests);
                async move {
                    let body = query_params_as_json(&uri);
                    requests.lock().await.push(body);
                    axum::Json(json!({
                        "results": {
                            "web": [{
                                "title": "Rust async guide",
                                "url": "https://example.com/rust-async",
                                "snippet": "Async Rust reference"
                            }],
                            "news": []
                        },
                        "metadata": {"provider": "mock-you"}
                    }))
                    .into_response()
                }
            }),
        );

        let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            url: format!("http://{addr}"),
            requests,
            handle,
        }
    }

    async fn request_bodies(&self) -> Vec<Value> {
        self.requests.lock().await.clone()
    }
}

fn query_params_as_json(uri: &Uri) -> Value {
    let mut params = serde_json::Map::new();
    for (key, value) in url::form_urlencoded::parse(uri.query().unwrap_or_default().as_bytes()) {
        params.insert(key.into_owned(), Value::String(value.into_owned()));
    }
    Value::Object(params)
}

impl Drop for MockYouSearchServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

enum MockResponse {
    Static(String),
    Gated {
        response: String,
        arrived: oneshot::Sender<()>,
        release: oneshot::Receiver<()>,
    },
    ObservedGated {
        response: String,
        arrived: oneshot::Sender<Value>,
        release: oneshot::Receiver<()>,
    },
    ObservedPhased {
        first_chunk: String,
        remaining_chunks: String,
        arrived: oneshot::Sender<Value>,
        release: oneshot::Receiver<()>,
    },
    Hanging {
        first_chunk: String,
        drop_tx: oneshot::Sender<()>,
    },
}

struct HangingSse {
    first_chunk: Option<Bytes>,
    drop_tx: Option<oneshot::Sender<()>>,
}

impl HangingSse {
    fn new(first_chunk: String, drop_tx: oneshot::Sender<()>) -> Self {
        Self {
            first_chunk: Some(Bytes::from(first_chunk)),
            drop_tx: Some(drop_tx),
        }
    }
}

impl futures::Stream for HangingSse {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(first_chunk) = self.first_chunk.take() {
            Poll::Ready(Some(Ok(first_chunk)))
        } else {
            Poll::Pending
        }
    }
}

impl Drop for HangingSse {
    fn drop(&mut self) {
        if let Some(drop_tx) = self.drop_tx.take() {
            let _ = drop_tx.send(());
        }
    }
}

impl MockResponsesServer {
    async fn start(responses: Vec<String>) -> Self {
        Self::start_with_responses(responses.into_iter().map(MockResponse::Static).collect()).await
    }

    async fn start_hanging(first_chunk: String) -> (Self, oneshot::Receiver<()>) {
        let (drop_tx, drop_rx) = oneshot::channel();
        let server = Self::start_with_responses(vec![MockResponse::Hanging { first_chunk, drop_tx }]).await;
        (server, drop_rx)
    }

    async fn start_gated(response: String) -> (Self, oneshot::Receiver<()>, oneshot::Sender<()>) {
        let (arrived_tx, arrived_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let server = Self::start_with_responses(vec![MockResponse::Gated {
            response,
            arrived: arrived_tx,
            release: release_rx,
        }])
        .await;
        (server, arrived_rx, release_tx)
    }

    async fn start_with_responses(responses: Vec<MockResponse>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let route_queue = Arc::clone(&queue);
        let route_requests = Arc::clone(&requests);

        let app = Router::new().route(
            "/v1/responses",
            post(move |body: Bytes| {
                let queue = Arc::clone(&route_queue);
                let requests = Arc::clone(&route_requests);
                async move {
                    let body = serde_json::from_slice::<Value>(&body).expect("request body should be JSON");
                    requests.lock().await.push(body.clone());
                    let response = queue.lock().await.pop_front().expect("mock response queue exhausted");
                    let body = match response {
                        MockResponse::Static(response) => axum::body::Body::from(response),
                        MockResponse::Gated {
                            response,
                            arrived,
                            release,
                        } => {
                            let _ = arrived.send(());
                            let _ = release.await;
                            axum::body::Body::from(response)
                        }
                        MockResponse::ObservedGated {
                            response,
                            arrived,
                            release,
                        } => {
                            let _ = arrived.send(body);
                            let _ = release.await;
                            axum::body::Body::from(response)
                        }
                        MockResponse::ObservedPhased {
                            first_chunk,
                            remaining_chunks,
                            arrived,
                            release,
                        } => {
                            let _ = arrived.send(body);
                            let first =
                                futures::stream::once(async move { Ok::<_, Infallible>(Bytes::from(first_chunk)) });
                            let remaining = futures::stream::once(async move {
                                let _ = release.await;
                                Ok::<_, Infallible>(Bytes::from(remaining_chunks))
                            });
                            axum::body::Body::from_stream(first.chain(remaining))
                        }
                        MockResponse::Hanging { first_chunk, drop_tx } => {
                            axum::body::Body::from_stream(HangingSse::new(first_chunk, drop_tx))
                        }
                    };
                    Response::builder()
                        .status(StatusCode::OK)
                        .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
                        .body(body)
                        .unwrap()
                        .into_response()
                }
            }),
        );

        let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            url: format!("http://{addr}"),
            requests,
            handle,
        }
    }

    async fn request_bodies(&self) -> Vec<Value> {
        self.requests.lock().await.clone()
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
        let path = std::env::temp_dir().join(format!("agentic_ws_test_{}.db", uuid::Uuid::now_v7()));
        Self { path }
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
    storage_backed_state_with_web_search(llm_url, None).await
}

fn persistence_disabled_state(llm_url: &str) -> AppState {
    let config = test_config(llm_url);
    let client = Arc::new(reqwest::Client::new());
    let exec_ctx = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
        client,
        config.llm_api_base.clone(),
    ));
    let proxy_state = ProxyState::new(config.clone()).expect("proxy state");
    AppState {
        proxy_state,
        exec_ctx,
        llm_readiness_client: agentic_core::readiness::llm_readiness_client().expect("readiness client"),
        readiness_tracker: agentic_server::app::ReadinessTracker::default(),
        shutdown_token: CancellationToken::new(),
        websocket_tracker: WebSocketTracker::default(),
        llm_api_base: config.llm_api_base,
        skip_llm_ready_check: config.skip_llm_ready_check,
        openai_api_key: config.openai_api_key,
    }
}

async fn storage_backed_state_with_web_search(llm_url: &str, web_search_base_url: Option<&str>) -> StorageBackedState {
    let db = TestDb::new();
    let db_url = db.url();
    let pool = create_pool_with_schema(Some(&db_url)).await.unwrap();
    let config = test_config(llm_url);
    let client = Arc::new(reqwest::Client::new());
    let mut exec_ctx = ExecutionContext::new(
        ConversationHandler::new(ConversationStore::new(Arc::clone(&pool))),
        ResponseHandler::new(ResponseStore::new(Arc::clone(&pool))),
        Arc::clone(&client),
        config.llm_api_base.clone(),
    );
    if let Some(base_url) = web_search_base_url {
        exec_ctx = exec_ctx.with_gateway_executor(Arc::new(WebSearchHandler::with_api_key(
            client,
            "test-you-key".to_owned(),
            base_url,
        )));
    }
    let exec_ctx = Arc::new(exec_ctx);
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
    };
    StorageBackedState { state, pool, _db: db }
}

const COMPETING_RESPONSE_ID: &str = "resp_competing";
const CONFLICT_MESSAGE: &str = "conversation changed while the response was being generated; retry the request";

async fn create_conversation(gateway_url: &str) -> String {
    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/conversations"))
        .json(&json!({"store": true}))
        .send()
        .await
        .expect("conversation request");
    assert_eq!(response.status(), StatusCode::OK);
    response.json::<Value>().await.expect("conversation response JSON")["id"]
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

async fn assert_conflicting_websocket_turn_not_persisted(
    pool: &Arc<DbPool>,
    conversation_id: &str,
    stale_response_id: &str,
) {
    let conversation_store = ConversationStore::new(Arc::clone(pool));
    assert_eq!(
        conversation_store
            .rehydrate(conversation_id)
            .await
            .expect("conversation history"),
        competing_turn_items()
    );
    let response_store = ResponseStore::new(Arc::clone(pool));
    let competing = response_store
        .get(COMPETING_RESPONSE_ID)
        .await
        .expect("competing response should remain");
    assert_eq!(competing.conversation_id.as_deref(), Some(conversation_id));
    let error = response_store
        .get(stale_response_id)
        .await
        .expect_err("rejected response must not be persisted");
    assert!(error.is_not_found(), "expected missing response, got {error}");
    let item_count = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM items")
        .fetch_one(pool.as_ref())
        .await
        .expect("stored item count");
    assert_eq!(item_count, 1, "rejected turn must not leave orphaned items");
}

fn ws_url(gateway_url: &str) -> String {
    format!("{}/v1/responses", gateway_url.replacen("http://", "ws://", 1))
}

async fn connect_responses_ws(url: &str) -> WebSocketStream<MaybeTlsStream<TcpStream>> {
    let (ws, response) = connect_async(ws_url(url)).await.expect("websocket handshake");
    assert_eq!(response.status(), StatusCode::SWITCHING_PROTOCOLS);
    ws
}

async fn recv_json(ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>) -> Value {
    loop {
        let message = tokio::time::timeout(std::time::Duration::from_secs(5), ws.next())
            .await
            .expect("timed out waiting for websocket message")
            .expect("websocket should yield a message")
            .expect("websocket message should be ok");
        match message {
            Message::Text(text) => return serde_json::from_str(&text).expect("message should be JSON"),
            Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => {}
            Message::Close(frame) => panic!("websocket closed before JSON event: {frame:?}"),
            Message::Binary(_) => panic!("unexpected binary websocket message"),
        }
    }
}

async fn recv_until_completed(ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>) -> Vec<Value> {
    let mut events = Vec::new();
    loop {
        let event = recv_json(ws).await;
        let is_done = matches!(
            event.get("type").and_then(Value::as_str),
            Some("response.completed" | "response.failed" | "response.incomplete" | "error")
        );
        events.push(event);
        if is_done {
            return events;
        }
    }
}

async fn wait_for_request_count(mock: &MockResponsesServer, count: usize) {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        if mock.request_bodies().await.len() >= count {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for mock request"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}

async fn recv_close_or_end(ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>) {
    let message = tokio::time::timeout(std::time::Duration::from_secs(2), ws.next())
        .await
        .expect("timed out waiting for websocket close");
    match message {
        None | Some(Ok(Message::Close(_)) | Err(_)) => {}
        Some(Ok(message)) => panic!("expected websocket close, got {message:?}"),
    }
}

async fn recv_clean_close(ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>) {
    let message = tokio::time::timeout(std::time::Duration::from_secs(2), ws.next())
        .await
        .expect("timed out waiting for clean websocket close");
    match message {
        Some(Ok(Message::Close(_))) => ws.flush().await.expect("failed to acknowledge websocket close"),
        None => panic!("websocket ended without a close frame"),
        Some(Err(error)) => panic!("websocket close failed: {error}"),
        Some(Ok(message)) => panic!("expected websocket close, got {message:?}"),
    }
}

async fn send_ping_and_wait_for_pong(ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>, payload: Bytes) {
    ws.send(Message::Ping(payload.clone())).await.unwrap();
    loop {
        let message = tokio::time::timeout(std::time::Duration::from_secs(2), ws.next())
            .await
            .expect("timed out waiting for websocket pong")
            .expect("websocket should yield a message")
            .expect("websocket message should be ok");
        match message {
            Message::Pong(actual) => {
                assert_eq!(actual, payload);
                break;
            }
            Message::Ping(_) | Message::Frame(_) => {}
            Message::Text(text) => panic!("unexpected text before pong: {text}"),
            Message::Close(frame) => panic!("websocket closed before pong: {frame:?}"),
            Message::Binary(_) => panic!("unexpected binary websocket message"),
        }
    }
}

async fn send_json(ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>, value: Value) {
    ws.send(Message::Text(value.to_string().into())).await.unwrap();
}

fn sse_response(response_id: &str, message_id: &str, text: &str) -> String {
    let created = json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": {"id": response_id, "status": "in_progress"}
    });
    let added = json!({
        "type": "response.output_item.added",
        "sequence_number": 1,
        "output_index": 0,
        "item": {"id": message_id, "type": "message"}
    });
    let delta = json!({
        "type": "response.output_text.delta",
        "sequence_number": 2,
        "item_id": message_id,
        "output_index": 0,
        "content_index": 0,
        "delta": text
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 3,
        "response": {"id": response_id, "status": "completed", "usage": null}
    });
    format!("data: {created}\n\ndata: {added}\n\ndata: {delta}\n\ndata: {completed}\n\ndata: [DONE]\n\n")
}

fn phased_sse_response(response_id: &str, message_id: &str, text: &str) -> (String, String) {
    let response = sse_response(response_id, message_id, text);
    let split_at = response.find("\n\n").expect("SSE response has a first event") + 2;
    let (first_chunk, remaining_chunks) = response.split_at(split_at);
    (first_chunk.to_owned(), remaining_chunks.to_owned())
}

fn sse_failed_response() -> String {
    let created = json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": {"id": "resp_failed_upstream", "status": "in_progress"}
    });
    let failed = json!({
        "type": "response.failed",
        "sequence_number": 1,
        "response": {
            "id": "resp_failed_upstream",
            "status": "failed",
            "error": {
                "code": "tool_catalog_too_large",
                "message": "Too many tools"
            },
            "incomplete_details": {
                "reason": "upstream_error"
            },
            "usage": null
        }
    });
    format!("data: {created}\n\ndata: {failed}\n\ndata: [DONE]\n\n")
}

fn sse_function_call_response(response_id: &str, call_name: &str) -> String {
    let created = json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": {"id": response_id, "status": "in_progress"}
    });
    let added = json!({
        "type": "response.output_item.added",
        "sequence_number": 1,
        "output_index": 0,
        "item": {
            "id": "fc_upstream_1",
            "type": "function_call",
            "status": "in_progress",
            "name": call_name,
            "call_id": "call_1",
            "arguments": ""
        }
    });
    let done = json!({
        "type": "response.output_item.done",
        "sequence_number": 2,
        "output_index": 0,
        "item": {
            "id": "fc_upstream_1",
            "type": "function_call",
            "status": "completed",
            "name": call_name,
            "call_id": "call_1",
            "arguments": "{\"numbers\":[8,0]}"
        }
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 3,
        "response": {"id": response_id, "status": "completed", "usage": null}
    });
    format!("data: {created}\n\ndata: {added}\n\ndata: {done}\n\ndata: {completed}\n\ndata: [DONE]\n\n")
}

fn sse_custom_tool_call_response() -> String {
    let created = json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": {"id": "resp_custom", "status": "in_progress"}
    });
    let added = json!({
        "type": "response.output_item.added",
        "sequence_number": 1,
        "output_index": 0,
        "item": {
            "id": "fc_upstream_1",
            "type": "function_call",
            "status": "in_progress",
            "name": "apply_patch",
            "call_id": "call_custom_1",
            "arguments": ""
        }
    });
    let delta = json!({
        "type": "response.function_call_arguments.delta",
        "sequence_number": 2,
        "output_index": 0,
        "item_id": "fc_upstream_1",
        "delta": "{\"input\":\"*** Begin Patch\\n*** End Patch\"}"
    });
    let input_done = json!({
        "type": "response.function_call_arguments.done",
        "sequence_number": 3,
        "output_index": 0,
        "item_id": "fc_upstream_1",
        "name": "apply_patch",
        "arguments": "{\"input\":\"*** Begin Patch\\n*** End Patch\"}"
    });
    let item_done = json!({
        "type": "response.output_item.done",
        "sequence_number": 4,
        "output_index": 0,
        "item": {
            "id": "fc_upstream_1",
            "type": "function_call",
            "status": "completed",
            "name": "apply_patch",
            "call_id": "call_custom_1",
            "arguments": "{\"input\":\"*** Begin Patch\\n*** End Patch\"}"
        }
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 5,
        "response": {"id": "resp_custom", "status": "completed", "usage": null}
    });
    format!(
        "data: {created}\n\ndata: {added}\n\ndata: {delta}\n\ndata: {input_done}\n\ndata: {item_done}\n\ndata: {completed}\n\ndata: [DONE]\n\n"
    )
}

fn web_search_function_call_sse_response() -> String {
    let created = json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": {"id": "resp_tool_call", "status": "in_progress", "usage": null}
    });
    let added = json!({
        "type": "response.output_item.added",
        "sequence_number": 1,
        "output_index": 0,
        "item": {
            "id": "fc_search",
            "type": "function_call",
            "call_id": "call_search",
            "name": "web_search",
            "arguments": "",
            "status": "in_progress"
        }
    });
    let done = json!({
        "type": "response.function_call_arguments.done",
        "sequence_number": 2,
        "item_id": "fc_search",
        "output_index": 0,
        "call_id": "call_search",
        "name": "web_search",
        "arguments": "{\"query\":\"rust async\",\"count\":2}"
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 3,
        "response": {"id": "resp_tool_call", "status": "completed", "usage": null}
    });
    format!("data: {created}\n\ndata: {added}\n\ndata: {done}\n\ndata: {completed}\n\ndata: [DONE]\n\n")
}

#[tokio::test]
async fn websocket_generate_false_rejects_oversized_events_before_persistence() {
    assert_oversized_local_completion_rejected(false).await;
}

#[tokio::test]
async fn websocket_generate_false_checks_completed_event_before_persistence_or_delivery() {
    assert_oversized_local_completion_rejected(true).await;
}

async fn assert_oversized_local_completion_rejected(only_completion_oversized: bool) {
    let fixture = storage_backed_state("http://127.0.0.1:9").await;
    let (gateway_url, gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;
    let mut payload = json!({
        "type": "response.create", "model": "test-model", "input": [],
        "generate": false, "stream_id": "oversized-local", "instructions": ""
    });
    let instruction_bytes = if only_completion_oversized {
        send_json(&mut ws, payload.clone()).await;
        let baseline = recv_until_completed(&mut ws).await;
        assert_eq!(baseline.len(), 2);
        let created_bytes = baseline[0].to_string().len();
        assert!(baseline[1].to_string().len() > created_bytes);
        // Fill the created event exactly to the wire limit. The completed event
        // is larger because it includes usage, so both must be checked up front.
        1024 * 1024 - created_bytes
    } else {
        2 * 1024 * 1024
    };
    payload["instructions"] = json!("x".repeat(instruction_bytes));
    payload["input"] = json!("must not be stored");
    send_json(&mut ws, payload).await;
    let event = recv_json(&mut ws).await;
    assert_eq!(
        event["type"], "error",
        "no lifecycle event may be emitted for an oversized local response"
    );
    assert_eq!(event["stream_id"], "oversized-local");
    assert!(event["error"]["message"].as_str().unwrap().contains("exceeded"));
    assert!(event.to_string().len() <= 1024 * 1024);
    let responses = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM responses")
        .fetch_one(fixture.pool.as_ref())
        .await
        .unwrap();
    assert_eq!(
        responses,
        i64::from(only_completion_oversized),
        "rejected response must not be persisted"
    );
    let items = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM items")
        .fetch_one(fixture.pool.as_ref())
        .await
        .unwrap();
    assert_eq!(items, 0, "rejected response must not leave orphaned items");
    gateway.abort();
}

#[tokio::test]
async fn test_websocket_generate_false_prewarm_persists_context_without_inference() {
    let mock = MockResponsesServer::start(vec![sse_response("resp_upstream_1", "msg_upstream_1", "READY")]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "instructions": "Follow the warmup rules.",
            "input": [{"type": "message", "role": "user", "content": "warmup prefix"}],
            "tools": [{
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch."
            }],
            "generate": false,
            "store": false,
            "stream": true
        }),
    )
    .await;

    let prewarm = recv_until_completed(&mut ws).await;
    assert_eq!(
        prewarm
            .iter()
            .map(|event| event["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["response.created", "response.completed"]
    );
    let prewarm_response = &prewarm.last().unwrap()["response"];
    let prewarm_response_id = prewarm_response["id"].as_str().unwrap().to_owned();
    assert_eq!(prewarm_response["status"], "completed");
    assert_eq!(prewarm_response["output"], json!([]));
    assert!(mock.request_bodies().await.is_empty());

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "instructions": "Follow the warmup rules.",
            "previous_response_id": prewarm_response_id,
            "input": [{"type": "message", "role": "user", "content": "first turn"}],
            "store": false,
            "stream": true
        }),
    )
    .await;

    let turn = recv_until_completed(&mut ws).await;
    assert_eq!(
        turn.last().unwrap()["response"]["output"][0]["content"][0]["text"],
        "READY"
    );

    let requests = mock.request_bodies().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["instructions"], "Follow the warmup rules.");
    assert_eq!(requests[0]["input"][0]["content"], "warmup prefix");
    assert_eq!(requests[0]["input"][1]["content"], "first turn");
    assert_eq!(requests[0]["tools"][0]["type"], "function");
    assert_eq!(requests[0]["tools"][0]["name"], "apply_patch");
    assert_eq!(
        requests[0]["tools"][0]["parameters"]["properties"]["input"]["type"],
        "string"
    );
    assert!(requests[0].get("generate").is_none());
}

#[tokio::test]
async fn test_websocket_generate_false_prewarm_redacts_mcp_runtime_credentials() {
    let mock = MockResponsesServer::start(vec![]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [],
            "tools": [{
                "type": "mcp",
                "server_label": "counter",
                "server_url": "https://mcp.example.com/mcp",
                "headers": {"X-API-Key": "secret"},
                "authorization": "bearer-secret",
                "require_approval": "never"
            }],
            "generate": false,
            "store": false,
            "stream": true
        }),
    )
    .await;

    let prewarm = recv_until_completed(&mut ws).await;
    let response_id = prewarm.last().unwrap()["response"]["id"]
        .as_str()
        .expect("prewarm response id")
        .to_owned();
    assert!(mock.request_bodies().await.is_empty());

    let request = serde_json::from_value::<RequestPayload>(json!({
        "model": "test-model",
        "input": [],
        "previous_response_id": response_id
    }))
    .expect("lookup request");
    let lookup_ctx = RequestContext {
        original_request: request.clone(),
        enriched_request: request,
        new_input_items: vec![],
        response_id: "resp_lookup".to_owned(),
        conversation_id: None,
        conversation_version: None,
    };
    let stored = fixture
        .state
        .exec_ctx
        .resp_handler
        .get(&lookup_ctx)
        .await
        .expect("stored prewarm response");
    let tools = stored.metadata.effective_tools.expect("persisted tools");
    let ResponsesTool::Mcp(tool) = &tools[0] else {
        panic!("expected MCP tool");
    };

    assert!(tool.headers.is_none());
    assert!(tool.authorization.is_none());
}

#[tokio::test]
async fn test_websocket_first_turn_forwards_incremental_events_and_final_payload() {
    let mock = MockResponsesServer::start(vec![sse_response("resp_upstream_1", "msg_upstream_1", "HELLO")]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "reasoning": {"effort": "high"},
            "text": {"format": {"type": "json_object"}, "verbosity": "high"},
            "store": true,
            "stream": true
        }),
    )
    .await;

    let events = recv_until_completed(&mut ws).await;
    let event_types = events
        .iter()
        .map(|event| event["type"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        event_types,
        vec![
            "response.created",
            "response.output_item.added",
            "response.output_text.delta",
            "response.completed"
        ]
    );
    assert_ne!(events[0]["response"]["id"], "resp_upstream_1");
    assert_eq!(events[2]["delta"], "HELLO");
    let response = &events.last().unwrap()["response"];
    assert_ne!(response["id"], "resp_upstream_1");
    assert_eq!(response["status"], "completed");
    assert_eq!(response["output"][0]["content"][0]["text"], "HELLO");
    let requests = mock.request_bodies().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["stream"], true);
    assert_eq!(requests[0]["input"][0]["content"], "hi");
    assert_eq!(requests[0]["reasoning"], json!({"effort": "high"}));
    assert_eq!(
        requests[0]["text"],
        json!({"format": {"type": "json_object"}, "verbosity": "high"})
    );
    assert!(requests[0].get("type").is_none());
}

#[tokio::test]
async fn websocket_distinct_stream_ids_run_concurrently_and_tag_every_event() {
    // Arrange: hold both upstream responses so the second request can arrive
    // only if the WebSocket handler starts distinct lanes concurrently.
    let (first_arrived_tx, first_arrived_rx) = oneshot::channel();
    let (first_release_tx, first_release_rx) = oneshot::channel();
    let (second_arrived_tx, second_arrived_rx) = oneshot::channel();
    let (second_release_tx, second_release_rx) = oneshot::channel();
    let (first_chunk, first_remaining) = phased_sse_response("resp_upstream_first", "msg_upstream_first", "FIRST");
    let (second_chunk, second_remaining) = phased_sse_response("resp_upstream_second", "msg_upstream_second", "SECOND");
    let mock = MockResponsesServer::start_with_responses(vec![
        MockResponse::Static(sse_response("resp_upstream_parent", "msg_upstream_parent", "PARENT")),
        MockResponse::ObservedPhased {
            first_chunk,
            remaining_chunks: first_remaining,
            arrived: first_arrived_tx,
            release: first_release_rx,
        },
        MockResponse::ObservedPhased {
            first_chunk: second_chunk,
            remaining_chunks: second_remaining,
            arrived: second_arrived_tx,
            release: second_release_rx,
        },
    ])
    .await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": "remember the parent context",
            "store": true
        }),
    )
    .await;
    let parent = recv_until_completed(&mut ws).await;
    let parent_response_id = parent
        .last()
        .and_then(|event| event["response"]["id"].as_str())
        .expect("parent response id")
        .to_owned();

    // Act: fork the stored parent into two independent lanes.
    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "stream_id": "lane-a",
            "model": "test-model",
            "previous_response_id": &parent_response_id,
            "input": "request-a",
            "metadata": {"test_lane": "lane-a"}
        }),
    )
    .await;
    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "stream_id": "lane-b",
            "model": "test-model",
            "previous_response_id": &parent_response_id,
            "input": "request-b",
            "metadata": {"test_lane": "lane-b"}
        }),
    )
    .await;

    let first_request = tokio::time::timeout(std::time::Duration::from_secs(2), first_arrived_rx)
        .await
        .expect("first lane did not reach upstream")
        .expect("first arrival sender dropped");
    let second_request = tokio::time::timeout(std::time::Duration::from_secs(2), second_arrived_rx)
        .await
        .expect("distinct WebSocket lane was serialized behind the active lane")
        .expect("second arrival sender dropped");
    assert!(first_request.get("stream_id").is_none());
    assert!(second_request.get("stream_id").is_none());

    let first_stream_id = first_request["metadata"]["test_lane"]
        .as_str()
        .expect("first request test lane");
    let second_stream_id = second_request["metadata"]["test_lane"]
        .as_str()
        .expect("second request test lane");
    assert_ne!(first_stream_id, second_stream_id);

    // Both lanes publish an initial event before either completes. Releasing
    // the first lane then proves its remaining events can follow an event from
    // the still-active second lane: first, second, first.
    let created_events = [recv_json(&mut ws).await, recv_json(&mut ws).await];
    assert!(created_events.iter().all(|event| event["type"] == "response.created"));
    assert!(created_events.iter().any(|event| event["stream_id"] == first_stream_id));
    assert!(
        created_events
            .iter()
            .any(|event| event["stream_id"] == second_stream_id)
    );

    first_release_tx.send(()).expect("release first upstream response");
    let first_events = recv_until_completed(&mut ws).await;
    assert_eq!(first_events.len(), 3);
    assert!(first_events.iter().all(|event| event["stream_id"] == first_stream_id));

    second_release_tx.send(()).expect("release second upstream response");
    let second_events = recv_until_completed(&mut ws).await;
    assert_eq!(second_events.len(), 3);
    assert!(second_events.iter().all(|event| event["stream_id"] == second_stream_id));
}

#[tokio::test]
async fn websocket_requests_with_the_same_stream_id_run_fifo() {
    let (first_arrived_tx, first_arrived_rx) = oneshot::channel();
    let (first_release_tx, first_release_rx) = oneshot::channel();
    let (second_arrived_tx, mut second_arrived_rx) = oneshot::channel();
    let (second_release_tx, second_release_rx) = oneshot::channel();
    let mock = MockResponsesServer::start_with_responses(vec![
        MockResponse::ObservedGated {
            response: sse_response("resp_upstream_first", "msg_upstream_first", "FIRST"),
            arrived: first_arrived_tx,
            release: first_release_rx,
        },
        MockResponse::ObservedGated {
            response: sse_response("resp_upstream_second", "msg_upstream_second", "SECOND"),
            arrived: second_arrived_tx,
            release: second_release_rx,
        },
    ])
    .await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    for input in ["first", "second"] {
        send_json(
            &mut ws,
            json!({
                "type": "response.create",
                "stream_id": "ordered-lane",
                "model": "test-model",
                "input": input
            }),
        )
        .await;
    }

    tokio::time::timeout(std::time::Duration::from_secs(2), first_arrived_rx)
        .await
        .expect("timed out waiting for first request to reach upstream")
        .expect("first arrival sender dropped");
    send_ping_and_wait_for_pong(&mut ws, Bytes::from_static(b"same-lane-request-received")).await;
    assert_eq!(
        mock.request_bodies().await.len(),
        1,
        "second request on the same lane started before the first completed"
    );

    first_release_tx.send(()).expect("release first response");
    let first_events = recv_until_completed(&mut ws).await;
    assert!(first_events.iter().all(|event| event["stream_id"] == "ordered-lane"));

    tokio::time::timeout(std::time::Duration::from_secs(2), &mut second_arrived_rx)
        .await
        .expect("second request did not start after first completed")
        .expect("second arrival sender dropped");
    second_release_tx.send(()).expect("release second response");
    let second_events = recv_until_completed(&mut ws).await;
    assert!(second_events.iter().all(|event| event["stream_id"] == "ordered-lane"));
}

async fn assert_validation_error_waits_for_active_lane(stream_id: Option<&str>) {
    let (mock, arrived, release) =
        MockResponsesServer::start_gated(sse_response("resp_upstream_first", "msg_upstream_first", "FIRST")).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    let mut valid = json!({
        "type": "response.create",
        "model": "test-model",
        "input": "first"
    });
    let mut invalid = json!({
        "type": "response.create",
        "input": "missing model"
    });
    if let Some(stream_id) = stream_id {
        valid["stream_id"] = json!(stream_id);
        invalid["stream_id"] = json!(stream_id);
    }

    send_json(&mut ws, valid).await;
    tokio::time::timeout(std::time::Duration::from_secs(2), arrived)
        .await
        .expect("timed out waiting for first request to reach upstream")
        .expect("first arrival sender dropped");
    send_json(&mut ws, invalid).await;
    send_ping_and_wait_for_pong(&mut ws, Bytes::from_static(b"validation-error-queued")).await;

    release.send(()).expect("release first response");
    let events = recv_until_completed(&mut ws).await;
    assert_eq!(events.last().expect("terminal event")["type"], "response.completed");

    let error = recv_json(&mut ws).await;
    assert_eq!(error["type"], "error");
    assert_eq!(error["status"], StatusCode::BAD_REQUEST.as_u16());
    if let Some(stream_id) = stream_id {
        assert_eq!(error["stream_id"], stream_id);
    } else {
        assert!(error.get("stream_id").is_none());
    }
}

#[tokio::test]
async fn websocket_validation_errors_wait_for_their_named_lane() {
    assert_validation_error_waits_for_active_lane(Some("ordered-lane")).await;
}

#[tokio::test]
async fn websocket_validation_errors_wait_for_the_default_lane() {
    assert_validation_error_waits_for_active_lane(None).await;
}

#[tokio::test]
async fn websocket_invalid_stream_ids_return_400_and_leave_connection_usable() {
    let mock = MockResponsesServer::start(vec![sse_response("resp_valid", "msg_valid", "done")]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    for stream_id in [json!(""), json!("x".repeat(257)), json!(42)] {
        send_json(
            &mut ws,
            json!({
                "type": "response.create",
                "stream_id": stream_id,
                "model": "test-model",
                "input": "invalid stream ID"
            }),
        )
        .await;
        let error = recv_json(&mut ws).await;
        assert_eq!(error["type"], "error");
        assert_eq!(error["status"], StatusCode::BAD_REQUEST.as_u16());
        assert!(error.get("stream_id").is_none());
        assert!(mock.request_bodies().await.is_empty());
    }

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "stream_id": "x",
            "model": "test-model",
            "input": "valid request"
        }),
    )
    .await;
    let events = recv_until_completed(&mut ws).await;
    assert!(events.iter().all(|event| event["stream_id"] == "x"));
    assert_eq!(mock.request_bodies().await.len(), 1);
}

#[tokio::test]
async fn websocket_rejects_more_than_64_outstanding_requests() {
    let first_chunk = format!(
        "data: {}\n\n",
        json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": {"id": "resp_upstream_hanging", "status": "in_progress"}
        })
    );
    let (mock, upstream_dropped) = MockResponsesServer::start_hanging(first_chunk).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    for index in 0..64 {
        send_json(
            &mut ws,
            json!({
                "type": "response.create",
                "stream_id": "capacity-lane",
                "model": "test-model",
                "input": format!("request {index}")
            }),
        )
        .await;
    }
    wait_for_request_count(&mock, 1).await;
    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "stream_id": "capacity-lane",
            "model": "test-model",
            "input": "request 65"
        }),
    )
    .await;

    let overload = loop {
        let event = recv_json(&mut ws).await;
        if event["type"] == "error" {
            break event;
        }
    };
    assert_eq!(overload["status"], StatusCode::TOO_MANY_REQUESTS.as_u16());
    assert_eq!(overload["error"]["code"], "rate_limit_exceeded");
    assert_eq!(overload["stream_id"], "capacity-lane");
    assert_eq!(mock.request_bodies().await.len(), 1);

    ws.close(None).await.expect("close websocket");
    tokio::time::timeout(std::time::Duration::from_secs(2), upstream_dropped)
        .await
        .expect("timed out waiting for upstream stream cancellation")
        .expect("upstream drop sender should notify");
}

#[tokio::test]
async fn websocket_rejects_a_65th_active_stream_lane() {
    let mut responses = Vec::new();
    let mut upstream_drops = Vec::new();
    for index in 0..64 {
        let first_chunk = format!(
            "data: {}\n\n",
            json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": {"id": format!("resp_active_{index}"), "status": "in_progress"}
            })
        );
        let (drop_tx, drop_rx) = oneshot::channel();
        responses.push(MockResponse::Hanging { first_chunk, drop_tx });
        upstream_drops.push(drop_rx);
    }
    let mock = MockResponsesServer::start_with_responses(responses).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    for index in 0..64 {
        send_json(
            &mut ws,
            json!({
                "type": "response.create",
                "stream_id": format!("active-{index}"),
                "model": "test-model",
                "input": format!("request {index}")
            }),
        )
        .await;
    }
    wait_for_request_count(&mock, 64).await;
    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "stream_id": "active-64",
            "model": "test-model",
            "input": "request 65"
        }),
    )
    .await;

    let overload = loop {
        let event = recv_json(&mut ws).await;
        if event["type"] == "error" {
            break event;
        }
    };
    assert_eq!(overload["status"], StatusCode::TOO_MANY_REQUESTS.as_u16());
    assert_eq!(overload["stream_id"], "active-64");
    assert_eq!(mock.request_bodies().await.len(), 64);

    ws.close(None).await.expect("close websocket");
    for upstream_dropped in upstream_drops {
        tokio::time::timeout(std::time::Duration::from_secs(2), upstream_dropped)
            .await
            .expect("timed out waiting for active upstream stream cancellation")
            .expect("upstream drop sender should notify");
    }
}

#[tokio::test]
async fn websocket_rejects_requests_over_the_aggregate_input_budget() {
    let (mock, arrived, release) =
        MockResponsesServer::start_gated(sse_response("resp_large", "msg_large", "done")).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "stream_id": "large-a",
            "model": "test-model",
            "input": "x".repeat(512 * 1024)
        }),
    )
    .await;
    tokio::time::timeout(std::time::Duration::from_secs(10), arrived)
        .await
        .expect("timed out waiting for large request to reach upstream")
        .expect("arrival sender dropped");

    for index in 1..23 {
        send_json(
            &mut ws,
            json!({
                "type": "response.create",
                "stream_id": "large-a",
                "model": "test-model",
                "input": "q".repeat(512 * 1024),
                "metadata": {"queued_index": index}
            }),
        )
        .await;
    }
    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "stream_id": "large-b",
            "model": "test-model",
            "input": "y".repeat(512 * 1024)
        }),
    )
    .await;
    let overload = recv_json(&mut ws).await;
    assert_eq!(overload["type"], "error");
    assert_eq!(overload["status"], StatusCode::TOO_MANY_REQUESTS.as_u16());
    assert_eq!(overload["error"]["code"], "rate_limit_exceeded");
    assert_eq!(overload["stream_id"], "large-b");
    assert_eq!(mock.request_bodies().await.len(), 1);

    release.send(()).expect("release large response");
    let events = recv_until_completed(&mut ws).await;
    assert!(events.iter().all(|event| event["stream_id"] == "large-a"));
}

#[tokio::test]
async fn websocket_reclaims_completed_request_capacity_before_admission() {
    let (first_arrived_tx, first_arrived_rx) = oneshot::channel();
    let (first_release_tx, first_release_rx) = oneshot::channel();
    let (second_drop_tx, second_drop_rx) = oneshot::channel();
    let second_chunk = format!(
        "data: {}\n\n",
        json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": {"id": "resp_capacity_second", "status": "in_progress"}
        })
    );
    let mock = MockResponsesServer::start_with_responses(vec![
        MockResponse::Gated {
            response: sse_response("resp_capacity_first", "msg_capacity_first", "first"),
            arrived: first_arrived_tx,
            release: first_release_rx,
        },
        MockResponse::Hanging {
            first_chunk: second_chunk,
            drop_tx: second_drop_tx,
        },
    ])
    .await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    for index in 0..64 {
        send_json(
            &mut ws,
            json!({
                "type": "response.create",
                "stream_id": "capacity-lane",
                "model": "test-model",
                "input": format!("request {index}")
            }),
        )
        .await;
    }
    tokio::time::timeout(std::time::Duration::from_secs(2), first_arrived_rx)
        .await
        .expect("timed out waiting for first request to reach upstream")
        .expect("first arrival sender dropped");
    first_release_tx.send(()).expect("release first response");
    let first_events = recv_until_completed(&mut ws).await;
    assert_eq!(
        first_events.last().expect("terminal event")["type"],
        "response.completed"
    );

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "stream_id": "capacity-lane",
            "model": "test-model",
            "input": "request after completed capacity is reclaimed"
        }),
    )
    .await;

    let second_created = recv_json(&mut ws).await;
    assert_eq!(second_created["type"], "response.created");
    assert_eq!(second_created["stream_id"], "capacity-lane");
    send_ping_and_wait_for_pong(&mut ws, Bytes::from_static(b"capacity-reclaimed")).await;

    ws.close(None).await.expect("close websocket");
    tokio::time::timeout(std::time::Duration::from_secs(2), second_drop_rx)
        .await
        .expect("timed out waiting for upstream stream cancellation")
        .expect("upstream drop sender should notify");
}

#[tokio::test]
async fn websocket_conversation_conflict_ends_request_without_persisting_stale_turn() {
    // Arrange
    let (mock, arrived, release) =
        MockResponsesServer::start_gated(sse_response("resp_upstream_stale_ws", "msg_upstream_stale_ws", "STALE"))
            .await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let conversation_id = create_conversation(&gateway_url).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    // Act
    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "stale turn"}],
            "conversation_id": conversation_id,
            "store": true,
            "stream": true
        }),
    )
    .await;
    arrived.await.expect("upstream request should arrive after rehydration");
    persist_competing_turn(&fixture.pool, &conversation_id).await;
    release.send(()).expect("release gated WebSocket response");
    let events = recv_until_completed(&mut ws).await;

    // Assert
    let stale_response_id = events
        .iter()
        .find(|event| event["type"] == "response.created")
        .and_then(|event| event["response"]["id"].as_str())
        .expect("gateway response ID from response.created")
        .to_owned();
    let error = events.last().expect("terminal conflict error");
    assert_eq!(error["type"], "error");
    assert_eq!(error["status"], StatusCode::BAD_REQUEST.as_u16());
    assert_eq!(
        error["error"],
        json!({
            "message": CONFLICT_MESSAGE,
            "type": "invalid_request_error",
            "code": "conversation_locked",
            "param": "conversation"
        })
    );
    assert!(events.iter().all(|event| event["type"] != "response.completed"));

    // A local response queued after the error is an ordering barrier: it can only
    // begin after the conflicted executor stream has ended.
    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [],
            "generate": false,
            "store": false,
            "stream": true
        }),
    )
    .await;
    let barrier_created = recv_json(&mut ws).await;
    assert_eq!(barrier_created["type"], "response.created");
    let barrier_response_id = barrier_created["response"]["id"]
        .as_str()
        .expect("barrier response ID")
        .to_owned();
    let barrier_completed = recv_json(&mut ws).await;
    assert_eq!(barrier_completed["type"], "response.completed");
    assert_eq!(barrier_completed["response"]["id"], barrier_response_id);
    assert_ne!(barrier_response_id, stale_response_id);

    assert_conflicting_websocket_turn_not_persisted(&fixture.pool, &conversation_id, &stale_response_id).await;
}

#[tokio::test]
async fn websocket_generate_false_conversation_conflict_rejects_stale_local_completion() {
    // Arrange
    let mock = MockResponsesServer::start(vec![]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (rehydrated, release) = fixture.state.websocket_tracker.install_local_completion_test_barrier();
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let conversation_id = create_conversation(&gateway_url).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    // Act: the one-shot barrier deterministically pauses local completion after
    // rehydration and before persistence, without relying on timing sleeps.
    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "stale local turn"}],
            "conversation_id": conversation_id,
            "generate": false,
            "store": true,
            "stream": true
        }),
    )
    .await;
    rehydrated
        .await
        .expect("local completion should pause after rehydration");
    persist_competing_turn(&fixture.pool, &conversation_id).await;
    release
        .send(())
        .expect("local completion should remain paused before persistence");
    let error = recv_json(&mut ws).await;

    // Assert
    assert_eq!(
        error,
        json!({
            "type": "error",
            "status": StatusCode::BAD_REQUEST.as_u16(),
            "error": {
                "message": CONFLICT_MESSAGE,
                "type": "invalid_request_error",
                "code": "conversation_locked",
                "param": "conversation"
            }
        })
    );

    // A local response queued after the error is an ordering barrier. Its first
    // event proves the stale turn emitted no response.completed after the error.
    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [],
            "generate": false,
            "store": false,
            "stream": true
        }),
    )
    .await;
    let barrier_created = recv_json(&mut ws).await;
    assert_eq!(barrier_created["type"], "response.created");
    let barrier_response_id = barrier_created["response"]["id"]
        .as_str()
        .expect("barrier response ID")
        .to_owned();
    let barrier_completed = recv_json(&mut ws).await;
    assert_eq!(barrier_completed["type"], "response.completed");
    assert_eq!(barrier_completed["response"]["id"], barrier_response_id);

    assert!(mock.request_bodies().await.is_empty());
    let conversation_store = ConversationStore::new(Arc::clone(&fixture.pool));
    assert_eq!(
        conversation_store
            .rehydrate(&conversation_id)
            .await
            .expect("conversation history"),
        competing_turn_items()
    );
    let response_ids = sqlx::query_scalar::<_, String>("SELECT id FROM responses WHERE id != $1 ORDER BY id")
        .bind(&barrier_response_id)
        .fetch_all(fixture.pool.as_ref())
        .await
        .expect("stored response IDs");
    assert_eq!(response_ids, vec![COMPETING_RESPONSE_ID]);
    let item_count = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM items")
        .fetch_one(fixture.pool.as_ref())
        .await
        .expect("stored item count");
    assert_eq!(item_count, 1, "rejected turn must not leave orphaned items");
}

#[tokio::test]
async fn test_websocket_streaming_persistence_error_uses_standard_envelope() {
    let mock = MockResponsesServer::start(vec![sse_response("resp_upstream_1", "msg_upstream_1", "HELLO")]).await;
    let (gateway_url, _gateway) = spawn_gateway(persistence_disabled_state(&mock.url)).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "store": true,
            "stream": true
        }),
    )
    .await;

    let events = recv_until_completed(&mut ws).await;
    let error = events.last().expect("persistence error event");
    assert_eq!(error["type"], "error");
    assert_eq!(error["status"], 500);
    assert_eq!(error["error"]["message"], "failed to persist response");
    assert_eq!(error["error"]["type"], "server_error");
    assert_eq!(error["error"]["code"], "server_error");
    assert!(events.iter().all(|event| event["type"] != "response.completed"));
}

#[tokio::test]
async fn test_websocket_preserves_upstream_failure_details() {
    let mock = MockResponsesServer::start(vec![sse_failed_response()]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": "fail",
            "store": true,
            "stream": true
        }),
    )
    .await;

    let events = recv_until_completed(&mut ws).await;
    let failed = events.last().unwrap();
    assert_eq!(failed["type"], "response.failed");
    assert_eq!(failed["response"]["status"], "error");
    assert_eq!(failed["response"]["error"]["code"], "tool_catalog_too_large");
    assert_eq!(failed["response"]["error"]["message"], "Too many tools");
    assert_eq!(failed["response"]["incomplete_details"]["reason"], "upstream_error");
}

#[tokio::test]
async fn test_websocket_generate_false_is_local_and_reusable() {
    let mock = MockResponsesServer::start(vec![sse_response("resp_upstream_1", "msg_upstream_1", "HELLO")]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "stream_id": "warmup-lane",
            "model": "test-model",
            "input": [],
            "generate": false,
            "store": false,
            "stream": true
        }),
    )
    .await;

    let warmup = recv_until_completed(&mut ws).await;
    assert_eq!(warmup.len(), 2);
    assert_eq!(warmup[0]["type"], "response.created");
    assert_eq!(warmup[1]["type"], "response.completed");
    assert_eq!(warmup[0]["response"]["id"], warmup[1]["response"]["id"]);
    assert_eq!(warmup[1]["response"]["output"], json!([]));
    assert_eq!(warmup[1]["response"]["usage"]["total_tokens"], 0);
    assert!(warmup.iter().all(|event| event["stream_id"] == "warmup-lane"));
    assert!(mock.request_bodies().await.is_empty());

    let warmup_id = warmup[1]["response"]["id"].as_str().unwrap();
    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "previous_response_id": warmup_id,
            "input": [{"type": "message", "role": "user", "content": "hello"}],
            "store": false,
            "stream": true
        }),
    )
    .await;

    let response = recv_until_completed(&mut ws).await;
    assert_eq!(response.last().unwrap()["response"]["previous_response_id"], warmup_id);
    assert!(response.iter().all(|event| event.get("stream_id").is_none()));
    let requests = mock.request_bodies().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["input"].as_array().unwrap().len(), 1);
    assert_eq!(requests[0]["input"][0]["role"], "user");
    assert_eq!(requests[0]["input"][0]["content"], "hello");
}

#[tokio::test]
async fn test_websocket_empty_input_without_generate_reaches_upstream() {
    let mock = MockResponsesServer::start(vec![sse_response("resp_upstream_1", "msg_upstream_1", "HELLO")]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [],
            "store": false,
            "stream": true
        }),
    )
    .await;

    let response = recv_until_completed(&mut ws).await;
    assert_eq!(
        response.last().unwrap()["response"]["output"][0]["content"][0]["text"],
        "HELLO"
    );
    let requests = mock.request_bodies().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["input"], json!([]));
}

#[tokio::test]
async fn test_websocket_restores_namespace_tool_call_events() {
    let mock = MockResponsesServer::start(vec![sse_function_call_response(
        "resp_upstream_1",
        "agentic_ns__mcp__agentic_fixture__add_numbers",
    )])
    .await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "use the tool"}],
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__agentic_fixture",
                    "tools": [
                        {
                            "type": "function",
                            "name": "add_numbers",
                            "parameters": {"type": "object"}
                        }
                    ]
                }
            ],
            "store": true,
            "stream": true
        }),
    )
    .await;

    let events = recv_until_completed(&mut ws).await;
    let added = events
        .iter()
        .find(|event| event["type"] == "response.output_item.added")
        .unwrap();
    let done = events
        .iter()
        .find(|event| event["type"] == "response.output_item.done")
        .unwrap();
    assert_eq!(added["item"]["namespace"], "mcp__agentic_fixture");
    assert_eq!(added["item"]["name"], "add_numbers");
    assert_eq!(done["item"]["namespace"], "mcp__agentic_fixture");
    assert_eq!(done["item"]["name"], "add_numbers");

    let completed = events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
    let response = &completed["response"];
    assert_eq!(response["output"][0]["namespace"], "mcp__agentic_fixture");
    assert_eq!(response["output"][0]["name"], "add_numbers");

    let requests = mock.request_bodies().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["tools"][0]["type"], "function");
    assert_eq!(
        requests[0]["tools"][0]["name"],
        "agentic_ns__mcp__agentic_fixture__add_numbers"
    );
}

#[tokio::test]
async fn test_websocket_bounds_and_restores_long_namespace_tool_name() {
    let namespace = "mcp__codex_apps__github";
    let member = "_remove_reaction_from_pr_review_comment";
    let upstream_name = model_visible_namespace_member_name(namespace, member);
    let mock = MockResponsesServer::start(vec![sse_function_call_response(
        "resp_upstream_long_namespace",
        &upstream_name,
    )])
    .await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": "use the long namespace tool",
            "tools": [{
                "type": "namespace",
                "name": namespace,
                "tools": [{
                    "type": "function",
                    "name": member,
                    "parameters": {"type": "object"}
                }]
            }],
            "store": true,
            "stream": true
        }),
    )
    .await;

    let events = recv_until_completed(&mut ws).await;
    let added = events
        .iter()
        .find(|event| event["type"] == "response.output_item.added")
        .unwrap();
    let done = events
        .iter()
        .find(|event| event["type"] == "response.output_item.done")
        .unwrap();
    assert_eq!(added["item"]["namespace"], namespace);
    assert_eq!(added["item"]["name"], member);
    assert_eq!(done["item"]["namespace"], namespace);
    assert_eq!(done["item"]["name"], member);

    let completed = events.last().unwrap();
    assert_eq!(completed["response"]["output"][0]["namespace"], namespace);
    assert_eq!(completed["response"]["output"][0]["name"], member);

    let requests = mock.request_bodies().await;
    let forwarded_name = requests[0]["tools"][0]["name"].as_str().unwrap();
    assert_eq!(forwarded_name, upstream_name);
    assert_eq!(forwarded_name.chars().count(), 64);
}

#[tokio::test]
async fn test_websocket_custom_tool_round_trip_and_continuation() {
    let mock = MockResponsesServer::start(vec![
        sse_custom_tool_call_response(),
        sse_response("resp_after_custom", "msg_after_custom", "CUSTOM TOOL COMPLETE"),
    ])
    .await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "apply the patch"}],
            "tools": [{
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch."
            }],
            "store": true,
            "stream": true
        }),
    )
    .await;

    let first_events = recv_until_completed(&mut ws).await;
    let event_types = first_events
        .iter()
        .filter_map(|event| event["type"].as_str())
        .collect::<Vec<_>>();
    assert!(event_types.contains(&"response.custom_tool_call_input.delta"));
    assert!(event_types.contains(&"response.custom_tool_call_input.done"));
    let first_completed = first_events.last().unwrap();
    assert_eq!(first_completed["response"]["output"][0]["type"], "custom_tool_call");
    assert_eq!(
        first_completed["response"]["output"][0]["input"],
        "*** Begin Patch\n*** End Patch"
    );
    let previous_response_id = first_completed["response"]["id"].as_str().unwrap();

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "previous_response_id": previous_response_id,
            "input": [{
                "type": "custom_tool_call_output",
                "call_id": "call_custom_1",
                "output": "Done!"
            }],
            "store": true,
            "stream": true
        }),
    )
    .await;

    let second_events = recv_until_completed(&mut ws).await;
    assert_eq!(
        second_events.last().unwrap()["response"]["output"][0]["content"][0]["text"],
        "CUSTOM TOOL COMPLETE"
    );

    let requests = mock.request_bodies().await;
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0]["tools"][0]["type"], "function");
    assert_eq!(
        requests[0]["tools"][0]["parameters"]["properties"]["input"]["type"],
        "string"
    );
    let continuation = requests[1]["input"].as_array().unwrap();
    assert!(continuation.iter().any(|item| {
        item["type"] == "function_call"
            && item["call_id"] == "call_custom_1"
            && item["arguments"] == "{\"input\":\"*** Begin Patch\\n*** End Patch\"}"
    }));
    assert!(continuation.iter().any(|item| {
        item["type"] == "function_call_output" && item["call_id"] == "call_custom_1" && item["output"] == "Done!"
    }));
    assert_eq!(requests[1]["tools"][0]["type"], "function");
}

#[tokio::test]
async fn test_websocket_executes_web_search_gateway_tool() {
    let mock_llm = MockResponsesServer::start(vec![
        web_search_function_call_sse_response(),
        sse_response("resp_final", "msg_final", "Use async carefully."),
    ])
    .await;
    let mock_you = MockYouSearchServer::start().await;
    let fixture = storage_backed_state_with_web_search(&mock_llm.url, Some(&mock_you.url)).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "search rust async"}],
            "tools": [{"type": "web_search_preview"}],
            "store": true,
            "stream": true
        }),
    )
    .await;

    let events = recv_until_completed(&mut ws).await;
    let event_types = events
        .iter()
        .filter_map(|event| event["type"].as_str())
        .collect::<Vec<_>>();

    assert!(event_types.contains(&"response.web_search_call.in_progress"));
    assert!(event_types.contains(&"response.web_search_call.searching"));
    assert!(event_types.contains(&"response.web_search_call.completed"));
    assert!(
        !events
            .iter()
            .any(|event| event["item"]["type"] == "function_call" && event["item"]["name"] == "web_search"),
        "internal gateway function_call events should not be forwarded"
    );
    assert_eq!(
        events.last().unwrap()["response"]["output"][1]["content"][0]["text"],
        "Use async carefully."
    );
    assert_eq!(mock_you.request_bodies().await[0]["query"], "rust async");

    let llm_requests = mock_llm.request_bodies().await;
    assert_eq!(llm_requests.len(), 2);
    assert_eq!(llm_requests[0]["tools"][0]["name"], "web_search");
    assert!(
        llm_requests[1]["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["type"] == "function_call_output" && item["call_id"] == "call_search")
    );
}

#[tokio::test]
async fn test_websocket_preserves_plain_function_tool_call_events() {
    let mock = MockResponsesServer::start(vec![sse_function_call_response("resp_upstream_1", "get_weather")]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "use the tool"}],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"}
                }
            ],
            "store": true,
            "stream": true
        }),
    )
    .await;

    let events = recv_until_completed(&mut ws).await;
    let added = events
        .iter()
        .find(|event| event["type"] == "response.output_item.added")
        .unwrap();
    let done = events
        .iter()
        .find(|event| event["type"] == "response.output_item.done")
        .unwrap();
    assert!(added["item"].get("namespace").is_none());
    assert_eq!(added["item"]["name"], "get_weather");
    assert!(done["item"].get("namespace").is_none());
    assert_eq!(done["item"]["name"], "get_weather");

    let completed = events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
    let response = &completed["response"];
    assert!(response["output"][0].get("namespace").is_none());
    assert_eq!(response["output"][0]["name"], "get_weather");

    let requests = mock.request_bodies().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["tools"][0]["type"], "function");
    assert_eq!(requests[0]["tools"][0]["name"], "get_weather");
}

#[tokio::test]
async fn test_websocket_continuation_rehydrates_previous_response() {
    let mock = MockResponsesServer::start(vec![
        sse_response("resp_upstream_1", "msg_upstream_1", "HELLO"),
        sse_response("resp_upstream_2", "msg_upstream_2", "WORLD"),
        sse_response("resp_upstream_3", "msg_upstream_3", "AGAIN"),
    ])
    .await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "text": {"verbosity": "low"},
            "store": true,
            "stream": true
        }),
    )
    .await;
    let first = recv_until_completed(&mut ws).await;
    let first_completed = first.last().unwrap();
    let previous_response_id = first_completed["response"]["id"].as_str().unwrap();

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "previous_response_id": previous_response_id,
            "input": [{"type": "message", "role": "user", "content": "continue"}],
            "text": {"verbosity": "high"},
            "store": true,
            "stream": true
        }),
    )
    .await;
    let second = recv_until_completed(&mut ws).await;
    let completed = second.last().unwrap();
    let event_types = second
        .iter()
        .map(|event| event["type"].as_str().unwrap())
        .collect::<Vec<_>>();

    assert_eq!(
        event_types,
        vec![
            "response.created",
            "response.output_item.added",
            "response.output_text.delta",
            "response.completed"
        ]
    );
    assert_eq!(second[2]["delta"], "WORLD");
    assert_eq!(completed["type"], "response.completed");
    let response = &completed["response"];
    assert_eq!(response["output"][0]["content"][0]["text"], "WORLD");
    assert_eq!(response["previous_response_id"], previous_response_id);

    let second_response_id = response["id"].as_str().unwrap();
    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "previous_response_id": second_response_id,
            "input": [{"type": "message", "role": "user", "content": "again"}],
            "store": true,
            "stream": true
        }),
    )
    .await;
    let third = recv_until_completed(&mut ws).await;
    assert_eq!(
        third.last().unwrap()["response"]["output"][0]["content"][0]["text"],
        "AGAIN"
    );

    let requests = mock.request_bodies().await;
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0]["text"], json!({"verbosity": "low"}));
    assert_eq!(requests[1]["text"], json!({"verbosity": "high"}));
    assert!(requests[2].get("text").is_none());
    assert!(requests[1].get("previous_response_id").is_none());
    assert_eq!(requests[1]["input"][0]["content"], "hi");
    assert_eq!(requests[1]["input"][1]["role"], "assistant");
    assert_eq!(requests[1]["input"][1]["content"][0]["text"], "HELLO");
    assert_eq!(requests[1]["input"][2]["content"], "continue");
}

#[tokio::test]
async fn test_websocket_unknown_previous_response_returns_error_event() {
    let mock = MockResponsesServer::start(vec![]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "stream_id": "missing-response-lane",
            "model": "test-model",
            "previous_response_id": "resp_missing",
            "input": [{"type": "message", "role": "user", "content": "continue"}],
            "store": true,
            "stream": true
        }),
    )
    .await;

    let error = recv_json(&mut ws).await;
    assert_eq!(error["type"], "error");
    assert_eq!(error["status"], StatusCode::NOT_FOUND.as_u16());
    assert_eq!(error["error"]["code"], "not_found");
    assert_eq!(error["stream_id"], "missing-response-lane");
    assert!(mock.request_bodies().await.is_empty());
}

#[tokio::test]
async fn test_websocket_rejects_binary_json_without_upstream_request() {
    let mock = MockResponsesServer::start(vec![sse_response("resp_upstream_1", "msg_upstream_1", "HELLO")]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    ws.send(Message::Binary(
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "store": true,
            "stream": true
        })
        .to_string()
        .into(),
    ))
    .await
    .unwrap();

    let error = recv_json(&mut ws).await;
    assert_eq!(error["type"], "error");
    assert_eq!(error["status"], StatusCode::BAD_REQUEST.as_u16());
    assert_eq!(error["error"]["code"], "invalid_request_error");
    assert!(mock.request_bodies().await.is_empty());
}

#[tokio::test]
async fn test_websocket_rejects_messages_larger_than_http_body_limit() {
    let mock = MockResponsesServer::start(vec![]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    if ws
        .send(Message::Text("x".repeat(10 * 1024 * 1024 + 1).into()))
        .await
        .is_ok()
    {
        let message = tokio::time::timeout(std::time::Duration::from_secs(2), ws.next())
            .await
            .expect("timed out waiting for websocket close/error")
            .expect("websocket should yield a close or error");
        assert!(message.is_err() || matches!(message, Ok(Message::Close(_))));
    }
    assert!(mock.request_bodies().await.is_empty());
}

#[tokio::test]
async fn test_websocket_ping_returns_pong_without_upstream_request() {
    let mock = MockResponsesServer::start(vec![]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_ping_and_wait_for_pong(&mut ws, Bytes::from_static(b"ping")).await;

    assert!(mock.request_bodies().await.is_empty());
}

#[tokio::test]
async fn test_websocket_shutdown_token_closes_idle_connection() {
    let mock = MockResponsesServer::start(vec![]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let shutdown_token = fixture.state.shutdown_token.clone();
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    shutdown_token.cancel();

    recv_close_or_end(&mut ws).await;
    assert!(mock.request_bodies().await.is_empty());
}

#[tokio::test]
async fn test_websocket_shutdown_drains_active_response_before_closing() {
    let (mock, arrived, release) =
        MockResponsesServer::start_gated(sse_response("resp_upstream_shutdown", "msg_upstream_shutdown", "DONE")).await;
    let fixture = storage_backed_state(&mock.url).await;
    let shutdown_token = fixture.state.shutdown_token.clone();
    let websocket_tracker = fixture.state.websocket_tracker.clone();
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "store": true,
            "stream": true
        }),
    )
    .await;
    tokio::time::timeout(std::time::Duration::from_secs(2), arrived)
        .await
        .expect("timed out waiting for shutdown request to reach upstream")
        .expect("arrival sender dropped");

    shutdown_token.cancel();
    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": "must not start during shutdown",
            "store": true,
            "stream": true
        }),
    )
    .await;
    let barrier = Bytes::from_static(b"shutdown-request-received");
    send_ping_and_wait_for_pong(&mut ws, barrier).await;
    release.send(()).unwrap();

    let events = recv_until_completed(&mut ws).await;
    assert_eq!(events.last().unwrap()["type"], "response.completed");
    recv_clean_close(&mut ws).await;
    tokio::time::timeout(std::time::Duration::from_secs(2), websocket_tracker.wait_until_idle())
        .await
        .expect("server did not receive the websocket close acknowledgement");
    assert_eq!(mock.request_bodies().await.len(), 1);
}

#[tokio::test]
async fn websocket_shutdown_drains_all_active_lanes_and_discards_queued_work() {
    let (first_arrived_tx, first_arrived_rx) = oneshot::channel();
    let (first_release_tx, first_release_rx) = oneshot::channel();
    let (second_arrived_tx, second_arrived_rx) = oneshot::channel();
    let (second_release_tx, second_release_rx) = oneshot::channel();
    let mock = MockResponsesServer::start_with_responses(vec![
        MockResponse::Gated {
            response: sse_response("resp_shutdown_a", "msg_shutdown_a", "A"),
            arrived: first_arrived_tx,
            release: first_release_rx,
        },
        MockResponse::Gated {
            response: sse_response("resp_shutdown_b", "msg_shutdown_b", "B"),
            arrived: second_arrived_tx,
            release: second_release_rx,
        },
    ])
    .await;
    let fixture = storage_backed_state(&mock.url).await;
    let shutdown_token = fixture.state.shutdown_token.clone();
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    for (stream_id, input) in [("shutdown-a", "active a"), ("shutdown-b", "active b")] {
        send_json(
            &mut ws,
            json!({
                "type": "response.create",
                "stream_id": stream_id,
                "model": "test-model",
                "input": input
            }),
        )
        .await;
    }
    tokio::time::timeout(std::time::Duration::from_secs(2), first_arrived_rx)
        .await
        .expect("timed out waiting for first shutdown lane")
        .expect("first arrival sender dropped");
    tokio::time::timeout(std::time::Duration::from_secs(2), second_arrived_rx)
        .await
        .expect("timed out waiting for second shutdown lane")
        .expect("second arrival sender dropped");
    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "stream_id": "shutdown-a",
            "model": "test-model",
            "input": "queued and discarded"
        }),
    )
    .await;
    send_ping_and_wait_for_pong(&mut ws, Bytes::from_static(b"queued-before-shutdown")).await;

    shutdown_token.cancel();
    first_release_tx.send(()).expect("release first shutdown lane");
    second_release_tx.send(()).expect("release second shutdown lane");

    let mut events = Vec::new();
    let mut terminal_count = 0;
    while terminal_count < 2 {
        let event = recv_json(&mut ws).await;
        if event["type"] == "response.completed" {
            terminal_count += 1;
        }
        events.push(event);
    }
    assert!(events.iter().any(|event| event["stream_id"] == "shutdown-a"));
    assert!(events.iter().any(|event| event["stream_id"] == "shutdown-b"));
    recv_clean_close(&mut ws).await;
    assert_eq!(mock.request_bodies().await.len(), 2);
}

#[tokio::test]
async fn test_websocket_client_close_cancels_hanging_upstream_stream() {
    let first_chunk = format!(
        "data: {}\n\n",
        json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": {"id": "resp_upstream_hanging", "status": "in_progress"}
        })
    );
    let (mock, upstream_dropped) = MockResponsesServer::start_hanging(first_chunk).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "store": true,
            "stream": true
        }),
    )
    .await;
    wait_for_request_count(&mock, 1).await;

    ws.close(None).await.unwrap();

    tokio::time::timeout(std::time::Duration::from_secs(2), upstream_dropped)
        .await
        .expect("timed out waiting for upstream stream to be dropped")
        .expect("upstream drop sender should notify");
    assert_eq!(mock.request_bodies().await.len(), 1);
}

#[tokio::test]
async fn websocket_client_close_cancels_all_active_stream_lanes() {
    let mut responses = Vec::new();
    let mut upstream_drops = Vec::new();
    for stream_id in ["disconnect-a", "disconnect-b"] {
        let first_chunk = format!(
            "data: {}\n\n",
            json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": {"id": format!("resp_{stream_id}"), "status": "in_progress"}
            })
        );
        let (drop_tx, drop_rx) = oneshot::channel();
        responses.push(MockResponse::Hanging { first_chunk, drop_tx });
        upstream_drops.push(drop_rx);
    }
    let mock = MockResponsesServer::start_with_responses(responses).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    for stream_id in ["disconnect-a", "disconnect-b"] {
        send_json(
            &mut ws,
            json!({
                "type": "response.create",
                "stream_id": stream_id,
                "model": "test-model",
                "input": "hang"
            }),
        )
        .await;
    }
    wait_for_request_count(&mock, 2).await;
    ws.close(None).await.expect("close websocket");

    for upstream_dropped in upstream_drops {
        tokio::time::timeout(std::time::Duration::from_secs(2), upstream_dropped)
            .await
            .expect("timed out waiting for active upstream stream cancellation")
            .expect("upstream drop sender should notify");
    }
}
