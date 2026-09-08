mod common;

use axum::Router;
use axum::body::Bytes;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::post;
use http::StatusCode;
use std::collections::VecDeque;
use std::convert::Infallible;
use std::fmt::Write as _;
use std::future::Future;
use std::num::NonZeroUsize;
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
use agentic_server::app::{AppState, DEFAULT_MAX_REQUEST_BODY_SIZE, WebSocketTracker};

use common::{spawn_gateway, spawn_mock_llm, test_config, test_state, test_state_with_max_request_body_size};

/// Deliberately tiny ceiling used to exercise the limit without large allocations.
const SMALL_REQUEST_SIZE_LIMIT: NonZeroUsize = NonZeroUsize::new(500).expect("nonzero");

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
        max_request_body_size: DEFAULT_MAX_REQUEST_BODY_SIZE,
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
    spawn_mock_vllm_json_capture_body(serde_json::json!({
        "id": "mock_id",
        "object": "response",
        "status": "completed",
        "model": "test",
        "output": [],
        "created_at": 0
    }))
    .await
}

async fn spawn_mock_vllm_json_capture_body(
    response_body: serde_json::Value,
) -> (String, Arc<Mutex<Vec<serde_json::Value>>>, tokio::task::JoinHandle<()>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let route_requests = Arc::clone(&requests);
    let response_body = Arc::new(response_body.to_string());
    let app = Router::new().route(
        "/v1/responses",
        post(move |body: Bytes| {
            let route_requests = Arc::clone(&route_requests);
            let response_body = Arc::clone(&response_body);
            async move {
                let body = serde_json::from_slice::<serde_json::Value>(&body).unwrap();
                route_requests.lock().await.push(body);
                axum::response::Response::builder()
                    .status(200)
                    .header("Content-Type", "application/json")
                    .body(axum::body::Body::from(response_body.as_str().to_owned()))
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

async fn spawn_mock_vllm_json_capture_bytes() -> (String, Arc<Mutex<Vec<Bytes>>>, tokio::task::JoinHandle<()>) {
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

async fn spawn_tool_search_sse_sequence(
    responses: Vec<String>,
) -> (String, Arc<Mutex<Vec<serde_json::Value>>>, tokio::task::JoinHandle<()>) {
    let responses = Arc::new(Mutex::new(VecDeque::from(responses)));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let route_responses = Arc::clone(&responses);
    let route_requests = Arc::clone(&requests);
    let app = Router::new().route(
        "/v1/responses",
        post(move |body: Bytes| {
            let route_responses = Arc::clone(&route_responses);
            let route_requests = Arc::clone(&route_requests);
            async move {
                route_requests
                    .lock()
                    .await
                    .push(serde_json::from_slice(&body).expect("request JSON"));
                let response = route_responses.lock().await.pop_front().expect("prepared SSE response");
                axum::response::Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
                    .body(axum::body::Body::from(response))
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

fn tool_search_sse() -> String {
    let events = [
        serde_json::json!({
            "type":"response.created",
            "response":{
                "id":"up_search","status":"in_progress",
                "tools":[{"type":"function","name":"tool_search","parameters":{"type":"object"}}]
            }
        }),
        serde_json::json!({
            "type":"response.output_item.added","output_index":0,
            "item":{"id":"fc_search","type":"function_call","status":"in_progress",
                "name":"tool_search","call_id":"call_search","arguments":""}
        }),
        serde_json::json!({
            "type":"response.function_call_arguments.delta","output_index":0,
            "item_id":"fc_search","delta":"{\"query\":\"weather\"}"
        }),
        serde_json::json!({
            "type":"response.function_call_arguments.done","output_index":0,
            "item_id":"fc_search","name":"tool_search","arguments":"{\"query\":\"weather\"}"
        }),
        serde_json::json!({
            "type":"response.output_item.done","output_index":0,
            "item":{"id":"fc_search","type":"function_call","status":"completed",
                "name":"tool_search","call_id":"call_search","arguments":"{\"query\":\"weather\"}"}
        }),
        serde_json::json!({"type":"response.completed","response":{"id":"up_search","status":"completed","usage":null}}),
    ];
    encode_sse_events(events)
}

fn encode_sse_events(events: impl IntoIterator<Item = serde_json::Value>) -> String {
    let mut response = String::new();
    for event in events {
        writeln!(&mut response, "data: {event}\n").expect("writing to String cannot fail");
    }
    response.push_str("data: [DONE]\n\n");
    response
}

fn function_call_sse(name: &str, item_id: &str, call_id: &str, arguments: &str) -> String {
    let events = [
        serde_json::json!({
            "type":"response.created",
            "response":{
                "id":"up_call","status":"in_progress",
                "tools":[{"type":"function","name":name,"parameters":{"type":"object"}}]
            }
        }),
        serde_json::json!({
            "type":"response.output_item.added","output_index":0,
            "item":{"id":item_id,"type":"function_call","status":"in_progress",
                "name":name,"call_id":call_id,"arguments":""}
        }),
        serde_json::json!({
            "type":"response.output_item.done","output_index":0,
            "item":{"id":item_id,"type":"function_call","status":"completed",
                "name":name,"call_id":call_id,"arguments":arguments}
        }),
        serde_json::json!({"type":"response.completed","response":{"id":"up_call","status":"completed","usage":null}}),
    ];
    encode_sse_events(events)
}

fn final_message_sse() -> String {
    let events = [
        serde_json::json!({"type":"response.created","response":{"id":"up_final","status":"in_progress"}}),
        serde_json::json!({
            "type":"response.output_item.added","output_index":0,
            "item":{"id":"msg_final","type":"message","role":"assistant","status":"in_progress","content":[]}
        }),
        serde_json::json!({
            "type":"response.output_text.delta","output_index":0,"content_index":0,
            "item_id":"msg_final","delta":"PARIS_WEATHER_OK"
        }),
        serde_json::json!({"type":"response.completed","response":{"id":"up_final","status":"completed","usage":null}}),
    ];
    encode_sse_events(events)
}

fn assert_public_search_sse(
    first_events: &[serde_json::Value],
    deferred_weather: &serde_json::Value,
) -> serde_json::Value {
    let public_tools = serde_json::json!([
        {
            "type":"tool_search","execution":"client","description":"Search tools",
            "parameters":{"type":"object","properties":{"query":{"type":"string"}}}
        },
        deferred_weather
    ]);
    let response_tool_envelopes = first_events
        .iter()
        .filter_map(|event| event.get("response"))
        .filter_map(|response| response.get("tools"))
        .collect::<Vec<_>>();
    assert!(!response_tool_envelopes.is_empty());
    assert!(response_tool_envelopes.iter().all(|tools| *tools == &public_tools));
    assert!(first_events.iter().all(|event| {
        event["response"]["tools"].as_array().is_none_or(|tools| {
            tools
                .iter()
                .all(|tool| !(tool["type"] == "function" && tool["name"] == "tool_search"))
        })
    }));
    assert_eq!(
        first_events
            .iter()
            .map(|event| event["sequence_number"].as_u64())
            .collect::<Vec<_>>(),
        (0..u64::try_from(first_events.len()).unwrap())
            .map(Some)
            .collect::<Vec<_>>()
    );
    assert!(first_events.iter().all(|event| {
        !matches!(
            event["type"].as_str(),
            Some("response.function_call_arguments.delta" | "response.function_call_arguments.done" | "error")
        )
    }));
    let search_lifecycle = first_events
        .iter()
        .filter(|event| {
            matches!(
                event["type"].as_str(),
                Some("response.output_item.added" | "response.output_item.done")
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(search_lifecycle.len(), 2);
    assert_eq!(search_lifecycle[0]["item"]["type"], "tool_search_call");
    assert_eq!(search_lifecycle[0]["item"]["status"], "in_progress");
    assert_eq!(search_lifecycle[0]["item"]["arguments"], serde_json::json!({}));
    assert_eq!(search_lifecycle[1]["item"]["type"], "tool_search_call");
    assert_eq!(search_lifecycle[1]["item"]["status"], "completed");
    assert_eq!(
        search_lifecycle[1]["item"]["arguments"],
        serde_json::json!({"query":"weather"})
    );
    assert_eq!(search_lifecycle[0]["item"]["id"], search_lifecycle[1]["item"]["id"]);
    assert_eq!(
        search_lifecycle[0]["item"]["call_id"],
        search_lifecycle[1]["item"]["call_id"]
    );
    assert_eq!(search_lifecycle[0]["output_index"], search_lifecycle[1]["output_index"]);
    let search_call = first_events.last().expect("first terminal")["response"]["output"][0].clone();
    assert_eq!(search_call["type"], "tool_search_call");
    assert_eq!(search_call["id"], "tsc_search");
    assert_eq!(search_call, search_lifecycle[1]["item"]);
    search_call
}

#[tokio::test]
async fn test_http_sse_tool_search_three_request_continuation_stays_public() {
    let (llm_url, requests, _llm) = spawn_tool_search_sse_sequence(vec![
        tool_search_sse(),
        function_call_sse("get_weather", "fc_weather", "call_weather", "{\"city\":\"Paris\"}"),
        final_message_sse(),
    ])
    .await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;
    let client = reqwest::Client::new();
    let deferred_weather = serde_json::json!({
        "type":"function","name":"get_weather","description":"Get weather",
        "parameters":{"type":"object","properties":{"city":{"type":"string"}}},
        "defer_loading":true
    });
    let first_response = client
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model":"test","input":"find weather","store":false,"stream":true,"parallel_tool_calls":false,
            "tools":[
                {"type":"tool_search","execution":"client","description":"Search tools",
                    "parameters":{"type":"object","properties":{"query":{"type":"string"}}}},
                deferred_weather.clone()
            ]
        }))
        .send()
        .await
        .expect("first response");
    assert_eq!(first_response.status(), StatusCode::OK);
    let first_body = first_response.text().await.expect("first SSE body");
    let first_events = sse_events(&first_body);
    let search_call = assert_public_search_sse(&first_events, &deferred_weather);

    let search_output = serde_json::json!({
        "type":"tool_search_output","call_id":"call_search","tools":[deferred_weather]
    });
    let second_response = client
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model":"test","input":[search_call.clone(),search_output.clone()],"store":false,"stream":true
        }))
        .send()
        .await
        .expect("second response");
    let second_body = second_response.text().await.expect("second SSE body");
    let second_events = sse_events(&second_body);
    let weather_call = second_events.last().unwrap()["response"]["output"][0].clone();
    assert_eq!(weather_call["type"], "function_call");
    assert_eq!(weather_call["name"], "get_weather");

    let third_response = client
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model":"test",
            "input":[
                search_call,search_output,weather_call,
                {"type":"function_call_output","call_id":"call_weather","output":"sunny"}
            ],
            "store":false,"stream":true
        }))
        .send()
        .await
        .expect("third response");
    let third_events = sse_events(&third_response.text().await.expect("third SSE body"));
    assert_eq!(
        third_events.last().unwrap()["response"]["output"][0]["content"][0]["text"],
        "PARIS_WEATHER_OK"
    );

    let requests = requests.lock().await;
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0]["tools"].as_array().map(Vec::len), Some(1));
    assert_eq!(requests[0]["tools"][0]["name"], "tool_search");
    assert!(
        requests[1]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "get_weather")
    );
    assert!(requests[1]["input"].as_array().unwrap().iter().any(|item| {
        item["type"] == "function_call" && item["name"] == "tool_search" && item["call_id"] == "call_search"
    }));
    assert!(
        requests[2]["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| { item["type"] == "function_call_output" && item["call_id"] == "call_weather" })
    );
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
async fn test_store_false_manual_tool_search_replay_loads_returned_function() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture_body(serde_json::json!({
        "id": "upstream_loaded_call",
        "object": "response",
        "status": "completed",
        "model": "test",
        "created_at": 0,
        "output": [{
            "type": "function_call",
            "id": "fc_weather_1",
            "call_id": "call_weather_1",
            "name": "get_weather",
            "arguments": "{\"city\":\"Paris\"}",
            "status": "completed"
        }]
    }))
    .await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": [
                {
                    "type": "tool_search_call",
                    "id": "tsc_1",
                    "call_id": "call_search_1",
                    "arguments": {"query": "weather"}
                },
                {
                    "type": "tool_search_output",
                    "call_id": "call_search_1",
                    "tools": [{
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"]
                        },
                        "defer_loading": true
                    }]
                }
            ],
            "tools": [],
            "store": false,
            "stream": false
        }))
        .send()
        .await
        .expect("gateway response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = response.json().await.expect("response JSON");
    assert_eq!(body["output"][0]["type"], "function_call");
    assert_eq!(body["output"][0]["name"], "get_weather");

    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert!(requests[0].get("previous_response_id").is_none());
    assert_eq!(requests[0]["input"][0]["type"], "function_call");
    assert_eq!(requests[0]["input"][0]["name"], "tool_search");
    assert_eq!(requests[0]["input"][0]["call_id"], "call_search_1");
    assert_eq!(requests[0]["input"][1]["type"], "function_call_output");
    assert_eq!(requests[0]["input"][1]["call_id"], "call_search_1");
    assert_eq!(requests[0]["tools"].as_array().map(Vec::len), Some(1));
    assert_eq!(requests[0]["tools"][0]["name"], "get_weather");
    assert!(requests[0]["tools"][0].get("defer_loading").is_none());
}

#[tokio::test]
async fn test_store_false_fresh_tool_search_lowers_and_translates_blocking_call() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture_body(serde_json::json!({
        "id": "upstream_search_call",
        "object": "response",
        "status": "completed",
        "model": "test",
        "created_at": 0,
        "output": [{
            "type": "function_call",
            "id": "fc_search_1",
            "call_id": "call_search_1",
            "name": "tool_search",
            "arguments": "{\"query\":\"weather\"}",
            "status": "completed"
        }]
    }))
    .await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": "find a weather tool",
            "tools": [
                {
                    "type": "tool_search",
                    "execution": "client",
                    "description": "Search the client catalog",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                },
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                    "defer_loading": true
                }
            ],
            "store": false,
            "stream": false
        }))
        .send()
        .await
        .expect("gateway response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = response.json().await.expect("response JSON");
    assert_eq!(
        body["output"][0],
        serde_json::json!({
            "type": "tool_search_call",
            "id": "tsc_search_1",
            "call_id": "call_search_1",
            "execution": "client",
            "arguments": {"query": "weather"},
            "status": "completed"
        })
    );

    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["tools"].as_array().map(Vec::len), Some(1));
    assert_eq!(requests[0]["tools"][0]["type"], "function");
    assert_eq!(requests[0]["tools"][0]["name"], "tool_search");
    assert!(
        requests[0]["tools"]
            .as_array()
            .is_some_and(|tools| tools.iter().all(|tool| tool["name"] != "get_weather")),
        "deferred function schema must not be a top-level private tool"
    );
    assert!(
        requests[0]["tools"][0]["description"]
            .as_str()
            .is_some_and(|description| description.contains("get_weather")),
        "safe synthetic catalog keeps function identity"
    );
}

#[tokio::test]
async fn test_blocking_tool_search_rejects_invalid_upstream_arguments_as_bad_gateway() {
    let (llm_url, _requests, _llm) = spawn_mock_vllm_json_capture_body(serde_json::json!({
        "id": "upstream_bad_search_call",
        "object": "response",
        "status": "completed",
        "model": "test",
        "created_at": 0,
        "output": [{
            "type": "function_call",
            "id": "fc_search_bad",
            "call_id": "call_search_bad",
            "name": "tool_search",
            "arguments": "not valid JSON",
            "status": "completed"
        }]
    }))
    .await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .json(&serde_json::json!({
            "model": "test",
            "input": "find a tool",
            "tools": [{
                "type": "tool_search",
                "execution": "client",
                "description": "Search",
                "parameters": {"type": "object"}
            }],
            "store": false,
            "stream": false
        }))
        .send()
        .await
        .expect("gateway response");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body: serde_json::Value = response.json().await.expect("error JSON");
    assert_eq!(body["error"]["type"], "tool_error");
}

#[tokio::test]
async fn test_store_false_proxies_unknown_text_format_verbatim() {
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture_bytes().await;
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
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture_bytes().await;
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
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture_bytes().await;
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
    let (llm_url, requests, _llm) = spawn_mock_vllm_json_capture_bytes().await;
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

/// The configured ceiling covers the serialized request, so a body only a few
/// hundred bytes long is rejected once the limit is lowered to match.
#[tokio::test]
async fn test_configured_request_size_limit_rejects_oversized_body() {
    // Arrange — a 500-byte ceiling; the LLM is never reached.
    let (llm_url, _h1) = spawn_mock_llm().await;
    let state = test_state_with_max_request_body_size(&test_config(&llm_url), SMALL_REQUEST_SIZE_LIMIT);
    let (gw_url, _h2) = spawn_gateway(state).await;

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses"))
        .header("Content-Type", "application/json")
        .body("x".repeat(SMALL_REQUEST_SIZE_LIMIT.get() + 1))
        .send()
        .await
        .unwrap();

    // Assert
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        resp.json::<serde_json::Value>().await.unwrap()["error"]["code"],
        "body_too_large"
    );
}

#[tokio::test]
async fn test_configured_request_size_limit_rejects_oversized_compaction_body() {
    // Arrange — `/v1/responses/compact` reads through the same configured ceiling.
    let (llm_url, _h1) = spawn_mock_llm().await;
    let state = test_state_with_max_request_body_size(&test_config(&llm_url), SMALL_REQUEST_SIZE_LIMIT);
    let (gw_url, _h2) = spawn_gateway(state).await;

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/responses/compact"))
        .header("Content-Type", "application/json")
        .body("x".repeat(SMALL_REQUEST_SIZE_LIMIT.get() + 1))
        .send()
        .await
        .unwrap();

    // Assert
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn test_raised_request_size_limit_admits_larger_body() {
    // Arrange — the same payload shape that a 500-byte ceiling would reject.
    let (llm_url, requests, _h1) = spawn_mock_vllm_json_capture().await;
    let state =
        test_state_with_max_request_body_size(&test_config(&llm_url), NonZeroUsize::new(64 * 1024).expect("nonzero"));
    let (gw_url, _h2) = spawn_gateway(state).await;
    let prompt = "x".repeat(4 * 1024);

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

    // Assert — the raised ceiling lets the request through to the upstream.
    assert_eq!(resp.status(), StatusCode::OK);
    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["input"][0]["content"].as_str().unwrap().len(), 4 * 1024);
}
