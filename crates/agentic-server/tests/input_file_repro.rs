//! Message-file preservation and rejection across typed Responses entry points.
//! All inference is intercepted by a loopback mock; no external model is used.

#[allow(dead_code)]
mod common;

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Duration;

use agentic_core::executor::{ConversationHandler, ExecutionContext, ResponseHandler};
use agentic_core::storage::pool::DbPool;
use agentic_core::storage::{ConversationStore, InOutItem, ResponseMetadata, ResponseStore, create_pool_with_schema};
use agentic_core::types::RequestPayload;
use axum::Router;
use axum::body::Bytes;
use axum::response::IntoResponse;
use axum::routing::post;
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

struct Fixture {
    url: String,
    client: reqwest::Client,
    requests: Arc<Mutex<Vec<Vec<u8>>>>,
    pool: Arc<DbPool>,
    gateway: JoinHandle<()>,
    model: JoinHandle<()>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.gateway.abort();
        self.model.abort();
    }
}

fn sse_response() -> String {
    let events = [
        json!({"type":"response.created", "sequence_number":0,
            "response":{"id":"resp_mock", "status":"in_progress"}}),
        json!({"type":"response.output_item.added", "sequence_number":1, "output_index":0,
            "item":{"id":"msg_mock", "type":"message"}}),
        json!({"type":"response.output_text.delta", "sequence_number":2, "item_id":"msg_mock",
            "output_index":0, "content_index":0, "delta":"mock summary"}),
        json!({"type":"response.completed", "sequence_number":3,
            "response":{"id":"resp_mock", "status":"completed", "usage":null}}),
    ];
    let mut body = String::new();
    for event in events {
        write!(&mut body, "data: {event}\n\n").expect("write to String");
    }
    body.push_str("data: [DONE]\n\n");
    body
}

impl Fixture {
    async fn new() -> Self {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let app = Router::new().route(
            "/v1/responses",
            post(move |body: Bytes| {
                let captured = Arc::clone(&captured);
                async move {
                    let request: Value = serde_json::from_slice(&body).expect("model request JSON");
                    captured.lock().await.push(body.to_vec());
                    if request["stream"] == true {
                        return ([("content-type", "text/event-stream")], sse_response()).into_response();
                    }
                    axum::Json(json!({
                        "id":"resp_mock", "object":"response", "created_at":0,
                        "model":"test-model", "status":"completed",
                        "output":[{"id":"msg_mock", "type":"message", "role":"assistant",
                            "status":"completed", "content":[{"type":"output_text",
                                "text":"mock summary", "annotations":[]}]}],
                        "usage":{"input_tokens":20, "output_tokens":5, "total_tokens":25}
                    }))
                    .into_response()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock");
        let address = listener.local_addr().expect("mock address");
        let model = tokio::spawn(async move { axum::serve(listener, app).await.expect("serve mock") });
        let config = common::test_config(&format!("http://{address}"));
        let pool = create_pool_with_schema(Some("sqlite://?mode=memory"))
            .await
            .expect("isolated in-memory store");
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("HTTP client");
        let mut state = common::test_state(&config);
        state.exec_ctx = Arc::new(ExecutionContext::new(
            ConversationHandler::new(ConversationStore::new(Arc::clone(&pool))),
            ResponseHandler::new(ResponseStore::new(Arc::clone(&pool))),
            Arc::new(client.clone()),
            config.llm_api_base,
        ));
        let (url, gateway) = common::spawn_gateway(state).await;
        Self {
            url,
            client,
            requests,
            pool,
            gateway,
            model,
        }
    }

    async fn post(&self, path: &str, body: Value) -> (reqwest::StatusCode, Value) {
        let response = self
            .client
            .post(format!("{}{path}", self.url))
            .json(&body)
            .send()
            .await
            .expect("HTTP response");
        let status = response.status();
        let bytes = response.bytes().await.expect("gateway body");
        let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| json!({"raw":String::from_utf8_lossy(&bytes)}));
        (status, body)
    }

    async fn captured(&self) -> Vec<Value> {
        self.requests
            .lock()
            .await
            .iter()
            .map(|bytes| serde_json::from_slice(bytes).unwrap())
            .collect()
    }

    async fn ws(&self, mut payload: Value) -> Value {
        payload["type"] = json!("response.create");
        let url = format!("{}/v1/responses", self.url.replacen("http://", "ws://", 1));
        let (mut socket, _) = connect_async(url).await.expect("WS handshake");
        socket
            .send(Message::Text(payload.to_string().into()))
            .await
            .expect("send turn");
        tokio::time::timeout(Duration::from_secs(5), async {
            for _ in 0..32 {
                let message = socket.next().await.expect("WS event").expect("valid frame");
                if let Message::Text(text) = message {
                    let event: Value = serde_json::from_str(&text).expect("WS JSON");
                    if matches!(
                        event["type"].as_str(),
                        Some("error" | "response.completed" | "response.failed" | "response.incomplete")
                    ) {
                        return event;
                    }
                }
            }
            panic!("no terminal event in 32 frames");
        })
        .await
        .expect("bounded WS response")
    }

    async fn row_counts(&self) -> (i64, i64) {
        let responses = sqlx::query_scalar("SELECT COUNT(*) FROM responses")
            .fetch_one(self.pool.as_ref())
            .await
            .expect("response count");
        let items = sqlx::query_scalar("SELECT COUNT(*) FROM items")
            .fetch_one(self.pool.as_ref())
            .await
            .expect("item count");
        (responses, items)
    }

    async fn seed_file_history(&self, conversation: bool) -> Value {
        let response_id = "resp_file_history";
        let metadata = ResponseMetadata {
            model: "test-model".to_owned(),
            ..Default::default()
        };
        let placeholder = InOutItem::Input(
            serde_json::from_value(json!({
                "role":"user", "content":"placeholder"
            }))
            .expect("placeholder message"),
        );
        let response_store = ResponseStore::new(Arc::clone(&self.pool));
        let reference = if conversation {
            let store = ConversationStore::new(Arc::clone(&self.pool));
            let data = store.create().await.expect("create conversation");
            store
                .persist(&data.conversation_id, response_id, None, vec![placeholder], &metadata)
                .await
                .expect("persist conversation fixture");
            json!({"conversation_id":data.conversation_id})
        } else {
            response_store
                .persist(response_id, None, vec![placeholder], &metadata)
                .await
                .expect("persist response fixture");
            json!({"previous_response_id":response_id})
        };
        let stored = response_store.get(response_id).await.expect("stored response");
        assert_eq!(stored.history_item_ids.len(), 1);
        // Install exact wire JSON independently of typed parsing. This preserves
        // the file even on the old baseline, which would deserialize it as Unknown.
        let mut raw = user_input(&file_part())[0].clone();
        raw["type"] = json!("message");
        raw["_agentic_item_kind"] = json!("input");
        let changed = sqlx::query("UPDATE items SET data = $1 WHERE id = $2")
            .bind(raw.to_string())
            .bind(&stored.history_item_ids[0])
            .execute(self.pool.as_ref())
            .await
            .expect("install raw history");
        assert_eq!(changed.rows_affected(), 1);
        let persisted: String = sqlx::query_scalar("SELECT data FROM items WHERE id = $1")
            .bind(&stored.history_item_ids[0])
            .fetch_one(self.pool.as_ref())
            .await
            .expect("read raw history");
        assert_eq!(serde_json::from_str::<Value>(&persisted).unwrap(), raw);
        reference
    }

    async fn assert_rejected(&self, path: &str, payload: Value) {
        let before = self.row_counts().await;
        let (status, body) = self.post(path, payload).await;
        assert_eq!(status, reqwest::StatusCode::BAD_REQUEST, "{body}");
        assert_file_error(&body);
        assert!(
            self.captured().await.is_empty(),
            "unsupported message file reached inference"
        );
        assert_eq!(self.row_counts().await, before, "rejected input was persisted");
    }
}

fn assert_file_error(body: &Value) {
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    let message = body["error"]["message"].as_str().expect("error message");
    assert!(
        message.contains("input_file") && message.contains("not supported"),
        "{message}"
    );
    assert!(
        !message.contains("example.invalid"),
        "do not reflect file contents in the error"
    );
}

fn file_part() -> Value {
    json!({"type":"input_file", "file_url":"https://example.invalid/document.pdf", "filename":"document.pdf"})
}

fn user_input(part: &Value) -> Value {
    // Keep meaningful text so compact does not reject an unrelated empty input.
    json!([{"role":"user", "content":[{"type":"input_text", "text":"Summarize this document"}, part]}])
}

fn assert_file_roundtrip(field: &str, value: &str) {
    let mut part = json!({"type":"input_file"});
    part[field] = json!(value);
    let parsed: RequestPayload =
        serde_json::from_value(json!({"model":"test-model", "input":user_input(&part)})).unwrap();
    let encoded = serde_json::to_value(parsed).unwrap();
    assert_eq!(
        encoded["input"][0]["content"][1], part,
        "typed parsing discarded {field}"
    );
}

#[test]
fn typed_file_data_survives() {
    assert_file_roundtrip("file_data", "data:application/pdf;base64,JVBERi0=");
}

#[test]
fn typed_file_id_survives() {
    assert_file_roundtrip("file_id", "file_test");
}

#[test]
fn typed_file_url_survives() {
    assert_file_roundtrip("file_url", "https://example.invalid/document.pdf");
}

#[test]
fn typed_filename_survives() {
    assert_file_roundtrip("filename", "document.pdf");
}

#[tokio::test]
async fn typed_http_rejects_user_file_before_inference() {
    let fixture = Fixture::new().await;
    let (status, body) = fixture
        .post(
            "/v1/responses",
            json!({
                "model":"test-model", "input":user_input(&file_part()), "store":true
            }),
        )
        .await;
    let captured = fixture.captured().await;
    eprintln!("HTTP status={status}; gateway={body}; upstream={captured:?}");
    assert_eq!(status, reqwest::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(captured.is_empty(), "unsupported user file reached inference");
}

#[tokio::test]
async fn explicit_compact_rejects_user_file_before_inference() {
    let fixture = Fixture::new().await;
    let (status, body) = fixture
        .post(
            "/v1/responses/compact",
            json!({
                "model":"test-model", "input":user_input(&file_part())
            }),
        )
        .await;
    let captured = fixture.captured().await;
    eprintln!("COMPACT status={status}; gateway={body}; upstream={captured:?}");
    assert_eq!(status, reqwest::StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        captured.is_empty(),
        "unsupported user file reached compaction inference"
    );
}

#[tokio::test]
async fn websocket_rejects_user_file_before_inference() {
    let fixture = Fixture::new().await;
    let url = format!("{}/v1/responses", fixture.url.replacen("http://", "ws://", 1));
    let (mut socket, _) = connect_async(url).await.expect("WS handshake");
    socket
        .send(Message::Text(
            json!({
                "type":"response.create", "model":"test-model", "store":true,
                "input":user_input(&file_part())
            })
            .to_string()
            .into(),
        ))
        .await
        .expect("send turn");
    let terminal = tokio::time::timeout(Duration::from_secs(5), async {
        for _ in 0..32 {
            let message = socket.next().await.expect("WS event").expect("valid WS frame");
            if let Message::Text(text) = message {
                let event: Value = serde_json::from_str(&text).expect("WS JSON");
                if matches!(
                    event["type"].as_str(),
                    Some("error" | "response.completed" | "response.failed" | "response.incomplete")
                ) {
                    return event;
                }
            }
        }
        panic!("no terminal event in 32 frames");
    })
    .await
    .expect("bounded WS response");
    let captured = fixture.captured().await;
    eprintln!("WS terminal={terminal}; upstream={captured:?}");
    assert_eq!(terminal["type"], "error");
    assert_eq!(terminal["status"], 400);
    assert_eq!(terminal["error"]["type"], "invalid_request_error");
    assert!(captured.is_empty(), "unsupported user file reached inference");
}

#[tokio::test]
async fn control_raw_proxy_preserves_exact_file_request_bytes() {
    let fixture = Fixture::new().await;
    let raw = format!(
        "{{\n  \"model\":\"test-model\", \"store\":false, \"input\": {}\n}}",
        user_input(&file_part())
    );
    let response = fixture
        .client
        .post(format!("{}/v1/responses", fixture.url))
        .header("content-type", "application/json")
        .body(raw.clone())
        .send()
        .await
        .expect("raw HTTP response");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let _ = response.bytes().await.expect("consume response");
    let captured = fixture.requests.lock().await;
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0], raw.as_bytes());
}

#[tokio::test]
async fn control_typed_text_and_image_are_preserved() {
    let fixture = Fixture::new().await;
    let image = json!({"type":"input_image", "image_url":"https://example.invalid/image.png", "detail":"low"});
    let input = user_input(&image);
    let (status, body) = fixture
        .post(
            "/v1/responses",
            json!({"model":"test-model", "store":true, "input":input}),
        )
        .await;
    assert_eq!(status, reqwest::StatusCode::OK, "{body}");
    let captured = fixture.captured().await;
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0]["input"][0]["content"], input[0]["content"]);
}

#[tokio::test]
async fn control_structured_tool_output_file_is_preserved() {
    let fixture = Fixture::new().await;
    let file = file_part();
    let input = json!([
        {"type":"function_call", "call_id":"call_test", "name":"read_file", "arguments":"{}"},
        {"type":"function_call_output", "call_id":"call_test", "output":[file.clone()]}
    ]);
    let (status, body) = fixture
        .post(
            "/v1/responses",
            json!({"model":"test-model", "store":true, "input":input}),
        )
        .await;
    assert_eq!(status, reqwest::StatusCode::OK, "{body}");
    let captured = fixture.captured().await;
    assert_eq!(captured.len(), 1);
    let output = captured[0]["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .expect("tool result forwarded");
    assert_eq!(output["output"], json!([file]));
}

#[test]
fn typed_file_all_fields_survive_together() {
    let part = json!({"type":"input_file", "file_data":"data:application/pdf;base64,JVBERi0=",
        "file_id":"file_test", "file_url":"https://example.invalid/document.pdf",
        "filename":"document.pdf", "detail":"auto"});
    let parsed: RequestPayload =
        serde_json::from_value(json!({"model":"test-model", "input":user_input(&part)})).unwrap();
    assert_eq!(serde_json::to_value(parsed).unwrap()["input"][0]["content"][1], part);
}

#[tokio::test]
async fn streaming_http_rejects_before_opening_sse() {
    let fixture = Fixture::new().await;
    fixture
        .assert_rejected(
            "/v1/responses",
            json!({"model":"test-model", "store":true, "stream":true, "input":user_input(&file_part())}),
        )
        .await;
}

#[tokio::test]
async fn automatic_compaction_rejects_before_summary_or_answer() {
    let fixture = Fixture::new().await;
    fixture
        .assert_rejected(
            "/v1/responses",
            json!({"model":"test-model", "store":false, "input":user_input(&file_part()),
            "context_management":[{"type":"compaction", "compact_threshold":1}]}),
        )
        .await;
}

#[tokio::test]
async fn compaction_trigger_cannot_bypass_file_validation() {
    let fixture = Fixture::new().await;
    let mut input = user_input(&file_part());
    input.as_array_mut().unwrap().push(json!({"type":"compaction_trigger"}));
    fixture
        .assert_rejected(
            "/v1/responses",
            json!({"model":"test-model", "store":false, "input":input}),
        )
        .await;
}

#[tokio::test]
async fn file_only_compaction_reports_unsupported_file_not_empty_input() {
    let fixture = Fixture::new().await;
    fixture
        .assert_rejected(
            "/v1/responses/compact",
            json!({"model":"test-model", "input":[{"role":"user", "content":[file_part()]}]}),
        )
        .await;
}

#[tokio::test]
async fn websocket_prewarm_rejects_without_persisting_file() {
    let fixture = Fixture::new().await;
    let terminal = fixture
        .ws(json!({"model":"test-model", "store":false, "generate":false,
        "input":user_input(&file_part())}))
        .await;
    assert_eq!(terminal["type"], "error", "{terminal}");
    assert_eq!(terminal["status"], 400);
    assert_file_error(&terminal);
    assert!(fixture.captured().await.is_empty());
    assert_eq!(fixture.row_counts().await, (0, 0));
}

async fn assert_history_http_rejected(conversation: bool, path: &str, extra: Value) {
    let fixture = Fixture::new().await;
    let mut payload = json!({"model":"test-model", "store":false, "input":"continue"});
    payload.as_object_mut().unwrap().extend(
        fixture
            .seed_file_history(conversation)
            .await
            .as_object()
            .unwrap()
            .clone(),
    );
    payload
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    fixture.assert_rejected(path, payload).await;
}

#[tokio::test]
async fn previous_response_file_is_rejected_after_rehydration() {
    assert_history_http_rejected(false, "/v1/responses", json!({})).await;
}

#[tokio::test]
async fn previous_response_file_is_rejected_before_streaming() {
    assert_history_http_rejected(false, "/v1/responses", json!({"stream":true})).await;
}

#[tokio::test]
async fn previous_response_file_is_rejected_by_explicit_compaction() {
    assert_history_http_rejected(false, "/v1/responses/compact", json!({})).await;
}

#[tokio::test]
async fn previous_response_file_is_rejected_before_automatic_compaction() {
    assert_history_http_rejected(
        false,
        "/v1/responses",
        json!({"context_management":[{"type":"compaction", "compact_threshold":1}]}),
    )
    .await;
}

#[tokio::test]
async fn previous_response_file_is_rejected_before_compaction_trigger() {
    assert_history_http_rejected(false, "/v1/responses", json!({"input":[{"type":"compaction_trigger"}]})).await;
}

#[tokio::test]
async fn conversation_file_is_rejected_after_rehydration() {
    assert_history_http_rejected(true, "/v1/responses", json!({})).await;
}

async fn assert_history_ws_rejected(conversation: bool, generate: bool) {
    let fixture = Fixture::new().await;
    let mut payload = json!({"model":"test-model", "input":"continue", "generate":generate});
    payload.as_object_mut().unwrap().extend(
        fixture
            .seed_file_history(conversation)
            .await
            .as_object()
            .unwrap()
            .clone(),
    );
    let before = fixture.row_counts().await;
    let terminal = fixture.ws(payload).await;
    assert_eq!(terminal["type"], "error", "{terminal}");
    assert_eq!(terminal["status"], 400);
    assert_file_error(&terminal);
    assert!(fixture.captured().await.is_empty());
    assert_eq!(fixture.row_counts().await, before);
}

#[tokio::test]
async fn previous_response_file_is_rejected_by_websocket() {
    assert_history_ws_rejected(false, true).await;
}

#[tokio::test]
async fn previous_response_file_is_rejected_by_websocket_prewarm() {
    assert_history_ws_rejected(false, false).await;
}

#[tokio::test]
async fn conversation_file_is_rejected_by_websocket() {
    assert_history_ws_rejected(true, true).await;
}

#[tokio::test]
async fn conversation_file_is_rejected_by_websocket_prewarm() {
    assert_history_ws_rejected(true, false).await;
}

#[test]
fn composed_upstream_request_cannot_bypass_file_validation() {
    use agentic_core::executor::{RequestContext, upstream_request};
    let request: RequestPayload =
        serde_json::from_value(json!({"model":"test-model", "input":user_input(&file_part())})).unwrap();
    let ctx = RequestContext {
        original_request: request.clone(),
        enriched_request: request,
        new_input_items: Vec::new(),
        response_id: "resp_composed".to_owned(),
        conversation_id: None,
        conversation_version: None,
    };
    for stream in [false, true] {
        let error = upstream_request(&ctx, stream).expect_err("direct typed upstream preparation must reject files");
        assert_eq!(error.http_status().as_u16(), 400);
        assert!(error.error_message().contains("input_file"));
    }
}

#[tokio::test]
async fn control_text_prewarm_still_persists_without_inference() {
    let fixture = Fixture::new().await;
    let terminal = fixture.ws(json!({"model":"test-model", "generate":false, "input":"Please read the local notes.md file with your read tool."})).await;
    assert_eq!(terminal["type"], "response.completed", "{terminal}");
    assert!(fixture.captured().await.is_empty());
    assert_eq!(fixture.row_counts().await, (1, 1));
}

#[tokio::test]
async fn control_streaming_raw_proxy_preserves_exact_file_request_bytes() {
    let fixture = Fixture::new().await;
    let raw = format!(
        "{{\n  \"model\":\"test-model\", \"store\":false, \"stream\":true, \"input\": {}\n}}",
        user_input(&file_part())
    );
    let response = fixture
        .client
        .post(format!("{}/v1/responses", fixture.url))
        .header("content-type", "application/json")
        .body(raw.clone())
        .send()
        .await
        .expect("proxy response");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let _ = response.bytes().await.expect("consume proxy stream");
    let captured = fixture.requests.lock().await;
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0], raw.as_bytes());
}

#[tokio::test]
async fn control_custom_tool_output_file_is_preserved() {
    let fixture = Fixture::new().await;
    let file = file_part();
    let input = json!([
        {"type":"custom_tool_call", "id":"ctc_test", "call_id":"call_test", "name":"read_file", "input":"document.pdf"},
        {"type":"custom_tool_call_output", "call_id":"call_test", "output":[file.clone()]}
    ]);
    let (status, body) = fixture
        .post(
            "/v1/responses",
            json!({"model":"test-model", "store":true, "input":input}),
        )
        .await;
    assert_eq!(status, reqwest::StatusCode::OK, "{body}");
    let captured = fixture.captured().await;
    assert_eq!(captured.len(), 1);
    let output = captured[0]["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .expect("normalized tool output");
    assert_eq!(output["output"], json!([file]));
}
