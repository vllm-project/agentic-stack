//! Conversation/responses rehydration benchmarks — full HTTP stack.
//!
//! Measures per-turn wall-clock time for both blocking and streaming paths
//! through the complete request pipeline: client → axum router → executor →
//! LLM (mock or real) → response.
//!
//! Prior turns are seeded **before** criterion starts timing so each sample
//! measures only the cost of the Nth turn with N-1 prior turns in the DB.
//!
//! # Environment variables
//!
//! | Variable       | Default | Description                                      |
//! |----------------|---------|--------------------------------------------------|
//! | `BENCH_TURNS`  | `2`     | Max turns; bench sweeps 1..=BENCH_TURNS           |
//! | `LLM_BASE_URL` | mock    | Real LLM URL; omit to use the built-in mock       |
//!
//! ```bash
//! # Mock LLM, 5 turns
//! BENCH_TURNS=5 cargo bench --bench benches -- conversation_rehydration
//!
//! # Real model, 3 turns
//! BENCH_TURNS=3 LLM_BASE_URL=http://localhost:9090 \
//!     cargo bench --bench benches -- conversation_rehydration
//! ```

use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::Request;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Router, serve};
use bytes::Bytes;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group};
use futures::stream;
use http::StatusCode;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;
use tokio_util::sync::CancellationToken;

use agentic_core::config::Config;
use agentic_core::executor::{ConversationHandler, ExecutionContext, ResponseHandler};
use agentic_core::proxy::ProxyState;
use agentic_core::storage::{ConversationStore, ResponseStore, create_pool_with_schema};
use agentic_server::app::{AppState, DEFAULT_MAX_REQUEST_BODY_SIZE, ServerConfig, WebSocketTracker, build_router};

fn bench_turns() -> usize {
    std::env::var("BENCH_TURNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2)
        .max(1)
}

fn llm_base_url() -> Option<String> {
    std::env::var("LLM_BASE_URL")
        .ok()
        .map(|u| u.trim_end_matches('/').to_owned())
}

/// Resolve the model name to use in requests.
///
/// Priority:
/// 1. `BENCH_MODEL` env var
/// 2. First model from `{LLM_BASE_URL}/v1/models` (when a real server is configured)
/// 3. `"mock"` — safe default for the built-in mock LLM
async fn bench_model(llm_url: &str) -> String {
    if let Ok(m) = std::env::var("BENCH_MODEL") {
        return m;
    }
    // Only auto-detect when pointing at a real server
    if std::env::var("LLM_BASE_URL").is_ok() {
        if let Ok(resp) = reqwest::get(format!("{llm_url}/v1/models")).await {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if let Some(id) = json["data"][0]["id"].as_str() {
                    return id.to_string();
                }
            }
        }
    }
    "mock".to_string()
}

async fn mock_responses(req: Request) -> Response {
    let body_bytes = axum::body::to_bytes(req.into_body(), 10 * 1024 * 1024)
        .await
        .unwrap_or_default();
    let streaming = serde_json::from_slice::<serde_json::Value>(&body_bytes)
        .ok()
        .and_then(|j| j.get("stream").and_then(serde_json::Value::as_bool))
        .unwrap_or(false);

    if streaming {
        let chunks: Vec<Result<Bytes, Infallible>> = vec![
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_item.added\",\
                 \"item\":{\"id\":\"msg_b\",\"type\":\"message\",\
                 \"status\":\"in_progress\",\"content\":[],\"role\":\"assistant\"}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
            )),
            Ok(Bytes::from(concat!(
                "data: {\"type\":\"response.completed\",\"response\":{",
                "\"id\":\"resp_mock\",\"object\":\"response\",\"status\":\"completed\",",
                "\"model\":\"mock\",\"created_at\":0,",
                "\"output\":[{\"type\":\"message\",\"id\":\"msg_b\",",
                "\"role\":\"assistant\",\"status\":\"completed\",",
                "\"content\":[{\"type\":\"output_text\",\"text\":\"ok\",\"annotations\":[]}]}],",
                "\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2,",
                "\"input_tokens_details\":{\"cached_tokens\":0},",
                "\"output_tokens_details\":{\"reasoning_tokens\":0}}}}\n\n",
            ))),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ];
        return (
            StatusCode::OK,
            [("content-type", "text/event-stream; charset=utf-8")],
            Body::from_stream(stream::iter(chunks)),
        )
            .into_response();
    }

    (
        StatusCode::OK,
        [("content-type", "application/json")],
        concat!(
            r#"{"id":"resp_mock","object":"response","status":"completed","model":"mock","#,
            r#""created_at":0,"output":[{"type":"message","id":"msg_b","role":"assistant","#,
            r#""status":"completed","content":[{"type":"output_text","text":"ok","annotations":[]}]}],"#,
            r#""usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,"#,
            r#""input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}}}"#,
        ),
    )
        .into_response()
}

async fn spawn_mock_llm() -> String {
    let app = Router::new()
        .route("/health", get(|| async { StatusCode::OK.into_response() }))
        .route("/v1/responses", post(mock_responses));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

async fn spawn_gateway(llm_url: &str) -> (Arc<reqwest::Client>, String) {
    let db_path = std::env::temp_dir().join(format!("gateway_bench_{}.db", uuid::Uuid::now_v7()));
    let pool = create_pool_with_schema(Some(&format!("sqlite://{}", db_path.display())))
        .await
        .expect("bench db");

    let config = Config {
        llm_api_base: llm_url.to_owned(),
        openai_api_key: None,
        llm_ready_timeout_s: 5.0,
        llm_ready_interval_s: 0.1,
        skip_llm_ready_check: false,
        db_url: Some(format!("sqlite://{}", db_path.display())),
        postgres: agentic_core::config::PostgresConfig::default(),
        sqlite: agentic_core::config::SqliteConfig::default(),
        tools: agentic_core::config::ToolRuntimeConfig::default(),
    };

    let proxy_state = ProxyState::new(config.clone()).unwrap();
    let exec_ctx = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::new(pool.clone())),
        ResponseHandler::new(ResponseStore::new(pool)),
        Arc::new(reqwest::Client::new()),
        config.llm_api_base.clone(),
    ));
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

    let router = build_router(state, &ServerConfig::from_env());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { serve(listener, router).await.unwrap() });

    (Arc::new(reqwest::Client::new()), format!("http://{addr}"))
}

/// Create a conversation and run `prior_turns` non-streaming turns.
///
/// Each turn passes both `conversation_id` and, from the second turn on,
/// `previous_response_id` from the immediately preceding turn — matching
/// real client usage.
///
/// Returns `(conversation_id, last_response_id)` for use in the benchmarked turn.
async fn seed_conversation(
    client: &reqwest::Client,
    gw: &str,
    model: &str,
    prior_turns: usize,
) -> (String, Option<String>) {
    let conv: serde_json::Value = client
        .post(format!("{gw}/v1/conversations"))
        .header("Content-Type", "application/json")
        .body("{}")
        .send()
        .await
        .expect("create conv")
        .json()
        .await
        .expect("conv json");

    let conv_id = conv["id"].as_str().expect("conv id").to_string();
    let mut prev_id: Option<String> = None;

    for _ in 0..prior_turns {
        let mut body = serde_json::json!({
            "model": model, "input": "bench",
            "store": true, "stream": false,
            "conversation_id": conv_id
        });
        if let Some(ref id) = prev_id {
            body["previous_response_id"] = serde_json::Value::String(id.clone());
        }
        let resp: serde_json::Value = client
            .post(format!("{gw}/v1/responses"))
            .json(&body)
            .send()
            .await
            .expect("seed turn")
            .json()
            .await
            .expect("seed turn json");
        prev_id = resp["id"].as_str().map(str::to_string);
    }

    (conv_id, prev_id)
}

/// Seed N-1 turns via `previous_response_id` chaining (no `conversation_id`).
/// Returns the last response id for the benchmarked turn.
async fn seed_response_chain(client: &reqwest::Client, gw: &str, model: &str, prior_turns: usize) -> Option<String> {
    let mut prev_id: Option<String> = None;
    for _ in 0..prior_turns {
        let mut body = serde_json::json!({
            "model": model, "input": "bench",
            "store": true, "stream": false
        });
        if let Some(ref id) = prev_id {
            body["previous_response_id"] = serde_json::Value::String(id.clone());
        }
        let resp: serde_json::Value = client
            .post(format!("{gw}/v1/responses"))
            .json(&body)
            .send()
            .await
            .expect("seed response turn")
            .json()
            .await
            .expect("seed response json");
        prev_id = resp["id"].as_str().map(str::to_string);
    }
    prev_id
}

/// # Panics
/// Panics if the gateway or DB setup fails — intentional in benchmarks.
#[allow(clippy::too_many_lines)]
pub fn gateway_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let max_turns = bench_turns();

    let llm_url = llm_base_url().unwrap_or_else(|| rt.block_on(spawn_mock_llm()));
    let model = rt.block_on(bench_model(&llm_url));
    let (client, gw_url) = rt.block_on(spawn_gateway(&llm_url));

    eprintln!("gateway bench  model={model}  turns=1..={max_turns}  llm={llm_url}");

    // ── conversation_rehydration — POST /v1/responses with conversation_id ──

    let mut group = c.benchmark_group("conversation_rehydration/non_streaming");
    for turns in 1..=max_turns {
        eprintln!(
            "  [seed] conversation_rehydration/non_streaming  turns={turns}  prior={}",
            turns - 1
        );
        let (conv_id, _) = rt.block_on(seed_conversation(&client, &gw_url, &model, turns - 1));
        group.bench_with_input(BenchmarkId::new("turns", turns), &turns, |b, _| {
            b.to_async(Runtime::new().unwrap()).iter_batched(
                || (conv_id.clone(), model.clone()),
                |(cid, mdl)| {
                    let client = Arc::clone(&client);
                    let url = format!("{gw_url}/v1/responses");
                    async move {
                        let body = serde_json::json!({
                            "model": mdl, "input": "bench",
                            "store": true, "stream": false, "conversation_id": cid
                        });
                        client
                            .post(&url)
                            .json(&body)
                            .send()
                            .await
                            .unwrap()
                            .bytes()
                            .await
                            .unwrap()
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();

    let mut group = c.benchmark_group("conversation_rehydration/streaming");
    for turns in 1..=max_turns {
        eprintln!(
            "  [seed] conversation_rehydration/streaming  turns={turns}  prior={}",
            turns - 1
        );
        let (conv_id, _) = rt.block_on(seed_conversation(&client, &gw_url, &model, turns - 1));
        group.bench_with_input(BenchmarkId::new("turns", turns), &turns, |b, _| {
            b.to_async(Runtime::new().unwrap()).iter_batched(
                || (conv_id.clone(), model.clone()),
                |(cid, mdl)| {
                    let client = Arc::clone(&client);
                    let url = format!("{gw_url}/v1/responses");
                    async move {
                        let body = serde_json::json!({
                            "model": mdl, "input": "bench",
                            "store": true, "stream": true, "conversation_id": cid
                        });
                        client
                            .post(&url)
                            .json(&body)
                            .send()
                            .await
                            .unwrap()
                            .bytes()
                            .await
                            .unwrap()
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();

    // ── response_rehydration — POST /v1/responses with previous_response_id ──

    let mut group = c.benchmark_group("response_rehydration/non_streaming");
    for turns in 1..=max_turns {
        eprintln!(
            "  [seed] response_rehydration/non_streaming  turns={turns}  prior={}",
            turns - 1
        );
        let prev_id = rt.block_on(seed_response_chain(&client, &gw_url, &model, turns - 1));
        group.bench_with_input(BenchmarkId::new("turns", turns), &turns, |b, _| {
            b.to_async(Runtime::new().unwrap()).iter_batched(
                || (prev_id.clone(), model.clone()),
                |(pid, mdl)| {
                    let client = Arc::clone(&client);
                    let url = format!("{gw_url}/v1/responses");
                    async move {
                        let mut body = serde_json::json!({"model": mdl, "input": "bench",
                            "store": true, "stream": false});
                        if let Some(id) = pid {
                            body["previous_response_id"] = serde_json::Value::String(id);
                        }
                        client
                            .post(&url)
                            .json(&body)
                            .send()
                            .await
                            .unwrap()
                            .bytes()
                            .await
                            .unwrap()
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();

    let mut group = c.benchmark_group("response_rehydration/streaming");
    for turns in 1..=max_turns {
        eprintln!(
            "  [seed] response_rehydration/streaming  turns={turns}  prior={}",
            turns - 1
        );
        let prev_id = rt.block_on(seed_response_chain(&client, &gw_url, &model, turns - 1));
        group.bench_with_input(BenchmarkId::new("turns", turns), &turns, |b, _| {
            b.to_async(Runtime::new().unwrap()).iter_batched(
                || (prev_id.clone(), model.clone()),
                |(pid, mdl)| {
                    let client = Arc::clone(&client);
                    let url = format!("{gw_url}/v1/responses");
                    async move {
                        let mut body = serde_json::json!({"model": mdl, "input": "bench",
                            "store": true, "stream": true});
                        if let Some(id) = pid {
                            body["previous_response_id"] = serde_json::Value::String(id);
                        }
                        client
                            .post(&url)
                            .json(&body)
                            .send()
                            .await
                            .unwrap()
                            .bytes()
                            .await
                            .unwrap()
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(gateway_benches, gateway_benchmarks);
