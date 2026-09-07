//! Throughput benchmarks for the executor agentic loop (`execute`).
//!
//! Measures wall-clock time per turn across chain depths 1–N, for both
//! blocking (non-streaming) and streaming execution paths.
//!
//! | Group              | What grows with depth                              |
//! |--------------------|----------------------------------------------------|
//! | `execute/blocking` | rehydrate cost (DB reads) + JSON fetch + persist   |
//! | `execute/streaming`| rehydrate cost + SSE accumulate + persist          |
//! | `rehydrate_only`   | pure rehydrate step, no LLM call                   |
//!
//! # Configuring max depth
//!
//! Set `BENCH_MAX_DEPTH` before running to control how many depths are swept:
//!
//! ```bash
//! BENCH_MAX_DEPTH=3 cargo bench --bench executor_throughput
//! ```
//!
//! Defaults to 5 when the variable is unset.
//!
//! # Sample size
//!
//! Pass `-- --sample-size=N` (criterion flag) to override the number of
//! iterations criterion collects per benchmark:
//!
//! ```bash
//! cargo bench --bench executor_throughput -- --sample-size=20
//! ```

use std::sync::{Arc, Mutex};

use axum::Router;
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::post;
use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group};
use either::Either;
use futures::StreamExt;

use agentic_core::executor::{ConversationHandler, ExecutionContext, ResponseHandler, execute, rehydrate_conversation};
use agentic_core::storage::{ConversationStore, DbPool, ResponseStore, create_pool_with_schema};
use agentic_core::types::io::{ResponsesInput, ToolChoice};
use agentic_core::types::request_response::RequestPayload;

fn max_depth() -> usize {
    std::env::var("BENCH_MAX_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(5)
        .max(1)
}

const NON_STREAMING_BODY: &str = r#"{
  "id": "resp_bench_upstream",
  "object": "response",
  "created_at": 1700000000,
  "status": "completed",
  "model": "test-model",
  "output": [{
    "type": "message",
    "id": "msg_bench",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "OK", "annotations": []}]
  }],
  "usage": {
    "input_tokens": 5, "output_tokens": 1, "total_tokens": 6,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0}
  }
}"#;

const STREAMING_BODY: &str = concat!(
    "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_bench_upstream\",\"status\":\"in_progress\"}}\n\n",
    "data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"msg_bench\",\"type\":\"message\",\"status\":\"in_progress\",\"content\":[],\"role\":\"assistant\"}}\n\n",
    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"OK\"}\n\n",
    "data: {\"type\":\"response.completed\",\"response\":{",
    "\"id\":\"resp_bench_upstream\",\"object\":\"response\",\"created_at\":1700000000,",
    "\"status\":\"completed\",\"model\":\"test-model\",",
    "\"output\":[{\"type\":\"message\",\"id\":\"msg_bench\",\"role\":\"assistant\",",
    "\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"OK\",\"annotations\":[]}]}],",
    "\"usage\":{\"input_tokens\":5,\"output_tokens\":1,\"total_tokens\":6,",
    "\"input_tokens_details\":{\"cached_tokens\":0},",
    "\"output_tokens_details\":{\"reasoning_tokens\":0}}",
    "}}\n\n",
    "data: [DONE]\n\n",
);

fn start_mock_server(rt: &tokio::runtime::Runtime) -> String {
    let listener = rt.block_on(async { tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap() });
    let addr = listener.local_addr().unwrap();

    rt.spawn(async move {
        let app = Router::new()
            .route(
                "/v1/responses",
                post(|body: axum::body::Bytes| async move {
                    let is_stream = serde_json::from_slice::<serde_json::Value>(&body)
                        .ok()
                        .and_then(|j| j["stream"].as_bool())
                        .unwrap_or(false);

                    if is_stream {
                        axum::http::Response::builder()
                            .status(200)
                            .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
                            .body(axum::body::Body::from(STREAMING_BODY))
                            .unwrap()
                            .into_response()
                    } else {
                        axum::http::Response::builder()
                            .status(200)
                            .header(header::CONTENT_TYPE, "application/json")
                            .body(axum::body::Body::from(NON_STREAMING_BODY))
                            .unwrap()
                            .into_response()
                    }
                }),
            )
            .route(
                "/v1/conversations",
                post(|| async { (axum::http::StatusCode::OK, "{}") }),
            );
        axum::serve(listener, app).await.ok();
    });

    format!("http://{addr}")
}

fn make_request(input: &str, stream: bool, prev_id: Option<String>) -> RequestPayload {
    RequestPayload {
        model: "test-model".to_string(),
        input: ResponsesInput::Text(input.to_string()),
        instructions: None,
        previous_response_id: prev_id,
        conversation_id: None,
        tools: None,
        tool_choice: Some(ToolChoice::Auto),
        stream,
        store: true,
        include: None,
        reasoning: None,
        text: None,
        temperature: None,
        top_p: None,
        max_output_tokens: None,
        truncation: None,
        metadata: None,
        parallel_tool_calls: None,
        cache_salt: None,
        context_management: None,
    }
}

fn build_exec_ctx(rt: &tokio::runtime::Runtime, mock_url: String) -> (Arc<ExecutionContext>, Arc<DbPool>) {
    let pool = rt.block_on(async { create_pool_with_schema(None).await.expect("bench pool creation failed") });
    let conv_handler = ConversationHandler::new(ConversationStore::new(pool.clone()));
    let resp_handler = ResponseHandler::new(ResponseStore::new(pool.clone()));
    let client = Arc::new(reqwest::Client::new());
    let exec_ctx = Arc::new(ExecutionContext::new(conv_handler, resp_handler, client, mock_url));
    (exec_ctx, pool)
}

/// Delete all rows from every table so the next bench group starts with a
/// clean state.  Accumulated rows from setup closures are removed; this
/// prevents cross-contamination between groups and unbounded DB growth.
fn clear_db(rt: &tokio::runtime::Runtime, pool: &DbPool) {
    rt.block_on(async {
        sqlx::query("DELETE FROM items").execute(pool).await.ok();
        sqlx::query("DELETE FROM responses").execute(pool).await.ok();
        sqlx::query("DELETE FROM conversations").execute(pool).await.ok();
    });
}

/// Build a chain of `depth - 1` non-streaming turns and return the last
/// response ID.  Called in the setup closure — cost does NOT count toward the
/// benchmark measurement.
async fn seed_chain(exec_ctx: &Arc<ExecutionContext>, depth: usize) -> Option<String> {
    let mut prev_id: Option<String> = None;
    for i in 0..depth.saturating_sub(1) {
        let req = make_request(&format!("seed {i}"), false, prev_id.take());
        if let Either::Left(p) = execute(req, Arc::clone(exec_ctx)).await.expect("seed") {
            prev_id = Some(p.id);
        }
    }
    prev_id
}

// Bench: blocking path, depths 1–max_depth
//
// The chain of N-1 prior turns is seeded with `rt.block_on()` BEFORE criterion
// starts the measurement loop, so only turn N is timed.
fn bench_execute_blocking(c: &mut Criterion, exec_ctx: &Arc<ExecutionContext>) {
    let mut group = c.benchmark_group("execute/blocking");
    let rt = tokio::runtime::Runtime::new().unwrap();

    for depth in 1..=max_depth() {
        // Pre-seed N-1 turns outside criterion — their cost is NOT measured.
        let prev_id = rt.block_on(seed_chain(exec_ctx, depth));

        group.bench_with_input(BenchmarkId::new("turns", depth), &depth, |b, _| {
            b.to_async(tokio::runtime::Runtime::new().unwrap()).iter_batched(
                // Synchronous setup: just hand the pre-seeded prev_id to each sample.
                || prev_id.clone(),
                |prev_id| {
                    let exec_ctx = Arc::clone(exec_ctx);
                    async move {
                        let req = make_request("bench turn", false, black_box(prev_id));
                        execute(req, exec_ctx).await.expect("execute")
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// Bench: streaming path, depths 1–max_depth (same pre-seed approach).
fn bench_execute_streaming(c: &mut Criterion, exec_ctx: &Arc<ExecutionContext>) {
    let mut group = c.benchmark_group("execute/streaming");
    let rt = tokio::runtime::Runtime::new().unwrap();

    for depth in 1..=max_depth() {
        let prev_id = rt.block_on(seed_chain(exec_ctx, depth));

        group.bench_with_input(BenchmarkId::new("turns", depth), &depth, |b, _| {
            b.to_async(tokio::runtime::Runtime::new().unwrap()).iter_batched(
                || prev_id.clone(),
                |prev_id| {
                    let exec_ctx = Arc::clone(exec_ctx);
                    async move {
                        let req = make_request("bench turn", true, black_box(prev_id));
                        let result = execute(req, exec_ctx).await.expect("execute");
                        if let Either::Right(stream) = result {
                            let mut stream = Box::pin(stream);
                            while stream.next().await.is_some() {}
                        }
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_rehydrate_only(c: &mut Criterion, exec_ctx: &Arc<ExecutionContext>) {
    let mut group = c.benchmark_group("rehydrate_only");

    // Grow the shared chain incrementally so deeper depths include all prior
    // history items; the chain_tip tracks the latest response ID.
    let chain_tip: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let rt = tokio::runtime::Runtime::new().unwrap();

    for depth in 1..=max_depth() {
        // Extend the chain to `depth` turns if not already deep enough.
        rt.block_on(async {
            let has_tip = chain_tip.lock().unwrap().is_some();
            if depth == 1 || !has_tip {
                let prev_id = chain_tip.lock().unwrap().clone();
                let req = make_request("seed", false, prev_id);
                if let Either::Left(p) = execute(req, Arc::clone(exec_ctx)).await.expect("seed") {
                    *chain_tip.lock().unwrap() = Some(p.id);
                }
            }
        });

        group.bench_with_input(BenchmarkId::new("prev_response_depth", depth), &depth, |b, _| {
            b.to_async(tokio::runtime::Runtime::new().unwrap()).iter_batched(
                || chain_tip.lock().unwrap().clone(),
                |prev_id| {
                    let exec_ctx = Arc::clone(exec_ctx);
                    async move {
                        let req = make_request("bench", false, black_box(prev_id));
                        rehydrate_conversation(req, &exec_ctx).await.expect("rehydrate")
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn init_benches(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mock_url = start_mock_server(&rt);
    let (exec_ctx, pool) = build_exec_ctx(&rt, mock_url);

    bench_execute_blocking(c, &exec_ctx);
    clear_db(&rt, &pool);

    bench_execute_streaming(c, &exec_ctx);
    clear_db(&rt, &pool);

    bench_rehydrate_only(c, &exec_ctx);
    clear_db(&rt, &pool);
}

criterion_group!(executor_benches, init_benches);
