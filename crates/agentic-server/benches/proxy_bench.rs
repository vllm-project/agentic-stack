use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::Request;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Router, serve};
use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, criterion_group};
use futures::stream;
use http::StatusCode;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;
use tokio_util::sync::CancellationToken;

use agentic_core::config::Config;
use agentic_core::executor::{ConversationHandler, ExecutionContext, ResponseHandler};
use agentic_core::proxy::ProxyState;
use agentic_core::storage::{ConversationStore, ResponseStore};
use agentic_server::app::{AppState, ServerConfig, WebSocketTracker, build_router};

const CONTENT_TYPE_JSON: &str = "application/json";
const PROMPT_SIZES: [usize; 3] = [1024, 10 * 1024, 100 * 1024];

fn bench_config(llm_url: &str) -> Config {
    Config {
        llm_api_base: llm_url.to_owned(),
        openai_api_key: Some("bench-key".to_owned()),
        llm_ready_timeout_s: 5.0,
        llm_ready_interval_s: 0.1,
        skip_llm_ready_check: false,
        db_url: None,
        postgres: agentic_core::config::PostgresConfig::default(),
        sqlite: agentic_core::config::SqliteConfig::default(),
        tools: agentic_core::config::ToolRuntimeConfig::default(),
    }
}

async fn health_handler() -> impl IntoResponse {
    StatusCode::OK
}

async fn responses_handler(req: Request) -> Response {
    let body_bytes = axum::body::to_bytes(req.into_body(), 10 * 1024 * 1024)
        .await
        .unwrap_or_default();

    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();

    if body.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false) {
        let chunks: Vec<Result<Bytes, Infallible>> = vec![
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n",
            )),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ];
        let body = Body::from_stream(stream::iter(chunks));
        return (
            StatusCode::OK,
            [("content-type", "text/event-stream; charset=utf-8")],
            body,
        )
            .into_response();
    }

    // Keep the mock response fixed so prompt-size runs isolate request upload overhead.
    let out = r#"{"id":"resp_bench","object":"response","status":"completed"}"#;
    (StatusCode::OK, [("content-type", CONTENT_TYPE_JSON)], out).into_response()
}

async fn spawn_llm() -> String {
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/responses", post(responses_handler));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve(listener, app).await.unwrap();
    });

    format!("http://{addr}")
}

async fn spawn_gateway(config: Config) -> String {
    let proxy_state = ProxyState::new(config.clone()).unwrap();
    let exec_ctx = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
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
        model_capabilities: std::sync::Arc::default(),
    };
    let server_config = ServerConfig::from_env();
    let router = build_router(state, &server_config);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve(listener, router).await.unwrap();
    });

    format!("http://{addr}")
}

fn responses_url(base_url: &str) -> String {
    format!("{base_url}/v1/responses")
}

fn request_body(prompt_bytes: usize, stream: bool) -> Bytes {
    let prompt = "x".repeat(prompt_bytes);
    let body = serde_json::json!({
        "model": "bench-model",
        "input": [{"role": "user", "content": prompt}],
        "store": false,
        "stream": stream
    });
    Bytes::from(serde_json::to_vec(&body).expect("benchmark request body serializes"))
}

async fn post_response(client: reqwest::Client, url: String, body: Bytes) -> usize {
    let resp = client
        .post(url)
        .header("content-type", CONTENT_TYPE_JSON)
        .body(body)
        .send()
        .await
        .expect("benchmark request succeeds");

    resp.bytes().await.expect("benchmark response body").len()
}

fn bench_single_request(c: &mut Criterion, rt: &Runtime, client: &reqwest::Client, llm_url: &str, gateway_url: &str) {
    let non_stream_body = request_body(5, false);
    let stream_body = request_body(5, true);

    let mut group = c.benchmark_group("non_stream");

    group.bench_function("direct", |b| {
        let url = responses_url(llm_url);
        let body = non_stream_body.clone();
        b.to_async(rt)
            .iter(|| post_response(client.clone(), url.clone(), body.clone()));
    });

    group.bench_function("proxied", |b| {
        let url = responses_url(gateway_url);
        let body = non_stream_body.clone();
        b.to_async(rt)
            .iter(|| post_response(client.clone(), url.clone(), body.clone()));
    });

    group.finish();

    let mut group = c.benchmark_group("stream");

    group.bench_function("direct", |b| {
        let url = responses_url(llm_url);
        let body = stream_body.clone();
        b.to_async(rt)
            .iter(|| post_response(client.clone(), url.clone(), body.clone()));
    });

    group.bench_function("proxied", |b| {
        let url = responses_url(gateway_url);
        let body = stream_body.clone();
        b.to_async(rt)
            .iter(|| post_response(client.clone(), url.clone(), body.clone()));
    });

    group.finish();
}

fn bench_prompt_size(c: &mut Criterion, rt: &Runtime, client: &reqwest::Client, llm_url: &str, gateway_url: &str) {
    let mut group = c.benchmark_group("non_stream/prompt_bytes");

    for prompt_bytes in PROMPT_SIZES {
        group.bench_with_input(
            BenchmarkId::new("direct", prompt_bytes),
            &prompt_bytes,
            |b, &prompt_bytes| {
                let url = responses_url(llm_url);
                let body = request_body(prompt_bytes, false);
                b.to_async(rt)
                    .iter(|| post_response(client.clone(), url.clone(), body.clone()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("proxied", prompt_bytes),
            &prompt_bytes,
            |b, &prompt_bytes| {
                let url = responses_url(gateway_url);
                let body = request_body(prompt_bytes, false);
                b.to_async(rt)
                    .iter(|| post_response(client.clone(), url.clone(), body.clone()));
            },
        );
    }

    group.finish();
}

fn proxy_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let (llm_url, gateway_url) = rt.block_on(async {
        let llm_url = spawn_llm().await;
        let config = bench_config(&llm_url);
        let gateway_url = spawn_gateway(config).await;
        (llm_url, gateway_url)
    });

    let client = reqwest::Client::new();

    bench_single_request(c, &rt, &client, &llm_url, &gateway_url);
    bench_prompt_size(c, &rt, &client, &llm_url, &gateway_url);
}

criterion_group!(proxy_benches, proxy_benchmarks);
