use std::sync::Arc;

use axum::Router;
use axum::response::IntoResponse;
use axum::routing::get;
use http::StatusCode;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use agentic_core::config::Config;
use agentic_core::executor::{ConversationHandler, ExecutionContext, ResponseHandler};
use agentic_core::proxy::ProxyState;
use agentic_core::readiness::llm_readiness_client;
use agentic_core::storage::{ConversationStore, ResponseStore};
use agentic_server::app::{AppState, ReadinessTracker, ServerConfig, WebSocketTracker, build_router};

pub fn test_config(llm_url: &str) -> Config {
    Config {
        llm_api_base: llm_url.to_owned(),
        openai_api_key: Some("test-key".to_owned()),
        llm_ready_timeout_s: 5.0,
        llm_ready_interval_s: 0.1,
        skip_llm_ready_check: false,
        db_url: None,
        postgres: agentic_core::config::PostgresConfig::default(),
        sqlite: agentic_core::config::SqliteConfig::default(),
        tools: agentic_core::config::ToolRuntimeConfig::default(),
    }
}

pub fn test_state(config: &Config) -> AppState {
    let exec_ctx = ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
        Arc::new(reqwest::Client::new()),
        config.llm_api_base.clone(),
    );
    let exec_ctx = Arc::new(exec_ctx);
    let proxy_state = ProxyState::new(config.clone()).expect("proxy state");
    AppState {
        proxy_state,
        exec_ctx,
        llm_readiness_client: llm_readiness_client().expect("readiness client"),
        readiness_tracker: ReadinessTracker::default(),
        shutdown_token: CancellationToken::new(),
        websocket_tracker: WebSocketTracker::default(),
        llm_api_base: config.llm_api_base.clone(),
        skip_llm_ready_check: config.skip_llm_ready_check,
        openai_api_key: config.openai_api_key.clone(),
        model_capabilities: std::sync::Arc::default(),
    }
}

/// Spawn a minimal mock LLM that responds to `GET /health` with 200.
pub async fn spawn_mock_llm() -> (String, tokio::task::JoinHandle<()>) {
    let app = Router::new().route("/health", get(|| async { StatusCode::OK.into_response() }));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), handle)
}

/// Spawn the gateway router bound to a random port.
pub async fn spawn_gateway(state: AppState) -> (String, tokio::task::JoinHandle<()>) {
    let router = build_router(state, &ServerConfig::from_env());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (format!("http://{addr}"), handle)
}
