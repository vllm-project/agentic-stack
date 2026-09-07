//! Build state, bind, serve, drain.

use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use agentic_core::config::Config;
use agentic_core::executor::ExecutionContext;

use crate::{BackendSecrets, BackendState, SecretError, build_router};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to build the execution context: {0}")]
    Context(#[from] agentic_core::error::Error),
    #[error("failed to bind or serve: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid backend secrets: {0}")]
    Secret(#[from] SecretError),
}

/// Matches the gateway's bound: a drain that never finishes would outlive the
/// pod's termination grace period and be killed mid-request anyway.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(8);

async fn drain_backend<F>(server: Pin<&mut F>) -> Result<(), Error>
where
    F: Future<Output = Result<(), std::io::Error>>,
{
    if let Ok(result) = tokio::time::timeout(DRAIN_TIMEOUT, server).await {
        result.map_err(Error::Io)
    } else {
        warn!(
            timeout_seconds = DRAIN_TIMEOUT.as_secs(),
            "drain timed out; closing remaining connections"
        );
        Ok(())
    }
}

/// Opens storage from `config` and serves the backend router until `shutdown`,
/// then drains in-flight requests for at most [`DRAIN_TIMEOUT`].
///
/// # Errors
/// If the storage pool cannot be opened, or the address cannot be bound.
pub async fn serve(
    config: &Config,
    signing_key: Vec<u8>,
    api_token: String,
    host: &str,
    port: u16,
    shutdown: CancellationToken,
) -> Result<(), Error> {
    let secrets = BackendSecrets::new(signing_key, api_token)?;
    let exec_ctx = Arc::new(ExecutionContext::from_config(config).await?);
    let listener = TcpListener::bind(format!("{host}:{port}")).await?;
    info!("agentic-llm-d listening on {host}:{port}, no proxy, no inference");
    let graceful = shutdown.clone();
    let serving = axum::serve(listener, build_router(BackendState { exec_ctx, secrets }))
        .with_graceful_shutdown(async move { graceful.cancelled().await })
        .into_future();
    tokio::pin!(serving);

    tokio::select! {
        result = &mut serving => result?,
        () = shutdown.cancelled() => {
            drain_backend(serving.as_mut()).await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentic_core::config::{PostgresConfig, SqliteConfig, ToolRuntimeConfig};

    #[tokio::test]
    async fn backend_drain_preserves_server_errors() {
        let server = std::future::ready(Err(std::io::Error::other("backend failed")));
        tokio::pin!(server);

        let error = drain_backend(server.as_mut()).await.unwrap_err();

        assert_eq!(error.to_string(), "failed to bind or serve: backend failed");
    }

    #[tokio::test]
    async fn serve_rejects_empty_secrets_before_binding() {
        let config = Config {
            llm_api_base: String::new(),
            openai_api_key: None,
            llm_ready_timeout_s: 0.0,
            llm_ready_interval_s: 0.0,
            skip_llm_ready_check: true,
            db_url: Some("sqlite://?mode=memory".to_owned()),
            postgres: PostgresConfig::default(),
            sqlite: SqliteConfig::default(),
            tools: ToolRuntimeConfig::default(),
        };
        let shutdown = CancellationToken::new();
        shutdown.cancel();

        let error = serve(&config, Vec::new(), String::new(), "127.0.0.1", 0, shutdown)
            .await
            .expect_err("empty secrets must be rejected");

        assert!(error.to_string().contains("at least 32 bytes"), "got: {error}");
    }
}
