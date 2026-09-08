use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use agentic_core::config::Config;
use agentic_core::error::Error as CoreError;
use agentic_core::executor::ExecutionContext;
use agentic_core::proxy::ProxyState;
use agentic_core::readiness::{llm_readiness_client, wait_llm_ready};
use agentic_server::app::{AppState, ReadinessTracker, ServerConfig, WebSocketTracker, build_router_with_auth};
use agentic_server::auth::{OidcAuthError, OidcAuthenticator, OidcConfig};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

const GATEWAY_DRAIN_TIMEOUT: Duration = Duration::from_secs(8);

/// Transport-level gateway settings resolved before the server starts.
///
/// These are deliberately separate from [`Config`], which carries inference,
/// storage, and tool concerns that core owns.
pub struct GatewayOptions<'a> {
    pub host: &'a str,
    pub port: u16,
    /// Ceiling on serialized inbound request bytes for HTTP bodies and
    /// WebSocket messages and frames.
    pub max_request_body_size: NonZeroUsize,
    pub oidc: Option<OidcConfig>,
}

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("failed to initialize OIDC authentication: {0}")]
    Oidc(#[source] OidcAuthError),
}

impl From<OidcAuthError> for ServerError {
    fn from(error: OidcAuthError) -> Self {
        Self::Oidc(error)
    }
}

async fn build_state(
    config: &Config,
    shutdown_token: CancellationToken,
    max_request_body_size: NonZeroUsize,
) -> Result<AppState, ServerError> {
    let proxy_state = ProxyState::new(config.clone())?;
    let exec_ctx = Arc::new(ExecutionContext::from_config(config).await?);

    Ok(AppState {
        proxy_state,
        exec_ctx,
        llm_readiness_client: llm_readiness_client()?,
        readiness_tracker: ReadinessTracker::default(),
        shutdown_token,
        websocket_tracker: WebSocketTracker::default(),
        llm_api_base: config.llm_api_base.clone(),
        skip_llm_ready_check: config.skip_llm_ready_check,
        openai_api_key: config.openai_api_key.clone(),
        max_request_body_size,
    })
}

async fn serve_gateway(
    state: AppState,
    host: &str,
    port: u16,
    authenticator: Option<OidcAuthenticator>,
) -> Result<(), ServerError> {
    let addr = format!("{host}:{port}");
    let server_config = ServerConfig::from_env();
    let shutdown_token = state.shutdown_token.clone();
    let websocket_tracker = state.websocket_tracker.clone();
    let router = build_router_with_auth(state, &server_config, authenticator);
    let listener = TcpListener::bind(&addr).await?;
    info!("gateway listening on {addr}");
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            shutdown_token.cancelled().await;
        })
        .await?;
    websocket_tracker.wait_until_idle().await;
    Ok(())
}

async fn serve_gateway_until_signal(
    state: AppState,
    host: &str,
    port: u16,
    authenticator: Option<OidcAuthenticator>,
) -> Result<(), ServerError> {
    let shutdown_token = state.shutdown_token.clone();
    let gateway = serve_gateway(state, host, port, authenticator);
    tokio::pin!(gateway);

    tokio::select! {
        result = &mut gateway => result,
        signal = shutdown_signal() => {
            signal?;
            info!("shutdown signal received");
            shutdown_token.cancel();
            drain_gateway(gateway.as_mut()).await
        }
    }
}

async fn drain_gateway<F>(gateway: Pin<&mut F>) -> Result<(), ServerError>
where
    F: Future<Output = Result<(), ServerError>>,
{
    if let Ok(result) = tokio::time::timeout(GATEWAY_DRAIN_TIMEOUT, gateway).await {
        result
    } else {
        warn!(
            timeout_seconds = GATEWAY_DRAIN_TIMEOUT.as_secs(),
            "gateway drain timed out; closing remaining connections"
        );
        Ok(())
    }
}

#[cfg(unix)]
async fn shutdown_signal() -> Result<(), std::io::Error> {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;

    tokio::select! {
        signal = tokio::signal::ctrl_c() => signal,
        _ = terminate.recv() => Ok(()),
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() -> Result<(), std::io::Error> {
    tokio::signal::ctrl_c().await
}

async fn wait_until_llm_ready(config: &Config) -> Result<(), ServerError> {
    if config.skip_llm_ready_check {
        info!("skipping LLM readiness check: {}", config.llm_api_base);
        return Ok(());
    }

    wait_llm_ready(config).await?;
    info!("LLM ready: {}", config.llm_api_base);
    Ok(())
}

/// Start the gateway after the LLM becomes ready.
///
/// # Errors
///
/// Returns an error if OIDC discovery or verification-key loading, DB
/// initialisation, LLM readiness polling, or the server binding fails.
pub async fn run(config: Config, gateway: GatewayOptions<'_>) -> Result<(), ServerError> {
    let GatewayOptions {
        host,
        port,
        max_request_body_size,
        oidc,
    } = gateway;
    let authenticator = match oidc {
        Some(oidc) => Some(OidcAuthenticator::discover(oidc).await?),
        None => None,
    };
    wait_until_llm_ready(&config).await?;
    let state = build_state(&config, CancellationToken::new(), max_request_body_size).await?;
    serve_gateway_until_signal(state, host, port, authenticator).await
}

/// Spawn vLLM as a subprocess and run the gateway in the foreground.
///
/// # Errors
///
/// Returns an error if OIDC discovery or verification-key loading fails, vLLM
/// fails to start, DB initialisation fails, or the gateway errors.
pub async fn run_with_llm(
    config: Config,
    gateway: GatewayOptions<'_>,
    llm_args: Vec<String>,
) -> Result<(), ServerError> {
    let GatewayOptions {
        host,
        port,
        max_request_body_size,
        oidc,
    } = gateway;
    let authenticator = match oidc {
        Some(oidc) => Some(OidcAuthenticator::discover(oidc).await?),
        None => None,
    };
    let mut cmd = tokio::process::Command::new("python");
    cmd.arg("-m").arg("vllm.entrypoints.openai.api_server");
    cmd.args(&llm_args);

    let mut child = cmd.spawn()?;
    info!("spawned vLLM subprocess (pid {})", child.id().unwrap_or(0));

    let readiness_result = if config.skip_llm_ready_check {
        info!("skipping LLM readiness check: {}", config.llm_api_base);
        Ok(false)
    } else {
        tokio::select! {
            ready = wait_llm_ready(&config) => ready.map(|()| true).map_err(ServerError::from),
            status = child.wait() => {
                let status = status?;
                Err(ServerError::from(CoreError::LlmProcessExited { status: status.to_string() }))
            }
        }
    };

    match readiness_result {
        Ok(true) => info!("LLM ready: {}", config.llm_api_base),
        Ok(false) => {}
        Err(err) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(err);
        }
    }

    let shutdown_token = CancellationToken::new();
    let state = match build_state(&config, shutdown_token.clone(), max_request_body_size).await {
        Ok(s) => s,
        Err(err) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(err);
        }
    };

    let gateway = serve_gateway(state, host, port, authenticator);
    tokio::pin!(gateway);

    let result = tokio::select! {
        gateway = &mut gateway => gateway,
        status = child.wait() => {
            shutdown_token.cancel();
            let status = status?;
            Err(ServerError::from(CoreError::LlmProcessExited { status: status.to_string() }))
        },
        signal = shutdown_signal() => {
            match signal {
                Ok(()) => {
                    info!("shutdown signal received");
                    shutdown_token.cancel();
                    drain_gateway(gateway.as_mut()).await
                }
                Err(err) => Err(err.into()),
            }
        }
    };

    let _ = child.kill().await;
    let _ = child.wait().await;
    result
}

#[cfg(test)]
mod tests {
    use super::{ServerError, drain_gateway};
    use agentic_core::error::Error as CoreError;

    #[tokio::test(start_paused = true)]
    async fn gateway_drain_is_bounded() {
        let gateway = std::future::pending::<Result<(), ServerError>>();
        tokio::pin!(gateway);

        drain_gateway(gateway.as_mut()).await.unwrap();
    }

    #[tokio::test]
    async fn gateway_drain_preserves_server_errors() {
        let gateway = std::future::ready(Err(ServerError::from(CoreError::Config("gateway failed".to_owned()))));
        tokio::pin!(gateway);

        let error = drain_gateway(gateway.as_mut()).await.unwrap_err();

        assert_eq!(error.to_string(), "gateway failed");
    }
}
