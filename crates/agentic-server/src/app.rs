use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

use axum::Router;
use axum::middleware;
use axum::routing::{get, post};
use http::HeaderValue;
#[cfg(debug_assertions)]
use tokio::sync::oneshot;
use tokio::sync::{Notify, Semaphore, SemaphorePermit};
use tokio_util::sync::CancellationToken;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

use agentic_core::executor::ExecutionContext;
use agentic_core::proxy::ProxyState;

use crate::auth::{ANTHROPIC_COUNT_TOKENS_PATH, ANTHROPIC_MESSAGES_PATH, OidcAuthenticator, require_oidc};
use crate::handler::{
    compact_response, conversations, count_tokens, health, messages, models, ready, responses, responses_ws_with_auth,
};
use crate::model_capabilities::ModelCapabilities;

#[derive(Clone, Default)]
pub struct WebSocketTracker {
    inner: Arc<WebSocketTrackerInner>,
}

#[derive(Default)]
struct WebSocketTrackerInner {
    active: AtomicUsize,
    idle: Notify,
    #[cfg(debug_assertions)]
    local_completion_barrier: std::sync::Mutex<Option<LocalCompletionBarrier>>,
}

/// Bounds readiness work and records dependency health transitions per server.
#[derive(Clone)]
pub struct ReadinessTracker {
    inner: Arc<ReadinessTrackerInner>,
}

struct ReadinessTrackerInner {
    probe: Semaphore,
    status: AtomicU8,
}

const READINESS_UNKNOWN: u8 = 0;
const READINESS_READY: u8 = 1;
const READINESS_NOT_READY: u8 = 2;

/// Exclusive ownership of one dependency readiness probe.
pub struct ReadinessProbe<'a> {
    tracker: &'a ReadinessTracker,
    _permit: SemaphorePermit<'a>,
}

impl Default for ReadinessTracker {
    fn default() -> Self {
        Self {
            inner: Arc::new(ReadinessTrackerInner {
                probe: Semaphore::new(1),
                status: AtomicU8::new(READINESS_UNKNOWN),
            }),
        }
    }
}

impl ReadinessTracker {
    /// Start the only allowed in-flight dependency probe.
    #[must_use]
    pub fn try_start_probe(&self) -> Option<ReadinessProbe<'_>> {
        let permit = self.inner.probe.try_acquire().ok()?;
        Some(ReadinessProbe {
            tracker: self,
            _permit: permit,
        })
    }

    /// Return the last completed dependency result, if any.
    #[must_use]
    pub fn last_result(&self) -> Option<bool> {
        match self.inner.status.load(Ordering::Relaxed) {
            READINESS_READY => Some(true),
            READINESS_NOT_READY => Some(false),
            _ => None,
        }
    }
}

impl ReadinessProbe<'_> {
    /// Complete this probe and return whether dependency readiness changed.
    #[must_use]
    pub fn finish(self, ready: bool) -> bool {
        let current = if ready { READINESS_READY } else { READINESS_NOT_READY };
        self.tracker.inner.status.swap(current, Ordering::Relaxed) != current
    }
}

#[cfg(debug_assertions)]
struct LocalCompletionBarrier {
    rehydrated: oneshot::Sender<()>,
    release: oneshot::Receiver<()>,
}

pub(crate) struct WebSocketGuard {
    inner: Arc<WebSocketTrackerInner>,
}

impl WebSocketTracker {
    pub(crate) fn track(&self) -> WebSocketGuard {
        self.inner.active.fetch_add(1, Ordering::AcqRel);
        WebSocketGuard {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Wait until every upgraded WebSocket task has finished.
    pub async fn wait_until_idle(&self) {
        loop {
            let idle = self.inner.idle.notified();
            tokio::pin!(idle);
            idle.as_mut().enable();
            if self.inner.active.load(Ordering::Acquire) == 0 {
                return;
            }
            idle.await;
        }
    }

    /// Installs a one-shot test barrier after local WebSocket rehydration.
    #[cfg(debug_assertions)]
    #[doc(hidden)]
    #[must_use]
    pub fn install_local_completion_test_barrier(&self) -> (oneshot::Receiver<()>, oneshot::Sender<()>) {
        let (rehydrated_tx, rehydrated_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let barrier = LocalCompletionBarrier {
            rehydrated: rehydrated_tx,
            release: release_rx,
        };
        self.inner
            .local_completion_barrier
            .lock()
            .expect("local completion test barrier mutex poisoned")
            .replace(barrier);
        (rehydrated_rx, release_tx)
    }

    #[cfg(debug_assertions)]
    pub(crate) async fn pause_local_completion_after_rehydration(&self) {
        let barrier = self
            .inner
            .local_completion_barrier
            .lock()
            .expect("local completion test barrier mutex poisoned")
            .take();
        if let Some(barrier) = barrier {
            if barrier.rehydrated.send(()).is_ok() {
                let _ = barrier.release.await;
            }
        }
    }
}

impl Drop for WebSocketGuard {
    fn drop(&mut self) {
        if self.inner.active.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.inner.idle.notify_waiters();
        }
    }
}

/// Server-level configuration read from environment variables.
pub struct ServerConfig {
    pub cors_allowed_origins: Vec<String>,
}

impl ServerConfig {
    #[must_use]
    pub fn from_env() -> Self {
        let cors_allowed_origins = std::env::var("CORS_ALLOWED_ORIGINS")
            .ok()
            .map(|s| {
                s.split(',')
                    .map(str::trim)
                    .filter(|o| !o.is_empty())
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Self { cors_allowed_origins }
    }

    fn cors_layer(&self) -> CorsLayer {
        let allow_origin = if self.cors_allowed_origins.is_empty() {
            AllowOrigin::any()
        } else {
            let origins: Vec<HeaderValue> = self
                .cors_allowed_origins
                .iter()
                .filter_map(|o| o.parse().ok())
                .collect();
            AllowOrigin::list(origins)
        };

        CorsLayer::new()
            .allow_origin(allow_origin)
            .allow_methods(Any)
            .allow_headers(Any)
    }
}

/// Shared application state injected into every handler.
///
/// Both states are always present:
/// - `proxy_state` handles `store=false` requests (direct passthrough to vLLM)
/// - `exec_ctx` handles `store=true` requests (stateful executor with DB)
#[derive(Clone)]
pub struct AppState {
    pub proxy_state: ProxyState,
    pub exec_ctx: Arc<ExecutionContext>,
    /// Dedicated no-redirect client for inference-service health probes.
    pub llm_readiness_client: reqwest::Client,
    /// Prevents public probes from multiplying dependency work and tracks transitions.
    pub readiness_tracker: ReadinessTracker,
    /// Shared cancellation signal used to drain long-lived handlers.
    pub shutdown_token: CancellationToken,
    /// Tracks upgraded WebSocket tasks, which Axum does not await during HTTP drain.
    pub websocket_tracker: WebSocketTracker,
    /// vLLM base URL — used by the `/ready` health probe.
    pub llm_api_base: String,
    /// Whether `/ready` should omit the upstream health check.
    pub skip_llm_ready_check: bool,
    /// Server-configured API key; used as fallback when the request carries no
    /// `Authorization` header on the executor path.
    pub openai_api_key: Option<String>,
    /// Configured per-model input-modality overrides applied to the Codex model catalog.
    pub model_capabilities: Arc<ModelCapabilities>,
}

pub fn build_router(state: AppState, server_config: &ServerConfig) -> Router {
    build_router_with_auth(state, server_config, None)
}

pub fn build_router_with_auth(
    state: AppState,
    server_config: &ServerConfig,
    authenticator: Option<OidcAuthenticator>,
) -> Router {
    let public_routes = Router::new().route("/health", get(health)).route("/ready", get(ready));
    let protected_routes = Router::new()
        .route("/v1/conversations", post(conversations))
        .route("/v1/models", get(models))
        .route(ANTHROPIC_MESSAGES_PATH, post(messages))
        .route(ANTHROPIC_COUNT_TOKENS_PATH, post(count_tokens))
        .route("/v1/responses", post(responses).get(responses_ws_with_auth))
        .route("/v1/responses/compact", post(compact_response));
    let protected_routes = match authenticator {
        Some(authenticator) => {
            protected_routes.route_layer(middleware::from_fn_with_state(authenticator, require_oidc))
        }
        None => protected_routes,
    };

    public_routes
        .merge(protected_routes)
        .layer(server_config.cors_layer())
        .with_state(state)
}
