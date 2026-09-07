//! Backend mode — agentic-api as state services for an orchestrator that runs
//! inference itself (the llm-d coordinator). Nothing here proxies or calls a model.

pub mod context;
pub mod handler;
pub mod runner;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::{Router, middleware};

use agentic_core::executor::ExecutionContext;

const MIN_SECRET_LEN: usize = 32;

/// Invalid backend authentication or signing material.
#[derive(Debug, thiserror::Error)]
pub enum SecretError {
    #[error("{name} must be at least {MIN_SECRET_LEN} bytes of independently generated randomness")]
    TooShort { name: &'static str },
    #[error("the signing key and workload token must be generated independently")]
    Reused,
}

/// Validated secrets shared by the backend handlers.
#[derive(Clone)]
pub struct BackendSecrets {
    signing_key: SigningKey,
    workload_token: Arc<str>,
}

/// Validated key used to sign and verify split-execution contexts.
#[derive(Clone)]
pub struct SigningKey(Arc<[u8]>);

impl SigningKey {
    /// Validates a context-signing key.
    ///
    /// # Errors
    /// Returns [`SecretError`] when the key is shorter than 32 non-whitespace
    /// bytes.
    pub fn new(value: Vec<u8>) -> Result<Self, SecretError> {
        if non_whitespace_ascii_len(&value) < MIN_SECRET_LEN {
            return Err(SecretError::TooShort { name: "signing key" });
        }
        Ok(Self(value.into()))
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl BackendSecrets {
    /// Validates and constructs the secrets required by the split routes.
    ///
    /// # Errors
    /// Returns [`SecretError`] when either value is too short or both values are
    /// identical.
    pub fn new(signing_key: Vec<u8>, workload_token: String) -> Result<Self, SecretError> {
        let signing_key = SigningKey::new(signing_key)?;
        if non_whitespace_ascii_len(workload_token.as_bytes()) < MIN_SECRET_LEN {
            return Err(SecretError::TooShort { name: "workload token" });
        }
        if signing_key.as_bytes() == workload_token.as_bytes() {
            return Err(SecretError::Reused);
        }
        Ok(Self {
            signing_key,
            workload_token: workload_token.into(),
        })
    }

    pub(crate) fn signing_key(&self) -> &SigningKey {
        &self.signing_key
    }

    pub(crate) fn workload_token(&self) -> &str {
        &self.workload_token
    }
}

fn non_whitespace_ascii_len(value: &[u8]) -> usize {
    value.iter().filter(|byte| !byte.is_ascii_whitespace()).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signing_keys_and_workload_tokens_must_be_long_and_independent() {
        assert!(matches!(
            BackendSecrets::new(Vec::new(), "b".repeat(MIN_SECRET_LEN)),
            Err(SecretError::TooShort { name: "signing key" })
        ));
        assert!(matches!(
            BackendSecrets::new(vec![b'a'; MIN_SECRET_LEN], String::new()),
            Err(SecretError::TooShort { name: "workload token" })
        ));
        assert!(matches!(
            BackendSecrets::new(vec![b'a'; MIN_SECRET_LEN], "a".repeat(MIN_SECRET_LEN)),
            Err(SecretError::Reused)
        ));
        let sparse = format!("a{}b", " ".repeat(MIN_SECRET_LEN - 2));
        assert!(matches!(
            SigningKey::new(sparse.as_bytes().to_vec()),
            Err(SecretError::TooShort { name: "signing key" })
        ));
        assert!(matches!(
            BackendSecrets::new(vec![b'a'; MIN_SECRET_LEN], sparse),
            Err(SecretError::TooShort { name: "workload token" })
        ));
        assert!(matches!(
            SigningKey::new(vec![b'a'; MIN_SECRET_LEN - 1]),
            Err(SecretError::TooShort { name: "signing key" })
        ));
        BackendSecrets::new(vec![b'a'; MIN_SECRET_LEN], "b".repeat(MIN_SECRET_LEN)).expect("independent secrets");
    }
}

/// All the endpoints need — far less than the gateway's `AppState`.
#[derive(Clone)]
pub struct BackendState {
    pub exec_ctx: Arc<ExecutionContext>,
    /// Validated signing and workload-authentication material.
    pub secrets: BackendSecrets,
}

/// The whole surface: two split-execution endpoints and two probes.
pub fn build_router(state: BackendState) -> Router {
    // Probes stay open so an orchestrator can check liveness without the secret.
    let probes = Router::new()
        .route("/health", get(handler::health))
        .route("/ready", get(handler::ready));
    let responses = Router::new()
        .route("/v1alpha/responses/hydrate", post(handler::hydrate))
        .route("/v1alpha/responses/persist", post(handler::persist))
        .route_layer(middleware::from_fn_with_state(state.clone(), handler::require_token));
    probes.merge(responses).with_state(state)
}
