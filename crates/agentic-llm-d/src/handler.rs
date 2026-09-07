//! The endpoints and their axum glue. The split routes require a shared token;
//! probes do not. Keep the listener cluster-internal regardless — the token
//! authenticates the calling workload, not a tenant.

use std::time::Duration;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use tracing::warn;

use agentic_core::executor::request::RequestContext;

use agentic_core::executor::{
    ExecutorError, UpstreamBody, commit, decode_upstream, rehydrate_conversation, upstream_request,
};
use agentic_core::types::request_response::RequestPayload;

use crate::BackendState;
use crate::context::{Hydration, ensure_splittable, seal, unseal};

const MAX_HYDRATE_BODY_SIZE: usize = 2 * 1024 * 1024;
const MAX_PERSIST_BODY_SIZE: usize = 16 * 1024 * 1024;
const MAX_CONTEXT_SIZE: usize = 6 * 1024 * 1024;
const MAX_UPSTREAM_BODY_SIZE: usize = 4 * 1024 * 1024;
/// The calling workload's shared secret.
pub const WORKLOAD_TOKEN_HEADER: &str = "x-agentic-workload-token";
/// Readiness means storage answers - llm-d owns the model fleet.
const STORAGE_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Body of `POST /v1alpha/responses/persist`: the context, plus one response form.
#[derive(Debug, Deserialize)]
pub struct PersistRequest {
    context: String,
    response: Option<Box<RawValue>>,
    sse: Option<String>,
}

/// Rejects any split-route call without the shared secret. The probes are
/// layered separately and stay open.
pub async fn require_token(State(state): State<BackendState>, request: Request, next: Next) -> Response {
    // Not `Authorization`: that stays free for the end user's token.
    let presented = request
        .headers()
        .get(WORKLOAD_TOKEN_HEADER)
        .and_then(|value| value.to_str().ok());
    match presented {
        Some(token) if token_matches(token, state.secrets.workload_token()) => next.run(request).await,
        _ => api_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid_workload_token",
            "missing or invalid workload token",
        ),
    }
}

/// No early return, so a wrong token takes the same time whatever byte differs.
fn token_matches(presented: &str, expected: &str) -> bool {
    presented.len() == expected.len()
        && presented
            .bytes()
            .zip(expected.bytes())
            .fold(0_u8, |differences, (a, b)| differences | (a ^ b))
            == 0
}

pub async fn health() -> StatusCode {
    StatusCode::OK
}

pub async fn ready(State(state): State<BackendState>) -> StatusCode {
    if state.exec_ctx.storage_ready(STORAGE_PROBE_TIMEOUT).await {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

pub async fn hydrate(State(state): State<BackendState>, req: Request) -> Response {
    let payload: RequestPayload = match read_json(req.into_body(), MAX_HYDRATE_BODY_SIZE).await {
        Ok(payload) => payload,
        Err(response) => return response,
    };
    match build_hydration(payload, &state).await {
        Ok(hydration) => axum::Json(hydration).into_response(),
        Err(error) => error_response(error),
    }
}

/// Rehydrates the turn and builds the request the caller forwards to a model.
#[allow(clippy::result_large_err)] // `ExecutorError` is core's; boxing it is not ours to decide
async fn build_hydration(
    request: RequestPayload,
    state: &BackendState,
) -> agentic_core::executor::ExecutorResult<Hydration> {
    ensure_splittable(&request)?;
    let ctx = rehydrate_conversation(request, state.exec_ctx.as_ref()).await?;
    // Rehydration can restore a gateway-owned tool from the stored turn, so
    // check what will actually run.
    ensure_splittable(&ctx.enriched_request)?;
    let stream = ctx.original_request.stream;
    let request = RawValue::from_string(upstream_request(&ctx, stream)?).map_err(ExecutorError::JsonError)?;
    let context = seal(ctx.into(), state.secrets.signing_key())?;
    if context.len() > MAX_CONTEXT_SIZE {
        return Err(ExecutorError::PayloadTooLarge(
            "hydrated context exceeds the split-execution size budget".to_owned(),
        ));
    }
    Ok(Hydration { request, context })
}

pub async fn persist(State(state): State<BackendState>, req: Request) -> Response {
    let PersistRequest { context, response, sse } = match read_json(req.into_body(), MAX_PERSIST_BODY_SIZE).await {
        Ok(request) => request,
        Err(response) => return response,
    };
    // serde rejects `RawValue` in `untagged`, so "exactly one of" is checked here.
    let upstream = match (response.as_deref(), sse.as_deref()) {
        (Some(json), None) => UpstreamBody::Json(json.get()),
        (None, Some(sse)) => UpstreamBody::Sse(sse),
        _ => {
            let message = "exactly one of `response` or `sse` is required".to_owned();
            return error_response(ExecutorError::InvalidRequest(message));
        }
    };
    if context.len() > MAX_CONTEXT_SIZE {
        return error_response(ExecutorError::PayloadTooLarge(
            "sealed context exceeds the split-execution size budget".to_owned(),
        ));
    }
    let upstream_size = match upstream {
        UpstreamBody::Json(body) | UpstreamBody::Sse(body) => body.len(),
    };
    if upstream_size > MAX_UPSTREAM_BODY_SIZE {
        return error_response(ExecutorError::PayloadTooLarge(
            "upstream response exceeds the split-execution size budget".to_owned(),
        ));
    }
    let context = match unseal(&context, state.secrets.signing_key()) {
        Ok(context) => context,
        Err(error) => return error_response(error),
    };
    let ctx = RequestContext::from(context);
    let stored = match decode_upstream(&ctx, upstream) {
        Ok(payload) => commit(ctx, payload, state.exec_ctx.as_ref()).await,
        Err(error) => Err(error),
    };
    match stored {
        Ok(payload) => axum::Json(payload).into_response(),
        Err(error) => error_response(error),
    }
}

/// Renders an error with the status and envelope core defines.
fn error_response(error: ExecutorError) -> Response {
    let status = error.http_status();
    warn!("backend error ({status}): {error}");
    json(status, error.into_response_body())
}

#[allow(clippy::result_large_err)] // an axum `Response` is the idiomatic error here
async fn read_json<T: DeserializeOwned>(body: Body, limit: usize) -> Result<T, Response> {
    let bytes = axum::body::to_bytes(body, limit)
        .await
        .map_err(|_| error_response(ExecutorError::PayloadTooLarge("request body too large".to_owned())))?;
    serde_json::from_slice(&bytes).map_err(|error| error_response(ExecutorError::from(error)))
}

#[derive(Serialize)]
struct ApiErrorEnvelope<'a> {
    error: ApiErrorBody<'a>,
}

#[derive(Serialize)]
struct ApiErrorBody<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    error_type: &'a str,
    code: &'a str,
}

fn api_error(status: StatusCode, error_type: &str, code: &str, message: &str) -> Response {
    let body = serde_json::to_vec(&ApiErrorEnvelope {
        error: ApiErrorBody {
            message,
            error_type,
            code,
        },
    })
    .expect("static API error serializes");
    json(status, body)
}

fn json(status: StatusCode, body: Vec<u8>) -> Response {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(body))
        .expect("valid response")
}
