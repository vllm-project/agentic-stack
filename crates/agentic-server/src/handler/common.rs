use std::num::NonZeroUsize;

use axum::body::Body;
use axum::http::HeaderMap;
use axum::response::Response;
use bytes::Bytes;
use futures::StreamExt;
use http::StatusCode;
use serde::de::DeserializeOwned;
use tracing::warn;

use agentic_core::executor::{BoxStream, ExecutorError};
use agentic_core::proxy::{ProxyAuth, ProxyBody, ProxyResponse, error_response_for_auth};

/// # Panics
/// Panics if the response builder produces an invalid response (unreachable in practice).
pub fn convert_response(resp: ProxyResponse) -> Response {
    let mut builder = Response::builder().status(resp.status);
    for (name, value) in &resp.headers {
        builder = builder.header(name, value);
    }
    match resp.body {
        ProxyBody::Full(bytes) => builder.body(Body::from(bytes)).expect("valid response"),
        ProxyBody::Stream(stream) => builder.body(Body::from_stream(stream)).expect("valid response"),
    }
}

/// # Panics
/// Panics if the response builder produces an invalid response (unreachable in practice).
pub fn executor_error_response(err: ExecutorError) -> Response {
    let status = err.http_status();
    if !matches!(err, ExecutorError::LLMRequest { .. }) {
        warn!("executor error ({status}): {err}");
    }
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(err.into_response_body()))
        .expect("valid error response")
}

#[allow(clippy::result_large_err)]
pub(super) async fn read_bytes(body: Body, limit: NonZeroUsize) -> Result<Bytes, Response> {
    read_bytes_with_auth(body, ProxyAuth::OpenAiBearer, limit).await
}

#[allow(clippy::result_large_err)]
pub(super) async fn read_bytes_with_auth(body: Body, auth: ProxyAuth, limit: NonZeroUsize) -> Result<Bytes, Response> {
    axum::body::to_bytes(body, limit.get()).await.map_err(|_| {
        convert_response(error_response_for_auth(
            StatusCode::PAYLOAD_TOO_LARGE,
            "body_too_large",
            "request body too large",
            auth,
        ))
    })
}

#[allow(clippy::result_large_err)]
pub(super) async fn read_json<T: DeserializeOwned>(body: Body, limit: NonZeroUsize) -> Result<T, Response> {
    let bytes = read_bytes(body, limit).await?;
    serde_json::from_slice::<T>(&bytes).map_err(|error| executor_error_response(ExecutorError::from(error)))
}

pub(super) fn extract_store(bytes: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()
        .and_then(|j| j.get("store").and_then(serde_json::Value::as_bool))
        .unwrap_or(true)
}

pub(super) fn extract_bearer(headers: &HeaderMap, config_key: Option<&str>) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| config_key.filter(|s| !s.is_empty()).map(str::to_string))
}

pub(super) fn sse_response(stream: BoxStream) -> Response {
    sse_response_with_headers(stream, HeaderMap::new())
}

pub(super) fn sse_response_with_headers(stream: BoxStream, mut headers: HeaderMap) -> Response {
    let byte_stream = stream.map(|line| Ok::<Bytes, std::convert::Infallible>(Bytes::from(line)));
    headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    headers.insert(http::header::CACHE_CONTROL, http::HeaderValue::from_static("no-cache"));
    headers.insert("x-accel-buffering", http::HeaderValue::from_static("no"));
    let mut builder = Response::builder().status(StatusCode::OK);
    for (name, value) in &headers {
        builder = builder.header(name, value);
    }
    builder
        .body(Body::from_stream(byte_stream))
        .expect("valid SSE response")
}
