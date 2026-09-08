//! Upstream error metadata must survive the non-streaming Responses executor path.
//!
//! Regression coverage for <https://github.com/vllm-project/agentic-api/issues/250>: when a
//! request is routed through the in-process executor (for example `store: true`) and the
//! upstream returns a non-2xx response, the gateway must preserve the upstream status, body,
//! and processed metadata headers (`retry-after`, request IDs, rate-limit headers) and must not
//! relabel a non-JSON error body as `application/json`.

// `common` is compiled into every test binary; this one never calls `spawn_mock_llm`,
// so silence the resulting dead-code warning (same as the other tests that skip it).
#[allow(dead_code)]
mod common;

use axum::Router;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use http::{HeaderMap, StatusCode};
use tokio::net::TcpListener;

use common::{spawn_gateway, test_config, test_state};

const NON_STREAMING_STORED_REQUEST: &str = r#"{"model":"test","input":"hi","store":true,"stream":false}"#;
const NON_STREAMING_PROXIED_REQUEST: &str = r#"{"model":"test","input":"hi","store":false,"stream":false}"#;

/// Spawn a mock upstream whose `POST /v1/responses` always answers with the given error.
async fn spawn_error_upstream(
    status: StatusCode,
    content_type: &'static str,
    body: &'static str,
    extra_headers: HeaderMap,
) -> (String, tokio::task::JoinHandle<()>) {
    let app = Router::new().route(
        "/v1/responses",
        post(move || {
            let extra_headers = extra_headers.clone();
            async move {
                let mut response = Response::builder()
                    .status(status)
                    .header("content-type", content_type)
                    .body(axum::body::Body::from(body))
                    .unwrap();
                response.headers_mut().extend(extra_headers);
                response.into_response()
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), handle)
}

fn rate_limit_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("retry-after", "7".parse().unwrap());
    headers.insert("x-request-id", "req_example".parse().unwrap());
    headers.insert("x-ratelimit-remaining-requests", "0".parse().unwrap());
    headers
}

async fn post_responses(gateway_url: &str, body: &'static str) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("{gateway_url}/v1/responses"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap()
}

#[tokio::test]
async fn executor_path_preserves_upstream_json_error_metadata() {
    let upstream_body = r#"{"error":{"message":"rate limited","type":"rate_limit_error"}}"#;
    let (llm_url, _upstream) = spawn_error_upstream(
        StatusCode::TOO_MANY_REQUESTS,
        "application/json",
        upstream_body,
        rate_limit_headers(),
    )
    .await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let response = post_responses(&gateway_url, NON_STREAMING_STORED_REQUEST).await;

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()["retry-after"], "7");
    assert_eq!(response.headers()["x-request-id"], "req_example");
    assert_eq!(response.headers()["x-ratelimit-remaining-requests"], "0");
    assert_eq!(response.headers()["content-type"], "application/json");
    assert_eq!(response.text().await.unwrap(), upstream_body);
}

#[tokio::test]
async fn executor_path_preserves_upstream_plain_text_error_content_type() {
    let upstream_body = "rate limited";
    let (llm_url, _upstream) = spawn_error_upstream(
        StatusCode::TOO_MANY_REQUESTS,
        "text/plain",
        upstream_body,
        rate_limit_headers(),
    )
    .await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let response = post_responses(&gateway_url, NON_STREAMING_STORED_REQUEST).await;

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()["retry-after"], "7");
    assert_eq!(response.headers()["x-request-id"], "req_example");
    assert_eq!(response.headers()["content-type"], "text/plain");
    assert_eq!(response.text().await.unwrap(), upstream_body);
}

#[tokio::test]
async fn executor_path_still_filters_connection_nominated_headers() {
    let mut headers = rate_limit_headers();
    headers.insert("connection", "x-upstream-hop".parse().unwrap());
    headers.insert("x-upstream-hop", "1".parse().unwrap());
    let (llm_url, _upstream) =
        spawn_error_upstream(StatusCode::TOO_MANY_REQUESTS, "text/plain", "rate limited", headers).await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let response = post_responses(&gateway_url, NON_STREAMING_STORED_REQUEST).await;

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()["retry-after"], "7");
    assert!(response.headers().get("x-upstream-hop").is_none());
}

/// Control: the stateless proxy path already preserved these headers and must keep doing so.
#[tokio::test]
async fn proxy_path_preserves_upstream_error_metadata() {
    let upstream_body = "rate limited";
    let (llm_url, _upstream) = spawn_error_upstream(
        StatusCode::TOO_MANY_REQUESTS,
        "text/plain",
        upstream_body,
        rate_limit_headers(),
    )
    .await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let response = post_responses(&gateway_url, NON_STREAMING_PROXIED_REQUEST).await;

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()["retry-after"], "7");
    assert_eq!(response.headers()["x-request-id"], "req_example");
    assert_eq!(response.headers()["content-type"], "text/plain");
    assert_eq!(response.text().await.unwrap(), upstream_body);
}

/// Control: gateway-originated errors keep their JSON envelope and content type.
#[tokio::test]
async fn malformed_client_json_still_returns_json_error_envelope() {
    let (llm_url, _upstream) = spawn_error_upstream(
        StatusCode::TOO_MANY_REQUESTS,
        "text/plain",
        "unreachable",
        rate_limit_headers(),
    )
    .await;
    let (gateway_url, _gateway) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    let response = post_responses(&gateway_url, r#"{"model":"test","input":"hi","store":true"#).await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(response.headers()["content-type"], "application/json");
    assert!(response.headers().get("retry-after").is_none());
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
}
