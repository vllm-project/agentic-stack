mod common;

use http::StatusCode;

use common::{spawn_gateway, spawn_mock_llm, test_config, test_state, test_state_with_max_request_body_size};

#[tokio::test]
async fn test_conversations_store_false_returns_400() {
    // Arrange
    let (llm_url, _h1) = spawn_mock_llm().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/conversations"))
        .json(&serde_json::json!({"store": false}))
        .send()
        .await
        .unwrap();

    // Assert
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "invalid_request_error");
}

#[tokio::test]
async fn test_conversations_empty_body_defaults_store_true_reaches_executor() {
    // Arrange — disabled store, so executor will error (not a 4xx)
    let (llm_url, _h1) = spawn_mock_llm().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    // Act — empty JSON body; store defaults to true
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/conversations"))
        .header("Content-Type", "application/json")
        .body("{}")
        .send()
        .await
        .unwrap();

    // Assert — reached executor path (storage disabled → 5xx, not a 4xx rejection)
    assert!(
        !resp.status().is_client_error(),
        "expected executor path, got client error"
    );
}

#[tokio::test]
async fn test_conversations_no_content_type_still_defaults_store_true() {
    // Arrange
    let (llm_url, _h1) = spawn_mock_llm().await;
    let (gw_url, _h2) = spawn_gateway(test_state(&test_config(&llm_url))).await;

    // Act — no body at all
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/conversations"))
        .send()
        .await
        .unwrap();

    // Assert — reached executor path (not a 4xx rejection)
    assert!(
        !resp.status().is_client_error(),
        "expected executor path, got client error"
    );
}

/// The request-size ceiling is one gateway-wide policy, so it also bounds the
/// small bodies this endpoint accepts.
#[tokio::test]
async fn test_conversations_honours_the_configured_request_size_limit() {
    // Arrange
    let (llm_url, _h1) = spawn_mock_llm().await;
    let limit = std::num::NonZeroUsize::new(500).expect("nonzero");
    let state = test_state_with_max_request_body_size(&test_config(&llm_url), limit);
    let (gw_url, _h2) = spawn_gateway(state).await;

    // Act
    let resp = reqwest::Client::new()
        .post(format!("{gw_url}/v1/conversations"))
        .header("Content-Type", "application/json")
        .body("x".repeat(limit.get() + 1))
        .send()
        .await
        .unwrap();

    // Assert
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}
