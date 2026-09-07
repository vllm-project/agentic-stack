use std::sync::Arc;

use agentic_core::executor::{ConversationHandler, ExecutionContext, ResponseHandler};
use agentic_core::storage::{ConversationStore, ResponseStore, create_pool_with_schema};
use agentic_llm_d::{BackendSecrets, BackendState, build_router};
use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;

const API_TOKEN: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const SIGNING_KEY: &[u8] = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

async fn state() -> BackendState {
    let pool = create_pool_with_schema(Some("sqlite://?mode=memory"))
        .await
        .expect("pool");
    BackendState {
        exec_ctx: Arc::new(ExecutionContext::new(
            ConversationHandler::new(ConversationStore::new(Arc::clone(&pool))),
            ResponseHandler::new(ResponseStore::new(pool)),
            Arc::new(reqwest::Client::new()),
            "http://localhost:8000".to_owned(),
        )),
        secrets: BackendSecrets::new(SIGNING_KEY.to_vec(), API_TOKEN.to_owned()).expect("valid secrets"),
    }
}

async fn response_json(response: axum::response::Response) -> Value {
    let bytes = to_bytes(response.into_body(), usize::MAX).await.expect("body");
    serde_json::from_slice(&bytes).expect("JSON response")
}

async fn hydrate_context(router: &Router, input: &str) -> String {
    let response = router
        .clone()
        .oneshot(
            Request::post("/v1alpha/responses/hydrate")
                .header("content-type", "application/json")
                .header("x-agentic-workload-token", API_TOKEN)
                .body(Body::from(
                    json!({"model": "test-model", "input": input, "store": true}).to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("router response");
    assert_eq!(response.status(), StatusCode::OK);
    response_json(response).await["context"]
        .as_str()
        .expect("sealed context")
        .to_owned()
}

async fn assert_body_too_large(response: axum::response::Response) {
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        response_json(response).await,
        json!({
            "error": {
                "message": "request body too large",
                "type": "invalid_request_error",
                "code": "body_too_large"
            }
        })
    );
}

#[tokio::test]
async fn missing_workload_token_has_a_standard_error_envelope() {
    let response = build_router(state().await)
        .oneshot(
            Request::post("/v1alpha/responses/hydrate")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response_json(response).await,
        json!({
            "error": {
                "message": "missing or invalid workload token",
                "type": "authentication_error",
                "code": "invalid_workload_token"
            }
        })
    );
}

#[tokio::test]
async fn hydrate_rejects_a_body_that_cannot_fit_the_round_trip_budget() {
    let body = json!({
        "model": "test-model",
        "input": "x".repeat(2 * 1024 * 1024),
        "store": true
    })
    .to_string();
    let response = build_router(state().await)
        .oneshot(
            Request::post("/v1alpha/responses/hydrate")
                .header("content-type", "application/json")
                .header("x-agentic-workload-token", API_TOKEN)
                .body(Body::from(body))
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        response_json(response).await,
        json!({
            "error": {
                "message": "request body too large",
                "type": "invalid_request_error",
                "code": "body_too_large"
            }
        })
    );
}

#[tokio::test]
async fn a_near_limit_hydrate_response_can_be_persisted() {
    let router = build_router(state().await);
    let body = json!({
        "model": "test-model",
        "input": "x".repeat(1_900_000),
        "store": true
    })
    .to_string();
    let hydrated = router
        .clone()
        .oneshot(
            Request::post("/v1alpha/responses/hydrate")
                .header("content-type", "application/json")
                .header("x-agentic-workload-token", API_TOKEN)
                .body(Body::from(body))
                .expect("request"),
        )
        .await
        .expect("router response");
    assert_eq!(hydrated.status(), StatusCode::OK);
    let context = response_json(hydrated).await["context"]
        .as_str()
        .expect("sealed context")
        .to_owned();

    let persisted = router
        .oneshot(
            Request::post("/v1alpha/responses/persist")
                .header("content-type", "application/json")
                .header("x-agentic-workload-token", API_TOKEN)
                .body(Body::from(
                    json!({
                        "context": context,
                        "response": {
                            "id": "resp_upstream",
                            "object": "response",
                            "created_at": 1_700_000_000,
                            "model": "test-model",
                            "status": "completed",
                            "output": []
                        }
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(persisted.status(), StatusCode::OK);
}

#[tokio::test]
async fn persist_rejects_each_size_budget_with_a_standard_envelope() {
    let router = build_router(state().await);
    let response = router
        .clone()
        .oneshot(
            Request::post("/v1alpha/responses/persist")
                .header("content-type", "application/json")
                .header("x-agentic-workload-token", API_TOKEN)
                .body(Body::from("x".repeat(16 * 1024 * 1024 + 1)))
                .expect("request"),
        )
        .await
        .expect("router response");
    assert_body_too_large(response).await;

    for body in [
        json!({
            "context": "x".repeat(6 * 1024 * 1024 + 1),
            "response": {"id": "resp_upstream", "status": "completed", "output": []}
        })
        .to_string(),
        json!({
            "context": "valid-size-but-not-sealed",
            "sse": "x".repeat(4 * 1024 * 1024 + 1)
        })
        .to_string(),
    ] {
        let response = router
            .clone()
            .oneshot(
                Request::post("/v1alpha/responses/persist")
                    .header("content-type", "application/json")
                    .header("x-agentic-workload-token", API_TOKEN)
                    .body(Body::from(body))
                    .expect("request"),
            )
            .await
            .expect("router response");
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(response_json(response).await["error"]["code"], "body_too_large");
    }
}

#[tokio::test]
async fn duplicate_persist_has_a_standard_conflict_envelope() {
    let router = build_router(state().await);
    let context = hydrate_context(&router, "What is 2+2?").await;
    let body = json!({
        "context": context,
        "response": {
            "id": "resp_upstream",
            "object": "response",
            "created_at": 1_700_000_000,
            "model": "test-model",
            "status": "completed",
            "output": []
        }
    })
    .to_string();

    let first = router
        .clone()
        .oneshot(
            Request::post("/v1alpha/responses/persist")
                .header("content-type", "application/json")
                .header("x-agentic-workload-token", API_TOKEN)
                .body(Body::from(body.clone()))
                .expect("request"),
        )
        .await
        .expect("router response");
    assert_eq!(first.status(), StatusCode::OK);
    let response_id = response_json(first).await["id"]
        .as_str()
        .expect("stored response id")
        .to_owned();

    let duplicate = router
        .oneshot(
            Request::post("/v1alpha/responses/persist")
                .header("content-type", "application/json")
                .header("x-agentic-workload-token", API_TOKEN)
                .body(Body::from(body))
                .expect("request"),
        )
        .await
        .expect("router response");
    assert_eq!(duplicate.status(), StatusCode::CONFLICT);
    assert_eq!(
        response_json(duplicate).await,
        json!({
            "error": {
                "message": format!("conflict: a turn is already stored under '{response_id}'"),
                "type": "conflict_error",
                "code": "response_already_stored"
            }
        })
    );
}
