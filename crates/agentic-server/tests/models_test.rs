// The shared helpers serve every integration test; this one needs a models-specific upstream.
#[allow(dead_code)]
mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use agentic_server::app::AppState;
use agentic_server::model_capabilities::{CodexCatalogCapabilities, InputModalities, ModelCapabilities};
use axum::Router;
use axum::response::IntoResponse;
use axum::routing::get;
use common::{spawn_gateway, test_config, test_state};
use http::StatusCode;
use serde_json::Value;
use tokio::net::TcpListener;

/// Deliberately irregular whitespace and an unknown field, so a pass-through response can be
/// compared byte for byte against what the upstream actually sent.
const UPSTREAM_MODELS: &str = r#"{"object":"list",  "data":[
  {"id":"vision-model","max_model_len":32768,"owned_by":"mock-vllm"},
  {"id":"upstream-image-model","capabilities":["image"]},
  {"id":"pinned-text-model","capabilities":["image"]},
  {"id":"plain-model"}
]}"#;

async fn spawn_upstream_models(body: &'static str, status: StatusCode) -> (String, tokio::task::JoinHandle<()>) {
    let app = Router::new().route(
        "/v1/models",
        get(move || async move { (status, [(http::header::CONTENT_TYPE, "application/json")], body).into_response() }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let upstream = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), upstream)
}

fn state_with_overrides(llm_url: &str, overrides: &[(&str, InputModalities)]) -> AppState {
    let config = test_config(llm_url);
    let overrides = overrides
        .iter()
        .map(|(model_id, modalities)| ((*model_id).to_owned(), *modalities))
        .collect::<BTreeMap<_, _>>();
    AppState {
        model_capabilities: Arc::new(ModelCapabilities::new(overrides)),
        ..test_state(&config)
    }
}

async fn spawn_configured_gateway(
    body: &'static str,
    status: StatusCode,
    overrides: &[(&str, InputModalities)],
) -> (String, tokio::task::JoinHandle<()>, tokio::task::JoinHandle<()>) {
    let (upstream_url, upstream) = spawn_upstream_models(body, status).await;
    let (gateway_url, gateway) = spawn_gateway(state_with_overrides(&upstream_url, overrides)).await;
    (gateway_url, upstream, gateway)
}

fn modalities(catalog: &Value, slug: &str) -> Value {
    catalog["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|model| model["slug"] == slug)
        .unwrap_or_else(|| panic!("catalog must contain {slug}"))["input_modalities"]
        .clone()
}

#[tokio::test]
async fn ordinary_model_listing_is_passed_through_unchanged() {
    let (gateway_url, _upstream, _gateway) = spawn_configured_gateway(
        UPSTREAM_MODELS,
        StatusCode::OK,
        &[("vision-model", InputModalities::TextAndImage)],
    )
    .await;

    let response = reqwest::get(format!("{gateway_url}/v1/models")).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[http::header::CONTENT_TYPE], "application/json");
    assert_eq!(
        response.text().await.unwrap(),
        UPSTREAM_MODELS,
        "a request without client_version must not be transformed"
    );
}

#[tokio::test]
async fn codex_catalog_resolves_capabilities_by_precedence() {
    let (gateway_url, _upstream, _gateway) = spawn_configured_gateway(
        UPSTREAM_MODELS,
        StatusCode::OK,
        &[
            ("vision-model", InputModalities::TextAndImage),
            ("pinned-text-model", InputModalities::Text),
        ],
    )
    .await;

    let catalog: Value = reqwest::get(format!("{gateway_url}/v1/models?client_version=1.2.3"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(
        modalities(&catalog, "vision-model"),
        serde_json::json!(["text", "image"]),
        "a configured vision model advertises images without upstream metadata"
    );
    assert_eq!(
        modalities(&catalog, "upstream-image-model"),
        serde_json::json!(["text", "image"])
    );
    assert_eq!(
        modalities(&catalog, "pinned-text-model"),
        serde_json::json!(["text"]),
        "an explicit text-only override wins over upstream image metadata"
    );
    assert_eq!(modalities(&catalog, "plain-model"), serde_json::json!(["text"]));
    assert_eq!(
        catalog["models"][0]["supports_image_detail_original"],
        serde_json::json!(false)
    );
}

#[tokio::test]
async fn codex_catalog_is_text_only_without_configuration() {
    let (gateway_url, _upstream, _gateway) = spawn_configured_gateway(UPSTREAM_MODELS, StatusCode::OK, &[]).await;

    let catalog: Value = reqwest::get(format!("{gateway_url}/v1/models?client_version=1.2.3"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(modalities(&catalog, "vision-model"), serde_json::json!(["text"]));
    assert_eq!(modalities(&catalog, "plain-model"), serde_json::json!(["text"]));
}

#[tokio::test]
async fn upstream_failures_are_forwarded_without_transformation() {
    let (gateway_url, _upstream, _gateway) =
        spawn_configured_gateway(r#"{"error":"upstream exploded"}"#, StatusCode::SERVICE_UNAVAILABLE, &[]).await;

    let response = reqwest::get(format!("{gateway_url}/v1/models?client_version=1.2.3"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(response.text().await.unwrap(), r#"{"error":"upstream exploded"}"#);
}

#[tokio::test]
async fn undecodable_upstream_payload_is_reported_as_a_bad_gateway() {
    let (gateway_url, _upstream, _gateway) =
        spawn_configured_gateway(r#"{"data":"not-a-list"}"#, StatusCode::OK, &[]).await;

    let response = reqwest::get(format!("{gateway_url}/v1/models?client_version=1.2.3"))
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::BAD_GATEWAY,
        "an undecodable payload must not be served as an empty catalog"
    );
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], "upstream_unavailable");
    assert_eq!(body["error"]["message"], "invalid model list from /v1/models");
}

#[tokio::test]
async fn upstream_without_models_yields_an_empty_catalog() {
    let (gateway_url, _upstream, _gateway) = spawn_configured_gateway("{}", StatusCode::OK, &[]).await;

    let catalog: Value = reqwest::get(format!("{gateway_url}/v1/models?client_version=1.2.3"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(catalog["models"], serde_json::json!([]));
}

/// The launcher reads the served catalog through [`CodexCatalogCapabilities`], so the catalog
/// written into an isolated Codex home can only match the HTTP catalog while that view keeps
/// parsing what this handler serves.
#[tokio::test]
async fn launcher_capability_view_parses_the_served_catalog() {
    let (gateway_url, _upstream, _gateway) = spawn_configured_gateway(
        UPSTREAM_MODELS,
        StatusCode::OK,
        &[
            ("vision-model", InputModalities::TextAndImage),
            ("pinned-text-model", InputModalities::Text),
        ],
    )
    .await;

    let catalog: CodexCatalogCapabilities = reqwest::get(format!("{gateway_url}/v1/models?client_version=1.2.3"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(
        catalog.select(None).expect("first entry").slug,
        "vision-model",
        "the launcher selects the first advertised model when none is requested"
    );
    assert_eq!(
        catalog
            .select(Some("vision-model"))
            .expect("configured vision model")
            .input_modalities,
        InputModalities::TextAndImage
    );
    assert_eq!(
        catalog
            .select(Some("upstream-image-model"))
            .expect("upstream image model")
            .input_modalities,
        InputModalities::TextAndImage
    );
    assert_eq!(
        catalog
            .select(Some("pinned-text-model"))
            .expect("pinned text model")
            .input_modalities,
        InputModalities::Text
    );
    assert!(catalog.select(Some("absent-model")).is_none());
}
