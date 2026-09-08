// Integration tests for the OpenAPI spec and Swagger UI endpoints.
#![cfg(feature = "openapi")]

mod common;

use agentic_server::app::{AppState, ServerConfig, build_router};
use common::{spawn_mock_llm, test_config, test_state};
use tokio::net::TcpListener;

async fn spawn_gateway_with_docs(state: AppState) -> (String, tokio::task::JoinHandle<()>) {
    let config = ServerConfig {
        enable_openapi_docs: true,
        ..ServerConfig::from_env()
    };
    let router = build_router(state, &config);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (format!("http://{addr}"), handle)
}

async fn fetch_spec() -> serde_json::Value {
    let (llm_url, _h1) = spawn_mock_llm().await;
    let (gw_url, _h2) = spawn_gateway_with_docs(test_state(&test_config(&llm_url))).await;

    let resp = reqwest::get(format!("{gw_url}/openapi.json")).await.unwrap();
    assert_eq!(resp.status(), 200);
    resp.json().await.unwrap()
}

#[tokio::test]
async fn openapi_json_returns_valid_spec() {
    let body = fetch_spec().await;

    assert!(
        body["openapi"].as_str().unwrap().starts_with("3."),
        "must be OpenAPI 3.x"
    );
    assert_eq!(body["info"]["title"], "vLLM Agentic API");

    let paths = body["paths"].as_object().expect("paths must be an object");
    for expected in [
        "/health",
        "/ready",
        "/v1/models",
        "/v1/responses",
        "/v1/responses/compact",
        "/v1/conversations",
        "/v1/messages",
        "/v1/messages/count_tokens",
    ] {
        assert!(paths.contains_key(expected), "missing path: {expected}");
    }

    let schemas = body["components"]["schemas"]
        .as_object()
        .expect("schemas must be an object");
    assert!(!schemas.is_empty(), "schemas must not be empty");
    for expected in ["RequestPayload", "ResponsePayload", "MessagesRequest"] {
        assert!(schemas.contains_key(expected), "missing schema: {expected}");
    }

    let security_schemes = body["components"]["securitySchemes"]
        .as_object()
        .expect("securitySchemes must be an object");
    assert!(
        security_schemes.contains_key("bearer_auth"),
        "bearer_auth scheme missing"
    );
}

#[tokio::test]
async fn streaming_endpoints_declare_dual_media_types() {
    let body = fetch_spec().await;

    for path in ["/v1/responses", "/v1/messages"] {
        let response_200 = &body["paths"][path]["post"]["responses"]["200"]["content"];
        let media_types: Vec<&str> = response_200
            .as_object()
            .unwrap_or_else(|| panic!("{path} 200 must have content"))
            .keys()
            .map(String::as_str)
            .collect();
        assert!(
            media_types.contains(&"application/json"),
            "{path} 200 missing application/json, got: {media_types:?}"
        );
        assert!(
            media_types.contains(&"text/event-stream"),
            "{path} 200 missing text/event-stream, got: {media_types:?}"
        );
    }
}

#[tokio::test]
async fn error_envelopes_match_api_style() {
    let body = fetch_spec().await;

    for path in ["/v1/messages", "/v1/messages/count_tokens"] {
        let err_400 = &body["paths"][path]["post"]["responses"]["400"];
        let err_ref = err_400["content"]["application/json"]["schema"]["$ref"]
            .as_str()
            .unwrap_or_else(|| panic!("{path} 400 must reference a schema"));
        assert!(
            err_ref.contains("AnthropicErrorResponse"),
            "{path} 400 should use AnthropicErrorResponse, got: {err_ref}"
        );
    }

    for path in ["/v1/responses", "/v1/responses/compact"] {
        let err_400 = &body["paths"][path]["post"]["responses"]["400"];
        let err_ref = err_400["content"]["application/json"]["schema"]["$ref"]
            .as_str()
            .unwrap_or_else(|| panic!("{path} 400 must reference a schema"));
        assert!(
            err_ref.contains("ApiErrorResponse"),
            "{path} 400 should use ApiErrorResponse, got: {err_ref}"
        );
    }
}

#[tokio::test]
async fn tagged_enum_discriminators_present() {
    let body = fetch_spec().await;
    let schemas = body["components"]["schemas"]
        .as_object()
        .expect("schemas must be an object");

    let responses_tool = &schemas["ResponsesTool"];
    let one_of = responses_tool["oneOf"].as_array().expect("ResponsesTool must be oneOf");
    let type_values: Vec<&str> = one_of
        .iter()
        .filter_map(|variant| variant["allOf"][0]["properties"]["type"]["enum"][0].as_str())
        .collect();
    for expected in ["function", "mcp", "web_search_preview", "custom"] {
        assert!(
            type_values.contains(&expected),
            "ResponsesTool missing discriminator value '{expected}', got: {type_values:?}"
        );
    }

    let input_content = &schemas["InputContent"];
    let one_of = input_content["oneOf"].as_array().expect("InputContent must be oneOf");
    let type_values: Vec<&str> = one_of
        .iter()
        .filter_map(|variant| variant["properties"]["type"]["enum"][0].as_str())
        .collect();
    for expected in ["input_text", "input_image", "output_text", "reasoning_text"] {
        assert!(
            type_values.contains(&expected),
            "InputContent missing discriminator value '{expected}', got: {type_values:?}"
        );
    }

    let input_item = &schemas["InputItem"];
    let one_of = input_item["oneOf"].as_array().expect("InputItem must be oneOf");
    let type_values: Vec<&str> = one_of
        .iter()
        .filter_map(|variant| {
            variant["properties"]["type"]["enum"][0]
                .as_str()
                .or_else(|| variant["allOf"][0]["properties"]["type"]["enum"][0].as_str())
        })
        .collect();
    assert!(
        type_values.contains(&"compaction_trigger"),
        "InputItem missing 'compaction_trigger', got: {type_values:?}"
    );
    assert!(
        type_values.contains(&"message"),
        "InputItem missing 'message', got: {type_values:?}"
    );
}

#[tokio::test]
async fn swagger_ui_returns_html() {
    let (llm_url, _h1) = spawn_mock_llm().await;
    let (gw_url, _h2) = spawn_gateway_with_docs(test_state(&test_config(&llm_url))).await;

    let resp = reqwest::get(format!("{gw_url}/swagger-ui/")).await.unwrap();
    assert_eq!(resp.status(), 200);

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.contains("text/html"),
        "swagger-ui should return HTML, got: {content_type}"
    );
}
