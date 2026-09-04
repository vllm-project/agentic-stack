#[allow(dead_code)]
mod common;

use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, HeaderValue, Response, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Extension, Json, Router, middleware};
use common::{test_config, test_state};
use futures::{SinkExt, StreamExt, stream};
use jsonwebtoken::jwk::{Jwk, KeyAlgorithm, PublicKeyUse};
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use rand::rngs::OsRng;
use rsa::RsaPrivateKey;
use rsa::pkcs1::EncodeRsaPrivateKey;
use serde::Serialize;
use serde_json::{Value, json};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::Message as TungsteniteMessage;

use agentic_server::app::{ServerConfig, build_router_with_auth};
use agentic_server::auth::{AuthenticatedPrincipal, OidcAuthError, OidcAuthenticator, OidcConfig, require_oidc};

const TEST_AUDIENCE: &str = "agentic-api";

struct TestKey {
    private_key_der: Vec<u8>,
    jwk: Jwk,
}

impl TestKey {
    fn generate() -> Self {
        let private_key = RsaPrivateKey::new(&mut OsRng, 2048).expect("generate test RSA key");
        let private_key_der = private_key.to_pkcs1_der().expect("encode test RSA key");
        let private_key_der = private_key_der.as_bytes().to_vec();
        let encoding_key = EncodingKey::from_rsa_der(&private_key_der);
        let mut jwk = Jwk::from_encoding_key(&encoding_key, Algorithm::RS256).expect("test JWK");
        jwk.common.key_algorithm = Some(KeyAlgorithm::RS256);
        jwk.common.public_key_use = Some(PublicKeyUse::Signature);
        Self { private_key_der, jwk }
    }

    fn with_id(&self, kid: &str) -> (Vec<u8>, Value) {
        let mut jwk = self.jwk.clone();
        jwk.common.key_id = Some(kid.to_owned());
        (
            self.private_key_der.clone(),
            serde_json::to_value(jwk).expect("serialize test JWK"),
        )
    }
}

static TEST_KEYS: LazyLock<[TestKey; 2]> = LazyLock::new(|| [TestKey::generate(), TestKey::generate()]);

struct TestGateway {
    address: std::net::SocketAddr,
    handle: JoinHandle<()>,
}

impl Drop for TestGateway {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

async fn spawn_gateway(authenticator: OidcAuthenticator, upstream_url: &str) -> TestGateway {
    let config = test_config(upstream_url);
    let router = build_router_with_auth(
        test_state(&config),
        &ServerConfig {
            cors_allowed_origins: Vec::new(),
            #[cfg(feature = "openapi")]
            enable_openapi_docs: false,
        },
        Some(authenticator),
    );
    spawn_router(router).await
}

async fn spawn_router(router: Router) -> TestGateway {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind test server");
    let address = listener.local_addr().expect("gateway address");
    let handle = tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve gateway");
    });
    TestGateway { address, handle }
}

async fn discover_test_authenticator(issuer: &str) -> OidcAuthenticator {
    OidcAuthenticator::discover(OidcConfig::new(issuer, TEST_AUDIENCE).expect("OIDC config"))
        .await
        .expect("OIDC discovery")
}

fn test_key() -> (Vec<u8>, Value) {
    test_key_with_id("test-key")
}

fn test_key_with_id(kid: &str) -> (Vec<u8>, Value) {
    TEST_KEYS[0].with_id(kid)
}

fn alternate_test_key_with_id(kid: &str) -> (Vec<u8>, Value) {
    TEST_KEYS[1].with_id(kid)
}

async fn spawn_rotating_oidc_provider() -> (String, Vec<u8>, Vec<u8>, std::sync::Arc<AtomicUsize>, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind rotating OIDC provider");
    let issuer = format!("http://{}", listener.local_addr().expect("OIDC provider address"));
    let discovery_issuer = issuer.clone();
    let discovery_jwks_uri = format!("{issuer}/jwks");
    let (old_private_key, old_jwk) = test_key_with_id("old-key");
    let (new_private_key, new_jwk) = alternate_test_key_with_id("new-key");
    let jwks_requests = std::sync::Arc::new(AtomicUsize::new(0));
    let observed_jwks_requests = std::sync::Arc::clone(&jwks_requests);

    let provider = Router::new()
        .route(
            "/.well-known/openid-configuration",
            get(move || {
                let issuer = discovery_issuer.clone();
                let jwks_uri = discovery_jwks_uri.clone();
                async move { Json(json!({"issuer": issuer, "jwks_uri": jwks_uri})) }
            }),
        )
        .route(
            "/jwks",
            get(move || {
                let old_jwk = old_jwk.clone();
                let new_jwk = new_jwk.clone();
                let observed_jwks_requests = std::sync::Arc::clone(&observed_jwks_requests);
                async move {
                    let request = observed_jwks_requests.fetch_add(1, Ordering::Relaxed);
                    let (cache_control, jwk) = if request == 0 {
                        ("max-age=0", old_jwk)
                    } else {
                        ("max-age=0", new_jwk)
                    };
                    (
                        [(reqwest::header::CACHE_CONTROL, cache_control)],
                        Json(json!({"keys": [jwk]})),
                    )
                }
            }),
        );
    let handle = tokio::spawn(async move {
        axum::serve(listener, provider)
            .await
            .expect("serve rotating OIDC provider");
    });

    (issuer, old_private_key, new_private_key, jwks_requests, handle)
}

async fn spawn_failing_refresh_provider() -> (String, Vec<u8>, std::sync::Arc<AtomicUsize>, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind failing OIDC provider");
    let issuer = format!("http://{}", listener.local_addr().expect("OIDC provider address"));
    let discovery_issuer = issuer.clone();
    let discovery_jwks_uri = format!("{issuer}/jwks");
    let (private_key, jwk) = test_key();
    let jwks_requests = std::sync::Arc::new(AtomicUsize::new(0));
    let observed_jwks_requests = std::sync::Arc::clone(&jwks_requests);

    let provider = Router::new()
        .route(
            "/.well-known/openid-configuration",
            get(move || {
                let issuer = discovery_issuer.clone();
                let jwks_uri = discovery_jwks_uri.clone();
                async move { Json(json!({"issuer": issuer, "jwks_uri": jwks_uri})) }
            }),
        )
        .route(
            "/jwks",
            get(move || {
                let jwk = jwk.clone();
                let observed_jwks_requests = std::sync::Arc::clone(&observed_jwks_requests);
                async move {
                    if observed_jwks_requests.fetch_add(1, Ordering::Relaxed) == 0 {
                        (
                            [(reqwest::header::CACHE_CONTROL, "max-age=0")],
                            Json(json!({"keys": [jwk]})),
                        )
                            .into_response()
                    } else {
                        StatusCode::SERVICE_UNAVAILABLE.into_response()
                    }
                }
            }),
        );
    let handle = tokio::spawn(async move {
        axum::serve(listener, provider)
            .await
            .expect("serve failing OIDC provider");
    });

    (issuer, private_key, jwks_requests, handle)
}

async fn spawn_metadata_provider(build_body: impl FnOnce(&str) -> String) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind metadata provider");
    let issuer = format!("http://{}", listener.local_addr().expect("metadata provider address"));
    let body = std::sync::Arc::new(build_body(&issuer));
    let provider = Router::new().route(
        "/.well-known/openid-configuration",
        get(move || {
            let body = std::sync::Arc::clone(&body);
            async move { ([(reqwest::header::CONTENT_TYPE, "application/json")], body.to_string()) }
        }),
    );
    let handle = tokio::spawn(async move {
        axum::serve(listener, provider).await.expect("serve metadata provider");
    });
    (issuer, handle)
}

async fn spawn_chunked_metadata_provider() -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind metadata provider");
    let issuer = format!("http://{}", listener.local_addr().expect("metadata provider address"));
    let provider = Router::new().route(
        "/.well-known/openid-configuration",
        get(|| async {
            let chunks = stream::iter([
                Ok::<_, std::convert::Infallible>(Bytes::from(vec![b' '; 768 * 1024])),
                Ok(Bytes::from(vec![b' '; 768 * 1024])),
            ]);
            Response::builder()
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(Body::from_stream(chunks))
                .expect("chunked metadata response")
        }),
    );
    let handle = tokio::spawn(async move {
        axum::serve(listener, provider).await.expect("serve metadata provider");
    });
    (issuer, handle)
}

async fn spawn_oidc_provider() -> (
    String,
    Vec<u8>,
    Vec<u8>,
    std::sync::Arc<AtomicUsize>,
    tokio::task::JoinHandle<()>,
) {
    spawn_oidc_provider_with_algorithm(KeyAlgorithm::RS256).await
}

async fn spawn_oidc_provider_with_algorithm(
    key_algorithm: KeyAlgorithm,
) -> (
    String,
    Vec<u8>,
    Vec<u8>,
    std::sync::Arc<AtomicUsize>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind OIDC provider");
    let issuer = format!("http://{}", listener.local_addr().expect("OIDC provider address"));
    let discovery_issuer = issuer.clone();
    let discovery_jwks_uri = format!("{issuer}/jwks");
    let (private_key_der, mut jwk) = test_key();
    jwk["alg"] = Value::String(
        match key_algorithm {
            KeyAlgorithm::RS256 => "RS256",
            KeyAlgorithm::RS512 => "RS512",
            _ => panic!("test provider only supports RS256 and RS512 metadata"),
        }
        .to_owned(),
    );
    let public_jwk = serde_json::to_vec(&jwk).expect("serialize public test JWK");
    let jwks_requests = std::sync::Arc::new(AtomicUsize::new(0));
    let observed_jwks_requests = std::sync::Arc::clone(&jwks_requests);

    let provider = Router::new()
        .route(
            "/.well-known/openid-configuration",
            get(move || {
                let issuer = discovery_issuer.clone();
                let jwks_uri = discovery_jwks_uri.clone();
                async move {
                    Json(json!({
                        "issuer": issuer,
                        "jwks_uri": jwks_uri
                    }))
                }
            }),
        )
        .route(
            "/jwks",
            get(move || {
                let jwk = jwk.clone();
                let observed_jwks_requests = std::sync::Arc::clone(&observed_jwks_requests);
                async move {
                    observed_jwks_requests.fetch_add(1, Ordering::Relaxed);
                    Json(json!({
                        "keys": [jwk]
                    }))
                }
            }),
        );
    let handle = tokio::spawn(async move {
        axum::serve(listener, provider).await.expect("serve OIDC provider");
    });

    (issuer, private_key_der, public_jwk, jwks_requests, handle)
}

fn identity_token(issuer: &str, audience: &str, expires_at: u64, kid: &str, private_key_der: &[u8]) -> String {
    #[derive(Serialize)]
    struct Claims<'a> {
        iss: &'a str,
        sub: &'a str,
        aud: &'a str,
        exp: u64,
    }

    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(kid.to_owned());
    encode(
        &header,
        &Claims {
            iss: issuer,
            sub: "github-user-123",
            aud: audience,
            exp: expires_at,
        },
        &EncodingKey::from_rsa_der(private_key_der),
    )
    .expect("encode test identity token")
}

fn identity_token_with_audiences(
    issuer: &str,
    audiences: &[&str],
    authorized_party: Option<&str>,
    private_key_der: &[u8],
) -> String {
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some("test-key".to_owned());
    encode(
        &header,
        &json!({
            "iss": issuer,
            "sub": "github-user-123",
            "aud": audiences,
            "azp": authorized_party,
            "exp": jsonwebtoken::get_current_timestamp() + 300
        }),
        &EncodingKey::from_rsa_der(private_key_der),
    )
    .expect("encode multi-audience identity token")
}

fn identity_token_with_authorized_party(
    issuer: &str,
    audience: &str,
    authorized_party: &str,
    private_key_der: &[u8],
) -> String {
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some("test-key".to_owned());
    encode(
        &header,
        &json!({
            "iss": issuer,
            "sub": "github-user-123",
            "aud": audience,
            "azp": authorized_party,
            "exp": jsonwebtoken::get_current_timestamp() + 300
        }),
        &EncodingKey::from_rsa_der(private_key_der),
    )
    .expect("encode identity token with authorized party")
}

fn custom_identity_token(header: &Header, claims: &Value, private_key_der: &[u8]) -> String {
    encode(header, claims, &EncodingKey::from_rsa_der(private_key_der)).expect("encode custom identity token")
}

fn hmac_identity_token(issuer: &str, audience: &str, secret: &[u8]) -> String {
    #[derive(Serialize)]
    struct Claims<'a> {
        iss: &'a str,
        sub: &'a str,
        aud: &'a str,
        exp: u64,
    }

    let mut header = Header::new(Algorithm::HS256);
    header.kid = Some("test-key".to_owned());
    encode(
        &header,
        &Claims {
            iss: issuer,
            sub: "github-user-123",
            aud: audience,
            exp: jsonwebtoken::get_current_timestamp() + 300,
        },
        &EncodingKey::from_secret(secret),
    )
    .expect("encode test HMAC identity token")
}

async fn spawn_models_upstream() -> (
    String,
    std::sync::Arc<std::sync::Mutex<Option<HeaderMap>>>,
    tokio::task::JoinHandle<()>,
) {
    let observed_headers = std::sync::Arc::new(std::sync::Mutex::new(None));
    let captured_headers = std::sync::Arc::clone(&observed_headers);
    let upstream = Router::new().route(
        "/v1/models",
        get(move |headers: HeaderMap| {
            let captured_headers = std::sync::Arc::clone(&captured_headers);
            async move {
                *captured_headers.lock().expect("capture headers") = Some(headers);
                Json(json!({"object": "list", "data": []}))
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind upstream");
    let address = listener.local_addr().expect("upstream address");
    let handle = tokio::spawn(async move {
        axum::serve(listener, upstream).await.expect("serve upstream");
    });
    (format!("http://{address}"), observed_headers, handle)
}

async fn spawn_anthropic_upstream() -> (
    String,
    std::sync::Arc<std::sync::Mutex<Option<HeaderMap>>>,
    tokio::task::JoinHandle<()>,
) {
    let observed_headers = std::sync::Arc::new(std::sync::Mutex::new(None));
    let captured_headers = std::sync::Arc::clone(&observed_headers);
    let upstream = Router::new().route(
        "/v1/messages",
        post(move |headers: HeaderMap| {
            let captured_headers = std::sync::Arc::clone(&captured_headers);
            async move {
                *captured_headers.lock().expect("capture headers") = Some(headers);
                Json(json!({
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "test-model",
                    "stop_reason": "end_turn",
                    "stop_sequence": null,
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }))
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind upstream");
    let address = listener.local_addr().expect("upstream address");
    let handle = tokio::spawn(async move {
        axum::serve(listener, upstream).await.expect("serve upstream");
    });
    (format!("http://{address}"), observed_headers, handle)
}

#[tokio::test]
async fn configured_oidc_rejects_missing_bearer_before_upstream() {
    let (issuer, _private_key, _public_jwk, _jwks_requests, _provider) = spawn_oidc_provider().await;
    let authenticator = discover_test_authenticator(&issuer).await;
    let gateway = spawn_gateway(authenticator, "http://127.0.0.1:9").await;

    let response = reqwest::get(format!("http://{}/v1/models", gateway.address))
        .await
        .expect("request gateway");

    assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
    let body = response.json::<Value>().await.expect("JSON error body");
    assert_eq!(body["error"]["code"], "missing_bearer_token");

    let health = reqwest::get(format!("http://{}/health", gateway.address))
        .await
        .expect("request health");
    assert_eq!(health.status(), reqwest::StatusCode::OK);
}

#[tokio::test]
async fn configured_oidc_rejects_invalid_bearer_before_upstream() {
    let (issuer, _private_key, _public_jwk, _jwks_requests, _provider) = spawn_oidc_provider().await;
    let authenticator = discover_test_authenticator(&issuer).await;
    let gateway = spawn_gateway(authenticator, "http://127.0.0.1:9").await;

    let response = reqwest::Client::new()
        .get(format!("http://{}/v1/models", gateway.address))
        .bearer_auth("not-a-jwt")
        .send()
        .await
        .expect("request gateway");

    assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
    let body = response.json::<Value>().await.expect("JSON error body");
    assert_eq!(body["error"]["code"], "invalid_token");

    let messages_response = reqwest::Client::new()
        .post(format!("http://{}/v1/messages", gateway.address))
        .bearer_auth("not-a-jwt")
        .send()
        .await
        .expect("request Messages API");
    assert_eq!(messages_response.status(), reqwest::StatusCode::UNAUTHORIZED);
    let messages_body = messages_response.json::<Value>().await.expect("JSON error body");
    assert_eq!(messages_body["type"], "error");
    assert_eq!(messages_body["error"]["type"], "authentication_error");
}

#[tokio::test]
async fn configured_oidc_rejects_hmac_tokens_signed_with_public_key_material() {
    let (issuer, _private_key, public_jwk, _jwks_requests, _provider) = spawn_oidc_provider().await;
    let authenticator = discover_test_authenticator(&issuer).await;
    let gateway = spawn_gateway(authenticator, "http://127.0.0.1:9").await;

    let response = reqwest::Client::new()
        .get(format!("http://{}/v1/models", gateway.address))
        .bearer_auth(hmac_identity_token(&issuer, "agentic-api", &public_jwk))
        .send()
        .await
        .expect("request gateway");

    assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
    let body = response.json::<Value>().await.expect("JSON error body");
    assert_eq!(body["error"]["code"], "invalid_token");
}

#[tokio::test]
async fn configured_oidc_rejects_token_and_jwk_algorithm_mismatch() {
    let (issuer, private_key, _public_jwk, _jwks_requests, _provider) =
        spawn_oidc_provider_with_algorithm(KeyAlgorithm::RS512).await;
    let authenticator = discover_test_authenticator(&issuer).await;
    let gateway = spawn_gateway(authenticator, "http://127.0.0.1:9").await;

    let response = reqwest::Client::new()
        .get(format!("http://{}/v1/models", gateway.address))
        .bearer_auth(identity_token(
            &issuer,
            TEST_AUDIENCE,
            jsonwebtoken::get_current_timestamp() + 300,
            "test-key",
            &private_key,
        ))
        .send()
        .await
        .expect("algorithm-mismatch request");

    assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn configured_oidc_accepts_valid_identity_and_uses_service_upstream_credential() {
    let (issuer, private_key_der, _public_jwk, _jwks_requests, _provider) = spawn_oidc_provider().await;
    let authenticator = discover_test_authenticator(&issuer).await;
    let (upstream_url, observed_headers, _upstream) = spawn_models_upstream().await;
    let gateway = spawn_gateway(authenticator, &upstream_url).await;
    let identity_token = identity_token(
        &issuer,
        "agentic-api",
        jsonwebtoken::get_current_timestamp() + 300,
        "test-key",
        &private_key_der,
    );
    let mut identity_headers = HeaderMap::new();
    identity_headers.append("x-api-key", HeaderValue::from_static("distinct-upstream-key"));

    let response = reqwest::Client::new()
        .get(format!("http://{}/v1/models", gateway.address))
        .bearer_auth(&identity_token)
        .headers(identity_headers)
        .send()
        .await
        .expect("request gateway");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let headers = observed_headers
        .lock()
        .expect("read captured headers")
        .clone()
        .expect("upstream request");
    assert_eq!(
        headers
            .get(reqwest::header::AUTHORIZATION)
            .expect("service credential")
            .to_str()
            .expect("valid authorization"),
        "Bearer test-key"
    );
    assert!(
        headers.get("x-api-key").is_none(),
        "caller-supplied OpenAI API key must not reach the upstream"
    );
}

#[tokio::test]
async fn configured_oidc_preserves_distinct_anthropic_upstream_credential() {
    let (issuer, private_key_der, _public_jwk, _jwks_requests, _provider) = spawn_oidc_provider().await;
    let authenticator = discover_test_authenticator(&issuer).await;
    let (upstream_url, observed_headers, _upstream) = spawn_anthropic_upstream().await;
    let gateway = spawn_gateway(authenticator, &upstream_url).await;
    let identity_token = identity_token(
        &issuer,
        TEST_AUDIENCE,
        jsonwebtoken::get_current_timestamp() + 300,
        "test-key",
        &private_key_der,
    );

    let response = reqwest::Client::new()
        .post(format!("http://{}/v1/messages", gateway.address))
        .bearer_auth(identity_token)
        .header("x-api-key", "distinct-upstream-key")
        .json(&json!({"model": "test-model", "max_tokens": 1, "messages": []}))
        .send()
        .await
        .expect("request gateway");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let headers = observed_headers
        .lock()
        .expect("read captured headers")
        .clone()
        .expect("upstream request");
    assert_eq!(
        headers.get("x-api-key"),
        Some(&HeaderValue::from_static("distinct-upstream-key"))
    );
    assert!(headers.get(reqwest::header::AUTHORIZATION).is_none());
}

#[tokio::test]
async fn configured_oidc_rejects_wrong_issuer_audience_and_expired_tokens() {
    let (issuer, private_key_der, _public_jwk, _jwks_requests, _provider) = spawn_oidc_provider().await;
    let authenticator = discover_test_authenticator(&issuer).await;
    let gateway = spawn_gateway(authenticator, "http://127.0.0.1:9").await;
    let now = jsonwebtoken::get_current_timestamp();
    let invalid_claims = [
        ("https://other-issuer.example", "agentic-api", now + 300),
        (issuer.as_str(), "other-audience", now + 300),
        (issuer.as_str(), "agentic-api", now - 120),
    ];

    for (token_issuer, audience, expires_at) in invalid_claims {
        let response = reqwest::Client::new()
            .get(format!("http://{}/v1/models", gateway.address))
            .bearer_auth(identity_token(
                token_issuer,
                audience,
                expires_at,
                "test-key",
                &private_key_der,
            ))
            .send()
            .await
            .expect("request gateway");

        assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
        let body = response.json::<Value>().await.expect("JSON error body");
        assert_eq!(body["error"]["code"], "invalid_token");
    }
}

#[tokio::test]
async fn configured_oidc_rejects_missing_claims_empty_subject_future_nbf_and_bad_signature() {
    let (issuer, private_key, _public_jwk, _jwks_requests, _provider) = spawn_oidc_provider().await;
    let (other_private_key, _) = alternate_test_key_with_id("test-key");
    let authenticator = discover_test_authenticator(&issuer).await;
    let gateway = spawn_gateway(authenticator, "http://127.0.0.1:9").await;
    let now = jsonwebtoken::get_current_timestamp();
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some("test-key".to_owned());
    let valid_claims = json!({
        "iss": issuer,
        "sub": "github-user-123",
        "aud": "agentic-api",
        "exp": now + 300
    });
    let mut missing_kid_header = Header::new(Algorithm::RS256);
    missing_kid_header.kid = None;
    let cases = [
        custom_identity_token(&missing_kid_header, &valid_claims, &private_key),
        custom_identity_token(
            &header,
            &json!({"iss": issuer, "sub": "", "aud": "agentic-api", "exp": now + 300}),
            &private_key,
        ),
        custom_identity_token(
            &header,
            &json!({
                "iss": issuer,
                "sub": "github-user-123",
                "aud": "agentic-api",
                "exp": now + 300,
                "nbf": now + 300
            }),
            &private_key,
        ),
        custom_identity_token(
            &header,
            &json!({"iss": issuer, "sub": "github-user-123", "aud": "agentic-api"}),
            &private_key,
        ),
        custom_identity_token(&header, &valid_claims, &other_private_key),
    ];

    for token in cases {
        let response = reqwest::Client::new()
            .get(format!("http://{}/v1/models", gateway.address))
            .bearer_auth(token)
            .send()
            .await
            .expect("invalid token request");
        assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
    }
}

#[tokio::test]
async fn unknown_key_ids_do_not_refresh_jwks_during_cooldown() {
    let (issuer, private_key_der, _public_jwk, jwks_requests, _provider) = spawn_oidc_provider().await;
    let authenticator = discover_test_authenticator(&issuer).await;
    assert_eq!(jwks_requests.load(Ordering::Relaxed), 1);
    let gateway = spawn_gateway(authenticator, "http://127.0.0.1:9").await;
    let expires_at = jsonwebtoken::get_current_timestamp() + 300;

    for kid in ["unknown-key-1", "unknown-key-2"] {
        let response = reqwest::Client::new()
            .get(format!("http://{}/v1/models", gateway.address))
            .bearer_auth(identity_token(
                &issuer,
                "agentic-api",
                expires_at,
                kid,
                &private_key_der,
            ))
            .send()
            .await
            .expect("request gateway");
        assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
    }

    assert_eq!(jwks_requests.load(Ordering::Relaxed), 1);
}

#[test]
fn oidc_configuration_enforces_secure_endpoints_and_nonempty_audience() {
    for issuer in ["https://issuer.example", "http://127.0.0.1:8080", "http://[::1]:8080"] {
        OidcConfig::new(issuer, "agentic-api").expect("accepted issuer");
    }

    for issuer in [
        "http://localhost:8080",
        "http://192.0.2.1:8080",
        "https://issuer.example?tenant=one",
        "https://issuer.example#fragment",
    ] {
        assert!(
            OidcConfig::new(issuer, "agentic-api").is_err(),
            "{issuer} must be rejected"
        );
    }
    assert!(OidcConfig::new("https://issuer.example", " \t").is_err());
}

#[tokio::test]
async fn discovery_rejects_mismatched_issuer_insecure_jwks_and_oversized_metadata() {
    let (issuer, _provider) = spawn_metadata_provider(|_| {
        json!({
            "issuer": "https://other-issuer.example",
            "jwks_uri": "https://other-issuer.example/jwks"
        })
        .to_string()
    })
    .await;
    assert!(matches!(
        OidcAuthenticator::discover(OidcConfig::new(&issuer, TEST_AUDIENCE).expect("OIDC config")).await,
        Err(OidcAuthError::IssuerMismatch { .. })
    ));

    let (issuer, _provider) = spawn_metadata_provider(|issuer| {
        json!({
            "issuer": issuer,
            "jwks_uri": "http://192.0.2.1/jwks"
        })
        .to_string()
    })
    .await;
    assert!(matches!(
        OidcAuthenticator::discover(OidcConfig::new(&issuer, TEST_AUDIENCE).expect("OIDC config")).await,
        Err(OidcAuthError::InsecureJwksUri)
    ));

    let (issuer, _provider) = spawn_metadata_provider(|_| " ".repeat(1024 * 1024 + 1)).await;
    assert!(matches!(
        OidcAuthenticator::discover(OidcConfig::new(&issuer, TEST_AUDIENCE).expect("OIDC config")).await,
        Err(OidcAuthError::ProviderResponseTooLarge)
    ));

    let (issuer, _provider) = spawn_chunked_metadata_provider().await;
    assert!(matches!(
        OidcAuthenticator::discover(OidcConfig::new(&issuer, TEST_AUDIENCE).expect("OIDC config")).await,
        Err(OidcAuthError::ProviderResponseTooLarge)
    ));
}

#[tokio::test]
async fn zero_ttl_rotated_jwks_is_coalesced_and_revokes_the_old_key() {
    let (issuer, old_private_key, new_private_key, jwks_requests, _provider) = spawn_rotating_oidc_provider().await;
    let authenticator = discover_test_authenticator(&issuer).await;
    assert_eq!(jwks_requests.load(Ordering::Relaxed), 1);
    let (upstream_url, _observed_headers, _upstream) = spawn_models_upstream().await;
    let gateway = spawn_gateway(authenticator, &upstream_url).await;
    let expires_at = jsonwebtoken::get_current_timestamp() + 300;
    let new_token = identity_token(&issuer, "agentic-api", expires_at, "new-key", &new_private_key);
    let client = reqwest::Client::new();

    let first = client
        .get(format!("http://{}/v1/models", gateway.address))
        .bearer_auth(&new_token)
        .send();
    let second = client
        .get(format!("http://{}/v1/models", gateway.address))
        .bearer_auth(&new_token)
        .send();
    let (first, second) = tokio::join!(first, second);
    assert_eq!(
        first.expect("first rotated-key request").status(),
        reqwest::StatusCode::OK
    );
    assert_eq!(
        second.expect("second rotated-key request").status(),
        reqwest::StatusCode::OK
    );
    assert_eq!(jwks_requests.load(Ordering::Relaxed), 2);

    let cached = client
        .get(format!("http://{}/v1/models", gateway.address))
        .bearer_auth(&new_token)
        .send()
        .await
        .expect("cached rotated-key request");
    assert_eq!(cached.status(), reqwest::StatusCode::OK);
    assert_eq!(jwks_requests.load(Ordering::Relaxed), 2);

    let revoked = client
        .get(format!("http://{}/v1/models", gateway.address))
        .bearer_auth(identity_token(
            &issuer,
            "agentic-api",
            expires_at,
            "old-key",
            &old_private_key,
        ))
        .send()
        .await
        .expect("revoked-key request");
    assert_eq!(revoked.status(), reqwest::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn jwks_refresh_failure_returns_protocol_specific_service_errors() {
    let (issuer, private_key, jwks_requests, _provider) = spawn_failing_refresh_provider().await;
    let authenticator = discover_test_authenticator(&issuer).await;
    let gateway = spawn_gateway(authenticator, "http://127.0.0.1:9").await;
    let token = identity_token(
        &issuer,
        "agentic-api",
        jsonwebtoken::get_current_timestamp() + 300,
        "test-key",
        &private_key,
    );
    let client = reqwest::Client::new();

    let openai_response = client
        .get(format!("http://{}/v1/models", gateway.address))
        .bearer_auth(&token)
        .send();
    let anthropic_response = client
        .post(format!("http://{}/v1/messages", gateway.address))
        .bearer_auth(&token)
        .send();
    let (response, anthropic_response) = tokio::join!(openai_response, anthropic_response);
    let response = response.expect("OpenAI-style dependency failure");
    assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
    assert!(response.headers().get(reqwest::header::WWW_AUTHENTICATE).is_none());
    let body = response.json::<Value>().await.expect("OpenAI service error");
    assert_eq!(body["error"]["code"], "authentication_service_unavailable");

    let response = anthropic_response.expect("Anthropic-style dependency failure");
    assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
    let body = response.json::<Value>().await.expect("Anthropic service error");
    assert_eq!(body["error"]["type"], "api_error");
    assert_eq!(jwks_requests.load(Ordering::Relaxed), 2);
}

#[tokio::test]
async fn tokens_require_only_trusted_audiences_and_matching_authorized_party() {
    let (issuer, private_key, _public_jwk, _jwks_requests, _provider) = spawn_oidc_provider().await;
    let authenticator = discover_test_authenticator(&issuer).await;
    let (upstream_url, _observed_headers, _upstream) = spawn_models_upstream().await;
    let gateway = spawn_gateway(authenticator, &upstream_url).await;

    for (audiences, authorized_party, expected_status) in [
        (&["agentic-api"][..], None, reqwest::StatusCode::OK),
        (&["agentic-api"][..], Some("agentic-api"), reqwest::StatusCode::OK),
        (
            &["agentic-api"][..],
            Some("other-client"),
            reqwest::StatusCode::UNAUTHORIZED,
        ),
        (
            &["agentic-api", "other-client"][..],
            Some("agentic-api"),
            reqwest::StatusCode::UNAUTHORIZED,
        ),
        (
            &["agentic-api", "other-client"][..],
            None,
            reqwest::StatusCode::UNAUTHORIZED,
        ),
    ] {
        let response = reqwest::Client::new()
            .get(format!("http://{}/v1/models", gateway.address))
            .bearer_auth(identity_token_with_audiences(
                &issuer,
                audiences,
                authorized_party,
                &private_key,
            ))
            .send()
            .await
            .expect("multi-audience request");
        assert_eq!(response.status(), expected_status);
    }

    let scalar_audience_with_conflicting_party = reqwest::Client::new()
        .get(format!("http://{}/v1/models", gateway.address))
        .bearer_auth(identity_token_with_authorized_party(
            &issuer,
            TEST_AUDIENCE,
            "other-client",
            &private_key,
        ))
        .send()
        .await
        .expect("scalar-audience request");
    assert_eq!(
        scalar_audience_with_conflicting_party.status(),
        reqwest::StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn authenticated_principal_is_inserted_and_identity_header_is_removed() {
    let (issuer, private_key, _public_jwk, _jwks_requests, _provider) = spawn_oidc_provider().await;
    let authenticator = discover_test_authenticator(&issuer).await;
    let router = Router::new()
        .route(
            "/v1/principal",
            get(
                |Extension(principal): Extension<AuthenticatedPrincipal>, headers: HeaderMap| async move {
                    Json(json!({
                        "issuer": principal.issuer(),
                        "subject": principal.subject(),
                        "authorization_present": headers.contains_key(reqwest::header::AUTHORIZATION)
                    }))
                },
            ),
        )
        .route_layer(middleware::from_fn_with_state(authenticator, require_oidc));
    let gateway = spawn_router(router).await;

    let response = reqwest::Client::new()
        .get(format!("http://{}/v1/principal", gateway.address))
        .bearer_auth(identity_token(
            &issuer,
            "agentic-api",
            jsonwebtoken::get_current_timestamp() + 300,
            "test-key",
            &private_key,
        ))
        .send()
        .await
        .expect("principal request");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body = response.json::<Value>().await.expect("principal JSON");
    assert_eq!(body["issuer"], issuer);
    assert_eq!(body["subject"], "github-user-123");
    assert_eq!(body["authorization_present"], false);
}

#[tokio::test]
async fn every_v1_route_rejects_missing_credentials() {
    let (issuer, _private_key, _public_jwk, _jwks_requests, _provider) = spawn_oidc_provider().await;
    let authenticator = discover_test_authenticator(&issuer).await;
    let gateway = spawn_gateway(authenticator, "http://127.0.0.1:9").await;
    let client = reqwest::Client::new();

    for path in [
        "/v1/conversations",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/responses",
        "/v1/responses/compact",
    ] {
        let response = client
            .post(format!("http://{}{path}", gateway.address))
            .send()
            .await
            .expect("protected POST");
        assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED, "{path}");
    }
    let models = client
        .get(format!("http://{}/v1/models", gateway.address))
        .send()
        .await
        .expect("protected models request");
    assert_eq!(models.status(), reqwest::StatusCode::UNAUTHORIZED);

    let websocket_error = tokio_tungstenite::connect_async(format!("ws://{}/v1/responses", gateway.address))
        .await
        .expect_err("missing bearer must reject WebSocket upgrade");
    match websocket_error {
        tokio_tungstenite::tungstenite::Error::Http(response) => {
            assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
        }
        error => panic!("unexpected WebSocket error: {error}"),
    }

    let ready = client
        .get(format!("http://{}/ready", gateway.address))
        .send()
        .await
        .expect("public readiness request");
    assert_ne!(ready.status(), reqwest::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn authenticated_websocket_rejects_requests_after_identity_expiry() {
    let (issuer, private_key, _public_jwk, _jwks_requests, _provider) = spawn_oidc_provider().await;
    let authenticator = discover_test_authenticator(&issuer).await;
    let gateway = spawn_gateway(authenticator, "http://127.0.0.1:9").await;
    let expires_at = jsonwebtoken::get_current_timestamp().saturating_sub(50);
    let token = identity_token(&issuer, TEST_AUDIENCE, expires_at, "test-key", &private_key);
    let mut request = format!("ws://{}/v1/responses", gateway.address)
        .into_client_request()
        .expect("WebSocket request");
    request.headers_mut().insert(
        reqwest::header::AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {token}")).expect("identity header"),
    );
    let (mut websocket, _response) = tokio_tungstenite::connect_async(request)
        .await
        .expect("authenticated WebSocket upgrade");

    while jsonwebtoken::get_current_timestamp() <= expires_at.saturating_add(60) {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    websocket
        .send(TungsteniteMessage::Text(
            json!({"type": "response.create", "response": {"input": "must not run"}})
                .to_string()
                .into(),
        ))
        .await
        .expect("send post-expiry request");
    let event = tokio::time::timeout(std::time::Duration::from_secs(2), websocket.next())
        .await
        .expect("expiry event must arrive promptly")
        .expect("expiry event")
        .expect("valid expiry frame")
        .into_text()
        .expect("text expiry frame");
    assert_eq!(
        serde_json::from_str::<Value>(&event).expect("expiry event JSON"),
        json!({
            "type": "error",
            "code": "invalid_token",
            "message": "OIDC bearer token expired",
            "param": null,
            "sequence_number": 0
        })
    );
    assert!(matches!(
        tokio::time::timeout(std::time::Duration::from_secs(2), websocket.next())
            .await
            .expect("WebSocket must close promptly after expiry"),
        Some(Ok(TungsteniteMessage::Close(_))) | None
    ));
}

#[tokio::test]
async fn anthropic_authentication_errors_include_matching_request_id() {
    let (issuer, _private_key, _public_jwk, _jwks_requests, _provider) = spawn_oidc_provider().await;
    let authenticator = discover_test_authenticator(&issuer).await;
    let gateway = spawn_gateway(authenticator, "http://127.0.0.1:9").await;

    let response = reqwest::Client::new()
        .post(format!("http://{}/v1/messages", gateway.address))
        .send()
        .await
        .expect("missing-credential Anthropic request");
    assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
    let request_id = response
        .headers()
        .get("request-id")
        .expect("Anthropic request-id header")
        .to_str()
        .expect("request ID must be ASCII")
        .to_owned();
    let body = response.json::<Value>().await.expect("Anthropic authentication error");

    assert!(request_id.starts_with("req_"));
    assert_eq!(body["request_id"], request_id);
    assert_eq!(body["error"]["type"], "authentication_error");
}
