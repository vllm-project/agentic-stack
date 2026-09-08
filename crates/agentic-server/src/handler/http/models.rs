use std::future::Future;

use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use http::StatusCode;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use agentic_core::proxy::{ProxyBody, ProxyResponse, error_response, proxy_get};
use agentic_core::readiness::{LLM_READINESS_PROBE_TIMEOUT, LlmReadiness, probe_llm_readiness};

use super::super::common::convert_response;
use crate::app::AppState;
use crate::model_capabilities::{InputModalities, ModelCapabilities, UpstreamCapabilities};

/// One model entry of an upstream OpenAI-compatible `/v1/models` payload.
///
/// Every field is optional: the payload comes from a third-party inference service whose
/// schema the gateway does not control, and an entry that omits metadata must degrade to the
/// conservative defaults rather than drop the model.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct UpstreamModel {
    id: Option<String>,
    name: Option<String>,
    /// vLLM reports the context window here.
    max_model_len: Option<i64>,
    /// Other OpenAI-compatible providers report the context window here.
    context_length: Option<i64>,
    /// Vendor-defined capability strings; only recognized values are honored.
    capabilities: Option<Vec<String>>,
}

/// An upstream OpenAI-compatible `/v1/models` payload.
#[derive(Debug, Default, Deserialize)]
struct UpstreamModelList {
    #[serde(default)]
    data: Vec<UpstreamModel>,
}

/// Reasoning effort levels Codex offers for a model.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum ReasoningEffort {
    Low,
    Medium,
    High,
}

/// One selectable reasoning level with its Codex-facing description.
#[derive(Debug, Serialize)]
struct SupportedReasoningLevel {
    effort: ReasoningEffort,
    description: &'static str,
}

/// How Codex truncates oversized tool output for a model.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum TruncationMode {
    Bytes,
}

/// The truncation policy advertised to Codex.
#[derive(Debug, Serialize)]
struct TruncationPolicy {
    mode: TruncationMode,
    limit: u32,
}

/// A Codex `ModelInfo` entry.
///
/// [`Default`] carries every field that is identical for all served models; the per-model
/// values are set by [`upstream_model_to_codex`].
#[derive(Debug, Serialize)]
// The boolean flags mirror Codex's `ModelInfo` schema; grouping them would change the wire shape.
#[allow(clippy::struct_excessive_bools)]
struct CodexModelInfo {
    slug: String,
    display_name: String,
    auto_review_model_override: String,
    supported_in_api: bool,
    priority: u8,
    shell_type: &'static str,
    visibility: &'static str,
    base_instructions: &'static str,
    supported_reasoning_levels: Vec<SupportedReasoningLevel>,
    supports_reasoning_summaries: bool,
    default_reasoning_summary: &'static str,
    support_verbosity: bool,
    /// Always null: the gateway does not advertise a verbosity default.
    default_verbosity: Option<String>,
    apply_patch_tool_type: &'static str,
    web_search_tool_type: &'static str,
    truncation_policy: TruncationPolicy,
    supports_parallel_tool_calls: bool,
    /// Image detail hints are never advertised; the gateway does not relay them upstream.
    supports_image_detail_original: bool,
    input_modalities: InputModalities,
    effective_context_window_percent: u8,
    experimental_supported_tools: Vec<String>,
    supports_search_tool: bool,
    use_responses_lite: bool,
    /// Always null: the gateway does not pin a Codex tool mode.
    tool_mode: Option<String>,
    /// Always null: the gateway does not advertise a multi-agent version.
    multi_agent_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_window: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_context_window: Option<i64>,
}

impl Default for CodexModelInfo {
    fn default() -> Self {
        Self {
            slug: String::new(),
            display_name: String::new(),
            auto_review_model_override: String::new(),
            supported_in_api: true,
            priority: 1,
            shell_type: "shell_command",
            visibility: "list",
            base_instructions: "",
            supported_reasoning_levels: vec![
                SupportedReasoningLevel {
                    effort: ReasoningEffort::Low,
                    description: "Fast responses with lighter reasoning",
                },
                SupportedReasoningLevel {
                    effort: ReasoningEffort::Medium,
                    description: "Balances speed and reasoning depth",
                },
                SupportedReasoningLevel {
                    effort: ReasoningEffort::High,
                    description: "Greater reasoning depth for complex problems",
                },
            ],
            supports_reasoning_summaries: false,
            default_reasoning_summary: "auto",
            support_verbosity: false,
            default_verbosity: None,
            apply_patch_tool_type: "freeform",
            web_search_tool_type: "text",
            truncation_policy: TruncationPolicy {
                mode: TruncationMode::Bytes,
                limit: 100_000,
            },
            supports_parallel_tool_calls: true,
            supports_image_detail_original: false,
            input_modalities: InputModalities::Text,
            effective_context_window_percent: 95,
            experimental_supported_tools: Vec::new(),
            supports_search_tool: false,
            use_responses_lite: false,
            tool_mode: None,
            multi_agent_version: None,
            context_window: None,
            max_context_window: None,
        }
    }
}

/// The Codex `ModelsResponse` returned to a Codex client.
#[derive(Debug, Serialize)]
struct CodexModelsResponse {
    models: Vec<CodexModelInfo>,
}

/// Transform a single upstream model entry into a Codex `ModelInfo` entry.
///
/// Returns `None` when the entry has no `id` field (malformed upstream data).
fn upstream_model_to_codex(model: &UpstreamModel, capabilities: &ModelCapabilities) -> Option<CodexModelInfo> {
    let id = model.id.as_deref()?;
    let context_window = model.max_model_len.or(model.context_length);
    let upstream = UpstreamCapabilities::from_advertised(model.capabilities.as_deref().unwrap_or_default());

    Some(CodexModelInfo {
        slug: id.to_owned(),
        display_name: model.name.as_deref().unwrap_or(id).to_owned(),
        auto_review_model_override: id.to_owned(),
        supports_reasoning_summaries: upstream.reasoning,
        input_modalities: capabilities.resolve(id, upstream),
        context_window,
        max_context_window: context_window,
        ..CodexModelInfo::default()
    })
}

/// Build the Codex `ModelsResponse` from a raw upstream vLLM models payload.
///
/// # Errors
///
/// Returns the deserialization error when the upstream payload is not a model list. Reporting
/// it keeps an undecodable upstream response from being served as an empty catalog, which
/// Codex would show as a gateway that serves no models.
fn build_codex_models_response(
    upstream_bytes: &[u8],
    capabilities: &ModelCapabilities,
) -> Result<CodexModelsResponse, serde_json::Error> {
    let upstream: UpstreamModelList = serde_json::from_slice(upstream_bytes)?;

    Ok(CodexModelsResponse {
        models: upstream
            .data
            .iter()
            .filter_map(|model| upstream_model_to_codex(model, capabilities))
            .collect(),
    })
}

pub async fn health() -> impl IntoResponse {
    StatusCode::OK
}

async fn upstream_is_ready(state: &AppState) -> bool {
    match probe_llm_readiness(
        &state.llm_readiness_client,
        &state.llm_api_base,
        state.openai_api_key.as_deref(),
        LLM_READINESS_PROBE_TIMEOUT,
    )
    .await
    {
        Ok(LlmReadiness::Ready) => true,
        Ok(LlmReadiness::Rejected(status)) => {
            debug!(http.status = %status, "LLM backend not ready");
            false
        }
        Ok(LlmReadiness::Unreachable(error)) => {
            debug!(error = ?error, "LLM backend unreachable");
            false
        }
        Ok(LlmReadiness::TimedOut) => {
            debug!("LLM backend readiness check timed out");
            false
        }
        Ok(_) => {
            debug!("LLM backend returned an unsupported readiness state");
            false
        }
        Err(error) => {
            debug!(error = ?error, "LLM backend readiness configuration invalid");
            false
        }
    }
}

async fn configured_upstream_is_ready(state: &AppState) -> bool {
    state.skip_llm_ready_check || upstream_is_ready(state).await
}

async fn dependencies_are_ready(
    storage_ready: impl Future<Output = bool>,
    upstream_ready: impl Future<Output = bool>,
) -> bool {
    tokio::pin!(storage_ready, upstream_ready);

    tokio::select! {
        storage_ready = &mut storage_ready => {
            if storage_ready {
                upstream_ready.await
            } else {
                debug!("database persistence not ready");
                false
            }
        }
        upstream_ready = &mut upstream_ready => {
            if upstream_ready {
                let storage_ready = storage_ready.await;
                if !storage_ready {
                    debug!("database persistence not ready");
                }
                storage_ready
            } else {
                false
            }
        }
    }
}

pub async fn ready(State(state): State<AppState>) -> impl IntoResponse {
    let Some(probe) = state.readiness_tracker.try_start_probe() else {
        let cached_ready = state.readiness_tracker.last_result().unwrap_or(false);
        debug!(
            readiness.ready = cached_ready,
            "returning cached readiness while dependency probe is in progress"
        );
        return if cached_ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        };
    };
    let dependencies_ready = dependencies_are_ready(
        state.exec_ctx.storage_ready(std::time::Duration::from_secs(1)),
        configured_upstream_is_ready(&state),
    )
    .await;

    if probe.finish(dependencies_ready) {
        if dependencies_ready {
            info!(readiness.ready = true, "gateway dependencies ready");
        } else {
            warn!(readiness.ready = false, "gateway dependencies not ready");
        }
    }

    if dependencies_ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

/// Query parameters for GET /v1/models.
///
/// Codex CLI appends `?client_version=<ver>` to identify itself; its presence
/// triggers transformation to the Codex `ModelsResponse` shape.
#[derive(serde::Deserialize)]
pub struct ModelsParams {
    client_version: Option<String>,
}

/// GET /v1/models — Codex-aware model list.
///
/// When `?client_version` is present (Codex CLI), fetches vLLM's model list via
/// [`proxy_get`] and transforms it into the Codex `ModelsResponse` shape
/// (`{ "models": [...] }` with rich metadata). Without `client_version`, the
/// upstream response is returned unchanged via [`proxy_get`].
pub async fn models(State(state): State<AppState>, headers: HeaderMap, Query(params): Query<ModelsParams>) -> Response {
    let upstream = proxy_get("/v1/models", &headers, &state.proxy_state).await;

    if params.client_version.is_none() {
        return convert_response(upstream);
    }

    let ProxyBody::Full(upstream_bytes) = upstream.body else {
        return convert_response(error_response(
            StatusCode::BAD_GATEWAY,
            "upstream_unavailable",
            "unexpected streaming response from /v1/models",
        ));
    };

    if !upstream.status.is_success() {
        return convert_response(ProxyResponse {
            body: ProxyBody::Full(upstream_bytes),
            ..upstream
        });
    }

    match build_codex_models_response(&upstream_bytes, &state.model_capabilities) {
        Ok(response) => axum::Json(response).into_response(),
        Err(error) => {
            warn!(error = %error, "upstream /v1/models payload could not be decoded");
            convert_response(error_response(
                StatusCode::BAD_GATEWAY,
                "upstream_unavailable",
                "invalid model list from /v1/models",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::future;
    use std::time::Duration;

    use serde_json::{Value, json};

    use super::{build_codex_models_response, dependencies_are_ready};
    use crate::app::ReadinessTracker;
    use crate::model_capabilities::{InputModalities, ModelCapabilities};

    const UPSTREAM_MODELS: &str = r#"{
        "object": "list",
        "data": [
            {"id": "vision-model", "max_model_len": 32768},
            {"id": "upstream-image-model", "capabilities": ["image", "reasoning"]},
            {"id": "pinned-text-model", "capabilities": ["image"]},
            {"id": "plain-model", "name": "Plain Model", "context_length": 8192}
        ]
    }"#;

    fn configured_capabilities() -> ModelCapabilities {
        ModelCapabilities::new(BTreeMap::from([
            ("vision-model".to_owned(), InputModalities::TextAndImage),
            ("pinned-text-model".to_owned(), InputModalities::Text),
        ]))
    }

    fn catalog(payload: &str, capabilities: &ModelCapabilities) -> Vec<Value> {
        let response = build_codex_models_response(payload.as_bytes(), capabilities).expect("decodable payload");
        let serialized = serde_json::to_value(&response).expect("serialize catalog");
        serialized["models"].as_array().cloned().expect("models array")
    }

    fn entry(models: &[Value], slug: &str) -> Value {
        models
            .iter()
            .find(|model| model["slug"] == slug)
            .unwrap_or_else(|| panic!("catalog must contain {slug}"))
            .clone()
    }

    #[test]
    fn catalog_resolves_modalities_by_configured_precedence() {
        let models = catalog(UPSTREAM_MODELS, &configured_capabilities());

        assert_eq!(
            entry(&models, "vision-model")["input_modalities"],
            json!(["text", "image"]),
            "a configured vision model must advertise images without upstream metadata"
        );
        assert_eq!(
            entry(&models, "upstream-image-model")["input_modalities"],
            json!(["text", "image"]),
            "recognized upstream metadata must be honored without an override"
        );
        assert_eq!(
            entry(&models, "pinned-text-model")["input_modalities"],
            json!(["text"]),
            "an explicit text-only override must win over upstream image metadata"
        );
        assert_eq!(
            entry(&models, "plain-model")["input_modalities"],
            json!(["text"]),
            "an unknown model without metadata must stay text-only"
        );
    }

    #[test]
    fn catalog_stays_text_only_without_configuration() {
        let models = catalog(UPSTREAM_MODELS, &ModelCapabilities::default());

        assert_eq!(
            entry(&models, "vision-model")["input_modalities"],
            json!(["text"]),
            "without an override a model with no upstream metadata stays text-only"
        );
        assert_eq!(
            entry(&models, "upstream-image-model")["input_modalities"],
            json!(["text", "image"]),
            "upstream image metadata is honored on its own"
        );
        assert_eq!(
            entry(&models, "pinned-text-model")["input_modalities"],
            json!(["text", "image"]),
            "the text-only pin comes from configuration, not from upstream metadata"
        );
        assert_eq!(entry(&models, "plain-model")["input_modalities"], json!(["text"]));
    }

    #[test]
    fn catalog_entries_keep_their_static_codex_settings() {
        let models = catalog(UPSTREAM_MODELS, &configured_capabilities());
        let model = entry(&models, "vision-model");

        assert_eq!(model["supports_image_detail_original"], json!(false));
        assert_eq!(model["shell_type"], json!("shell_command"));
        assert_eq!(model["apply_patch_tool_type"], json!("freeform"));
        assert_eq!(model["web_search_tool_type"], json!("text"));
        assert_eq!(model["truncation_policy"], json!({"mode": "bytes", "limit": 100_000}));
        assert_eq!(model["effective_context_window_percent"], json!(95));
        assert_eq!(model["supported_in_api"], json!(true));
        assert_eq!(model["visibility"], json!("list"));
        assert_eq!(model["default_verbosity"], Value::Null);
        assert_eq!(model["tool_mode"], Value::Null);
        assert_eq!(model["multi_agent_version"], Value::Null);
        assert_eq!(
            model["supported_reasoning_levels"],
            json!([
                {"effort": "low", "description": "Fast responses with lighter reasoning"},
                {"effort": "medium", "description": "Balances speed and reasoning depth"},
                {"effort": "high", "description": "Greater reasoning depth for complex problems"}
            ])
        );
    }

    #[test]
    fn catalog_entries_keep_the_codex_key_set() {
        let models = catalog(UPSTREAM_MODELS, &configured_capabilities());
        let model = entry(&models, "vision-model");
        let mut keys: Vec<&str> = model
            .as_object()
            .expect("catalog entry is an object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();

        assert_eq!(
            keys,
            [
                "apply_patch_tool_type",
                "auto_review_model_override",
                "base_instructions",
                "context_window",
                "default_reasoning_summary",
                "default_verbosity",
                "display_name",
                "effective_context_window_percent",
                "experimental_supported_tools",
                "input_modalities",
                "max_context_window",
                "multi_agent_version",
                "priority",
                "shell_type",
                "slug",
                "support_verbosity",
                "supported_in_api",
                "supported_reasoning_levels",
                "supports_image_detail_original",
                "supports_parallel_tool_calls",
                "supports_reasoning_summaries",
                "supports_search_tool",
                "tool_mode",
                "truncation_policy",
                "use_responses_lite",
                "visibility",
                "web_search_tool_type",
            ],
            "the Codex ModelInfo key set is a wire contract"
        );
    }

    #[test]
    fn catalog_entries_keep_their_upstream_identity_and_context_window() {
        let models = catalog(UPSTREAM_MODELS, &configured_capabilities());

        let vision = entry(&models, "vision-model");
        assert_eq!(vision["display_name"], json!("vision-model"));
        assert_eq!(vision["auto_review_model_override"], json!("vision-model"));
        assert_eq!(vision["context_window"], json!(32768));
        assert_eq!(vision["max_context_window"], json!(32768));

        let plain = entry(&models, "plain-model");
        assert_eq!(plain["display_name"], json!("Plain Model"));
        assert_eq!(plain["context_window"], json!(8192), "context_length is the fallback");

        let without_window = entry(&models, "pinned-text-model");
        assert!(
            without_window.get("context_window").is_none(),
            "an unknown context window must stay absent"
        );
        assert!(without_window.get("max_context_window").is_none());
    }

    #[test]
    fn reasoning_capability_drives_reasoning_summaries() {
        let models = catalog(UPSTREAM_MODELS, &configured_capabilities());

        assert_eq!(
            entry(&models, "upstream-image-model")["supports_reasoning_summaries"],
            json!(true)
        );
        assert_eq!(
            entry(&models, "vision-model")["supports_reasoning_summaries"],
            json!(false)
        );
    }

    #[test]
    fn unrecognized_capability_strings_never_advertise_images() {
        let payload = r#"{"data": [{"id": "vl-model", "capabilities": ["vision", "multimodal", "IMAGE"]}]}"#;
        let models = catalog(payload, &ModelCapabilities::default());

        assert_eq!(entry(&models, "vl-model")["input_modalities"], json!(["text"]));
    }

    #[test]
    fn entries_without_an_identifier_are_skipped() {
        let payload = r#"{"data": [{"object": "model"}, {"id": "kept-model"}]}"#;
        let models = catalog(payload, &ModelCapabilities::default());

        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["slug"], json!("kept-model"));
    }

    #[test]
    fn a_payload_without_models_yields_an_empty_catalog() {
        assert!(catalog("{}", &ModelCapabilities::default()).is_empty());
        assert!(catalog(r#"{"object": "list", "data": []}"#, &ModelCapabilities::default()).is_empty());
    }

    #[test]
    fn undecodable_payloads_are_reported_instead_of_emptied() {
        for payload in [
            "not json",
            r#"{"data": "not-a-list"}"#,
            r#"{"data": [{"id": "a-model", "max_model_len": "32k"}]}"#,
        ] {
            assert!(
                build_codex_models_response(payload.as_bytes(), &ModelCapabilities::default()).is_err(),
                "{payload} must not be served as an empty catalog"
            );
        }
    }

    #[test]
    fn readiness_tracker_reports_only_state_transitions() {
        let tracker = ReadinessTracker::default();

        assert_eq!(tracker.last_result(), None);
        assert!(tracker.try_start_probe().unwrap().finish(false));
        assert_eq!(tracker.last_result(), Some(false));
        assert!(!tracker.try_start_probe().unwrap().finish(false));
        assert!(tracker.try_start_probe().unwrap().finish(true));
        assert_eq!(tracker.last_result(), Some(true));
        assert!(!tracker.try_start_probe().unwrap().finish(true));
        assert!(tracker.try_start_probe().unwrap().finish(false));
        assert_eq!(tracker.last_result(), Some(false));
    }

    #[test]
    fn unfinished_readiness_probe_releases_permit_without_changing_result() {
        let tracker = ReadinessTracker::default();
        assert!(tracker.try_start_probe().unwrap().finish(true));

        let unfinished = tracker.try_start_probe().unwrap();
        drop(unfinished);

        assert_eq!(tracker.last_result(), Some(true));
        assert!(tracker.try_start_probe().is_some());
    }

    #[tokio::test(start_paused = true)]
    async fn dependency_check_fails_fast_when_upstream_is_unready() {
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            dependencies_are_ready(future::pending(), future::ready(false)),
        )
        .await
        .expect("upstream failure must win before the pending storage check");

        assert!(!result);
    }
}
