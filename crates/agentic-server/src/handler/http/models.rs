use std::future::Future;
use std::sync::OnceLock;

use axum::extract::{Query, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use http::StatusCode;
use serde_json::{Value, json};
use tracing::{debug, info, warn};

use agentic_core::proxy::{ProxyBody, ProxyResponse, error_response, proxy_get};
use agentic_core::readiness::{LLM_READINESS_PROBE_TIMEOUT, LlmReadiness, probe_llm_readiness};

use super::super::common::convert_response;
use crate::app::AppState;

/// Static fields shared by every Codex `ModelInfo` entry.
///
/// Built once on first use; cloned per model and patched with the per-model
/// values (`slug`, `display_name`, `auto_review_model_override`,
/// `supports_reasoning_summaries`, `input_modalities`, and optionally
/// `context_window` / `max_context_window`).
fn codex_model_template() -> &'static Value {
    static TEMPLATE: OnceLock<Value> = OnceLock::new();
    TEMPLATE.get_or_init(|| {
        json!({
            "supported_in_api": true,
            "priority": 1,
            "shell_type": "shell_command",
            "visibility": "list",
            "base_instructions": "",
            "supported_reasoning_levels": [
                {"effort": "low",    "description": "Fast responses with lighter reasoning"},
                {"effort": "medium", "description": "Balances speed and reasoning depth"},
                {"effort": "high",   "description": "Greater reasoning depth for complex problems"}
            ],
            "default_reasoning_summary": "auto",
            "support_verbosity": false,
            "default_verbosity": null,
            "apply_patch_tool_type": "freeform",
            "web_search_tool_type": "text",
            "truncation_policy": {"mode": "bytes", "limit": 100_000},
            "supports_parallel_tool_calls": true,
            "supports_image_detail_original": false,
            "effective_context_window_percent": 95,
            "experimental_supported_tools": [],
            "supports_search_tool": false,
            "use_responses_lite": false,
            "tool_mode": null,
            "multi_agent_version": null,
        })
    })
}

/// Transform a single upstream model entry into a Codex `ModelInfo` object.
///
/// Returns `None` when the entry has no `id` field (malformed upstream data).
fn upstream_model_to_codex(m: &Value) -> Option<Value> {
    let id = m["id"].as_str()?.to_owned();
    let display_name = m.get("name").and_then(Value::as_str).unwrap_or(&id).to_owned();
    // vLLM uses max_model_len; other providers may use context_length
    let context_length = m["max_model_len"].as_i64().or_else(|| m["context_length"].as_i64());
    // Single pass over capabilities for both flags
    let (supports_reasoning, supports_image) = m["capabilities"].as_array().map_or((false, false), |c| {
        c.iter().fold((false, false), |(r, i), v| {
            let s = v.as_str();
            (r || s == Some("reasoning"), i || s == Some("image"))
        })
    });
    let input_modalities = if supports_image {
        json!(["text", "image"])
    } else {
        json!(["text"])
    };

    let mut model = codex_model_template().clone();
    let obj = model.as_object_mut().expect("template is object");
    obj.insert("slug".into(), json!(id));
    obj.insert("display_name".into(), json!(display_name));
    obj.insert("auto_review_model_override".into(), json!(id));
    obj.insert("supports_reasoning_summaries".into(), json!(supports_reasoning));
    obj.insert("input_modalities".into(), input_modalities);
    if let Some(ctx) = context_length {
        obj.insert("context_window".into(), json!(ctx));
        obj.insert("max_context_window".into(), json!(ctx));
    }

    Some(model)
}

/// Build the Codex `ModelsResponse` from a raw upstream vLLM models payload.
fn build_codex_models_response(upstream_bytes: &[u8]) -> Value {
    let models: Vec<Value> = serde_json::from_slice::<Value>(upstream_bytes)
        .ok()
        .and_then(|mut v| match v["data"].take() {
            Value::Array(arr) => Some(arr),
            _ => None,
        })
        .into_iter()
        .flatten()
        .filter_map(|m| upstream_model_to_codex(&m))
        .collect();
    json!({ "models": models })
}

#[cfg_attr(feature = "openapi", utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Server is alive"),
    ),
    tag = "health",
))]
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

#[cfg_attr(feature = "openapi", utoipa::path(
    get,
    path = "/ready",
    responses(
        (status = 200, description = "Server and dependencies are ready"),
        (status = 503, description = "One or more dependencies are not ready"),
    ),
    tag = "health",
))]
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
#[cfg_attr(feature = "openapi", utoipa::path(
    get,
    path = "/v1/models",
    params(
        ("client_version" = Option<String>, Query, description = "Codex CLI version; triggers Codex-compatible model list shape"),
    ),
    responses(
        (status = 200, description = "Model list"),
        (status = 502, description = "Upstream unavailable", body = crate::openapi::ApiErrorResponse),
    ),
    security(("bearer_auth" = [])),
    tag = "models",
))]
pub async fn models(State(state): State<AppState>, headers: HeaderMap, Query(params): Query<ModelsParams>) -> Response {
    let upstream = proxy_get("/v1/models", &headers, &state.proxy_state).await;

    if params.client_version.is_none() {
        return convert_response(upstream);
    }

    let ProxyBody::Full(upstream_bytes) = upstream.body else {
        return convert_response(error_response(
            http::StatusCode::BAD_GATEWAY,
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

    axum::Json(build_codex_models_response(&upstream_bytes)).into_response()
}

#[cfg(test)]
mod tests {
    use std::future;
    use std::time::Duration;

    use super::dependencies_are_ready;
    use crate::app::ReadinessTracker;

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
