//! The wire protocol between `/v1alpha/responses/hydrate` and `.../persist`. Only
//! this crate speaks it; core keeps the operations it composes.
#![allow(clippy::result_large_err)] // `ExecutorError` is core's; boxing it is not ours to decide

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use agentic_core::executor::request::RequestContext;
use agentic_core::executor::{ExecutorError, ExecutorResult};
use agentic_core::types::io::{ResponsesInput, ToolChoice};
use agentic_core::types::request_response::RequestPayload;
use agentic_core::types::tools::ResponsesTool;

use crate::SigningKey;

/// What `hydrate` returns. Raw JSON: the caller forwards it uninterpreted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hydration {
    pub request: Box<RawValue>,
    /// Sealed: echo back to `persist` unchanged. Opaque to the caller.
    pub context: String,
}

/// Wire form of a [`RequestContext`]. `enriched_request` and `new_input_items`
/// are absent on purpose: both are rebuilt on return.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitContext {
    pub response_id: String,
    pub original_request: RequestPayload,
    /// Inherited from the continued turn; the request's own is rejected.
    pub conversation_id: Option<String>,
    /// The only part of `enriched_request` that `original_request` cannot supply.
    pub effective_tools: Option<Vec<ResponsesTool>>,
    pub effective_tool_choice: Option<ToolChoice>,
}

impl From<RequestContext> for SplitContext {
    fn from(ctx: RequestContext) -> Self {
        Self {
            response_id: ctx.response_id,
            original_request: ctx.original_request,
            conversation_id: ctx.conversation_id,
            effective_tools: ctx.enriched_request.tools,
            effective_tool_choice: ctx.enriched_request.tool_choice,
        }
    }
}

impl From<SplitContext> for RequestContext {
    fn from(wire: SplitContext) -> Self {
        let new_input_items = Vec::from(&wire.original_request.input);
        let mut enriched_request = wire.original_request.clone();
        enriched_request.previous_response_id = None;
        enriched_request.input = ResponsesInput::Items(new_input_items.clone());
        enriched_request.tools = wire.effective_tools;
        enriched_request.tool_choice = wire.effective_tool_choice;
        Self {
            original_request: wire.original_request,
            enriched_request,
            new_input_items,
            response_id: wire.response_id,
            conversation_id: wire.conversation_id,
            // Conversation mode is rejected, so there is no version to resume.
            conversation_version: None,
            continuation: None,
        }
    }
}

/// Rejects requests needing state the in-process flow keeps between steps.
///
/// # Errors
/// [`ExecutorError::InvalidRequest`] naming the feature that cannot be split.
pub fn ensure_splittable(request: &RequestPayload) -> ExecutorResult<()> {
    if let Some(feature) = request.in_process_feature() {
        return Err(ExecutorError::InvalidRequest(format!(
            "{feature} is not supported for split execution"
        )));
    }
    Ok(())
}

/// Only has to outlive one inference call, so generous rather than tuned.
const CONTEXT_TTL: Duration = Duration::from_secs(600);
const AUDIENCE: &str = "agentic-llm-d";

#[derive(Serialize, Deserialize)]
struct SealedClaims {
    exp: u64,
    aud: String,
    ctx: SplitContext,
}

/// Seals a context so `persist` can prove `hydrate` issued it.
///
/// # Errors
/// [`ExecutorError::InvalidRequest`] if the token cannot be produced.
pub fn seal(context: SplitContext, key: &SigningKey) -> ExecutorResult<String> {
    let expires = SystemTime::now()
        .checked_add(CONTEXT_TTL)
        .and_then(|at| at.duration_since(UNIX_EPOCH).ok())
        .ok_or_else(|| ExecutorError::InvalidRequest("cannot compute context expiry".to_owned()))?;
    let claims = SealedClaims {
        exp: expires.as_secs(),
        aud: AUDIENCE.to_owned(),
        ctx: context,
    };
    encode(
        &Header::new(Algorithm::HS256),
        &claims,
        &EncodingKey::from_secret(key.as_bytes()),
    )
    .map_err(|error| ExecutorError::InvalidRequest(format!("cannot seal context: {error}")))
}

/// Opens a sealed context, rejecting one that was tampered with or has expired.
///
/// # Errors
/// [`ExecutorError::InvalidRequest`] for a bad signature, a wrong audience, or
/// a context past its expiry.
pub fn unseal(token: &str, key: &SigningKey) -> ExecutorResult<SplitContext> {
    let mut validation = Validation::new(Algorithm::HS256);
    validation.set_audience(&[AUDIENCE]);
    validation.set_required_spec_claims(&["exp", "aud"]);
    decode::<SealedClaims>(token, &DecodingKey::from_secret(key.as_bytes()), &validation)
        .map(|data| data.claims.ctx)
        .map_err(|error| ExecutorError::InvalidRequest(format!("context rejected: {error}")))
}
