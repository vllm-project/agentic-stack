//! Step 3 of the conversation pipeline — response persistence.
//!
//! Writes the completed response and output items to storage, routing to the
//! appropriate handler based on whether the turn belongs to a conversation.

use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::modes::{ConversationHandler, ResponseHandler};
use crate::executor::prepare::prepare_request_tools;
use crate::executor::request::{ExecutionContext, RequestContext};
use crate::storage::ResponseMetadata;
use crate::tool::{ToolSearchMetadata, ToolSearchState};
use crate::types::event::ResponseStatus;
use crate::types::io::OutputItem;
use crate::types::request_response::ResponsePayload;
use tracing::error;

#[must_use]
pub(crate) fn should_persist(ctx: &RequestContext) -> bool {
    ctx.original_request.store
        || ctx.original_request.previous_response_id.is_some()
        || ctx.original_request.conversation_id.is_some()
}

pub(crate) async fn persist_if_needed(
    payload: ResponsePayload,
    ctx: RequestContext,
    tool_search_metadata: Option<ToolSearchMetadata>,
    conv_handler: ConversationHandler,
    resp_handler: ResponseHandler,
) -> ExecutorResult<()> {
    if should_persist(&ctx) {
        match persist_prepared_response(payload, ctx, tool_search_metadata, conv_handler, resp_handler).await {
            Err(error @ ExecutorError::Conflict(_)) => Err(error),
            Err(source) => {
                error!(error = ?source, "failed to persist response");
                Err(ExecutorError::Persistence(Box::new(source)))
            }
            Ok(()) => Ok(()),
        }
    } else {
        Ok(())
    }
}

/// Step 3 — Persist the completed response to storage.
///
/// Skipped if [`ResponseStatus`] is not `Completed`/`Incomplete` or `payload.id` is empty.
/// Routes explicit `conversation_id` requests to [`ConversationHandler`] and
/// all other requests, including `previous_response_id` continuations, to [`ResponseHandler`].
///
/// # Errors
/// Returns [`ExecutorError`] if the storage operation fails.
pub async fn persist_response(
    payload: ResponsePayload,
    ctx: RequestContext,
    conv_handler: ConversationHandler,
    resp_handler: ResponseHandler,
) -> ExecutorResult<()> {
    // Use typed enum — no hardcoded status strings.
    if !matches!(
        payload.status.parse::<ResponseStatus>().unwrap_or_default(),
        ResponseStatus::Completed | ResponseStatus::Incomplete
    ) || payload.id.is_empty()
    {
        return Ok(());
    }

    let (ctx, tool_search_state) = prepare_request_tools(ctx, &conv_handler, &resp_handler).await?;
    let tool_search_metadata = tool_search_state.map(ToolSearchState::into_public_metadata);
    persist_prepared_turn(ctx, tool_search_metadata, payload.output, &conv_handler, &resp_handler).await
}

async fn persist_prepared_response(
    payload: ResponsePayload,
    ctx: RequestContext,
    tool_search_metadata: Option<ToolSearchMetadata>,
    conv_handler: ConversationHandler,
    resp_handler: ResponseHandler,
) -> ExecutorResult<()> {
    if !matches!(
        payload.status.parse::<ResponseStatus>().unwrap_or_default(),
        ResponseStatus::Completed | ResponseStatus::Incomplete
    ) || payload.id.is_empty()
    {
        return Ok(());
    }

    persist_prepared_turn(ctx, tool_search_metadata, payload.output, &conv_handler, &resp_handler).await
}

/// Persists one completed turn with the handler selected by its explicit conversation discriminator.
///
/// # Errors
/// Returns [`ExecutorError`] if the selected storage operation fails.
pub async fn persist_turn(
    ctx: RequestContext,
    output_items: Vec<OutputItem>,
    conv_handler: &ConversationHandler,
    resp_handler: &ResponseHandler,
) -> ExecutorResult<()> {
    let (ctx, tool_search_state) = prepare_request_tools(ctx, conv_handler, resp_handler).await?;
    let tool_search_metadata = tool_search_state.map(ToolSearchState::into_public_metadata);
    persist_prepared_turn(ctx, tool_search_metadata, output_items, conv_handler, resp_handler).await
}

pub(crate) async fn persist_prepared_turn(
    mut ctx: RequestContext,
    tool_search_metadata: Option<ToolSearchMetadata>,
    output_items: Vec<OutputItem>,
    conv_handler: &ConversationHandler,
    resp_handler: &ResponseHandler,
) -> ExecutorResult<()> {
    let mut metadata = ResponseMetadata {
        model: std::mem::take(&mut ctx.enriched_request.model),
        previous_response_id: ctx.original_request.previous_response_id.take(),
        effective_tools: ctx.enriched_request.tools.take(),
        tool_search_loaded_tools: None,
        effective_tool_choice: ctx.enriched_request.tool_choice.take().unwrap_or_default(),
        effective_instructions: ctx.enriched_request.instructions.take(),
    };
    if let Some(tool_search_metadata) = tool_search_metadata {
        metadata.effective_tools = tool_search_metadata.effective_tools;
        metadata.tool_search_loaded_tools = Some(tool_search_metadata.loaded_tools);
    }
    if ctx.original_request.conversation_id.is_some() {
        conv_handler
            .execute_turn_with_metadata(ctx, output_items, metadata)
            .await
    } else {
        resp_handler
            .execute_turn_with_metadata(ctx, output_items, metadata)
            .await
    }
}

/// Stores a decoded turn for a caller that ran inference itself. A failed turn is
/// returned unstored, as the in-process flow does.
///
/// # Errors
/// [`ExecutorError::InvalidRequest`] for an unusable id or unfinished response,
/// [`ExecutorError::Conflict`] for an id already stored, or a storage error.
pub async fn commit(
    ctx: RequestContext,
    payload: ResponsePayload,
    exec_ctx: &ExecutionContext,
) -> ExecutorResult<ResponsePayload> {
    if ctx.response_id.is_empty() {
        return Err(ExecutorError::InvalidRequest(
            "context has no reserved response id".to_owned(),
        ));
    }

    // Storing an unfinished turn would return an id that can never be continued.
    if payload.status.parse::<ResponseStatus>().unwrap_or_default() == ResponseStatus::InProgress {
        return Err(ExecutorError::InvalidRequest(format!(
            "upstream response status '{}' is not terminal",
            payload.status
        )));
    }

    persist_if_needed(
        payload.clone(),
        ctx,
        None,
        exec_ctx.conv_handler.clone(),
        exec_ctx.resp_handler.clone(),
    )
    .await?;
    Ok(payload)
}
