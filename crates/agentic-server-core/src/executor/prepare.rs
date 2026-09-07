//! Explicit request-scoped tool preparation after public rehydration.

use crate::executor::error::ExecutorResult;
use crate::executor::modes::{ConversationHandler, ResponseHandler};
use crate::executor::rehydrate::apply_effective_settings;
use crate::executor::request::RequestContext;
use crate::tool::{ToolSearchHandler, ToolSearchState};
use crate::types::tools::ResponsesTool;

/// Prepare the tool-search projection for a fully rehydrated public request.
///
/// Compaction may remove the call/output pair that records which deferred
/// definitions were loaded. Only that path performs a targeted metadata read;
/// ordinary rehydration does not gain an additional storage query.
pub(crate) async fn prepare_request_tools(
    mut ctx: RequestContext,
    conv_handler: &ConversationHandler,
    resp_handler: &ResponseHandler,
) -> ExecutorResult<(RequestContext, Option<ToolSearchState>)> {
    let restored_loaded_tools = restored_loaded_tools(&mut ctx, conv_handler, resp_handler).await?;
    let restore_only_declared = ctx.original_request.tools.is_some();
    let state =
        ToolSearchHandler::prepare_request(&mut ctx.enriched_request, &restored_loaded_tools, restore_only_declared)?;
    Ok((ctx, state))
}

async fn restored_loaded_tools(
    ctx: &mut RequestContext,
    conv_handler: &ConversationHandler,
    resp_handler: &ResponseHandler,
) -> ExecutorResult<Vec<ResponsesTool>> {
    if !ctx.enriched_request.input.contains_compaction() {
        return Ok(Vec::new());
    }

    if ctx.original_request.previous_response_id.is_some() {
        return Ok(resp_handler
            .get(ctx)
            .await?
            .metadata
            .tool_search_loaded_tools
            .unwrap_or_default());
    }

    let Some(version) = ctx.conversation_version.as_ref() else {
        return Ok(Vec::new());
    };
    let metadata = conv_handler.response_metadata_at_version(ctx, version).await?;
    if let Some(metadata) = metadata {
        let restored = metadata.tool_search_loaded_tools.clone().unwrap_or_default();
        if metadata.tool_search_loaded_tools.is_some() {
            apply_effective_settings(ctx, &metadata);
        }
        return Ok(restored);
    }
    Ok(Vec::new())
}
