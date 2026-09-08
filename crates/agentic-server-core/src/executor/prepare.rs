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
        // Session parents may never have been written to durable storage.
        // Their checkpoint retains the same public metadata as a stored response.
        if let Some(parent) = ctx.continuation.as_ref().and_then(|lease| lease.parent.as_ref()) {
            return Ok(parent.metadata.tool_search_loaded_tools.clone().unwrap_or_default());
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::ResponseSession;
    use crate::storage::{ConversationStore, ResponseMetadata, ResponseStore};
    use crate::types::request_response::RequestPayload;
    use std::num::NonZeroUsize;

    #[tokio::test]
    async fn compacted_session_restores_loaded_tools_without_durable_storage() {
        let session = ResponseSession::new(NonZeroUsize::new(100).unwrap(), NonZeroUsize::new(100_000).unwrap());
        let loaded: ResponsesTool = serde_json::from_value(serde_json::json!({
            "type": "function", "name": "weather", "parameters": {"type": "object"}
        }))
        .unwrap();
        let metadata = ResponseMetadata {
            tool_search_loaded_tools: Some(vec![loaded.clone()]),
            ..ResponseMetadata::default()
        };
        let lease = session.begin(None).unwrap();
        let checkpoint = lease
            .checkpoint("resp_parent".to_owned(), None, &metadata, &[], false)
            .unwrap();
        lease.publish(checkpoint).unwrap();
        let request: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test", "store": false, "previous_response_id": "resp_parent",
            "input": [{"type": "compaction", "id": "cmp_1", "encrypted_content": "summary"}]
        }))
        .unwrap();
        let mut ctx = RequestContext {
            original_request: request.clone(),
            enriched_request: request,
            new_input_items: Vec::new(),
            response_id: "resp_child".to_owned(),
            conversation_id: None,
            conversation_version: None,
            continuation: Some(session.begin(Some("resp_parent")).unwrap()),
        };
        let restored = restored_loaded_tools(
            &mut ctx,
            &ConversationHandler::new(ConversationStore::disabled()),
            &ResponseHandler::new(ResponseStore::disabled()),
        )
        .await
        .unwrap();
        assert_eq!(serde_json::to_value(restored).unwrap(), serde_json::json!([loaded]));
    }
}
