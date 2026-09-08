//! Response storage handler — owns all response store operations.

use crate::storage::{InOutItem, ResponseData, ResponseMetadata, ResponseStore};
use crate::types::io::OutputItem;

use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::request::RequestContext;

/// Handles all response store operations: lookup, rehydration, and persistence.
#[derive(Clone, Debug)]
pub struct ResponseHandler {
    store: ResponseStore,
}

impl ResponseHandler {
    #[must_use]
    pub fn new(store: ResponseStore) -> Self {
        Self { store }
    }

    /// Retrieves the stored response for `previous_response_id`.
    ///
    /// Reads `previous_response_id` from `ctx.original_request`.
    ///
    /// # Errors
    /// Returns `ExecutorError` if `previous_response_id` is absent, the response
    /// is not found, the store is disabled, or the database query fails.
    pub async fn get(&self, ctx: &RequestContext) -> ExecutorResult<ResponseData> {
        let prev_id = ctx
            .original_request
            .previous_response_id
            .as_deref()
            .ok_or_else(|| ExecutorError::InvalidRequest("previous_response_id is required for get".into()))?;
        self.store.get(prev_id).await.map_err(ExecutorError::Storage)
    }

    /// Validates that the response for `previous_response_id` exists.
    ///
    /// Used in the `store=false` path where we only need to confirm the ID is
    /// valid without loading any history.
    ///
    /// # Errors
    /// Returns `ExecutorError` if `previous_response_id` is absent, the response
    /// is not found, or the store is disabled.
    pub async fn validate_exists(&self, ctx: &RequestContext) -> ExecutorResult<()> {
        self.get(ctx).await.map(|_| ())
    }

    /// Loads all history items referenced by the previous response.
    ///
    /// Reads `previous_response_id` from `ctx.original_request`. Returns an empty
    /// vec if there is no previous response.
    ///
    /// # Errors
    /// Returns `ExecutorError` if the store is disabled or the database query fails.
    pub async fn rehydrate(&self, ctx: &RequestContext) -> ExecutorResult<Vec<InOutItem>> {
        let Some(prev_id) = ctx.original_request.previous_response_id.as_deref() else {
            return Ok(vec![]);
        };
        self.store.rehydrate(prev_id).await.map_err(ExecutorError::Storage)
    }

    /// Completes response-scoped persistence and optional transient checkpoint retention.
    ///
    /// Takes `ctx` and `output_items` by value so fields can be moved directly
    /// into [`ResponseMetadata`]. Normally only this turn's items are inserted;
    /// promoting a transient parent to a stored child requires its full canonical
    /// history because there are no durable parent item IDs to reference.
    ///
    /// # Errors
    /// Returns `ExecutorError` if the store is disabled or the database operation fails.
    pub async fn execute_turn(&self, mut ctx: RequestContext, output_items: Vec<OutputItem>) -> ExecutorResult<()> {
        let metadata = ResponseMetadata {
            model: std::mem::take(&mut ctx.enriched_request.model),
            previous_response_id: ctx.original_request.previous_response_id.take(),
            effective_tools: ctx.enriched_request.tools.take(),
            tool_search_loaded_tools: None,
            effective_tool_choice: ctx.enriched_request.tool_choice.take().unwrap_or_default(),
            effective_instructions: ctx.enriched_request.instructions.take(),
        };

        self.execute_turn_with_metadata(ctx, output_items, metadata).await
    }

    /// Persists a response using metadata prepared by request-scoped tool behavior.
    pub(crate) async fn execute_turn_with_metadata(
        &self,
        mut ctx: RequestContext,
        output_items: Vec<OutputItem>,
        metadata: ResponseMetadata,
    ) -> ExecutorResult<()> {
        let continuation = ctx.continuation.take();
        let write_durable = continuation.is_none() || ctx.original_request.store;
        let mut new_items = Vec::with_capacity(ctx.new_input_items.len() + output_items.len());
        new_items.extend(ctx.new_input_items.into_iter().map(InOutItem::Input));
        new_items.extend(
            output_items
                .into_iter()
                .enumerate()
                .filter(|(index, item)| {
                    continuation
                        .as_ref()
                        .is_none_or(|lease| lease.retains_output(*index, item))
                })
                .map(|(_, item)| InOutItem::Output(item)),
        );

        let checkpoint = continuation
            .as_ref()
            .map(|lease| {
                lease.checkpoint(
                    ctx.response_id.clone(),
                    ctx.conversation_id.clone(),
                    &metadata,
                    &new_items,
                    write_durable,
                )
            })
            .transpose()?;

        // A stored child of a transient parent needs its complete canonical
        // checkpoint, not a database reference to a response that was never stored.
        let transient_parent = continuation
            .as_ref()
            .filter(|lease| lease.parent.as_ref().is_some_and(|parent| !parent.durable));
        let previous_durable_id = if let Some(lease) = transient_parent {
            if write_durable {
                let mut full_items = lease.parent_items().cloned().map(InOutItem::Input).collect::<Vec<_>>();
                full_items.append(&mut new_items);
                new_items = full_items;
            }
            None
        } else {
            metadata.previous_response_id.as_deref()
        };

        let result = if write_durable {
            self.store
                .persist_with_conversation_id(
                    &ctx.response_id,
                    ctx.conversation_id.as_deref(),
                    previous_durable_id,
                    new_items,
                    &metadata,
                )
                .await
        } else {
            Ok(())
        };
        match result {
            Err(error) if error.is_unique_violation() => Err(ExecutorError::Conflict(format!(
                "a turn is already stored under '{}'",
                ctx.response_id
            ))),
            Err(error) => Err(ExecutorError::Storage(error)),
            Ok(()) => {
                if let Some((lease, checkpoint)) = continuation.zip(checkpoint) {
                    lease.publish(checkpoint)?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::io::ResponsesInput;
    use crate::types::request_response::RequestPayload;

    fn disabled_handler() -> ResponseHandler {
        ResponseHandler::new(ResponseStore::disabled())
    }

    fn make_ctx(previous_response_id: Option<&str>) -> RequestContext {
        let req = RequestPayload {
            model: "test".into(),
            input: ResponsesInput::Text("hi".into()),
            instructions: None,
            previous_response_id: previous_response_id.map(str::to_string),
            conversation_id: None,
            tools: None,
            tool_choice: None,
            stream: false,
            store: true,
            include: None,
            reasoning: None,
            text: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            truncation: None,
            metadata: None,
            parallel_tool_calls: None,
            cache_salt: None,
            context_management: None,
        };
        RequestContext {
            enriched_request: req.clone(),
            original_request: req,
            new_input_items: vec![],
            response_id: "resp_test".into(),
            conversation_id: None,
            conversation_version: None,
            continuation: None,
        }
    }

    #[tokio::test]
    async fn test_get_missing_prev_id_returns_error() {
        let result = disabled_handler().get(&make_ctx(None)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validate_exists_missing_prev_id_returns_error() {
        let result = disabled_handler().validate_exists(&make_ctx(None)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_rehydrate_no_prev_id_returns_empty() {
        let result = disabled_handler().rehydrate(&make_ctx(None)).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_rehydrate_disabled_store_returns_error() {
        let result = disabled_handler().rehydrate(&make_ctx(Some("resp_prev"))).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_turn_disabled_store_returns_error() {
        let result = disabled_handler().execute_turn(make_ctx(None), vec![]).await;
        assert!(result.is_err());
    }
}
