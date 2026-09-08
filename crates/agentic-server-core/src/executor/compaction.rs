use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::rehydrate::rehydrate_conversation;
use crate::executor::request::{ExecutionContext, RequestContext};
use crate::executor::upstream::fetch_blocking_payload;
use crate::types::event::MessageStatus;
use crate::types::io::input::latest_compaction_window;
use crate::types::io::{
    CompactionItem, InputContent, InputItem, InputMessage, InputMessageContent, OutputItem, ResponseUsage,
    ResponsesInput,
};
use crate::types::request_response::{CompactRequest, CompactedResponse, RequestPayload, ResponsePayload};
use crate::utils::common::{serialize_to_string, utcnow_str, uuid7_str};

const COMPACTION_PROMPT: &str = "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a concise handoff summary that preserves current progress, decisions, constraints, unresolved work, and critical references for the next model. Return only the summary.";

fn retained_user_window(items: &[InputItem]) -> Vec<InputItem> {
    let window = latest_compaction_window(items);
    items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            let InputItem::Message(message) = item else {
                return None;
            };
            if message.role != "user"
                || window.is_some_and(|window| index < window.latest_index() && !window.retains_user_item(index, item))
            {
                return None;
            }
            let mut retained = message.clone();
            retained.id = Some(uuid7_str("msg_"));
            retained.status = Some(MessageStatus::Completed);
            Some(InputItem::Message(retained))
        })
        .collect()
}

fn finish_compacted_window(mut output: Vec<InputItem>, summary: String) -> Vec<InputItem> {
    output.push(InputItem::Compaction(CompactionItem {
        id: Some(uuid7_str("cmp_")),
        encrypted_content: summary,
    }));
    output
}

#[cfg(test)]
fn build_compacted_window(items: &[InputItem], summary: String) -> Vec<InputItem> {
    finish_compacted_window(retained_user_window(items), summary)
}

fn response_output_text(output: &[OutputItem]) -> Option<String> {
    let text = output
        .iter()
        .filter_map(|item| match item {
            OutputItem::Message(message) => Some(message),
            _ => None,
        })
        .flat_map(|message| message.content.iter())
        .map(|content| content.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    (!text.is_empty()).then_some(text)
}

fn value_has_content(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::String(text) => !text.trim().is_empty(),
        serde_json::Value::Array(values) => values.iter().any(value_has_content),
        serde_json::Value::Object(values) => values.values().any(value_has_content),
        serde_json::Value::Bool(_) | serde_json::Value::Number(_) => true,
    }
}

fn item_has_meaningful_context(item: &InputItem) -> bool {
    match item {
        InputItem::Message(message) => match &message.content {
            InputMessageContent::Text(text) => !text.trim().is_empty(),
            InputMessageContent::Parts(parts) => parts.iter().any(|part| match part {
                InputContent::InputText(text) | InputContent::OutputText(text) | InputContent::ReasoningText(text) => {
                    !text.text.trim().is_empty()
                }
                InputContent::InputImage(image) => image.image_url.as_deref().is_some_and(|url| !url.trim().is_empty()),
                InputContent::InputFile(file) => file.has_reference(),
                InputContent::Unknown => false,
            }),
        },
        InputItem::FunctionCall(call) => !call.name.trim().is_empty() || !call.arguments.trim().is_empty(),
        InputItem::FunctionCallOutput(output) => output.output.has_content(),
        InputItem::CustomToolCall(call) => !call.name.trim().is_empty() || !call.input.trim().is_empty(),
        InputItem::CustomToolCallOutput(output) => output.output.has_content(),
        InputItem::Reasoning(reasoning) => {
            reasoning.content.iter().any(|content| !content.text.trim().is_empty())
                || reasoning.summary.iter().any(value_has_content)
                || reasoning.encrypted_content.as_ref().is_some_and(value_has_content)
        }
        InputItem::Compaction(compaction) => !compaction.encrypted_content.trim().is_empty(),
        InputItem::McpListTools(_) | InputItem::CompactionTrigger | InputItem::Unknown => false,
    }
}

fn completed_summary_text(response: &ResponsePayload) -> ExecutorResult<String> {
    if response.status != "completed" || response.error.is_some() {
        let details = response
            .error
            .as_ref()
            .and_then(|error| serialize_to_string(error).ok())
            .or_else(|| {
                response
                    .incomplete_details
                    .as_ref()
                    .and_then(|details| details.reason.clone())
            })
            .unwrap_or_else(|| "upstream returned no failure details".to_owned());
        return Err(ExecutorError::CompactionFailed {
            status: response.status.clone(),
            details,
        });
    }
    response_output_text(&response.output).ok_or_else(|| ExecutorError::CompactionFailed {
        status: response.status.clone(),
        details: "upstream returned no summary text".to_owned(),
    })
}

/// Estimate the current model-facing context size without requiring a model-specific tokenizer.
///
/// The approximation deliberately includes JSON structure and rounds up at four UTF-8 bytes per
/// token. It is deterministic, inexpensive, and errs slightly toward compacting early.
#[must_use]
pub(crate) fn estimate_input_tokens(input: &ResponsesInput) -> u64 {
    let serialized = serialize_to_string(&input.model_input()).unwrap_or_default();
    let bytes = u64::try_from(serialized.len()).unwrap_or(u64::MAX);
    bytes.saturating_add(3) / 4
}

fn request_payload(model: String, input: ResponsesInput, instructions: Option<String>) -> RequestPayload {
    RequestPayload {
        model,
        input,
        instructions,
        previous_response_id: None,
        conversation_id: None,
        tools: None,
        tool_choice: None,
        stream: false,
        store: false,
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
    }
}

/// Summarize an already-resolved item history and return its canonical compacted window.
///
/// # Errors
///
/// Returns an invalid-request error for empty input, an upstream error for an unusable model
/// summary, and propagates inference and serialization failures.
pub(crate) async fn compact_items(
    model: &str,
    input: ResponsesInput,
    instructions: Option<&str>,
    exec_ctx: &ExecutionContext,
    auth: Option<&str>,
) -> ExecutorResult<(Vec<InputItem>, ResponseUsage)> {
    let original_items = Vec::from(input);
    if !original_items.iter().any(item_has_meaningful_context) {
        return Err(ExecutorError::InvalidRequest(
            "compaction requires non-empty input or previous_response_id context".to_owned(),
        ));
    }

    let compacted = retained_user_window(&original_items);
    let mut summary_items: Vec<InputItem> = original_items
        .into_iter()
        .filter(|item| !item.is_compaction_trigger())
        .collect();
    summary_items.push(InputItem::Message(InputMessage {
        id: None,
        role: "user".to_owned(),
        status: None,
        content: InputMessageContent::Text(COMPACTION_PROMPT.to_owned()),
    }));
    let instructions = instructions.map(str::to_owned);
    let original_request = request_payload(
        model.to_owned(),
        ResponsesInput::Items(Vec::new()),
        instructions.clone(),
    );
    let enriched_request = request_payload(model.to_owned(), ResponsesInput::Items(summary_items), instructions);
    let ctx = RequestContext {
        original_request,
        enriched_request,
        new_input_items: Vec::new(),
        response_id: uuid7_str("resp_"),
        conversation_id: None,
        conversation_version: None,
    };
    let response = fetch_blocking_payload(&ctx, exec_ctx, auth).await?;
    let summary = completed_summary_text(&response)?;

    Ok((
        finish_compacted_window(compacted, summary),
        response.usage.unwrap_or_default(),
    ))
}

/// Apply the first configured compaction threshold to the resolved request history.
///
/// Returns the summarization usage when compaction ran. The configuration remains on the
/// client-facing request context but is never part of [`crate::types::request_response::UpstreamRequest`].
///
/// # Errors
///
/// Propagates inference errors from the summarization request.
pub(crate) async fn maybe_compact_context(
    ctx: &mut RequestContext,
    exec_ctx: &ExecutionContext,
    auth: Option<&str>,
) -> ExecutorResult<Option<ResponseUsage>> {
    let threshold = ctx
        .enriched_request
        .context_management
        .as_deref()
        .unwrap_or_default()
        .iter()
        .find(|entry| entry.type_ == "compaction")
        .and_then(|entry| entry.compact_threshold);
    let Some(threshold) = threshold else {
        return Ok(None);
    };
    let estimated_tokens = estimate_input_tokens(&ctx.enriched_request.input);
    if estimated_tokens <= threshold {
        return Ok(None);
    }

    tracing::debug!(
        estimated_tokens,
        threshold,
        "automatically compacting resolved response input"
    );
    let model = ctx.enriched_request.model.clone();
    let instructions = ctx.enriched_request.instructions.clone();
    let input = std::mem::replace(&mut ctx.enriched_request.input, ResponsesInput::Items(Vec::new()));
    let (compacted, usage) = compact_items(&model, input, instructions.as_deref(), exec_ctx, auth).await?;
    ctx.enriched_request.input = ResponsesInput::Items(compacted.clone());
    ctx.new_input_items = compacted;
    Ok(Some(usage))
}

/// Compact direct input or a stored previous-response chain into a reusable item window.
///
/// # Errors
///
/// Returns an invalid-request error when neither input nor a previous response ID is supplied,
/// and propagates history, inference, and persistence failures.
pub async fn compact_response(
    request: CompactRequest,
    exec_ctx: &ExecutionContext,
    auth: Option<&str>,
) -> ExecutorResult<CompactedResponse> {
    if request.input.is_none() && request.previous_response_id.is_none() {
        return Err(ExecutorError::InvalidRequest(
            "compaction requires input or previous_response_id".to_owned(),
        ));
    }

    let mut payload = request_payload(
        request.model,
        request.input.unwrap_or_else(|| ResponsesInput::Items(Vec::new())),
        request.instructions,
    );
    payload.previous_response_id = request.previous_response_id;
    let mut ctx = rehydrate_conversation(payload, exec_ctx).await?;
    let model = ctx.enriched_request.model.clone();
    let instructions = ctx.enriched_request.instructions.clone();
    let input = std::mem::replace(&mut ctx.enriched_request.input, ResponsesInput::Items(Vec::new()));
    let (output, usage) = compact_items(&model, input, instructions.as_deref(), exec_ctx, auth).await?;

    let response_id = ctx.response_id.clone();
    ctx.new_input_items.clone_from(&output);
    match exec_ctx.resp_handler.execute_turn(ctx, Vec::new()).await {
        Ok(()) | Err(ExecutorError::Storage(crate::StorageError::NotConfigured)) => {}
        Err(error) => return Err(error),
    }

    Ok(CompactedResponse {
        id: response_id,
        object: "response.compaction".to_owned(),
        created_at: utcnow_str(),
        output,
        usage,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::Router;
    use axum::routing::post;

    use crate::executor::modes::{ConversationHandler, ResponseHandler};
    use crate::executor::request::ExecutionContext;
    use crate::storage::{ConversationStore, InOutItem, ResponseMetadata, ResponseStore, create_pool_with_schema};
    use crate::types::event::MessageStatus;
    use crate::types::io::{
        CompactionItem, FunctionToolResultMessage, InputItem, InputMessage, InputMessageContent, ResponsesInput,
    };

    use super::{build_compacted_window, compact_response, completed_summary_text, estimate_input_tokens};

    fn user_message(text: &str) -> InputItem {
        InputItem::Message(InputMessage {
            id: None,
            role: "user".to_owned(),
            status: None,
            content: InputMessageContent::Text(text.to_owned()),
        })
    }

    async fn mock_execution_context(response_store: ResponseStore) -> (ExecutionContext, tokio::task::JoinHandle<()>) {
        let app = Router::new().route(
            "/v1/responses",
            post(|| async {
                axum::Json(serde_json::json!({
                    "id": "resp_upstream",
                    "object": "response",
                    "created_at": 0,
                    "model": "test-model",
                    "status": "completed",
                    "output": [{
                        "id": "msg_upstream",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{
                            "type": "output_text",
                            "text": "durable summary",
                            "annotations": []
                        }]
                    }],
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "total_tokens": 15
                    },
                    "incomplete_details": null,
                    "error": null,
                    "previous_response_id": null,
                    "conversation_id": null,
                    "instructions": null
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock inference server");
        let address = listener.local_addr().expect("mock server address");
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        let exec_ctx = ExecutionContext::new(
            ConversationHandler::new(ConversationStore::disabled()),
            ResponseHandler::new(response_store),
            Arc::new(reqwest::Client::new()),
            format!("http://{address}"),
        );
        (exec_ctx, server)
    }

    fn compact_request() -> crate::CompactRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "input": [{"role": "user", "content": "important context"}]
        }))
        .expect("valid compact request")
    }

    #[test]
    fn compacted_window_retains_user_messages_and_replaces_old_compaction() {
        let items = vec![
            InputItem::Compaction(CompactionItem {
                id: Some("cmp_old".to_owned()),
                encrypted_content: "old summary".to_owned(),
            }),
            user_message("first"),
            InputItem::FunctionCallOutput(FunctionToolResultMessage {
                call_id: "call_1".to_owned(),
                output: "tool output".into(),
            }),
            user_message("second"),
        ];

        let output = build_compacted_window(&items, "new summary".to_owned());

        assert_eq!(
            output
                .iter()
                .filter(|item| matches!(item, InputItem::Message(_)))
                .count(),
            2
        );
        assert_eq!(
            output
                .iter()
                .filter(|item| matches!(item, InputItem::Compaction(_)))
                .count(),
            1
        );
        assert!(matches!(output.last(), Some(InputItem::Compaction(_))));
        for item in output.iter().filter_map(|item| match item {
            InputItem::Message(message) => Some(message),
            _ => None,
        }) {
            assert_eq!(item.status, Some(MessageStatus::Completed));
            assert!(item.id.as_deref().is_some_and(|id| id.starts_with("msg_")));
        }
    }

    #[test]
    fn token_estimate_counts_replayed_content() {
        let input = ResponsesInput::Items(vec![
            user_message("hello context"),
            InputItem::FunctionCallOutput(FunctionToolResultMessage {
                call_id: "call_1".to_owned(),
                output: "substantial tool output".into(),
            }),
        ]);

        assert!(estimate_input_tokens(&input) > 0);
    }

    #[test]
    fn partial_text_from_incomplete_or_failed_summary_is_rejected() {
        for (status, error, incomplete_details) in [
            (
                "incomplete",
                serde_json::Value::Null,
                serde_json::json!({"reason": "max_output_tokens"}),
            ),
            (
                "failed",
                serde_json::json!({"message": "model failed"}),
                serde_json::Value::Null,
            ),
        ] {
            let response: crate::ResponsePayload = serde_json::from_value(serde_json::json!({
                "id": "resp_upstream",
                "object": "response",
                "created_at": 0,
                "model": "test-model",
                "status": status,
                "output": [{
                    "id": "msg_partial",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "partial summary", "annotations": []}]
                }],
                "usage": null,
                "incomplete_details": incomplete_details,
                "error": error,
                "previous_response_id": null,
                "conversation_id": null,
                "instructions": null
            }))
            .expect("valid upstream response");

            let error = completed_summary_text(&response).expect_err("partial summary must be rejected");
            assert!(matches!(error, crate::executor::ExecutorError::CompactionFailed { .. }));
        }
    }

    #[test]
    fn completed_response_without_summary_is_an_upstream_failure() {
        let response: crate::ResponsePayload = serde_json::from_value(serde_json::json!({
            "id": "resp_upstream",
            "object": "response",
            "created_at": 0,
            "model": "test-model",
            "status": "completed",
            "output": [],
            "usage": null,
            "incomplete_details": null,
            "error": null,
            "previous_response_id": null,
            "conversation_id": null,
            "instructions": null
        }))
        .expect("valid upstream response");

        let error = completed_summary_text(&response).expect_err("missing summary must be rejected");

        assert!(matches!(
            error,
            crate::executor::ExecutorError::CompactionFailed {
                ref status,
                ref details
            } if status == "completed" && details.contains("no summary text")
        ));
        assert_eq!(error.http_status(), http::StatusCode::BAD_GATEWAY);
        assert_eq!(error.error_code(), "upstream_error");
    }

    #[tokio::test]
    async fn disabled_storage_does_not_fail_compaction() {
        let (exec_ctx, server) = mock_execution_context(ResponseStore::disabled()).await;

        let response = compact_response(compact_request(), &exec_ctx, None)
            .await
            .expect("disabled persistence should be ignored");

        assert_eq!(response.object, "response.compaction");
        assert_eq!(response.usage.total_tokens, 15);
        assert!(response.id.starts_with("resp_"));
        assert!(
            matches!(response.output.last(), Some(InputItem::Compaction(item)) if item.encrypted_content == "durable summary")
        );
        server.abort();
    }

    #[tokio::test]
    async fn empty_context_is_rejected_before_summarization() {
        let (exec_ctx, server) = mock_execution_context(ResponseStore::disabled()).await;
        for input in [
            serde_json::json!(""),
            serde_json::json!([{"role": "user", "content": "   "}]),
            serde_json::json!([{"type": "future_item"}]),
        ] {
            let request = serde_json::from_value(serde_json::json!({
                "model": "test-model",
                "input": input
            }))
            .expect("structurally valid compact request");
            let error = compact_response(request, &exec_ctx, None)
                .await
                .expect_err("empty context must fail");
            assert!(matches!(error, crate::executor::ExecutorError::InvalidRequest(_)));
        }
        server.abort();
    }

    #[tokio::test]
    async fn compaction_persists_a_reusable_response_checkpoint() {
        let pool = create_pool_with_schema(Some("sqlite::memory:"))
            .await
            .expect("create response store");
        let response_store = ResponseStore::new(pool);
        let (exec_ctx, server) = mock_execution_context(response_store.clone()).await;

        let response = compact_response(compact_request(), &exec_ctx, None)
            .await
            .expect("compaction succeeds");
        let history = response_store
            .rehydrate(&response.id)
            .await
            .expect("compaction checkpoint can be rehydrated");

        assert_eq!(history.len(), 2);
        assert!(matches!(history[0], InOutItem::Input(InputItem::Message(_))));
        assert!(matches!(history[1], InOutItem::Input(InputItem::Compaction(_))));
        server.abort();
    }

    #[tokio::test]
    async fn compaction_resolves_and_replaces_previous_response_history() {
        let pool = create_pool_with_schema(Some("sqlite::memory:"))
            .await
            .expect("create response store");
        let response_store = ResponseStore::new(pool);
        response_store
            .persist(
                "resp_previous",
                None,
                vec![InOutItem::Input(user_message("remember banana"))],
                &ResponseMetadata {
                    model: "test-model".to_owned(),
                    previous_response_id: None,
                    effective_tools: None,
                    effective_tool_choice: crate::ToolChoice::Auto,
                    effective_instructions: None,
                },
            )
            .await
            .expect("seed previous response");
        let (exec_ctx, server) = mock_execution_context(response_store.clone()).await;
        let request = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "previous_response_id": "resp_previous"
        }))
        .expect("valid previous-response compact request");

        let response = compact_response(request, &exec_ctx, None)
            .await
            .expect("previous response compacts");
        let history = response_store
            .rehydrate(&response.id)
            .await
            .expect("compacted continuation rehydrates");
        let model_input = ResponsesInput::Items(InOutItem::into_input_items(history));
        let serialized = serde_json::to_value(model_input.model_input()).expect("model input serializes");

        assert_eq!(serialized.as_array().map(Vec::len), Some(2));
        assert_eq!(serialized[0]["content"], "remember banana");
        assert_eq!(serialized[1]["role"], "assistant");
        assert_eq!(serialized[1]["content"][0]["text"], "durable summary");
        server.abort();
    }
}
