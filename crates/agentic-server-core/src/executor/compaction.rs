use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::rehydrate::rehydrate_conversation;
use crate::executor::request::{ExecutionContext, RequestContext};
use crate::executor::upstream::fetch_blocking_payload;
use crate::types::event::MessageStatus;
use crate::types::io::input::latest_compaction_window;
use crate::types::io::{
    CompactionItem, InputContent, InputItem, InputMessage, InputMessageContent, OutputItem, ResponseUsage,
    ResponsesInput, ToolCallOutput, ToolOutputContent,
};
use crate::types::request_response::{CompactRequest, CompactedResponse, RequestPayload, ResponsePayload};
use crate::utils::common::{serialize_to_string, utcnow_str, uuid7_str};

const COMPACTION_PROMPT: &str = "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a concise handoff summary that preserves current progress, decisions, constraints, unresolved work, and critical references for the next model. Return only the summary.";
const ESTIMATED_BYTES_PER_TOKEN: u64 = 4;
const ESTIMATED_INPUT_OVERHEAD_TOKENS: u64 = 1;
const ESTIMATED_ITEM_OVERHEAD_TOKENS: u64 = 12;
const ESTIMATED_CONTENT_PART_OVERHEAD_TOKENS: u64 = 7;
const ESTIMATED_JSON_VALUE_OVERHEAD_TOKENS: u64 = 1;

/// Fixed model-agnostic allowance for one image, including its content-part framing.
///
/// Actual vision-token usage depends on the model, processor, image dimensions, and detail
/// setting. A conservative fixed budget avoids treating images as free without mistaking their
/// base64 transport encoding for model-visible text.
const ESTIMATED_IMAGE_TOKENS: u64 = 1_024;

#[derive(Default)]
struct InputTokenEstimate {
    text_bytes: u64,
    fixed_tokens: u64,
}

impl InputTokenEstimate {
    fn add_text(&mut self, text: &str) {
        let bytes = u64::try_from(text.len()).unwrap_or(u64::MAX);
        self.text_bytes = self.text_bytes.saturating_add(bytes);
    }

    fn add_optional_text(&mut self, text: Option<&str>) {
        if let Some(text) = text {
            self.add_text(text);
        }
    }

    const fn add_tokens(&mut self, tokens: u64) {
        self.fixed_tokens = self.fixed_tokens.saturating_add(tokens);
    }

    fn add_json_value(&mut self, value: &serde_json::Value) {
        self.add_tokens(ESTIMATED_JSON_VALUE_OVERHEAD_TOKENS);
        match value {
            serde_json::Value::String(text) => self.add_text(text),
            serde_json::Value::Array(values) => {
                for value in values {
                    self.add_json_value(value);
                }
            }
            serde_json::Value::Object(values) => {
                for (key, value) in values {
                    self.add_text(key);
                    self.add_json_value(value);
                }
            }
            serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
        }
    }

    fn total_tokens(self) -> u64 {
        let text_tokens =
            self.text_bytes / ESTIMATED_BYTES_PER_TOKEN + u64::from(self.text_bytes % ESTIMATED_BYTES_PER_TOKEN != 0);
        self.fixed_tokens.saturating_add(text_tokens)
    }
}

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

fn add_message_content(estimate: &mut InputTokenEstimate, content: &InputMessageContent) {
    match content {
        InputMessageContent::Text(text) => estimate.add_text(text),
        InputMessageContent::Parts(parts) => {
            for part in parts {
                match part {
                    InputContent::InputText(text)
                    | InputContent::OutputText(text)
                    | InputContent::ReasoningText(text) => {
                        estimate.add_tokens(ESTIMATED_CONTENT_PART_OVERHEAD_TOKENS);
                        estimate.add_text(&text.text);
                    }
                    InputContent::InputImage(_) => estimate.add_tokens(ESTIMATED_IMAGE_TOKENS),
                    InputContent::Unknown => estimate.add_tokens(ESTIMATED_CONTENT_PART_OVERHEAD_TOKENS),
                }
            }
        }
    }
}

fn add_tool_call_output(estimate: &mut InputTokenEstimate, output: &ToolCallOutput) {
    match output {
        ToolCallOutput::Text(text) => estimate.add_text(text),
        ToolCallOutput::Content(parts) => {
            for part in parts {
                match part {
                    ToolOutputContent::InputText(text) => {
                        estimate.add_tokens(ESTIMATED_CONTENT_PART_OVERHEAD_TOKENS);
                        estimate.add_text(&text.text);
                    }
                    ToolOutputContent::InputImage(_) => estimate.add_tokens(ESTIMATED_IMAGE_TOKENS),
                    ToolOutputContent::InputFile(file) => {
                        estimate.add_tokens(ESTIMATED_CONTENT_PART_OVERHEAD_TOKENS);
                        estimate.add_optional_text(file.file_data.as_deref());
                        estimate.add_optional_text(file.file_id.as_deref());
                        estimate.add_optional_text(file.file_url.as_deref());
                        estimate.add_optional_text(file.filename.as_deref());
                        estimate.add_optional_text(file.detail.as_deref());
                    }
                }
            }
        }
    }
}

fn add_input_item(estimate: &mut InputTokenEstimate, item: &InputItem) {
    if matches!(item, InputItem::McpListTools(_) | InputItem::CompactionTrigger) {
        return;
    }
    estimate.add_tokens(ESTIMATED_ITEM_OVERHEAD_TOKENS);

    match item {
        InputItem::Message(message) => {
            estimate.add_optional_text(message.id.as_deref());
            estimate.add_text(&message.role);
            add_message_content(estimate, &message.content);
        }
        InputItem::FunctionCall(call) => {
            estimate.add_optional_text(call.id.as_deref());
            estimate.add_text(&call.call_id);
            estimate.add_text(&call.name);
            estimate.add_optional_text(call.namespace.as_deref());
            estimate.add_text(&call.arguments);
        }
        InputItem::FunctionCallOutput(output) => {
            estimate.add_text(&output.call_id);
            add_tool_call_output(estimate, &output.output);
        }
        InputItem::CustomToolCall(call) => {
            estimate.add_text(&call.id);
            estimate.add_text(&call.call_id);
            estimate.add_text(&call.name);
            estimate.add_text(&call.input);
        }
        InputItem::CustomToolCallOutput(output) => {
            estimate.add_text(&output.call_id);
            estimate.add_optional_text(output.name.as_deref());
            add_tool_call_output(estimate, &output.output);
        }
        InputItem::Reasoning(reasoning) => {
            estimate.add_text(&reasoning.id);
            estimate.add_optional_text(reasoning.status.as_deref());
            for content in &reasoning.content {
                estimate.add_tokens(ESTIMATED_CONTENT_PART_OVERHEAD_TOKENS);
                estimate.add_text(&content.text);
            }
            for summary in &reasoning.summary {
                estimate.add_json_value(summary);
            }
            if let Some(encrypted_content) = &reasoning.encrypted_content {
                estimate.add_json_value(encrypted_content);
            }
        }
        InputItem::Compaction(compaction) => {
            // `model_input` presents the checkpoint as one assistant output-text message.
            estimate.add_text("assistant");
            estimate.add_tokens(ESTIMATED_CONTENT_PART_OVERHEAD_TOKENS);
            estimate.add_text(&compaction.encrypted_content);
        }
        InputItem::Unknown | InputItem::McpListTools(_) | InputItem::CompactionTrigger => {}
    }
}

/// Estimate the current model-facing context size without requiring a model-specific tokenizer.
///
/// Textual fields are aggregated at four UTF-8 bytes per token, with fixed allowances for
/// Responses framing. Images receive [`ESTIMATED_IMAGE_TOKENS`] each; their URLs and inline bytes
/// are deliberately excluded because vision-token usage is unrelated to base64 transport size.
#[must_use]
pub(crate) fn estimate_input_tokens(input: &ResponsesInput) -> u64 {
    let mut estimate = InputTokenEstimate::default();
    estimate.add_tokens(ESTIMATED_INPUT_OVERHEAD_TOKENS);
    match input {
        ResponsesInput::Text(text) => estimate.add_text(text),
        ResponsesInput::Items(_) => {
            for item in input.model_items() {
                add_input_item(&mut estimate, item);
            }
        }
    }
    estimate.total_tokens()
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
    use crate::executor::request::{ExecutionContext, RequestContext};
    use crate::storage::{ConversationStore, InOutItem, ResponseMetadata, ResponseStore, create_pool_with_schema};
    use crate::types::event::MessageStatus;
    use crate::types::io::{
        CompactionItem, CustomToolCallOutputMessage, FunctionToolResultMessage, InputContent, InputImageContent,
        InputItem, InputMessage, InputMessageContent, ResponsesInput, ToolCallOutput, ToolOutputContent,
    };
    use crate::types::request_response::ContextManagement;

    use super::{
        ESTIMATED_IMAGE_TOKENS, build_compacted_window, compact_response, completed_summary_text,
        estimate_input_tokens, maybe_compact_context, request_payload,
    };

    fn user_message(text: &str) -> InputItem {
        InputItem::Message(InputMessage {
            id: None,
            role: "user".to_owned(),
            status: None,
            content: InputMessageContent::Text(text.to_owned()),
        })
    }

    fn inline_image(encoded_bytes: usize) -> InputImageContent {
        InputImageContent {
            file_id: None,
            image_url: Some(format!("data:image/png;base64,{}", "A".repeat(encoded_bytes))),
            detail: Some("auto".to_owned()),
        }
    }

    fn image_message(encoded_bytes: usize) -> InputItem {
        InputItem::Message(InputMessage {
            id: None,
            role: "user".to_owned(),
            status: None,
            content: InputMessageContent::Parts(vec![InputContent::InputImage(inline_image(encoded_bytes))]),
        })
    }

    fn function_image_output(encoded_bytes: usize) -> InputItem {
        InputItem::FunctionCallOutput(FunctionToolResultMessage {
            call_id: "call_view_image".to_owned(),
            output: ToolCallOutput::Content(vec![ToolOutputContent::InputImage(inline_image(encoded_bytes))]),
        })
    }

    fn custom_image_output(encoded_bytes: usize) -> InputItem {
        InputItem::CustomToolCallOutput(CustomToolCallOutputMessage {
            call_id: "call_view_image".to_owned(),
            name: Some("view_image".to_owned()),
            output: ToolCallOutput::Content(vec![ToolOutputContent::InputImage(inline_image(encoded_bytes))]),
        })
    }

    fn context_with_threshold(input: ResponsesInput, threshold: u64) -> RequestContext {
        let original_request = request_payload("test-model".to_owned(), ResponsesInput::Items(Vec::new()), None);
        let mut enriched_request = request_payload("test-model".to_owned(), input, None);
        enriched_request.context_management = Some(vec![ContextManagement {
            type_: "compaction".to_owned(),
            compact_threshold: Some(threshold),
        }]);
        RequestContext {
            original_request,
            enriched_request,
            new_input_items: Vec::new(),
            response_id: "resp_test".to_owned(),
            conversation_id: None,
            conversation_version: None,
        }
    }

    fn assert_text_growth(cases: impl IntoIterator<Item = (&'static str, serde_json::Value, serde_json::Value)>) {
        for (label, short, long) in cases {
            let short: ResponsesInput = serde_json::from_value(short).expect("valid short input");
            let long: ResponsesInput = serde_json::from_value(long).expect("valid long input");
            assert!(
                estimate_input_tokens(&long) > estimate_input_tokens(&short),
                "{label} should contribute to the estimate"
            );
        }
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
    fn inline_image_payload_size_does_not_affect_user_message_estimate() {
        let small = estimate_input_tokens(&ResponsesInput::Items(vec![image_message(100 * 1_024)]));
        let large = estimate_input_tokens(&ResponsesInput::Items(vec![image_message(5 * 1_024 * 1_024)]));

        assert_eq!(small, large);
    }

    #[test]
    fn inline_image_payload_size_does_not_affect_structured_tool_output_estimate() {
        for make_output in [
            function_image_output as fn(usize) -> InputItem,
            custom_image_output as fn(usize) -> InputItem,
        ] {
            let small = estimate_input_tokens(&ResponsesInput::Items(vec![make_output(100 * 1_024)]));
            let large = estimate_input_tokens(&ResponsesInput::Items(vec![make_output(5 * 1_024 * 1_024)]));

            assert_eq!(small, large);
        }
    }

    #[test]
    fn each_image_adds_the_fixed_image_budget() {
        let estimate_with_images = |count| {
            let parts = (0..count).map(|_| InputContent::InputImage(inline_image(1))).collect();
            estimate_input_tokens(&ResponsesInput::Items(vec![InputItem::Message(InputMessage {
                id: None,
                role: "user".to_owned(),
                status: None,
                content: InputMessageContent::Parts(parts),
            })]))
        };

        let without_image = estimate_with_images(0);
        let one_image = estimate_with_images(1);
        let two_images = estimate_with_images(2);

        assert_eq!(one_image - without_image, ESTIMATED_IMAGE_TOKENS);
        assert_eq!(two_images - one_image, ESTIMATED_IMAGE_TOKENS);
    }

    #[test]
    fn message_and_tool_textual_fields_increase_estimates() {
        let long_text = "substantial context ".repeat(256);
        assert_text_growth([
            (
                "message text",
                serde_json::json!([{"role": "user", "content": "x"}]),
                serde_json::json!([{"role": "user", "content": long_text}]),
            ),
            (
                "message content part",
                serde_json::json!([{
                    "role": "user",
                    "content": [{"type": "input_text", "text": "x"}]
                }]),
                serde_json::json!([{
                    "role": "user",
                    "content": [{"type": "input_text", "text": long_text}]
                }]),
            ),
            (
                "function arguments",
                serde_json::json!([{
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{}"
                }]),
                serde_json::json!([{
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": long_text
                }]),
            ),
            (
                "function call output",
                serde_json::json!([{
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "x"
                }]),
                serde_json::json!([{
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": long_text
                }]),
            ),
            (
                "custom tool input",
                serde_json::json!([{
                    "type": "custom_tool_call",
                    "call_id": "call_1",
                    "name": "shell",
                    "input": "x"
                }]),
                serde_json::json!([{
                    "type": "custom_tool_call",
                    "call_id": "call_1",
                    "name": "shell",
                    "input": long_text
                }]),
            ),
            (
                "structured custom tool output",
                serde_json::json!([{
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "output": [{"type": "input_text", "text": "x"}]
                }]),
                serde_json::json!([{
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "output": [{"type": "input_text", "text": long_text}]
                }]),
            ),
        ]);
    }

    #[test]
    fn reasoning_textual_fields_increase_estimates() {
        let long_text = "substantial reasoning context ".repeat(256);
        assert_text_growth([
            (
                "assistant reasoning text",
                serde_json::json!([{
                    "type": "reasoning",
                    "id": "rs_1",
                    "content": [{"type": "reasoning_text", "text": "x"}],
                    "summary": []
                }]),
                serde_json::json!([{
                    "type": "reasoning",
                    "id": "rs_1",
                    "content": [{"type": "reasoning_text", "text": long_text}],
                    "summary": []
                }]),
            ),
            (
                "reasoning summary",
                serde_json::json!([{
                    "type": "reasoning",
                    "id": "rs_1",
                    "content": [],
                    "summary": [{"type": "summary_text", "text": "x"}]
                }]),
                serde_json::json!([{
                    "type": "reasoning",
                    "id": "rs_1",
                    "content": [],
                    "summary": [{"type": "summary_text", "text": long_text}]
                }]),
            ),
        ]);
    }

    #[test]
    fn text_only_estimate_remains_close_to_json_size_baseline() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "hello context"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{\"topic\":\"compaction\"}"
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "tool context"}
        ]))
        .expect("valid text-only input");
        let serialized = serde_json::to_string(&input.model_input()).expect("model input serializes");
        let serialized_bytes = u64::try_from(serialized.len()).expect("serialized fixture length fits in u64");
        let json_size_estimate = serialized_bytes.div_ceil(4);
        let typed_estimate = estimate_input_tokens(&input);

        assert!(
            typed_estimate.abs_diff(json_size_estimate) <= json_size_estimate.div_ceil(4),
            "typed estimate {typed_estimate} should remain within 25% of JSON estimate {json_size_estimate}"
        );
    }

    #[test]
    fn estimate_uses_only_the_effective_compacted_window() {
        let input_with_stale_text = |stale_text: String| {
            ResponsesInput::Items(vec![
                user_message(&stale_text),
                InputItem::Compaction(CompactionItem {
                    id: Some("cmp_1".to_owned()),
                    encrypted_content: "durable summary".to_owned(),
                }),
                user_message("continue"),
            ])
        };

        let short = input_with_stale_text("old".to_owned());
        let large = input_with_stale_text("old".repeat(100_000));

        assert_eq!(estimate_input_tokens(&short), estimate_input_tokens(&large));
    }

    #[tokio::test]
    async fn automatic_compaction_ignores_image_bytes_but_triggers_for_text_and_preserves_image() {
        let large_image_input = ResponsesInput::Items(vec![image_message(5 * 1_024 * 1_024)]);
        let image_estimate = estimate_input_tokens(&large_image_input);
        let threshold = image_estimate.saturating_add(100);
        let (exec_ctx, server) = mock_execution_context(ResponseStore::disabled()).await;
        let mut image_context = context_with_threshold(large_image_input, threshold);

        assert!(
            maybe_compact_context(&mut image_context, &exec_ctx, None)
                .await
                .expect("image-only threshold check succeeds")
                .is_none()
        );

        let retained_image = inline_image(100 * 1_024);
        let expected_url = retained_image.image_url.clone().expect("inline image URL");
        let text_input = ResponsesInput::Items(vec![
            InputItem::Message(InputMessage {
                id: None,
                role: "user".to_owned(),
                status: None,
                content: InputMessageContent::Parts(vec![InputContent::InputImage(retained_image)]),
            }),
            user_message(&"genuine textual context ".repeat(1_000)),
        ]);
        assert!(estimate_input_tokens(&text_input) > threshold);
        let mut text_context = context_with_threshold(text_input, threshold);

        assert!(
            maybe_compact_context(&mut text_context, &exec_ctx, None)
                .await
                .expect("long text compacts")
                .is_some()
        );
        let ResponsesInput::Items(compacted) = &text_context.enriched_request.input else {
            panic!("compaction should produce item input");
        };
        let retained_url = compacted.iter().find_map(|item| {
            let InputItem::Message(message) = item else {
                return None;
            };
            let InputMessageContent::Parts(parts) = &message.content else {
                return None;
            };
            parts.iter().find_map(|part| match part {
                InputContent::InputImage(image) => image.image_url.as_deref(),
                _ => None,
            })
        });
        assert_eq!(retained_url, Some(expected_url.as_str()));
        server.abort();
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
