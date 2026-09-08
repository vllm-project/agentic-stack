//! Step 1 of the conversation pipeline — history rehydration.
//!
//! Builds a [`RequestContext`] by loading prior turns from storage and
//! injecting them into the enriched request before it is forwarded to the LLM.

use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::pending_calls::pending_calls;
use crate::executor::request::{ExecutionContext, RequestContext};
use crate::storage::InOutItem;
use crate::tool::ToolError;
use crate::types::io::{
    InputContent, InputItem, InputMessageContent, ReasoningOutput, ReasoningTextContent, ResponsesInput,
    resolve_tool_choice, resolve_tools,
};
use crate::types::request_response::RequestPayload;
use crate::utils::uuid7_str;

/// Reject unsupported message files on typed paths, including restored history.
///
/// Keep this out of deserialization: eligible raw proxy requests must retain their
/// original bytes and leave support decisions to the upstream. Structured tool
/// call outputs have a separate content type and are deliberately not rejected.
pub(super) fn validate_message_files(input: &ResponsesInput) -> ExecutorResult<()> {
    let ResponsesInput::Items(items) = input else {
        return Ok(());
    };
    let has_file = items.iter().any(|item| {
        matches!(item, InputItem::Message(message)
            if matches!(&message.content, InputMessageContent::Parts(parts)
                if parts.iter().any(|part| matches!(part, InputContent::InputFile(_)))))
    });
    if has_file {
        return Err(ExecutorError::InvalidRequest(
            "input_file content in messages is not supported by the typed Responses executor; provide input_text or input_image content instead"
                .to_owned(),
        ));
    }
    Ok(())
}

fn has_plaintext_reasoning(reasoning: &ReasoningOutput) -> bool {
    reasoning.content.iter().any(|content| !content.text.is_empty())
}

fn has_opaque_reasoning_state(reasoning: &ReasoningOutput) -> bool {
    reasoning
        .encrypted_content
        .as_ref()
        .is_some_and(|encrypted| !encrypted.is_null())
}

/// Reject opaque reasoning that vLLM cannot replay before any normal inference call.
pub(super) fn validate_reasoning_for_vllm(input: &ResponsesInput) -> ExecutorResult<()> {
    let ResponsesInput::Items(items) = input else {
        return Ok(());
    };

    if items.iter().any(|item| {
        matches!(item, InputItem::Reasoning(reasoning) if has_opaque_reasoning_state(reasoning) && !has_plaintext_reasoning(reasoning))
    }) {
        return Err(ExecutorError::InvalidRequest(
            "reasoning item contains encrypted state without plaintext reasoning content and cannot be replayed to vLLM"
                .to_owned(),
        ));
    }

    Ok(())
}

/// Prepare reasoning in the vLLM-bound request copy.
///
/// vLLM can replay plaintext reasoning content but cannot interpret opaque provider state. Its generic Responses
/// conversion reads only the first reasoning content part and falls back to a summary when content is absent, while
/// its Harmony conversion joins all content parts. Normalize usable plaintext into one ordered, newline-delimited part
/// and remove summaries so both paths receive the same continuation state. Summary-only items have no usable vLLM
/// state and are omitted. [`RequestContext`] keeps the original request and new input items separately, so none of
/// these changes mutate persisted state.
pub(super) fn prepare_reasoning_for_vllm(input: &mut ResponsesInput) -> ExecutorResult<()> {
    // Validate the complete input before mutation so an error never leaves a partially prepared request behind.
    validate_reasoning_for_vllm(input)?;

    let ResponsesInput::Items(items) = input else {
        return Ok(());
    };

    items.retain_mut(|item| {
        let InputItem::Reasoning(reasoning) = item else {
            return true;
        };
        if !has_plaintext_reasoning(reasoning) {
            return false;
        }

        let plaintext = reasoning
            .content
            .iter()
            .map(|content| content.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        reasoning.content = vec![ReasoningTextContent::new(plaintext)];
        reasoning.summary.clear();
        reasoning.encrypted_content = None;
        true
    });
    Ok(())
}

/// Step 1 — Build [`RequestContext`] by rehydrating conversation history.
///
/// `request` is moved into the context as `enriched_request`; one clone is taken
/// for `original_request` so the engine retains an unmodified copy for persistence
/// and ID resolution.
///
/// Dispatches based on `store` flag and which ID is present:
/// - `previous_response_id`: rehydrate from the prior response checkpoint
/// - `conversation_id`:      rehydrate from the conversation
/// - no ids:                 forward only the new input
///
/// # Errors
/// Returns [`ExecutorError`] if storage is unavailable, a referenced ID does not exist,
/// or the new/resolved input contains unsupported message files.
pub async fn rehydrate_conversation(
    request: RequestPayload,
    exec_ctx: &ExecutionContext,
) -> ExecutorResult<RequestContext> {
    // Fail before storage work for new files; check again once history is resolved.
    validate_message_files(&request.input)?;
    let response_id = uuid7_str("resp_");
    let new_input_items: Vec<InputItem> = Vec::from(&request.input);

    // One clone for the unmodified original; `request` is moved as enriched_request.
    let original_request = request.clone();
    let mut ctx = RequestContext {
        enriched_request: request,
        original_request,
        new_input_items,
        response_id,
        conversation_id: None,
        conversation_version: None,
    };

    if ctx.original_request.conversation_id.is_some() && ctx.original_request.previous_response_id.is_some() {
        return Err(ExecutorError::InvalidRequest(
            "provide only one of conversation_id or previous_response_id".into(),
        ));
    }

    if ctx.original_request.conversation_id.is_some() {
        from_conversation(&mut ctx, exec_ctx).await?;
    } else if ctx.original_request.previous_response_id.is_some() {
        from_response(&mut ctx, exec_ctx).await?;
    } else {
        ctx.enriched_request.input = ResponsesInput::Items(ctx.new_input_items.clone());
    }

    validate_message_files(&ctx.enriched_request.input)?;
    Ok(ctx)
}

/// Hydrates `ctx` from the previous response chain.
///
/// Loads the stored response, rehydrates its history items, resolves effective
/// tools and tool choice from the stored metadata, and prepends the history to
/// the enriched request input.
async fn from_response(ctx: &mut RequestContext, exec_ctx: &ExecutionContext) -> ExecutorResult<()> {
    let stored = exec_ctx.resp_handler.get(ctx).await?;
    let history = exec_ctx.resp_handler.rehydrate(ctx).await?;

    let mut items = InOutItem::into_input_items(history);
    items.reserve(ctx.new_input_items.len());
    items.extend(ctx.new_input_items.iter().cloned());
    if let Some(pending) = pending_calls(&items)?.into_iter().next() {
        return Err(ExecutorError::Tool(ToolError::MissingOutput {
            call_id: pending.call_id,
        }));
    }

    ctx.enriched_request.previous_response_id = None;
    ctx.enriched_request.input = ResponsesInput::Items(items);
    ctx.enriched_request.tools = resolve_tools(
        ctx.original_request.tools.as_deref(),
        stored.metadata.effective_tools.as_deref(),
        ctx.original_request.tools.is_some(),
    );
    ctx.enriched_request.tool_choice = Some(resolve_tool_choice(
        ctx.original_request.tool_choice.as_ref(),
        &stored.metadata.effective_tool_choice,
        ctx.original_request.tool_choice.is_some(),
    ));
    ctx.conversation_id = stored.conversation_id;
    Ok(())
}

/// Hydrates `ctx` from the conversation store.
///
/// Gets or creates the conversation (depending on `store`) and rehydrates its
/// history in parallel, then prepends the history items to the enriched request input.
async fn from_conversation(ctx: &mut RequestContext, exec_ctx: &ExecutionContext) -> ExecutorResult<()> {
    let (conv_data, snapshot) = tokio::try_join!(
        async {
            if ctx.original_request.store {
                exec_ctx.conv_handler.get_or_create(ctx).await
            } else {
                exec_ctx.conv_handler.get(ctx).await
            }
        },
        exec_ctx.conv_handler.rehydrate_snapshot(ctx),
    )?;

    let mut items = InOutItem::into_input_items(snapshot.items);
    items.reserve(ctx.new_input_items.len());
    items.extend(ctx.new_input_items.iter().cloned());
    if let Some(pending) = pending_calls(&items)?.into_iter().next() {
        return Err(ExecutorError::Tool(ToolError::MissingOutput {
            call_id: pending.call_id,
        }));
    }

    ctx.enriched_request.input = ResponsesInput::Items(items);
    ctx.conversation_id = Some(conv_data.conversation_id);
    ctx.conversation_version = Some(snapshot.version);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::executor::modes::{ConversationHandler, ResponseHandler};
    use crate::storage::{
        ConversationStore, ConversationVersion, InOutItem, ResponseMetadata, ResponseStore, create_pool_with_schema,
    };
    use crate::types::io::output::{McpListTools, OutputItem};
    use crate::types::request_response::RequestPayload;

    fn reasoning_item(content: &[&str], encrypted_content: Option<serde_json::Value>) -> InputItem {
        InputItem::Reasoning(ReasoningOutput {
            id: "rs_prior".to_owned(),
            content: content.iter().map(|text| ReasoningTextContent::new(*text)).collect(),
            summary: vec![serde_json::json!({"type": "summary_text", "text": "public summary"})],
            encrypted_content,
            status: Some("completed".to_owned()),
        })
    }

    #[test]
    fn plaintext_reasoning_is_normalized_for_both_vllm_paths() {
        let mut input = ResponsesInput::Items(vec![reasoning_item(
            &["first continuation part", "second continuation part"],
            Some(serde_json::json!({"ciphertext": "opaque-provider-state"})),
        )]);

        prepare_reasoning_for_vllm(&mut input).expect("plaintext reasoning is replayable");

        let ResponsesInput::Items(items) = input else {
            panic!("expected structured input");
        };
        let InputItem::Reasoning(reasoning) = &items[0] else {
            panic!("expected reasoning item");
        };
        assert_eq!(reasoning.id, "rs_prior");
        assert_eq!(reasoning.content.len(), 1);
        assert_eq!(
            reasoning.content[0].text,
            "first continuation part\nsecond continuation part"
        );
        assert!(reasoning.summary.is_empty());
        assert_eq!(reasoning.status.as_deref(), Some("completed"));
        assert_eq!(reasoning.encrypted_content, None);
    }

    #[test]
    fn encrypted_reasoning_requires_nonempty_plaintext_content() {
        for content in [Vec::new(), vec![""], vec!["", ""]] {
            let mut input = ResponsesInput::Items(vec![reasoning_item(
                &content,
                Some(serde_json::json!("opaque-provider-state")),
            )]);

            let error =
                prepare_reasoning_for_vllm(&mut input).expect_err("encrypted-only reasoning must not reach vLLM");

            assert_eq!(error.http_status(), http::StatusCode::BAD_REQUEST);
            assert!(
                error
                    .to_string()
                    .contains("encrypted state without plaintext reasoning content")
            );
            assert!(!error.to_string().contains("opaque-provider-state"));
        }
    }

    #[test]
    fn plaintext_reasoning_with_null_encrypted_state_is_normalized_without_summary() {
        let mut item = reasoning_item(&["plaintext continuation"], Some(serde_json::Value::Null));
        let InputItem::Reasoning(reasoning) = &mut item else {
            panic!("expected reasoning item");
        };
        reasoning.content[0].type_ = "unexpected_provider_type".to_owned();
        let mut input = ResponsesInput::Items(vec![item]);

        prepare_reasoning_for_vllm(&mut input).expect("null encrypted state is valid");

        let ResponsesInput::Items(items) = input else {
            panic!("expected structured input");
        };
        let InputItem::Reasoning(reasoning) = &items[0] else {
            panic!("expected reasoning item");
        };
        assert_eq!(reasoning.content[0].type_, "reasoning_text");
        assert_eq!(reasoning.content[0].text, "plaintext continuation");
        assert!(reasoning.summary.is_empty());
        assert_eq!(reasoning.encrypted_content, None);
    }

    #[test]
    fn summary_only_reasoning_without_opaque_state_is_removed_from_vllm_copy() {
        for encrypted_content in [None, Some(serde_json::Value::Null)] {
            let mut input = ResponsesInput::Items(vec![reasoning_item(&[], encrypted_content)]);

            prepare_reasoning_for_vllm(&mut input).expect("summary-only reasoning has no usable vLLM state");

            let ResponsesInput::Items(items) = input else {
                panic!("expected structured input");
            };
            assert!(
                items.is_empty(),
                "a reasoning summary must never be promoted to reasoning text"
            );
        }
    }

    #[test]
    fn validation_failure_does_not_partially_mutate_input() {
        let valid = reasoning_item(
            &["plaintext continuation"],
            Some(serde_json::json!("first-opaque-state")),
        );
        let invalid = reasoning_item(&[], Some(serde_json::json!("second-opaque-state")));
        let mut input = ResponsesInput::Items(vec![valid.clone(), invalid]);

        prepare_reasoning_for_vllm(&mut input).expect_err("the complete input must validate before normalization");

        let ResponsesInput::Items(items) = input else {
            panic!("expected structured input");
        };
        let (InputItem::Reasoning(actual), InputItem::Reasoning(expected)) = (&items[0], &valid) else {
            panic!("expected reasoning items");
        };
        assert_eq!(actual.content[0].text, expected.content[0].text);
        assert_eq!(actual.summary, expected.summary);
        assert_eq!(actual.encrypted_content, expected.encrypted_content);
    }

    #[test]
    fn text_input_is_unchanged() {
        let mut input = ResponsesInput::Text("plain user input".to_owned());

        prepare_reasoning_for_vllm(&mut input).expect("text input contains no reasoning item");

        assert!(matches!(input, ResponsesInput::Text(ref text) if text == "plain user input"));
    }

    fn request(conversation_id: Option<&str>, previous_response_id: Option<&str>) -> RequestPayload {
        RequestPayload {
            model: "test".into(),
            input: ResponsesInput::Text("new input".into()),
            instructions: None,
            previous_response_id: previous_response_id.map(str::to_owned),
            conversation_id: conversation_id.map(str::to_owned),
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
        }
    }

    fn execution_context(conversation_store: ConversationStore, response_store: ResponseStore) -> ExecutionContext {
        ExecutionContext::new(
            ConversationHandler::new(conversation_store),
            ResponseHandler::new(response_store),
            Arc::new(reqwest::Client::new()),
            "http://localhost:8000".to_owned(),
        )
    }

    #[tokio::test]
    async fn new_conversation_rehydration_captures_empty_version() -> Result<(), Box<dyn std::error::Error>> {
        let pool = create_pool_with_schema(Some("sqlite://?mode=memory")).await?;
        let conversation_store = ConversationStore::new(pool);
        let conversation = conversation_store.create().await?;
        let exec_ctx = execution_context(conversation_store, ResponseStore::disabled());

        let ctx = rehydrate_conversation(request(Some(&conversation.conversation_id), None), &exec_ctx).await?;

        assert_eq!(ctx.conversation_version, Some(ConversationVersion::Empty));
        Ok(())
    }

    #[tokio::test]
    async fn existing_conversation_rehydration_captures_last_sequence() -> Result<(), Box<dyn std::error::Error>> {
        let pool = create_pool_with_schema(Some("sqlite://?mode=memory")).await?;
        let conversation_store = ConversationStore::new(pool);
        let conversation = conversation_store.create().await?;
        let prior_items = Vec::<InputItem>::from(&ResponsesInput::Text("prior input".into()))
            .into_iter()
            .map(InOutItem::Input)
            .collect();
        conversation_store
            .persist(
                &conversation.conversation_id,
                "resp_prior",
                None,
                prior_items,
                &ResponseMetadata::default(),
            )
            .await?;
        let exec_ctx = execution_context(conversation_store, ResponseStore::disabled());

        let ctx = rehydrate_conversation(request(Some(&conversation.conversation_id), None), &exec_ctx).await?;

        assert_eq!(ctx.conversation_version, Some(ConversationVersion::LastSequence(0)));
        Ok(())
    }

    #[tokio::test]
    async fn conversation_rehydration_remembers_listed_mcp_servers() -> Result<(), Box<dyn std::error::Error>> {
        let pool = create_pool_with_schema(Some("sqlite://?mode=memory")).await?;
        let conversation_store = ConversationStore::new(pool);
        let conversation = conversation_store.create().await?;
        conversation_store
            .persist(
                &conversation.conversation_id,
                "resp_prior",
                None,
                vec![InOutItem::Output(OutputItem::McpListTools(McpListTools::new(
                    "mcpl_prior",
                    "counter",
                    Vec::new(),
                )))],
                &ResponseMetadata::default(),
            )
            .await?;
        let exec_ctx = execution_context(conversation_store, ResponseStore::disabled());

        let ctx = rehydrate_conversation(request(Some(&conversation.conversation_id), None), &exec_ctx).await?;

        let ResponsesInput::Items(items) = &ctx.enriched_request.input else {
            panic!("rehydrated input should contain items");
        };
        assert!(
            matches!(items.first(), Some(InputItem::McpListTools(list_tools)) if list_tools.server_label == "counter")
        );
        let ResponsesInput::Items(model_items) = ctx.enriched_request.input.model_input().into_owned() else {
            panic!("model input should contain items");
        };
        assert!(
            model_items
                .iter()
                .all(|item| !matches!(item, InputItem::McpListTools(_)))
        );
        Ok(())
    }

    #[tokio::test]
    async fn request_without_continuation_has_no_conversation_version() -> Result<(), Box<dyn std::error::Error>> {
        let exec_ctx = execution_context(ConversationStore::disabled(), ResponseStore::disabled());

        let ctx = rehydrate_conversation(request(None, None), &exec_ctx).await?;

        assert_eq!(ctx.conversation_version, None);
        Ok(())
    }

    #[tokio::test]
    async fn previous_response_rehydration_has_no_conversation_version() -> Result<(), Box<dyn std::error::Error>> {
        let pool = create_pool_with_schema(Some("sqlite://?mode=memory")).await?;
        let response_store = ResponseStore::new(pool);
        response_store
            .persist("resp_prior", None, Vec::new(), &ResponseMetadata::default())
            .await?;
        let exec_ctx = execution_context(ConversationStore::disabled(), response_store);

        let ctx = rehydrate_conversation(request(None, Some("resp_prior")), &exec_ctx).await?;

        assert_eq!(ctx.conversation_version, None);
        Ok(())
    }

    #[tokio::test]
    async fn previous_response_rehydration_remembers_listed_mcp_servers() -> Result<(), Box<dyn std::error::Error>> {
        let pool = create_pool_with_schema(Some("sqlite://?mode=memory")).await?;
        let response_store = ResponseStore::new(pool);
        response_store
            .persist(
                "resp_prior",
                None,
                vec![InOutItem::Output(OutputItem::McpListTools(McpListTools::new(
                    "mcpl_prior",
                    "counter",
                    Vec::new(),
                )))],
                &ResponseMetadata::default(),
            )
            .await?;
        let exec_ctx = execution_context(ConversationStore::disabled(), response_store);

        let ctx = rehydrate_conversation(request(None, Some("resp_prior")), &exec_ctx).await?;

        let ResponsesInput::Items(items) = &ctx.enriched_request.input else {
            panic!("rehydrated input should contain items");
        };
        assert!(
            matches!(items.first(), Some(InputItem::McpListTools(list_tools)) if list_tools.server_label == "counter")
        );
        let ResponsesInput::Items(model_items) = ctx.enriched_request.input.model_input().into_owned() else {
            panic!("model input should contain items");
        };
        assert!(
            model_items
                .iter()
                .all(|item| !matches!(item, InputItem::McpListTools(_)))
        );
        Ok(())
    }
}
