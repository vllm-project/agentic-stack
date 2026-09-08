use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::event::MessageStatus;
use crate::utils::common::deserialize_from_value;

use super::output::{CustomToolCall, FunctionToolCall, McpListTools, ReasoningOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputTextContent {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputImageContent {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputFileContent {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl InputFileContent {
    /// Whether the part names a file at all. The gateway never fetches or
    /// decodes the file; it only needs to know the part is not an empty shell
    /// when deciding whether an item still carries context.
    #[must_use]
    pub fn has_reference(&self) -> bool {
        [
            self.file_data.as_deref(),
            self.file_id.as_deref(),
            self.file_url.as_deref(),
            self.filename.as_deref(),
        ]
        .into_iter()
        .flatten()
        .any(|value| !value.trim().is_empty())
    }
}

/// Content item inside a message input.
///
/// Uses an internally-tagged enum — serde consumes `"type"` for the variant
/// discriminant so the inner structs must NOT redeclare a `type_` field.
/// `output_text` and `reasoning_text` reuse `InputTextContent` since they
/// carry only a `text` field; they are preserved so vLLM sees the full history.
/// `input_image` and `input_file` mirror the media parts [`ToolOutputContent`]
/// models, so a media part keeps its structure whether it arrives in a message
/// or as a client-owned tool result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContent {
    InputText(InputTextContent),
    InputImage(InputImageContent),
    InputFile(InputFileContent),
    /// Assistant output text in rehydrated history.
    OutputText(InputTextContent),
    /// Reasoning step text in rehydrated history.
    ReasoningText(InputTextContent),
    /// A content type this gateway does not model.
    ///
    /// The variant is unit-only, so it cannot round-trip the part it replaced:
    /// serializing it would forward a synthetic `{"type": "unknown"}` part that
    /// no client sent. [`ResponsesInput::model_input`] drops it instead, which
    /// keeps the surrounding message — and the ordering of the parts the
    /// gateway does model — intact.
    #[serde(other)]
    Unknown,
}

impl InputContent {
    /// Whether this part is a content type the gateway cannot represent.
    #[must_use]
    pub const fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMessage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<MessageStatus>,
    pub content: InputMessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputMessageContent {
    Text(String),
    Parts(Vec<InputContent>),
}

impl InputMessageContent {
    /// Whether any part is a content type the gateway cannot represent.
    #[must_use]
    fn has_unknown_parts(&self) -> bool {
        matches!(self, Self::Parts(parts) if parts.iter().any(InputContent::is_unknown))
    }

    /// The same content with unrepresentable parts removed, preserving the
    /// order of the parts the gateway does model.
    #[must_use]
    fn without_unknown_parts(&self) -> Self {
        match self {
            Self::Text(text) => Self::Text(text.clone()),
            Self::Parts(parts) => Self::Parts(parts.iter().filter(|part| !part.is_unknown()).cloned().collect()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionToolResultMessage {
    pub call_id: String,
    pub output: ToolCallOutput,
}

/// Text or structured content returned by a client-owned tool call.
///
/// The Responses API accepts either a string or an array containing text,
/// image, and file input content. Keeping the array structured preserves its
/// media semantics when a custom-tool output is normalized to a function-tool
/// output for the upstream model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolCallOutput {
    Text(String),
    Content(Vec<ToolOutputContent>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolOutputContent {
    InputText(InputTextContent),
    InputImage(InputImageContent),
    InputFile(InputFileContent),
}

impl ToolCallOutput {
    #[must_use]
    pub fn has_content(&self) -> bool {
        match self {
            Self::Text(text) => !text.trim().is_empty(),
            Self::Content(content) => !content.is_empty(),
        }
    }
}

impl From<String> for ToolCallOutput {
    fn from(output: String) -> Self {
        Self::Text(output)
    }
}

impl From<&str> for ToolCallOutput {
    fn from(output: &str) -> Self {
        Self::Text(output.to_owned())
    }
}

/// A model-generated function call replayed as Responses input.
///
/// Input replay is intentionally more permissive than [`FunctionToolCall`]
/// output: clients may omit `id` and `status` when passing prior items to a
/// later request or to the compact endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputFunctionToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub call_id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    pub arguments: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<MessageStatus>,
}

impl From<FunctionToolCall> for InputFunctionToolCall {
    fn from(call: FunctionToolCall) -> Self {
        Self {
            id: Some(call.id),
            call_id: call.call_id,
            name: call.name,
            namespace: call.namespace,
            arguments: call.arguments,
            status: Some(call.status),
        }
    }
}

impl From<CustomToolCall> for InputFunctionToolCall {
    fn from(call: CustomToolCall) -> Self {
        Self {
            id: function_call_item_id(&call.id),
            call_id: call.call_id,
            name: call.name,
            namespace: None,
            arguments: serde_json::json!({ "input": call.input }).to_string(),
            status: call.status,
        }
    }
}

/// An opaque compacted context checkpoint accepted as Responses input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub encrypted_content: String,
}

/// Client result for a freeform custom tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomToolCallOutputMessage {
    pub call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub output: ToolCallOutput,
}

impl From<CustomToolCallOutputMessage> for FunctionToolResultMessage {
    fn from(output: CustomToolCallOutputMessage) -> Self {
        Self {
            call_id: output.call_id,
            output: output.output,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum InputItem {
    #[serde(rename = "message")]
    Message(InputMessage),
    /// The model's tool invocation — appears in rehydrated history so vLLM sees
    /// the full call/output pair across turns.
    #[serde(rename = "function_call")]
    FunctionCall(InputFunctionToolCall),
    #[serde(rename = "function_call_output")]
    FunctionCallOutput(FunctionToolResultMessage),
    /// The public freeform invocation accepted from a client request.
    #[serde(rename = "custom_tool_call")]
    CustomToolCall(CustomToolCall),
    #[serde(rename = "custom_tool_call_output")]
    CustomToolCallOutput(CustomToolCallOutputMessage),
    #[serde(rename = "reasoning")]
    Reasoning(ReasoningOutput),
    /// Internal history record used by gateway orchestration to remember that
    /// an MCP server's tools were already listed. It is never sent to the model.
    #[serde(rename = "mcp_list_tools")]
    McpListTools(McpListTools),
    #[serde(rename = "compaction")]
    Compaction(CompactionItem),
    /// Codex CLI's remote-compaction V2 marker. Signals the server to run its
    /// own summarization turn and return exactly one `compaction` output item.
    /// Carries no payload; it is never forwarded to the upstream model.
    #[serde(rename = "compaction_trigger")]
    CompactionTrigger,
    #[serde(other)]
    Unknown,
}

impl<'de> Deserialize<'de> for InputItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let item = match value.get("type").and_then(Value::as_str) {
            None | Some("message") => deserialize_from_value(value).map(Self::Message),
            Some("function_call") => deserialize_from_value(value).map(Self::FunctionCall),
            Some("function_call_output") => deserialize_from_value(value).map(Self::FunctionCallOutput),
            Some("custom_tool_call") => deserialize_from_value(value).map(Self::CustomToolCall),
            Some("custom_tool_call_output") => deserialize_from_value(value).map(Self::CustomToolCallOutput),
            Some("reasoning") => deserialize_from_value(value).map(Self::Reasoning),
            Some("mcp_list_tools") => deserialize_from_value(value).map(Self::McpListTools),
            Some("compaction") => deserialize_from_value(value).map(Self::Compaction),
            Some("compaction_trigger") => Ok(Self::CompactionTrigger),
            Some(_) => return Ok(Self::Unknown),
        };
        item.map_err(serde::de::Error::custom)
    }
}

impl InputItem {
    #[must_use]
    pub(crate) fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }

    #[must_use]
    pub(crate) fn is_compaction_trigger(&self) -> bool {
        matches!(self, Self::CompactionTrigger)
    }

    #[must_use]
    pub(crate) fn is_model_visible(&self) -> bool {
        !matches!(self, Self::McpListTools(_) | Self::CompactionTrigger)
    }

    /// Whether the item carries message content the gateway cannot represent.
    #[must_use]
    pub(crate) fn has_unknown_content(&self) -> bool {
        matches!(self, Self::Message(message) if message.content.has_unknown_parts())
    }

    /// The item as the model should see it, with unrepresentable content parts
    /// dropped. The message itself is kept so the turn structure — and the
    /// order of the remaining parts — survives.
    #[must_use]
    fn without_unknown_content(&self) -> Self {
        let Self::Message(message) = self else {
            return self.clone();
        };
        if !message.content.has_unknown_parts() {
            return self.clone();
        }
        let mut message = message.clone();
        message.content = message.content.without_unknown_parts();
        Self::Message(message)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<InputItem>),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CompactionWindow {
    latest_index: usize,
    retained_start: usize,
}

impl CompactionWindow {
    #[must_use]
    pub(crate) const fn latest_index(self) -> usize {
        self.latest_index
    }

    #[must_use]
    pub(crate) fn retains_user_item(self, index: usize, item: &InputItem) -> bool {
        index >= self.retained_start
            && index < self.latest_index
            && matches!(item, InputItem::Message(message)
                if message.role == "user"
                    && message.id.is_some()
                    && message.status == Some(MessageStatus::Completed))
    }

    pub(crate) fn retained_user_items(self, items: &[InputItem]) -> impl Iterator<Item = &InputItem> {
        items
            .iter()
            .enumerate()
            .filter(move |(index, item)| self.retains_user_item(*index, item))
            .map(|(_, item)| item)
    }
}

#[must_use]
pub(crate) fn latest_compaction_window(items: &[InputItem]) -> Option<CompactionWindow> {
    let latest_index = items
        .iter()
        .rposition(|item| matches!(item, InputItem::Compaction(_)))?;
    let retained_start = items[..latest_index]
        .iter()
        .rposition(|item| matches!(item, InputItem::Compaction(_)))
        .map_or(0, |index| index + 1);
    Some(CompactionWindow {
        latest_index,
        retained_start,
    })
}

impl ResponsesInput {
    #[must_use]
    pub fn contains_compaction(&self) -> bool {
        matches!(self, Self::Items(items) if items.iter().any(|item| matches!(item, InputItem::Compaction(_))))
    }

    #[must_use]
    pub fn has_compaction_trigger(&self) -> bool {
        matches!(self, Self::Items(items) if items.iter().any(InputItem::is_compaction_trigger))
    }

    /// Return the canonical context sent to vLLM.
    ///
    /// vLLM does not understand public `compaction` items, so the latest item
    /// becomes an assistant message containing the locally generated summary.
    /// Items before that checkpoint are superseded and are omitted.
    /// Internal MCP-list records and `compaction_trigger` markers are stripped
    /// and never reach the model. Message content parts the gateway does not
    /// model are dropped rather than forwarded as `{"type": "unknown"}`.
    #[must_use]
    pub fn model_input(&self) -> Cow<'_, Self> {
        let Self::Items(items) = self else {
            return Cow::Borrowed(self);
        };

        let Some(window) = latest_compaction_window(items) else {
            if items
                .iter()
                .any(|item| !item.is_model_visible() || item.has_unknown_content())
            {
                let stripped = items
                    .iter()
                    .filter(|item| item.is_model_visible())
                    .map(InputItem::without_unknown_content)
                    .collect();
                return Cow::Owned(Self::Items(stripped));
            }
            return Cow::Borrowed(self);
        };

        let model_items = window
            .retained_user_items(items)
            .chain(items[window.latest_index()..].iter())
            .filter(|item| item.is_model_visible())
            .map(|item| match item {
                InputItem::Compaction(compaction) => InputItem::Message(InputMessage {
                    id: None,
                    role: "assistant".to_owned(),
                    status: None,
                    content: InputMessageContent::Parts(vec![InputContent::OutputText(InputTextContent {
                        text: compaction.encrypted_content.clone(),
                    })]),
                }),
                other => other.without_unknown_content(),
            })
            .collect();
        Cow::Owned(Self::Items(model_items))
    }
}

fn function_call_item_id(item_id: &str) -> Option<String> {
    if item_id.is_empty() {
        return None;
    }
    if let Some(suffix) = item_id.strip_prefix("ctc_").filter(|suffix| !suffix.is_empty()) {
        return Some(format!("fc_{suffix}"));
    }
    Some(item_id.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_call_input_accepts_missing_status() {
        let item: InputItem = serde_json::from_value(serde_json::json!({
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}"
        }))
        .expect("valid replay input");

        let InputItem::FunctionCall(call) = item else {
            panic!("expected function call");
        };
        assert_eq!(call.status, None);
    }

    #[test]
    fn malformed_known_type_is_not_reinterpreted_as_shorthand_message() {
        let result = serde_json::from_value::<InputItem>(serde_json::json!({
            "type": "function_call",
            "role": "user",
            "content": "not a function call"
        }));

        assert!(result.is_err());
    }

    #[test]
    fn structured_custom_tool_output_is_preserved_when_normalized() {
        let content = serde_json::json!([
            {"type": "input_text", "text": "diagram"},
            {"type": "input_image", "image_url": "data:image/png;base64,abc", "detail": "low"},
            {"type": "input_file", "file_id": "file_123", "filename": "report.pdf"}
        ]);
        let item: InputItem = serde_json::from_value(serde_json::json!({
            "type": "custom_tool_call_output",
            "call_id": "call_1",
            "output": content
        }))
        .expect("valid structured custom-tool output");

        let InputItem::CustomToolCallOutput(output) = item else {
            panic!("expected custom-tool output");
        };
        let normalized = FunctionToolResultMessage::from(output);
        let value = serde_json::to_value(normalized).expect("normalized output serializes");

        assert_eq!(value["output"], content);
    }

    #[test]
    fn structured_function_tool_output_preserves_image_array() {
        let content = serde_json::json!([
            {"type": "input_text", "text": "attached local image path: diagram.png"},
            {"type": "input_image", "image_url": "data:image/png;base64,abc"}
        ]);
        let item: InputItem = serde_json::from_value(serde_json::json!({
            "type": "function_call_output",
            "call_id": "call_view_image_1",
            "output": content
        }))
        .expect("valid structured function-tool output");

        let InputItem::FunctionCallOutput(output) = &item else {
            panic!("expected function-tool output");
        };
        assert!(matches!(output.output, ToolCallOutput::Content(_)));

        let value = serde_json::to_value(&item).expect("output serializes");
        assert_eq!(value["output"], content, "structured output must not be stringified");
    }

    #[test]
    fn input_file_message_content_round_trips_as_a_modeled_part() {
        let part = serde_json::json!({
            "type": "input_file",
            "file_id": "file_1",
            "filename": "report.pdf"
        });
        let item: InputItem = serde_json::from_value(serde_json::json!({
            "type": "message",
            "role": "user",
            "content": [part]
        }))
        .expect("valid message input");

        let InputItem::Message(message) = &item else {
            panic!("expected message item");
        };
        let InputMessageContent::Parts(parts) = &message.content else {
            panic!("expected message parts");
        };
        let [InputContent::InputFile(file)] = parts.as_slice() else {
            panic!("expected a modeled input_file part");
        };
        assert!(file.has_reference());

        let value = serde_json::to_value(&item).expect("message serializes");
        assert_eq!(value["content"][0], part);
    }

    #[test]
    fn unknown_message_content_parts_are_dropped_from_model_input() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "before"},
                {"type": "input_audio", "audio_url": "https://example.com/clip.wav"},
                {"type": "input_image", "image_url": "data:image/png;base64,abc", "detail": "low"}
            ]
        }]))
        .expect("valid message input");

        let serialized = serde_json::to_value(input.model_input()).expect("model input serializes");
        let parts = serialized[0]["content"].as_array().expect("content parts");
        assert_eq!(
            parts.iter().map(|part| &part["type"]).collect::<Vec<_>>(),
            vec!["input_text", "input_image"],
            "an unrepresentable part must be dropped, not forwarded as {{\"type\": \"unknown\"}}"
        );
        assert_eq!(parts[1]["image_url"], "data:image/png;base64,abc");
        assert_eq!(parts[1]["detail"], "low");
    }

    #[test]
    fn message_with_only_unknown_content_keeps_its_turn() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([{
            "role": "user",
            "content": [{"type": "input_audio", "audio_url": "https://example.com/clip.wav"}]
        }]))
        .expect("valid message input");

        let serialized = serde_json::to_value(input.model_input()).expect("model input serializes");
        assert_eq!(serialized.as_array().map(Vec::len), Some(1));
        assert_eq!(serialized[0]["role"], "user");
        assert_eq!(serialized[0]["content"].as_array().map(Vec::len), Some(0));
    }

    #[test]
    fn unknown_content_is_dropped_from_retained_messages_after_compaction() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {
                "type": "message",
                "id": "msg_keep",
                "role": "user",
                "status": "completed",
                "content": [
                    {"type": "input_audio", "audio_url": "https://example.com/clip.wav"},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc"}
                ]
            },
            {"type": "compaction", "encrypted_content": "summary"}
        ]))
        .expect("valid compacted history");

        let serialized = serde_json::to_value(input.model_input()).expect("model input serializes");
        let parts = serialized[0]["content"].as_array().expect("retained content parts");
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["type"], "input_image");
    }

    #[test]
    fn custom_tool_output_rejects_unsupported_shapes() {
        for output in [
            serde_json::json!({"result": "not a supported top-level object"}),
            serde_json::json!([{"type": "output_text", "text": "wrong content type"}]),
            serde_json::json!(["content items must be objects"]),
        ] {
            let result = serde_json::from_value::<InputItem>(serde_json::json!({
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": output
            }));

            assert!(result.is_err(), "unsupported custom-tool output should fail");
        }
    }

    #[test]
    fn compaction_item_becomes_assistant_model_context() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([{
            "type": "compaction",
            "id": "cmp_1",
            "encrypted_content": "summary"
        }]))
        .expect("valid compaction input");

        let model_input = input.model_input();
        let serialized = serde_json::to_value(model_input).expect("model input serializes");
        assert_eq!(serialized[0]["role"], "assistant");
        assert_eq!(serialized[0]["content"][0]["type"], "output_text");
        assert_eq!(serialized[0]["content"][0]["text"], "summary");
    }

    #[test]
    fn compaction_trigger_parses_as_dedicated_variant() {
        let item: InputItem = serde_json::from_value(serde_json::json!({"type": "compaction_trigger"}))
            .expect("compaction_trigger parses");
        assert!(item.is_compaction_trigger());

        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "history"},
            {"type": "compaction_trigger"}
        ]))
        .expect("trigger input parses");
        assert!(input.has_compaction_trigger());
        assert!(!input.contains_compaction());
    }

    #[test]
    fn model_input_strips_compaction_trigger_without_window() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "history"},
            {"type": "compaction_trigger"}
        ]))
        .expect("trigger input parses");

        let serialized = serde_json::to_value(input.model_input()).expect("model input serializes");
        assert_eq!(serialized.as_array().map(Vec::len), Some(1));
        assert_eq!(serialized[0]["type"], "message");
        assert_eq!(serialized[0]["content"], "history");
    }

    #[test]
    fn model_input_strips_internal_mcp_list_tools() {
        let input = ResponsesInput::Items(vec![
            InputItem::McpListTools(McpListTools::new("mcpl_1", "counter", Vec::new())),
            InputItem::Message(InputMessage {
                id: None,
                role: "user".to_owned(),
                status: None,
                content: InputMessageContent::Text("continue".to_owned()),
            }),
        ]);

        let serialized = serde_json::to_value(input.model_input()).expect("model input serializes");
        assert_eq!(serialized.as_array().map(Vec::len), Some(1));
        assert_eq!(serialized[0]["content"], "continue");
        assert!(
            serialized
                .as_array()
                .is_some_and(|items| { items.iter().all(|item| item["type"] != "mcp_list_tools") })
        );
    }

    #[test]
    fn model_input_strips_compaction_trigger_after_window() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "discard me"},
            {"type": "compaction", "encrypted_content": "summary"},
            {"type": "message", "id": "msg_keep", "role": "user", "status": "completed", "content": "retained"},
            {"type": "compaction_trigger"}
        ]))
        .expect("trigger input parses");

        let serialized = serde_json::to_value(input.model_input()).expect("model input serializes");
        assert_eq!(serialized.as_array().map(Vec::len), Some(2));
        assert_eq!(serialized[0]["role"], "assistant");
        assert_eq!(serialized[0]["content"][0]["text"], "summary");
        assert_eq!(serialized[1]["content"], "retained");
        assert!(
            serialized
                .as_array()
                .is_some_and(|items| items.iter().all(|item| item["type"] != "compaction_trigger"))
        );
    }

    #[test]
    fn latest_compaction_preserves_canonical_user_messages_and_supersedes_prior_context() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "discard me"},
            {"type": "compaction", "encrypted_content": "old summary"},
            {"role": "assistant", "content": "also discard me"},
            {"type": "message", "id": "msg_keep", "role": "user", "status": "completed", "content": "retained user"},
            {"type": "compaction", "encrypted_content": "latest summary"},
            {"role": "user", "content": "keep me"}
        ]))
        .expect("valid compacted history");

        let serialized = serde_json::to_value(input.model_input()).expect("model input serializes");
        assert_eq!(serialized.as_array().map(Vec::len), Some(3));
        assert_eq!(serialized[0]["content"], "retained user");
        assert_eq!(serialized[1]["role"], "assistant");
        assert_eq!(serialized[1]["content"][0]["text"], "latest summary");
        assert_eq!(serialized[2]["content"], "keep me");
    }

    #[test]
    fn custom_items_convert_to_function_history() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {
                "type": "custom_tool_call",
                "id": "ctc_1",
                "call_id": "call_1",
                "name": "raw_echo",
                "input": "hello",
                "status": "completed"
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call_1",
                "output": "done"
            }
        ]))
        .expect("custom history");

        let canonical_value = serde_json::to_value(Vec::<InputItem>::from(&input)).expect("canonical items");
        assert_eq!(canonical_value[0]["type"], "function_call");
        assert_eq!(canonical_value[0]["id"], "fc_1");
        assert_eq!(canonical_value[0]["arguments"], r#"{"input":"hello"}"#);
        assert_eq!(canonical_value[1]["type"], "function_call_output");
        assert_eq!(canonical_value[1]["output"], "done");

        let public_value = serde_json::to_value(input).expect("public input");
        assert_eq!(public_value[0]["type"], "custom_tool_call");
        assert_eq!(public_value[1]["type"], "custom_tool_call_output");
    }
}
