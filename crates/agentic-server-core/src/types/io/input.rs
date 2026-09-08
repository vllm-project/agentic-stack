use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::event::MessageStatus;
use crate::types::tools::{ResponsesTool, ToolSearchExecution, ToolSearchStatus};
use crate::utils::common::deserialize_from_value;

use super::output::{CustomToolCall, FunctionToolCall, McpListTools, ReasoningOutput, ToolSearchCall};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct InputTextContent {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct InputImageContent {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
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

/// Content item inside a message input.
///
/// Uses an internally-tagged enum — serde consumes `"type"` for the variant
/// discriminant so the inner structs must NOT redeclare a `type_` field.
/// `output_text` and `reasoning_text` reuse `InputTextContent` since they
/// carry only a `text` field; they are preserved so vLLM sees the full history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContent {
    InputText(InputTextContent),
    InputImage(InputImageContent),
    /// Assistant output text in rehydrated history.
    OutputText(InputTextContent),
    /// Reasoning step text in rehydrated history.
    ReasoningText(InputTextContent),
    /// Any other content type — drop silently.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
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
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolOutputContent {
    InputText(InputTextContent),
    InputImage(InputImageContent),
    InputFile(InputFileContent),
}

#[cfg(feature = "openapi")]
mod openapi_schemas {
    use super::{InputContent, InputItem, InputMessageContent, ResponsesInput, ToolCallOutput};
    use utoipa::openapi::schema::{ArrayBuilder, OneOfBuilder, Schema, SchemaType, Type};
    use utoipa::openapi::{ObjectBuilder, Ref, RefOr};

    fn string_schema() -> RefOr<Schema> {
        ObjectBuilder::new().schema_type(SchemaType::new(Type::String)).into()
    }

    impl utoipa::PartialSchema for InputMessageContent {
        fn schema() -> RefOr<Schema> {
            OneOfBuilder::new()
                .item(string_schema())
                .item(ArrayBuilder::new().items(Ref::from_schema_name("InputContent")))
                .into()
        }
    }
    impl utoipa::ToSchema for InputMessageContent {
        fn name() -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("InputMessageContent")
        }
    }

    impl utoipa::PartialSchema for ToolCallOutput {
        fn schema() -> RefOr<Schema> {
            OneOfBuilder::new()
                .item(string_schema())
                .item(ArrayBuilder::new().items(Ref::from_schema_name("ToolOutputContent")))
                .into()
        }
    }
    impl utoipa::ToSchema for ToolCallOutput {
        fn name() -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("ToolCallOutput")
        }
    }

    impl utoipa::PartialSchema for ResponsesInput {
        fn schema() -> RefOr<Schema> {
            OneOfBuilder::new()
                .item(string_schema())
                .item(ArrayBuilder::new().items(Ref::from_schema_name("InputItem")))
                .into()
        }
    }
    impl utoipa::ToSchema for ResponsesInput {
        fn name() -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("ResponsesInput")
        }
    }

    fn tagged_text_variant(type_value: &str) -> RefOr<Schema> {
        ObjectBuilder::new()
            .property(
                "type",
                ObjectBuilder::new()
                    .schema_type(SchemaType::new(Type::String))
                    .enum_values(Some([type_value])),
            )
            .required("type")
            .property("text", ObjectBuilder::new().schema_type(SchemaType::new(Type::String)))
            .required("text")
            .into()
    }

    impl utoipa::PartialSchema for InputContent {
        fn schema() -> RefOr<Schema> {
            OneOfBuilder::new()
                .discriminator(Some(utoipa::openapi::schema::Discriminator::new("type")))
                .item(tagged_text_variant("input_text"))
                .item(
                    ObjectBuilder::new()
                        .property(
                            "type",
                            ObjectBuilder::new()
                                .schema_type(SchemaType::new(Type::String))
                                .enum_values(Some(["input_image"])),
                        )
                        .required("type")
                        .property(
                            "file_id",
                            ObjectBuilder::new().schema_type(SchemaType::new(Type::String)),
                        )
                        .property(
                            "image_url",
                            ObjectBuilder::new().schema_type(SchemaType::new(Type::String)),
                        )
                        .property(
                            "detail",
                            ObjectBuilder::new().schema_type(SchemaType::new(Type::String)),
                        ),
                )
                .item(tagged_text_variant("output_text"))
                .item(tagged_text_variant("reasoning_text"))
                .into()
        }
    }
    impl utoipa::ToSchema for InputContent {
        fn name() -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("InputContent")
        }
    }

    fn tagged_ref(type_value: &str, schema_name: &str) -> RefOr<Schema> {
        use utoipa::openapi::schema::AllOfBuilder;
        AllOfBuilder::new()
            .item(
                ObjectBuilder::new()
                    .property(
                        "type",
                        ObjectBuilder::new()
                            .schema_type(SchemaType::new(Type::String))
                            .enum_values(Some([type_value])),
                    )
                    .required("type"),
            )
            .item(Ref::from_schema_name(schema_name))
            .into()
    }

    impl utoipa::PartialSchema for InputItem {
        fn schema() -> RefOr<Schema> {
            use utoipa::openapi::schema::AllOfBuilder;
            let message_branch: RefOr<Schema> = AllOfBuilder::new()
                .item(
                    ObjectBuilder::new().property(
                        "type",
                        ObjectBuilder::new()
                            .schema_type(SchemaType::new(Type::String))
                            .enum_values(Some(["message"])),
                    ),
                )
                .item(Ref::from_schema_name("InputMessage"))
                .into();
            OneOfBuilder::new()
                .discriminator(Some(utoipa::openapi::schema::Discriminator::new("type")))
                .item(message_branch)
                .item(tagged_ref("function_call", "InputFunctionToolCall"))
                .item(tagged_ref("function_call_output", "FunctionToolResultMessage"))
                .item(tagged_ref("custom_tool_call", "CustomToolCall"))
                .item(tagged_ref("custom_tool_call_output", "CustomToolCallOutputMessage"))
                .item(tagged_ref("reasoning", "ReasoningOutput"))
                .item(tagged_ref("mcp_list_tools", "McpListTools"))
                .item(tagged_ref("compaction", "CompactionItem"))
                .item(
                    ObjectBuilder::new()
                        .property(
                            "type",
                            ObjectBuilder::new()
                                .schema_type(SchemaType::new(Type::String))
                                .enum_values(Some(["compaction_trigger"])),
                        )
                        .required("type"),
                )
                .into()
        }
    }
    impl utoipa::ToSchema for InputItem {
        fn name() -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("InputItem")
        }
    }
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
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
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

pub(super) fn deserialize_non_blank_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    if value.trim().is_empty() {
        return Err(serde::de::Error::custom("value must not be blank"));
    }
    Ok(value)
}

/// A public model-generated tool-search call replayed as Responses input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputToolSearchCall {
    #[serde(deserialize_with = "deserialize_non_blank_string")]
    pub id: String,
    #[serde(deserialize_with = "deserialize_non_blank_string")]
    pub call_id: String,
    #[serde(default)]
    pub execution: ToolSearchExecution,
    pub arguments: Value,
    #[serde(default)]
    pub status: ToolSearchStatus,
}

impl TryFrom<&ToolSearchCall> for InputToolSearchCall {
    type Error = ToolSearchStatus;

    fn try_from(call: &ToolSearchCall) -> Result<Self, Self::Error> {
        if call.status != ToolSearchStatus::Completed {
            return Err(call.status);
        }
        Ok(Self {
            id: call.id.clone(),
            call_id: call.call_id.clone(),
            execution: call.execution,
            arguments: call.arguments.clone(),
            status: ToolSearchStatus::Completed,
        })
    }
}

/// Client-returned declarations resolving a public tool-search call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSearchOutputMessage {
    #[serde(deserialize_with = "deserialize_non_blank_string")]
    pub call_id: String,
    #[serde(default)]
    pub execution: ToolSearchExecution,
    #[serde(default)]
    pub status: ToolSearchStatus,
    pub tools: Vec<ResponsesTool>,
}

/// An opaque compacted context checkpoint accepted as Responses input.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CompactionItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub encrypted_content: String,
}

/// Client result for a freeform custom tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
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
    #[serde(rename = "tool_search_call")]
    ToolSearchCall(InputToolSearchCall),
    #[serde(rename = "tool_search_output")]
    ToolSearchOutput(ToolSearchOutputMessage),
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
            Some("tool_search_call") => deserialize_from_value(value).map(Self::ToolSearchCall),
            Some("tool_search_output") => deserialize_from_value(value).map(Self::ToolSearchOutput),
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
    /// and never reach the model.
    #[must_use]
    pub fn model_input(&self) -> Cow<'_, Self> {
        let Self::Items(items) = self else {
            return Cow::Borrowed(self);
        };

        let Some(window) = latest_compaction_window(items) else {
            if items.iter().any(|item| !item.is_model_visible()) {
                let stripped = items.iter().filter(|item| item.is_model_visible()).cloned().collect();
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
                other => other.clone(),
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
    fn tool_search_replay_defaults_are_canonicalized() {
        let call: InputItem = serde_json::from_value(serde_json::json!({
            "type": "tool_search_call",
            "id": "tsc_1",
            "call_id": "call_search_1",
            "arguments": ["weather", "timezone"]
        }))
        .expect("valid replayed search call");
        let output: InputItem = serde_json::from_value(serde_json::json!({
            "type": "tool_search_output",
            "call_id": "call_search_1",
            "tools": []
        }))
        .expect("valid empty search result");

        assert_eq!(
            serde_json::to_value(call).expect("call serializes"),
            serde_json::json!({
                "type": "tool_search_call",
                "id": "tsc_1",
                "call_id": "call_search_1",
                "execution": "client",
                "arguments": ["weather", "timezone"],
                "status": "completed"
            })
        );
        assert_eq!(
            serde_json::to_value(output).expect("output serializes"),
            serde_json::json!({
                "type": "tool_search_output",
                "call_id": "call_search_1",
                "execution": "client",
                "status": "completed",
                "tools": []
            })
        );
    }

    #[test]
    fn tool_search_items_accept_documented_statuses() {
        for status in ["in_progress", "completed", "incomplete"] {
            let call: InputItem = serde_json::from_value(serde_json::json!({
                "type": "tool_search_call",
                "id": "tsc_1",
                "call_id": "call_search_1",
                "arguments": {"query": "weather"},
                "status": status
            }))
            .expect("documented tool-search call status");
            let output: InputItem = serde_json::from_value(serde_json::json!({
                "type": "tool_search_output",
                "call_id": "call_search_1",
                "status": status,
                "tools": []
            }))
            .expect("documented tool-search output status");

            assert_eq!(serde_json::to_value(call).expect("call serializes")["status"], status);
            assert_eq!(
                serde_json::to_value(output).expect("output serializes")["status"],
                status
            );
        }
    }

    #[test]
    fn tool_search_replay_rejects_invalid_known_shapes() {
        for item in [
            serde_json::json!({
                "type": "tool_search_call",
                "call_id": "call_search_1",
                "arguments": {"query": "missing required item id"}
            }),
            serde_json::json!({
                "type": "tool_search_call",
                "id": "   ",
                "call_id": "call_search_1",
                "arguments": {"query": "blank item id"}
            }),
            serde_json::json!({
                "type": "tool_search_call",
                "id": "tsc_1",
                "call_id": "   ",
                "arguments": {"query": "blank call id"}
            }),
            serde_json::json!({
                "type": "tool_search_call",
                "id": "tsc_1",
                "call_id": "call_search_1",
                "execution": "server",
                "arguments": {"query": "unsupported execution"}
            }),
            serde_json::json!({
                "type": "tool_search_call",
                "id": "tsc_1",
                "call_id": "call_search_1",
                "status": "completed"
            }),
            serde_json::json!({
                "type": "tool_search_output",
                "call_id": "call_search_1"
            }),
        ] {
            assert!(
                serde_json::from_value::<InputItem>(item).is_err(),
                "malformed known tool-search item must not become Unknown"
            );
        }

        let future: InputItem = serde_json::from_value(serde_json::json!({
            "type": "future_search_item",
            "payload": {"opaque": true}
        }))
        .expect("unrelated future item remains forward-compatible");
        assert!(matches!(future, InputItem::Unknown));
    }

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
