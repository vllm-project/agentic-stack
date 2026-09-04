//! Anthropic Messages API request types (`POST /v1/messages`).
//!
//! These mirror the Anthropic wire shape. Unmodeled fields are preserved via
//! `extra` and pass through unchanged. The Messages-native loop normalizes
//! gateway-owned native server-tool declarations before forwarding, then reads
//! the tools and assistant turn; see [`super::tool_seam`] and
//! [`crate::executor::messages_loop`].

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Top-level Anthropic Messages request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<MessageParam>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolParam>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Anthropic `output_config` (reasoning effort and related settings).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
    /// Any other top-level field (e.g. `metadata`, `stop_sequences`) preserved verbatim.
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Anthropic `output_config` object. Unmodeled keys are preserved via `extra`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct OutputConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Reasoning effort requested through `output_config.effort`.
///
/// Known tiers are modeled explicitly; any other string is preserved verbatim
/// so the gateway never rejects or rewrites vocabulary it does not know.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ReasoningEffort {
    Level(ReasoningEffortLevel),
    Other(String),
}

/// Effort tiers understood by the gateway: the Anthropic/OpenAI standard
/// `low`/`medium`/`high`/`max` plus the `xhigh` top tier used by Qwen chat templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffortLevel {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

#[cfg(feature = "openapi")]
impl utoipa::PartialSchema for ReasoningEffort {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        utoipa::openapi::ObjectBuilder::new()
            .schema_type(utoipa::openapi::schema::SchemaType::new(
                utoipa::openapi::schema::Type::String,
            ))
            .into()
    }
}
#[cfg(feature = "openapi")]
impl utoipa::ToSchema for ReasoningEffort {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("ReasoningEffort")
    }
}

impl ReasoningEffort {
    /// Map the standard top tiers (`high`, `max`) onto `xhigh`, the top tier Qwen
    /// chat templates accept. Returns `None` when no change is needed.
    #[must_use]
    pub fn clamped_for_qwen(&self) -> Option<Self> {
        match self {
            Self::Level(ReasoningEffortLevel::High | ReasoningEffortLevel::Max) => {
                Some(Self::Level(ReasoningEffortLevel::Xhigh))
            }
            Self::Level(_) | Self::Other(_) => None,
        }
    }
}

/// Anthropic `system` accepts either a bare string or an array of text blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[cfg(feature = "openapi")]
impl utoipa::PartialSchema for SystemPrompt {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::{
            ObjectBuilder,
            schema::{ArrayBuilder, OneOfBuilder, SchemaType, Type},
        };
        OneOfBuilder::new()
            .item(ObjectBuilder::new().schema_type(SchemaType::new(Type::String)))
            .item(ArrayBuilder::new().items(utoipa::openapi::Ref::from_schema_name("SystemBlock")))
            .into()
    }
}
#[cfg(feature = "openapi")]
impl utoipa::ToSchema for SystemPrompt {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("SystemPrompt")
    }
}

impl SystemPrompt {
    /// Flatten into a single instructions string (block texts joined by newlines).
    #[must_use]
    pub fn to_instructions(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Blocks(blocks) => blocks.iter().map(|b| b.text.as_str()).collect::<Vec<_>>().join("\n"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct SystemBlock {
    #[serde(default)]
    pub text: String,
}

/// One entry in the `messages` array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct MessageParam {
    pub role: String,
    pub content: MessageContent,
}

/// Anthropic message content is either a bare string or an array of blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[cfg(feature = "openapi")]
impl utoipa::PartialSchema for MessageContent {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::{
            ObjectBuilder,
            schema::{ArrayBuilder, OneOfBuilder, SchemaType, Type},
        };
        OneOfBuilder::new()
            .item(ObjectBuilder::new().schema_type(SchemaType::new(Type::String)))
            .item(ArrayBuilder::new().items(utoipa::openapi::Ref::from_schema_name("ContentBlock")))
            .into()
    }
}
#[cfg(feature = "openapi")]
impl utoipa::ToSchema for MessageContent {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("MessageContent")
    }
}

/// A content block inside a message. Internally tagged by `type`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        #[serde(default)]
        text: String,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: ToolResultContent,
    },
    /// Any block type the gateway does not model — dropped on the way in.
    #[serde(other)]
    Unknown,
}

/// `tool_result.content` may be a plain string or an array of blocks (Anthropic
/// allows both). Normalised to a single string by [`ToolResultContent::to_text`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultBlock>),
}

#[cfg(feature = "openapi")]
impl utoipa::PartialSchema for ContentBlock {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::{
            ObjectBuilder,
            schema::{OneOfBuilder, SchemaType, Type},
        };

        let str_type = || ObjectBuilder::new().schema_type(SchemaType::new(Type::String));

        OneOfBuilder::new()
            .discriminator(Some(utoipa::openapi::schema::Discriminator::new("type")))
            .item(
                ObjectBuilder::new()
                    .property("type", str_type().enum_values(Some(["text"])))
                    .required("type")
                    .property("text", str_type()),
            )
            .item(
                ObjectBuilder::new()
                    .property("type", str_type().enum_values(Some(["thinking"])))
                    .required("type")
                    .property("thinking", str_type())
                    .property("signature", str_type()),
            )
            .item(
                ObjectBuilder::new()
                    .property("type", str_type().enum_values(Some(["tool_use"])))
                    .required("type")
                    .property("id", str_type())
                    .required("id")
                    .property("name", str_type())
                    .required("name")
                    .property("input", ObjectBuilder::new()),
            )
            .item(
                ObjectBuilder::new()
                    .property("type", str_type().enum_values(Some(["tool_result"])))
                    .required("type")
                    .property("tool_use_id", str_type())
                    .required("tool_use_id")
                    .property("content", utoipa::openapi::Ref::from_schema_name("ToolResultContent")),
            )
            .into()
    }
}
#[cfg(feature = "openapi")]
impl utoipa::ToSchema for ContentBlock {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("ContentBlock")
    }
}

#[cfg(feature = "openapi")]
impl utoipa::PartialSchema for ToolResultContent {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::{
            ObjectBuilder,
            schema::{ArrayBuilder, OneOfBuilder, SchemaType, Type},
        };
        OneOfBuilder::new()
            .item(ObjectBuilder::new().schema_type(SchemaType::new(Type::String)))
            .item(ArrayBuilder::new().items(utoipa::openapi::Ref::from_schema_name("ToolResultBlock")))
            .into()
    }
}
#[cfg(feature = "openapi")]
impl utoipa::ToSchema for ToolResultContent {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("ToolResultContent")
    }
}

impl Default for ToolResultContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl ToolResultContent {
    /// Flatten to a single output string for the internal `function_call_output`.
    #[must_use]
    pub fn to_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| b.text.as_deref())
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ToolResultBlock {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// A tool declared in the request. Anthropic's shape is `{name, description,
/// input_schema}`; server tools may additionally carry a versioned `type`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ToolParam {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<Value>,
    /// Anthropic server tools carry a versioned `type` (e.g. `web_search_20250305`).
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[cfg(test)]
mod effort_tests {
    use super::*;

    #[test]
    fn output_config_effort_clamps_top_tiers_and_preserves_extras() {
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "high", "format": {"type": "json"}},
            "metadata": {"user_id": "u"}
        }))
        .unwrap();
        let effort = request.output_config.as_ref().unwrap().effort.as_ref().unwrap();
        assert_eq!(
            effort.clamped_for_qwen(),
            Some(ReasoningEffort::Level(ReasoningEffortLevel::Xhigh))
        );
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["output_config"]["format"]["type"], "json");
        assert_eq!(json["metadata"]["user_id"], "u");

        for (input, expect_change) in [("max", true), ("medium", false), ("xhigh", false), ("minimal", false)] {
            let effort: ReasoningEffort = serde_json::from_value(serde_json::json!(input)).unwrap();
            assert_eq!(effort.clamped_for_qwen().is_some(), expect_change, "{input}");
            assert_eq!(serde_json::to_value(&effort).unwrap(), input);
        }
    }
}
