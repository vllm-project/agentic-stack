use std::borrow::Cow;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::io::{
    FunctionTool, InputItem, InputMessage, InputMessageContent, OutputItem, ResponseUsage, ResponsesInput, ToolChoice,
};
use super::tools::ResponsesTool;
use crate::tool::{CodexNamespaceHandler, CustomHandler, ToolError};
use crate::utils::common::serialize_to_string;

/// Standard Responses API reasoning generation settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate_summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Responses text-generation settings forwarded to the upstream service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTextConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<ResponseTextFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
    /// Unmodeled extension fields preserved for upstream compatibility.
    #[serde(default)]
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Output format requested through [`ResponseTextConfig`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ResponseTextFormat {
    Text {
        /// Unmodeled extension fields preserved for upstream compatibility.
        #[serde(default)]
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    JsonObject {
        /// Unmodeled extension fields preserved for upstream compatibility.
        #[serde(default)]
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    JsonSchema {
        name: String,
        schema: serde_json::Map<String, Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
        /// Unmodeled extension fields preserved for upstream compatibility.
        #[serde(default)]
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "Box<T>: Serialize", deserialize = "Box<T>: Deserialize<'de>"))]
pub struct RequestPayload<T: ?Sized = ResponseTextConfig> {
    pub model: String,
    pub input: ResponsesInput,
    pub instructions: Option<String>,
    pub previous_response_id: Option<String>,
    pub conversation_id: Option<String>,
    pub tools: Option<Vec<ResponsesTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_true")]
    pub store: bool,
    pub include: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Box<ReasoningConfig>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<Box<T>>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_output_tokens: Option<u32>,
    pub truncation: Option<String>,
    pub metadata: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_salt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Vec<ContextManagement>>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub struct UpstreamRequest<'a> {
    pub model: &'a str,
    pub input: Cow<'a, ResponsesInput>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<&'a str>,
    /// Tools forwarded to vLLM. Function-like declarations are normalized to
    /// ordinary function tools.
    /// Skipped when empty so vLLM does not receive an empty array.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<UpstreamTool>>,
    #[serde(
        skip_serializing_if = "is_absent_or_default_tool_choice",
        serialize_with = "serialize_upstream_tool_choice"
    )]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<&'a Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<&'a ReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<&'a ResponseTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_salt: Option<&'a str>,
}

/// A normalized tool declaration supported by the upstream Responses endpoint.
///
/// Gateway and client tool declarations are converted to function tools before
/// entering this upstream-only payload.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum UpstreamTool {
    Function(FunctionTool),
}

// serde's `skip_serializing_if` requires a `&Option<T>` receiver, so the
// idiomatic `Option<&T>` clippy suggests does not apply here.
#[allow(clippy::ref_option)]
fn is_absent_or_default_tool_choice(choice: &Option<ToolChoice>) -> bool {
    choice.as_ref().is_none_or(|choice| matches!(choice, ToolChoice::Auto))
}

// serde's `serialize_with` passes a reference to the field's concrete type.
#[allow(clippy::ref_option)]
fn serialize_upstream_tool_choice<S>(choice: &Option<ToolChoice>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    choice
        .as_ref()
        .map(ToolChoice::normalized_for_upstream)
        .serialize(serializer)
}

impl<T: ?Sized> RequestPayload<T> {
    /// Names the feature in this request that only the in-process executor
    /// implements, if any — neither the passthrough proxy nor split execution
    /// can serve it.
    #[must_use]
    pub fn in_process_feature(&self) -> Option<&'static str> {
        if self.conversation_id.is_some() {
            return Some("conversation_id");
        }
        if self
            .tools
            .as_ref()
            .is_some_and(|tools| tools.iter().any(|tool| !matches!(tool, ResponsesTool::Function(_))))
        {
            return Some("gateway-owned tools");
        }
        if self.input.contains_compaction() || self.input.has_compaction_trigger() {
            return Some("compaction input");
        }
        if self
            .context_management
            .as_ref()
            .is_some_and(|entries| !entries.is_empty())
        {
            return Some("context_management");
        }
        None
    }

    /// Transform the text configuration while moving all other request fields
    /// without reparsing or reallocating them.
    ///
    /// # Errors
    ///
    /// Returns the mapping function's error when the text configuration cannot
    /// be transformed.
    pub fn try_map_text<U: ?Sized, E>(
        self,
        map: impl FnOnce(Box<T>) -> Result<Box<U>, E>,
    ) -> Result<RequestPayload<U>, E> {
        let text = self.text.map(map).transpose()?;
        Ok(RequestPayload {
            model: self.model,
            input: self.input,
            instructions: self.instructions,
            previous_response_id: self.previous_response_id,
            conversation_id: self.conversation_id,
            tools: self.tools,
            tool_choice: self.tool_choice,
            stream: self.stream,
            store: self.store,
            include: self.include,
            reasoning: self.reasoning,
            text,
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.max_output_tokens,
            truncation: self.truncation,
            metadata: self.metadata,
            parallel_tool_calls: self.parallel_tool_calls,
            cache_salt: self.cache_salt,
            context_management: self.context_management,
        })
    }
}

impl RequestPayload {
    /// Construct an `UpstreamRequest` suitable for forwarding to vLLM.
    ///
    /// Codex `namespace` tools' members are first renamed to their flat,
    /// model-visible names via [`CodexNamespaceHandler::resolve_namespace_members`].
    /// Namespace, gateway, and custom tools are then normalized to function
    /// declarations. `tool_choice` is resolved the same way via
    /// [`CodexNamespaceHandler::resolve_tool_choice`].
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] when a Codex namespace member's generated
    /// flat name collides with a top-level function tool or another namespace
    /// member, or when a custom tool declares a format whose constrained
    /// decoding cannot be preserved upstream.
    pub fn to_upstream_request(&self, stream: bool) -> Result<UpstreamRequest<'_>, ToolError> {
        // This is only the upstream model-generation preference: it controls
        // whether the model may emit parallel calls, not how the gateway
        // schedules calls after inference. Forward an explicit client value;
        // preserve this gateway's existing default of `false` when omitted.
        // `GatewayScheduler` independently applies its bounded fan-out and each
        // handler's same-tool parallel-safety policy to whatever calls appear.
        let parallel_tool_calls = Some(self.parallel_tool_calls.unwrap_or(false));

        let renamed_tools = self
            .tools
            .as_deref()
            .map(|tools| CodexNamespaceHandler.resolve_namespace_members(tools))
            .transpose()?;
        if let Some(tools) = &renamed_tools {
            for tool in tools {
                tool.validate()?;
            }
        }
        let tools: Option<Vec<UpstreamTool>> = renamed_tools.map(|tools| {
            tools
                .iter()
                .flat_map(ResponsesTool::to_function_tools)
                .map(UpstreamTool::Function)
                .collect()
        });
        let tools = tools.filter(|tools| !tools.is_empty());
        let namespace_map = CodexNamespaceHandler.build_namespace_map(self.tools.as_deref())?;
        let tool_choice = CodexNamespaceHandler.resolve_tool_choice(namespace_map.as_ref(), self.tool_choice.as_ref());
        CustomHandler::validate_tool_choice(self.tools.as_deref(), &tool_choice)?;
        Ok(UpstreamRequest {
            model: &self.model,
            input: self.input.model_input(),
            stream,
            instructions: self.instructions.as_deref(),
            tools,
            tool_choice: Some(tool_choice),
            include: self.include.as_ref(),
            reasoning: self.reasoning.as_deref(),
            text: self.text.as_deref(),
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.max_output_tokens,
            truncation: self.truncation.as_deref(),
            metadata: self.metadata.as_ref(),
            parallel_tool_calls,
            cache_salt: self.cache_salt.as_deref(),
        })
    }
}

/// Server-side context management configuration for a Responses request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextManagement {
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compact_threshold: Option<u64>,
}

/// Request body accepted by `POST /v1/responses/compact`.
#[derive(Debug, Clone, Deserialize)]
pub struct CompactRequest {
    pub model: String,
    #[serde(default)]
    pub input: Option<ResponsesInput>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    /// Compatibility fields sent by current SDK and Codex clients.
    #[serde(flatten)]
    pub compatibility: HashMap<String, Value>,
}

/// Result returned by `POST /v1/responses/compact`.
#[derive(Debug, Clone, Serialize)]
pub struct CompactedResponse {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    pub output: Vec<InputItem>,
    pub usage: ResponseUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncompleteDetails {
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsePayload {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    pub model: String,
    pub status: String,
    #[serde(default)]
    pub output: Vec<OutputItem>,
    pub usage: Option<ResponseUsage>,
    pub incomplete_details: Option<IncompleteDetails>,
    pub error: Option<Value>,
    pub previous_response_id: Option<String>,
    pub conversation_id: Option<String>,
    pub instructions: Option<String>,
}

impl ResponsePayload {
    #[must_use]
    pub fn as_created_response_chunk(&self) -> String {
        let mut response = self.clone();
        "in_progress".clone_into(&mut response.status);
        let event = json!({
            "type": "response.created",
            "response": response,
        });
        let json_str = serialize_to_string(&event).unwrap_or_else(|_| String::new());
        format!("data: {json_str}\n\n")
    }

    #[must_use]
    pub fn as_responses_chunk(&self) -> String {
        let json_str = serialize_to_string(self).unwrap_or_else(|_| String::new());
        format!("data: {json_str}\n\n")
    }

    #[must_use]
    pub fn as_terminal_response_chunk(&self) -> String {
        let event = json!({
            "type": self.terminal_event_type(),
            "response": self,
        });
        let json_str = serialize_to_string(&event).unwrap_or_else(|_| String::new());
        format!("data: {json_str}\n\n")
    }

    pub(crate) fn terminal_event_type(&self) -> &'static str {
        match self.status.as_str() {
            "incomplete" => "response.incomplete",
            "failed" | "error" => "response.failed",
            "in_progress" => "response.in_progress",
            _ => "response.completed",
        }
    }
}

impl From<&ResponsesInput> for Vec<InputItem> {
    fn from(input: &ResponsesInput) -> Self {
        match input {
            ResponsesInput::Text(text) => vec![InputItem::Message(InputMessage {
                id: None,
                role: "user".into(),
                status: None,
                content: InputMessageContent::Text(text.clone()),
            })],
            ResponsesInput::Items(items) => items
                .iter()
                .filter_map(|item| match item {
                    InputItem::Unknown => None,
                    InputItem::ShellCall(call) => Some(InputItem::FunctionCall(call.clone().into())),
                    InputItem::ShellCallOutput(output) => Some(InputItem::FunctionCallOutput(output.clone().into())),
                    InputItem::CustomToolCall(call) => Some(InputItem::FunctionCall(call.clone().into())),
                    InputItem::CustomToolCallOutput(output) => {
                        Some(InputItem::FunctionCallOutput(output.clone().into()))
                    }
                    item => Some(item.clone()),
                })
                .collect(),
        }
    }
}

impl From<ResponsesInput> for Vec<InputItem> {
    fn from(input: ResponsesInput) -> Self {
        match input {
            ResponsesInput::Text(text) => vec![InputItem::Message(InputMessage {
                id: None,
                role: "user".into(),
                status: None,
                content: InputMessageContent::Text(text),
            })],
            ResponsesInput::Items(items) => items
                .into_iter()
                .filter_map(|item| match item {
                    InputItem::Unknown => None,
                    InputItem::ShellCall(call) => Some(InputItem::FunctionCall(call.into())),
                    InputItem::ShellCallOutput(output) => Some(InputItem::FunctionCallOutput(output.into())),
                    InputItem::CustomToolCall(call) => Some(InputItem::FunctionCall(call.into())),
                    InputItem::CustomToolCallOutput(output) => Some(InputItem::FunctionCallOutput(output.into())),
                    item => Some(item),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_request_accepts_codex_compatibility_fields() {
        let request: CompactRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "input": [{"role": "user", "content": "hello"}],
            "tools": [],
            "parallel_tool_calls": true,
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "low"}
        }))
        .expect("compact request should parse");

        assert_eq!(request.model, "test-model");
        assert!(request.input.is_some());
        assert_eq!(request.compatibility.len(), 4);
    }

    #[test]
    fn request_payload_omits_absent_and_forwards_present_cache_salt_upstream() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "input": "hello"
        }))
        .expect("request should deserialize");

        let upstream = serde_json::to_value(payload.to_upstream_request(false).expect("request should normalize"))
            .expect("upstream request should serialize");

        assert!(upstream.get("cache_salt").is_none());

        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "input": "hello",
            "cache_salt": "tenant-a"
        }))
        .expect("request should deserialize");

        let upstream = serde_json::to_value(payload.to_upstream_request(false).expect("request should normalize"))
            .expect("upstream request should serialize");

        assert_eq!(upstream["cache_salt"], "tenant-a");
    }

    #[test]
    fn request_payload_forwards_reasoning_configuration_upstream() {
        let reasoning = serde_json::json!({
            "context": "all_turns",
            "effort": "high",
            "generate_summary": "concise",
            "mode": "pro",
            "summary": "detailed"
        });
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "input": "hello",
            "reasoning": reasoning
        }))
        .expect("request should deserialize");

        for stream in [false, true] {
            let upstream = serde_json::to_value(payload.to_upstream_request(stream).expect("request should normalize"))
                .expect("upstream request should serialize");

            assert_eq!(upstream["reasoning"], reasoning);
            assert_eq!(upstream["stream"], stream);
        }
    }

    #[test]
    fn request_payload_forwards_text_configuration_upstream() {
        let text = serde_json::json!({
            "format": {
                "type": "json_schema",
                "name": "weather",
                "description": "A weather report",
                "schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "temperature": {"type": "number"}
                    },
                    "required": ["city", "temperature"],
                    "additionalProperties": false
                },
                "strict": true
            },
            "verbosity": "low"
        });
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "input": "hello",
            "text": text
        }))
        .expect("request should deserialize");

        for stream in [false, true] {
            let upstream = serde_json::to_value(payload.to_upstream_request(stream).expect("request should normalize"))
                .expect("upstream request should serialize");

            assert_eq!(upstream["text"], text);
            assert_eq!(upstream["stream"], stream);
        }
    }

    #[test]
    fn request_payload_handles_text_configuration_boundaries() {
        for text in [
            serde_json::json!({}),
            serde_json::json!({"format": {"type": "text"}}),
            serde_json::json!({"format": {"type": "json_object"}}),
            serde_json::json!({"verbosity": "medium"}),
            serde_json::json!({
                "format": {"type": "text", "x-format-extension": true},
                "x-text-extension": {"enabled": true}
            }),
        ] {
            let payload: RequestPayload = serde_json::from_value(serde_json::json!({
                "model": "test-model",
                "input": "hello",
                "text": text
            }))
            .expect("valid text configuration should deserialize");
            let upstream = serde_json::to_value(payload.to_upstream_request(false).expect("request should normalize"))
                .expect("upstream request should serialize");

            assert_eq!(upstream["text"], text);
        }

        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "input": "hello",
            "text": null
        }))
        .expect("null text configuration should remain absent");
        let upstream = serde_json::to_value(payload.to_upstream_request(false).expect("request should normalize"))
            .expect("upstream request should serialize");
        assert!(upstream.get("text").is_none());

        for text in [
            serde_json::json!("json_object"),
            serde_json::json!({"format": "json_object"}),
            serde_json::json!({"format": {"type": 7}}),
            serde_json::json!({"format": {"type": "unknown"}}),
            serde_json::json!({"format": {"type": "json_schema", "schema": {}}}),
            serde_json::json!({"format": {"type": "json_schema", "name": "missing_schema"}}),
            serde_json::json!({"format": {"type": "json_schema", "schema": []}}),
            serde_json::json!({"verbosity": 1}),
        ] {
            let parsed = serde_json::from_value::<RequestPayload>(serde_json::json!({
                "model": "test-model",
                "input": "hello",
                "text": text
            }));

            assert!(parsed.is_err(), "malformed text configuration should fail: {text}");
        }
    }

    #[test]
    fn text_configuration_preserves_extension_field_order() {
        let config: ResponseTextConfig = serde_json::from_str(
            r#"{
                "format": {
                    "type": "json_schema",
                    "name": "ordered",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "second": {"type": "string"},
                            "first": {"type": "number"}
                        }
                    },
                    "x-format-first": 1,
                    "x-format-second": 2,
                    "x-format-third": 3,
                    "x-format-fourth": 4,
                    "x-format-fifth": 5,
                    "x-format-sixth": 6
                },
                "verbosity": "low",
                "x-text-first": 1,
                "x-text-second": 2,
                "x-text-third": 3,
                "x-text-fourth": 4,
                "x-text-fifth": 5,
                "x-text-sixth": 6
            }"#,
        )
        .expect("text configuration should deserialize");

        let serialized = serde_json::to_string(&config).expect("text configuration should serialize");

        assert_eq!(
            serialized,
            r#"{"format":{"type":"json_schema","name":"ordered","schema":{"type":"object","properties":{"second":{"type":"string"},"first":{"type":"number"}}},"x-format-first":1,"x-format-second":2,"x-format-third":3,"x-format-fourth":4,"x-format-fifth":5,"x-format-sixth":6},"verbosity":"low","x-text-first":1,"x-text-second":2,"x-text-third":3,"x-text-fourth":4,"x-text-fifth":5,"x-text-sixth":6}"#
        );
    }

    #[test]
    fn request_payload_handles_reasoning_boundaries() {
        for reasoning in [serde_json::json!({}), serde_json::json!({"effort": "minimal"})] {
            let payload: RequestPayload = serde_json::from_value(serde_json::json!({
                "model": "test-model",
                "input": "hello",
                "reasoning": reasoning
            }))
            .expect("valid reasoning object should deserialize");
            let upstream = serde_json::to_value(payload.to_upstream_request(false).expect("request should normalize"))
                .expect("upstream request should serialize");

            assert_eq!(upstream["reasoning"], reasoning);
        }

        for reasoning in [
            serde_json::Value::Null,
            serde_json::json!("high"),
            serde_json::json!({"effort": 3}),
        ] {
            let parsed = serde_json::from_value::<RequestPayload>(serde_json::json!({
                "model": "test-model",
                "input": "hello",
                "reasoning": reasoning
            }));

            if reasoning.is_null() {
                let upstream = serde_json::to_value(
                    parsed
                        .expect("null should be treated as absent")
                        .to_upstream_request(false)
                        .expect("request should normalize"),
                )
                .expect("upstream request should serialize");
                assert!(upstream.get("reasoning").is_none());
            } else {
                assert!(parsed.is_err(), "non-object reasoning configuration should be rejected");
            }
        }
    }

    #[test]
    fn request_payload_uses_option_tool_choice_for_missing_vs_explicit() {
        let absent: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi"
        }))
        .unwrap();
        assert_eq!(absent.tool_choice, None);

        let explicit: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tool_choice": "none"
        }))
        .unwrap();
        assert_eq!(explicit.tool_choice, Some(ToolChoice::None));
    }

    #[test]
    fn to_upstream_request_carries_instructions_forward() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "instructions": "rules",
            "input": "hi"
        }))
        .unwrap();

        assert_eq!(payload.instructions.as_deref(), Some("rules"));
        assert!(matches!(&payload.input, ResponsesInput::Text(text) if text == "hi"));

        let upstream = payload.to_upstream_request(false).expect("valid upstream request");
        let value = serde_json::to_value(upstream).unwrap();
        assert_eq!(value["instructions"], "rules");
        assert_eq!(value["input"], "hi");
    }

    #[test]
    fn to_upstream_request_preserves_parallel_tool_calls() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "parallel_tool_calls": false
        }))
        .unwrap();

        let upstream = payload.to_upstream_request(false).expect("valid upstream request");
        let value = serde_json::to_value(upstream).unwrap();
        assert_eq!(value["parallel_tool_calls"], false);
    }

    #[test]
    fn to_upstream_request_allows_parallel_tool_calls_for_client_function_tools() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "parallel_tool_calls": true,
            "tools": [{"type": "function", "name": "get_weather"}]
        }))
        .unwrap();

        let upstream = payload
            .to_upstream_request(false)
            .expect("function tools allow parallel calls");
        let value = serde_json::to_value(upstream).unwrap();
        assert_eq!(value["parallel_tool_calls"], true);
    }

    #[test]
    fn to_upstream_request_preserves_parallel_tool_calls_for_mixed_tools() {
        for built_in_tool in builtin_tool_declarations() {
            for parallel_tool_calls in [false, true] {
                let payload: RequestPayload = serde_json::from_value(serde_json::json!({
                    "model": "test",
                    "input": "hi",
                    "parallel_tool_calls": parallel_tool_calls,
                    "tools": [
                        {"type": "function", "name": "get_weather"},
                        built_in_tool.clone()
                    ]
                }))
                .unwrap();

                let value = serde_json::to_value(
                    payload
                        .to_upstream_request(false)
                        .expect("mixed tools preserve the client's parallel_tool_calls value"),
                )
                .unwrap();
                assert_eq!(value["parallel_tool_calls"], parallel_tool_calls);
            }
        }
    }

    #[test]
    fn to_upstream_request_defaults_parallel_tool_calls_to_false_when_omitted() {
        for tool in builtin_tool_declarations() {
            let payload: RequestPayload = serde_json::from_value(serde_json::json!({
                "model": "test",
                "input": "hi",
                "tools": [tool]
            }))
            .unwrap();

            let upstream = payload
                .to_upstream_request(false)
                .expect("omitted parallel_tool_calls defaults to false");
            let value = serde_json::to_value(upstream).unwrap();
            assert_eq!(value["parallel_tool_calls"], false);
        }
    }

    #[test]
    fn to_upstream_request_allows_parallel_tool_calls_for_builtin_tools() {
        for tool in builtin_tool_declarations() {
            let payload: RequestPayload = serde_json::from_value(serde_json::json!({
                "model": "test",
                "input": "hi",
                "parallel_tool_calls": true,
                "tools": [tool]
            }))
            .unwrap();

            let upstream = payload
                .to_upstream_request(false)
                .expect("built-in tools allow parallel calls");
            let value = serde_json::to_value(upstream).unwrap();
            assert_eq!(value["parallel_tool_calls"], true);
        }
    }

    #[test]
    fn to_upstream_request_allows_builtin_tools_with_serial_tool_calls() {
        for tool in builtin_tool_declarations() {
            let payload: RequestPayload = serde_json::from_value(serde_json::json!({
                "model": "test",
                "input": "hi",
                "parallel_tool_calls": false,
                "tools": [tool]
            }))
            .unwrap();

            let upstream = payload
                .to_upstream_request(false)
                .expect("serial built-in tool request is valid");
            let value = serde_json::to_value(upstream).unwrap();
            assert_eq!(value["parallel_tool_calls"], false);
        }
    }

    fn builtin_tool_declarations() -> Vec<Value> {
        vec![
            serde_json::json!({
                "type": "mcp",
                "server_label": "repo",
                "server_url": "http://localhost:9001/mcp"
            }),
            serde_json::json!({"type": "web_search_preview"}),
            serde_json::json!({"type": "file_search", "vector_store_ids": ["vs_abc"]}),
            serde_json::json!({"type": "code_interpreter"}),
        ]
    }

    #[test]
    fn to_upstream_request_flattens_namespace_and_skips_unknown_tools() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__shell",
                    "tools": [
                        {"type": "function", "name": "run", "parameters": {"type": "object"}},
                        {"type": "future_member", "opaque": true}
                    ]
                },
                {"type": "future_tool", "opaque": true}
            ]
        }))
        .unwrap();

        let tools = payload.tools.as_ref().expect("tools should preserve explicit presence");
        assert_eq!(tools.len(), 2);
        let ResponsesTool::Namespace(namespace) = &tools[0] else {
            panic!("expected namespace tool");
        };
        assert_eq!(namespace.tools.len(), 2);

        let upstream = payload.to_upstream_request(false).expect("valid upstream request");
        let value = serde_json::to_value(upstream).unwrap();
        assert_eq!(value["tools"].as_array().expect("upstream tools").len(), 1);
        assert_eq!(value["tools"][0]["name"], "agentic_ns__mcp__shell__run");
    }

    #[test]
    fn to_upstream_request_rejects_namespace_collisions() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tools": [
                {"type": "function", "name": "agentic_ns__mcp__shell__run"},
                {
                    "type": "namespace",
                    "name": "mcp__shell",
                    "tools": [{"type": "function", "name": "run"}]
                }
            ]
        }))
        .unwrap();

        let Err(err) = payload.to_upstream_request(false) else {
            panic!("colliding namespace member should be rejected");
        };

        assert!(err.to_string().contains("collides with a declared function tool"));
    }

    #[test]
    fn to_upstream_request_normalizes_custom_tools_to_functions() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tool_choice": {
                "type": "custom",
                "name": "apply_patch"
            },
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {"type": "object"}
                },
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch.",
                    "x-provider-field": {"mode": "strict"}
                }
            ]
        }))
        .unwrap();

        let request = payload.to_upstream_request(false).unwrap();
        let tools = request.tools.as_ref().expect("mixed upstream tools");
        let UpstreamTool::Function(first) = &tools[0];
        let UpstreamTool::Function(second) = &tools[1];
        assert_eq!(first.name, "read_file");
        assert_eq!(second.name, "apply_patch");

        let upstream = serde_json::to_value(request).unwrap();
        assert_eq!(upstream["tools"][0]["type"], "function");
        assert_eq!(upstream["tools"][0]["name"], "read_file");
        assert_eq!(upstream["tools"][1]["type"], "function");
        assert_eq!(upstream["tools"][1]["name"], "apply_patch");
        let custom_description = upstream["tools"][1]["description"]
            .as_str()
            .expect("custom tool description");
        assert!(custom_description.contains("Apply a patch."));
        assert!(custom_description.contains("raw tool input in the `input` string field"));
        assert!(custom_description.contains("x-provider-field"));
        assert_eq!(
            upstream["tools"][1]["parameters"]["properties"]["input"]["type"],
            "string"
        );
        assert_eq!(upstream["tools"][1]["parameters"]["required"][0], "input");
        assert_eq!(upstream["tool_choice"]["type"], "function");
        assert_eq!(upstream["tool_choice"]["name"], "apply_patch");

        let deserialized: FunctionTool =
            serde_json::from_value(upstream["tools"][1].clone()).expect("upstream function tool should deserialize");
        assert_eq!(deserialized.name, "apply_patch");
    }

    #[test]
    fn to_upstream_request_rejects_custom_tool_grammar_formats() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tools": [{
                "type": "custom",
                "name": "constrained_input",
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": "start: value"
                }
            }]
        }))
        .expect("request");

        let error = payload
            .to_upstream_request(false)
            .expect_err("unsupported grammar must fail closed");
        assert!(error.to_string().contains("cannot preserve constrained decoding"));
    }

    #[test]
    fn to_upstream_request_normalizes_custom_allowed_tool_choices() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [
                    {"type": "function", "name": "read_file"},
                    {"type": "custom", "name": "apply_patch"}
                ]
            },
            "tools": [
                {"type": "function", "name": "read_file"},
                {"type": "custom", "name": "apply_patch"}
            ]
        }))
        .unwrap();

        let public_choice = serde_json::to_value(payload.tool_choice.as_ref().unwrap()).unwrap();
        assert_eq!(public_choice["tools"][1]["type"], "custom");

        let upstream = serde_json::to_value(payload.to_upstream_request(false).unwrap()).unwrap();
        assert_eq!(upstream["tool_choice"]["type"], "allowed_tools");
        assert_eq!(upstream["tool_choice"]["mode"], "required");
        assert_eq!(upstream["tool_choice"]["tools"][0]["type"], "function");
        assert_eq!(upstream["tool_choice"]["tools"][1]["type"], "function");
        assert_eq!(upstream["tool_choice"]["tools"][1]["name"], "apply_patch");
    }

    #[test]
    fn to_upstream_request_rejects_custom_choice_for_function_declaration() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tool_choice": {"type": "custom", "name": "echo"},
            "tools": [{"type": "function", "name": "echo"}]
        }))
        .expect("request");

        let error = payload
            .to_upstream_request(false)
            .expect_err("a custom selector must match a custom declaration");
        assert!(error.to_string().contains("no matching custom tool is declared"));
    }

    #[test]
    fn to_upstream_request_rejects_unknown_custom_allowed_tool() {
        let payload: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "input": "hi",
            "tool_choice": {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "custom", "name": "missing"}]
            },
            "tools": [{"type": "custom", "name": "apply_patch"}]
        }))
        .expect("request");

        let error = payload
            .to_upstream_request(false)
            .expect_err("an allowed custom selector must match a custom declaration");
        assert!(error.to_string().contains("no matching custom tool is declared"));
    }

    #[test]
    fn responses_input_discards_unknown_items_when_converted_for_storage() {
        let input: ResponsesInput = serde_json::from_value(serde_json::json!([
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "future_item", "payload": {"a": 1}}
        ]))
        .unwrap();

        let items = Vec::<InputItem>::from(&input);
        assert_eq!(items.len(), 1);
        assert!(matches!(items[0], InputItem::Message(_)));
    }

    #[test]
    fn response_payload_terminal_chunk_uses_status_specific_event_type() {
        let mut payload = ResponsePayload {
            id: "resp_test".to_string(),
            object: "response".to_string(),
            created_at: 0,
            model: "test-model".to_string(),
            status: "completed".to_string(),
            output: Vec::new(),
            usage: None,
            incomplete_details: None,
            error: None,
            previous_response_id: None,
            conversation_id: None,
            instructions: None,
        };

        for (status, expected_type) in [
            ("completed", "response.completed"),
            ("incomplete", "response.incomplete"),
            ("failed", "response.failed"),
            ("error", "response.failed"),
            ("in_progress", "response.in_progress"),
        ] {
            payload.status = status.to_string();
            let chunk = payload.as_terminal_response_chunk();
            let data = chunk.trim().strip_prefix("data: ").unwrap();
            let event: Value = serde_json::from_str(data).unwrap();
            assert_eq!(event["type"], expected_type);
            assert_eq!(event["response"]["status"], status);
        }
    }

    #[test]
    fn response_payload_created_chunk_uses_in_progress_status() {
        let payload = ResponsePayload {
            id: "resp_test".to_string(),
            object: "response".to_string(),
            created_at: 0,
            model: "test-model".to_string(),
            status: "completed".to_string(),
            output: Vec::new(),
            usage: None,
            incomplete_details: None,
            error: None,
            previous_response_id: None,
            conversation_id: None,
            instructions: None,
        };

        let chunk = payload.as_created_response_chunk();
        let data = chunk.trim().strip_prefix("data: ").unwrap();
        let event: Value = serde_json::from_str(data).unwrap();
        assert_eq!(event["type"], "response.created");
        assert_eq!(event["response"]["id"], "resp_test");
        assert_eq!(event["response"]["status"], "in_progress");
    }
}
