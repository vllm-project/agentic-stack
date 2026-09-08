use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Error returned when a tool name is empty.
///
/// Kept in `types/` so the wire-shape module stays self-contained and does
/// not import from the behavioral layer (`tool/`).
#[derive(Debug, thiserror::Error)]
#[error("tool name must not be empty")]
pub struct EmptyToolNameError;

/// A non-empty tool name, validated at construction.
///
/// Eliminates scattered empty-name checks by making the invalid state
/// (`name = ""`) unrepresentable. Use [`TryFrom<String>`] or
/// [`TryFrom<&str>`] to construct; serde rejects empty strings automatically.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct NonEmptyToolName(String);

impl NonEmptyToolName {
    /// Returns the name as a `&str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[cfg(feature = "openapi")]
impl utoipa::PartialSchema for NonEmptyToolName {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        utoipa::openapi::ObjectBuilder::new()
            .schema_type(utoipa::openapi::schema::SchemaType::new(
                utoipa::openapi::schema::Type::String,
            ))
            .min_length(Some(1))
            .into()
    }
}

#[cfg(feature = "openapi")]
impl utoipa::ToSchema for NonEmptyToolName {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("NonEmptyToolName")
    }
}

impl TryFrom<String> for NonEmptyToolName {
    type Error = EmptyToolNameError;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        if s.is_empty() {
            Err(EmptyToolNameError)
        } else {
            Ok(Self(s))
        }
    }
}

impl TryFrom<&str> for NonEmptyToolName {
    type Error = EmptyToolNameError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::try_from(s.to_owned())
    }
}

impl From<NonEmptyToolName> for String {
    fn from(n: NonEmptyToolName) -> String {
        n.0
    }
}

impl AsRef<str> for NonEmptyToolName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NonEmptyToolName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

// Request-side tool params  (serde-enum-representation, api-non-exhaustive)

/// Wire-compatible with the existing `{"type":"function",...}` format.
///
/// Marked `#[non_exhaustive]` because the Responses API adds new tool types
/// (e.g. `computer_use_preview`). Downstream match arms must include a catch-all.
/// Codex `namespace` tools stay in this public request/storage shape and are
/// flattened inside the upstream request conversion path.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesTool {
    #[serde(rename = "function")]
    Function(FunctionToolParam),
    #[serde(rename = "tool_search")]
    ToolSearch(ToolSearchToolParam),
    #[serde(rename = "mcp")]
    Mcp(McpToolParam),
    #[serde(
        rename = "web_search_preview",
        alias = "web_search",
        alias = "web_search_preview_2025_03_11",
        alias = "web_search_2025_08_26"
    )]
    WebSearch(WebSearchToolParam),
    #[serde(rename = "file_search")]
    FileSearch(FileSearchToolParam),
    #[serde(rename = "code_interpreter")]
    CodeInterpreter(CodeInterpreterToolParam),
    #[serde(rename = "namespace")]
    Namespace(CodexNamespaceToolParam),
    /// A freeform tool declaration. Unlike a function tool, calls carry raw
    /// text in `custom_tool_call.input` rather than JSON arguments.
    #[serde(rename = "custom")]
    Custom(CustomToolParam),
    #[serde(rename = "unknown", other)]
    Unknown,
}

/// Parameters for a user-defined function tool.
///
/// Does NOT carry a `type` field — serde consumes the tag during
/// deserialization and the payload struct must not also carry it.
///
/// `name` is a [`NonEmptyToolName`]: serde rejects empty strings at
/// deserialization time, making the invalid state unrepresentable.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct FunctionToolParam {
    pub name: NonEmptyToolName,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
    #[serde(default)]
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Parameters for a freeform (`type: "custom"`) tool.
///
/// `format` remains opaque at the wire boundary so declarations round-trip,
/// but the gateway rejects formatted custom tools before normalization because
/// it cannot preserve their constrained-decoding semantics upstream.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CustomToolParam {
    pub name: NonEmptyToolName,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
    #[serde(default)]
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Only client-executed tool search is part of the public gateway contract.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSearchExecution {
    #[default]
    Client,
    // TODO: Support `Server` execution type for gateway built-in tool
}

/// Lifecycle status of a public tool-search call or output item.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSearchStatus {
    InProgress,
    #[default]
    Completed,
    Incomplete,
}

/// Parameters for a client-executed tool-search declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSearchToolParam {
    pub execution: ToolSearchExecution,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

/// Parameters for a gateway MCP built-in tool declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct McpToolParam {
    pub server_label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connector_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_approval: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
    /// Request-scoped `tools/list` results used by MCP normalization. This
    /// field is populated internally and ignored on the public request wire.
    #[serde(
        rename = "_agentic_discovered_tools",
        default,
        skip_deserializing,
        skip_serializing_if = "Vec::is_empty"
    )]
    #[cfg_attr(feature = "openapi", schema(ignore))]
    pub(crate) discovered_tools: Vec<McpDiscoveredToolParam>,
}

/// Parameters for a discovered MCP (Model Context Protocol) server tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpDiscoveredToolParam {
    pub server_label: String,
    pub tool_name: String,
    pub internal_name: String,
    pub tool: rmcp::model::Tool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum WebSearchContextSize {
    Low,
    Medium,
    High,
}

impl WebSearchContextSize {
    pub(crate) const fn default_count(self) -> u8 {
        match self {
            Self::Low => 3,
            Self::Medium => 5,
            Self::High => 10,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct WebSearchFilters {
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct WebSearchUserLocation {
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub city: Option<String>,
    pub country: Option<String>,
    pub region: Option<String>,
    pub timezone: Option<String>,
}

/// Parameters for a web search tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct WebSearchToolParam {
    pub search_context_size: Option<WebSearchContextSize>,
    pub filters: Option<WebSearchFilters>,
    pub user_location: Option<WebSearchUserLocation>,
}

/// Parameters for a file search tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct FileSearchToolParam {
    pub vector_store_ids: Option<Vec<String>>,
}

/// Parameters for a code interpreter tool (no required fields).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CodeInterpreterToolParam {}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CodexNamespaceToolParam {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub tools: Vec<CodexNamespaceMember>,
    #[serde(default)]
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CodexNamespaceMember {
    #[serde(rename = "function")]
    Function(FunctionToolParam),
    #[serde(rename = "unknown", other)]
    Unknown,
}

#[cfg(feature = "openapi")]
impl utoipa::PartialSchema for ResponsesTool {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::Ref;
        use utoipa::openapi::schema::{AllOfBuilder, ObjectBuilder, SchemaType, Type};

        fn tagged(type_value: &str, schema: &str) -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
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
                .item(Ref::from_schema_name(schema))
                .into()
        }

        utoipa::openapi::schema::OneOfBuilder::new()
            .discriminator(Some(utoipa::openapi::schema::Discriminator::new("type")))
            .item(tagged("function", "FunctionToolParam"))
            .item(tagged("mcp", "McpToolParam"))
            .item(
                AllOfBuilder::new()
                    .item(
                        ObjectBuilder::new()
                            .property(
                                "type",
                                ObjectBuilder::new()
                                    .schema_type(SchemaType::new(Type::String))
                                    .enum_values(Some([
                                        "web_search_preview",
                                        "web_search",
                                        "web_search_preview_2025_03_11",
                                        "web_search_2025_08_26",
                                    ])),
                            )
                            .required("type"),
                    )
                    .item(Ref::from_schema_name("WebSearchToolParam")),
            )
            .item(tagged("file_search", "FileSearchToolParam"))
            .item(tagged("code_interpreter", "CodeInterpreterToolParam"))
            .item(tagged("namespace", "CodexNamespaceToolParam"))
            .item(tagged("custom", "CustomToolParam"))
            .into()
    }
}
#[cfg(feature = "openapi")]
impl utoipa::ToSchema for ResponsesTool {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("ResponsesTool")
    }
}

#[cfg(feature = "openapi")]
impl utoipa::PartialSchema for CodexNamespaceMember {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::Ref;
        use utoipa::openapi::schema::{AllOfBuilder, ObjectBuilder, SchemaType, Type};

        utoipa::openapi::schema::OneOfBuilder::new()
            .discriminator(Some(utoipa::openapi::schema::Discriminator::new("type")))
            .item(
                AllOfBuilder::new()
                    .item(
                        ObjectBuilder::new()
                            .property(
                                "type",
                                ObjectBuilder::new()
                                    .schema_type(SchemaType::new(Type::String))
                                    .enum_values(Some(["function"])),
                            )
                            .required("type"),
                    )
                    .item(Ref::from_schema_name("FunctionToolParam")),
            )
            .into()
    }
}
#[cfg(feature = "openapi")]
impl utoipa::ToSchema for CodexNamespaceMember {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("CodexNamespaceMember")
    }
}

impl ResponsesTool {
    #[must_use]
    pub fn original_type(&self) -> Option<&str> {
        match self {
            Self::Function(_) => Some("function"),
            Self::ToolSearch(_) => Some("tool_search"),
            Self::Mcp(_) => Some("mcp"),
            Self::WebSearch(_) => Some("web_search_preview"),
            Self::FileSearch(_) => Some("file_search"),
            Self::CodeInterpreter(_) => Some("code_interpreter"),
            Self::Namespace(_) => Some("namespace"),
            Self::Custom(_) => Some("custom"),
            Self::Unknown => None,
        }
    }

    /// Removes request-scoped MCP state before a tool declaration is persisted
    /// as effective response metadata.
    pub(crate) fn sanitize_for_persistence(&mut self) {
        if let Self::Mcp(param) = self {
            param.headers = None;
            param.authorization = None;
            param.discovered_tools.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_empty_name_accepts_valid() {
        let n = NonEmptyToolName::try_from("get_weather").unwrap();
        assert_eq!(n.as_str(), "get_weather");
    }

    #[test]
    fn non_empty_name_rejects_empty() {
        assert!(NonEmptyToolName::try_from(String::new()).is_err());
        assert!(NonEmptyToolName::try_from("").is_err());
    }

    #[test]
    fn non_empty_name_serde_round_trips() {
        let json = serde_json::json!("get_weather");
        let n: NonEmptyToolName = serde_json::from_value(json).unwrap();
        assert_eq!(n.as_str(), "get_weather");
        assert_eq!(serde_json::to_value(&n).unwrap(), serde_json::json!("get_weather"));
    }

    #[test]
    fn non_empty_name_serde_rejects_empty() {
        assert!(serde_json::from_value::<NonEmptyToolName>(serde_json::json!("")).is_err());
    }

    #[test]
    fn responses_tool_function_round_trips() {
        let json = serde_json::json!({
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            "x-extra": "kept"
        });
        let tool: ResponsesTool = serde_json::from_value(json).unwrap();
        assert!(matches!(tool, ResponsesTool::Function(_)));
        if let ResponsesTool::Function(ref p) = tool {
            assert_eq!(p.name.as_str(), "get_weather");
        }
        let back = serde_json::to_value(&tool).unwrap();
        assert_eq!(back["type"], "function");
        assert_eq!(back["name"], "get_weather");
        assert_eq!(back["x-extra"], "kept");
    }

    #[test]
    fn responses_tool_mcp_round_trips_with_field_values() {
        let json = serde_json::json!({
            "type": "mcp",
            "server_label": "repo",
            "server_url": "http://localhost:9001/mcp",
            "headers": {"X-Request-ID": "request-1"},
            "authorization": "token",
            "allowed_tools": ["read_file"],
            "require_approval": "never",
            "defer_loading": false
        });
        let tool: ResponsesTool = serde_json::from_value(json).unwrap();
        let back = serde_json::to_value(&tool).unwrap();
        assert_eq!(back["type"], "mcp");
        assert_eq!(back["server_label"], "repo");
        assert_eq!(back["server_url"], "http://localhost:9001/mcp");
        assert_eq!(back["defer_loading"], false);
        if let ResponsesTool::Mcp(ref p) = tool {
            assert_eq!(p.server_label, "repo");
            assert_eq!(p.server_url.as_deref(), Some("http://localhost:9001/mcp"));
        }
    }

    #[test]
    fn responses_tool_mcp_removes_request_scoped_state_for_persistence() {
        let mut tool = serde_json::from_value::<ResponsesTool>(serde_json::json!({
            "type": "mcp",
            "server_label": "repo",
            "server_url": "https://mcp.example.test/mcp",
            "headers": {
                "Authorization": "Bearer header-secret",
                "X-Request-ID": "request-1"
            },
            "authorization": "field-secret",
            "allowed_tools": ["read_file"],
            "require_approval": "never"
        }))
        .unwrap();

        let ResponsesTool::Mcp(param) = &mut tool else {
            panic!("expected MCP tool");
        };
        param.discovered_tools.push(McpDiscoveredToolParam {
            server_label: "repo".to_owned(),
            tool_name: "read_file".to_owned(),
            internal_name: "mcp__repo__read_file".to_owned(),
            tool: serde_json::from_value(serde_json::json!({
                "name": "read_file",
                "inputSchema": {"type": "object"}
            }))
            .expect("discovered MCP tool"),
        });

        tool.sanitize_for_persistence();

        let persisted = serde_json::to_value(tool).unwrap();
        assert!(persisted.get("headers").is_none());
        assert!(persisted.get("authorization").is_none());
        assert!(persisted.get("_agentic_discovered_tools").is_none());
        assert_eq!(persisted["server_label"], "repo");
        assert_eq!(persisted["server_url"], "https://mcp.example.test/mcp");
        assert_eq!(persisted["allowed_tools"], serde_json::json!(["read_file"]));
        assert_eq!(persisted["require_approval"], "never");
    }

    #[test]
    fn responses_tool_search_declaration_round_trips_exactly() {
        let declaration = serde_json::json!({
            "type": "tool_search",
            "execution": "client",
            "description": "Find a tool for the requested task",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        });

        let tool: ResponsesTool = serde_json::from_value(declaration.clone()).expect("valid tool-search declaration");

        assert_eq!(tool.original_type(), Some("tool_search"));
        assert_eq!(tool.tool_type(), Some(crate::tool::ToolType::ToolSearch));
        assert!(
            !tool.is_gateway_owned(),
            "client-executed tool search must bypass gateway dispatch"
        );
        assert_eq!(
            serde_json::to_value(tool.to_function_tools()).unwrap(),
            serde_json::json!([{
                "type": "function",
                "name": "tool_search",
                "description": "Find a tool for the requested task",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                },
                "strict": false
            }]),
            "the upstream-normalization boundary lowers tool search exactly once"
        );
        assert_eq!(serde_json::to_value(tool).expect("tool serializes"), declaration);
    }

    #[test]
    fn responses_tool_search_declaration_omits_optional_fields() {
        let declaration = serde_json::json!({
            "type": "tool_search",
            "execution": "client"
        });

        let tool: ResponsesTool = serde_json::from_value(declaration.clone()).expect("valid minimal declaration");

        tool.validate().expect("omitted optional fields are valid");
        assert_eq!(serde_json::to_value(tool).expect("tool serializes"), declaration);
    }

    #[test]
    fn responses_tool_search_declaration_rejects_invalid_wire_shapes() {
        for declaration in [
            serde_json::json!({
                "type": "tool_search",
                "description": "Missing execution",
                "parameters": {"type": "object"}
            }),
            serde_json::json!({
                "type": "tool_search",
                "execution": "server",
                "description": "Hosted execution is excluded",
                "parameters": {"type": "object"}
            }),
        ] {
            assert!(
                serde_json::from_value::<ResponsesTool>(declaration).is_err(),
                "invalid tool-search wire shape must not fall back to an unknown tool"
            );
        }
    }

    #[test]
    fn responses_tool_search_preserves_unknown_parameters_before_behavioral_validation() {
        let declaration = serde_json::json!({
            "type": "tool_search",
            "execution": "client",
            "parameters": ["not", "a", "schema", "object"]
        });
        let tool: ResponsesTool = serde_json::from_value(declaration.clone()).expect("wire value is retained");

        assert_eq!(serde_json::to_value(&tool).expect("tool serializes"), declaration);
        assert!(
            tool.validate()
                .expect_err("private function lowering requires an object schema")
                .to_string()
                .contains("parameters must be a JSON object")
        );
    }

    #[test]
    fn responses_tool_search_declaration_accepts_model_facing_values_for_private_normalization() {
        for (description, parameters) in [
            ("   ", serde_json::json!({"type": "object"})),
            ("Find a tool", serde_json::json!({})),
            ("Find a tool", serde_json::json!({"type": "array"})),
        ] {
            let tool: ResponsesTool = serde_json::from_value(serde_json::json!({
                "type": "tool_search",
                "execution": "client",
                "description": description,
                "parameters": parameters
            }))
            .expect("structurally valid declaration");

            tool.validate()
                .expect("typed public values are normalized only when building the private synthetic function");
        }
    }

    #[test]
    fn responses_tool_mcp_ignores_unknown_fields() {
        let tool = serde_json::from_value::<ResponsesTool>(serde_json::json!({
            "type": "mcp",
            "name": "increment",
            "server_label": "repo",
            "server_url": "http://localhost:9001/mcp",
            "future_field": true
        }))
        .unwrap();

        let back = serde_json::to_value(tool).unwrap();
        assert_eq!(back["server_label"], "repo");
        assert!(back.get("name").is_none());
        assert!(back.get("future_field").is_none());
    }

    #[test]
    fn responses_tool_mcp_ignores_internal_discovery_field_from_request() {
        let tool = serde_json::from_value::<ResponsesTool>(serde_json::json!({
            "type": "mcp",
            "server_label": "repo",
            "server_url": "http://localhost:9001/mcp",
            "_agentic_discovered_tools": [{"not": "a discovered tool"}]
        }))
        .unwrap();

        let ResponsesTool::Mcp(param) = tool else {
            panic!("expected MCP tool");
        };
        assert!(param.discovered_tools.is_empty());
    }

    #[test]
    fn responses_tool_web_search_round_trips() {
        let json = serde_json::json!({"type": "web_search_preview"});
        let tool: ResponsesTool = serde_json::from_value(json).unwrap();
        assert!(matches!(tool, ResponsesTool::WebSearch(_)));
        assert_eq!(serde_json::to_value(&tool).unwrap()["type"], "web_search_preview");
    }

    #[test]
    fn responses_tool_web_search_accepts_openai_aliases() {
        for type_name in [
            "web_search",
            "web_search_preview",
            "web_search_preview_2025_03_11",
            "web_search_2025_08_26",
        ] {
            let json = serde_json::json!({"type": type_name});
            let tool: ResponsesTool = serde_json::from_value(json).unwrap();
            assert!(matches!(tool, ResponsesTool::WebSearch(_)));
        }
    }

    #[test]
    fn responses_tool_file_search_round_trips() {
        let json = serde_json::json!({"type": "file_search", "vector_store_ids": ["vs_abc"]});
        let tool: ResponsesTool = serde_json::from_value(json).unwrap();
        assert!(matches!(tool, ResponsesTool::FileSearch(_)));
        let back = serde_json::to_value(&tool).unwrap();
        assert_eq!(back["type"], "file_search");
        assert_eq!(back["vector_store_ids"][0], "vs_abc");
    }

    #[test]
    fn responses_tool_code_interpreter_round_trips() {
        let json = serde_json::json!({"type": "code_interpreter"});
        let tool: ResponsesTool = serde_json::from_value(json).unwrap();
        assert!(matches!(tool, ResponsesTool::CodeInterpreter(_)));
        assert_eq!(serde_json::to_value(&tool).unwrap()["type"], "code_interpreter");
    }

    #[test]
    fn mcp_tool_param_round_trips_with_tool_schema() {
        let json = serde_json::json!({
            "server_label": "my_server",
            "tool_name": "fetch",
            "internal_name": "mcp__my_server__fetch",
            "tool": {
                "name": "fetch",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"}
                    }
                }
            }
        });
        let param: McpDiscoveredToolParam = serde_json::from_value(json).unwrap();
        let back = serde_json::to_value(&param).unwrap();
        assert_eq!(back["server_label"], "my_server");
        assert_eq!(back["tool"]["inputSchema"]["properties"]["id"]["type"], "string");
    }

    #[test]
    fn codex_namespace_tool_shape_round_trips_and_unknowns_are_minimal() {
        let tools_json = serde_json::json!([
            {
                "type": "namespace",
                "name": "mcp__shell",
                "tools": [
                    {"type": "function", "name": "run", "parameters": {"type": "object"}},
                    {"type": "future_member", "opaque": true}
                ],
                "x-extra": "kept"
            },
            {
                "type": "future_tool",
                "opaque": true
            }
        ]);

        let tools: Vec<ResponsesTool> = serde_json::from_value(tools_json).unwrap();
        assert!(matches!(tools[0], ResponsesTool::Namespace(_)));
        assert!(matches!(tools[1], ResponsesTool::Unknown));
        if let ResponsesTool::Namespace(namespace) = &tools[0] {
            assert!(matches!(namespace.tools[0], CodexNamespaceMember::Function(_)));
            assert!(matches!(namespace.tools[1], CodexNamespaceMember::Unknown));
        }

        let serialized = serde_json::to_value(&tools).unwrap();
        assert_eq!(serialized[0]["tools"][0]["type"], "function");
        assert_eq!(serialized[0]["tools"][1], serde_json::json!({"type": "unknown"}));
        assert_eq!(serialized[1], serde_json::json!({"type": "unknown"}));
    }

    #[test]
    fn custom_tool_shape_round_trips_without_interpreting_its_format() {
        let tool: ResponsesTool = serde_json::from_value(serde_json::json!({
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch.",
            "defer_loading": false,
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": "start: patch",
                "future_option": true
            }
        }))
        .unwrap();

        assert!(matches!(tool, ResponsesTool::Custom(_)));
        let serialized = serde_json::to_value(tool).unwrap();
        assert_eq!(serialized["type"], "custom");
        assert_eq!(serialized["defer_loading"], false);
        assert_eq!(serialized["format"]["syntax"], "lark");
        assert_eq!(serialized["format"]["future_option"], true);
    }
}
