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
    #[serde(rename = "shell")]
    Shell(ShellToolParam),
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
pub struct CustomToolParam {
    pub name: NonEmptyToolName,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    #[serde(default)]
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Parameters for a gateway MCP built-in tool declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Request-scoped `tools/list` results used by MCP normalization. This
    /// field is populated internally and ignored on the public request wire.
    #[serde(
        rename = "_agentic_discovered_tools",
        default,
        skip_deserializing,
        skip_serializing_if = "Vec::is_empty"
    )]
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
pub struct WebSearchFilters {
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
pub struct WebSearchToolParam {
    pub search_context_size: Option<WebSearchContextSize>,
    pub filters: Option<WebSearchFilters>,
    pub user_location: Option<WebSearchUserLocation>,
}

/// Parameters for a file search tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileSearchToolParam {
    pub vector_store_ids: Option<Vec<String>>,
}

/// Parameters for a code interpreter tool (no required fields).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodeInterpreterToolParam {}

/// Parameters for the shell built-in tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellToolParam {
    pub environment: ShellEnvironment,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_callers: Option<Vec<String>>,
    #[serde(default, flatten)]
    pub extra: HashMap<String, Value>,
}

/// Environment in which shell calls are executed.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum ShellEnvironment {
    Local(LocalShellEnvironment),
    Unknown(Value),
}

impl Serialize for ShellEnvironment {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut value = match self {
            Self::Local(environment) => serde_json::to_value(environment).map_err(serde::ser::Error::custom)?,
            Self::Unknown(value) => return value.serialize(serializer),
        };
        let object = value
            .as_object_mut()
            .ok_or_else(|| serde::ser::Error::custom("shell environment must serialize as an object"))?;
        object.insert("type".to_owned(), Value::String("local".to_owned()));
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ShellEnvironment {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let mut value = Value::deserialize(deserializer)?;
        if value.get("type").and_then(Value::as_str) != Some("local") {
            return Ok(Self::Unknown(value));
        }
        value
            .as_object_mut()
            .expect("a value with a string type field must be an object")
            .remove("type");
        serde_json::from_value(value)
            .map(Self::Local)
            .map_err(serde::de::Error::custom)
    }
}

/// Caller-provided local environment configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LocalShellEnvironment {
    #[serde(default, flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl ResponsesTool {
    #[must_use]
    pub fn original_type(&self) -> Option<&str> {
        match self {
            Self::Function(_) => Some("function"),
            Self::Mcp(_) => Some("mcp"),
            Self::WebSearch(_) => Some("web_search_preview"),
            Self::FileSearch(_) => Some("file_search"),
            Self::CodeInterpreter(_) => Some("code_interpreter"),
            Self::Shell(_) => Some("shell"),
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
            "require_approval": "never"
        });
        let tool: ResponsesTool = serde_json::from_value(json).unwrap();
        let back = serde_json::to_value(&tool).unwrap();
        assert_eq!(back["type"], "mcp");
        assert_eq!(back["server_label"], "repo");
        assert_eq!(back["server_url"], "http://localhost:9001/mcp");
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
    fn responses_tool_shell_local_environment_round_trips() {
        let json = serde_json::json!({
            "type": "shell",
            "environment": {
                "type": "local",
                "skills": [{"name": "repo", "path": "/workspace/repo"}]
            },
            "allowed_callers": ["assistant"],
            "future_tool_field": true
        });
        let tool: ResponsesTool = serde_json::from_value(json).unwrap();
        assert!(matches!(tool, ResponsesTool::Shell(_)));

        let serialized = serde_json::to_value(tool).unwrap();
        assert_eq!(serialized["type"], "shell");
        assert_eq!(serialized["environment"]["type"], "local");
        assert_eq!(serialized["environment"]["skills"][0]["name"], "repo");
        assert_eq!(serialized["allowed_callers"][0], "assistant");
        assert_eq!(serialized["future_tool_field"], true);
    }

    #[test]
    fn responses_tool_shell_preserves_unknown_environment() {
        let json = serde_json::json!({
            "type": "shell",
            "environment": {
                "type": "container_reference",
                "container": "cntr_123",
                "future_environment_field": true
            }
        });
        let tool: ResponsesTool = serde_json::from_value(json.clone()).unwrap();
        assert!(matches!(
            tool,
            ResponsesTool::Shell(ShellToolParam {
                environment: ShellEnvironment::Unknown(_),
                ..
            })
        ));

        assert_eq!(serde_json::to_value(tool).unwrap(), json);
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
        assert_eq!(serialized["format"]["syntax"], "lark");
        assert_eq!(serialized["format"]["future_option"], true);
    }
}
