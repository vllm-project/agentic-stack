// Feature-gated OpenAPI spec assembly and Swagger UI router.
#![cfg(feature = "openapi")]

use utoipa::OpenApi;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "vLLM Agentic API",
        description = "Stateful agentic layer for vLLM — OpenAI-compatible Responses API and Anthropic-compatible Messages API.",
        license(name = "Apache-2.0"),
    ),
    paths(
        crate::handler::http::models::health,
        crate::handler::http::models::ready,
        crate::handler::http::models::models,
        crate::handler::http::responses::responses,
        crate::handler::http::responses::compact_response,
        crate::handler::http::conversations::conversations,
        crate::handler::http::messages::messages,
        crate::handler::http::messages::count_tokens,
    ),
    components(schemas(
        agentic_core::types::request_response::RequestPayload,
        agentic_core::types::request_response::ResponsePayload,
        agentic_core::types::request_response::CompactRequest,
        agentic_core::types::request_response::CompactedResponse,
        agentic_core::types::request_response::ContextManagement,
        agentic_core::types::request_response::IncompleteDetails,
        agentic_core::types::request_response::ResponseTextConfig,
        agentic_core::types::request_response::ResponseTextFormat,
        agentic_core::types::request_response::ReasoningConfig,
        agentic_core::types::io::ResponsesInput,
        agentic_core::types::io::InputItem,
        agentic_core::types::io::InputMessage,
        agentic_core::types::io::InputMessageContent,
        agentic_core::types::io::InputTextContent,
        agentic_core::types::io::InputImageContent,
        agentic_core::types::io::InputFileContent,
        agentic_core::types::io::InputContent,
        agentic_core::types::io::InputFunctionToolCall,
        agentic_core::types::io::FunctionToolResultMessage,
        agentic_core::types::io::ToolCallOutput,
        agentic_core::types::io::ToolOutputContent,
        agentic_core::types::io::CompactionItem,
        agentic_core::types::io::CustomToolCallOutputMessage,
        agentic_core::types::io::OutputItem,
        agentic_core::types::io::OutputTextContent,
        agentic_core::types::io::OutputMessage,
        agentic_core::types::io::FunctionToolCall,
        agentic_core::types::io::CustomToolCall,
        agentic_core::types::io::WebSearchCall,
        agentic_core::types::io::WebSearchAction,
        agentic_core::types::io::WebSearchActionSearch,
        agentic_core::types::io::WebSearchActionOpenPage,
        agentic_core::types::io::WebSearchActionFindInPage,
        agentic_core::types::io::WebSearchSource,
        agentic_core::types::io::McpCall,
        agentic_core::types::io::McpCallError,
        agentic_core::types::io::McpToolExecutionError,
        agentic_core::types::io::McpToolExecutionErrorContent,
        agentic_core::types::io::McpListTools,
        agentic_core::types::io::McpListTool,
        agentic_core::types::io::ReasoningOutput,
        agentic_core::types::io::ReasoningTextContent,
        agentic_core::types::io::ResponseUsage,
        agentic_core::types::io::InputTokenDetails,
        agentic_core::types::io::OutputTokenDetails,
        agentic_core::types::io::FunctionTool,
        agentic_core::types::io::ToolChoice,
        agentic_core::types::io::AllowedTool,
        agentic_core::types::io::AllowedToolsMode,
        agentic_core::types::io::GatewayCallStatus,
        agentic_core::types::io::McpCallStatus,
        agentic_core::types::event::ResponseStatus,
        agentic_core::types::event::MessageStatus,
        agentic_core::types::tools::ResponsesTool,
        agentic_core::types::tools::FunctionToolParam,
        agentic_core::types::tools::CustomToolParam,
        agentic_core::types::tools::McpToolParam,
        agentic_core::types::tools::WebSearchToolParam,
        agentic_core::types::tools::WebSearchContextSize,
        agentic_core::types::tools::WebSearchFilters,
        agentic_core::types::tools::WebSearchUserLocation,
        agentic_core::types::tools::FileSearchToolParam,
        agentic_core::types::tools::CodeInterpreterToolParam,
        agentic_core::types::tools::CodexNamespaceToolParam,
        agentic_core::types::tools::CodexNamespaceMember,
        agentic_core::types::tools::NonEmptyToolName,
        agentic_core::types::messages::MessagesRequest,
        agentic_core::types::messages::MessageParam,
        agentic_core::types::messages::MessageContent,
        agentic_core::types::messages::ContentBlock,
        agentic_core::types::messages::SystemPrompt,
        agentic_core::types::messages::SystemBlock,
        agentic_core::types::messages::ToolParam,
        agentic_core::types::messages::ToolResultContent,
        agentic_core::types::messages::ToolResultBlock,
        agentic_core::types::messages::OutputConfig,
        agentic_core::types::messages::ReasoningEffort,
        agentic_core::types::messages::ReasoningEffortLevel,
        ApiErrorResponse,
        ApiError,
        AnthropicErrorResponse,
        AnthropicError,
        CreateConversationRequest,
        ConversationResponse,
    )),
    modifiers(&SecurityAddon),
    tags(
        (name = "health", description = "Health and readiness probes"),
        (name = "models", description = "Model listing"),
        (name = "responses", description = "OpenAI-compatible Responses API"),
        (name = "conversations", description = "Conversation management"),
        (name = "messages", description = "Anthropic-compatible Messages API"),
    )
)]
pub struct ApiDoc;

struct SecurityAddon;

impl utoipa::Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        if let Some(components) = openapi.components.as_mut() {
            components.add_security_scheme(
                "bearer_auth",
                utoipa::openapi::security::SecurityScheme::Http(utoipa::openapi::security::Http::new(
                    utoipa::openapi::security::HttpAuthScheme::Bearer,
                )),
            );
        }
    }
}

/// OpenAI-style error envelope for Responses/Models endpoints.
#[derive(serde::Serialize, utoipa::ToSchema)]
pub struct ApiErrorResponse {
    pub error: ApiError,
}

#[derive(serde::Serialize, utoipa::ToSchema)]
pub struct ApiError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

/// Anthropic-style error envelope for Messages endpoints.
#[derive(serde::Serialize, utoipa::ToSchema)]
pub struct AnthropicErrorResponse {
    #[serde(rename = "type")]
    pub response_type: String,
    pub error: AnthropicError,
}

#[derive(serde::Serialize, utoipa::ToSchema)]
pub struct AnthropicError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

/// Request body for POST /v1/conversations.
#[derive(serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
pub struct CreateConversationRequest {
    pub store: bool,
}

/// Response from POST /v1/conversations.
#[derive(serde::Serialize, utoipa::ToSchema)]
pub struct ConversationResponse {
    pub id: String,
    pub created_at: i64,
    pub object: String,
    pub metadata: serde_json::Value,
}

pub fn swagger_ui_router<S>() -> axum::Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    let router: axum::Router<()> = utoipa_swagger_ui::SwaggerUi::new("/swagger-ui")
        .url("/openapi.json", ApiDoc::openapi())
        .into();
    router.with_state(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use utoipa::OpenApi;

    #[test]
    fn spec_contains_expected_paths() {
        let spec = ApiDoc::openapi();
        let paths: Vec<&str> = spec.paths.paths.keys().map(String::as_str).collect();

        for expected in [
            "/health",
            "/ready",
            "/v1/models",
            "/v1/responses",
            "/v1/responses/compact",
            "/v1/conversations",
            "/v1/messages",
            "/v1/messages/count_tokens",
        ] {
            assert!(paths.contains(&expected), "missing path: {expected}");
        }
    }

    #[test]
    fn spec_contains_core_schemas() {
        let spec = ApiDoc::openapi();
        let schemas = spec
            .components
            .as_ref()
            .expect("components must exist")
            .schemas
            .keys()
            .cloned()
            .collect::<Vec<_>>();

        for expected in [
            "RequestPayload",
            "ResponsePayload",
            "MessagesRequest",
            "ApiErrorResponse",
        ] {
            assert!(schemas.iter().any(|s| s == expected), "missing schema: {expected}");
        }
    }

    #[test]
    fn spec_has_security_scheme() {
        let spec = ApiDoc::openapi();
        let schemes = &spec
            .components
            .as_ref()
            .expect("components must exist")
            .security_schemes;
        assert!(
            schemes.contains_key("bearer_auth"),
            "bearer_auth security scheme missing"
        );
    }

    #[test]
    fn spec_serializes_to_valid_json() {
        let spec = ApiDoc::openapi();
        let json = serde_json::to_string_pretty(&spec).expect("spec must serialize to JSON");
        assert!(json.contains("\"openapi\""));
        assert!(json.contains("\"3."));
    }

    #[test]
    fn spec_schema_refs_resolve() {
        let spec = ApiDoc::openapi();
        let json = serde_json::to_value(&spec).expect("spec must serialize");
        let schemas = json["components"]["schemas"].as_object().expect("schemas must exist");

        let spec_str = serde_json::to_string(&json).unwrap();
        let prefix = "#/components/schemas/";
        let mut unresolved = Vec::new();
        for (idx, _) in spec_str.match_indices(prefix) {
            let rest = &spec_str[idx + prefix.len()..];
            let name = rest.split('"').next().unwrap_or("");
            if !name.is_empty() && !schemas.contains_key(name) {
                unresolved.push(name.to_owned());
            }
        }
        unresolved.sort();
        unresolved.dedup();
        assert!(unresolved.is_empty(), "unresolved $ref schemas: {unresolved:?}");
    }

    #[test]
    fn spec_paths_have_operations() {
        let spec = ApiDoc::openapi();
        for (path, item) in &spec.paths.paths {
            let has_op = item.get.is_some()
                || item.post.is_some()
                || item.put.is_some()
                || item.delete.is_some()
                || item.patch.is_some();
            assert!(has_op, "path {path} has no operations");
        }
    }

    #[test]
    fn spec_has_required_info_fields() {
        let spec = ApiDoc::openapi();
        assert!(!spec.info.title.is_empty(), "info.title must not be empty");
        assert_eq!(
            spec.info.version,
            env!("CARGO_PKG_VERSION"),
            "info.version should match crate version"
        );
        assert!(spec.info.license.is_some(), "info.license should be set");
    }

    /// Validates that JSON fixtures representing each tagged-enum variant
    /// pass the hand-written `OpenAPI` schema. Catches schema drift that
    /// structural tests (ref resolution, meta-schema) cannot.
    #[test]
    fn serialized_fixtures_validate_against_component_schemas() {
        let spec = ApiDoc::openapi();
        let spec_json = serde_json::to_value(&spec).expect("spec must serialize");
        let schemas = spec_json["components"]["schemas"]
            .as_object()
            .expect("schemas must exist");

        let validate = |schema_name: &str, fixture: &serde_json::Value| {
            let wrapper = serde_json::json!({
                "components": { "schemas": schemas },
                "$ref": format!("#/components/schemas/{schema_name}")
            });
            let validator = jsonschema::validator_for(&wrapper)
                .unwrap_or_else(|e| panic!("failed to compile schema for {schema_name}: {e}"));
            let errors: Vec<String> = validator
                .iter_errors(fixture)
                .map(|e| format!("  {} at {}", e, e.instance_path))
                .collect();
            assert!(
                errors.is_empty(),
                "{schema_name} fixture failed validation:\n{}\nfixture: {}",
                errors.join("\n"),
                serde_json::to_string_pretty(fixture).unwrap()
            );
        };

        // -- InputItem: every variant the deserializer accepts --
        let input_items = vec![
            serde_json::json!({"type": "message", "role": "user", "content": "hello"}),
            serde_json::json!({"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}),
            serde_json::json!({"type": "function_call_output", "call_id": "c1", "output": "ok"}),
            serde_json::json!({"type": "custom_tool_call", "id": "ct1", "name": "t", "input": "d"}),
            serde_json::json!({"type": "custom_tool_call_output", "call_id": "c1", "output": "r"}),
            serde_json::json!({"type": "reasoning", "id": "r1", "content": [{"type": "reasoning_text", "text": "think"}]}),
            serde_json::json!({"type": "mcp_list_tools", "id": "mlt1", "server_label": "s", "tools": []}),
            serde_json::json!({"type": "compaction", "id": "cmp1", "encrypted_content": "enc"}),
            serde_json::json!({"type": "compaction_trigger"}),
        ];
        for fixture in &input_items {
            validate("InputItem", fixture);
        }

        // -- OutputItem: every variant --
        let output_items = vec![
            serde_json::json!({"type": "message", "id": "m1", "role": "assistant", "status": "completed"}),
            serde_json::json!({"type": "function_call", "id": "fc1", "call_id": "c1", "name": "f", "arguments": "{}", "status": "completed"}),
            serde_json::json!({"type": "custom_tool_call", "id": "ct1", "name": "t", "input": "d"}),
            serde_json::json!({"type": "web_search_call", "id": "ws1", "status": "completed", "action": {"type": "search", "query": "q"}}),
            serde_json::json!({"type": "mcp_call", "id": "mc1", "server_label": "s", "name": "n", "arguments": "{}"}),
            serde_json::json!({"type": "mcp_list_tools", "id": "mlt1", "server_label": "s", "tools": []}),
            serde_json::json!({"type": "reasoning", "id": "r1", "content": [{"type": "reasoning_text", "text": "think"}]}),
            serde_json::json!({"type": "compaction", "encrypted_content": "enc"}),
        ];
        for fixture in &output_items {
            validate("OutputItem", fixture);
        }

        // -- ResponsesTool: every variant --
        let tools = vec![
            serde_json::json!({"type": "function", "name": "f"}),
            serde_json::json!({"type": "mcp", "server_label": "s"}),
            serde_json::json!({"type": "web_search_preview"}),
            serde_json::json!({"type": "file_search"}),
            serde_json::json!({"type": "code_interpreter"}),
            serde_json::json!({"type": "namespace", "name": "ns", "tools": []}),
            serde_json::json!({"type": "custom", "name": "c"}),
        ];
        for fixture in &tools {
            validate("ResponsesTool", fixture);
        }

        // Serde aliases for WebSearch must validate against the schema AND round-trip.
        for alias in ["web_search", "web_search_preview_2025_03_11", "web_search_2025_08_26"] {
            let fixture = serde_json::json!({"type": alias});
            validate("ResponsesTool", &fixture);
            let _: agentic_core::types::tools::ResponsesTool =
                serde_json::from_value(fixture).expect("alias must deserialize");
        }
    }

    #[test]
    fn spec_validates_against_openapi_3_1_meta_schema() {
        let meta_schema: serde_json::Value =
            serde_json::from_str(include_str!("../tests/openapi-3.1-schema.json")).expect("meta-schema must parse");
        let validator = jsonschema::validator_for(&meta_schema).expect("meta-schema must compile");

        let spec = ApiDoc::openapi();
        let spec_json = serde_json::to_value(&spec).expect("spec must serialize");

        let errors: Vec<String> = validator
            .iter_errors(&spec_json)
            .map(|e| format!("{} at {}", e, e.instance_path))
            .collect();
        assert!(
            errors.is_empty(),
            "OpenAPI spec failed meta-schema validation:\n{}",
            errors.join("\n")
        );
    }
}
