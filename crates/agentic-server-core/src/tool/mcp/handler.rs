use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::tool::handler::MAX_GATEWAY_TOOL_OUTPUT_BYTES;
use crate::tool::{GatewayExecutor, GatewayToolEventPlan, ToolError, ToolHandler, ToolOutput, ToolType};
use crate::types::io::FunctionTool;
use crate::types::io::output::{
    FunctionToolCall, GatewayCallStatus, McpCall, McpCallError, McpCallStatus, McpListTool, McpListTools, OutputItem,
};
use crate::types::tools::{McpDiscoveredToolParam, McpToolParam, ResponsesTool};
use crate::utils::common::{deserialize_from_str, deserialize_from_str_opt, serialize_to_string};
use crate::utils::uuid7_str;

use super::{McpClient, McpError};

#[must_use]
pub(crate) fn output_item(
    call: &FunctionToolCall,
    output: &ToolOutput,
    status: GatewayCallStatus,
    server_label: &str,
    tool_name: &str,
) -> OutputItem {
    let error = if status == GatewayCallStatus::Failed {
        Some(McpCallError::tool_execution(error_text_from_output(&output.output)))
    } else {
        None
    };
    let successful_output = (status == GatewayCallStatus::Completed).then(|| output.output.clone());

    OutputItem::McpCall(McpCall::new(
        call_output_id(call),
        server_label.to_owned(),
        tool_name.to_owned(),
        call.arguments.clone(),
        status.into(),
        successful_output,
        error,
    ))
}

#[must_use]
pub(crate) fn started_output_item(call: &FunctionToolCall, server_label: &str, tool_name: &str) -> OutputItem {
    OutputItem::McpCall(McpCall::new(
        call_output_id(call),
        server_label.to_owned(),
        tool_name.to_owned(),
        "",
        McpCallStatus::InProgress,
        None,
        None,
    ))
}

#[must_use]
pub(crate) fn list_tools_output_item(item: &McpListTools) -> OutputItem {
    OutputItem::McpListTools(item.clone())
}

#[must_use]
pub(crate) fn started_list_tools_output_item(item: &McpListTools) -> OutputItem {
    OutputItem::McpListTools(McpListTools::new(item.id.clone(), item.server_label.clone(), vec![]))
}

/// Executes one tool discovered from an MCP server.
///
/// A handler with no client is used only while normalizing the discovered tool
/// metadata stored on `McpToolParam` into model-visible function tools.
pub struct McpHandler {
    client: Option<Arc<McpClient>>,
}

#[derive(Clone)]
pub struct McpDiscoveredHandler {
    pub param: McpDiscoveredToolParam,
    pub handler: Arc<McpHandler>,
}

#[derive(Clone)]
pub(crate) struct McpServerToolSet {
    pub discovered_handlers: Vec<McpDiscoveredHandler>,
    pub list_tools_item: McpListTools,
}

impl McpHandler {
    /// Validates request-level MCP server identities before any discovery I/O.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] when multiple MCP declarations use the
    /// same `server_label`.
    pub(crate) fn validate_server_labels(tools: &[ResponsesTool]) -> Result<(), ToolError> {
        let mut server_labels = HashSet::new();
        for param in tools.iter().filter_map(|tool| match tool {
            ResponsesTool::Mcp(param) => Some(param),
            _ => None,
        }) {
            if !server_labels.insert(param.server_label.clone()) {
                return Err(ToolError::Config(format!(
                    "duplicate MCP declarations are not allowed for server_label '{}'",
                    param.server_label
                )));
            }
        }
        Ok(())
    }

    #[must_use]
    pub const fn discovered_tool_spec_only() -> Self {
        Self { client: None }
    }

    #[must_use]
    pub fn tool_call(client: Arc<McpClient>) -> Self {
        Self { client: Some(client) }
    }

    /// Discovers and normalizes the tools exposed by one MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Execution`] when the server's `tools/list`
    /// operation fails or times out.
    pub async fn discovered_tool_handlers(
        server_label: &str,
        client: Arc<McpClient>,
        allowed_tools: Option<&[String]>,
    ) -> Result<Vec<McpDiscoveredHandler>, ToolError> {
        let tools = client
            .list_tools()
            .await
            .map_err(|error| mcp_discovery_error(server_label, &error))?;

        let mut discovered_handlers = Vec::new();
        let mut internal_names = HashMap::new();
        for tool in tools {
            let tool_name = tool.name.to_string();
            if allowed_tools.is_some_and(|allowed| !allowed.iter().any(|name| name == &tool_name)) {
                continue;
            }
            let internal_name = internal_mcp_tool_name(server_label, &tool_name, &mut internal_names);
            let handler = Arc::new(Self::tool_call(Arc::clone(&client)));
            discovered_handlers.push(McpDiscoveredHandler {
                param: McpDiscoveredToolParam {
                    server_label: server_label.to_owned(),
                    tool_name,
                    internal_name,
                    tool,
                },
                handler,
            });
        }

        Ok(discovered_handlers)
    }

    pub(crate) async fn discover_tools(
        server_label: &str,
        client: Arc<McpClient>,
        allowed_tools: Option<&[String]>,
    ) -> Result<McpServerToolSet, ToolError> {
        let handlers = Self::discovered_tool_handlers(server_label, client, allowed_tools).await?;
        Ok(Self::server_tool_set_from_handlers(server_label, handlers))
    }

    #[must_use]
    pub(crate) fn server_tool_set_from_handlers(
        server_label: &str,
        discovered_handlers: Vec<McpDiscoveredHandler>,
    ) -> McpServerToolSet {
        let tools = discovered_handlers
            .iter()
            .map(|discovered| mcp_list_tool(&discovered.param))
            .collect();

        McpServerToolSet {
            discovered_handlers,
            list_tools_item: McpListTools::new(uuid7_str("mcpl_"), server_label, tools),
        }
    }

    #[must_use]
    pub(crate) fn failed_list_tools_item(server_label: &str, error: &ToolError) -> McpListTools {
        let mut item = McpListTools::new(uuid7_str("mcpl_"), server_label, Vec::new());
        item.error = Some(error.to_string());
        item
    }

    /// Returns the spec-only MCP tool handler used during request normalization.
    #[must_use]
    pub const fn spec_from_param(_param: &McpToolParam) -> Self {
        Self::discovered_tool_spec_only()
    }
}

fn mcp_discovery_error(server_label: &str, error: &McpError) -> ToolError {
    ToolError::Execution(format!("tools/list failed for MCP server '{server_label}': {error}"))
}

fn mcp_list_tool(param: &McpDiscoveredToolParam) -> McpListTool {
    let tool = &param.tool;
    let read_only = tool
        .annotations
        .as_ref()
        .and_then(|annotations| annotations.read_only_hint)
        .unwrap_or(false);
    let annotations = Value::Object([("read_only".to_owned(), Value::Bool(read_only))].into_iter().collect());

    McpListTool::new(
        param.tool_name.clone(),
        tool.description.as_deref().map(str::to_owned),
        Value::Object(tool.input_schema.as_ref().clone()),
        Some(annotations),
    )
}

impl ToolHandler for McpHandler {
    type ToolParams = McpToolParam;

    fn tool_type(&self) -> ToolType {
        ToolType::Mcp
    }

    fn validate(&self, params: &McpToolParam) -> Result<(), ToolError> {
        if params.defer_loading == Some(true) {
            return Err(ToolError::Config(format!(
                "MCP tool '{}' cannot use defer_loading because deferred MCP tools are not supported",
                params.server_label
            )));
        }
        Ok(())
    }

    fn normalize(&self, params: &McpToolParam) -> Vec<FunctionTool> {
        params
            .discovered_tools
            .iter()
            .map(discovered_mcp_function_tool)
            .collect()
    }
}

impl GatewayExecutor for McpHandler {
    type ExecutionParams = McpDiscoveredToolParam;

    fn execute(
        &self,
        call_id: &str,
        _tool_name: &str,
        arguments: &str,
        params: &McpDiscoveredToolParam,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let call_id = call_id.to_owned();
        let arguments = arguments.to_owned();
        let client = self.client.clone();
        let server_label = params.server_label.clone();
        let tool_name = params.tool_name.clone();

        Box::pin(async move {
            let Some(client) = &client else {
                return Err(ToolError::Config(
                    "MCP tool spec-only handler cannot execute tools".to_owned(),
                ));
            };
            let output = execute_tool_call(client, &server_label, &tool_name, &arguments).await?;

            Ok(ToolOutput { call_id, output })
        })
    }

    fn supports_parallel_execution(&self) -> bool {
        true
    }

    fn plan_gateway_events(&self, call: &FunctionToolCall, params: &McpDiscoveredToolParam) -> GatewayToolEventPlan {
        GatewayToolEventPlan::new(Some(started_output_item(call, &params.server_label, &params.tool_name)))
    }

    fn public_output(
        &self,
        call: &FunctionToolCall,
        output: &ToolOutput,
        status: GatewayCallStatus,
        params: &McpDiscoveredToolParam,
    ) -> Option<OutputItem> {
        Some(output_item(
            call,
            output,
            status,
            &params.server_label,
            &params.tool_name,
        ))
    }
}

async fn execute_tool_call(
    client: &McpClient,
    server_label: &str,
    mcp_tool_name: &str,
    arguments: &str,
) -> Result<String, ToolError> {
    let args = parse_tool_arguments(arguments)?;

    let result = client
        .call_tool(mcp_tool_name, Some(args))
        .await
        .map_err(|error| ToolError::Execution(format!("tools/call failed for MCP server '{server_label}': {error}")))?;

    mcp_tool_result_text(&result)
}

fn parse_tool_arguments(arguments: &str) -> Result<Value, ToolError> {
    let arguments = deserialize_from_str::<Value>(arguments)
        .map_err(|error| ToolError::Execution(format!("invalid MCP tool arguments: {error}")))?;
    if !arguments.is_object() {
        return Err(ToolError::Execution(
            "MCP tool arguments must be a JSON object".to_owned(),
        ));
    }
    Ok(arguments)
}

fn mcp_tool_result_text(result: &rmcp::model::CallToolResult) -> Result<String, ToolError> {
    let mut text = String::new();
    for part in result
        .content
        .iter()
        .filter_map(|content| content.as_text().map(|text| text.text.as_str()))
    {
        let separator_bytes = usize::from(!text.is_empty());
        ensure_mcp_output_size(text.len().saturating_add(separator_bytes).saturating_add(part.len()))?;
        if separator_bytes == 1 {
            text.push('\n');
        }
        text.push_str(part);
    }
    let output = if !text.is_empty() {
        text
    } else if let Some(structured_content) = &result.structured_content {
        let output = serialize_to_string(structured_content)
            .map_err(|error| ToolError::Execution(format!("failed to serialize MCP structured content: {error}")))?;
        ensure_mcp_output_size(output.len())?;
        output
    } else {
        let output = serialize_to_string(&result.content)
            .map_err(|error| ToolError::Execution(format!("failed to serialize MCP tool content: {error}")))?;
        ensure_mcp_output_size(output.len())?;
        output
    };

    if result.is_error == Some(true) {
        Err(ToolError::Execution(output))
    } else {
        Ok(output)
    }
}

fn ensure_mcp_output_size(bytes: usize) -> Result<(), ToolError> {
    if bytes > MAX_GATEWAY_TOOL_OUTPUT_BYTES {
        return Err(ToolError::Execution(format!(
            "MCP tool output exceeded {MAX_GATEWAY_TOOL_OUTPUT_BYTES} bytes"
        )));
    }
    Ok(())
}

pub(crate) fn discovered_mcp_function_tool(param: &McpDiscoveredToolParam) -> FunctionTool {
    mcp_tool_to_function_tool(&param.internal_name, &param.tool)
}

const INTERNAL_MCP_PREFIX: &str = "mcp__";
const MAX_INTERNAL_TOOL_NAME_LEN: usize = 64;

fn internal_mcp_tool_name(server_label: &str, tool_name: &str, used: &mut HashMap<String, (String, String)>) -> String {
    let identity = (server_label.to_owned(), tool_name.to_owned());
    let base = sanitize_internal_tool_name(&format!("{INTERNAL_MCP_PREFIX}{server_label}__{tool_name}"));
    if base.len() <= MAX_INTERNAL_TOOL_NAME_LEN && used.get(&base).is_none_or(|existing| existing == &identity) {
        used.insert(base.clone(), identity);
        return base;
    }

    let mut attempt = 0_u32;
    loop {
        let hash_input = if attempt == 0 {
            format!("{server_label}:{tool_name}")
        } else {
            format!("{server_label}:{tool_name}:{attempt}")
        };
        let suffix = format!("__{:010x}", stable_name_hash(&hash_input) & 0xff_ffff_ffff);
        let prefix_len = MAX_INTERNAL_TOOL_NAME_LEN.saturating_sub(suffix.len());
        let candidate = format!("{}{}", &base[..base.len().min(prefix_len)], suffix);
        if used.get(&candidate).is_none_or(|existing| existing == &identity) {
            used.insert(candidate.clone(), identity);
            return candidate;
        }
        attempt = attempt.saturating_add(1);
    }
}

fn sanitize_internal_tool_name(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn stable_name_hash(value: &str) -> u64 {
    value.as_bytes().iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn mcp_tool_to_function_tool(name: &str, tool: &rmcp::model::Tool) -> FunctionTool {
    let mut parameters = Value::Object(tool.input_schema.as_ref().clone());

    if let Value::Object(object) = &mut parameters
        && object.get("properties").is_none_or(Value::is_null)
    {
        object.insert("properties".to_owned(), Value::Object(serde_json::Map::new()));
    }

    FunctionTool {
        type_: "function".to_owned(),
        name: name.to_owned(),
        description: tool.description.as_ref().map(ToString::to_string),
        parameters: Some(parameters),
        strict: Some(false),
    }
}

fn error_text_from_output(output: &str) -> String {
    deserialize_from_str_opt::<Value>(output)
        .and_then(|value| value.get("error").and_then(Value::as_str).map(str::to_owned))
        .filter(|error| !error.trim().is_empty())
        .unwrap_or_else(|| output.to_owned())
}

fn call_output_id(call: &FunctionToolCall) -> String {
    if let Some(suffix) = call.id.strip_prefix("fc_").filter(|suffix| !suffix.is_empty()) {
        return format!("mcp_{suffix}");
    }
    if let Some(suffix) = call.call_id.strip_prefix("call_").filter(|suffix| !suffix.is_empty()) {
        return format!("mcp_{suffix}");
    }
    let source_identity = format!("{}\0{}", call.id, call.call_id);
    format!("mcp_{:016x}", stable_name_hash(&source_identity))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn discovered_param() -> McpDiscoveredToolParam {
        McpDiscoveredToolParam {
            server_label: "counter".to_owned(),
            tool_name: "increment".to_owned(),
            internal_name: "mcp__counter__increment".to_owned(),
            tool: serde_json::from_value(serde_json::json!({
                "name": "increment",
                "description": "Increment the counter",
                "inputSchema": {"type": "object"}
            }))
            .expect("valid MCP tool"),
        }
    }

    #[test]
    fn native_mcp_param_without_discovery_normalizes_to_no_functions() {
        let param = serde_json::from_value::<McpToolParam>(serde_json::json!({
            "server_label": "counter",
            "server_url": "http://127.0.0.1:8000/mcp"
        }))
        .expect("MCP tool param");

        let handler = McpHandler::spec_from_param(&param);

        assert!(handler.normalize(&param).is_empty());
    }

    #[test]
    fn deferred_mcp_tools_are_rejected() {
        let param = serde_json::from_value::<McpToolParam>(serde_json::json!({
            "server_label": "counter",
            "server_url": "http://127.0.0.1:8000/mcp",
            "defer_loading": true
        }))
        .expect("MCP tool param");

        let error = McpHandler::spec_from_param(&param)
            .validate(&param)
            .expect_err("deferred MCP tools are not supported");
        assert!(error.to_string().contains("deferred MCP tools are not supported"));
    }

    #[test]
    fn discovered_tool_normalizes_to_function_tool() {
        let handler = McpHandler::discovered_tool_spec_only();
        let mut params = serde_json::from_value::<McpToolParam>(serde_json::json!({
            "server_label": "counter"
        }))
        .expect("MCP tool param");
        params.discovered_tools.push(discovered_param());

        let normalized = handler.normalize(&params);

        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0].name, "mcp__counter__increment");
        assert_eq!(
            normalized[0].parameters.as_ref().unwrap()["properties"],
            serde_json::json!({})
        );
    }

    #[test]
    fn discovery_builds_openai_list_tools_item_from_mcp_tools() {
        let mut read_only_param = discovered_param();
        read_only_param.tool_name = "get_value".to_owned();
        read_only_param.internal_name = "mcp__counter__get_value".to_owned();
        read_only_param.tool.name = "stale_raw_name".to_owned().into();
        read_only_param.tool.annotations = Some(rmcp::model::ToolAnnotations::new().read_only(true));

        let handlers = vec![discovered_param(), read_only_param]
            .into_iter()
            .map(|param| McpDiscoveredHandler {
                param,
                handler: Arc::new(McpHandler::discovered_tool_spec_only()),
            })
            .collect();

        let tool_set = McpHandler::server_tool_set_from_handlers("counter", handlers);

        assert!(tool_set.list_tools_item.id.starts_with("mcpl_"));
        assert_eq!(tool_set.list_tools_item.server_label, "counter");
        assert_eq!(tool_set.discovered_handlers.len(), 2);
        assert_eq!(
            tool_set
                .list_tools_item
                .tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            ["increment", "get_value"]
        );
        assert_eq!(
            tool_set.list_tools_item.tools[0].input_schema,
            serde_json::json!({"type": "object"})
        );
        assert_eq!(
            tool_set.list_tools_item.tools[0].annotations,
            Some(serde_json::json!({"read_only": false}))
        );
        assert_eq!(
            tool_set.list_tools_item.tools[1].annotations,
            Some(serde_json::json!({"read_only": true}))
        );
    }

    #[test]
    fn list_tools_output_items_share_identity_across_lifecycle() {
        let list_tools = McpListTools::new(
            "mcpl_1",
            "counter",
            vec![McpListTool::new(
                "increment",
                Some("Increment the counter".to_owned()),
                serde_json::json!({"type": "object", "properties": {}}),
                Some(serde_json::json!({"read_only": false})),
            )],
        );

        let OutputItem::McpListTools(started) = started_list_tools_output_item(&list_tools) else {
            panic!("expected started mcp_list_tools");
        };
        let OutputItem::McpListTools(completed) = list_tools_output_item(&list_tools) else {
            panic!("expected completed mcp_list_tools");
        };

        assert_eq!(started.id, "mcpl_1");
        assert_eq!(started.server_label, "counter");
        assert!(started.tools.is_empty());
        assert!(started.error.is_none());
        assert_eq!(completed.id, started.id);
        assert_eq!(completed.server_label, started.server_label);
        assert_eq!(completed.tools.len(), 1);
        assert_eq!(completed.tools[0].name, "increment");
    }

    #[test]
    fn tools_list_failure_preserves_upstream_cause_as_execution_error() {
        let upstream_error = super::super::McpError::Timeout {
            operation: super::super::McpOperation::ListTools,
        };

        let error = mcp_discovery_error("counter", &upstream_error);

        assert!(matches!(error, ToolError::Execution(_)));
        assert!(error.to_string().contains("tools/list failed for MCP server 'counter'"));
        assert!(error.to_string().contains("timed out during tools/list"));
    }

    #[test]
    fn mcp_tool_arguments_require_valid_json_object() {
        assert_eq!(
            parse_tool_arguments(r#"{"amount":1}"#).unwrap(),
            serde_json::json!({"amount": 1})
        );

        let malformed = parse_tool_arguments(r#"{"amount":"#).unwrap_err();
        assert!(matches!(
            malformed,
            ToolError::Execution(message) if message.contains("invalid MCP tool arguments")
        ));

        let non_object = parse_tool_arguments("null").unwrap_err();
        assert!(matches!(
            non_object,
            ToolError::Execution(message) if message == "MCP tool arguments must be a JSON object"
        ));
    }

    #[test]
    fn internal_tool_names_include_server_and_tool_identity() {
        let mut used = HashMap::new();

        let name = internal_mcp_tool_name("counter server", "increment/value", &mut used);

        assert_eq!(name, "mcp__counter_server__increment_value");
    }

    #[test]
    fn discovered_tool_output_uses_public_mcp_identity() {
        let call = FunctionToolCall {
            id: "fc_1".to_owned(),
            call_id: "call_1".to_owned(),
            name: "mcp__counter__increment".to_owned(),
            arguments: "{}".to_owned(),
            status: crate::types::event::MessageStatus::Completed,
            namespace: None,
        };
        let output = ToolOutput {
            call_id: call.call_id.clone(),
            output: "1".to_owned(),
        };

        let OutputItem::McpCall(item) =
            output_item(&call, &output, GatewayCallStatus::Completed, "counter", "increment")
        else {
            panic!("expected mcp_call");
        };

        assert_eq!(item.server_label, "counter");
        assert_eq!(item.name, "increment");
        assert_eq!(item.arguments, "{}");
        assert_eq!(item.output.as_deref(), Some("1"));
    }

    #[test]
    fn prefixless_function_ids_reuse_public_mcp_id_across_lifecycle() {
        let call = FunctionToolCall {
            id: "provider-item-1".to_owned(),
            call_id: "provider-call-1".to_owned(),
            name: "mcp__counter__increment".to_owned(),
            arguments: "{}".to_owned(),
            status: crate::types::event::MessageStatus::Completed,
            namespace: None,
        };
        let output = ToolOutput {
            call_id: call.call_id.clone(),
            output: "1".to_owned(),
        };

        let OutputItem::McpCall(started) = started_output_item(&call, "counter", "increment") else {
            panic!("expected started mcp_call");
        };
        let OutputItem::McpCall(completed) =
            output_item(&call, &output, GatewayCallStatus::Completed, "counter", "increment")
        else {
            panic!("expected completed mcp_call");
        };

        assert!(started.id.starts_with("mcp_"));
        assert_eq!(started.id, completed.id);
    }

    #[test]
    fn idless_parallel_calls_receive_distinct_public_mcp_ids() {
        let calls = [
            serde_json::from_value::<FunctionToolCall>(serde_json::json!({
                "name": "mcp__counter__increment",
                "arguments": "{}"
            }))
            .expect("valid first function call"),
            serde_json::from_value::<FunctionToolCall>(serde_json::json!({
                "name": "mcp__counter__increment",
                "arguments": "{}"
            }))
            .expect("valid second function call"),
        ];

        let public_ids = calls
            .iter()
            .map(|call| {
                let OutputItem::McpCall(started) = started_output_item(call, "counter", "increment") else {
                    panic!("expected started mcp_call");
                };
                let output = ToolOutput {
                    call_id: call.call_id.clone(),
                    output: "1".to_owned(),
                };
                let OutputItem::McpCall(completed) =
                    output_item(call, &output, GatewayCallStatus::Completed, "counter", "increment")
                else {
                    panic!("expected completed mcp_call");
                };

                assert_eq!(started.id, completed.id);
                started.id
            })
            .collect::<Vec<_>>();

        assert_ne!(public_ids[0], public_ids[1]);
    }

    #[test]
    fn successful_mcp_result_exposes_text_instead_of_protocol_envelope() {
        let result = serde_json::from_value::<rmcp::model::CallToolResult>(serde_json::json!({
            "content": [{"type": "text", "text": "42"}],
            "isError": false
        }))
        .expect("valid MCP result");

        assert_eq!(mcp_tool_result_text(&result).unwrap(), "42");
    }

    #[test]
    fn mcp_error_result_becomes_execution_failure() {
        let result = serde_json::from_value::<rmcp::model::CallToolResult>(serde_json::json!({
            "content": [{"type": "text", "text": "missing field `b`"}],
            "isError": true
        }))
        .expect("valid MCP result");

        let error = mcp_tool_result_text(&result).unwrap_err();
        assert!(matches!(error, ToolError::Execution(message) if message == "missing field `b`"));
    }

    #[test]
    fn mcp_text_result_is_bounded_before_joining_content() {
        let part = "x".repeat(MAX_GATEWAY_TOOL_OUTPUT_BYTES / 2 + 1);
        let result = serde_json::from_value::<rmcp::model::CallToolResult>(serde_json::json!({
            "content": [
                {"type": "text", "text": part},
                {"type": "text", "text": part}
            ],
            "isError": false
        }))
        .expect("valid MCP result");

        let error = mcp_tool_result_text(&result).expect_err("oversized MCP text must fail");
        assert!(error.to_string().contains("MCP tool output exceeded"));
    }

    #[test]
    fn failed_mcp_output_uses_openai_structured_error() {
        let call = FunctionToolCall {
            id: "fc_1".to_owned(),
            call_id: "call_1".to_owned(),
            name: "mcp__counter__sum".to_owned(),
            arguments: r#"{"a":40}"#.to_owned(),
            status: crate::types::event::MessageStatus::Completed,
            namespace: None,
        };
        let output = ToolOutput {
            call_id: call.call_id.clone(),
            output: r#"{"error":"missing field `b`"}"#.to_owned(),
        };
        let item = output_item(&call, &output, GatewayCallStatus::Failed, "counter", "sum");
        let json = serde_json::to_value(item).expect("serializable mcp_call");

        assert_eq!(json["status"], "failed");
        assert!(json["output"].is_null());
        assert_eq!(json["error"]["type"], "mcp_tool_execution_error");
        assert_eq!(json["error"]["content"][0]["type"], "text");
        assert_eq!(json["error"]["content"][0]["text"], "missing field `b`");
        assert!(json["error"]["content"][0]["annotations"].is_null());
        assert!(json["error"]["content"][0]["meta"].is_null());
    }
}
