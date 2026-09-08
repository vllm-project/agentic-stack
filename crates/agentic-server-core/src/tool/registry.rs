use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::future::{Future, ready};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::codex::insert_namespace_entries;
use super::custom::{CustomHandler, CustomToolMap, insert_custom_entry};
use super::executors::GatewayExecutors;
use super::function::insert_function_entry;
use super::mcp::registry::insert_discovered_mcp_entry;
use super::ownership::{GatewayBinding, ToolOwnership};
use super::tool_search::{
    TOOL_SEARCH_NAME, ensure_request_prepared, insert_tool_search_entry, validate_blocking_response,
};
use super::web_search::insert_web_search_entry;
use super::{
    CodexNamespaceHandler, McpHandler, NamespaceMap, ToolError, ToolOutput, ToolSearchMetadata, ToolSearchState,
};
use crate::events::WireEvent;

use crate::types::io::output::{FunctionToolCall, McpListTools};
use crate::types::io::{InputItem, OutputItem, ResponsesInput};
use crate::types::request_response::RequestPayload;
use crate::types::tools::{CodeInterpreterToolParam, FileSearchToolParam, ResponsesTool};
use crate::utils::common::serialize_to_value;

const MAX_MCP_SERVERS_PER_REQUEST: usize = 64;
const MAX_DISCOVERED_MCP_TOOLS_PER_REQUEST: usize = 128;
const MAX_MCP_DISCOVERY_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
    ToolSearch,
    Custom,
    CodexNamespace,
    Mcp,
    /// Internal routing discriminant. Serializes as `"web_search"`.
    /// Note: the corresponding `ResponsesTool` wire tag is `"web_search_preview"`.
    /// `ToolType` is not used in wire-facing types so the names differ intentionally.
    WebSearch,
    FileSearch,
    CodeInterpreter,
}

impl ToolType {
    #[must_use]
    pub(crate) const fn description(self) -> &'static str {
        match self {
            Self::Function => "function tool",
            Self::ToolSearch => "tool search",
            Self::Custom => "custom tool",
            Self::CodexNamespace => "Codex namespace tool",
            Self::Mcp => "MCP tool",
            Self::WebSearch => "web search tool",
            Self::FileSearch => "file search tool",
            Self::CodeInterpreter => "code interpreter tool",
        }
    }

    /// Whether this kind of tool is gateway-owned by design, independent of
    /// any specific registry entry. Used before a `ToolEntry` exists (e.g.
    /// classifying a raw declaration); once an entry exists, prefer
    /// `ToolOwnership::is_gateway` on it directly.
    #[must_use]
    pub const fn is_gateway_owned(self) -> bool {
        !matches!(
            self,
            Self::Function | Self::ToolSearch | Self::Custom | Self::CodexNamespace
        )
    }
}

/// Per-request routing entry keyed by the tool name the model will call.
#[derive(Clone)]
pub struct ToolEntry {
    pub tool_type: ToolType,
    /// For MCP tools: which server this tool belongs to.
    pub server_label: Option<String>,
    pub ownership: ToolOwnership,
}

impl std::fmt::Debug for ToolEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolEntry")
            .field("tool_type", &self.tool_type)
            .field("server_label", &self.server_label)
            .field("is_gateway", &self.ownership.is_gateway())
            .finish()
    }
}

impl ToolEntry {
    /// Builds a client-owned entry. `tool_type.is_gateway_owned()` is the
    /// single source of truth for the ownership discriminant; this asserts
    /// the caller picked the constructor matching its own tool type.
    pub(crate) fn client(tool_type: ToolType, server_label: Option<String>) -> Self {
        debug_assert!(!tool_type.is_gateway_owned());
        Self {
            tool_type,
            server_label,
            ownership: ToolOwnership::Client,
        }
    }

    /// Builds a gateway-owned entry. `binding` is `None` for tool types that
    /// are gateway-owned in principle but have no executor yet.
    pub(crate) fn gateway(tool_type: ToolType, server_label: Option<String>, binding: Option<GatewayBinding>) -> Self {
        debug_assert!(tool_type.is_gateway_owned());
        Self {
            tool_type,
            server_label,
            ownership: ToolOwnership::Gateway(binding),
        }
    }
}

fn insert_unique_tool_entries(
    entries: &mut HashMap<String, ToolEntry>,
    insert: impl FnOnce(&mut HashMap<String, ToolEntry>),
) -> Result<(), ToolError> {
    let mut resolved = HashMap::new();
    insert(&mut resolved);
    for (name, entry) in resolved {
        match entries.entry(name) {
            Entry::Occupied(existing) => {
                return Err(ToolError::Config(format!(
                    "{} registry name '{}' conflicts with existing {}",
                    entry.tool_type.description(),
                    existing.key(),
                    existing.get().tool_type.description()
                )));
            }
            Entry::Vacant(vacant) => {
                vacant.insert(entry);
            }
        }
    }
    Ok(())
}

pub struct GatewayDispatchResult {
    pub tool_type: ToolType,
    pub output: Result<ToolOutput, ToolError>,
}

// TODO: move to a dedicated file_search module alongside its `ToolHandler`
// once file_search execution is implemented.
fn insert_file_search_entry(entries: &mut HashMap<String, ToolEntry>, _params: &FileSearchToolParam) {
    entries.insert(
        "file_search".to_owned(),
        ToolEntry::gateway(ToolType::FileSearch, None, None),
    );
}

// TODO: move to a dedicated code_interpreter module alongside its `ToolHandler`
// once code_interpreter execution is implemented.
fn insert_code_interpreter_entry(entries: &mut HashMap<String, ToolEntry>, _params: &CodeInterpreterToolParam) {
    entries.insert(
        "code_interpreter".to_owned(),
        ToolEntry::gateway(ToolType::CodeInterpreter, None, None),
    );
}

/// Request-scoped registry built from `RequestPayload.tools`.
/// Maps the name the LLM sees → routing metadata.
#[derive(Debug, Default)]
pub struct ToolRegistry {
    entries: HashMap<String, ToolEntry>,

    /// Prepared public/private tool-search projection for this request.
    tool_search: Option<Box<ToolSearchState>>,

    /// Built once from the declared tools, so final payload and streaming event
    /// restoration don't rebuild it on every call.
    namespace_map: Option<NamespaceMap>,

    /// Maps normalized custom function names back to their public declarations
    /// for response lifecycle metadata restoration.
    custom_tool_map: Option<CustomToolMap>,

    /// MCP tool-list items grouped by server label. Current discovery is stored
    /// first while building and rehydrated historical records are appended.
    /// Insertion order preserves the MCP declaration order for public output.
    mcp_list_tools_items: IndexMap<String, Vec<McpListTools>>,
}

impl ToolRegistry {
    /// Build a registry from declared tools and attach gateway handlers for dispatchable tool types.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] when Codex namespace member flattening
    /// would collide with another declared tool name, when discovered MCP
    /// tools derive the same internal model-visible name, or when an MCP
    /// declaration is itself invalid (for example, a request tries to
    /// override a gateway-configured server's connection). Transient MCP
    /// discovery failures (the server could not be reached) are not
    /// returned as errors; they are recorded as failed [`McpListTools`]
    /// metadata so the rest of the request can proceed. Returns
    /// [`ToolError::Execution`] when discovered MCP metadata exceeds its byte
    /// or tool-count limit.
    pub async fn build_with_handlers(
        tools: &mut [ResponsesTool],
        executors: &mut GatewayExecutors,
    ) -> Result<Self, ToolError> {
        let mut remaining = MAX_MCP_DISCOVERY_BYTES;
        Self::build_with_handlers_guarded(
            tools,
            executors,
            |bytes| {
                remaining = remaining.checked_sub(bytes).ok_or_else(|| {
                    ToolError::Execution(format!(
                        "MCP discovery metadata exceeded {MAX_MCP_DISCOVERY_BYTES} bytes"
                    ))
                })?;
                Ok(())
            },
            || ready(()),
        )
        .await
    }

    /// Builds a registry while charging each guarded MCP discovery to a caller-owned budget.
    ///
    /// The callback runs before discovered metadata is copied into the registry or request, so
    /// a request-wide executor budget can reject aggregate results without retaining the item
    /// that crossed the limit. Keeping the callback generic preserves the dependency direction:
    /// tool registration does not depend on executor error or policy types.
    pub(crate) async fn build_with_handlers_guarded<E, Acquire, AcquireFuture, Guard>(
        tools: &mut [ResponsesTool],
        executors: &mut GatewayExecutors,
        mut consume_materialized: impl FnMut(usize) -> Result<(), E>,
        mut acquire_materialization: Acquire,
    ) -> Result<Self, E>
    where
        E: From<ToolError>,
        Acquire: FnMut() -> AcquireFuture,
        AcquireFuture: Future<Output = Guard>,
    {
        let mut entries = HashMap::with_capacity(tools.len());
        let mut mcp_list_tools_items = IndexMap::<String, Vec<McpListTools>>::new();
        let mut discovered_mcp_tools = 0usize;
        // Namespace members must be keyed by the same flat, model-visible name
        // the model will call, so resolve them first — the same pure pass used
        // to build the upstream request.
        let resolved_tools = CodexNamespaceHandler.resolve_namespace_members(tools)?;
        McpHandler::validate_server_labels(&resolved_tools)?;
        validate_mcp_server_count(&resolved_tools)?;

        for (index, tool) in resolved_tools.iter().enumerate() {
            match tool {
                ResponsesTool::Function(p) => {
                    insert_unique_tool_entries(&mut entries, |resolved| insert_function_entry(resolved, p))?;
                }
                ResponsesTool::ToolSearch(param) => {
                    insert_unique_tool_entries(&mut entries, |resolved| {
                        insert_tool_search_entry(resolved, param);
                    })?;
                }
                ResponsesTool::Mcp(p) => {
                    let _materialization_guard = acquire_materialization().await;
                    let tool_set = match executors.mcp_server_tools(p).await {
                        Ok(tool_set) => tool_set,
                        // Config errors mean the declaration is invalid; the client can fix it.
                        Err(error @ ToolError::Config(_)) => return Err(error.into()),
                        Err(error) => {
                            let list_tools_item = McpHandler::failed_list_tools_item(&p.server_label, &error);
                            consume_serialized(&list_tools_item, &mut consume_materialized)?;
                            mcp_list_tools_items
                                .entry(p.server_label.clone())
                                .or_default()
                                .push(list_tools_item);
                            continue;
                        }
                    };
                    let handlers = tool_set.discovered_handlers;
                    discovered_mcp_tools = discovered_mcp_tools.checked_add(handlers.len()).ok_or_else(|| {
                        E::from(ToolError::Execution(
                            "discovered MCP tool count overflowed the platform limit".to_owned(),
                        ))
                    })?;
                    if discovered_mcp_tools > MAX_DISCOVERED_MCP_TOOLS_PER_REQUEST {
                        return Err(ToolError::Execution(format!(
                            "request discovered {discovered_mcp_tools} MCP tools; at most {MAX_DISCOVERED_MCP_TOOLS_PER_REQUEST} discovered MCP tools are allowed"
                        ))
                        .into());
                    }
                    for handler in &handlers {
                        consume_serialized(&handler.param, &mut consume_materialized)?;
                    }
                    consume_serialized(&tool_set.list_tools_item, &mut consume_materialized)?;
                    mcp_list_tools_items
                        .entry(p.server_label.clone())
                        .or_default()
                        .push(tool_set.list_tools_item);
                    if let ResponsesTool::Mcp(declaration) = &mut tools[index] {
                        declaration.discovered_tools = handlers.iter().map(|item| item.param.clone()).collect();
                    }
                    for discovered in handlers {
                        insert_unique_tool_entries(&mut entries, |resolved| {
                            insert_discovered_mcp_entry(resolved, discovered);
                        })?;
                    }
                }
                ResponsesTool::WebSearch(p) => {
                    insert_unique_tool_entries(&mut entries, |resolved| {
                        insert_web_search_entry(resolved, p, executors.web_search_handler());
                    })?;
                }
                ResponsesTool::FileSearch(p) => {
                    insert_unique_tool_entries(&mut entries, |resolved| insert_file_search_entry(resolved, p))?;
                }
                ResponsesTool::CodeInterpreter(p) => {
                    insert_unique_tool_entries(&mut entries, |resolved| {
                        insert_code_interpreter_entry(resolved, p);
                    })?;
                }
                ResponsesTool::Namespace(p) => {
                    insert_unique_tool_entries(&mut entries, |resolved| insert_namespace_entries(resolved, p))?;
                }
                ResponsesTool::Custom(p) => {
                    insert_unique_tool_entries(&mut entries, |resolved| insert_custom_entry(resolved, p))?;
                }
                ResponsesTool::Unknown => {
                    tracing::debug!("unknown tool declared but skipped in registry");
                }
            }
        }

        let namespace_map = CodexNamespaceHandler.build_namespace_map((!tools.is_empty()).then_some(tools))?;
        let custom_tool_map = CustomHandler::build_tool_map(tools);

        Ok(Self {
            entries,
            tool_search: None,
            namespace_map,
            custom_tool_map,
            mcp_list_tools_items,
        })
    }

    pub(crate) fn install_tool_search_state(&mut self, state: Option<ToolSearchState>) -> Result<(), ToolError> {
        if let Some(state) = state {
            self.validate_tool_search_state(&state)?;
            self.tool_search = Some(Box::new(state));
        }
        Ok(())
    }

    /// Public declarations to expose in response metadata. `Some([])` is
    /// intentionally distinct from an inactive request.
    #[must_use]
    pub(crate) fn tool_search_response_tools(&self) -> Option<Vec<ResponsesTool>> {
        let state = self.tool_search.as_deref().filter(|state| state.is_active())?;
        let mut tools = state.public_response_tools();
        for tool in &mut tools {
            tool.sanitize_for_persistence();
        }
        Some(tools)
    }

    /// Move the public tool projection into response persistence metadata.
    pub(crate) fn take_tool_search_metadata(&mut self) -> Option<ToolSearchMetadata> {
        self.tool_search
            .take()
            .filter(|state| state.is_active())
            .map(|state| (*state).into_public_metadata())
    }

    pub(crate) fn validate_blocking_response(&self, body: &str) -> Result<(), ToolError> {
        let empty = HashSet::new();
        let state = self.tool_search.as_deref();
        validate_blocking_response(
            body,
            state.is_some_and(ToolSearchState::is_active),
            state.map_or(&empty, ToolSearchState::withheld_function_names),
        )
    }

    /// Ensure tool-search requests went through the request-scoped preparation seam.
    pub(crate) fn ensure_request_prepared(&self, request: &RequestPayload) -> Result<(), ToolError> {
        ensure_request_prepared(request, self.tool_search.is_some())
    }

    pub(crate) fn restore_tool_search_response_tools(&self, wire: &mut WireEvent) -> Result<(), ToolError> {
        let Some(response) = wire.rest.get_mut("response").and_then(Value::as_object_mut) else {
            return Ok(());
        };
        if !response.contains_key("tools") {
            return Ok(());
        }
        let Some(tools) = self.tool_search_response_tools() else {
            return Ok(());
        };
        response.insert(
            "tools".to_owned(),
            serialize_to_value(&tools).map_err(|_| super::tool_search::invalid_upstream_search_call())?,
        );
        Ok(())
    }

    #[must_use]
    pub fn lookup(&self, tool_name: &str) -> Option<&ToolEntry> {
        self.entries.get(tool_name)
    }

    pub(crate) fn tool_type(&self, name: &str) -> ToolType {
        if name == TOOL_SEARCH_NAME && self.tool_search.as_deref().is_some_and(ToolSearchState::is_active) {
            return ToolType::ToolSearch;
        }
        self.entries
            .get(name)
            .map_or(ToolType::Function, |entry| entry.tool_type)
    }

    pub(crate) fn is_withheld_function(&self, name: &str) -> bool {
        self.tool_search
            .as_deref()
            .is_some_and(|state| state.withheld_function_names().contains(name))
    }

    pub(crate) fn tool_search_is_active(&self) -> bool {
        self.tool_type(TOOL_SEARCH_NAME) == ToolType::ToolSearch
    }

    #[cfg(test)]
    pub(crate) fn from_tool_types(tool_types: HashMap<String, ToolType>) -> Self {
        let entries = tool_types
            .into_iter()
            .map(|(name, tool_type)| {
                let entry = if tool_type.is_gateway_owned() {
                    ToolEntry::gateway(tool_type, None, None)
                } else {
                    ToolEntry::client(tool_type, None)
                };
                (name, entry)
            })
            .collect();
        Self {
            entries,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn contains_mcp_server_label(&self, server_label: &str) -> bool {
        self.entries
            .values()
            .any(|entry| entry.tool_type == ToolType::Mcp && entry.server_label.as_deref() == Some(server_label))
    }

    pub(crate) fn cache_listed_mcp_tools(&mut self, input: &ResponsesInput) {
        if let ResponsesInput::Items(items) = input {
            for item in items {
                let InputItem::McpListTools(list_tools) = item else {
                    continue;
                };
                // Only a currently declared MCP server can emit discovery in
                // this request. Ignore history for labels absent from the
                // current registry instead of turning it into a new candidate.
                if let Some(items) = self.mcp_list_tools_items.get_mut(&list_tools.server_label) {
                    items.push(list_tools.clone());
                }
            }
        }
    }

    /// Current discovery items without an equivalent successful historical
    /// lifecycle. Failed or changed discovery remains visible to the client.
    pub(crate) fn mcp_list_tool_items(&self) -> impl Iterator<Item = &McpListTools> {
        self.mcp_list_tools_items.values().filter_map(|items| {
            let (current, history) = items.split_first()?;
            let already_emitted = current.error.is_none()
                && history
                    .iter()
                    .any(|previous| previous.error.is_none() && previous.tools == current.tools);
            (!already_emitted).then_some(current)
        })
    }

    /// Marks the current request's discovery lifecycle as consumed.
    pub(crate) fn clear_mcp_list_tool_items(&mut self) {
        self.mcp_list_tools_items.clear();
    }

    /// Validate the private dispatch table against prepared tool-search state.
    fn validate_tool_search_state(&self, state: &ToolSearchState) -> Result<(), ToolError> {
        if !state.is_active() {
            return Ok(());
        }
        if self
            .entries
            .keys()
            .any(|name| state.withheld_function_names().contains(name))
        {
            return Err(ToolError::Config(
                "a loaded tool collides with a withheld function name".to_owned(),
            ));
        }
        let Some(_) = state.synthetic_tool_search() else {
            return Ok(());
        };
        let entry = self.entries.get(TOOL_SEARCH_NAME).ok_or_else(|| {
            ToolError::Config("prepared tool-search declaration is missing from the private registry".to_owned())
        })?;
        if entry.tool_type != ToolType::ToolSearch || !matches!(entry.ownership, ToolOwnership::Client) {
            return Err(ToolError::Config(
                "prepared tool-search declaration has invalid private registry ownership".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn restore_final_payload_output(&self, output: &mut [OutputItem]) {
        CodexNamespaceHandler.restore_output_items(output, self.namespace_map.as_ref());
    }

    pub fn restore_stream_event_wire(&self, wire: &mut WireEvent) -> bool {
        let custom_restored = CustomHandler::restore_response_wire(wire, self.custom_tool_map.as_ref());
        CodexNamespaceHandler.restore_response_wire(wire, self.namespace_map.as_ref()) | custom_restored
    }

    /// Returns the subset of `calls` whose names map to gateway-owned tools.
    #[must_use]
    pub fn gateway_owned<'a>(&self, calls: &'a [FunctionToolCall]) -> Vec<&'a FunctionToolCall> {
        calls
            .iter()
            .filter(|c| self.entries.get(&c.name).is_some_and(|e| e.ownership.is_gateway()))
            .collect()
    }

    #[must_use]
    pub fn is_gateway_owned_name(&self, name: &str) -> bool {
        self.entries.get(name).is_some_and(|entry| entry.ownership.is_gateway())
    }

    #[must_use]
    pub fn is_client_custom_name(&self, name: &str) -> bool {
        self.entries
            .get(name)
            .is_some_and(|entry| entry.tool_type == ToolType::Custom)
    }

    /// Returns the subset of `calls` whose names map to client-owned tools
    /// (`Function`, Codex namespace members, or unknown names).
    #[must_use]
    pub fn client_owned<'a>(&self, calls: &'a [FunctionToolCall]) -> Vec<&'a FunctionToolCall> {
        calls
            .iter()
            .filter(|c| self.entries.get(&c.name).is_none_or(|e| !e.ownership.is_gateway()))
            .collect()
    }

    pub async fn dispatch(&self, call: &FunctionToolCall) -> Option<GatewayDispatchResult> {
        let entry = self.entries.get(&call.name)?;
        let ToolOwnership::Gateway(Some(binding)) = &entry.ownership else {
            return None;
        };
        let tool_type = entry.tool_type;
        Some(GatewayDispatchResult {
            tool_type,
            output: binding.execute(&call.call_id, &call.name, &call.arguments).await,
        })
    }
}

fn validate_mcp_server_count(tools: &[ResponsesTool]) -> Result<(), ToolError> {
    let mcp_server_count = tools
        .iter()
        .filter(|tool| matches!(tool, ResponsesTool::Mcp(_)))
        .count();
    if mcp_server_count > MAX_MCP_SERVERS_PER_REQUEST {
        return Err(ToolError::Config(format!(
            "request declared {mcp_server_count} MCP servers; at most {MAX_MCP_SERVERS_PER_REQUEST} MCP server declarations are allowed"
        )));
    }
    Ok(())
}

fn consume_serialized<T, E>(value: &T, consume_materialized: &mut impl FnMut(usize) -> Result<(), E>) -> Result<(), E>
where
    T: Serialize,
    E: From<ToolError>,
{
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ToolError::Execution(format!("failed to account for MCP discovery metadata: {error}")))?;
    consume_materialized(bytes.len())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::tool::executors::GatewayExecutorRegistration;
    use crate::tool::mcp::{McpDiscoveredHandler, McpHandler};
    use crate::types::event::MessageStatus;
    use crate::types::io::output::McpListTool;
    use crate::types::tools::McpDiscoveredToolParam;

    fn declaration(server_label: &str) -> ResponsesTool {
        serde_json::from_value(serde_json::json!({
            "type": "mcp",
            "server_label": server_label,
            "server_url": "http://127.0.0.1:8000/mcp",
            "require_approval": "never"
        }))
        .expect("MCP declaration")
    }

    /// A request-declared MCP tool for a server the gateway already has
    /// configured. Configured servers reject a request-supplied `server_url`,
    /// so declarations for them must omit it.
    fn configured_declaration(server_label: &str) -> ResponsesTool {
        serde_json::from_value(serde_json::json!({
            "type": "mcp",
            "server_label": server_label
        }))
        .expect("MCP declaration")
    }

    fn discovered_handler(server_label: &str, tool_name: &str, internal_name: &str) -> McpDiscoveredHandler {
        discovered_handler_with_description(server_label, tool_name, internal_name, "Discovered test tool")
    }

    fn discovered_handler_with_description(
        server_label: &str,
        tool_name: &str,
        internal_name: &str,
        description: &str,
    ) -> McpDiscoveredHandler {
        let param = McpDiscoveredToolParam {
            server_label: server_label.to_owned(),
            tool_name: tool_name.to_owned(),
            internal_name: internal_name.to_owned(),
            tool: serde_json::from_value(serde_json::json!({
                "name": tool_name,
                "description": description,
                "inputSchema": {"type": "object"}
            }))
            .expect("discovered MCP tool"),
        };
        McpDiscoveredHandler {
            param,
            handler: Arc::new(McpHandler::discovered_tool_spec_only()),
        }
    }

    fn discovered_handlers(server_label: &str, start: usize, count: usize) -> Vec<McpDiscoveredHandler> {
        (start..start + count)
            .map(|index| {
                discovered_handler(
                    server_label,
                    &format!("tool-{index}"),
                    &format!("mcp__{server_label}__tool_{index}"),
                )
            })
            .collect()
    }

    fn mixed_tool_declarations() -> Vec<ResponsesTool> {
        serde_json::from_value(serde_json::json!([
            {
                "type": "function",
                "name": "echo",
                "parameters": {"type": "object"}
            },
            {
                "type": "mcp",
                "server_label": "counter"
            },
            {"type": "web_search_preview", "search_context_size": "low"},
            {"type": "file_search", "vector_store_ids": ["vs_test"]},
            {"type": "code_interpreter"},
            {
                "type": "namespace",
                "name": "mcp__shell",
                "tools": [{"type": "function", "name": "run"}]
            },
            {"type": "custom", "name": "freeform"},
            {"type": "future_tool", "opaque": true}
        ]))
        .expect("mixed tool declarations")
    }

    fn assert_namespace_call_restoration(registry: &ToolRegistry) {
        let mut output = vec![OutputItem::FunctionCall(FunctionToolCall {
            id: "fc_1".to_owned(),
            call_id: "call_1".to_owned(),
            name: "agentic_ns__mcp__shell__run".to_owned(),
            namespace: None,
            arguments: "{}".to_owned(),
            status: MessageStatus::Completed,
        })];
        registry.restore_final_payload_output(&mut output);
        let OutputItem::FunctionCall(call) = &output[0] else {
            panic!("expected restored function call");
        };
        assert_eq!(call.namespace.as_deref(), Some("mcp__shell"));
        assert_eq!(call.name, "run");
    }

    fn assert_mcp_list_tools_metadata(registry: &ToolRegistry) {
        let [list_tools] = registry.mcp_list_tools_items["counter"].as_slice() else {
            panic!("expected one MCP list-tools item");
        };
        assert!(list_tools.id.starts_with("mcpl_"));
        assert_eq!(list_tools.server_label, "counter");
        assert_eq!(
            list_tools
                .tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            ["increment", "get_value"]
        );
        assert_eq!(list_tools.tools[0].description.as_deref(), Some("Discovered test tool"));
        assert_eq!(list_tools.tools[0].input_schema, serde_json::json!({"type": "object"}));
        assert_eq!(
            list_tools.tools[0].annotations,
            Some(serde_json::json!({"read_only": false}))
        );
    }

    #[test]
    fn ignores_mcp_list_history_for_servers_absent_from_current_registry() {
        let mut registry = ToolRegistry::default();
        let input = ResponsesInput::Items(vec![
            InputItem::McpListTools(McpListTools::new("mcpl_1", "counter", Vec::new())),
            InputItem::McpListTools(McpListTools::new("mcpl_2", "search", Vec::new())),
        ]);

        registry.cache_listed_mcp_tools(&input);

        assert_eq!(registry.mcp_list_tool_items().count(), 0);
        assert!(registry.mcp_list_tools_items.is_empty());
    }

    #[test]
    fn mcp_list_tools_map_suppresses_equivalent_successful_history_and_clears_after_emission() {
        let mut registry = ToolRegistry::default();
        registry.mcp_list_tools_items.insert(
            "counter".to_owned(),
            vec![
                McpListTools::new("mcpl_current", "counter", Vec::new()),
                McpListTools::new("mcpl_prior", "counter", Vec::new()),
            ],
        );
        registry.mcp_list_tools_items.insert(
            "search".to_owned(),
            vec![McpListTools::new("mcpl_search_current", "search", Vec::new())],
        );

        let current = registry.mcp_list_tool_items().collect::<Vec<_>>();

        assert_eq!(
            current.iter().map(|item| item.id.as_str()).collect::<Vec<_>>(),
            ["mcpl_search_current"]
        );
        assert_eq!(registry.mcp_list_tools_items["counter"].len(), 2);

        registry.clear_mcp_list_tool_items();

        assert_eq!(registry.mcp_list_tool_items().count(), 0);
        assert!(registry.mcp_list_tools_items.is_empty());
    }

    fn listed_tool(name: &str) -> McpListTool {
        McpListTool::new(
            name,
            Some(format!("{name} description")),
            serde_json::json!({"type": "object"}),
            Some(serde_json::json!({"read_only": true})),
        )
    }

    fn visible_discovery_ids(current: McpListTools, history: Vec<McpListTools>) -> Vec<String> {
        let mut items = vec![current];
        items.extend(history);
        let mut registry = ToolRegistry::default();
        registry.mcp_list_tools_items.insert("server".to_owned(), items);
        registry.mcp_list_tool_items().map(|item| item.id.clone()).collect()
    }

    #[test]
    fn mcp_list_tools_map_keeps_failed_or_changed_current_discovery_visible() {
        let current_success = McpListTools::new("mcpl_current", "server", vec![listed_tool("current")]);
        let mut previous_failure = McpListTools::new("mcpl_failed", "server", vec![listed_tool("current")]);
        previous_failure.error = Some("discovery failed".to_owned());
        assert_eq!(
            visible_discovery_ids(current_success, vec![previous_failure]),
            ["mcpl_current"]
        );

        let current_changed = McpListTools::new("mcpl_changed", "server", vec![listed_tool("new")]);
        let previous_success = McpListTools::new("mcpl_previous", "server", vec![listed_tool("old")]);
        assert_eq!(
            visible_discovery_ids(current_changed, vec![previous_success]),
            ["mcpl_changed"]
        );

        let mut current_failure = McpListTools::new("mcpl_current_failed", "server", Vec::new());
        current_failure.error = Some("current discovery failed".to_owned());
        let previous_empty_success = McpListTools::new("mcpl_previous_success", "server", Vec::new());
        assert_eq!(
            visible_discovery_ids(current_failure, vec![previous_empty_success]),
            ["mcpl_current_failed"]
        );
    }

    #[test]
    fn mcp_list_tool_items_preserve_server_declaration_order() {
        let mut registry = ToolRegistry::default();
        registry.mcp_list_tools_items.insert(
            "second-alphabetically".to_owned(),
            vec![McpListTools::new("mcpl_first", "second-alphabetically", Vec::new())],
        );
        registry.mcp_list_tools_items.insert(
            "first-alphabetically".to_owned(),
            vec![McpListTools::new("mcpl_second", "first-alphabetically", Vec::new())],
        );

        assert_eq!(
            registry
                .mcp_list_tool_items()
                .map(|item| item.server_label.as_str())
                .collect::<Vec<_>>(),
            ["second-alphabetically", "first-alphabetically"]
        );
    }

    #[tokio::test]
    async fn build_with_handlers_registers_mixed_tools_and_runtime_metadata() {
        let mut executors = GatewayExecutors::from_env(Arc::new(reqwest::Client::new()));
        executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "counter".to_owned(),
            handlers: vec![
                discovered_handler("counter", "increment", "mcp__counter__increment"),
                discovered_handler("counter", "get_value", "mcp__counter__get_value"),
            ],
        });
        let mut tools = mixed_tool_declarations();

        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("mixed registry");

        assert_eq!(registry.len(), 8);
        assert!(registry.contains_mcp_server_label("counter"));
        assert!(!registry.contains_mcp_server_label("missing"));
        assert_mcp_list_tools_metadata(&registry);

        let expected_entries = [
            ("echo", ToolType::Function, None, false),
            ("freeform", ToolType::Custom, None, false),
            ("mcp__counter__increment", ToolType::Mcp, Some("counter"), true),
            ("mcp__counter__get_value", ToolType::Mcp, Some("counter"), true),
            ("web_search", ToolType::WebSearch, None, true),
            ("file_search", ToolType::FileSearch, None, false),
            ("code_interpreter", ToolType::CodeInterpreter, None, false),
            (
                "agentic_ns__mcp__shell__run",
                ToolType::CodexNamespace,
                Some("mcp__shell"),
                false,
            ),
        ];
        for (name, tool_type, server_label, has_handler) in expected_entries {
            let entry = registry
                .lookup(name)
                .unwrap_or_else(|| panic!("missing registry entry '{name}'"));
            assert_eq!(entry.tool_type, tool_type, "unexpected type for '{name}'");
            assert_eq!(
                entry.server_label.as_deref(),
                server_label,
                "unexpected server label for '{name}'"
            );
            assert_eq!(
                matches!(entry.ownership, ToolOwnership::Gateway(Some(_))),
                has_handler,
                "unexpected handler for '{name}'"
            );
        }
        for name in [
            "mcp__counter__increment",
            "mcp__counter__get_value",
            "web_search",
            "file_search",
            "code_interpreter",
        ] {
            assert!(registry.is_gateway_owned_name(name), "'{name}' should be gateway-owned");
        }
        for name in ["echo", "freeform", "agentic_ns__mcp__shell__run"] {
            assert!(!registry.is_gateway_owned_name(name), "'{name}' should be client-owned");
        }

        let ResponsesTool::Mcp(declared) = &tools[1] else {
            panic!("expected MCP declaration");
        };
        assert_eq!(declared.discovered_tools.len(), 2);
        assert_eq!(
            tools[1]
                .to_function_tools()
                .into_iter()
                .map(|tool| tool.name)
                .collect::<Vec<_>>(),
            ["mcp__counter__increment", "mcp__counter__get_value"]
        );

        let ResponsesTool::Namespace(namespace) = &tools[5] else {
            panic!("expected namespace declaration");
        };
        assert!(matches!(
            namespace.tools.as_slice(),
            [crate::types::tools::CodexNamespaceMember::Function(function)] if function.name.as_str() == "run"
        ));
        assert_namespace_call_restoration(&registry);
    }

    #[tokio::test]
    async fn build_with_handlers_retains_mcp_discovery_failure_output() {
        let mut tools = vec![declaration("unreachable")];
        let mut executors = GatewayExecutors::default();

        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("discovery failures should become response metadata");

        let [list_tools] = registry.mcp_list_tools_items["unreachable"].as_slice() else {
            panic!("expected one MCP list-tools item");
        };
        assert_eq!(list_tools.server_label, "unreachable");
        assert!(list_tools.tools.is_empty());
        assert!(
            list_tools
                .error
                .as_deref()
                .is_some_and(|error| error.contains("failed"))
        );
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn build_with_handlers_guarded_rejects_aggregate_mcp_discovery_bytes() {
        let mut executors = GatewayExecutors::default();
        let description = "x".repeat(600);
        for server_label in ["first", "second"] {
            executors.insert(GatewayExecutorRegistration::Mcp {
                server_label: server_label.to_owned(),
                handlers: vec![discovered_handler_with_description(
                    server_label,
                    "tool",
                    &format!("mcp__{server_label}__tool"),
                    &description,
                )],
            });
        }
        let mut tools = vec![configured_declaration("first"), configured_declaration("second")];
        let mut consumed = 0usize;

        let error = ToolRegistry::build_with_handlers_guarded(
            &mut tools,
            &mut executors,
            |bytes| {
                consumed = consumed.saturating_add(bytes);
                if consumed > 2_048 {
                    return Err(ToolError::Execution("test MCP discovery budget exceeded".to_owned()));
                }
                Ok(())
            },
            || std::future::ready(()),
        )
        .await
        .expect_err("aggregate MCP discovery must use the caller's shared budget");

        assert!(matches!(error, ToolError::Execution(message) if message.contains("budget exceeded")));
    }

    #[tokio::test]
    async fn build_with_handlers_applies_a_default_mcp_discovery_budget() {
        let mut executors = GatewayExecutors::default();
        let description = "x".repeat(MAX_MCP_DISCOVERY_BYTES / 4);
        for server_label in ["first", "second"] {
            executors.insert(GatewayExecutorRegistration::Mcp {
                server_label: server_label.to_owned(),
                handlers: vec![discovered_handler_with_description(
                    server_label,
                    "tool",
                    &format!("mcp__{server_label}__tool"),
                    &description,
                )],
            });
        }
        let mut tools = vec![configured_declaration("first"), configured_declaration("second")];

        let error = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect_err("the standalone registry builder must enforce its own discovery budget");

        assert!(matches!(
            error,
            ToolError::Execution(message)
                if message.contains("MCP discovery metadata exceeded")
                    && message.contains(&MAX_MCP_DISCOVERY_BYTES.to_string())
        ));
    }

    #[tokio::test]
    async fn build_with_handlers_rejects_too_many_mcp_declarations_before_discovery() {
        let mut accepted_executors = GatewayExecutors::default();
        let mut exact_limit = Vec::new();
        for index in 0..MAX_MCP_SERVERS_PER_REQUEST {
            let server_label = format!("accepted-server-{index}");
            accepted_executors.insert(GatewayExecutorRegistration::Mcp {
                server_label: server_label.clone(),
                handlers: vec![discovered_handler(
                    &server_label,
                    "tool",
                    &format!("mcp__accepted_server_{index}__tool"),
                )],
            });
            exact_limit.push(configured_declaration(&server_label));
        }
        ToolRegistry::build_with_handlers(&mut exact_limit, &mut accepted_executors)
            .await
            .expect("the documented MCP declaration limit must be accepted");

        let mut tools = (0..=MAX_MCP_SERVERS_PER_REQUEST)
            .map(|index| configured_declaration(&format!("server-{index}")))
            .collect::<Vec<_>>();
        let mut executors = GatewayExecutors::default();

        let error = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect_err("too many MCP declarations must fail before discovery");

        assert!(matches!(
            error,
            ToolError::Config(message)
                if message.contains("MCP server declarations")
                    && message.contains(&MAX_MCP_SERVERS_PER_REQUEST.to_string())
        ));
    }

    #[tokio::test]
    async fn build_with_handlers_rejects_too_many_discovered_mcp_tools() {
        let first_count = MAX_DISCOVERED_MCP_TOOLS_PER_REQUEST / 2;
        let second_count = MAX_DISCOVERED_MCP_TOOLS_PER_REQUEST - first_count;
        let mut accepted_executors = GatewayExecutors::default();
        accepted_executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "accepted-first".to_owned(),
            handlers: discovered_handlers("accepted-first", 0, first_count),
        });
        accepted_executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "accepted-second".to_owned(),
            handlers: discovered_handlers("accepted-second", first_count, second_count),
        });
        let mut accepted_tools = vec![
            configured_declaration("accepted-first"),
            configured_declaration("accepted-second"),
        ];
        ToolRegistry::build_with_handlers(&mut accepted_tools, &mut accepted_executors)
            .await
            .expect("the documented discovered MCP tool limit must be accepted across servers");

        let mut executors = GatewayExecutors::default();
        executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "first".to_owned(),
            handlers: discovered_handlers("first", 0, first_count),
        });
        executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "second".to_owned(),
            handlers: discovered_handlers("second", first_count, second_count + 1),
        });
        let mut tools = vec![configured_declaration("first"), configured_declaration("second")];

        let error = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect_err("too many discovered MCP tools must fail");

        assert!(matches!(
            error,
            ToolError::Execution(message)
                if message.contains("discovered MCP tools")
                    && message.contains(&MAX_DISCOVERED_MCP_TOOLS_PER_REQUEST.to_string())
        ));
    }

    #[tokio::test]
    async fn duplicate_mcp_server_labels_are_rejected() {
        let mut tools = vec![declaration("counter"), declaration("counter")];
        let mut executors = GatewayExecutors::default();

        let error = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect_err("duplicate server_label must fail");

        assert!(
            matches!(error, ToolError::Config(message) if message.contains("duplicate MCP declarations") && message.contains("counter"))
        );
    }

    #[tokio::test]
    async fn cross_server_internal_name_collisions_are_rejected() {
        let internal_name = "mcp__foo__bar__baz";
        let mut executors = GatewayExecutors::default();
        executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "foo".to_owned(),
            handlers: vec![discovered_handler("foo", "bar__baz", internal_name)],
        });
        executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "foo__bar".to_owned(),
            handlers: vec![discovered_handler("foo__bar", "baz", internal_name)],
        });
        let mut tools = vec![configured_declaration("foo"), configured_declaration("foo__bar")];

        let error = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect_err("colliding derived MCP names must fail");

        assert!(matches!(
            error,
            ToolError::Config(message)
                if message.contains(internal_name) && message.matches("MCP tool").count() == 2
        ));
    }

    #[tokio::test]
    async fn discovered_mcp_name_collision_with_function_is_rejected_in_any_order() {
        let internal_name = "mcp__counter__increment";

        for mcp_first in [false, true] {
            let function = serde_json::from_value(serde_json::json!({
                "type": "function",
                "name": internal_name
            }))
            .expect("function declaration");
            let mcp = configured_declaration("counter");
            let mut tools = if mcp_first {
                vec![mcp, function]
            } else {
                vec![function, mcp]
            };
            let mut executors = GatewayExecutors::default();
            executors.insert(GatewayExecutorRegistration::Mcp {
                server_label: "counter".to_owned(),
                handlers: vec![discovered_handler("counter", "increment", internal_name)],
            });

            let error = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
                .await
                .expect_err("MCP internal name must not overwrite a function");

            assert!(matches!(
                error,
                ToolError::Config(message)
                    if message.contains(internal_name)
                        && message.contains("MCP tool")
                        && message.contains("function tool")
            ));
        }
    }
}
