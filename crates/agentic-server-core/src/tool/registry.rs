use std::collections::HashMap;
use std::collections::hash_map::Entry;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::codex::insert_namespace_entries;
use super::custom::{CustomHandler, CustomToolMap, insert_custom_entry};
use super::executors::GatewayExecutors;
use super::function::insert_function_entry;
use super::mcp::registry::insert_discovered_mcp_entry;
use super::ownership::{GatewayBinding, ToolOwnership};
use super::web_search::insert_web_search_entry;
use super::{CodexNamespaceHandler, McpHandler, NamespaceMap, ToolError, ToolOutput};
use crate::events::WireEvent;

use crate::types::io::output::{FunctionToolCall, McpListTools};
use crate::types::io::{InputItem, OutputItem, ResponsesInput};
use crate::types::tools::{CodeInterpreterToolParam, FileSearchToolParam, ResponsesTool};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
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
        !matches!(self, Self::Function | Self::Custom | Self::CodexNamespace)
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
    /// metadata so the rest of the request can proceed.
    ///
    /// # Panics
    ///
    /// Panics if serialization of a tool param struct fails, which cannot happen
    /// for the types defined in this module (`#[derive(Serialize)]` on plain structs).
    pub async fn build_with_handlers(
        tools: &mut [ResponsesTool],
        executors: &mut GatewayExecutors,
    ) -> Result<Self, ToolError> {
        let mut entries = HashMap::with_capacity(tools.len());
        let mut mcp_list_tools_items = IndexMap::<String, Vec<McpListTools>>::new();
        // Namespace members must be keyed by the same flat, model-visible name
        // the model will call, so resolve them first — the same pure pass used
        // to build the upstream request.
        let resolved_tools = CodexNamespaceHandler.resolve_namespace_members(tools)?;
        McpHandler::validate_server_labels(&resolved_tools)?;

        for (index, tool) in resolved_tools.iter().enumerate() {
            match tool {
                ResponsesTool::Function(p) => {
                    insert_unique_tool_entries(&mut entries, |resolved| insert_function_entry(resolved, p))?;
                }
                ResponsesTool::Mcp(p) => {
                    let tool_set = match executors.mcp_server_tools(p).await {
                        Ok(tool_set) => tool_set,
                        // Config errors mean the declaration is invalid; the client can fix it.
                        Err(error @ ToolError::Config(_)) => return Err(error),
                        Err(error) => {
                            mcp_list_tools_items
                                .entry(p.server_label.clone())
                                .or_default()
                                .push(McpHandler::failed_list_tools_item(&p.server_label, &error));
                            continue;
                        }
                    };
                    let handlers = tool_set.discovered_handlers;
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
                ResponsesTool::Shell(_) => {
                    tracing::debug!("shell tool declared but skipped until a handler is registered");
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
            namespace_map,
            custom_tool_map,
            mcp_list_tools_items,
        })
    }

    #[must_use]
    pub fn lookup(&self, tool_name: &str) -> Option<&ToolEntry> {
        self.entries.get(tool_name)
    }

    pub(crate) fn tool_type_map(&self) -> HashMap<String, ToolType> {
        self.entries
            .iter()
            .map(|(name, entry)| (name.clone(), entry.tool_type))
            .collect()
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
        let param = McpDiscoveredToolParam {
            server_label: server_label.to_owned(),
            tool_name: tool_name.to_owned(),
            internal_name: internal_name.to_owned(),
            tool: serde_json::from_value(serde_json::json!({
                "name": tool_name,
                "description": "Discovered test tool",
                "inputSchema": {"type": "object"}
            }))
            .expect("discovered MCP tool"),
        };
        McpDiscoveredHandler {
            param,
            handler: Arc::new(McpHandler::discovered_tool_spec_only()),
        }
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
