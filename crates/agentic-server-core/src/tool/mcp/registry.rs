use std::collections::HashMap;

use crate::tool::{GatewayBinding, ToolEntry, ToolType};

use super::McpDiscoveredHandler;

/// Registers one tool returned by MCP `tools/list`, keyed by its internal
/// model-visible name while binding its typed discovered-tool parameters to
/// the matching MCP executor.
pub(crate) fn insert_discovered_mcp_entry(entries: &mut HashMap<String, ToolEntry>, discovered: McpDiscoveredHandler) {
    let McpDiscoveredHandler { param, handler } = discovered;
    let server_label = param.server_label.clone();
    let internal_name = param.internal_name.clone();
    entries.insert(
        internal_name,
        ToolEntry::gateway(
            ToolType::Mcp,
            Some(server_label),
            Some(GatewayBinding::new(handler, param)),
        ),
    );
}
