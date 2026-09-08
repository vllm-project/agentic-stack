use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use super::mcp::handler::McpServerToolSet;
use super::mcp::{McpClientPool, McpDiscoveredHandler, McpHandler};
use super::web_search::{WebSearchExecutor, WebSearchHandler};
use super::{GatewayExecutor, ToolError};
use crate::config::ToolRuntimeConfig;
use crate::types::tools::McpToolParam;

pub enum GatewayExecutorRegistration {
    /// Explicitly grant shell execution to an application-provided sandbox.
    Shell(Arc<dyn super::shell::ShellExecutor>),
    WebSearch(Arc<WebSearchExecutor>),
    Mcp {
        server_label: String,
        handlers: Vec<McpDiscoveredHandler>,
    },
}

impl<T> From<Arc<T>> for GatewayExecutorRegistration
where
    T: GatewayExecutor<
            ToolParams = crate::types::tools::WebSearchToolParam,
            ExecutionParams = crate::types::tools::WebSearchToolParam,
        >,
{
    fn from(executor: Arc<T>) -> Self {
        Self::WebSearch(executor)
    }
}

impl From<Arc<WebSearchExecutor>> for GatewayExecutorRegistration {
    fn from(executor: Arc<WebSearchExecutor>) -> Self {
        Self::WebSearch(executor)
    }
}

/// Shared, per-server registry of gateway-owned tool executors.
///
/// Built once at startup ([`GatewayExecutors::from_env`]) and reused across
/// every request. Configured and request-declared MCP servers are discovered
/// lazily unless handlers were registered with [`Self::insert`].
#[derive(Clone, Default)]
pub struct GatewayExecutors {
    mcp: HashMap<String, Vec<McpDiscoveredHandler>>,
    mcp_configs: HashMap<String, super::mcp::McpServerEntry>,
    mcp_clients: Arc<RwLock<HashMap<String, Arc<super::mcp::McpClient>>>>,
    mcp_discovered: Arc<RwLock<HashMap<String, Vec<McpDiscoveredHandler>>>>,
    mcp_allowed_hosts: Vec<String>,
    web_search: Option<Arc<WebSearchExecutor>>,
    shell: Option<Arc<dyn super::shell::ShellExecutor>>,
}

impl GatewayExecutors {
    #[must_use]
    pub fn from_env(client: Arc<reqwest::Client>) -> Self {
        Self {
            shell: None,
            mcp: HashMap::new(),
            mcp_configs: HashMap::new(),
            mcp_clients: Arc::new(RwLock::new(HashMap::new())),
            mcp_discovered: Arc::new(RwLock::new(HashMap::new())),
            mcp_allowed_hosts: super::mcp::pool::allowed_hosts_from_env(),
            web_search: Some(Arc::new(WebSearchHandler::from_env(client))),
        }
    }

    /// Builds the shared executors without contacting configured MCP servers.
    ///
    /// Configured `allowed_tools` are applied during discovery, so the stored
    /// handler set is the maximum set that a request may use.
    ///
    /// # Errors
    ///
    /// Invalid policy configuration is returned as an error. Connection and
    /// discovery happen when a configured server is requested.
    pub fn from_config(client: Arc<reqwest::Client>, config: &ToolRuntimeConfig) -> Result<Self, ToolError> {
        let executors = Self {
            shell: None,
            mcp: HashMap::new(),
            mcp_configs: config.mcp_servers.clone(),
            mcp_clients: Arc::new(RwLock::new(HashMap::new())),
            mcp_discovered: Arc::new(RwLock::new(HashMap::new())),
            mcp_allowed_hosts: if config.mcp_allowed_hosts.is_empty() {
                super::mcp::pool::allowed_hosts_from_env()
            } else {
                config.mcp_allowed_hosts.clone()
            },
            web_search: Some(Arc::new(WebSearchHandler::from_values(
                client,
                config.web_search.api_key.clone(),
                config.web_search.base_url.clone(),
                config.max_concurrent_gateway_calls,
            ))),
        };
        if config.mcp_servers.is_empty() {
            return Ok(executors);
        }

        for (server_label, entry) in &config.mcp_servers {
            if entry.require_approval() != Some("never") {
                return Err(ToolError::Config(format!(
                    "configured MCP server '{server_label}' must set require_approval to 'never'"
                )));
            }
        }

        Ok(executors)
    }

    pub fn insert(&mut self, registration: impl Into<GatewayExecutorRegistration>) {
        match registration.into() {
            GatewayExecutorRegistration::Shell(executor) => self.shell = Some(executor),
            GatewayExecutorRegistration::WebSearch(executor) => self.web_search = Some(executor),
            GatewayExecutorRegistration::Mcp { server_label, handlers } => {
                if handlers.is_empty() {
                    tracing::debug!(server_label, "empty MCP discovered handler registration skipped");
                    return;
                }
                if self.mcp.insert(server_label.clone(), handlers).is_some() {
                    tracing::debug!(server_label, "replaced MCP discovered handler registration");
                }
            }
        }
    }

    /// Always returns a real handler — falls back to [`WebSearchHandler::spec_only`]
    /// when no provider was configured, so callers never need to handle a
    /// missing gateway-owned `web_search` handler themselves.
    #[must_use]
    pub fn web_search_handler(&self) -> Arc<WebSearchExecutor> {
        self.web_search
            .clone()
            .unwrap_or_else(|| Arc::new(WebSearchHandler::spec_only()))
    }

    #[must_use]
    pub(crate) fn shell_executor(&self) -> Option<Arc<dyn super::shell::ShellExecutor>> {
        self.shell.clone()
    }

    #[must_use]
    pub(crate) fn request_scoped(&self) -> Self {
        self.clone()
    }

    /// Returns the discovered handlers for one request-declared MCP server.
    ///
    /// # Errors
    ///
    /// Returns a configuration error for an invalid declaration or an empty
    /// allowed tool set, and an execution error when the server cannot connect.
    pub async fn mcp_handler(&mut self, param: &McpToolParam) -> Result<Vec<McpDiscoveredHandler>, ToolError> {
        Ok(self.mcp_server_tools(param).await?.discovered_handlers)
    }

    /// Returns the request-scoped tools and public discovery item for one MCP server.
    ///
    /// # Errors
    ///
    /// Returns a configuration error for an invalid declaration or an empty
    /// allowed tool set, and an execution error when the server cannot connect.
    pub(crate) async fn mcp_server_tools(&mut self, param: &McpToolParam) -> Result<McpServerToolSet, ToolError> {
        let server_label = param.server_label.trim();
        if server_label.is_empty() {
            return Err(ToolError::Config(
                "MCP declaration requires a non-empty server_label".to_owned(),
            ));
        }
        let configured_handlers = self.mcp.get(server_label);
        let configured_server = self.mcp_configs.contains_key(server_label);
        validate_mcp_execution_options(param, configured_server || configured_handlers.is_some())?;
        if (configured_server || configured_handlers.is_some()) && param.server_url.is_some() {
            return Err(ToolError::Config(format!(
                "MCP server '{server_label}' is configured by the gateway; omit server_url from the request"
            )));
        }
        if let Some(configured_handlers) = configured_handlers {
            let discovered_handlers = require_non_empty_mcp_handlers(
                server_label,
                filter_allowed_mcp_handlers(configured_handlers, param.allowed_tools.as_deref()),
            )?;
            return Ok(McpHandler::server_tool_set_from_handlers(
                server_label,
                discovered_handlers,
            ));
        }

        if configured_server {
            let Some(entry) = self.mcp_configs.get(server_label).cloned() else {
                return Err(ToolError::Config(format!(
                    "configured MCP server '{server_label}' is missing"
                )));
            };
            let cached_client = self.mcp_clients.read().await.get(server_label).cloned();
            let client = if let Some(client) = cached_client {
                client
            } else {
                let mut servers = HashMap::new();
                servers.insert(server_label.to_owned(), entry.clone());
                let pool = McpClientPool::from_config(servers).await;
                let Some(client) = pool.get(server_label).cloned() else {
                    return Err(ToolError::Execution(format!(
                        "configured MCP server '{server_label}' failed to connect: {}",
                        pool.connection_error(server_label)
                            .unwrap_or("unknown connection error")
                    )));
                };
                self.mcp_clients
                    .write()
                    .await
                    .insert(server_label.to_owned(), Arc::clone(&client));
                client
            };
            let discovered_handlers = if let Some(discovered_handlers) =
                self.mcp_discovered.read().await.get(server_label).cloned()
            {
                discovered_handlers
            } else {
                let tool_set = McpHandler::discover_tools(server_label, client, entry.allowed_tools()).await?;
                let discovered_handlers = require_non_empty_mcp_handlers(server_label, tool_set.discovered_handlers)?;
                self.mcp_discovered
                    .write()
                    .await
                    .insert(server_label.to_owned(), discovered_handlers.clone());
                discovered_handlers
            };
            let discovered_handlers = require_non_empty_mcp_handlers(
                server_label,
                filter_allowed_mcp_handlers(&discovered_handlers, param.allowed_tools.as_deref()),
            )?;
            return Ok(McpHandler::server_tool_set_from_handlers(
                server_label,
                discovered_handlers,
            ));
        }

        let pool =
            McpClientPool::from_params_with_allowed_hosts(std::slice::from_ref(param), &self.mcp_allowed_hosts).await;
        let Some(client) = pool.get(server_label).cloned() else {
            return Err(pool.connection_error(server_label).map_or_else(
                || {
                    ToolError::Config(format!(
                        "MCP server '{server_label}' has no valid request-declared configuration"
                    ))
                },
                |error| ToolError::Execution(format!("MCP server '{server_label}' failed to connect: {error}")),
            ));
        };
        let tool_set = McpHandler::discover_tools(server_label, client, param.allowed_tools.as_deref()).await?;
        let discovered_handlers = require_non_empty_mcp_handlers(server_label, tool_set.discovered_handlers)?;
        self.mcp.insert(server_label.to_owned(), discovered_handlers.clone());
        Ok(McpHandler::server_tool_set_from_handlers(
            server_label,
            discovered_handlers,
        ))
    }
}

fn filter_allowed_mcp_handlers(
    handlers: &[McpDiscoveredHandler],
    allowed_tools: Option<&[String]>,
) -> Vec<McpDiscoveredHandler> {
    handlers
        .iter()
        .filter(|handler| {
            allowed_tools.is_none_or(|allowed| allowed.iter().any(|name| name == &handler.param.tool_name))
        })
        .cloned()
        .collect()
}

fn require_non_empty_mcp_handlers(
    server_label: &str,
    handlers: Vec<McpDiscoveredHandler>,
) -> Result<Vec<McpDiscoveredHandler>, ToolError> {
    if handlers.is_empty() {
        return Err(ToolError::Config(format!(
            "MCP server '{server_label}' has an empty final allowed tool set"
        )));
    }
    Ok(handlers)
}

fn validate_mcp_execution_options(param: &McpToolParam, configured_server: bool) -> Result<(), ToolError> {
    if param.connector_id.is_some() {
        return Err(ToolError::Config(
            "MCP connector_id is not supported; configure server_url instead".to_owned(),
        ));
    }
    if param
        .require_approval
        .as_deref()
        .is_some_and(|policy| policy != "never")
    {
        return Err(ToolError::Config(
            "MCP require_approval supports only 'never'; approval gating is not yet supported".to_owned(),
        ));
    }
    if !configured_server && param.require_approval.is_none() {
        return Err(ToolError::Config(
            "MCP require_approval must be set to 'never' in gateway configuration or the request".to_owned(),
        ));
    }
    Ok(())
}

impl std::fmt::Debug for GatewayExecutors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatewayExecutors")
            .field("shell", &self.shell.is_some())
            .field("mcp_server_handlers", &self.mcp.len())
            .field("mcp_server_configs", &self.mcp_configs.len())
            .field("mcp_clients", &Arc::strong_count(&self.mcp_clients))
            .field("mcp_discovered", &Arc::strong_count(&self.mcp_discovered))
            .field("mcp_allowed_hosts", &self.mcp_allowed_hosts)
            .field("web_search", &self.web_search.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::{GatewayExecutorRegistration, GatewayExecutors, validate_mcp_execution_options};
    use crate::config::ToolRuntimeConfig;
    use crate::tool::mcp::McpServerEntry;
    use crate::tool::mcp::{McpDiscoveredHandler, McpHandler};
    use crate::types::tools::{McpDiscoveredToolParam, McpToolParam};

    fn mcp_param(value: serde_json::Value) -> McpToolParam {
        serde_json::from_value(value).unwrap()
    }

    fn discovered_handler(tool_name: &str) -> McpDiscoveredHandler {
        McpDiscoveredHandler {
            param: McpDiscoveredToolParam {
                server_label: "counter".to_owned(),
                tool_name: tool_name.to_owned(),
                internal_name: format!("mcp__counter__{tool_name}"),
                tool: serde_json::from_value(serde_json::json!({
                    "name": tool_name,
                    "inputSchema": {"type": "object"}
                }))
                .unwrap(),
            },
            handler: Arc::new(McpHandler::discovered_tool_spec_only()),
        }
    }

    #[test]
    fn mcp_execution_allows_explicit_never_approval_policy() {
        let param = mcp_param(serde_json::json!({
            "server_label": "counter",
            "server_url": "http://localhost:8000/mcp",
            "require_approval": "never"
        }));

        validate_mcp_execution_options(&param, false).unwrap();
    }

    #[test]
    fn mcp_execution_uses_configured_never_approval_policy() {
        let param = mcp_param(serde_json::json!({
            "server_label": "counter"
        }));

        validate_mcp_execution_options(&param, true).unwrap();
    }

    #[test]
    fn mcp_execution_rejects_omitted_approval_policy() {
        let param = mcp_param(serde_json::json!({
            "server_label": "counter",
            "server_url": "http://localhost:8000/mcp"
        }));

        let error = validate_mcp_execution_options(&param, false).unwrap_err();
        assert!(error.to_string().contains("gateway configuration or the request"));
    }

    #[test]
    fn mcp_execution_rejects_unsupported_approval_policy() {
        let param = mcp_param(serde_json::json!({
            "server_label": "counter",
            "server_url": "http://localhost:8000/mcp",
            "require_approval": "always"
        }));

        let error = validate_mcp_execution_options(&param, false).unwrap_err();
        assert!(error.to_string().contains("approval gating is not yet supported"));
    }

    #[test]
    fn mcp_execution_rejects_connector_id() {
        let param = mcp_param(serde_json::json!({
            "server_label": "counter",
            "connector_id": "connector_dropbox"
        }));

        let error = validate_mcp_execution_options(&param, false).unwrap_err();
        assert!(error.to_string().contains("connector_id is not supported"));
    }

    #[tokio::test]
    async fn configured_mcp_server_rejects_request_connection_override() {
        let mut executors = GatewayExecutors::default();
        executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "counter".to_owned(),
            handlers: vec![discovered_handler("read")],
        });
        let param = mcp_param(serde_json::json!({
            "server_label": "counter",
            "server_url": "http://localhost:8000/mcp",
            "require_approval": "never"
        }));

        let Err(error) = executors.mcp_server_tools(&param).await else {
            panic!("request connection override must fail");
        };
        assert!(error.to_string().contains("configured by the gateway"));
        assert!(error.to_string().contains("omit server_url"));
    }

    #[tokio::test]
    async fn unavailable_configured_mcp_server_does_not_block_startup() {
        let mut servers = HashMap::new();
        servers.insert(
            "unavailable".to_owned(),
            McpServerEntry::Http {
                url: "http://127.0.0.1:1/mcp".to_owned(),
                headers: None,
                allowed_tools: Some(vec!["read".to_owned()]),
                require_approval: Some("never".to_owned()),
            },
        );
        let config = ToolRuntimeConfig {
            mcp_servers: servers,
            ..ToolRuntimeConfig::default()
        };

        let executors = GatewayExecutors::from_config(Arc::new(reqwest::Client::new()), &config);

        assert!(executors.is_ok());
    }

    #[tokio::test]
    async fn configured_allowed_tools_cannot_be_expanded_by_request() {
        let mut executors = GatewayExecutors::default();
        executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "counter".to_owned(),
            handlers: vec![discovered_handler("read")],
        });
        let param = mcp_param(serde_json::json!({
            "server_label": "counter",
            "allowed_tools": ["read", "delete"]
        }));

        let tools = executors.mcp_server_tools(&param).await.unwrap();

        assert_eq!(tools.discovered_handlers.len(), 1);
        assert_eq!(tools.discovered_handlers[0].param.tool_name, "read");
    }

    #[tokio::test]
    async fn cached_mcp_server_tools_apply_request_allowed_tools_with_fresh_output_id() {
        let mut executors = GatewayExecutors::default();
        executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "counter".to_owned(),
            handlers: vec![discovered_handler("read"), discovered_handler("delete")],
        });
        let param = mcp_param(serde_json::json!({
            "server_label": "counter",
            "allowed_tools": ["read"],
            "require_approval": "never"
        }));

        let first = executors.mcp_server_tools(&param).await.unwrap();
        let first_output_id = first.list_tools_item.id.clone();
        let second = executors.mcp_server_tools(&param).await.unwrap();

        assert_eq!(first.discovered_handlers.len(), 1);
        assert_eq!(first.discovered_handlers[0].param.tool_name, "read");
        assert_eq!(first.list_tools_item.tools.len(), 1);
        assert_eq!(first.list_tools_item.tools[0].name, "read");
        assert_ne!(first_output_id, second.list_tools_item.id);
    }

    #[tokio::test]
    async fn cached_mcp_handlers_reject_empty_final_allowed_set() {
        let mut executors = GatewayExecutors::default();
        executors.insert(GatewayExecutorRegistration::Mcp {
            server_label: "counter".to_owned(),
            handlers: vec![discovered_handler("delete")],
        });
        let param = mcp_param(serde_json::json!({
            "server_label": "counter",
            "allowed_tools": ["read"],
            "require_approval": "never"
        }));

        let Err(error) = executors.mcp_server_tools(&param).await else {
            panic!("expected empty allowed set to be rejected");
        };

        assert!(error.to_string().contains("empty final allowed tool set"));
    }
}
