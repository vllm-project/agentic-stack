//! Tool framework — registry, handler trait, and normalization pipeline.
//!
//! Wire format types (`ResponsesTool`, param structs) live in [`crate::types::tools`].
//! This module owns the behavioral layer: routing, handler interface, and normalization.

pub mod codex;
pub mod custom;
pub mod executors;
pub mod function;
pub mod handler;
pub mod mcp;
pub mod normalize;
pub mod ownership;
pub mod registry;
pub mod shell;
pub mod web_search;

pub use codex::{CodexNamespaceHandler, NamespaceMap, model_visible_namespace_member_name};
pub use custom::CustomHandler;
pub use executors::{GatewayExecutorRegistration, GatewayExecutors};
pub use function::FunctionHandler;
pub use handler::{GatewayExecutor, GatewayToolEventPlan, ToolError, ToolHandler, ToolOutput};
pub use mcp::{McpClient, McpClientPool, McpDiscoveredHandler, McpError, McpHandler, McpOperation, McpServerEntry};
pub use ownership::{GatewayBinding, ToolOwnership};
pub use registry::{GatewayDispatchResult, ToolEntry, ToolRegistry, ToolType};
pub use shell::{ShellExecutor, ShellHandler};
pub use web_search::WebSearchHandler;
