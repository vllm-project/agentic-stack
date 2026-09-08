use std::future::Future;
use std::pin::Pin;

use crate::types::io::FunctionTool;
use crate::types::io::output::{FunctionToolCall, GatewayCallStatus, OutputItem};

pub(crate) const MAX_GATEWAY_TOOL_OUTPUT_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub call_id: String,
    pub output: String,
}

/// Tool-owned public lifecycle projection for one scheduled gateway call.
///
/// Handlers provide typed Responses output items while the gateway scheduler
/// remains responsible for output indexes, SSE framing, and event ordering.
#[derive(Debug, Clone, Default)]
pub struct GatewayToolEventPlan {
    started_output: Option<OutputItem>,
}

impl GatewayToolEventPlan {
    #[must_use]
    pub const fn new(started_output: Option<OutputItem>) -> Self {
        Self { started_output }
    }

    #[must_use]
    pub(crate) fn into_started_output(self) -> Option<OutputItem> {
        self.started_output
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("execution failed: {0}")]
    Execution(String),
    #[error("invalid tool config: {0}")]
    Config(String),
    #[error("upstream returned an invalid tool-search call")]
    InvalidUpstreamToolSearch,
    #[error("upstream returned a call for a function that has not been loaded")]
    UpstreamWithheldFunctionCall,
    /// A continuation request omitted the output for a pending function call
    /// from the prior turn.
    #[error("No tool output found for function call {call_id}.")]
    MissingOutput { call_id: String },
}

/// Trait implemented by every tool type — client-owned and gateway-owned alike.
///
/// Covers typed validation and normalization: the steps that apply to all
/// tools regardless of who executes them.
pub trait ToolHandler: Send + Sync {
    /// The public declaration parameters handled by this implementation.
    type ToolParams: Send + Sync;

    #[must_use]
    fn tool_type(&self) -> super::registry::ToolType;

    /// Validate the typed tool declaration parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] for obviously invalid configurations.
    fn validate(&self, params: &Self::ToolParams) -> Result<(), ToolError>;

    /// Normalise this tool declaration into vLLM-compatible `FunctionTool` entries.
    #[must_use]
    fn normalize(&self, params: &Self::ToolParams) -> Vec<FunctionTool>;
}

/// Extension of [`ToolHandler`] for tool types that are executed by the gateway.
///
/// Only executable gateway handlers implement this trait. MCP and web search
/// implement it today. File search and code interpreter are gateway-owned in
/// the registry but do not yet have executors. Client-owned tools (`Function`,
/// `ToolSearch`, `Custom`, `CodexNamespace`) do not implement it, so they cannot
/// be dispatched through this interface.
///
/// ## Note on `async fn` in traits
///
/// Native `async fn` in traits (Rust 1.75+) is not yet `dyn`-compatible, so this
/// trait uses explicit `Pin<Box<dyn Future>>` return types. Concrete executors
/// are paired with their typed parameters before the pair is erased for storage
/// in the heterogeneous tool registry.
pub trait GatewayExecutor: ToolHandler + 'static {
    /// Request-scoped parameters for one model-visible executable tool.
    ///
    /// These may differ from [`ToolHandler::ToolParams`]. MCP, for example,
    /// normalizes an [`McpToolParam`](crate::types::tools::McpToolParam) server
    /// declaration but executes one
    /// [`McpDiscoveredToolParam`](crate::types::tools::McpDiscoveredToolParam).
    type ExecutionParams: Clone + Send + Sync + 'static;

    /// Execute a tool call and return the result.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Execution`] if the tool call fails.
    fn execute(
        &self,
        call_id: &str,
        tool_name: &str,
        arguments: &str,
        params: &Self::ExecutionParams,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>>;

    /// Whether multiple calls to this same model-visible tool name may overlap.
    /// Defaults to `false`, which serializes only same-name calls; calls to
    /// different tools may still execute concurrently in the same round.
    #[must_use]
    fn supports_parallel_execution(&self) -> bool {
        false
    }

    /// Plans the typed public lifecycle for one gateway call.
    ///
    /// The returned plan must not assign protocol indexes or construct SSE
    /// frames. Defaults to an empty lifecycle for tools that have no public
    /// gateway-specific call item.
    #[must_use]
    fn plan_gateway_events(&self, call: &FunctionToolCall, params: &Self::ExecutionParams) -> GatewayToolEventPlan {
        GatewayToolEventPlan::new(self.started_output(call, params))
    }

    /// The placeholder output item shown while this call is in progress.
    ///
    /// Kept as the compatibility hook for existing gateway executors;
    /// implementations that need richer planning should override
    /// [`Self::plan_gateway_events`] instead.
    #[must_use]
    fn started_output(&self, call: &FunctionToolCall, params: &Self::ExecutionParams) -> Option<OutputItem> {
        let _ = (call, params);
        None
    }

    /// The public output item for a completed or failed call.
    /// Defaults to `None` (no gateway-specific shape).
    #[must_use]
    fn public_output(
        &self,
        call: &FunctionToolCall,
        output: &ToolOutput,
        status: GatewayCallStatus,
        params: &Self::ExecutionParams,
    ) -> Option<OutputItem> {
        let _ = (call, output, status, params);
        None
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    // Compile-time check: a GatewayExecutor with fixed associated parameter
    // types remains dyn-compatible for typed executor slots.
    fn _assert_gateway_executor_dyn_compatible(_: Arc<dyn GatewayExecutor<ToolParams = (), ExecutionParams = ()>>) {}
}
