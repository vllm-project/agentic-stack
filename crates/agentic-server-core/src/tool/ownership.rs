//! Whether a tool is executed by the gateway itself or handed back to the client.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::handler::{GatewayExecutor, GatewayToolEventPlan, ToolError, ToolOutput};
use crate::types::io::OutputItem;
use crate::types::io::output::{FunctionToolCall, GatewayCallStatus};

/// Object-safe execution surface for a handler that has already been paired
/// with the exact parameter type declared by its [`GatewayExecutor`]
/// implementation.
trait ErasedGatewayExecutor: Send + Sync {
    fn execute(
        &self,
        call_id: &str,
        tool_name: &str,
        arguments: &str,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>>;

    fn plan_gateway_events(&self, call: &FunctionToolCall) -> GatewayToolEventPlan;

    fn public_output(
        &self,
        call: &FunctionToolCall,
        output: &ToolOutput,
        status: GatewayCallStatus,
    ) -> Option<OutputItem>;
}

struct TypedGatewayExecutor<E>
where
    E: GatewayExecutor + ?Sized,
{
    executor: Arc<E>,
    params: E::ExecutionParams,
}

impl<E> ErasedGatewayExecutor for TypedGatewayExecutor<E>
where
    E: GatewayExecutor + ?Sized,
{
    fn execute(
        &self,
        call_id: &str,
        tool_name: &str,
        arguments: &str,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        self.executor.execute(call_id, tool_name, arguments, &self.params)
    }

    fn plan_gateway_events(&self, call: &FunctionToolCall) -> GatewayToolEventPlan {
        self.executor.plan_gateway_events(call, &self.params)
    }

    fn public_output(
        &self,
        call: &FunctionToolCall,
        output: &ToolOutput,
        status: GatewayCallStatus,
    ) -> Option<OutputItem> {
        self.executor.public_output(call, output, status, &self.params)
    }
}

/// A resolved gateway handler plus its per-tool-name concurrency policy.
pub struct GatewayBinding {
    executor: Arc<dyn ErasedGatewayExecutor>,
    /// `Some` when this handler must not run concurrently with a second call
    /// to the SAME tool name (built from `!handler.supports_parallel_execution()`
    /// at registration time); `None` when it's safe to call itself concurrently.
    /// Never gates against other tool names.
    pub self_exclusion: Option<Arc<tokio::sync::Semaphore>>,
}

impl Clone for GatewayBinding {
    fn clone(&self) -> Self {
        Self {
            executor: Arc::clone(&self.executor),
            self_exclusion: self.self_exclusion.clone(),
        }
    }
}

impl GatewayBinding {
    #[must_use]
    pub(crate) fn new<E>(executor: Arc<E>, params: E::ExecutionParams) -> Self
    where
        E: GatewayExecutor + ?Sized,
    {
        let self_exclusion =
            (!executor.supports_parallel_execution()).then(|| Arc::new(tokio::sync::Semaphore::new(1)));
        Self {
            executor: Arc::new(TypedGatewayExecutor { executor, params }),
            self_exclusion,
        }
    }

    pub(crate) fn execute(
        &self,
        call_id: &str,
        tool_name: &str,
        arguments: &str,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        self.executor.execute(call_id, tool_name, arguments)
    }

    #[must_use]
    pub(crate) fn plan_gateway_events(&self, call: &FunctionToolCall) -> GatewayToolEventPlan {
        self.executor.plan_gateway_events(call)
    }

    #[must_use]
    pub(crate) fn public_output(
        &self,
        call: &FunctionToolCall,
        output: &ToolOutput,
        status: GatewayCallStatus,
    ) -> Option<OutputItem> {
        self.executor.public_output(call, output, status)
    }
}

pub enum ToolOwnership {
    Client,
    /// `None` means this tool type is gateway-owned in principle but has no
    /// handler implemented yet (e.g. `FileSearch`/`CodeInterpreter` today).
    Gateway(Option<GatewayBinding>),
}

impl Clone for ToolOwnership {
    fn clone(&self) -> Self {
        match self {
            Self::Client => Self::Client,
            Self::Gateway(binding) => Self::Gateway(binding.clone()),
        }
    }
}

impl ToolOwnership {
    #[must_use]
    pub fn is_gateway(&self) -> bool {
        matches!(self, Self::Gateway(_))
    }
}
