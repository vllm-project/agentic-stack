use std::num::NonZeroUsize;
use std::time::Duration;

use futures::future::join_all;
use tokio::sync::Semaphore;

use crate::config::DEFAULT_MAX_CONCURRENT_GATEWAY_CALLS;
use crate::events::SSEEventType;
use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::gateway_accumulator::{GatewayStreamAccumulator, StreamEvent, emit_sse_frame, synthetic_event};
use crate::executor::request::RequestContext;
use crate::tool::{GatewayBinding, ToolError, ToolOutput, ToolOwnership, ToolRegistry};
use crate::types::io::output::{FunctionToolCall, GatewayCallStatus, McpCallStatus};
use crate::types::io::{InputItem, OutputItem, ResponsesInput};
use crate::types::request_response::ResponsePayload;
use crate::utils::common::{serialize_to_string, serialize_to_value};

/// Request-independent execution policy owned by one [`ExecutionContext`].
///
/// The nonzero type makes `.buffered(0)` unrepresentable. Distinct execution
/// contexts retain their own limits instead of sharing process-global state.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GatewaySchedulerPolicy {
    max_concurrent_calls: NonZeroUsize,
}

impl GatewaySchedulerPolicy {
    #[must_use]
    pub(crate) const fn new(max_concurrent_calls: NonZeroUsize) -> Self {
        Self { max_concurrent_calls }
    }
}

impl Default for GatewaySchedulerPolicy {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_CONCURRENT_GATEWAY_CALLS)
    }
}

/// Per-call wall-clock budget. A tool exceeding this yields an error output fed
/// back to the model (never a whole-request failure). `Duration::ZERO` disables
/// the timeout — for providers that manage their own.
///
/// Note: this bounds a single call, not the whole request. Worst-case request
/// latency scales with rounds and fan-out; an outer request-level deadline
/// would be the place to cap total time end-to-end.
const GATEWAY_TOOL_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone)]
pub(super) struct GatewayCallResult {
    pub(super) item_index: usize,
    pub(super) input_item: InputItem,
    pub(super) public_output: Option<OutputItem>,
}

/// Supplies the public output that completes a gateway event plan.
pub(super) trait GatewayPublicOutputSource {
    fn public_output(&self) -> Option<&OutputItem>;
}

impl GatewayPublicOutputSource for GatewayCallResult {
    fn public_output(&self) -> Option<&OutputItem> {
        self.public_output.as_ref()
    }
}

impl GatewayPublicOutputSource for OutputItem {
    fn public_output(&self) -> Option<&OutputItem> {
        Some(self)
    }
}

#[derive(Clone)]
pub(super) struct GatewayEventPlan {
    output_index: u32,
    started_output: Option<OutputItem>,
    completed_output: Option<OutputItem>,
    arguments: Option<String>,
}

#[derive(Clone)]
enum GatewayExecutionPlan {
    Bound(GatewayBinding),
    MissingHandler,
}

/// One gateway-owned call planned from the model output.
///
/// Execution and lifecycle data live in the same slot so their positional
/// relationship cannot diverge.
#[derive(Clone)]
struct GatewayCallPlan {
    item_index: usize,
    call: FunctionToolCall,
    execution: GatewayExecutionPlan,
    events: GatewayEventPlan,
}

/// Plans and executes one model-output round of gateway-owned calls.
///
/// The scheduler is the single source of gateway-call membership, original
/// item indexes, public output indexes, execution bindings, and lifecycle
/// plans. Tool handlers provide typed lifecycle projections; the scheduler
/// assigns protocol positions and emits the resulting events.
pub(super) struct GatewayScheduler {
    calls: Vec<GatewayCallPlan>,
    policy: GatewaySchedulerPolicy,
    timeout: Duration,
}

impl GatewayScheduler {
    pub(super) fn plan(
        output_items: &[OutputItem],
        registry: &ToolRegistry,
        output_offset: usize,
        policy: GatewaySchedulerPolicy,
    ) -> Self {
        Self::plan_with_timeout(output_items, registry, output_offset, policy, GATEWAY_TOOL_TIMEOUT)
    }

    fn plan_with_timeout(
        output_items: &[OutputItem],
        registry: &ToolRegistry,
        output_offset: usize,
        policy: GatewaySchedulerPolicy,
        timeout: Duration,
    ) -> Self {
        let calls = output_items
            .iter()
            .enumerate()
            .filter_map(|(item_index, item)| {
                let OutputItem::FunctionCall(call) = item else {
                    return None;
                };
                let entry = registry.lookup(&call.name)?;
                let ToolOwnership::Gateway(binding) = &entry.ownership else {
                    return None;
                };

                let started_output = binding
                    .as_ref()
                    .and_then(|binding| binding.plan_gateway_events(call).into_started_output());
                let execution = binding
                    .as_ref()
                    .map_or(GatewayExecutionPlan::MissingHandler, |binding| {
                        GatewayExecutionPlan::Bound(binding.clone())
                    });
                Some(GatewayCallPlan {
                    item_index,
                    call: call.clone(),
                    execution,
                    events: GatewayEventPlan {
                        output_index: u32::try_from(output_offset.saturating_add(item_index)).unwrap_or(u32::MAX),
                        started_output,
                        completed_output: None,
                        arguments: Some(call.arguments.clone()),
                    },
                })
            })
            .collect();
        Self { calls, policy, timeout }
    }

    #[cfg(test)]
    fn plan_with_test_timeout(
        output_items: &[OutputItem],
        registry: &ToolRegistry,
        output_offset: usize,
        timeout: Duration,
    ) -> Self {
        Self::plan_with_timeout(
            output_items,
            registry,
            output_offset,
            GatewaySchedulerPolicy::default(),
            timeout,
        )
    }

    pub(super) fn event_plans(&self) -> impl Iterator<Item = &GatewayEventPlan> {
        self.calls.iter().map(|call| &call.events)
    }

    pub(super) fn event_plan(&self, call_index: usize) -> Option<&GatewayEventPlan> {
        self.calls.get(call_index).map(|call| &call.events)
    }

    pub(super) fn call_index_for_item(&self, item_index: usize) -> Option<usize> {
        self.calls
            .binary_search_by_key(&item_index, |call| call.item_index)
            .ok()
    }

    /// Number of leading scheduled calls whose start lifecycle may be emitted
    /// before execution without overtaking an earlier client-owned call.
    pub(super) fn initial_event_run_len(&self, output_items: &[OutputItem], registry: &ToolRegistry) -> usize {
        let Some(first) = self.calls.first().map(|call| call.item_index) else {
            return 0;
        };
        if output_items[..first]
            .iter()
            .any(|item| matches!(item, OutputItem::FunctionCall(call) if registry.is_client_custom_name(&call.name)))
        {
            return 0;
        }
        self.calls
            .iter()
            .enumerate()
            .take_while(|(offset, call)| call.item_index == first.saturating_add(*offset))
            .count()
    }

    /// Executes planned calls under the bounded-concurrency policy and records
    /// each tool's completed public lifecycle on the same slot.
    pub(super) async fn execute(&mut self) -> ExecutorResult<Vec<GatewayCallResult>> {
        let execution_slots = Semaphore::new(self.policy.max_concurrent_calls.get());
        let results = join_all(
            self.calls
                .iter()
                .cloned()
                .map(|call| self.run_one(call, &execution_slots)),
        )
        .await
        .into_iter()
        .collect::<ExecutorResult<Vec<_>>>()?;

        debug_assert_eq!(self.calls.len(), results.len());
        for (planned, result) in self.calls.iter_mut().zip(&results) {
            debug_assert_eq!(planned.item_index, result.item_index);
            planned.events.completed_output.clone_from(&result.public_output);
        }
        Ok(results)
    }

    async fn run_one(&self, plan: GatewayCallPlan, execution_slots: &Semaphore) -> ExecutorResult<GatewayCallResult> {
        let GatewayCallPlan {
            item_index,
            call,
            execution,
            ..
        } = plan;
        let GatewayExecutionPlan::Bound(binding) = execution else {
            let output = execution_error_output(
                &call,
                &format!("gateway tool '{}' has no registered handler", call.name),
            )?;
            return Ok(GatewayCallResult {
                item_index,
                input_item: InputItem::FunctionCallOutput(output.into()),
                public_output: None,
            });
        };

        let _permit = match &binding.self_exclusion {
            Some(semaphore) => Some(semaphore.acquire().await.expect("semaphore is never closed")),
            None => None,
        };
        let _execution_slot = execution_slots.acquire().await.expect("semaphore is never closed");

        let dispatched = if self.timeout.is_zero() {
            binding.execute(&call.call_id, &call.name, &call.arguments).await
        } else {
            match tokio::time::timeout(
                self.timeout,
                binding.execute(&call.call_id, &call.name, &call.arguments),
            )
            .await
            {
                Ok(output) => output,
                Err(_elapsed) => Err(ToolError::Execution(format!(
                    "gateway tool '{}' timed out after {:?}",
                    call.name, self.timeout
                ))),
            }
        };
        let (output, status) = match dispatched {
            Ok(output) => (output, GatewayCallStatus::Completed),
            Err(ToolError::Execution(message) | ToolError::Config(message)) => {
                (execution_error_output(&call, &message)?, GatewayCallStatus::Failed)
            }
            Err(error @ ToolError::MissingOutput { .. }) => return Err(ExecutorError::from(error)),
        };
        let public_output = binding.public_output(&call, &output, status);
        Ok(GatewayCallResult {
            item_index,
            input_item: InputItem::FunctionCallOutput(output.into()),
            public_output,
        })
    }
}

pub(super) fn has_client_owned_calls(output_items: &[OutputItem], registry: &ToolRegistry) -> bool {
    output_items.iter().any(|item| item.requires_client_action(registry))
}

fn execution_error_output(call: &FunctionToolCall, message: &str) -> ExecutorResult<ToolOutput> {
    let output = serialize_to_string(&serde_json::json!({ "error": message })).map_err(ExecutorError::JsonError)?;
    Ok(ToolOutput {
        call_id: call.call_id.clone(),
        output,
    })
}

pub(super) fn public_output_items(
    output_items: &[OutputItem],
    registry: &ToolRegistry,
    gateway_results: &[GatewayCallResult],
) -> Vec<OutputItem> {
    output_items
        .iter()
        .enumerate()
        .map(|(item_index, item)| match item {
            OutputItem::FunctionCall(call) if registry.is_client_custom_name(&call.name) => {
                crate::tool::CustomHandler::output_item(call)
            }
            OutputItem::FunctionCall(call) if registry.is_gateway_owned_name(&call.name) => gateway_results
                .iter()
                .find(|result| result.item_index == item_index)
                .and_then(|result| result.public_output.clone())
                .unwrap_or_else(|| OutputItem::FunctionCall(call.clone())),
            other => other.clone(),
        })
        .collect()
}

pub(super) fn mcp_list_tools_event_plans(
    public_output_items: &[OutputItem],
    output_offset: usize,
) -> Vec<GatewayEventPlan> {
    public_output_items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            let OutputItem::McpListTools(list_tools) = item else {
                return None;
            };
            Some(GatewayEventPlan {
                output_index: u32::try_from(output_offset.saturating_add(index)).unwrap_or(u32::MAX),
                started_output: Some(crate::tool::mcp::handler::started_list_tools_output_item(list_tools)),
                completed_output: Some(item.clone()),
                arguments: None,
            })
        })
        .collect()
}

pub(super) fn compaction_event_plans(
    public_output_items: &[OutputItem],
    output_offset: usize,
) -> Vec<GatewayEventPlan> {
    public_output_items
        .iter()
        .enumerate()
        .filter(|(_, item)| matches!(item, OutputItem::Compaction(_)))
        .map(|(index, item)| GatewayEventPlan {
            output_index: u32::try_from(output_offset.saturating_add(index)).unwrap_or(u32::MAX),
            started_output: Some(item.clone()),
            completed_output: Some(item.clone()),
            arguments: None,
        })
        .collect()
}

fn output_item_value(item: &OutputItem) -> ExecutorResult<serde_json::Value> {
    serde_json::to_value(item).map_err(ExecutorError::JsonError)
}

pub(super) fn emit_response_start_events(
    payload: &ResponsePayload,
    stream_accumulator: &mut GatewayStreamAccumulator,
    stream_sender: &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
) -> ExecutorResult<()> {
    let mut response = payload.clone();
    "in_progress".clone_into(&mut response.status);
    response.output.clear();
    response.usage = None;
    let response = serialize_to_value(&response).map_err(ExecutorError::JsonError)?;
    for event_type in [SSEEventType::ResponseCreated, SSEEventType::ResponseInProgress] {
        let mut event = synthetic_event(event_type, [("response".to_owned(), response.clone())])?;
        emit_gateway_event(&mut event, stream_accumulator, stream_sender)?;
    }
    Ok(())
}

#[cfg(test)]
fn complete_gateway_event_plans<T: GatewayPublicOutputSource>(plans: &mut [GatewayEventPlan], completed: &[T]) {
    for (plan, source) in plans.iter_mut().zip(completed) {
        plan.completed_output = source.public_output().cloned();
    }
}

pub(super) fn emit_gateway_start_events<'a>(
    plans: impl IntoIterator<Item = &'a GatewayEventPlan>,
    stream_accumulator: &mut GatewayStreamAccumulator,
    stream_sender: &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
) -> ExecutorResult<()> {
    for plan in plans {
        let Some(output_item) = &plan.started_output else {
            continue;
        };
        let item = output_item_value(output_item)?;
        let mut added_event = synthetic_event(
            SSEEventType::OutputItemAdded,
            [
                ("output_index".to_owned(), serde_json::json!(plan.output_index)),
                ("item".to_owned(), item),
            ],
        )?;
        emit_gateway_event(&mut added_event, stream_accumulator, stream_sender)?;
        match output_item {
            OutputItem::WebSearchCall(web_search_call) => {
                let mut in_progress_event = synthetic_event(
                    SSEEventType::WebSearchCallInProgress,
                    [
                        ("item_id".to_owned(), serde_json::json!(web_search_call.id)),
                        ("output_index".to_owned(), serde_json::json!(plan.output_index)),
                    ],
                )?;
                emit_gateway_event(&mut in_progress_event, stream_accumulator, stream_sender)?;
                let mut searching_event = synthetic_event(
                    SSEEventType::WebSearchCallSearching,
                    [
                        ("item_id".to_owned(), serde_json::json!(web_search_call.id)),
                        ("output_index".to_owned(), serde_json::json!(plan.output_index)),
                    ],
                )?;
                emit_gateway_event(&mut searching_event, stream_accumulator, stream_sender)?;
            }
            OutputItem::McpCall(mcp_call) => {
                let mut in_progress_event = synthetic_event(
                    SSEEventType::McpCallInProgress,
                    [
                        ("item_id".to_owned(), serde_json::json!(mcp_call.id)),
                        ("output_index".to_owned(), serde_json::json!(plan.output_index)),
                    ],
                )?;
                emit_gateway_event(&mut in_progress_event, stream_accumulator, stream_sender)?;
                let arguments = plan.arguments.as_deref().unwrap_or_default();
                let mut arguments_delta_event = synthetic_event(
                    SSEEventType::McpCallArgumentsDelta,
                    [
                        ("delta".to_owned(), serde_json::json!(arguments)),
                        ("item_id".to_owned(), serde_json::json!(mcp_call.id)),
                        ("output_index".to_owned(), serde_json::json!(plan.output_index)),
                    ],
                )?;
                emit_gateway_event(&mut arguments_delta_event, stream_accumulator, stream_sender)?;
                let mut arguments_done_event = synthetic_event(
                    SSEEventType::McpCallArgumentsDone,
                    [
                        ("arguments".to_owned(), serde_json::json!(arguments)),
                        ("item_id".to_owned(), serde_json::json!(mcp_call.id)),
                        ("output_index".to_owned(), serde_json::json!(plan.output_index)),
                    ],
                )?;
                emit_gateway_event(&mut arguments_done_event, stream_accumulator, stream_sender)?;
            }
            OutputItem::McpListTools(list_tools) => {
                let mut in_progress_event = synthetic_event(
                    SSEEventType::McpListToolsInProgress,
                    [
                        ("item_id".to_owned(), serde_json::json!(list_tools.id)),
                        ("output_index".to_owned(), serde_json::json!(plan.output_index)),
                    ],
                )?;
                emit_gateway_event(&mut in_progress_event, stream_accumulator, stream_sender)?;
            }
            OutputItem::Message(_)
            | OutputItem::FunctionCall(_)
            | OutputItem::CustomToolCall(_)
            | OutputItem::Reasoning(_)
            | OutputItem::Compaction(_)
            | OutputItem::Unknown => {}
        }
    }
    Ok(())
}

pub(super) fn emit_gateway_completed_events<'a, T: GatewayPublicOutputSource>(
    results: &[T],
    plans: impl IntoIterator<Item = &'a GatewayEventPlan>,
    stream_accumulator: &mut GatewayStreamAccumulator,
    stream_sender: &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
) -> ExecutorResult<()> {
    for (index, plan) in plans.into_iter().enumerate() {
        let Some(public_output) = plan
            .completed_output
            .as_ref()
            .or_else(|| results.get(index).and_then(GatewayPublicOutputSource::public_output))
        else {
            continue;
        };
        let output_index = plan.output_index;
        let completed_event = match public_output {
            OutputItem::WebSearchCall(web_search_call) => {
                Some((SSEEventType::WebSearchCallCompleted, web_search_call.id.as_str()))
            }
            OutputItem::McpCall(mcp_call) => Some((
                if mcp_call.status == Some(McpCallStatus::Failed) {
                    SSEEventType::McpCallFailed
                } else {
                    SSEEventType::McpCallCompleted
                },
                mcp_call.id.as_str(),
            )),
            OutputItem::McpListTools(list_tools) => Some((
                if list_tools.error.is_some() {
                    SSEEventType::McpListToolsFailed
                } else {
                    SSEEventType::McpListToolsCompleted
                },
                list_tools.id.as_str(),
            )),
            OutputItem::Compaction(_) => None,
            OutputItem::Message(_)
            | OutputItem::FunctionCall(_)
            | OutputItem::CustomToolCall(_)
            | OutputItem::Reasoning(_)
            | OutputItem::Unknown => continue,
        };
        let item = output_item_value(public_output)?;
        if let Some((event_type, item_id)) = completed_event {
            let mut completed_fields = serde_json::Map::from_iter([
                ("item_id".to_owned(), serde_json::json!(item_id)),
                ("output_index".to_owned(), serde_json::json!(output_index)),
            ]);
            if matches!(public_output, OutputItem::WebSearchCall(_)) {
                completed_fields.insert("item".to_owned(), item.clone());
            }
            let mut completed_event = synthetic_event(event_type, completed_fields)?;
            emit_gateway_event(&mut completed_event, stream_accumulator, stream_sender)?;
        }
        let mut done_event = synthetic_event(
            SSEEventType::OutputItemDone,
            [
                ("output_index".to_owned(), serde_json::json!(output_index)),
                ("item".to_owned(), item),
            ],
        )?;
        emit_gateway_event(&mut done_event, stream_accumulator, stream_sender)?;
    }
    Ok(())
}

pub(super) async fn execute_and_emit_output_calls(
    output_items: &[OutputItem],
    registry: &ToolRegistry,
    output_offset: usize,
    policy: GatewaySchedulerPolicy,
    mut stream: Option<(
        &mut GatewayStreamAccumulator,
        &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    )>,
) -> ExecutorResult<Vec<GatewayCallResult>> {
    let mut scheduler = GatewayScheduler::plan(output_items, registry, output_offset, policy);
    if let Some((stream_accumulator, stream_sender)) = stream.as_mut() {
        emit_gateway_start_events(scheduler.event_plans(), stream_accumulator, stream_sender)?;
    }
    let gateway_results = scheduler.execute().await?;
    if let Some((stream_accumulator, stream_sender)) = stream.as_mut() {
        emit_gateway_completed_events(
            &gateway_results,
            scheduler.event_plans(),
            stream_accumulator,
            stream_sender,
        )?;
    }
    Ok(gateway_results)
}

fn emit_gateway_event(
    frame: &mut crate::events::EventFrame,
    stream_accumulator: &mut GatewayStreamAccumulator,
    stream_sender: &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
) -> ExecutorResult<()> {
    if stream_accumulator.process_event(frame, 0) {
        emit_sse_frame(stream_sender, frame)?;
    }
    Ok(())
}

pub(super) fn append_input_item(input: &mut ResponsesInput, item: InputItem) {
    match input {
        ResponsesInput::Items(items) => items.push(item),
        ResponsesInput::Text(text) => {
            let text_input = ResponsesInput::Text(std::mem::take(text));
            let mut items = Vec::<InputItem>::from(&text_input);
            items.push(item);
            *input = ResponsesInput::Items(items);
        }
    }
}

pub(super) fn append_output_items_to_input(input: &mut ResponsesInput, output_items: &[OutputItem]) {
    for input_item in output_items.iter().filter_map(OutputItem::to_input_item) {
        append_input_item(input, input_item);
    }
}

pub(super) fn append_tool_outputs(ctx: &mut RequestContext, tool_outputs: Vec<InputItem>) {
    for output in tool_outputs {
        ctx.new_input_items.push(output.clone());
        append_input_item(&mut ctx.enriched_request.input, output);
    }
}

pub(super) fn append_gateway_calls_to_new_input(
    ctx: &mut RequestContext,
    output_items: &[OutputItem],
    registry: &ToolRegistry,
) {
    ctx.new_input_items.extend(output_items.iter().filter_map(|item| {
        let OutputItem::FunctionCall(call) = item else {
            return None;
        };
        registry
            .is_gateway_owned_name(&call.name)
            .then(|| InputItem::FunctionCall(call.clone().into()))
    }));
}

#[cfg(test)]
mod tests {
    use super::GatewayCallResult;
    use crate::executor::accumulator::ResponseAccumulator;
    use crate::types::io::output::{FunctionToolCall, McpListTool, McpListTools};
    use crate::types::io::{CompactionItem, InputItem, McpCallStatus};
    use tokio::sync::{Notify, mpsc};

    fn parse_named_sse_event(content: &str) -> Value {
        let body = content.strip_suffix("\n\n").expect("SSE event terminator");
        let (event_line, data_line) = body.split_once('\n').expect("named SSE event and data lines");
        let event_name = event_line.strip_prefix("event: ").expect("SSE event prefix");
        let data = data_line.strip_prefix("data: ").expect("SSE data prefix");
        let event = serde_json::from_str::<Value>(data).expect("event JSON");
        assert_eq!(event["type"].as_str(), Some(event_name));
        event
    }

    use std::num::NonZeroUsize;
    use std::pin::Pin;
    use std::sync::Arc;

    use serde_json::Value;

    use super::{GatewayCallPlan, GatewayEventPlan, GatewayExecutionPlan, GatewayScheduler, GatewaySchedulerPolicy};
    use crate::tool::{
        GatewayBinding, GatewayExecutor, GatewayExecutors, GatewayToolEventPlan, ToolError, ToolHandler, ToolOutput,
        ToolRegistry, ToolType,
    };
    use crate::types::io::OutputItem;
    use crate::types::io::output::GatewayCallStatus;
    use crate::types::io::tools::FunctionTool;
    use crate::types::tools::{ResponsesTool, WebSearchToolParam};

    /// A gateway executor that sleeps ~50ms — comfortably longer than the tiny
    /// timeout the test injects, forcing the timeout path without a paused clock.
    struct SlowExecutor;

    impl ToolHandler for SlowExecutor {
        type ToolParams = WebSearchToolParam;

        fn tool_type(&self) -> ToolType {
            ToolType::WebSearch
        }
        fn validate(&self, _params: &WebSearchToolParam) -> Result<(), ToolError> {
            Ok(())
        }
        fn normalize(&self, _params: &WebSearchToolParam) -> Vec<FunctionTool> {
            Vec::new()
        }
    }

    impl GatewayExecutor for SlowExecutor {
        type ExecutionParams = WebSearchToolParam;

        fn execute(
            &self,
            call_id: &str,
            _tool_name: &str,
            _arguments: &str,
            _params: &WebSearchToolParam,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            let call_id = call_id.to_owned();
            Box::pin(async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                Ok(ToolOutput {
                    call_id,
                    output: "unreachable".to_owned(),
                })
            })
        }

        fn supports_parallel_execution(&self) -> bool {
            true
        }

        fn plan_gateway_events(&self, call: &FunctionToolCall, _params: &WebSearchToolParam) -> GatewayToolEventPlan {
            GatewayToolEventPlan::new(crate::tool::web_search::started_output_item(call))
        }

        fn public_output(
            &self,
            call: &FunctionToolCall,
            output: &ToolOutput,
            status: GatewayCallStatus,
            _params: &WebSearchToolParam,
        ) -> Option<OutputItem> {
            crate::tool::web_search::output_item(call, output, status)
        }
    }

    fn web_search_call(call_id: &str) -> FunctionToolCall {
        FunctionToolCall {
            id: format!("fc_{call_id}"),
            call_id: call_id.to_owned(),
            name: "web_search".to_owned(),
            arguments: "{}".to_owned(),
            status: crate::types::event::MessageStatus::Completed,
            namespace: None,
        }
    }

    #[tokio::test]
    async fn hung_gateway_call_times_out_into_error_output() {
        let web_search: ResponsesTool =
            serde_json::from_value(serde_json::json!({"type": "web_search_preview"})).expect("web_search tool param");
        let mut executors = GatewayExecutors::default();
        executors.insert(Arc::new(SlowExecutor));
        let mut tools = [web_search];
        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("registry builds");

        // 1ms budget vs a 50ms tool → the timeout fires. Must return (not hang):
        // the stuck call becomes an error output the loop can feed back.
        let output_items = [OutputItem::FunctionCall(web_search_call("call_hang"))];
        let mut scheduler =
            GatewayScheduler::plan_with_test_timeout(&output_items, &registry, 0, std::time::Duration::from_millis(1));
        let result = scheduler
            .execute()
            .await
            .expect("timeout is isolated as an error output, not a dispatch failure")
            .remove(0);

        assert_eq!(result.item_index, 0);
        // A failed web_search still yields a public web_search_call item.
        assert!(matches!(result.public_output, Some(OutputItem::WebSearchCall(_))));
        // The fed-back tool output is an error JSON mentioning the timeout.
        let InputItem::FunctionCallOutput(msg) = &result.input_item else {
            panic!("expected a function_call_output");
        };
        let body = serde_json::to_string(msg).expect("serialize output");
        assert!(
            body.contains("timed out"),
            "error output should mention the timeout: {body}"
        );
    }

    #[tokio::test]
    async fn gateway_call_without_configured_provider_becomes_error_output() {
        // Declare web_search but build the registry with NO provider for it —
        // `web_search_handler()` falls back to `WebSearchHandler::spec_only()`,
        // so the entry still has a real handler (public_output/started_output
        // work normally) but `execute()` fails. This must surface an error
        // output, not fail the whole request.
        let web_search: ResponsesTool =
            serde_json::from_value(serde_json::json!({"type": "web_search_preview"})).expect("web_search tool param");
        let mut tools = [web_search];
        let mut executors = GatewayExecutors::default();
        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("registry builds");

        let output_items = [OutputItem::FunctionCall(web_search_call("call_no_handler"))];
        let mut scheduler =
            GatewayScheduler::plan_with_test_timeout(&output_items, &registry, 0, std::time::Duration::ZERO);
        let result = scheduler
            .execute()
            .await
            .expect("a missing provider is isolated as an error output, not a dispatch failure")
            .remove(0);

        assert_eq!(result.item_index, 0);
        assert!(matches!(result.public_output, Some(OutputItem::WebSearchCall(_))));
        let InputItem::FunctionCallOutput(msg) = &result.input_item else {
            panic!("expected a function_call_output");
        };
        let body = serde_json::to_string(msg).expect("serialize output");
        assert!(
            body.contains("spec-only handler cannot execute tools"),
            "error output should mention the missing provider: {body}"
        );
    }

    #[tokio::test]
    async fn scheduler_retains_event_slot_for_gateway_tool_without_handler() {
        let file_search: ResponsesTool = serde_json::from_value(serde_json::json!({
            "type": "file_search",
            "vector_store_ids": ["vs_test"]
        }))
        .expect("file_search tool param");
        let web_search: ResponsesTool =
            serde_json::from_value(serde_json::json!({"type": "web_search_preview"})).expect("web_search tool param");
        let mut tools = [file_search, web_search];
        let mut executors = GatewayExecutors::default();
        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("registry builds");

        let mut file_search_call = web_search_call("call_file");
        file_search_call.name = "file_search".to_owned();
        let mut search_call = web_search_call("call_web");
        search_call.arguments = r#"{"query":"weather"}"#.to_owned();
        let output_items = [
            OutputItem::FunctionCall(file_search_call),
            OutputItem::FunctionCall(search_call),
        ];
        let mut scheduler = GatewayScheduler::plan(&output_items, &registry, 7, GatewaySchedulerPolicy::default());

        {
            let plans = scheduler.event_plans().collect::<Vec<_>>();
            assert_eq!(plans.len(), 2, "every gateway-owned call needs one scheduled slot");
            assert_eq!(plans[0].output_index, 7);
            assert!(plans[0].started_output.is_none());
            assert_eq!(plans[1].output_index, 8);
            assert!(matches!(plans[1].started_output, Some(OutputItem::WebSearchCall(_))));
        }

        let results = scheduler.execute().await.expect("all scheduled slots complete");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].item_index, 0);
        assert!(results[0].public_output.is_none());
        assert_eq!(results[1].item_index, 1);
        assert!(matches!(results[1].public_output, Some(OutputItem::WebSearchCall(_))));

        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut stream_accumulator = crate::executor::gateway_accumulator::GatewayStreamAccumulator::new();
        super::emit_gateway_start_events(scheduler.event_plans(), &mut stream_accumulator, &sender)
            .expect("start events");
        super::emit_gateway_completed_events(&results, scheduler.event_plans(), &mut stream_accumulator, &sender)
            .expect("completed events");

        let events = std::iter::from_fn(|| receiver.try_recv().ok())
            .map(|event| parse_named_sse_event(&event.content))
            .collect::<Vec<_>>();
        assert!(!events.is_empty());
        assert!(events.iter().all(|event| event["output_index"] == 8));
        assert!(events.iter().all(|event| {
            event["item"]["type"]
                .as_str()
                .is_none_or(|type_| type_ == "web_search_call")
        }));
    }

    /// Proves that a handler which opts into same-tool parallel execution really
    /// overlaps calls under the global window. Four 50ms calls sequentially would
    /// take ~200ms; with the default window of five they should finish near one slot.
    #[tokio::test]
    async fn gateway_scheduler_overlaps_parallel_safe_same_tool_calls() {
        let web_search: ResponsesTool =
            serde_json::from_value(serde_json::json!({"type": "web_search_preview"})).expect("web_search tool param");
        let mut executors = GatewayExecutors::default();
        executors.insert(Arc::new(SlowExecutor));
        let mut tools = [web_search];
        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("registry builds");

        let output_items: Vec<OutputItem> = ["call_a", "call_b", "call_c", "call_d"]
            .into_iter()
            .map(|call_id| OutputItem::FunctionCall(web_search_call(call_id)))
            .collect();

        let started = std::time::Instant::now();
        let mut scheduler = GatewayScheduler::plan(&output_items, &registry, 0, GatewaySchedulerPolicy::default());
        let results = scheduler.execute().await.expect("all calls execute");
        let elapsed = started.elapsed();

        assert_eq!(results.len(), 4);
        // Well under the ~200ms four sequential 50ms calls would take, and close to
        // one 50ms slot -- proves the calls actually overlapped in wall-clock time.
        assert!(
            elapsed < std::time::Duration::from_millis(150),
            "four concurrent 50ms calls took {elapsed:?}, expected well under 150ms"
        );
        // The scheduler restores original item order regardless of completion order.
        assert_eq!(
            results.iter().map(|result| result.item_index).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }

    #[tokio::test]
    async fn gateway_scheduler_policy_bounds_parallel_safe_calls() {
        let web_search: ResponsesTool =
            serde_json::from_value(serde_json::json!({"type": "web_search_preview"})).expect("web_search tool param");
        let mut executors = GatewayExecutors::default();
        executors.insert(Arc::new(SlowExecutor));
        let mut tools = [web_search];
        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("registry builds");
        let output_items = ["call_a", "call_b", "call_c", "call_d"]
            .into_iter()
            .map(|call_id| OutputItem::FunctionCall(web_search_call(call_id)))
            .collect::<Vec<_>>();
        let policy = GatewaySchedulerPolicy::new(NonZeroUsize::MIN);

        let started = std::time::Instant::now();
        let mut scheduler = GatewayScheduler::plan(&output_items, &registry, 0, policy);
        let results = scheduler.execute().await.expect("all calls execute");
        let elapsed = started.elapsed();

        assert_eq!(results.len(), 4);
        assert!(
            elapsed >= std::time::Duration::from_millis(180),
            "a one-call scheduler window completed four 50ms calls in {elapsed:?}"
        );
    }

    /// A handler that inherits the conservative `false` default serializes calls
    /// to its own tool name. This does not prevent different tool names from
    /// overlapping elsewhere in the same round.
    struct ExclusiveSlowExecutor;

    impl ToolHandler for ExclusiveSlowExecutor {
        type ToolParams = WebSearchToolParam;

        fn tool_type(&self) -> ToolType {
            ToolType::WebSearch
        }
        fn validate(&self, _params: &WebSearchToolParam) -> Result<(), ToolError> {
            Ok(())
        }
        fn normalize(&self, _params: &WebSearchToolParam) -> Vec<FunctionTool> {
            Vec::new()
        }
    }

    impl GatewayExecutor for ExclusiveSlowExecutor {
        type ExecutionParams = WebSearchToolParam;

        fn execute(
            &self,
            call_id: &str,
            _tool_name: &str,
            _arguments: &str,
            _params: &WebSearchToolParam,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            let call_id = call_id.to_owned();
            Box::pin(async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                Ok(ToolOutput {
                    call_id,
                    output: "unreachable".to_owned(),
                })
            })
        }
    }

    #[tokio::test]
    async fn gateway_scheduler_serializes_non_parallel_safe_same_tool_calls() {
        let web_search: ResponsesTool =
            serde_json::from_value(serde_json::json!({"type": "web_search_preview"})).expect("web_search tool param");
        let mut executors = GatewayExecutors::default();
        executors.insert(Arc::new(ExclusiveSlowExecutor));
        let mut tools = [web_search];
        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .expect("registry builds");

        let output_items: Vec<OutputItem> = ["call_a", "call_b", "call_c", "call_d"]
            .into_iter()
            .map(|call_id| OutputItem::FunctionCall(web_search_call(call_id)))
            .collect();

        let started = std::time::Instant::now();
        let mut scheduler = GatewayScheduler::plan(&output_items, &registry, 0, GatewaySchedulerPolicy::default());
        let results = scheduler.execute().await.expect("all calls execute");
        let elapsed = started.elapsed();

        assert_eq!(results.len(), 4);
        // Four 50ms calls run sequentially (one at a time) should take close to
        // 200ms -- well above what four *concurrent* 50ms calls would take.
        assert!(
            elapsed >= std::time::Duration::from_millis(180),
            "four exclusive 50ms calls took {elapsed:?}, expected close to 200ms"
        );
    }

    struct SchedulingProbeExecutor {
        blocked_call_id: &'static str,
        observed_call_id: &'static str,
        blocked_started: Arc<Notify>,
        observed_started: Arc<Notify>,
        release_blocked: Arc<Notify>,
        supports_parallel_execution: bool,
    }

    impl ToolHandler for SchedulingProbeExecutor {
        type ToolParams = WebSearchToolParam;

        fn tool_type(&self) -> ToolType {
            ToolType::WebSearch
        }

        fn validate(&self, _params: &WebSearchToolParam) -> Result<(), ToolError> {
            Ok(())
        }

        fn normalize(&self, _params: &WebSearchToolParam) -> Vec<FunctionTool> {
            Vec::new()
        }
    }

    impl GatewayExecutor for SchedulingProbeExecutor {
        type ExecutionParams = WebSearchToolParam;

        fn execute(
            &self,
            call_id: &str,
            _tool_name: &str,
            _arguments: &str,
            _params: &WebSearchToolParam,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
            let call_id = call_id.to_owned();
            let blocked_started = Arc::clone(&self.blocked_started);
            let observed_started = Arc::clone(&self.observed_started);
            let release_blocked = Arc::clone(&self.release_blocked);
            Box::pin(async move {
                if call_id == self.blocked_call_id {
                    blocked_started.notify_one();
                    release_blocked.notified().await;
                } else if call_id == self.observed_call_id {
                    observed_started.notify_one();
                }
                Ok(ToolOutput {
                    call_id,
                    output: "completed".to_owned(),
                })
            })
        }

        fn supports_parallel_execution(&self) -> bool {
            self.supports_parallel_execution
        }
    }

    fn scheduler_test_params() -> WebSearchToolParam {
        serde_json::from_value(serde_json::json!({"type": "web_search_preview"})).expect("web_search tool param")
    }

    fn gateway_call_plan(item_index: usize, call_id: &str, name: &str, binding: GatewayBinding) -> GatewayCallPlan {
        let mut call = web_search_call(call_id);
        call.name = name.to_owned();
        GatewayCallPlan {
            item_index,
            events: GatewayEventPlan {
                output_index: u32::try_from(item_index).expect("test item index fits in u32"),
                started_output: None,
                completed_output: None,
                arguments: Some(call.arguments.clone()),
            },
            call,
            execution: GatewayExecutionPlan::Bound(binding),
        }
    }

    #[tokio::test]
    async fn gateway_scheduler_keeps_same_tool_waiter_outside_global_execution_window() {
        let blocked_started = Arc::new(Notify::new());
        let unrelated_started = Arc::new(Notify::new());
        let release_blocked = Arc::new(Notify::new());
        let executor = Arc::new(SchedulingProbeExecutor {
            blocked_call_id: "call_a1",
            observed_call_id: "call_b",
            blocked_started: Arc::clone(&blocked_started),
            observed_started: Arc::clone(&unrelated_started),
            release_blocked: Arc::clone(&release_blocked),
            supports_parallel_execution: false,
        });
        let binding_a = GatewayBinding::new(Arc::clone(&executor), scheduler_test_params());
        let binding_b = GatewayBinding::new(executor, scheduler_test_params());
        let calls = vec![
            gateway_call_plan(0, "call_a1", "tool_a", binding_a.clone()),
            gateway_call_plan(1, "call_a2", "tool_a", binding_a),
            gateway_call_plan(2, "call_b", "tool_b", binding_b),
        ];
        let mut scheduler = GatewayScheduler {
            calls,
            policy: GatewaySchedulerPolicy::new(NonZeroUsize::new(2).expect("nonzero test limit")),
            timeout: std::time::Duration::ZERO,
        };

        let execution = tokio::spawn(async move { scheduler.execute().await });
        tokio::time::timeout(std::time::Duration::from_secs(1), blocked_started.notified())
            .await
            .expect("first same-tool call starts");
        tokio::time::timeout(std::time::Duration::from_secs(1), unrelated_started.notified())
            .await
            .expect("unrelated call starts while the same-tool waiter remains queued");
        release_blocked.notify_one();

        let results = execution
            .await
            .expect("scheduler task joins")
            .expect("scheduler succeeds");
        assert_eq!(
            results.iter().map(|result| result.item_index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[tokio::test]
    async fn gateway_scheduler_refills_execution_window_before_earlier_call_finishes() {
        let blocked_started = Arc::new(Notify::new());
        let refilled_call_started = Arc::new(Notify::new());
        let release_blocked = Arc::new(Notify::new());
        let executor = Arc::new(SchedulingProbeExecutor {
            blocked_call_id: "call_a",
            observed_call_id: "call_c",
            blocked_started: Arc::clone(&blocked_started),
            observed_started: Arc::clone(&refilled_call_started),
            release_blocked: Arc::clone(&release_blocked),
            supports_parallel_execution: true,
        });
        let binding = GatewayBinding::new(executor, scheduler_test_params());
        let calls = vec![
            gateway_call_plan(0, "call_a", "tool", binding.clone()),
            gateway_call_plan(1, "call_b", "tool", binding.clone()),
            gateway_call_plan(2, "call_c", "tool", binding),
        ];
        let mut scheduler = GatewayScheduler {
            calls,
            policy: GatewaySchedulerPolicy::new(NonZeroUsize::new(2).expect("nonzero test limit")),
            timeout: std::time::Duration::ZERO,
        };

        let execution = tokio::spawn(async move { scheduler.execute().await });
        tokio::time::timeout(std::time::Duration::from_secs(1), blocked_started.notified())
            .await
            .expect("first call starts");
        tokio::time::timeout(std::time::Duration::from_secs(1), refilled_call_started.notified())
            .await
            .expect("third call refills the slot released by the completed second call");
        release_blocked.notify_one();

        let results = execution
            .await
            .expect("scheduler task joins")
            .expect("scheduler succeeds");
        assert_eq!(
            results.iter().map(|result| result.item_index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn mcp_list_tools_uses_shared_gateway_event_lifecycle() {
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
        let discovered_output = crate::tool::mcp::handler::list_tools_output_item(&list_tools);
        let public_output = super::public_output_items(&[discovered_output], &ToolRegistry::default(), &[]);
        let plans = super::mcp_list_tools_event_plans(&public_output, 0);
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut stream_accumulator = crate::executor::gateway_accumulator::GatewayStreamAccumulator::new();
        stream_accumulator
            .process_sse_line(r#"data: {"type":"response.created"}"#, 0)
            .expect("response.created");
        stream_accumulator
            .process_sse_line(r#"data: {"type":"response.in_progress"}"#, 0)
            .expect("response.in_progress");

        super::emit_gateway_start_events(&plans, &mut stream_accumulator, &sender).expect("start events");
        super::emit_gateway_completed_events(&public_output, &plans, &mut stream_accumulator, &sender)
            .expect("completed events");

        let events = std::iter::from_fn(|| receiver.try_recv().ok())
            .map(|event| parse_named_sse_event(&event.content))
            .collect::<Vec<_>>();
        assert_eq!(
            events
                .iter()
                .map(|event| event["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec![
                "response.output_item.added",
                "response.mcp_list_tools.in_progress",
                "response.mcp_list_tools.completed",
                "response.output_item.done",
            ]
        );
        assert_eq!(
            events
                .iter()
                .map(|event| event["sequence_number"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![2, 3, 4, 5]
        );
        assert_eq!(events[0]["item"]["tools"], serde_json::json!([]));
        assert_eq!(events[3]["item"]["tools"][0]["name"], "increment");
    }

    #[test]
    fn compaction_uses_shared_gateway_event_lifecycle_without_intermediate_event() {
        let public_output = [OutputItem::Compaction(CompactionItem {
            id: Some("cmp_1".to_owned()),
            encrypted_content: "durable summary".to_owned(),
        })];
        let plans = super::compaction_event_plans(&public_output, 0);
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut stream_accumulator = crate::executor::gateway_accumulator::GatewayStreamAccumulator::new();

        super::emit_gateway_start_events(&plans, &mut stream_accumulator, &sender).expect("start events");
        super::emit_gateway_completed_events(&public_output, &plans, &mut stream_accumulator, &sender)
            .expect("completed events");

        let chunks = std::iter::from_fn(|| receiver.try_recv().ok())
            .map(|event| event.content)
            .collect::<Vec<_>>();
        let events = chunks
            .iter()
            .map(|chunk| parse_named_sse_event(chunk))
            .collect::<Vec<_>>();
        assert_eq!(
            events
                .iter()
                .map(|event| event["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["response.output_item.added", "response.output_item.done"]
        );
        assert_eq!(events[0]["item"], events[1]["item"]);
        assert_eq!(events[1]["item"]["encrypted_content"], "durable summary");

        let data_lines = chunks
            .iter()
            .filter_map(|chunk| chunk.lines().find(|line| line.starts_with("data: ")).map(str::to_owned));
        let response = ResponseAccumulator::from_sse_lines(data_lines, None)
            .expect("valid SSE stream")
            .finalize("test-model", None, None);
        assert_eq!(response.output.len(), 1);
        assert!(matches!(response.output[0], OutputItem::Compaction(_)));
    }

    #[test]
    fn mcp_gateway_events_follow_openai_lifecycle() {
        let call = FunctionToolCall {
            id: "fc_1".to_owned(),
            call_id: "call_1".to_owned(),
            name: "mcp__counter__increment".to_owned(),
            arguments: "{}".to_owned(),
            status: crate::types::event::MessageStatus::Completed,
            namespace: None,
        };
        let started = OutputItem::McpCall(crate::types::io::McpCall::new(
            "mcp_1",
            "counter",
            "increment",
            "",
            McpCallStatus::InProgress,
            None,
            None,
        ));
        let mut plans = vec![super::GatewayEventPlan {
            output_index: 0,
            started_output: Some(started),
            completed_output: None,
            arguments: Some(call.arguments.clone()),
        }];
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut stream_accumulator = crate::executor::gateway_accumulator::GatewayStreamAccumulator::new();

        super::emit_gateway_start_events(&plans, &mut stream_accumulator, &sender).expect("start events");

        let mut start_events = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            start_events.push(parse_named_sse_event(&event.content));
        }
        assert_eq!(
            start_events
                .iter()
                .map(|event| event["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec![
                "response.output_item.added",
                "response.mcp_call.in_progress",
                "response.mcp_call_arguments.delta",
                "response.mcp_call_arguments.done"
            ]
        );
        assert_eq!(start_events[0]["item"]["type"], "mcp_call");
        assert_eq!(start_events[0]["item"]["arguments"], "");
        assert_eq!(start_events[2]["delta"], "{}");
        assert_eq!(start_events[3]["arguments"], "{}");
        assert_eq!(
            start_events
                .iter()
                .map(|event| event["sequence_number"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );

        let final_item = OutputItem::McpCall(crate::types::io::McpCall::new(
            "mcp_1",
            "counter",
            "increment",
            "{}",
            McpCallStatus::Completed,
            Some("1".to_owned()),
            None,
        ));
        let results = vec![GatewayCallResult {
            item_index: 0,
            input_item: InputItem::FunctionCallOutput(
                ToolOutput {
                    call_id: "call_1".to_owned(),
                    output: "1".to_owned(),
                }
                .into(),
            ),
            public_output: Some(final_item),
        }];

        super::complete_gateway_event_plans(&mut plans, &results);
        super::emit_gateway_completed_events(&results, &plans, &mut stream_accumulator, &sender)
            .expect("completed events");

        let completed = receiver.try_recv().expect("mcp_call.completed");
        let completed = parse_named_sse_event(&completed.content);
        assert_eq!(completed["type"], "response.mcp_call.completed");
        assert_eq!(completed["sequence_number"], 4);
        assert!(completed.get("item").is_none());

        let done = receiver.try_recv().expect("output_item.done");
        let done = parse_named_sse_event(&done.content);
        assert_eq!(done["type"], "response.output_item.done");
        assert_eq!(done["sequence_number"], 5);
        assert_eq!(done["item"]["type"], "mcp_call");
        assert_eq!(done["item"]["output"], "1");
    }

    #[test]
    fn failed_mcp_gateway_events_keep_contiguous_sequence_numbers() {
        let call = FunctionToolCall {
            id: "fc_1".to_owned(),
            call_id: "call_1".to_owned(),
            name: "mcp__counter__increment".to_owned(),
            arguments: "{}".to_owned(),
            status: crate::types::event::MessageStatus::Completed,
            namespace: None,
        };
        let mut plans = vec![super::GatewayEventPlan {
            output_index: 0,
            started_output: Some(OutputItem::McpCall(crate::types::io::McpCall::new(
                "mcp_1",
                "counter",
                "increment",
                "",
                McpCallStatus::InProgress,
                None,
                None,
            ))),
            completed_output: None,
            arguments: Some(call.arguments.clone()),
        }];
        let results = vec![GatewayCallResult {
            item_index: 0,
            input_item: InputItem::FunctionCallOutput(
                ToolOutput {
                    call_id: "call_1".to_owned(),
                    output: r#"{"error":"boom"}"#.to_owned(),
                }
                .into(),
            ),
            public_output: Some(OutputItem::McpCall(crate::types::io::McpCall::new(
                "mcp_1",
                "counter",
                "increment",
                "{}",
                McpCallStatus::Failed,
                None,
                Some(crate::types::io::McpCallError::tool_execution("boom")),
            ))),
        }];
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let mut stream_accumulator = crate::executor::gateway_accumulator::GatewayStreamAccumulator::new();

        super::emit_gateway_start_events(&plans, &mut stream_accumulator, &sender).expect("start events");
        super::complete_gateway_event_plans(&mut plans, &results);
        super::emit_gateway_completed_events(&results, &plans, &mut stream_accumulator, &sender)
            .expect("failed events");

        let events = std::iter::from_fn(|| receiver.try_recv().ok())
            .map(|event| parse_named_sse_event(&event.content))
            .collect::<Vec<_>>();
        assert_eq!(
            events
                .iter()
                .map(|event| event["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec![
                "response.output_item.added",
                "response.mcp_call.in_progress",
                "response.mcp_call_arguments.delta",
                "response.mcp_call_arguments.done",
                "response.mcp_call.failed",
                "response.output_item.done",
            ]
        );
        assert_eq!(
            events
                .iter()
                .map(|event| event["sequence_number"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
    }
}
