use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures::StreamExt;
use serde::Deserialize;
use serde_json::Value;

use crate::events::{EventFrame, SSEEventType, WireEvent, normalize_sse_value};
use crate::executor::accumulator::ResponseAccumulator;
use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::function_sse::FunctionSseTranslator;
use crate::executor::gateway::{
    emit_gateway_completed_events, emit_gateway_start_events, mcp_list_tools_event_plans, public_output_items,
};
use crate::executor::gateway_accumulator::{GatewayStreamAccumulator, StreamEvent, emit_sse_frame};
use crate::executor::inference::{call_inference, fetch_response_json};
use crate::executor::request::{ExecutionContext, RequestContext};
use crate::tool::{ToolRegistry, ToolSearchHandler};
use crate::types::io::OutputItem;
use crate::types::request_response::ResponsePayload;
use crate::utils::common::{deserialize_from_str, serialize_to_string};

const MAX_DEFERRED_STREAM_BYTES: usize = 256 * 1024;

struct StreamEmitContext<'a> {
    request: &'a RequestContext,
    registry: &'a ToolRegistry,
    sender: &'a tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    accumulator: &'a mut GatewayStreamAccumulator,
    output_offset: usize,
}

pub(super) struct StreamPayload {
    pub(super) payload: ResponsePayload,
    pub(super) deferred_events: Vec<EventFrame>,
}

/// Builds the JSON body sent upstream: history inlined, continuation and storage
/// fields removed.
///
/// # Errors
/// A tool-configuration or serialization failure.
pub fn upstream_request(ctx: &RequestContext, stream: bool) -> ExecutorResult<String> {
    let request = ctx.enriched_request.to_upstream_request(stream)?;
    serialize_to_string(&request).map_err(ExecutorError::JsonError)
}

pub(super) async fn fetch_blocking_payload(
    ctx: &RequestContext,
    exec_ctx: &ExecutionContext,
    auth: Option<&str>,
    registry: &ToolRegistry,
) -> ExecutorResult<ResponsePayload> {
    let url = exec_ctx.responses_url();
    registry.ensure_request_prepared(&ctx.enriched_request)?;
    // Non-streaming request: stream=false -> full JSON body -> from_json.
    let upstream_json = upstream_request(ctx, false)?;

    let body = fetch_response_json(upstream_json, &url, &exec_ctx.client, auth).await?;
    registry.validate_blocking_response(&body)?;
    let mut payload = payload_from_upstream(ctx, UpstreamBody::Json(&body))?;
    let status = payload.status.parse().unwrap_or_default();
    ToolSearchHandler::normalize_response_output(registry, &mut payload.output, status, &HashSet::new())?;
    Ok(payload)
}

/// A complete upstream response, in whichever form the caller received it.
#[derive(Debug, Clone, Copy)]
pub enum UpstreamBody<'a> {
    Json(&'a str),
    /// Frames of a streamed response, already relayed by the caller.
    Sse(&'a str),
}

fn absorb_line(acc: &mut ResponseAccumulator, ctx: &RequestContext, line: &str) -> bool {
    if let Some(frame) = acc.process_sse_line(line) {
        log_upstream_failure(&frame, &ctx.response_id);
        return true;
    }
    // Only a `data:` line that produced no frame is malformed.
    !is_data_frame(line)
}

/// A `data:` payload the accumulator should have understood; `[DONE]` carries none.
fn is_data_frame(line: &str) -> bool {
    line.strip_prefix("data:")
        .map(str::trim)
        .is_some_and(|payload| !payload.is_empty() && payload != "[DONE]")
}

#[derive(Default)]
enum ResponseLifecycle {
    #[default]
    AwaitingCreated,
    Created(String),
    InProgress(String),
    Terminal,
}

#[derive(Default)]
struct RelayedStreamValidator {
    active_items: HashMap<u32, ActiveItem>,
    completed_items: HashMap<u32, ActiveItem>,
    seen_item_ids: HashSet<String>,
    seen_output_indexes: HashSet<u32>,
    lifecycle: ResponseLifecycle,
}

struct ActiveItem {
    id: String,
    item_type: String,
}

impl RelayedStreamValidator {
    fn validate_line(&mut self, line: &str) -> ExecutorResult<Option<EventFrame>> {
        let Some(payload) = line.strip_prefix("data:").map(str::trim) else {
            return Ok(None);
        };
        if payload.is_empty() || payload == "[DONE]" {
            return Ok(None);
        }

        let event: Value = deserialize_from_str(payload).map_err(ExecutorError::JsonError)?;
        let event_name = required_str(&event, "type", "streaming event")?.to_owned();
        if matches!(self.lifecycle, ResponseLifecycle::Terminal) {
            return Err(ExecutorError::InvalidRequest(
                "upstream stream contains an event after its terminal event".to_owned(),
            ));
        }

        let event_type = SSEEventType::from(event_name.as_str());
        match event_type {
            SSEEventType::ResponseCreated
            | SSEEventType::ResponseInProgress
            | SSEEventType::ResponseCompleted
            | SSEEventType::ResponseFailed
            | SSEEventType::ResponseIncomplete => self.validate_response_event(&event, event_type, &event_name)?,
            SSEEventType::OutputItemAdded => {
                self.require_in_progress(&event_name)?;
                let output_index = required_u32(&event, "output_index", &event_name)?;
                let item = event.get("item").ok_or_else(|| missing_field(&event_name, "item"))?;
                let item_id = required_str(item, "id", "output item")?;
                let item_type = required_str(item, "type", "output item")?;
                ensure_supported_output_item_type(item_type)?;
                if self.seen_output_indexes.contains(&output_index) || self.seen_item_ids.contains(item_id) {
                    return Err(ExecutorError::InvalidRequest(format!(
                        "upstream stream repeats output item '{item_id}'"
                    )));
                }
                self.seen_output_indexes.insert(output_index);
                self.seen_item_ids.insert(item_id.to_owned());
                self.active_items.insert(
                    output_index,
                    ActiveItem {
                        id: item_id.to_owned(),
                        item_type: item_type.to_owned(),
                    },
                );
            }
            SSEEventType::OutputItemDone => {
                self.require_in_progress(&event_name)?;
                let output_index = required_u32(&event, "output_index", &event_name)?;
                let item = event.get("item").ok_or_else(|| missing_field(&event_name, "item"))?;
                let item_id = required_str(item, "id", "output item")?;
                let item_type = required_str(item, "type", "output item")?;
                ensure_supported_output_item_type(item_type)?;
                OutputItem::deserialize(item).map_err(|error| {
                    ExecutorError::InvalidRequest(format!("upstream stream output item is invalid: {error}"))
                })?;
                self.finish_item(output_index, item_id, item_type, &event_name)?;
            }
            SSEEventType::Other => self.require_in_progress(&event_name)?,
            event_type => {
                self.require_in_progress(&event_name)?;
                let output_index = required_u32(&event, "output_index", &event_name)?;
                let item_id = required_str(&event, "item_id", &event_name)?;
                self.require_active_item(output_index, item_id, expected_item_type(event_type), &event_name)?;
                validate_event_fields(&event, event_type, &event_name)?;
            }
        }
        normalize_sse_value(event).map(Some).ok_or_else(|| {
            ExecutorError::InvalidRequest(format!("upstream stream event '{event_name}' could not be normalized"))
        })
    }

    fn require_in_progress(&self, event_name: &str) -> ExecutorResult<()> {
        if matches!(self.lifecycle, ResponseLifecycle::InProgress(_)) {
            return Ok(());
        }
        Err(ExecutorError::InvalidRequest(format!(
            "upstream stream event '{event_name}' is out of lifecycle order"
        )))
    }

    fn validate_response_event(
        &mut self,
        event: &Value,
        event_type: SSEEventType,
        event_name: &str,
    ) -> ExecutorResult<()> {
        let response = event
            .get("response")
            .ok_or_else(|| missing_field(event_name, "response"))?;
        let response_id = required_str(response, "id", "upstream response")?;
        let status = required_str(response, "status", "upstream response")?;
        let expected_status = match event_type {
            SSEEventType::ResponseCreated | SSEEventType::ResponseInProgress => "in_progress",
            SSEEventType::ResponseCompleted => "completed",
            SSEEventType::ResponseFailed => "failed",
            SSEEventType::ResponseIncomplete => "incomplete",
            _ => return Ok(()),
        };
        if status != expected_status {
            return Err(ExecutorError::InvalidRequest(format!(
                "upstream stream event '{event_name}' has status '{status}', expected '{expected_status}'"
            )));
        }

        match (&self.lifecycle, event_type) {
            (ResponseLifecycle::AwaitingCreated, SSEEventType::ResponseCreated) => {
                self.lifecycle = ResponseLifecycle::Created(response_id.to_owned());
            }
            (ResponseLifecycle::Created(created_id), SSEEventType::ResponseInProgress) if created_id == response_id => {
                self.lifecycle = ResponseLifecycle::InProgress(response_id.to_owned());
            }
            (
                ResponseLifecycle::InProgress(in_progress_id),
                SSEEventType::ResponseCompleted | SSEEventType::ResponseFailed | SSEEventType::ResponseIncomplete,
            ) if in_progress_id == response_id => {
                if !self.active_items.is_empty() {
                    return Err(ExecutorError::InvalidRequest(
                        "upstream stream ended with unfinished output items".to_owned(),
                    ));
                }
                self.validate_terminal_output(response)?;
                self.lifecycle = ResponseLifecycle::Terminal;
            }
            _ => {
                return Err(ExecutorError::InvalidRequest(format!(
                    "upstream stream event '{event_name}' is out of lifecycle order or changes the response id"
                )));
            }
        }
        Ok(())
    }

    fn require_active_item(
        &self,
        output_index: u32,
        item_id: &str,
        expected_type: &str,
        event_name: &str,
    ) -> ExecutorResult<()> {
        let Some(active) = self.active_items.get(&output_index) else {
            return Err(ExecutorError::InvalidRequest(format!(
                "upstream stream event '{event_name}' has no active output item"
            )));
        };
        if item_id != active.id || expected_type != active.item_type {
            return Err(ExecutorError::InvalidRequest(format!(
                "upstream stream event '{event_name}' does not match its active output item"
            )));
        }
        Ok(())
    }

    fn finish_item(
        &mut self,
        output_index: u32,
        item_id: &str,
        item_type: &str,
        event_name: &str,
    ) -> ExecutorResult<()> {
        self.require_active_item(output_index, item_id, item_type, event_name)?;
        let item = self.active_items.remove(&output_index).ok_or_else(|| {
            ExecutorError::InvalidRequest(format!(
                "upstream stream event '{event_name}' has no active output item"
            ))
        })?;
        self.completed_items.insert(output_index, item);
        Ok(())
    }

    fn validate_terminal_output(&self, response: &Value) -> ExecutorResult<()> {
        let output = response
            .get("output")
            .and_then(Value::as_array)
            .ok_or_else(|| missing_field("terminal upstream response", "output"))?;
        if output.len() != self.completed_items.len() {
            return Err(ExecutorError::InvalidRequest(
                "terminal upstream response output does not match completed item events".to_owned(),
            ));
        }
        for (index, item) in output.iter().enumerate() {
            let output_index = u32::try_from(index).map_err(|_| {
                ExecutorError::InvalidRequest("terminal upstream response has too many output items".to_owned())
            })?;
            let item_id = required_str(item, "id", "terminal output item")?;
            let item_type = required_str(item, "type", "terminal output item")?;
            ensure_supported_output_item_type(item_type)?;
            let Some(completed) = self.completed_items.get(&output_index) else {
                return Err(ExecutorError::InvalidRequest(
                    "terminal upstream response output does not match completed item events".to_owned(),
                ));
            };
            if item_id != completed.id || item_type != completed.item_type {
                return Err(ExecutorError::InvalidRequest(
                    "terminal upstream response output does not match completed item events".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

fn ensure_supported_output_item_type(item_type: &str) -> ExecutorResult<()> {
    if matches!(
        item_type,
        "message"
            | "function_call"
            | "tool_search_call"
            | "custom_tool_call"
            | "web_search_call"
            | "mcp_call"
            | "mcp_list_tools"
            | "reasoning"
            | "compaction"
    ) {
        return Ok(());
    }
    Err(ExecutorError::InvalidRequest(format!(
        "upstream output item type '{item_type}' is unsupported"
    )))
}

fn expected_item_type(event_type: SSEEventType) -> &'static str {
    match event_type {
        SSEEventType::OutputTextDelta
        | SSEEventType::OutputTextDone
        | SSEEventType::ContentPartAdded
        | SSEEventType::ContentPartDone => "message",
        SSEEventType::FunctionCallArgumentsDelta | SSEEventType::FunctionCallArgumentsDone => "function_call",
        SSEEventType::CustomToolCallInputDelta | SSEEventType::CustomToolCallInputDone => "custom_tool_call",
        SSEEventType::ReasoningTextDelta
        | SSEEventType::ReasoningTextDone
        | SSEEventType::ReasoningPartAdded
        | SSEEventType::ReasoningPartDone
        | SSEEventType::ReasoningSummaryTextDelta
        | SSEEventType::ReasoningSummaryTextDone => "reasoning",
        SSEEventType::FileSearchCallSearching | SSEEventType::FileSearchCallCompleted => "file_search_call",
        SSEEventType::WebSearchCallInProgress
        | SSEEventType::WebSearchCallSearching
        | SSEEventType::WebSearchCallCompleted => "web_search_call",
        SSEEventType::McpCallInProgress
        | SSEEventType::McpCallArgumentsDelta
        | SSEEventType::McpCallArgumentsDone
        | SSEEventType::McpCallCompleted
        | SSEEventType::McpCallFailed => "mcp_call",
        SSEEventType::McpListToolsInProgress
        | SSEEventType::McpListToolsCompleted
        | SSEEventType::McpListToolsFailed => "mcp_list_tools",
        SSEEventType::ResponseCreated
        | SSEEventType::ResponseInProgress
        | SSEEventType::ResponseCompleted
        | SSEEventType::ResponseFailed
        | SSEEventType::ResponseIncomplete
        | SSEEventType::OutputItemAdded
        | SSEEventType::OutputItemDone
        | SSEEventType::Other => unreachable!("only item events are classified"),
    }
}

fn validate_event_fields(event: &Value, event_type: SSEEventType, event_name: &str) -> ExecutorResult<()> {
    match event_type {
        SSEEventType::OutputTextDelta
        | SSEEventType::OutputTextDone
        | SSEEventType::ContentPartAdded
        | SSEEventType::ContentPartDone
        | SSEEventType::ReasoningTextDelta
        | SSEEventType::ReasoningTextDone
        | SSEEventType::ReasoningPartAdded
        | SSEEventType::ReasoningPartDone => {
            let _ = required_u32(event, "content_index", event_name)?;
        }
        SSEEventType::ReasoningSummaryTextDelta | SSEEventType::ReasoningSummaryTextDone => {
            let _ = required_u32(event, "summary_index", event_name)?;
        }
        _ => {}
    }
    let required = match event_type {
        SSEEventType::OutputTextDelta
        | SSEEventType::FunctionCallArgumentsDelta
        | SSEEventType::CustomToolCallInputDelta
        | SSEEventType::ReasoningTextDelta
        | SSEEventType::ReasoningSummaryTextDelta
        | SSEEventType::McpCallArgumentsDelta => Some("delta"),
        SSEEventType::OutputTextDone | SSEEventType::ReasoningTextDone | SSEEventType::ReasoningSummaryTextDone => {
            Some("text")
        }
        SSEEventType::FunctionCallArgumentsDone => {
            let _ = required_str(event, "name", event_name)?;
            Some("arguments")
        }
        SSEEventType::McpCallArgumentsDone => Some("arguments"),
        SSEEventType::CustomToolCallInputDone => Some("input"),
        SSEEventType::ContentPartAdded
        | SSEEventType::ContentPartDone
        | SSEEventType::ReasoningPartAdded
        | SSEEventType::ReasoningPartDone => {
            let _ = required_object(event, "part", event_name)?;
            None
        }
        SSEEventType::ResponseCreated
        | SSEEventType::ResponseInProgress
        | SSEEventType::ResponseCompleted
        | SSEEventType::ResponseFailed
        | SSEEventType::ResponseIncomplete
        | SSEEventType::OutputItemAdded
        | SSEEventType::OutputItemDone
        | SSEEventType::FileSearchCallSearching
        | SSEEventType::FileSearchCallCompleted
        | SSEEventType::WebSearchCallInProgress
        | SSEEventType::WebSearchCallSearching
        | SSEEventType::WebSearchCallCompleted
        | SSEEventType::McpCallInProgress
        | SSEEventType::McpCallCompleted
        | SSEEventType::McpCallFailed
        | SSEEventType::McpListToolsInProgress
        | SSEEventType::McpListToolsCompleted
        | SSEEventType::McpListToolsFailed
        | SSEEventType::Other => None,
    };
    if let Some(field) = required {
        let _ = required_string(event, field, event_name)?;
    }
    Ok(())
}

fn required_str<'a>(value: &'a Value, field: &str, owner: &str) -> ExecutorResult<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| missing_field(owner, field))
}

fn required_string<'a>(value: &'a Value, field: &str, owner: &str) -> ExecutorResult<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| missing_field(owner, field))
}

fn required_object<'a>(
    value: &'a Value,
    field: &str,
    owner: &str,
) -> ExecutorResult<&'a serde_json::Map<String, Value>> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| missing_field(owner, field))
}

fn required_u32(value: &Value, field: &str, owner: &str) -> ExecutorResult<u32> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| missing_field(owner, field))
}

fn missing_field(owner: &str, field: &str) -> ExecutorError {
    ExecutorError::InvalidRequest(format!("{owner} has no valid '{field}'"))
}

/// Rejects a body [`ResponseAccumulator::from_json`] would accept too generously:
/// it defaults a missing `status` to `completed` and drops unreadable items, which
/// is safe for our own fetch but not for a body an outside caller supplied.
///
/// # Errors
/// [`ExecutorError::InvalidRequest`] naming the field that is missing or invalid.
pub fn ensure_strict_response(body: &str) -> ExecutorResult<()> {
    let json: Value = deserialize_from_str(body).map_err(ExecutorError::JsonError)?;
    let Some(status) = json["status"].as_str() else {
        return Err(ExecutorError::InvalidRequest(
            "upstream response has no 'status'".to_owned(),
        ));
    };
    if !matches!(status, "completed" | "failed" | "incomplete") {
        return Err(ExecutorError::InvalidRequest(format!(
            "upstream response status '{status}' is not terminal"
        )));
    }
    let Some(items) = json["output"].as_array() else {
        return Err(ExecutorError::InvalidRequest(
            "upstream response has no 'output' array".to_owned(),
        ));
    };
    let mut item_ids = HashSet::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let owner = format!("upstream response output[{index}]");
        let item_id = required_str(item, "id", &owner)?;
        let item_type = required_str(item, "type", &owner)?;
        ensure_supported_output_item_type(item_type)?;
        OutputItem::deserialize(item).map_err(|error| {
            ExecutorError::InvalidRequest(format!(
                "upstream response output[{index}] is not a valid item: {error}"
            ))
        })?;
        if !item_ids.insert(item_id) {
            return Err(ExecutorError::InvalidRequest(format!(
                "upstream response repeats output item '{item_id}'"
            )));
        }
    }
    Ok(())
}

/// Decodes a complete upstream response, checking a caller-supplied body first.
///
/// # Errors
/// [`ExecutorError::InvalidRequest`] for an incomplete response, or a parse error.
pub fn decode_upstream(ctx: &RequestContext, upstream: UpstreamBody<'_>) -> ExecutorResult<ResponsePayload> {
    if let UpstreamBody::Json(body) = upstream {
        ensure_strict_response(body)?;
    }
    payload_from_upstream(ctx, upstream)
}

pub(super) fn payload_from_upstream(
    ctx: &RequestContext,
    upstream: UpstreamBody<'_>,
) -> ExecutorResult<ResponsePayload> {
    let acc = match upstream {
        UpstreamBody::Json(body) => ResponseAccumulator::from_json(body, ctx.conversation_id.as_deref())?,
        UpstreamBody::Sse(sse) => {
            let mut acc = ResponseAccumulator::new(ctx.response_id.clone(), ctx.conversation_id.clone());
            let mut validator = RelayedStreamValidator::default();
            for line in sse.lines() {
                if let Some(frame) = validator.validate_line(line)? {
                    acc.process_normalized_event(&frame);
                    log_upstream_failure(&frame, &ctx.response_id);
                }
            }
            if !acc.saw_terminal_frame() {
                return Err(ExecutorError::InvalidRequest(
                    "upstream stream ended without a terminal event".to_owned(),
                ));
            }
            acc.finish_stream();
            acc
        }
    };
    Ok(finalize_payload(ctx, acc))
}

/// The tail both legs share: request-derived fields in, our ids stamped on.
fn finalize_payload(ctx: &RequestContext, acc: ResponseAccumulator) -> ResponsePayload {
    let mut payload = acc.finalize(
        &ctx.enriched_request.model,
        ctx.original_request.previous_response_id.as_deref(),
        ctx.original_request.instructions.as_deref(),
    );
    ctx.inject_ids(&mut payload);
    payload
}

pub(super) async fn fetch_stream_payload(
    ctx: &RequestContext,
    exec_ctx: &ExecutionContext,
    auth: Option<&str>,
    registry: &ToolRegistry,
    mut stream: Option<(
        &mut GatewayStreamAccumulator,
        &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    )>,
    output_offset: usize,
) -> ExecutorResult<StreamPayload> {
    let url = exec_ctx.responses_url();
    registry.ensure_request_prepared(&ctx.enriched_request)?;
    let upstream_json = upstream_request(ctx, true)?;
    let mut line_stream = Box::pin(call_inference(
        upstream_json,
        url,
        Arc::clone(&exec_ctx.client),
        auth.map(str::to_owned),
        exec_ctx.streaming_timeout,
    ));
    let mut acc = ResponseAccumulator::new(ctx.response_id.clone(), ctx.conversation_id.clone());
    let mut function_sse = FunctionSseTranslator::new(registry);
    let mut defer_from_output_index = None;
    let mut deferred_events = Vec::new();
    let mut deferred_bytes = 0;
    while let Some(line_result) = line_stream.next().await {
        let line = line_result?;
        if stream.is_none() && !registry.tool_search_is_active() {
            let _ = absorb_line(&mut acc, ctx, &line);
            continue;
        }
        if let Some(translation) = acc.process_sse_line_with_translator(&line, &mut function_sse)? {
            let previous_defer_from_output_index = defer_from_output_index;
            defer_from_output_index = translation.defer_from_output_index.map(u64::from);
            for frame in &translation.frames {
                log_upstream_failure(frame, &ctx.response_id);
            }
            if let Some((accumulator, sender)) = stream.as_mut() {
                let mut emit_ctx = StreamEmitContext {
                    request: ctx,
                    registry,
                    sender,
                    accumulator,
                    output_offset,
                };
                for frame in translation.frames {
                    if !is_terminal_response_event(frame.event_type) {
                        let event_type = frame.event_type;
                        let emitted = emit_or_defer_stream_frame(
                            frame,
                            &mut emit_ctx,
                            defer_from_output_index,
                            &mut deferred_events,
                            &mut deferred_bytes,
                        )?;
                        if event_type == SSEEventType::ResponseInProgress && emitted {
                            emit_mcp_discovery_lifecycle(registry, emit_ctx.accumulator, emit_ctx.sender)?;
                        }
                    }
                }
                if defer_from_output_index != previous_defer_from_output_index {
                    flush_released_stream_frames(
                        &mut emit_ctx,
                        defer_from_output_index,
                        &mut deferred_events,
                        &mut deferred_bytes,
                    )?;
                }
            }
        }
    }
    let function_sse_outcome = function_sse.finish()?;
    acc.finish_stream();
    let mut payload = finalize_payload(ctx, acc);
    let status = payload.status.parse().unwrap_or_default();
    ToolSearchHandler::normalize_response_output(
        registry,
        &mut payload.output,
        status,
        &function_sse_outcome.unfinished_tool_search_item_ids,
    )?;
    Ok(StreamPayload {
        payload,
        deferred_events,
    })
}

fn log_upstream_failure(frame: &EventFrame, gateway_response_id: &str) {
    if frame.event_type != SSEEventType::ResponseFailed {
        return;
    }

    let response = frame.wire.rest.get("response").unwrap_or(&Value::Null);
    let error = &response["error"];
    let error_code = error.get("code").and_then(Value::as_str).unwrap_or_default();
    let error_message = error
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| error.as_str())
        .unwrap_or_default();
    let incomplete_reason = response["incomplete_details"]
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or_default();

    tracing::warn!(
        response_id = %gateway_response_id,
        upstream_response_id = response["id"].as_str().unwrap_or_default(),
        error_code,
        error_message,
        incomplete_reason,
        "upstream response failed"
    );
}

pub(super) fn emit_deferred_stream_events(
    deferred_events: Vec<EventFrame>,
    request: &RequestContext,
    registry: &ToolRegistry,
    accumulator: &mut GatewayStreamAccumulator,
    sender: &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    output_offset: usize,
) -> ExecutorResult<()> {
    let mut emit_ctx = StreamEmitContext {
        request,
        registry,
        sender,
        accumulator,
        output_offset,
    };
    for mut frame in deferred_events {
        emit_stream_frame(&mut frame, &mut emit_ctx)?;
    }
    Ok(())
}

fn should_defer_stream_event(frame: &EventFrame, defer_from_output_index: Option<u64>) -> bool {
    defer_from_output_index.is_some_and(|first_hidden_index| {
        frame
            .wire
            .output_index
            .is_some_and(|output_index| output_index >= first_hidden_index)
    })
}

fn emit_stream_frame(frame: &mut EventFrame, emit_ctx: &mut StreamEmitContext<'_>) -> ExecutorResult<bool> {
    apply_context_response_ids(&mut frame.wire, emit_ctx.request);
    emit_ctx.registry.restore_tool_search_response_tools(&mut frame.wire)?;
    emit_ctx.registry.restore_stream_event_wire(&mut frame.wire);
    let emitted = emit_ctx.accumulator.process_event(frame, emit_ctx.output_offset);
    if emitted {
        emit_sse_frame(emit_ctx.sender, frame)?;
    }
    Ok(emitted)
}

fn emit_or_defer_stream_frame(
    mut frame: EventFrame,
    emit_ctx: &mut StreamEmitContext<'_>,
    defer_from_output_index: Option<u64>,
    deferred_events: &mut Vec<EventFrame>,
    deferred_bytes: &mut usize,
) -> ExecutorResult<bool> {
    if should_defer_stream_event(&frame, defer_from_output_index) {
        let frame_bytes = serialize_to_string(&frame.wire)
            .map_err(ExecutorError::JsonError)?
            .len();
        let next_bytes = deferred_bytes.saturating_add(frame_bytes);
        if next_bytes > MAX_DEFERRED_STREAM_BYTES {
            return Err(ExecutorError::StreamError(format!(
                "deferred stream exceeded {MAX_DEFERRED_STREAM_BYTES} buffered bytes"
            )));
        }
        deferred_events.push(frame);
        *deferred_bytes = next_bytes;
        return Ok(false);
    }
    emit_stream_frame(&mut frame, emit_ctx)
}

fn flush_released_stream_frames(
    emit_ctx: &mut StreamEmitContext<'_>,
    defer_from_output_index: Option<u64>,
    deferred_events: &mut Vec<EventFrame>,
    deferred_bytes: &mut usize,
) -> ExecutorResult<()> {
    let mut pending = std::mem::take(deferred_events);
    *deferred_bytes = 0;
    pending.sort_by_key(|frame| frame.wire.output_index);
    for frame in pending {
        emit_or_defer_stream_frame(
            frame,
            emit_ctx,
            defer_from_output_index,
            deferred_events,
            deferred_bytes,
        )?;
    }
    Ok(())
}

fn emit_mcp_discovery_lifecycle(
    registry: &ToolRegistry,
    stream_accumulator: &mut GatewayStreamAccumulator,
    stream_sender: &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
) -> ExecutorResult<()> {
    let discovered_output = registry
        .mcp_list_tool_items()
        .map(crate::tool::mcp::handler::list_tools_output_item)
        .collect::<Vec<_>>();
    let public_output = public_output_items(&discovered_output, registry, &[]);
    let event_plans = mcp_list_tools_event_plans(&public_output, 0);

    emit_gateway_start_events(&event_plans, stream_accumulator, stream_sender)?;
    emit_gateway_completed_events(&public_output, &event_plans, stream_accumulator, stream_sender)
}

fn is_terminal_response_event(event_type: SSEEventType) -> bool {
    matches!(
        event_type,
        SSEEventType::ResponseCompleted | SSEEventType::ResponseFailed | SSEEventType::ResponseIncomplete
    )
}

fn apply_context_response_ids(wire: &mut WireEvent, ctx: &RequestContext) {
    let Some(response) = wire.rest.get_mut("response").and_then(Value::as_object_mut) else {
        return;
    };
    response.insert("id".to_owned(), Value::String(ctx.response_id.clone()));
    if let Some(previous_response_id) = &ctx.original_request.previous_response_id {
        response.insert(
            "previous_response_id".to_owned(),
            Value::String(previous_response_id.clone()),
        );
    }
    if let Some(conversation_id) = &ctx.conversation_id {
        response.insert("conversation_id".to_owned(), Value::String(conversation_id.clone()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventPayload;
    use crate::types::io::ResponsesInput;
    use crate::types::request_response::RequestPayload;

    fn request_context() -> RequestContext {
        let request = RequestPayload {
            model: "test".to_owned(),
            input: ResponsesInput::Text("hi".to_owned()),
            instructions: None,
            previous_response_id: None,
            conversation_id: None,
            tools: None,
            tool_choice: None,
            stream: true,
            store: false,
            include: None,
            reasoning: None,
            text: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            truncation: None,
            metadata: None,
            parallel_tool_calls: None,
            cache_salt: None,
            context_management: None,
        };
        RequestContext {
            original_request: request.clone(),
            enriched_request: request,
            new_input_items: Vec::new(),
            response_id: "resp_test".to_owned(),
            conversation_id: None,
            conversation_version: None,
        }
    }

    fn frame(output_index: u64, payload: Value) -> EventFrame {
        let mut wire = WireEvent::new("response.output_item.added");
        wire.output_index = Some(output_index);
        wire.rest.insert("item".to_owned(), payload);
        EventFrame {
            event_type: SSEEventType::OutputItemAdded,
            payload: EventPayload::None,
            wire,
        }
    }

    #[test]
    fn released_frames_are_emitted_in_output_index_order() {
        let request = request_context();
        let registry = ToolRegistry::default();
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let mut accumulator = GatewayStreamAccumulator::new();
        let mut emit_ctx = StreamEmitContext {
            request: &request,
            registry: &registry,
            sender: &sender,
            accumulator: &mut accumulator,
            output_offset: 0,
        };
        let mut deferred = vec![
            frame(3, serde_json::json!({"id": "msg_3"})),
            frame(2, serde_json::json!({"id": "msg_2"})),
        ];
        let mut deferred_bytes = deferred
            .iter()
            .map(|frame| serialize_to_string(&frame.wire).unwrap().len())
            .sum();

        flush_released_stream_frames(&mut emit_ctx, None, &mut deferred, &mut deferred_bytes).expect("flush succeeds");
        assert_eq!(deferred_bytes, 0);

        let indices = [receiver.try_recv().unwrap(), receiver.try_recv().unwrap()].map(|event| {
            let data_line = event
                .content
                .lines()
                .find(|line| line.starts_with("data: "))
                .expect("SSE data line");
            crate::events::normalize_sse_line(data_line)
                .and_then(|frame| frame.wire.output_index)
                .expect("output index")
        });
        assert_eq!(indices, [2, 3]);
    }

    #[test]
    fn deferred_frames_have_a_shared_byte_limit() {
        let request = request_context();
        let registry = ToolRegistry::default();
        let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
        let mut accumulator = GatewayStreamAccumulator::new();
        let mut emit_ctx = StreamEmitContext {
            request: &request,
            registry: &registry,
            sender: &sender,
            accumulator: &mut accumulator,
            output_offset: 0,
        };
        let mut deferred = Vec::new();
        let mut deferred_bytes = 0;
        let oversized = frame(0, Value::String("x".repeat(256 * 1024 + 1)));

        let error = emit_or_defer_stream_frame(oversized, &mut emit_ctx, Some(0), &mut deferred, &mut deferred_bytes)
            .expect_err("oversized deferred stream must fail");
        assert!(error.to_string().contains("deferred stream exceeded"));
    }
}
