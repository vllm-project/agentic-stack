//! Response accumulation and parsing utilities.
//!
//! Handles both streaming (SSE) and non-streaming JSON response formats,
//! accumulating chunks into a unified `ResponsePayload` structure.
//!
//! Streaming path uses a channel + `spawn_blocking` so that SSE JSON parsing
//! runs on a blocking thread while the async task continues reading from the
//! network — keeping the tokio executor thread free between chunk arrivals.

use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::mpsc;

use indexmap::IndexMap;

use futures::{Stream, StreamExt};

use crate::events::{
    EventFrame, EventPayload, SSEEventType, SSEItemType, ValidatedFrame, is_data_frame, normalize_sse_line,
    output_item_identity, validate_frame,
};
use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::function_sse::{FunctionSseTranslation, FunctionSseTranslator};
use crate::types::event::{MessageStatus, ResponseStatus};
use crate::types::io::output::McpListTools;
use crate::types::io::{
    ApplyDone, CompactionItem, CustomToolCall, FunctionToolCall, OutputItem, OutputMessage, OutputTextContent,
    ReasoningOutput, ResponseUsage,
};
use crate::types::io::{McpCall, WebSearchCall};
use crate::types::request_response::{IncompleteDetails, ResponsePayload};
use crate::utils::common::{deserialize_from_str, deserialize_from_value_opt};
use crate::utils::uuid7_str;

/// Tracks a single output item currently being streamed, together with its
/// accumulated text/arguments buffer.
#[derive(Clone)]
enum InFlight {
    Message { item: OutputMessage, text: String },
    Reasoning { item: ReasoningOutput },
    FunctionCall { item: FunctionToolCall, arguments: String },
    CustomToolCall { item: CustomToolCall, input: String },
    WebSearchCall { item: Option<WebSearchCall> },
    McpCall { item: McpCall },
    McpListTools { item: McpListTools },
    Compaction { item: CompactionItem },
}

impl std::fmt::Debug for InFlight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Message { .. } => write!(f, "InFlight::Message {{ .. }}"),
            Self::Reasoning { .. } => write!(f, "InFlight::Reasoning {{ .. }}"),
            Self::FunctionCall { .. } => write!(f, "InFlight::FunctionCall {{ .. }}"),
            Self::CustomToolCall { .. } => write!(f, "InFlight::CustomToolCall {{ .. }}"),
            Self::WebSearchCall { .. } => write!(f, "InFlight::WebSearchCall {{ .. }}"),
            Self::McpCall { .. } => write!(f, "InFlight::McpCall {{ .. }}"),
            Self::McpListTools { .. } => write!(f, "InFlight::McpListTools {{ .. }}"),
            Self::Compaction { .. } => write!(f, "InFlight::Compaction {{ .. }}"),
        }
    }
}

impl InFlight {
    fn finalize(self) -> Option<OutputItem> {
        match self {
            Self::Reasoning { item } => Some(OutputItem::Reasoning(item)),
            Self::FunctionCall { mut item, arguments } => {
                if !arguments.is_empty() && item.arguments.is_empty() {
                    item.arguments = arguments;
                }
                item.status = MessageStatus::Completed;
                Some(OutputItem::FunctionCall(item))
            }
            Self::Message { mut item, text } => {
                if !text.is_empty() {
                    item.content.push(OutputTextContent::new(text));
                }
                item.status = MessageStatus::Completed;
                Some(OutputItem::Message(item))
            }
            Self::CustomToolCall { mut item, input } => {
                if item.input.is_empty() {
                    item.input = input;
                }
                item.status = Some(MessageStatus::Completed);
                Some(OutputItem::CustomToolCall(item))
            }
            Self::WebSearchCall { item } => item.map(OutputItem::WebSearchCall),
            Self::McpCall { item } => Some(OutputItem::McpCall(item)),
            Self::McpListTools { item } => Some(OutputItem::McpListTools(item)),
            Self::Compaction { item } => Some(OutputItem::Compaction(item)),
        }
    }
}

#[derive(Debug)]
struct InFlightEntry {
    output_index: u32,
    item_id: String,
    item_type: SSEItemType,
    item: InFlight,
    done_item: Option<OutputItem>,
}

#[derive(Debug, Default)]
struct CallIdObservation {
    first: Option<String>,
    changed: bool,
}

impl CallIdObservation {
    fn observe(&mut self, call_id: Option<&str>) {
        let Some(call_id) = call_id.filter(|call_id| !call_id.is_empty()) else {
            return;
        };
        match self.first.as_deref() {
            Some(first) if first != call_id => self.changed = true,
            None => self.first = Some(call_id.to_owned()),
            Some(_) => {}
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum StreamLifecycle {
    #[default]
    AwaitingCreated,
    Created,
    InProgress,
    Terminal,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResolvePolicy {
    Lenient,
    Authoritative,
    Strict,
}

#[derive(Clone, Copy)]
pub(super) struct AccumulatedFunctionCall<'a> {
    pub(super) item: &'a FunctionToolCall,
    pub(super) output_index: u32,
    arguments: &'a str,
}

impl AccumulatedFunctionCall<'_> {
    pub(super) fn arguments(&self) -> &str {
        if self.item.arguments.is_empty() {
            self.arguments
        } else {
            &self.item.arguments
        }
    }
}

/// Accumulates LLM response chunks from streaming or non-streaming sources.
#[derive(Debug)]
pub struct ResponseAccumulator {
    response_id: String,
    conversation_id: Option<String>,
    output: Vec<OutputItem>,
    usage: Option<ResponseUsage>,
    status: ResponseStatus,
    incomplete_details: Option<IncompleteDetails>,
    error: Option<serde_json::Value>,
    /// In-flight output items keyed by `item_id` or an internal fallback key.
    in_flight: IndexMap<String, InFlightEntry>,
    /// Explicit wire output indexes mapped to their in-flight key.
    in_flight_indexes: HashMap<u32, String>,
    /// Completed streaming items waiting to be emitted in `output_index` order.
    completed: Vec<(u32, OutputItem)>,
    strict_call_ids: HashMap<u32, CallIdObservation>,
    stream_lifecycle: StreamLifecycle,
}

impl ResponseAccumulator {
    /// Creates a new response accumulator.
    #[must_use]
    pub fn new(response_id: String, conversation_id: Option<String>) -> Self {
        Self {
            response_id,
            conversation_id,
            output: Vec::new(),
            usage: None,
            status: ResponseStatus::InProgress,
            incomplete_details: None,
            error: None,
            in_flight: IndexMap::new(),
            in_flight_indexes: HashMap::new(),
            completed: Vec::new(),
            strict_call_ids: HashMap::new(),
            stream_lifecycle: StreamLifecycle::AwaitingCreated,
        }
    }

    /// Parses a non-streaming JSON response body.
    ///
    /// # Errors
    /// Returns `ExecutorError::ParseError` if JSON parsing fails or required fields are missing.
    pub fn from_json(body: &str, conversation_id: Option<&str>) -> ExecutorResult<Self> {
        let mut json: serde_json::Value = deserialize_from_str(body).map_err(ExecutorError::JsonError)?;

        let response_id = json["id"]
            .as_str()
            .ok_or_else(|| ExecutorError::ParseError("missing 'id' field in response".into()))?
            .to_string();

        let output = deserialize_from_value_opt::<Vec<serde_json::Value>>(json["output"].take())
            .map(|items| {
                let mut out = Vec::with_capacity(items.len());
                out.extend(items.into_iter().filter_map(deserialize_from_value_opt::<OutputItem>));
                out
            })
            .unwrap_or_default();

        let status = json["status"]
            .as_str()
            .map_or(ResponseStatus::Completed, |s| s.parse().unwrap_or_default());

        let usage = deserialize_from_value_opt::<ResponseUsage>(json["usage"].take());
        let incomplete_details = deserialize_from_value_opt::<IncompleteDetails>(json["incomplete_details"].take());
        let error = (!json["error"].is_null()).then(|| json["error"].take());

        Ok(Self {
            response_id,
            conversation_id: conversation_id.map(str::to_string),
            output,
            usage,
            status,
            incomplete_details,
            error,
            in_flight: IndexMap::new(),
            in_flight_indexes: HashMap::new(),
            completed: Vec::new(),
            strict_call_ids: HashMap::new(),
            stream_lifecycle: StreamLifecycle::Terminal,
        })
    }

    /// Accumulates an async stream of raw SSE lines with parallel processing.
    ///
    /// The async task feeds raw SSE lines through a channel while a `spawn_blocking`
    /// worker handles JSON parsing on a blocking thread — keeping the tokio executor
    /// free between chunk arrivals.
    ///
    /// # Errors
    /// Returns [`ExecutorError::InvalidRequest`] when repeated authoritative
    /// output-item content conflicts, or [`ExecutorError::StreamError`] when
    /// the input stream or worker encounters an error.
    pub async fn from_stream(
        mut stream: Pin<Box<dyn Stream<Item = Result<String, ExecutorError>> + Send>>,
        conversation_id: Option<&str>,
    ) -> ExecutorResult<Self> {
        let (tx, rx) = mpsc::channel::<String>();
        // Convert to owned here — spawn_blocking closure must be 'static.
        let conv_id_owned = conversation_id.map(str::to_string);

        // Spawn blocking task: JSON parsing is CPU-bound, runs off the async executor.
        let worker_handle = tokio::task::spawn_blocking(move || Self::process_stream_chunks(rx, conv_id_owned));

        // Feed raw SSE lines from the async stream to the blocking worker.
        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    if tx.send(chunk).is_err() {
                        break;
                    }
                }
                Err(e) => return Err(e),
            }
        }

        // Signal EOF to worker.
        drop(tx);

        // Properly async join — does not block the tokio executor thread.
        worker_handle
            .await
            .map_err(|_| ExecutorError::StreamError("Worker thread panicked".into()))?
    }

    /// Worker function that processes SSE lines from the channel (runs on blocking thread).
    fn process_stream_chunks(rx: mpsc::Receiver<String>, conversation_id: Option<String>) -> ExecutorResult<Self> {
        let mut acc = Self::new(uuid7_str("resp_"), conversation_id);
        for line in rx {
            let _ = acc.process_lenient_sse_line(&line)?;
        }
        acc.finish_stream();
        Ok(acc)
    }

    /// Processes pre-collected raw SSE lines synchronously.
    ///
    /// Useful when lines have already been buffered (e.g. replaying a recorded stream).
    /// Prefer [`from_stream`](Self::from_stream) for live async streams.
    /// Malformed data frames are skipped for compatibility.
    ///
    /// # Errors
    /// Returns [`ExecutorError::InvalidRequest`] when repeated authoritative
    /// output-item content conflicts with the previously accumulated value.
    pub fn from_sse_lines(
        lines: impl IntoIterator<Item = String>,
        conversation_id: Option<&str>,
    ) -> ExecutorResult<Self> {
        let mut acc = Self::new(uuid7_str("resp_"), conversation_id.map(str::to_string));
        for line in lines {
            let _ = acc.process_lenient_sse_line(&line)?;
        }
        acc.finalize_all();
        Ok(acc)
    }

    /// Finalizes all streaming items in upstream `output_index` order.
    pub(crate) fn finalize_all(&mut self) {
        self.completed.extend(
            self.in_flight
                .drain(..)
                .filter_map(|(_, entry)| entry.item.finalize().map(|item| (entry.output_index, item))),
        );
        self.in_flight_indexes.clear();
        self.completed.sort_by_key(|(output_index, _)| *output_index);
        self.output
            .extend(self.completed.drain(..).map(|(_, output_item)| output_item));
    }

    pub(super) fn process_lenient_sse_line(&mut self, line: &str) -> ExecutorResult<Option<EventFrame>> {
        let Some(frame) = normalize_sse_line(line) else {
            return Ok(None);
        };
        if !self.process_normalized_event(&frame, None, false)? {
            return Ok(None);
        }
        Ok(Some(frame))
    }

    pub(crate) fn process_strict_sse_line(&mut self, line: &str) -> ExecutorResult<Option<EventFrame>> {
        let Some(frame) = normalize_sse_line(line) else {
            if is_data_frame(line) {
                return Err(ExecutorError::InvalidRequest(
                    "upstream stream contains a malformed data frame".to_owned(),
                ));
            }
            return Ok(None);
        };
        let validated = validate_frame(&frame).map_err(|error| ExecutorError::InvalidRequest(error.to_string()))?;
        self.validate_strict_transition(&frame, &validated)?;
        let _ = self.process_normalized_event(&frame, Some(&validated), true)?;
        if let Some(item) = validated
            .item
            .as_ref()
            .filter(|_| frame.event_type == SSEEventType::OutputItemDone)
        {
            self.completed.extend(
                self.in_flight
                    .shift_remove(item.item_id)
                    .and_then(|entry| entry.item.finalize().map(|output| (item.output_index, output))),
            );
            self.in_flight_indexes.remove(&item.output_index);
        }
        Ok(Some(frame))
    }

    fn process_normalized_event(
        &mut self,
        frame: &EventFrame,
        validated: Option<&ValidatedFrame<'_>>,
        strict: bool,
    ) -> ExecutorResult<bool> {
        self.capture_terminal_details_if_needed(frame);
        self.process_event_checked(frame, validated, strict)
    }

    fn validate_strict_transition(&mut self, frame: &EventFrame, validated: &ValidatedFrame<'_>) -> ExecutorResult<()> {
        let event_name = frame.wire.event_type.as_deref().unwrap_or("streaming event");
        if self.stream_lifecycle == StreamLifecycle::Terminal {
            return Err(invalid_stream(
                "upstream stream contains an event after its terminal event",
            ));
        }

        match (&frame.event_type, &frame.payload) {
            (SSEEventType::ResponseCreated, EventPayload::Response { .. }) => {
                if self.stream_lifecycle != StreamLifecycle::AwaitingCreated {
                    return Err(invalid_lifecycle(event_name));
                }
            }
            (SSEEventType::ResponseInProgress, EventPayload::Response { id, .. }) => {
                if self.stream_lifecycle != StreamLifecycle::Created || id != &self.response_id {
                    return Err(invalid_lifecycle_or_id(event_name));
                }
            }
            (
                SSEEventType::ResponseCompleted | SSEEventType::ResponseFailed | SSEEventType::ResponseIncomplete,
                EventPayload::Response { id, .. },
            ) => {
                if self.stream_lifecycle != StreamLifecycle::InProgress || id != &self.response_id {
                    return Err(invalid_lifecycle_or_id(event_name));
                }
                if !self.in_flight.is_empty() {
                    return Err(invalid_stream("upstream stream ended with unfinished output items"));
                }
                self.validate_terminal_output(frame, frame.event_type != SSEEventType::ResponseFailed)?;
            }
            (SSEEventType::OutputItemAdded, _) => {
                self.require_in_progress(event_name)?;
                let item = validated
                    .item
                    .as_ref()
                    .ok_or_else(|| invalid_stream("validated output item is missing"))?;
                if self.has_output_index(item.output_index) || self.has_item_id(item.item_id) {
                    return Err(invalid_stream(format!(
                        "upstream stream repeats output item '{}'",
                        item.item_id
                    )));
                }
            }
            _ => self.require_in_progress(event_name)?,
        }
        Ok(())
    }

    fn require_in_progress(&self, event_name: &str) -> ExecutorResult<()> {
        if self.stream_lifecycle == StreamLifecycle::InProgress {
            return Ok(());
        }
        Err(invalid_lifecycle(event_name))
    }

    fn resolve_active(
        &mut self,
        output_index: u32,
        item_id: &str,
        expected_type: SSEItemType,
        event_name: &str,
        output_index_is_explicit: bool,
        policy: ResolvePolicy,
    ) -> ExecutorResult<Option<&mut InFlightEntry>> {
        let Some(position) = self.active_position(item_id, output_index) else {
            return if policy == ResolvePolicy::Strict {
                Err(invalid_stream(format!(
                    "upstream stream event '{event_name}' has no active output item"
                )))
            } else {
                Ok(None)
            };
        };
        let index_matches = self
            .in_flight_indexes
            .get(&output_index)
            .and_then(|key| self.in_flight.get_index_of(key))
            == Some(position);
        let Some((_, active)) = self.in_flight.get_index_mut(position) else {
            return Ok(None);
        };
        let id_matches = active.item_id.is_empty() || item_id.is_empty() || active.item_id == item_id;
        let index_is_consistent = active.output_index == output_index;
        let type_matches = active.item_type == expected_type;
        if output_index_is_explicit && !index_is_consistent {
            return Err(invalid_stream(format!(
                "upstream stream event '{event_name}' does not match its active output item"
            )));
        }
        if !type_matches || (!id_matches && (policy == ResolvePolicy::Strict || !index_matches)) {
            return if policy == ResolvePolicy::Lenient {
                Ok(None)
            } else {
                Err(invalid_stream(format!(
                    "upstream stream event '{event_name}' does not match its active output item"
                )))
            };
        }
        if active.item_id.is_empty() && !item_id.is_empty() {
            item_id.clone_into(&mut active.item_id);
        }
        Ok(Some(active))
    }

    fn active_position(&self, item_id: &str, output_index: u32) -> Option<usize> {
        self.in_flight.get_index_of(item_id).or_else(|| {
            self.in_flight_indexes
                .get(&output_index)
                .and_then(|key| self.in_flight.get_index_of(key))
        })
    }

    fn has_output_index(&self, output_index: u32) -> bool {
        self.in_flight_indexes.contains_key(&output_index)
            || self.completed.iter().any(|(index, _)| *index == output_index)
    }

    fn has_item_id(&self, item_id: &str) -> bool {
        self.in_flight.contains_key(item_id)
            || self
                .completed
                .iter()
                .any(|(_, item)| item.id().is_some_and(|id| id == item_id))
    }

    fn validate_terminal_output(&mut self, frame: &EventFrame, enforce_call_id_stability: bool) -> ExecutorResult<()> {
        if enforce_call_id_stability {
            self.ensure_stable_strict_call_ids()?;
        }
        let response = frame
            .wire
            .rest
            .get("response")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| invalid_stream("terminal upstream response has no valid 'response'"))?;
        let Some(output) = response.get("output") else {
            return Ok(());
        };
        if output.is_null() {
            return Ok(());
        }
        let output = output
            .as_array()
            .ok_or_else(|| invalid_stream("terminal upstream response has no valid 'output'"))?;
        if output.len() != self.completed.len() {
            return Err(invalid_stream(
                "terminal upstream response output does not match completed item events",
            ));
        }

        let mut terminal_item_ids = HashSet::with_capacity(output.len());
        for (output_index, item) in output.iter().enumerate() {
            let output_index = u32::try_from(output_index)
                .map_err(|_| invalid_stream("terminal upstream response has too many output items"))?;
            let item = item
                .as_object()
                .ok_or_else(|| invalid_stream("terminal upstream response contains an invalid output item"))?;
            let (item_id, item_type) = output_item_identity(item, "terminal output item")
                .map_err(|error| invalid_stream(error.to_string()))?;
            if !terminal_item_ids.insert(item_id) {
                return Err(invalid_stream(format!(
                    "terminal upstream response repeats output item '{item_id}'"
                )));
            }
            let Some((_, completed)) = self.completed.iter().find(|(index, _)| *index == output_index) else {
                return Err(invalid_stream(
                    "terminal upstream response output does not match completed item events",
                ));
            };
            if SSEItemType::try_from(completed) != Ok(item_type) {
                return Err(invalid_stream(
                    "terminal upstream response output does not match completed item events",
                ));
            }
            let mut canonical = serde_json::Value::Object(item.clone());
            if canonical
                .get("id")
                .and_then(serde_json::Value::as_str)
                .is_none_or(str::is_empty)
            {
                canonical["id"] = serde_json::Value::String(item_id.to_owned());
            }
            let terminal_item = serde_json::from_value(canonical).map_err(|error| {
                invalid_stream(format!("terminal upstream response output item is invalid: {error}"))
            })?;
            self.observe_strict_call_id(output_index, output_item_call_id(&terminal_item));
        }
        if enforce_call_id_stability {
            self.ensure_stable_strict_call_ids()?;
        }
        Ok(())
    }

    fn ensure_stable_strict_call_ids(&self) -> ExecutorResult<()> {
        if let Some(output_index) = self
            .strict_call_ids
            .iter()
            .filter_map(|(output_index, observation)| observation.changed.then_some(*output_index))
            .min()
        {
            return Err(invalid_stream(format!(
                "upstream stream changes 'call_id' for output[{output_index}]"
            )));
        }
        Ok(())
    }

    fn observe_strict_call_id(&mut self, output_index: u32, call_id: Option<&str>) {
        self.strict_call_ids.entry(output_index).or_default().observe(call_id);
    }

    pub(super) fn process_sse_line_with_translator(
        &mut self,
        line: &str,
        translator: &mut FunctionSseTranslator,
    ) -> ExecutorResult<Option<FunctionSseTranslation>> {
        let Some(frame) = self.process_lenient_sse_line(line)? else {
            return Ok(None);
        };
        let call_key = function_event_key(&frame.payload);
        let call = call_key.and_then(|(item_id, output_index)| self.accumulated_function_call(item_id, output_index));
        translator.translate(frame, call).map(Some)
    }

    fn accumulated_function_call(&self, item_id: &str, output_index: u32) -> Option<AccumulatedFunctionCall<'_>> {
        if let Some(InFlightEntry {
            item_id: _,
            item_type: SSEItemType::FunctionCall,
            item: InFlight::FunctionCall { item, arguments },
            ..
        }) = self
            .active_position(item_id, output_index)
            .and_then(|position| self.in_flight.get_index(position).map(|(_, entry)| entry))
        {
            return Some(AccumulatedFunctionCall {
                item,
                output_index,
                arguments,
            });
        }

        self.completed.iter().rev().find_map(|(completed_index, item)| {
            let OutputItem::FunctionCall(item) = item else {
                return None;
            };
            (*completed_index == output_index).then_some(AccumulatedFunctionCall {
                item,
                output_index: *completed_index,
                arguments: &item.arguments,
            })
        })
    }

    fn capture_terminal_details(&mut self, frame: &EventFrame) {
        let Some(response) = frame.wire.rest.get("response") else {
            return;
        };

        self.incomplete_details = response
            .get("incomplete_details")
            .cloned()
            .and_then(deserialize_from_value_opt::<IncompleteDetails>);
        self.error = response.get("error").filter(|error| !error.is_null()).cloned();
    }

    fn capture_terminal_details_if_needed(&mut self, frame: &EventFrame) {
        if matches!(
            frame.event_type,
            SSEEventType::ResponseFailed | SSEEventType::ResponseIncomplete
        ) {
            self.capture_terminal_details(frame);
        }
    }

    pub(crate) fn finish_strict_stream(&mut self) -> ExecutorResult<()> {
        if self.stream_lifecycle != StreamLifecycle::Terminal {
            return Err(ExecutorError::InvalidRequest(
                "upstream stream ended without a terminal event".to_owned(),
            ));
        }
        self.finish_stream();
        Ok(())
    }

    pub(crate) fn finish_stream(&mut self) {
        self.finalize_all();
        if self.status == ResponseStatus::InProgress {
            self.status = ResponseStatus::Completed;
        }
        self.stream_lifecycle = StreamLifecycle::Terminal;
    }

    /// Processes a typed [`EventFrame`], updating accumulator state.
    ///
    /// This is the core state machine — callers that already have a normalized
    /// frame (e.g. [`StreamTee`](future)) can call this directly without
    /// re-parsing from a raw line.
    #[cfg(test)]
    fn process_event(&mut self, frame: &EventFrame) {
        let _ = self.process_event_checked(frame, None, false);
    }

    fn process_event_checked(
        &mut self,
        frame: &EventFrame,
        validated: Option<&ValidatedFrame<'_>>,
        strict: bool,
    ) -> ExecutorResult<bool> {
        let event_name = frame.wire.event_type.as_deref().unwrap_or("streaming event");
        let resolve_policy = if strict {
            ResolvePolicy::Strict
        } else {
            ResolvePolicy::Lenient
        };
        let output_index_is_explicit = has_explicit_output_index(frame);
        let mut emit = true;
        match (&frame.event_type, &frame.payload) {
            (SSEEventType::ResponseCreated, EventPayload::Response { id, .. }) if !id.is_empty() => {
                self.response_id.clone_from(id);
                self.stream_lifecycle = StreamLifecycle::Created;
            }
            (SSEEventType::ResponseInProgress, EventPayload::Response { .. }) => {
                self.stream_lifecycle = StreamLifecycle::InProgress;
            }
            (SSEEventType::OutputItemAdded, EventPayload::OutputItemAdded { .. }) => {
                self.process_output_item_added(frame, strict);
            }
            (SSEEventType::OutputItemDone, payload @ EventPayload::OutputItemDone { .. }) => {
                let done_item = validated
                    .and_then(|frame| frame.item.as_ref())
                    .and_then(|item| item.done_item.as_ref());
                emit = self.complete_call_item(payload, done_item, event_name, output_index_is_explicit, strict)?;
            }
            (SSEEventType::ReasoningTextDone | SSEEventType::ReasoningSummaryTextDone, payload) => {
                self.process_reasoning_done(payload, event_name, output_index_is_explicit, resolve_policy)?;
            }
            (SSEEventType::FunctionCallArgumentsDelta | SSEEventType::FunctionCallArgumentsDone, _) => {
                self.process_function_event(&frame.payload, event_name, output_index_is_explicit, strict)?;
            }
            (SSEEventType::CustomToolCallInputDelta | SSEEventType::CustomToolCallInputDone, payload) => {
                self.process_custom_tool_event(payload, event_name, output_index_is_explicit, resolve_policy)?;
            }
            (
                SSEEventType::OutputTextDelta,
                EventPayload::TextDelta {
                    delta,
                    item_id,
                    output_index,
                    ..
                },
            ) => {
                if let Some(entry) = self.resolve_active(
                    *output_index,
                    item_id,
                    SSEItemType::Message,
                    event_name,
                    output_index_is_explicit,
                    resolve_policy,
                )? && let InFlight::Message { text, .. } = &mut entry.item
                {
                    text.push_str(delta);
                }
            }
            (
                event_type @ (SSEEventType::ResponseCompleted
                | SSEEventType::ResponseFailed
                | SSEEventType::ResponseIncomplete),
                EventPayload::Response { usage, .. },
            ) => self.finish_response_event(*event_type, *usage),
            _ => {
                if strict && let Some(item) = validated.and_then(|frame| frame.item.as_ref()) {
                    let _ = self.resolve_active(
                        item.output_index,
                        item.item_id,
                        item.item_type,
                        event_name,
                        true,
                        ResolvePolicy::Strict,
                    )?;
                }
            }
        }
        Ok(emit)
    }

    fn process_reasoning_done(
        &mut self,
        payload: &EventPayload,
        event_name: &str,
        output_index_is_explicit: bool,
        resolve_policy: ResolvePolicy,
    ) -> ExecutorResult<()> {
        let (item_id, output_index) = match payload {
            EventPayload::ReasoningTextDone {
                item_id, output_index, ..
            }
            | EventPayload::ReasoningSummaryTextDone {
                item_id, output_index, ..
            } => (item_id, *output_index),
            _ => return Ok(()),
        };
        if let Some(entry) = self.resolve_active(
            output_index,
            item_id,
            SSEItemType::Reasoning,
            event_name,
            output_index_is_explicit,
            resolve_policy,
        )? && let InFlight::Reasoning { item } = &mut entry.item
        {
            item.apply_done(payload, &mut String::new());
        }
        Ok(())
    }

    fn process_custom_tool_event(
        &mut self,
        payload: &EventPayload,
        event_name: &str,
        output_index_is_explicit: bool,
        resolve_policy: ResolvePolicy,
    ) -> ExecutorResult<()> {
        let (item_id, output_index) = match payload {
            EventPayload::CustomToolCallInputDelta {
                item_id, output_index, ..
            }
            | EventPayload::CustomToolCallInputDone {
                item_id, output_index, ..
            } => (item_id, *output_index),
            _ => return Ok(()),
        };
        let Some(entry) = self.resolve_active(
            output_index,
            item_id,
            SSEItemType::CustomToolCall,
            event_name,
            output_index_is_explicit,
            resolve_policy,
        )?
        else {
            return Ok(());
        };
        match (&mut entry.item, payload) {
            (InFlight::CustomToolCall { input, .. }, EventPayload::CustomToolCallInputDelta { delta, .. }) => {
                input.push_str(delta);
            }
            (InFlight::CustomToolCall { item, input }, EventPayload::CustomToolCallInputDone { .. }) => {
                item.apply_done(payload, input);
            }
            _ => {}
        }
        Ok(())
    }

    fn finish_response_event(&mut self, event_type: SSEEventType, usage: Option<ResponseUsage>) {
        let status = match event_type {
            SSEEventType::ResponseCompleted => ResponseStatus::Completed,
            SSEEventType::ResponseFailed => ResponseStatus::Error,
            SSEEventType::ResponseIncomplete => ResponseStatus::Incomplete,
            _ => return,
        };
        self.finish_response(status, usage);
    }

    fn process_output_item_added(&mut self, frame: &EventFrame, strict: bool) {
        let payload = &frame.payload;
        if strict
            && let EventPayload::OutputItemAdded {
                output_index, call_id, ..
            } = payload
        {
            self.observe_strict_call_id(*output_index, call_id.as_deref());
        }
        self.start_output_item(payload, has_explicit_output_index(frame));
    }

    fn process_function_event(
        &mut self,
        payload: &EventPayload,
        event_name: &str,
        output_index_is_explicit: bool,
        strict: bool,
    ) -> ExecutorResult<()> {
        let (item_id, output_index, call_id) = match payload {
            EventPayload::FunctionCallArgsDelta {
                item_id,
                output_index,
                call_id,
                ..
            }
            | EventPayload::FunctionCallArgsDone {
                item_id,
                output_index,
                call_id,
                ..
            } => (item_id.as_str(), *output_index, call_id.as_deref()),
            _ => return Ok(()),
        };
        if strict {
            self.observe_strict_call_id(output_index, call_id);
        }
        let resolve_policy = if strict {
            ResolvePolicy::Strict
        } else {
            ResolvePolicy::Lenient
        };
        let Some(entry) = self.resolve_active(
            output_index,
            item_id,
            SSEItemType::FunctionCall,
            event_name,
            output_index_is_explicit,
            resolve_policy,
        )?
        else {
            return Ok(());
        };
        match (&mut entry.item, payload) {
            (InFlight::FunctionCall { arguments, .. }, EventPayload::FunctionCallArgsDelta { delta, .. }) => {
                arguments.push_str(delta);
            }
            (InFlight::FunctionCall { item, arguments }, EventPayload::FunctionCallArgsDone { .. }) => {
                item.apply_done(payload, arguments);
            }
            _ => {}
        }
        Ok(())
    }

    fn start_output_item(&mut self, payload: &EventPayload, has_explicit_output_index: bool) {
        let EventPayload::OutputItemAdded {
            item_id,
            item_type,
            output_index,
            ..
        } = payload
        else {
            return;
        };
        let item = match item_type {
            SSEItemType::Reasoning => ReasoningOutput::try_from(payload)
                .ok()
                .map(|item| InFlight::Reasoning { item }),
            SSEItemType::FunctionCall => FunctionToolCall::try_from(payload)
                .ok()
                .map(|item| InFlight::FunctionCall {
                    item,
                    arguments: String::with_capacity(128),
                }),
            SSEItemType::CustomToolCall => {
                CustomToolCall::try_from(payload)
                    .ok()
                    .map(|item| InFlight::CustomToolCall {
                        item,
                        input: String::with_capacity(256),
                    })
            }
            SSEItemType::Message => OutputMessage::try_from(payload).ok().map(|item| InFlight::Message {
                item,
                text: String::with_capacity(256),
            }),
            SSEItemType::WebSearchCall if !item_id.is_empty() => Some(InFlight::WebSearchCall { item: None }),
            SSEItemType::Compaction => CompactionItem::try_from(payload)
                .ok()
                .map(|item| InFlight::Compaction { item }),
            SSEItemType::WebSearchCall => None,
            SSEItemType::McpCall => McpCall::try_from(payload).ok().map(|item| InFlight::McpCall { item }),
            SSEItemType::McpListTools => McpListTools::try_from(payload)
                .ok()
                .map(|item| InFlight::McpListTools { item }),
        };
        if let Some(item) = item {
            let mut key = if !item_id.is_empty() && !self.in_flight.contains_key(item_id) {
                item_id.clone()
            } else {
                format!("__output_index_{output_index}")
            };
            while self.in_flight.contains_key(&key) {
                key.push('_');
            }
            if has_explicit_output_index {
                self.in_flight_indexes
                    .entry(*output_index)
                    .or_insert_with(|| key.clone());
            }
            self.in_flight.insert(
                key,
                InFlightEntry {
                    output_index: *output_index,
                    item_id: item_id.clone(),
                    item_type: *item_type,
                    item,
                    done_item: None,
                },
            );
        }
    }

    fn finish_response(&mut self, status: ResponseStatus, usage: Option<ResponseUsage>) {
        self.finalize_all();
        self.status = status;
        self.usage = usage;
        self.stream_lifecycle = StreamLifecycle::Terminal;
    }

    fn complete_call_item(
        &mut self,
        payload: &EventPayload,
        validated_done_item: Option<&OutputItem>,
        event_name: &str,
        output_index_is_explicit: bool,
        strict: bool,
    ) -> ExecutorResult<bool> {
        let EventPayload::OutputItemDone {
            item_id,
            item_type,
            output_index,
            item: raw_item,
            ..
        } = payload
        else {
            return Ok(true);
        };
        let parsed_done_item = validated_done_item.cloned().or_else(|| {
            deserialize_from_value_opt::<OutputItem>(raw_item.clone()).or_else(|| {
                (*item_type == SSEItemType::Reasoning)
                    .then(|| ReasoningOutput::try_from(payload).ok().map(OutputItem::Reasoning))
                    .flatten()
            })
        });
        if strict {
            self.observe_strict_call_id(*output_index, parsed_done_item.as_ref().and_then(output_item_call_id));
        }

        let resolve_policy = if strict {
            ResolvePolicy::Strict
        } else {
            ResolvePolicy::Authoritative
        };
        if let Some(entry) = self.resolve_active(
            *output_index,
            item_id,
            *item_type,
            event_name,
            output_index_is_explicit,
            resolve_policy,
        )? {
            let mut candidate = entry.item.clone();
            apply_output_item_done(&mut candidate, payload, parsed_done_item.as_ref(), &entry.item_id);
            let candidate_done = candidate.clone().finalize();
            if let Some(previous) = &entry.done_item {
                if candidate_done
                    .as_ref()
                    .is_some_and(|candidate| output_items_semantically_equal(previous, candidate))
                {
                    return Ok(false);
                }
                return Err(invalid_stream(format!(
                    "upstream stream contains conflicting repeated output item.done for output[{output_index}]"
                )));
            }
            entry.item = candidate;
            entry.done_item = candidate_done;
            return Ok(true);
        }

        if let Some((_, previous)) = self.completed.iter().find(|(index, _)| index == output_index) {
            if parsed_done_item
                .as_ref()
                .is_some_and(|candidate| output_items_semantically_equal(previous, candidate))
            {
                return Ok(false);
            }
            return Err(invalid_stream(format!(
                "upstream stream contains conflicting repeated output item.done for output[{output_index}]"
            )));
        }

        if let Some(
            mut output_item @ (OutputItem::Reasoning(_)
            | OutputItem::FunctionCall(_)
            | OutputItem::CustomToolCall(_)
            | OutputItem::WebSearchCall(_)
            | OutputItem::McpCall(_)
            | OutputItem::McpListTools(_)
            | OutputItem::Compaction(_)),
        ) = parsed_done_item
        {
            let OutputItem::WebSearchCall(call) = &mut output_item else {
                self.completed.push((*output_index, output_item));
                return Ok(true);
            };
            if call.id.is_empty() {
                call.id = uuid7_str("ws_");
            }
            self.completed.push((*output_index, output_item));
        }
        Ok(true)
    }

    /// Marks the response as incomplete due to an error or interruption.
    pub fn mark_incomplete(&mut self, reason: impl Into<String>) {
        self.status = ResponseStatus::Incomplete;
        self.incomplete_details = Some(IncompleteDetails {
            reason: Some(reason.into()),
        });
    }

    /// Finalizes the accumulator into a `ResponsePayload`.
    ///
    /// The caller supplies fields that come from the original request, not from
    /// the LLM response stream.
    #[must_use]
    pub fn finalize(
        self,
        model: &str,
        previous_response_id: Option<&str>,
        instructions: Option<&str>,
    ) -> ResponsePayload {
        ResponsePayload {
            id: self.response_id,
            object: "response".to_string(),
            created_at: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            status: self.status.as_str().to_string(),
            output: self.output,
            usage: self.usage,
            incomplete_details: self.incomplete_details,
            error: self.error,
            previous_response_id: previous_response_id.map(str::to_string),
            conversation_id: self.conversation_id,
            instructions: instructions.map(str::to_string),
        }
    }
}

fn apply_output_item_done(
    in_flight: &mut InFlight,
    payload: &EventPayload,
    done_item: Option<&OutputItem>,
    item_id: &str,
) {
    match (in_flight, done_item) {
        (InFlight::Message { item, text }, Some(OutputItem::Message(done))) => {
            item.clone_from(done);
            text.clear();
        }
        (InFlight::Reasoning { item }, Some(OutputItem::Reasoning(done))) => {
            let mut done = done.clone();
            let raw = match payload {
                EventPayload::OutputItemDone { item, .. } => item.as_object(),
                _ => None,
            };
            if raw.is_some_and(|raw| !raw.contains_key("content")) {
                done.content.clone_from(&item.content);
            }
            if raw.is_some_and(|raw| !raw.contains_key("summary")) {
                done.summary.clone_from(&item.summary);
            }
            if done.id.is_empty() {
                done.id.clone_from(&item.id);
            }
            *item = done;
        }
        (InFlight::FunctionCall { item, arguments }, Some(OutputItem::FunctionCall(done))) => {
            let mut done = done.clone();
            if done.id.is_empty() {
                done.id.clone_from(&item.id);
            }
            if done.call_id.is_empty() {
                done.call_id.clone_from(&item.call_id);
            }
            if done.name.is_empty() {
                done.name.clone_from(&item.name);
            }
            if done.namespace.is_none() {
                done.namespace.clone_from(&item.namespace);
            }
            if done.arguments.is_empty() {
                done.arguments = if item.arguments.is_empty() {
                    std::mem::take(arguments)
                } else {
                    item.arguments.clone()
                };
            } else {
                arguments.clear();
            }
            *item = done;
        }
        (InFlight::CustomToolCall { item, input }, Some(OutputItem::CustomToolCall(done))) => {
            let mut done = done.clone();
            if done.id.is_empty() {
                done.id.clone_from(&item.id);
            }
            if done.call_id.is_empty() {
                done.call_id.clone_from(&item.call_id);
            }
            if done.name.is_empty() {
                done.name.clone_from(&item.name);
            }
            if done.input.is_empty() {
                done.input = if item.input.is_empty() {
                    std::mem::take(input)
                } else {
                    item.input.clone()
                };
            } else {
                input.clear();
            }
            *item = done;
        }
        (InFlight::WebSearchCall { item }, Some(OutputItem::WebSearchCall(done))) => {
            let mut done = done.clone();
            if done.id.is_empty() {
                done.id = if item_id.is_empty() {
                    uuid7_str("ws_")
                } else {
                    item_id.to_owned()
                };
            }
            *item = Some(done);
        }
        (InFlight::McpCall { item }, Some(OutputItem::McpCall(done))) => item.clone_from(done),
        (InFlight::McpListTools { item }, Some(OutputItem::McpListTools(done))) => item.clone_from(done),
        (InFlight::Compaction { item }, Some(OutputItem::Compaction(done))) => {
            let mut done = done.clone();
            if done.id.as_deref().is_none_or(str::is_empty) {
                done.id.clone_from(&item.id);
            }
            *item = done;
        }
        (InFlight::Reasoning { item }, None) => item.apply_done(payload, &mut String::new()),
        (InFlight::FunctionCall { item, arguments }, None) => item.apply_done(payload, arguments),
        (InFlight::CustomToolCall { item, input }, None) => item.apply_done(payload, input),
        (InFlight::McpCall { item }, None) => item.apply_done(payload, &mut String::new()),
        (InFlight::McpListTools { item }, None) => item.apply_done(payload, &mut String::new()),
        (InFlight::Compaction { item }, None) => item.apply_done(payload, &mut String::new()),
        _ => {}
    }
}

fn output_item_call_id(item: &OutputItem) -> Option<&str> {
    match item {
        OutputItem::FunctionCall(call) => Some(&call.call_id),
        OutputItem::CustomToolCall(call) => Some(&call.call_id),
        _ => None,
    }
}

fn output_items_semantically_equal(left: &OutputItem, right: &OutputItem) -> bool {
    match (serde_json::to_value(left), serde_json::to_value(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn invalid_lifecycle(event_name: &str) -> ExecutorError {
    invalid_stream(format!(
        "upstream stream event '{event_name}' is out of lifecycle order"
    ))
}

fn invalid_lifecycle_or_id(event_name: &str) -> ExecutorError {
    invalid_stream(format!(
        "upstream stream event '{event_name}' is out of lifecycle order or changes the response id"
    ))
}

fn invalid_stream(message: impl Into<String>) -> ExecutorError {
    ExecutorError::InvalidRequest(message.into())
}

fn function_event_key(payload: &EventPayload) -> Option<(&str, u32)> {
    match payload {
        EventPayload::OutputItemAdded {
            item_id,
            item_type: SSEItemType::FunctionCall,
            output_index,
            ..
        }
        | EventPayload::OutputItemDone {
            item_id,
            item_type: SSEItemType::FunctionCall,
            output_index,
            ..
        }
        | EventPayload::FunctionCallArgsDelta {
            item_id, output_index, ..
        }
        | EventPayload::FunctionCallArgsDone {
            item_id, output_index, ..
        } => Some((item_id, *output_index)),
        _ => None,
    }
}

fn has_explicit_output_index(frame: &EventFrame) -> bool {
    frame
        .wire
        .output_index
        .and_then(|index| u32::try_from(index).ok())
        .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::WireEvent;
    use crate::types::io::{McpCallError, McpCallStatus, WebSearchCallStatus};

    fn from_sse_lines(lines: impl IntoIterator<Item = String>, conversation_id: Option<&str>) -> ResponseAccumulator {
        ResponseAccumulator::from_sse_lines(lines, conversation_id).expect("valid SSE stream")
    }

    #[test]
    fn test_accumulator_new() {
        let acc = ResponseAccumulator::new("resp_123".into(), Some("conv_456".into()));
        assert_eq!(acc.response_id, "resp_123");
        assert_eq!(acc.conversation_id, Some("conv_456".into()));
        assert_eq!(acc.status, ResponseStatus::InProgress);
    }

    #[test]
    fn test_accumulator_mark_incomplete() {
        let mut acc = ResponseAccumulator::new("resp_123".into(), None);
        acc.mark_incomplete("Stream interrupted");
        assert_eq!(acc.status, ResponseStatus::Incomplete);
        assert!(acc.incomplete_details.is_some());
    }

    #[test]
    fn test_accumulator_preserves_streamed_failure_details() {
        let acc = from_sse_lines(
            [r#"data: {"type":"response.failed","response":{"id":"resp_failed","status":"failed","error":{"code":"tool_catalog_too_large","message":"Too many tools"},"incomplete_details":{"reason":"upstream_error"}}}"#.to_owned()],
            None,
        );
        let payload = acc.finalize("test-model", None, None);

        assert_eq!(payload.status, "error");
        assert_eq!(payload.error.as_ref().unwrap()["code"], "tool_catalog_too_large");
        assert_eq!(
            payload.incomplete_details.unwrap().reason.as_deref(),
            Some("upstream_error")
        );
    }

    #[test]
    fn test_accumulator_finalize() {
        let acc = ResponseAccumulator::new("resp_123".into(), Some("conv_456".into()));
        let payload = acc.finalize("gpt-4o", Some("resp_prev"), Some("be helpful"));
        assert_eq!(payload.id, "resp_123");
        assert_eq!(payload.model, "gpt-4o");
        assert_eq!(payload.conversation_id, Some("conv_456".into()));
        assert_eq!(payload.previous_response_id, Some("resp_prev".into()));
        assert_eq!(payload.instructions, Some("be helpful".into()));
        assert_eq!(payload.status, ResponseStatus::InProgress.as_str());
    }

    #[test]
    fn test_accumulator_from_sse_lines_empty() {
        let acc = from_sse_lines(vec![], None);
        assert_eq!(acc.status, ResponseStatus::InProgress);
        assert!(acc.output.is_empty());
    }

    #[test]
    fn test_accumulator_text_delta_assigned_to_message() {
        let lines = vec![
            r#"data: {"type":"response.created","response":{"id":"resp_abc"}}"#.to_string(),
            r#"data: {"type":"response.output_item.added","item":{"id":"msg_1"}}"#.to_string(),
            r#"data: {"type":"response.output_text.delta","delta":"Hello","item_id":"msg_1"}"#.to_string(),
            r#"data: {"type":"response.output_text.delta","delta":" world","item_id":"msg_1"}"#.to_string(),
            r#"data: {"type":"response.done","response":{"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.status, ResponseStatus::Completed);
        assert_eq!(acc.output.len(), 1);

        if let OutputItem::Message(msg) = &acc.output[0] {
            assert_eq!(msg.content.len(), 1);
            assert_eq!(msg.content[0].text, "Hello world");
        } else {
            panic!("expected OutputItem::Message");
        }

        assert!(acc.usage.is_some());
        let usage = acc.usage.unwrap();
        assert_eq!(usage.total_tokens, 7);
    }

    #[test]
    fn test_message_status_enum() {
        assert_eq!(MessageStatus::Completed.as_str(), "completed");
        assert_eq!(MessageStatus::InProgress.as_str(), "in_progress");
    }

    #[test]
    fn test_process_event_response_created_sets_id() {
        let mut acc = ResponseAccumulator::new("resp_old".into(), None);
        let frame = EventFrame {
            event_type: SSEEventType::ResponseCreated,
            payload: EventPayload::Response {
                id: "resp_new".into(),
                status: "in_progress".into(),
                usage: None,
            },
            wire: WireEvent::new("test"),
        };
        acc.process_event(&frame);
        assert_eq!(acc.response_id, "resp_new");
    }

    #[test]
    fn test_process_event_response_created_empty_id_no_overwrite() {
        let mut acc = ResponseAccumulator::new("resp_keep".into(), None);
        let frame = EventFrame {
            event_type: SSEEventType::ResponseCreated,
            payload: EventPayload::Response {
                id: String::new(),
                status: "in_progress".into(),
                usage: None,
            },
            wire: WireEvent::new("test"),
        };
        acc.process_event(&frame);
        assert_eq!(acc.response_id, "resp_keep");
    }

    #[test]
    fn test_process_event_text_delta_accumulates() {
        let mut acc = ResponseAccumulator::new("resp_1".into(), None);

        acc.process_event(&EventFrame {
            event_type: SSEEventType::OutputItemAdded,
            payload: EventPayload::OutputItemAdded {
                item_id: "msg_1".into(),
                item_type: "message".into(),
                output_index: 0,
                name: None,
                namespace: None,
                call_id: None,
            },
            wire: WireEvent::new("test"),
        });

        acc.process_event(&EventFrame {
            event_type: SSEEventType::OutputTextDelta,
            payload: EventPayload::TextDelta {
                delta: "Hello".into(),
                item_id: "msg_1".into(),
                output_index: 0,
                content_index: 0,
            },
            wire: WireEvent::new("test"),
        });
        acc.process_event(&EventFrame {
            event_type: SSEEventType::OutputTextDelta,
            payload: EventPayload::TextDelta {
                delta: " world".into(),
                item_id: "msg_1".into(),
                output_index: 0,
                content_index: 0,
            },
            wire: WireEvent::new("test"),
        });

        acc.process_event(&EventFrame {
            event_type: SSEEventType::ResponseCompleted,
            payload: EventPayload::Response {
                id: "resp_1".into(),
                status: "completed".into(),
                usage: None,
            },
            wire: WireEvent::new("test"),
        });

        assert_eq!(acc.status, ResponseStatus::Completed);
        assert_eq!(acc.output.len(), 1);
        if let OutputItem::Message(msg) = &acc.output[0] {
            assert_eq!(msg.content[0].text, "Hello world");
        } else {
            panic!("expected Message");
        }
    }

    #[test]
    fn test_process_event_mcp_call_done_accumulates_output() {
        let lines = vec![
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"mcp_call","id":"mcp_1","server_label":"counter","name":"increment","arguments":"","status":"in_progress","approval_request_id":null,"output":null,"error":null}}"#.to_string(),
            r#"data: {"type":"response.mcp_call.in_progress","item_id":"mcp_1","output_index":0}"#.to_string(),
            r#"data: {"type":"response.mcp_call_arguments.delta","delta":"{}","item_id":"mcp_1","output_index":0}"#.to_string(),
            r#"data: {"type":"response.mcp_call_arguments.done","arguments":"{}","item_id":"mcp_1","output_index":0}"#.to_string(),
            r#"data: {"type":"response.mcp_call.completed","item_id":"mcp_1","output_index":0}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"mcp_call","id":"mcp_1","server_label":"counter","name":"increment","arguments":"{}","status":"completed","approval_request_id":null,"output":"1","error":null}}"#.to_string(),
            r#"data: {"type":"response.done","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.status, ResponseStatus::Completed);
        assert_eq!(acc.output.len(), 1);
        assert!(matches!(acc.output[0], OutputItem::McpCall(_)));
    }

    #[test]
    fn test_process_event_mcp_list_tools_done_accumulates_output() {
        let added = r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"mcp_list_tools","id":"mcpl_1","server_label":"counter","tools":[]}}"#;
        let remaining = [
            r#"data: {"type":"response.mcp_list_tools.in_progress","item_id":"mcpl_1","output_index":0}"#.to_string(),
            r#"data: {"type":"response.mcp_list_tools.completed","item_id":"mcpl_1","output_index":0}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"mcp_list_tools","id":"mcpl_1","server_label":"counter","tools":[{"name":"increment","description":"Increment the counter","input_schema":{"type":"object","properties":{}},"annotations":{"read_only":false}}]}}"#.to_string(),
            r#"data: {"type":"response.done","response":{"id":"resp_1","status":"completed"}}"#.to_string(),
        ];

        let mut acc = ResponseAccumulator::new("resp_1".to_owned(), None);
        let _ = acc.process_lenient_sse_line(added).expect("valid SSE event");
        let Some(InFlightEntry {
            item: InFlight::McpListTools { item },
            ..
        }) = acc.in_flight.get("mcpl_1")
        else {
            panic!("expected in-flight mcp_list_tools");
        };
        assert!(item.server_label.is_empty());
        assert!(item.tools.is_empty());

        for line in remaining {
            let _ = acc.process_lenient_sse_line(&line).expect("valid SSE event");
        }
        acc.finalize_all();

        assert_eq!(acc.status, ResponseStatus::Completed);
        assert_eq!(acc.output.len(), 1);
        let OutputItem::McpListTools(item) = &acc.output[0] else {
            panic!("expected mcp_list_tools");
        };
        assert_eq!(item.id, "mcpl_1");
        assert_eq!(item.server_label, "counter");
        assert_eq!(item.tools.len(), 1);
        assert_eq!(item.tools[0].name, "increment");
        assert_eq!(item.tools[0].annotations, Some(serde_json::json!({"read_only": false})));
    }

    #[test]
    fn compaction_added_and_done_accumulate_typed_output() {
        let done = r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"compaction","id":"cmp_1","encrypted_content":"durable summary"}}"#;
        let lines = [
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"compaction","id":"cmp_1","encrypted_content":"durable summary"}}"#.to_owned(),
            done.to_owned(),
            r#"data: {"type":"response.completed","response":{"id":"resp_1","status":"completed"}}"#.to_owned(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_compaction_output(&acc.output);

        let done_only = from_sse_lines([done.to_owned()], None);
        assert_compaction_output(&done_only.output);
    }

    fn assert_compaction_output(output: &[OutputItem]) {
        assert_eq!(output.len(), 1);
        let OutputItem::Compaction(item) = &output[0] else {
            panic!("expected compaction output");
        };
        assert_eq!(item.id.as_deref(), Some("cmp_1"));
        assert_eq!(item.encrypted_content, "durable summary");
    }

    #[test]
    fn test_accumulator_reasoning_before_mcp_call_preserves_order() {
        let lines = vec![
            r#"data: {"type":"response.created","response":{"id":"resp_abc"}}"#.to_string(),
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}"#.to_string(),
            r#"data: {"type":"response.reasoning_text.done","text":"thinking...","item_id":"rs_1"}"#.to_string(),
            r#"data: {"type":"response.output_item.added","output_index":1,"item":{"type":"mcp_call","id":"mcp_1","server_label":"counter","name":"increment","arguments":"","status":"in_progress","approval_request_id":null,"output":null,"error":null}}"#.to_string(),
            r#"data: {"type":"response.mcp_call.completed","item_id":"mcp_1","output_index":1}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":1,"item":{"type":"mcp_call","id":"mcp_1","server_label":"counter","name":"increment","arguments":"{}","status":"completed","approval_request_id":null,"output":"1","error":null}}"#.to_string(),
            r#"data: {"type":"response.done","response":{"id":"resp_abc","status":"completed","usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.output.len(), 2);
        assert!(matches!(acc.output[0], OutputItem::Reasoning(_)));
        assert!(matches!(acc.output[1], OutputItem::McpCall(_)));
    }

    #[test]
    fn test_accumulator_reasoning_before_done_only_mcp_call_preserves_order() {
        let lines = vec![
            r#"data: {"type":"response.created","response":{"id":"resp_abc"}}"#.to_string(),
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}"#.to_string(),
            r#"data: {"type":"response.reasoning_text.done","text":"thinking...","item_id":"rs_1"}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":1,"item":{"type":"mcp_call","id":"mcp_1","server_label":"counter","name":"increment","arguments":"{}","status":"completed","approval_request_id":null,"output":"1","error":null}}"#.to_string(),
            r#"data: {"type":"response.done","response":{"id":"resp_abc","status":"completed","usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.output.len(), 2);
        assert!(matches!(acc.output[0], OutputItem::Reasoning(_)));
        assert!(matches!(acc.output[1], OutputItem::McpCall(_)));
    }

    #[test]
    fn test_accumulator_reasoning_before_web_search_call_preserves_order() {
        let lines = vec![
            r#"data: {"type":"response.created","response":{"id":"resp_abc"}}"#.to_string(),
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}"#.to_string(),
            r#"data: {"type":"response.reasoning_text.done","text":"thinking...","item_id":"rs_1"}"#.to_string(),
            r#"data: {"type":"response.output_item.added","output_index":1,"item":{"type":"web_search_call","id":"ws_1","status":"in_progress","action":{"type":"search","query":"","sources":[]}}}"#.to_string(),
            r#"data: {"type":"response.web_search_call.in_progress","item_id":"ws_1","output_index":1}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":1,"item":{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"rust","sources":[]}}}"#.to_string(),
            r#"data: {"type":"response.done","response":{"id":"resp_abc","status":"completed","usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.output.len(), 2);
        assert!(matches!(acc.output[0], OutputItem::Reasoning(_)));
        let OutputItem::WebSearchCall(call) = &acc.output[1] else {
            panic!("expected web_search_call");
        };
        assert_eq!(call.status, WebSearchCallStatus::Completed);
        assert_eq!(call.action.as_search().unwrap().query, "rust");
    }

    #[test]
    fn test_accumulator_preserves_open_page_web_search_action() {
        let lines = vec![
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"ws_1","status":"in_progress"}}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"open_page","url":"https://example.com"}}}"#.to_string(),
            r#"data: {"type":"response.done","response":{"id":"resp_1","status":"completed"}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.output.len(), 1);
        let action = match &acc.output[0] {
            OutputItem::WebSearchCall(call) => serde_json::to_value(&call.action).unwrap(),
            _ => panic!("expected web_search_call"),
        };
        assert_eq!(action["type"], "open_page");
        assert_eq!(action["url"], "https://example.com");
    }

    #[test]
    fn test_accumulator_preserves_find_in_page_web_search_action() {
        let lines = vec![
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"ws_1","status":"in_progress"}}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"find_in_page","url":"https://example.com","pattern":"needle"}}}"#.to_string(),
            r#"data: {"type":"response.done","response":{"id":"resp_1","status":"completed"}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.output.len(), 1);
        let action = match &acc.output[0] {
            OutputItem::WebSearchCall(call) => serde_json::to_value(&call.action).unwrap(),
            _ => panic!("expected web_search_call"),
        };
        assert_eq!(action["type"], "find_in_page");
        assert_eq!(action["url"], "https://example.com");
        assert_eq!(action["pattern"], "needle");
    }

    #[test]
    fn test_accumulator_drops_unfinished_web_search_placeholder() {
        let lines = vec![
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"ws_1","status":"in_progress"}}"#.to_string(),
            r#"data: {"type":"response.done","response":{"id":"resp_1","status":"completed"}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert!(acc.output.is_empty());
    }

    #[test]
    fn test_accumulator_empty_added_id_then_stable_done_does_not_duplicate() {
        let lines = vec![
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"","status":"in_progress"}}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"rust","sources":[]}}}"#.to_string(),
            r#"data: {"type":"response.done","response":{"id":"resp_1","status":"completed"}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.output.len(), 1);
        let OutputItem::WebSearchCall(call) = &acc.output[0] else {
            panic!("expected web_search_call");
        };
        assert_eq!(call.id, "ws_1");
    }

    #[test]
    fn test_accumulator_stable_added_id_survives_empty_done_id() {
        let lines = vec![
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"ws_added","status":"in_progress"}}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","id":"","status":"completed","action":{"type":"search","query":"rust","sources":[]}}}"#.to_string(),
            r#"data: {"type":"response.done","response":{"id":"resp_1","status":"completed"}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.output.len(), 1);
        let OutputItem::WebSearchCall(call) = &acc.output[0] else {
            panic!("expected web_search_call");
        };
        assert_eq!(call.id, "ws_added");
    }

    #[test]
    fn test_accumulator_uses_authoritative_done_item_with_item_id_fallback() {
        let lines = vec![
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}"#.to_string(),
            r#"data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"item_id":"msg_1","delta":"partial"}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":null,"item_id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"authoritative","annotations":[]}]}}"#.to_string(),
            r#"data: {"type":"response.done","response":{"id":"resp_1","status":"completed"}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.output.len(), 1);
        let OutputItem::Message(message) = &acc.output[0] else {
            panic!("expected message");
        };
        assert_eq!(message.id, "msg_1");
        assert_eq!(message.content.len(), 1);
        assert_eq!(message.content[0].text, "authoritative");
    }

    #[test]
    fn repeated_function_call_done_does_not_duplicate_lenient_output() {
        let added = r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":""}}"#;
        let done = r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"}}"#;
        let terminal = r#"data: {"type":"response.completed","response":{"id":"resp_1","status":"completed"}}"#;

        let acc = from_sse_lines([added, done, done, terminal].map(str::to_owned), None);

        assert_eq!(acc.output.len(), 1);
        let OutputItem::FunctionCall(call) = &acc.output[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.id, "fc_1");
        assert_eq!(call.call_id, "call_1");
        assert_eq!(call.arguments, "{}");
    }

    #[test]
    fn repeated_identical_done_is_not_emitted_by_lenient_translation() {
        let added = r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":""}}"#;
        let done = r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"}}"#;
        let mut acc = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::default();

        acc.process_sse_line_with_translator(added, &mut translator)
            .expect("added item is valid");
        acc.process_sse_line_with_translator(done, &mut translator)
            .expect("first done item is valid");
        let repeated = acc
            .process_sse_line_with_translator(done, &mut translator)
            .expect("identical repeated done is valid");

        assert!(repeated.is_none());
    }

    #[test]
    fn repeated_function_call_done_rejects_conflicting_lenient_authoritative_content() {
        let added = r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":""}}"#;
        let first = r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"}}"#;
        let conflicting = r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"query\":\"rust\"}","status":"completed"}}"#;
        let mut acc = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::default();

        acc.process_sse_line_with_translator(added, &mut translator)
            .expect("added item is valid");
        acc.process_sse_line_with_translator(first, &mut translator)
            .expect("first done item is valid");
        let error = acc
            .process_sse_line_with_translator(conflicting, &mut translator)
            .expect_err("conflicting repeated done must be rejected");

        assert!(error.to_string().contains("conflicting repeated output item.done"));
    }

    #[test]
    fn from_sse_lines_propagates_conflicting_lenient_authoritative_content() {
        let added = r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":""}}"#;
        let first = r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"}}"#;
        let conflicting = r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"query\":\"rust\"}","status":"completed"}}"#;

        let error = ResponseAccumulator::from_sse_lines([added, first, conflicting].map(str::to_owned), None)
            .expect_err("conflicting authoritative content must propagate from the constructor");

        assert!(error.to_string().contains("conflicting repeated output item.done"));
    }

    #[tokio::test]
    async fn from_stream_propagates_conflicting_lenient_authoritative_content() {
        let lines = [
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":""}}"#,
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"}}"#,
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"query\":\"rust\"}","status":"completed"}}"#,
        ];
        let stream = futures::stream::iter(lines.into_iter().map(|line| Ok::<_, ExecutorError>(line.to_owned())));

        let error = ResponseAccumulator::from_stream(Box::pin(stream), None)
            .await
            .expect_err("conflicting authoritative content must propagate from the async constructor");

        assert!(error.to_string().contains("conflicting repeated output item.done"));
    }

    #[test]
    fn repeated_done_rejects_a_conflicting_lenient_item_type() {
        let added = r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":""}}"#;
        let conflicting = r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"custom_tool_call","id":"fc_1","call_id":"call_1","name":"lookup","input":"{}","status":"completed"}}"#;
        let mut acc = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::default();

        acc.process_sse_line_with_translator(added, &mut translator)
            .expect("added item is valid");
        let error = acc
            .process_sse_line_with_translator(conflicting, &mut translator)
            .expect_err("an authoritative done item must not change type");

        assert!(error.to_string().contains("does not match its active output item"));
    }

    #[test]
    fn lenient_events_reject_contradictory_explicit_item_id_and_output_index() {
        let added_first = r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":""}}"#;
        let added_second = r#"data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"lookup","arguments":""}}"#;
        let contradictory = r#"data: {"type":"response.function_call_arguments.delta","output_index":1,"item_id":"fc_1","call_id":"call_1","delta":"{}"}"#;
        let mut acc = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut translator = FunctionSseTranslator::default();

        acc.process_sse_line_with_translator(added_first, &mut translator)
            .expect("first item is valid");
        acc.process_sse_line_with_translator(added_second, &mut translator)
            .expect("second item is valid");
        let error = acc
            .process_sse_line_with_translator(contradictory, &mut translator)
            .expect_err("explicit item id and output index must resolve to the same item");

        assert!(error.to_string().contains("does not match its active output item"));
    }

    #[test]
    fn test_unknown_mcp_call_error_shape_is_not_dropped() {
        let lines = vec![
            r#"data: {"type":"response.created","response":{"id":"resp_abc"}}"#.to_string(),
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"mcp_call","id":"mcp_1","server_label":"counter","name":"increment","arguments":"","status":"in_progress","approval_request_id":null,"output":null,"error":null}}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"mcp_call","id":"mcp_1","server_label":"counter","name":"increment","arguments":"{}","status":"failed","approval_request_id":null,"output":null,"error":{"type":"mcp_protocol_error","code":-32000,"message":"boom"}}}"#.to_string(),
            r#"data: {"type":"response.done","response":{"id":"resp_abc","status":"completed","usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.output.len(), 1);
        let OutputItem::McpCall(call) = &acc.output[0] else {
            panic!("expected mcp_call");
        };
        let Some(McpCallError::Unknown(error)) = &call.error else {
            panic!("expected unknown MCP error payload");
        };
        assert_eq!(error["type"], "mcp_protocol_error");
        assert_eq!(error["code"], -32000);
        assert_eq!(error["message"], "boom");
    }

    #[test]
    fn test_streaming_preserves_all_documented_mcp_call_statuses() {
        let lines = vec![
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"mcp_call","id":"mcp_calling","server_label":"counter","name":"increment","arguments":"{}","status":"calling","approval_request_id":null,"output":null,"error":null}}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":1,"item":{"type":"mcp_call","id":"mcp_incomplete","server_label":"counter","name":"increment","arguments":"{}","status":"incomplete","approval_request_id":null,"output":null,"error":null}}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":2,"item":{"type":"mcp_call","id":"mcp_omitted","server_label":"counter","name":"increment","arguments":"{}","approval_request_id":null,"output":"1","error":null}}"#.to_string(),
            r#"data: {"type":"response.done","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        let statuses = acc
            .output
            .iter()
            .map(|item| match item {
                OutputItem::McpCall(call) => call.status,
                _ => panic!("expected mcp_call"),
            })
            .collect::<Vec<_>>();

        assert_eq!(
            statuses,
            vec![Some(McpCallStatus::Calling), Some(McpCallStatus::Incomplete), None]
        );
    }

    #[test]
    fn test_process_event_web_search_done_accumulates_output() {
        let lines = vec![
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"ws_1","status":"in_progress","action":{"type":"search","query":"rust","sources":[]}}}"#.to_string(),
            r#"data: {"type":"response.web_search_call.in_progress","item_id":"ws_1","output_index":0}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"rust","sources":[]}}}"#.to_string(),
            r#"data: {"type":"response.done","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.status, ResponseStatus::Completed);
        assert_eq!(acc.output.len(), 1);
        assert!(matches!(acc.output[0], OutputItem::WebSearchCall(_)));
    }

    #[test]
    fn test_process_event_completed_with_usage() {
        let mut acc = ResponseAccumulator::new("resp_1".into(), None);
        let frame = EventFrame {
            event_type: SSEEventType::ResponseCompleted,
            payload: EventPayload::Response {
                id: "resp_1".into(),
                status: "completed".into(),
                usage: Some(ResponseUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    total_tokens: 15,
                    ..Default::default()
                }),
            },
            wire: WireEvent::new("test"),
        };
        acc.process_event(&frame);
        assert_eq!(acc.status, ResponseStatus::Completed);
        assert!(acc.usage.is_some());
        assert_eq!(acc.usage.unwrap().total_tokens, 15);
    }

    #[test]
    fn test_process_event_failed_sets_error_status() {
        let mut acc = ResponseAccumulator::new("resp_1".into(), None);
        acc.process_event(&EventFrame {
            event_type: SSEEventType::ResponseFailed,
            payload: EventPayload::Response {
                id: "resp_1".into(),
                status: "failed".into(),
                usage: None,
            },
            wire: WireEvent::new("response.failed"),
        });
        assert_eq!(acc.status, ResponseStatus::Error);
    }

    #[test]
    fn test_process_event_incomplete_sets_incomplete_status() {
        let mut acc = ResponseAccumulator::new("resp_1".into(), None);
        acc.process_event(&EventFrame {
            event_type: SSEEventType::ResponseIncomplete,
            payload: EventPayload::Response {
                id: "resp_1".into(),
                status: "incomplete".into(),
                usage: None,
            },
            wire: WireEvent::new("test"),
        });
        assert_eq!(acc.status, ResponseStatus::Incomplete);
    }

    #[test]
    fn test_process_event_unknown_payload_ignored() {
        let mut acc = ResponseAccumulator::new("resp_1".into(), None);
        let frame = EventFrame {
            event_type: SSEEventType::ContentPartAdded,
            payload: EventPayload::Raw(serde_json::json!({"type": "response.content_part.added"})),
            wire: WireEvent::new("test"),
        };
        acc.process_event(&frame);
        assert_eq!(acc.response_id, "resp_1");
        assert_eq!(acc.status, ResponseStatus::InProgress);
        assert!(acc.output.is_empty());
    }

    #[test]
    fn test_accumulator_reasoning_and_message_from_sse() {
        let lines = vec![
            r#"data: {"type":"response.created","response":{"id":"resp_abc"}}"#.to_string(),
            r#"data: {"type":"response.output_item.added","item":{"id":"rs_1","type":"reasoning","summary":[]}}"#.to_string(),
            r#"data: {"type":"response.reasoning_text.delta","delta":"Let me ","item_id":"rs_1"}"#.to_string(),
            r#"data: {"type":"response.reasoning_text.delta","delta":"think.","item_id":"rs_1"}"#.to_string(),
            r#"data: {"type":"response.reasoning_text.done","text":"Let me think.","item_id":"rs_1"}"#.to_string(),
            r#"data: {"type":"response.output_item.added","item":{"id":"msg_1","type":"message"}}"#.to_string(),
            r#"data: {"type":"response.output_text.delta","delta":"Hello","item_id":"msg_1"}"#.to_string(),
            r#"data: {"type":"response.done","response":{"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.status, ResponseStatus::Completed);
        assert_eq!(acc.output.len(), 2);

        if let OutputItem::Reasoning(r) = &acc.output[0] {
            assert_eq!(r.id, "rs_1");
            assert_eq!(r.content.len(), 1);
            assert_eq!(r.content[0].text, "Let me think.");
        } else {
            panic!("expected OutputItem::Reasoning, got {:?}", acc.output[0]);
        }

        if let OutputItem::Message(msg) = &acc.output[1] {
            assert_eq!(msg.id, "msg_1");
            assert_eq!(msg.content[0].text, "Hello");
        } else {
            panic!("expected OutputItem::Message");
        }
    }

    #[test]
    fn completed_reasoning_replaces_partial_deltas_without_duplication() {
        let lines = [
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","content":[],"summary":[]}}"#.to_owned(),
            r#"data: {"type":"response.reasoning_text.delta","item_id":"rs_1","output_index":0,"content_index":0,"delta":"partial content"}"#.to_owned(),
            r#"data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"partial summary"}"#.to_owned(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","content":[{"type":"reasoning_text","text":"complete content"},{"type":"reasoning_text","text":"second content"}],"summary":[{"type":"summary_text","text":"complete summary"}],"encrypted_content":"opaque-state","status":"completed"}}"#.to_owned(),
            r#"data: {"type":"response.completed","response":{"id":"resp_1","status":"completed"}}"#.to_owned(),
        ];

        let acc = from_sse_lines(lines, None);

        assert_eq!(acc.output.len(), 1);
        assert_eq!(
            serde_json::to_value(&acc.output[0]).unwrap(),
            serde_json::json!({
                "type": "reasoning",
                "id": "rs_1",
                "content": [
                    {"type": "reasoning_text", "text": "complete content"},
                    {"type": "reasoning_text", "text": "second content"},
                ],
                "summary": [{"type": "summary_text", "text": "complete summary"}],
                "encrypted_content": "opaque-state",
                "status": "completed",
            })
        );
    }

    #[test]
    fn reasoning_done_events_keep_part_index_order() {
        let lines = [
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning"}}"#.to_owned(),
            r#"data: {"type":"response.reasoning_text.done","item_id":"rs_1","output_index":0,"content_index":1,"text":"second content"}"#.to_owned(),
            r#"data: {"type":"response.reasoning_text.done","item_id":"rs_1","output_index":0,"content_index":0,"text":"first content"}"#.to_owned(),
            r#"data: {"type":"response.reasoning_summary_text.done","item_id":"rs_1","output_index":0,"summary_index":1,"text":"second summary"}"#.to_owned(),
            r#"data: {"type":"response.reasoning_summary_text.done","item_id":"rs_1","output_index":0,"summary_index":0,"text":"first summary"}"#.to_owned(),
            r#"data: {"type":"response.completed","response":{"id":"resp_1","status":"completed"}}"#.to_owned(),
        ];

        let acc = from_sse_lines(lines, None);
        let OutputItem::Reasoning(reasoning) = &acc.output[0] else {
            panic!("expected reasoning output");
        };

        assert_eq!(
            reasoning
                .content
                .iter()
                .map(|part| part.text.as_str())
                .collect::<Vec<_>>(),
            ["first content", "second content"]
        );
        assert_eq!(
            reasoning.summary,
            [
                serde_json::json!({"type": "summary_text", "text": "first summary"}),
                serde_json::json!({"type": "summary_text", "text": "second summary"}),
            ]
        );
    }

    #[test]
    fn completed_reasoning_preserves_done_fields_when_omitted() {
        let lines = [
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning"}}"#.to_owned(),
            r#"data: {"type":"response.reasoning_text.done","item_id":"rs_1","output_index":0,"content_index":0,"text":"completed content"}"#.to_owned(),
            r#"data: {"type":"response.reasoning_summary_text.done","item_id":"rs_1","output_index":0,"summary_index":0,"text":"completed summary"}"#.to_owned(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","encrypted_content":{"token":"opaque"},"status":"completed"}}"#.to_owned(),
            r#"data: {"type":"response.completed","response":{"id":"resp_1","status":"completed"}}"#.to_owned(),
        ];

        let acc = from_sse_lines(lines, None);
        let OutputItem::Reasoning(reasoning) = &acc.output[0] else {
            panic!("expected reasoning output");
        };

        assert_eq!(reasoning.content[0].text, "completed content");
        assert_eq!(
            reasoning.summary,
            [serde_json::json!({"type": "summary_text", "text": "completed summary"})]
        );
        assert_eq!(
            reasoning.encrypted_content,
            Some(serde_json::json!({"token": "opaque"}))
        );
        assert_eq!(reasoning.status.as_deref(), Some("completed"));
    }

    #[test]
    fn completed_reasoning_null_and_empty_fields_are_authoritative_independently() {
        let content_null = [
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_content_null","type":"reasoning"}}"#.to_owned(),
            r#"data: {"type":"response.reasoning_text.done","item_id":"rs_content_null","output_index":0,"content_index":0,"text":"discarded content"}"#.to_owned(),
            r#"data: {"type":"response.reasoning_summary_text.done","item_id":"rs_content_null","output_index":0,"summary_index":0,"text":"kept summary"}"#.to_owned(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_content_null","type":"reasoning","content":null}}"#.to_owned(),
        ];
        let summary_empty = [
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_summary_empty","type":"reasoning"}}"#.to_owned(),
            r#"data: {"type":"response.reasoning_text.done","item_id":"rs_summary_empty","output_index":0,"content_index":0,"text":"kept content"}"#.to_owned(),
            r#"data: {"type":"response.reasoning_summary_text.done","item_id":"rs_summary_empty","output_index":0,"summary_index":0,"text":"discarded summary"}"#.to_owned(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_summary_empty","type":"reasoning","summary":[]}}"#.to_owned(),
        ];

        let content_null = from_sse_lines(content_null, None);
        let OutputItem::Reasoning(content_null) = &content_null.output[0] else {
            panic!("expected reasoning output");
        };
        assert!(content_null.content.is_empty());
        assert_eq!(content_null.summary[0]["text"], "kept summary");

        let summary_empty = from_sse_lines(summary_empty, None);
        let OutputItem::Reasoning(summary_empty) = &summary_empty.output[0] else {
            panic!("expected reasoning output");
        };
        assert_eq!(summary_empty.content[0].text, "kept content");
        assert!(summary_empty.summary.is_empty());
    }

    #[test]
    fn streaming_and_nonstreaming_nullable_reasoning_fields_are_equivalent() {
        let streaming = from_sse_lines(
            [r#"data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","content":null,"summary":null,"encrypted_content":null,"status":"completed"}}"#.to_owned()],
            None,
        );
        let nonstreaming = ResponseAccumulator::from_json(
            r#"{"id":"resp_1","status":"completed","output":[{"id":"rs_1","type":"reasoning","content":null,"summary":null,"encrypted_content":null,"status":"completed"}]}"#,
            None,
        )
        .unwrap();

        assert_eq!(
            serde_json::to_value(&streaming.output).unwrap(),
            serde_json::to_value(&nonstreaming.output).unwrap()
        );
    }

    #[test]
    fn done_only_reasoning_uses_output_index_order() {
        let lines = [
            r#"data: {"type":"response.output_item.added","output_index":1,"item":{"id":"msg_1","type":"message"}}"#.to_owned(),
            r#"data: {"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"delta":"answer"}"#.to_owned(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","content":[{"type":"reasoning_text","text":"thinking"}],"summary":[],"encrypted_content":null,"status":"completed"}}"#.to_owned(),
            r#"data: {"type":"response.completed","response":{"id":"resp_1","status":"completed"}}"#.to_owned(),
        ];

        let acc = from_sse_lines(lines, None);

        assert_eq!(acc.output.len(), 2);
        assert!(matches!(acc.output[0], OutputItem::Reasoning(_)));
        assert!(matches!(acc.output[1], OutputItem::Message(_)));
    }

    #[test]
    fn malformed_completed_reasoning_retains_done_fields() {
        let lines = [
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning"}}"#.to_owned(),
            r#"data: {"type":"response.reasoning_text.done","item_id":"rs_1","output_index":0,"content_index":0,"text":"completed content"}"#.to_owned(),
            r#"data: {"type":"response.reasoning_summary_text.done","item_id":"rs_1","output_index":0,"summary_index":0,"text":"completed summary"}"#.to_owned(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","content":"malformed","summary":[{"type":"summary_text","text":"ignored completion"}],"encrypted_content":"ignored"}}"#.to_owned(),
            r#"data: {"type":"response.completed","response":{"id":"resp_1","status":"completed"}}"#.to_owned(),
        ];

        let acc = from_sse_lines(lines, None);
        let OutputItem::Reasoning(reasoning) = &acc.output[0] else {
            panic!("expected reasoning output");
        };

        assert_eq!(reasoning.content[0].text, "completed content");
        assert_eq!(reasoning.summary[0]["text"], "completed summary");
        assert!(reasoning.encrypted_content.is_none());
    }

    #[test]
    fn test_accumulator_message_then_reasoning_preserves_order() {
        let lines = vec![
            r#"data: {"type":"response.created","response":{"id":"resp_abc"}}"#.to_string(),
            r#"data: {"type":"response.output_item.added","item":{"id":"msg_1","type":"message"}}"#.to_string(),
            r#"data: {"type":"response.output_text.delta","delta":"Hello","item_id":"msg_1"}"#.to_string(),
            r#"data: {"type":"response.output_item.added","item":{"id":"rs_1","type":"reasoning","summary":[]}}"#.to_string(),
            r#"data: {"type":"response.reasoning_text.done","text":"thinking...","item_id":"rs_1"}"#.to_string(),
            r#"data: {"type":"response.done","response":{"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.output.len(), 2);
        assert!(matches!(acc.output[0], OutputItem::Message(_)));
        assert!(matches!(acc.output[1], OutputItem::Reasoning(_)));
    }

    #[test]
    fn test_accumulator_reasoning_done_without_delta_uses_text() {
        let lines = vec![
            r#"data: {"type":"response.output_item.added","item":{"id":"rs_1","type":"reasoning","summary":[]}}"#.to_string(),
            r#"data: {"type":"response.reasoning_text.done","text":"done only","item_id":"rs_1"}"#.to_string(),
            r#"data: {"type":"response.done","response":{"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        if let OutputItem::Reasoning(reasoning) = &acc.output[0] {
            assert_eq!(reasoning.content.len(), 1);
            assert_eq!(reasoning.content[0].text, "done only");
        } else {
            panic!("expected reasoning output");
        }
    }

    #[test]
    fn test_accumulator_reasoning_from_json() {
        let body = serde_json::json!({
            "id": "resp_xyz",
            "status": "completed",
            "output": [
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"text": "thinking...", "type": "reasoning_text"}],
                    "encrypted_content": null,
                    "status": null
                },
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "answer", "annotations": []}]
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        });

        let acc = ResponseAccumulator::from_json(&body.to_string(), None).unwrap();
        assert_eq!(acc.output.len(), 2);
        assert!(matches!(acc.output[0], OutputItem::Reasoning(_)));
        assert!(matches!(acc.output[1], OutputItem::Message(_)));
    }

    #[test]
    fn test_blocking_preserves_all_documented_mcp_call_statuses() {
        let cases: [(Option<&str>, Option<McpCallStatus>); 3] = [
            (Some("calling"), Some(McpCallStatus::Calling)),
            (Some("incomplete"), Some(McpCallStatus::Incomplete)),
            (None, None),
        ];

        for (status, expected) in cases {
            let mut item = serde_json::json!({
                "type": "mcp_call",
                "id": "mcp_1",
                "server_label": "counter",
                "name": "increment",
                "arguments": "{}",
                "approval_request_id": null,
                "output": null,
                "error": null
            });
            if let Some(status) = status {
                item["status"] = serde_json::json!(status);
            }
            let body = serde_json::json!({
                "id": "resp_1",
                "status": "completed",
                "output": [item],
                "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
            });

            let acc = ResponseAccumulator::from_json(&body.to_string(), None).unwrap();
            assert_eq!(acc.output.len(), 1);
            let OutputItem::McpCall(call) = &acc.output[0] else {
                panic!("expected mcp_call");
            };
            assert_eq!(call.status, expected);
        }
    }

    #[test]
    fn test_function_call_accumulation_basic() {
        let mut acc = ResponseAccumulator::new("resp_1".into(), None);

        acc.process_event(&EventFrame {
            event_type: SSEEventType::OutputItemAdded,
            payload: EventPayload::OutputItemAdded {
                item_id: "fc_1".into(),
                item_type: "function_call".into(),
                output_index: 0,
                name: Some("get_weather".into()),
                namespace: Some("mcp__weather".into()),
                call_id: Some("call_abc".into()),
            },
            wire: WireEvent::new("test"),
        });

        acc.process_event(&EventFrame {
            event_type: SSEEventType::FunctionCallArgumentsDelta,
            payload: EventPayload::FunctionCallArgsDelta {
                delta: r#"{"location""#.into(),
                call_id: Some("call_abc".into()),
                item_id: "fc_1".into(),
                output_index: 0,
            },
            wire: WireEvent::new("test"),
        });

        acc.process_event(&EventFrame {
            event_type: SSEEventType::FunctionCallArgumentsDelta,
            payload: EventPayload::FunctionCallArgsDelta {
                delta: r#":"Paris"}"#.into(),
                call_id: Some("call_abc".into()),
                item_id: "fc_1".into(),
                output_index: 0,
            },
            wire: WireEvent::new("test"),
        });

        acc.process_event(&EventFrame {
            event_type: SSEEventType::FunctionCallArgumentsDone,
            payload: EventPayload::FunctionCallArgsDone {
                arguments: r#"{"location":"Paris"}"#.into(),
                call_id: Some("call_abc".into()),
                item_id: "fc_1".into(),
                name: "get_weather".into(),
                output_index: 0,
            },
            wire: WireEvent::new("test"),
        });

        acc.process_event(&EventFrame {
            event_type: SSEEventType::ResponseCompleted,
            payload: EventPayload::Response {
                id: "resp_1".into(),
                status: "completed".into(),
                usage: None,
            },
            wire: WireEvent::new("test"),
        });

        assert_eq!(acc.status, ResponseStatus::Completed);
        assert_eq!(acc.output.len(), 1);
        if let OutputItem::FunctionCall(fc) = &acc.output[0] {
            assert_eq!(fc.id, "fc_1");
            assert_eq!(fc.call_id, "call_abc");
            assert_eq!(fc.name, "get_weather");
            assert_eq!(fc.namespace.as_deref(), Some("mcp__weather"));
            assert_eq!(fc.arguments, r#"{"location":"Paris"}"#);
            assert_eq!(fc.status, MessageStatus::Completed);
        } else {
            panic!("expected FunctionCall");
        }
    }

    #[test]
    fn test_function_call_done_uses_deltas_when_arguments_empty() {
        let mut acc = ResponseAccumulator::new("resp_1".into(), None);

        acc.process_event(&EventFrame {
            event_type: SSEEventType::OutputItemAdded,
            payload: EventPayload::OutputItemAdded {
                item_id: "fc_1".into(),
                item_type: "function_call".into(),
                output_index: 0,
                name: Some("search".into()),
                namespace: None,
                call_id: Some("call_1".into()),
            },
            wire: WireEvent::new("test"),
        });

        acc.process_event(&EventFrame {
            event_type: SSEEventType::FunctionCallArgumentsDelta,
            payload: EventPayload::FunctionCallArgsDelta {
                delta: r#"{"q":"rust"}"#.into(),
                call_id: Some("call_1".into()),
                item_id: "fc_1".into(),
                output_index: 0,
            },
            wire: WireEvent::new("test"),
        });

        acc.process_event(&EventFrame {
            event_type: SSEEventType::FunctionCallArgumentsDone,
            payload: EventPayload::FunctionCallArgsDone {
                arguments: String::new(),
                call_id: Some("call_1".into()),
                item_id: "fc_1".into(),
                name: "search".into(),
                output_index: 0,
            },
            wire: WireEvent::new("test"),
        });

        acc.finalize_all();
        assert_eq!(acc.output.len(), 1);
        if let OutputItem::FunctionCall(fc) = &acc.output[0] {
            assert_eq!(fc.arguments, r#"{"q":"rust"}"#);
        } else {
            panic!("expected FunctionCall");
        }
    }

    #[test]
    fn test_function_call_multiple_parallel() {
        let mut acc = ResponseAccumulator::new("resp_1".into(), None);

        acc.process_event(&EventFrame {
            event_type: SSEEventType::OutputItemAdded,
            payload: EventPayload::OutputItemAdded {
                item_id: "fc_1".into(),
                item_type: "function_call".into(),
                output_index: 0,
                name: Some("get_weather".into()),
                namespace: None,
                call_id: Some("call_1".into()),
            },
            wire: WireEvent::new("test"),
        });
        acc.process_event(&EventFrame {
            event_type: SSEEventType::FunctionCallArgumentsDone,
            payload: EventPayload::FunctionCallArgsDone {
                arguments: r#"{"city":"NYC"}"#.into(),
                call_id: Some("call_1".into()),
                item_id: "fc_1".into(),
                name: "get_weather".into(),
                output_index: 0,
            },
            wire: WireEvent::new("test"),
        });

        acc.process_event(&EventFrame {
            event_type: SSEEventType::OutputItemAdded,
            payload: EventPayload::OutputItemAdded {
                item_id: "fc_2".into(),
                item_type: "function_call".into(),
                output_index: 1,
                name: Some("get_time".into()),
                namespace: None,
                call_id: Some("call_2".into()),
            },
            wire: WireEvent::new("test"),
        });
        acc.process_event(&EventFrame {
            event_type: SSEEventType::FunctionCallArgumentsDone,
            payload: EventPayload::FunctionCallArgsDone {
                arguments: r#"{"tz":"EST"}"#.into(),
                call_id: Some("call_2".into()),
                item_id: "fc_2".into(),
                name: "get_time".into(),
                output_index: 1,
            },
            wire: WireEvent::new("test"),
        });

        acc.process_event(&EventFrame {
            event_type: SSEEventType::ResponseCompleted,
            payload: EventPayload::Response {
                id: "resp_1".into(),
                status: "completed".into(),
                usage: None,
            },
            wire: WireEvent::new("test"),
        });

        assert_eq!(acc.output.len(), 2);
        assert!(matches!(&acc.output[0], OutputItem::FunctionCall(fc) if fc.name == "get_weather"));
        assert!(matches!(&acc.output[1], OutputItem::FunctionCall(fc) if fc.name == "get_time"));
    }

    #[test]
    fn test_function_call_interleaved_with_message() {
        let mut acc = ResponseAccumulator::new("resp_1".into(), None);

        acc.process_event(&EventFrame {
            event_type: SSEEventType::OutputItemAdded,
            payload: EventPayload::OutputItemAdded {
                item_id: "msg_1".into(),
                item_type: "message".into(),
                output_index: 0,
                name: None,
                namespace: None,
                call_id: None,
            },
            wire: WireEvent::new("test"),
        });
        acc.process_event(&EventFrame {
            event_type: SSEEventType::OutputTextDelta,
            payload: EventPayload::TextDelta {
                delta: "Let me check".into(),
                item_id: "msg_1".into(),
                output_index: 0,
                content_index: 0,
            },
            wire: WireEvent::new("test"),
        });

        acc.process_event(&EventFrame {
            event_type: SSEEventType::OutputItemAdded,
            payload: EventPayload::OutputItemAdded {
                item_id: "fc_1".into(),
                item_type: "function_call".into(),
                output_index: 1,
                name: Some("lookup".into()),
                namespace: None,
                call_id: Some("call_x".into()),
            },
            wire: WireEvent::new("test"),
        });
        acc.process_event(&EventFrame {
            event_type: SSEEventType::FunctionCallArgumentsDone,
            payload: EventPayload::FunctionCallArgsDone {
                arguments: "{}".into(),
                call_id: Some("call_x".into()),
                item_id: "fc_1".into(),
                name: "lookup".into(),
                output_index: 1,
            },
            wire: WireEvent::new("test"),
        });

        acc.process_event(&EventFrame {
            event_type: SSEEventType::ResponseCompleted,
            payload: EventPayload::Response {
                id: "resp_1".into(),
                status: "completed".into(),
                usage: None,
            },
            wire: WireEvent::new("test"),
        });

        assert_eq!(acc.output.len(), 2);
        assert!(matches!(&acc.output[0], OutputItem::Message(m) if m.content[0].text == "Let me check"));
        assert!(matches!(&acc.output[1], OutputItem::FunctionCall(fc) if fc.name == "lookup"));
    }

    #[test]
    fn test_function_call_done_updates_metadata() {
        let mut acc = ResponseAccumulator::new("resp_1".into(), None);

        acc.process_event(&EventFrame {
            event_type: SSEEventType::OutputItemAdded,
            payload: EventPayload::OutputItemAdded {
                item_id: "fc_1".into(),
                item_type: "function_call".into(),
                output_index: 0,
                name: Some("old_name".into()),
                namespace: None,
                call_id: Some("old_call".into()),
            },
            wire: WireEvent::new("test"),
        });

        acc.process_event(&EventFrame {
            event_type: SSEEventType::FunctionCallArgumentsDone,
            payload: EventPayload::FunctionCallArgsDone {
                arguments: "{}".into(),
                call_id: Some("new_call".into()),
                item_id: "fc_1".into(),
                name: "new_name".into(),
                output_index: 0,
            },
            wire: WireEvent::new("test"),
        });

        acc.finalize_all();
        if let OutputItem::FunctionCall(fc) = &acc.output[0] {
            assert_eq!(fc.call_id, "new_call");
            assert_eq!(fc.name, "new_name");
        } else {
            panic!("expected FunctionCall");
        }
    }

    #[test]
    fn test_output_item_done_restores_initially_unnamed_function_call() {
        let lines = vec![
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"","name":"","arguments":"","status":"in_progress"}}"#.to_string(),
            r#"data: {"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":"{\"input\":\"hello\"}"}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"raw_echo","arguments":"","status":"completed"}}"#.to_string(),
            r#"data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":null}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.output.len(), 1);
        let OutputItem::FunctionCall(call) = &acc.output[0] else {
            panic!("expected function_call");
        };
        assert_eq!(call.id, "fc_1");
        assert_eq!(call.call_id, "call_1");
        assert_eq!(call.name, "raw_echo");
        assert_eq!(call.arguments, r#"{"input":"hello"}"#);
        assert_eq!(call.status, MessageStatus::Completed);
    }

    #[test]
    fn test_function_call_done_matches_empty_added_id_by_output_index() {
        let lines = vec![
            r#"data: {"type":"response.output_item.added","output_index":3,"item":{"type":"function_call","id":"","call_id":"","name":"","arguments":"","status":"in_progress"}}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":3,"item":{"type":"function_call","id":"fc_done","call_id":"call_done","name":"raw_echo","arguments":"{}","status":"completed"}}"#.to_string(),
            r#"data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":null}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.output.len(), 1);
        let OutputItem::FunctionCall(call) = &acc.output[0] else {
            panic!("expected function_call");
        };
        assert_eq!(call.id, "fc_done");
        assert_eq!(call.call_id, "call_done");
        assert_eq!(call.name, "raw_echo");
    }

    #[test]
    fn test_done_only_function_call_is_completed() {
        let lines = vec![
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}","status":"completed"}}"#.to_string(),
            r#"data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":null}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.output.len(), 1);
        let OutputItem::FunctionCall(call) = &acc.output[0] else {
            panic!("expected function_call");
        };
        assert_eq!(call.id, "fc_1");
        assert_eq!(call.call_id, "call_1");
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn test_function_call_empty_item_id_generates_uuid() {
        let mut acc = ResponseAccumulator::new("resp_1".into(), None);

        acc.process_event(&EventFrame {
            event_type: SSEEventType::OutputItemAdded,
            payload: EventPayload::OutputItemAdded {
                item_id: String::new(),
                item_type: "function_call".into(),
                output_index: 0,
                name: Some("tool".into()),
                namespace: None,
                call_id: Some("c1".into()),
            },
            wire: WireEvent::new("test"),
        });

        acc.process_event(&EventFrame {
            event_type: SSEEventType::FunctionCallArgumentsDone,
            payload: EventPayload::FunctionCallArgsDone {
                arguments: "{}".into(),
                call_id: Some("c1".into()),
                item_id: String::new(),
                name: "tool".into(),
                output_index: 0,
            },
            wire: WireEvent::new("test"),
        });

        acc.finalize_all();
        if let OutputItem::FunctionCall(fc) = &acc.output[0] {
            assert!(fc.id.starts_with("fc_"), "expected fc_ prefix, got: {}", fc.id);
        } else {
            panic!("expected FunctionCall");
        }
    }

    /// Orphaned delta (no active function call for this `item_id`) is silently dropped.
    #[test]
    fn test_function_call_orphaned_delta_safe() {
        let mut acc = ResponseAccumulator::new("resp_1".into(), None);

        acc.process_event(&EventFrame {
            event_type: SSEEventType::FunctionCallArgumentsDelta,
            payload: EventPayload::FunctionCallArgsDelta {
                delta: "orphan".into(),
                call_id: None,
                item_id: String::new(),
                output_index: 0,
            },
            wire: WireEvent::new("test"),
        });

        assert!(acc.output.is_empty());
        assert!(acc.in_flight.is_empty());
    }

    #[test]
    fn test_function_call_finalized_on_response_completed() {
        let mut acc = ResponseAccumulator::new("resp_1".into(), None);

        acc.process_event(&EventFrame {
            event_type: SSEEventType::OutputItemAdded,
            payload: EventPayload::OutputItemAdded {
                item_id: "fc_1".into(),
                item_type: "function_call".into(),
                output_index: 0,
                name: Some("partial".into()),
                namespace: None,
                call_id: Some("c1".into()),
            },
            wire: WireEvent::new("test"),
        });
        acc.process_event(&EventFrame {
            event_type: SSEEventType::FunctionCallArgumentsDelta,
            payload: EventPayload::FunctionCallArgsDelta {
                delta: r#"{"x":1}"#.into(),
                call_id: Some("c1".into()),
                item_id: "fc_1".into(),
                output_index: 0,
            },
            wire: WireEvent::new("test"),
        });

        acc.process_event(&EventFrame {
            event_type: SSEEventType::ResponseCompleted,
            payload: EventPayload::Response {
                id: "resp_1".into(),
                status: "completed".into(),
                usage: None,
            },
            wire: WireEvent::new("test"),
        });

        assert_eq!(acc.output.len(), 1);
        if let OutputItem::FunctionCall(fc) = &acc.output[0] {
            assert_eq!(fc.arguments, r#"{"x":1}"#);
            assert_eq!(fc.status, MessageStatus::Completed);
        } else {
            panic!("expected FunctionCall");
        }
    }

    #[test]
    fn test_function_call_from_sse_lines() {
        let lines = vec![
            r#"data: {"type":"response.created","response":{"id":"resp_fc"}}"#.to_string(),
            r#"data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","name":"get_weather","call_id":"call_abc"}}"#.to_string(),
            r#"data: {"type":"response.function_call_arguments.delta","delta":"{\"city\":","item_id":"fc_1"}"#.to_string(),
            r#"data: {"type":"response.function_call_arguments.delta","delta":"\"SF\"}}","item_id":"fc_1"}"#.to_string(),
            r#"data: {"type":"response.function_call_arguments.done","arguments":"{\"city\":\"SF\"}","call_id":"call_abc","name":"get_weather","item_id":"fc_1"}"#.to_string(),
            r#"data: {"type":"response.done","response":{"id":"resp_fc","usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, Some("conv_1"));
        assert_eq!(acc.status, ResponseStatus::Completed);
        assert_eq!(acc.output.len(), 1);

        if let OutputItem::FunctionCall(fc) = &acc.output[0] {
            assert_eq!(fc.name, "get_weather");
            assert_eq!(fc.arguments, r#"{"city":"SF"}"#);
            assert_eq!(fc.call_id, "call_abc");
        } else {
            panic!("expected FunctionCall");
        }

        assert_eq!(acc.usage.unwrap().total_tokens, 15);
    }

    #[test]
    fn test_custom_tool_call_accumulates_freeform_input() {
        let lines = vec![
            r#"data: {"type":"response.created","response":{"id":"resp_custom"}}"#.to_string(),
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"id":"ctc_1","type":"custom_tool_call","call_id":"","name":"","input":"","status":"in_progress"}}"#.to_string(),
            r#"data: {"type":"response.custom_tool_call_input.delta","item_id":"ctc_1","output_index":0,"delta":"*** Begin"}"#.to_string(),
            r#"data: {"type":"response.custom_tool_call_input.delta","item_id":"ctc_1","output_index":0,"delta":" Patch"}"#.to_string(),
            r#"data: {"type":"response.custom_tool_call_input.done","item_id":"ctc_1","output_index":0,"input":""}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"id":"ctc_1","type":"custom_tool_call","call_id":"call_1","name":"apply_patch","input":"","status":"completed"}}"#.to_string(),
            r#"data: {"type":"response.completed","response":{"id":"resp_custom","status":"completed","usage":null}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.output.len(), 1);
        let OutputItem::CustomToolCall(call) = &acc.output[0] else {
            panic!("expected CustomToolCall");
        };
        assert_eq!(call.call_id, "call_1");
        assert_eq!(call.name, "apply_patch");
        assert_eq!(call.input, "*** Begin Patch");
        assert_eq!(call.status, Some(MessageStatus::Completed));
    }

    #[test]
    fn test_reasoning_before_done_only_custom_tool_call_preserves_order() {
        let lines = vec![
            r#"data: {"type":"response.created","response":{"id":"resp_custom"}}"#.to_string(),
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}"#.to_string(),
            r#"data: {"type":"response.reasoning_text.done","text":"thinking...","item_id":"rs_1"}"#.to_string(),
            r#"data: {"type":"response.output_item.done","output_index":1,"item":{"id":"ctc_1","type":"custom_tool_call","call_id":"call_1","name":"raw_echo","input":"hello","status":"completed"}}"#.to_string(),
            r#"data: {"type":"response.completed","response":{"id":"resp_custom","status":"completed","usage":null}}"#.to_string(),
        ];

        let acc = from_sse_lines(lines, None);
        assert_eq!(acc.output.len(), 2);
        assert!(matches!(acc.output[0], OutputItem::Reasoning(_)));
        let OutputItem::CustomToolCall(call) = &acc.output[1] else {
            panic!("expected CustomToolCall");
        };
        assert_eq!(call.call_id, "call_1");
        assert_eq!(call.name, "raw_echo");
        assert_eq!(call.input, "hello");
    }
}
