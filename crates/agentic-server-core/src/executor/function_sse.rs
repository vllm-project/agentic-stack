use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::events::{EventFrame, EventPayload, SSEEventType, SSEItemType};
use crate::executor::accumulator::AccumulatedFunctionCall;
use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::gateway_accumulator::synthetic_event;
use crate::tool::{ToolRegistry, ToolType, tool_search};
use crate::types::io::OutputItem;
use crate::utils::common::{serialize_to_string, serialize_to_value};

const MAX_PENDING_FUNCTION_BYTES: usize = 256 * 1024;

#[derive(Debug)]
enum FunctionCallShape {
    PublicFunction,
    GatewayOwned,
    Custom(CustomCallState),
    ToolSearch { internal_item_id: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamTerminalState {
    Open,
    Completed,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolSearchCallSource {
    Synthetic,
    Native,
}

#[derive(Debug)]
struct ToolSearchIdentity {
    source: ToolSearchCallSource,
    output_index: u32,
    call_id: String,
}

#[derive(Debug)]
struct CustomCallState {
    public_item_id: String,
    output_index: u32,
    emitted_input: String,
    input_start: Option<usize>,
    input_cursor: usize,
    input_done: bool,
}

#[derive(Debug, Default)]
struct PendingFunctionCall {
    output_index: u32,
    internal_item_id: Option<String>,
    frames: Vec<EventFrame>,
    bytes: usize,
}

#[derive(Debug, Default)]
pub(super) struct FunctionSseTranslation {
    pub(super) frames: Vec<EventFrame>,
    pub(super) defer_from_output_index: Option<u32>,
}

#[derive(Debug, Default)]
pub(super) struct FunctionSseOutcome {
    pub(super) unfinished_tool_search_item_ids: HashSet<String>,
}

/// Restores normalized upstream function-call SSE to the public call shape.
/// Tool routing remains in the request-scoped registry; this type borrows its
/// classification facts while owning only per-stream lifecycle state.
#[derive(Debug)]
pub(super) struct FunctionSseTranslator<'a> {
    registry: &'a ToolRegistry,
    active: HashMap<u32, FunctionCallShape>,
    pending_unnamed: HashMap<u32, PendingFunctionCall>,
    pending_bytes: usize,
    first_gateway_output_index: Option<u32>,
    active_native_tool_search: HashSet<u32>,
    tool_search_identity: Option<ToolSearchIdentity>,
    terminal: StreamTerminalState,
}

impl<'a> FunctionSseTranslator<'a> {
    pub(super) fn new(registry: &'a ToolRegistry) -> Self {
        Self {
            registry,
            active: HashMap::new(),
            pending_unnamed: HashMap::new(),
            pending_bytes: 0,
            first_gateway_output_index: None,
            active_native_tool_search: HashSet::new(),
            tool_search_identity: None,
            terminal: StreamTerminalState::Open,
        }
    }

    pub(super) fn translate(
        &mut self,
        frame: EventFrame,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        self.validate_frame(&frame)?;
        self.track_native_tool_search(&frame)?;
        self.terminal = match frame.event_type {
            SSEEventType::ResponseCompleted => StreamTerminalState::Completed,
            SSEEventType::ResponseFailed | SSEEventType::ResponseIncomplete => StreamTerminalState::Aborted,
            _ => self.terminal,
        };
        let mut translated = match &frame.payload {
            EventPayload::OutputItemAdded {
                item_id,
                item_type: SSEItemType::FunctionCall,
                output_index,
                name: Some(name),
                ..
            } => self.start_call(item_id, name, *output_index, Some(frame.clone()), call),
            EventPayload::OutputItemAdded {
                item_id,
                item_type: SSEItemType::FunctionCall,
                output_index,
                name: None,
                ..
            } => {
                let item_id = item_id.clone();
                self.buffer_unnamed(&item_id, *output_index, frame, call)
            }
            EventPayload::FunctionCallArgsDelta {
                item_id, output_index, ..
            } => self.translate_delta(item_id, *output_index, frame.clone(), call),
            EventPayload::FunctionCallArgsDone {
                item_id,
                name,
                output_index,
                ..
            } => self.finish_arguments(item_id, name, *output_index, frame.clone(), call),
            EventPayload::OutputItemDone {
                item_id,
                item_type: SSEItemType::FunctionCall,
                output_index,
                item,
            } => {
                let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                self.finish_call(item_id, name, *output_index, frame.clone(), call)
            }
            _ => Ok(FunctionSseTranslation {
                frames: vec![frame],
                defer_from_output_index: None,
            }),
        }?;
        translated.defer_from_output_index = self.defer_from_output_index();
        Ok(translated)
    }

    pub(super) fn finish(self) -> ExecutorResult<FunctionSseOutcome> {
        let unfinished_tool_search_item_ids = self.unfinished_tool_search_item_ids();
        if self.terminal != StreamTerminalState::Aborted
            && (!unfinished_tool_search_item_ids.is_empty() || !self.active_native_tool_search.is_empty())
        {
            return Err(tool_search::invalid_upstream_search_call().into());
        }
        Ok(FunctionSseOutcome {
            unfinished_tool_search_item_ids,
        })
    }

    fn start_call(
        &mut self,
        item_id: &str,
        name: &str,
        output_index: u32,
        original: Option<EventFrame>,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        let tool_type = self.registry.tool_type(name);
        if tool_type == ToolType::ToolSearch
            && let Some(original) = original.as_ref()
        {
            validate_tool_search_added(original, name)?;
        }
        match tool_type {
            ToolType::Custom => {
                let public_item_id = call.as_ref().map_or_else(
                    || crate::tool::custom::public_item_id(item_id),
                    |call| crate::tool::custom::public_item_id(&call.item.id),
                );
                self.active.insert(
                    output_index,
                    FunctionCallShape::Custom(CustomCallState {
                        public_item_id,
                        output_index,
                        emitted_input: String::new(),
                        input_start: None,
                        input_cursor: 0,
                        input_done: false,
                    }),
                );
                Ok(FunctionSseTranslation {
                    frames: call
                        .map(|call| custom_added_frame(&call))
                        .transpose()?
                        .into_iter()
                        .collect(),
                    defer_from_output_index: None,
                })
            }
            ToolType::Mcp | ToolType::WebSearch | ToolType::FileSearch | ToolType::CodeInterpreter => {
                if self.first_gateway_output_index.is_none_or(|first| output_index < first) {
                    self.first_gateway_output_index = Some(output_index);
                }
                self.active.insert(output_index, FunctionCallShape::GatewayOwned);
                Ok(FunctionSseTranslation::default())
            }
            ToolType::ToolSearch => {
                let call = call.ok_or_else(|| ExecutorError::Tool(tool_search::invalid_upstream_search_call()))?;
                let public = tool_search::started_public_call(call.item)?;
                self.start_tool_search_call(ToolSearchCallSource::Synthetic, output_index, &call.item.call_id)?;
                self.active.insert(
                    output_index,
                    FunctionCallShape::ToolSearch {
                        internal_item_id: call.item.id.clone(),
                    },
                );
                Ok(FunctionSseTranslation {
                    frames: vec![tool_search_frame(SSEEventType::OutputItemAdded, output_index, &public)?],
                    defer_from_output_index: None,
                })
            }
            ToolType::Function | ToolType::CodexNamespace => {
                self.active.insert(output_index, FunctionCallShape::PublicFunction);
                Ok(FunctionSseTranslation {
                    frames: original.into_iter().collect(),
                    defer_from_output_index: None,
                })
            }
        }
    }

    fn translate_delta(
        &mut self,
        item_id: &str,
        output_index: u32,
        original: EventFrame,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        match self.active.get_mut(&output_index) {
            Some(FunctionCallShape::PublicFunction) => Ok(FunctionSseTranslation {
                frames: vec![original],
                defer_from_output_index: None,
            }),
            Some(FunctionCallShape::GatewayOwned) => Ok(FunctionSseTranslation::default()),
            Some(FunctionCallShape::Custom(state)) => {
                let frame = match call {
                    Some(call) => incremental_custom_delta(state, call.arguments())?,
                    None => None,
                };
                Ok(FunctionSseTranslation {
                    frames: frame.into_iter().collect(),
                    defer_from_output_index: None,
                })
            }
            Some(FunctionCallShape::ToolSearch { .. }) => {
                if let Some(call) = call {
                    ensure_function_call_size(call.arguments())?;
                }
                Ok(FunctionSseTranslation::default())
            }
            None => self.buffer_unnamed(item_id, output_index, original, call),
        }
    }

    fn finish_arguments(
        &mut self,
        item_id: &str,
        name: &str,
        output_index: u32,
        original: EventFrame,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        let mut translated = self.resolve_pending(item_id, name, output_index, call)?;
        match self.active.get_mut(&output_index) {
            Some(FunctionCallShape::PublicFunction) | None => translated.frames.push(original),
            Some(FunctionCallShape::GatewayOwned) => {}
            Some(FunctionCallShape::Custom(state)) => {
                if let Some(call) = call {
                    translated.frames.extend(finish_custom_input(state, call.arguments())?);
                }
            }
            Some(FunctionCallShape::ToolSearch { .. }) => {
                let call = call.ok_or_else(|| ExecutorError::Tool(tool_search::invalid_upstream_search_call()))?;
                ensure_function_call_size(call.arguments())?;
                tool_search::validate_public_arguments(call.arguments())?;
            }
        }
        Ok(translated)
    }

    fn finish_call(
        &mut self,
        item_id: &str,
        name: &str,
        output_index: u32,
        original: EventFrame,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        let mut translated = self.resolve_pending(item_id, name, output_index, call)?;
        match self.active.remove(&output_index) {
            Some(FunctionCallShape::PublicFunction) | None => translated.frames.push(original),
            Some(FunctionCallShape::GatewayOwned) => {}
            Some(FunctionCallShape::Custom(mut state)) => {
                if let Some(call) = call {
                    translated
                        .frames
                        .extend(finish_custom_input(&mut state, call.arguments())?);
                    translated.frames.push(custom_done_frame(&state, &call)?);
                }
            }
            Some(shape @ FunctionCallShape::ToolSearch { .. }) => {
                let call = call.ok_or_else(|| ExecutorError::Tool(tool_search::invalid_upstream_search_call()))?;
                ensure_function_call_size(call.arguments())?;
                if call.item.status == crate::types::event::MessageStatus::Completed {
                    let public = tool_search::completed_public_call(call.item)?;
                    translated
                        .frames
                        .push(tool_search_frame(SSEEventType::OutputItemDone, output_index, &public)?);
                } else {
                    self.active.insert(output_index, shape);
                }
            }
        }
        Ok(translated)
    }

    fn resolve_pending(
        &mut self,
        item_id: &str,
        name: &str,
        output_index: u32,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        if self.active.contains_key(&output_index) {
            return Ok(FunctionSseTranslation::default());
        }

        let pending = self.take_pending(output_index);
        let original_added = pending.iter().find(|frame| {
            matches!(
                frame.payload,
                EventPayload::OutputItemAdded {
                    item_type: SSEItemType::FunctionCall,
                    ..
                }
            )
        });
        let mut translated = self.start_call(item_id, name, output_index, original_added.cloned(), call)?;

        for frame in pending {
            if let EventPayload::FunctionCallArgsDelta { output_index, .. } = &frame.payload {
                let delta = self.translate_delta(item_id, *output_index, frame.clone(), call)?;
                translated.frames.extend(delta.frames);
            }
        }
        Ok(translated)
    }

    fn validate_frame(&self, frame: &EventFrame) -> ExecutorResult<()> {
        let lifecycle_name = match &frame.payload {
            EventPayload::OutputItemAdded {
                item_type: SSEItemType::FunctionCall,
                name: Some(name),
                ..
            }
            | EventPayload::FunctionCallArgsDone { name, .. } => Some(name.as_str()),
            EventPayload::OutputItemDone {
                item_type: SSEItemType::FunctionCall,
                item,
                ..
            } => item.get("name").and_then(Value::as_str),
            _ => None,
        };
        if let Some(name) = lifecycle_name {
            tool_search::ensure_function_is_available(self.registry.is_withheld_function(name))?;
        }

        match &frame.payload {
            EventPayload::OutputItemDone {
                item_type: SSEItemType::FunctionCall,
                item,
                ..
            } if item
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| self.registry.tool_type(name) == ToolType::ToolSearch) =>
            {
                tool_search::strict_function_call(item)?;
            }
            EventPayload::Response { .. }
                if matches!(
                    frame.event_type,
                    SSEEventType::ResponseCompleted | SSEEventType::ResponseFailed | SSEEventType::ResponseIncomplete
                ) =>
            {
                self.validate_terminal_output(frame)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn validate_terminal_output(&self, frame: &EventFrame) -> ExecutorResult<()> {
        let Some(output) = frame
            .wire
            .rest
            .get("response")
            .and_then(|response| response.get("output"))
            .and_then(Value::as_array)
        else {
            return Ok(());
        };
        let completed = frame.event_type == SSEEventType::ResponseCompleted;
        let mut saw_tool_search_call = false;
        for item in output {
            match item.get("type").and_then(Value::as_str) {
                Some("function_call") => {
                    let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                    tool_search::ensure_function_is_available(self.registry.is_withheld_function(name))?;
                    if self.registry.tool_type(name) == ToolType::ToolSearch {
                        if saw_tool_search_call {
                            return Err(tool_search::invalid_upstream_search_call().into());
                        }
                        saw_tool_search_call = true;
                        let call = tool_search::strict_function_call(item)?;
                        ensure_function_call_size(&call.arguments)?;
                        if completed && call.status != crate::types::event::MessageStatus::Completed {
                            return Err(tool_search::invalid_upstream_search_call().into());
                        }
                    }
                }
                Some("tool_search_call") => {
                    if saw_tool_search_call {
                        return Err(tool_search::invalid_upstream_search_call().into());
                    }
                    saw_tool_search_call = true;
                    let call = tool_search::strict_native_call(item.clone())?;
                    let arguments = serialize_to_string(&call.arguments).map_err(ExecutorError::JsonError)?;
                    ensure_function_call_size(&arguments)?;
                    if completed && call.status != crate::types::tools::ToolSearchStatus::Completed {
                        return Err(tool_search::invalid_upstream_search_call().into());
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn track_native_tool_search(&mut self, frame: &EventFrame) -> ExecutorResult<()> {
        match &frame.payload {
            EventPayload::OutputItemAdded {
                item_type: SSEItemType::ToolSearchCall,
                output_index,
                ..
            } => {
                let call = validate_native_tool_search_frame(frame)?;
                self.start_tool_search_call(ToolSearchCallSource::Native, *output_index, &call.call_id)?;
                self.active_native_tool_search.insert(*output_index);
            }
            EventPayload::OutputItemDone {
                item_type: SSEItemType::ToolSearchCall,
                output_index,
                ..
            } => {
                let call = validate_native_tool_search_frame(frame)?;
                self.observe_tool_search_call(ToolSearchCallSource::Native, *output_index, &call.call_id)?;
                if call.status == crate::types::tools::ToolSearchStatus::Completed {
                    self.active_native_tool_search.remove(output_index);
                } else {
                    self.active_native_tool_search.insert(*output_index);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn start_tool_search_call(
        &mut self,
        source: ToolSearchCallSource,
        output_index: u32,
        call_id: &str,
    ) -> ExecutorResult<()> {
        if self.tool_search_identity.is_some() {
            return Err(tool_search::invalid_upstream_search_call().into());
        }
        self.tool_search_identity = Some(ToolSearchIdentity {
            source,
            output_index,
            call_id: call_id.to_owned(),
        });
        Ok(())
    }

    fn observe_tool_search_call(
        &mut self,
        source: ToolSearchCallSource,
        output_index: u32,
        call_id: &str,
    ) -> ExecutorResult<()> {
        match self.tool_search_identity.as_ref() {
            Some(identity)
                if identity.source != source
                    || identity.output_index != output_index
                    || identity.call_id != call_id =>
            {
                Err(tool_search::invalid_upstream_search_call().into())
            }
            Some(_) => Ok(()),
            None => {
                self.tool_search_identity = Some(ToolSearchIdentity {
                    source,
                    output_index,
                    call_id: call_id.to_owned(),
                });
                Ok(())
            }
        }
    }

    fn unfinished_tool_search_item_ids(&self) -> HashSet<String> {
        let active = self.active.values().filter_map(|shape| match shape {
            FunctionCallShape::ToolSearch { internal_item_id } => Some(internal_item_id.clone()),
            FunctionCallShape::PublicFunction | FunctionCallShape::GatewayOwned | FunctionCallShape::Custom(_) => None,
        });
        let pending = self
            .registry
            .tool_search_is_active()
            .then(|| {
                self.pending_unnamed
                    .values()
                    .filter_map(|pending| pending.internal_item_id.clone())
            })
            .into_iter()
            .flatten();
        active.chain(pending).collect()
    }

    fn defer_from_output_index(&self) -> Option<u32> {
        self.first_gateway_output_index
            .into_iter()
            .chain(self.pending_unnamed.values().map(|pending| pending.output_index))
            .min()
    }

    fn buffer_unnamed(
        &mut self,
        item_id: &str,
        output_index: u32,
        frame: EventFrame,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        let bytes = serialize_to_string(&frame.wire)
            .map_err(ExecutorError::JsonError)?
            .len();
        if self.pending_bytes.saturating_add(bytes) > MAX_PENDING_FUNCTION_BYTES {
            return Err(ExecutorError::StreamError(format!(
                "unnamed function-call SSE exceeded {MAX_PENDING_FUNCTION_BYTES} buffered bytes"
            )));
        }
        let pending = self
            .pending_unnamed
            .entry(output_index)
            .or_insert_with(|| PendingFunctionCall {
                output_index,
                ..PendingFunctionCall::default()
            });
        if let Some(internal_item_id) = call
            .map(|call| call.item.id.as_str())
            .filter(|item_id| !item_id.is_empty())
        {
            pending.internal_item_id = Some(internal_item_id.to_owned());
        } else if pending.internal_item_id.is_none() && !item_id.is_empty() {
            pending.internal_item_id = Some(item_id.to_owned());
        }
        pending.frames.push(frame);
        pending.bytes = pending.bytes.saturating_add(bytes);
        self.pending_bytes = self.pending_bytes.saturating_add(bytes);
        Ok(FunctionSseTranslation::default())
    }

    fn take_pending(&mut self, output_index: u32) -> Vec<EventFrame> {
        let Some(pending) = self.pending_unnamed.remove(&output_index) else {
            return Vec::new();
        };
        self.pending_bytes = self.pending_bytes.saturating_sub(pending.bytes);
        pending.frames
    }
}

fn validate_tool_search_added(frame: &EventFrame, name: &str) -> ExecutorResult<()> {
    let Some(item) = frame.wire.rest.get("item") else {
        return Err(tool_search::invalid_upstream_search_call().into());
    };
    let mut item = item.clone();
    match item.get("name") {
        None | Some(Value::Null) => {
            item.as_object_mut()
                .ok_or_else(tool_search::invalid_upstream_search_call)?
                .insert("name".to_owned(), Value::String(name.to_owned()));
        }
        Some(Value::String(_)) => {}
        Some(_) => return Err(tool_search::invalid_upstream_search_call().into()),
    }
    tool_search::strict_started_function(&item)?;
    Ok(())
}

fn validate_native_tool_search_frame(frame: &EventFrame) -> ExecutorResult<crate::types::io::ToolSearchCall> {
    let item = frame
        .wire
        .rest
        .get("item")
        .cloned()
        .ok_or_else(tool_search::invalid_upstream_search_call)?;
    let call = tool_search::strict_native_call(item)?;
    let arguments = serialize_to_string(&call.arguments).map_err(ExecutorError::JsonError)?;
    ensure_function_call_size(&arguments)?;
    if frame.event_type == SSEEventType::OutputItemAdded
        && (call.status != crate::types::tools::ToolSearchStatus::InProgress
            || !matches!(&call.arguments, Value::Object(arguments) if arguments.is_empty()))
    {
        return Err(tool_search::invalid_upstream_search_call().into());
    }
    Ok(call)
}

fn tool_search_frame(
    event_type: SSEEventType,
    output_index: u32,
    call: &crate::types::io::ToolSearchCall,
) -> ExecutorResult<EventFrame> {
    let item = serialize_to_value(&OutputItem::ToolSearchCall(call.clone())).map_err(ExecutorError::JsonError)?;
    let mut frame = synthetic_event(event_type, [("item".to_owned(), item)])?;
    frame.wire.output_index = Some(u64::from(output_index));
    Ok(frame)
}

fn custom_added_frame(call: &AccumulatedFunctionCall<'_>) -> ExecutorResult<EventFrame> {
    custom_frame(
        SSEEventType::OutputItemAdded,
        call.output_index,
        [(
            "item".to_owned(),
            serde_json::json!({
                "id": crate::tool::custom::public_item_id(&call.item.id),
                "type": "custom_tool_call",
                "status": "in_progress",
                "call_id": call.item.call_id,
                "input": "",
                "name": call.item.name,
            }),
        )],
    )
}

fn incremental_custom_delta(state: &mut CustomCallState, arguments: &str) -> ExecutorResult<Option<EventFrame>> {
    ensure_function_call_size(arguments)?;
    let Some(delta) = partial_custom_input(state, arguments)? else {
        return Ok(None);
    };
    state.emitted_input.push_str(&delta);
    custom_frame(
        SSEEventType::CustomToolCallInputDelta,
        state.output_index,
        [
            ("delta".to_owned(), Value::String(delta)),
            ("item_id".to_owned(), Value::String(state.public_item_id.clone())),
        ],
    )
    .map(Some)
}

fn finish_custom_input(state: &mut CustomCallState, arguments: &str) -> ExecutorResult<Vec<EventFrame>> {
    if state.input_done {
        return Ok(Vec::new());
    }
    ensure_function_call_size(arguments)?;
    let input = crate::tool::custom::input_from_arguments(arguments);
    let Some(remaining) = input.strip_prefix(&state.emitted_input) else {
        return Err(ExecutorError::StreamError(
            "authoritative custom tool input contradicts streamed custom tool input".to_owned(),
        ));
    };
    let remaining = (!remaining.is_empty()).then(|| remaining.to_owned());
    state.emitted_input.clone_from(&input);
    state.input_done = true;

    let mut frames = Vec::with_capacity(2);
    if let Some(delta) = remaining {
        frames.push(custom_frame(
            SSEEventType::CustomToolCallInputDelta,
            state.output_index,
            [
                ("delta".to_owned(), Value::String(delta)),
                ("item_id".to_owned(), Value::String(state.public_item_id.clone())),
            ],
        )?);
    }
    frames.push(custom_frame(
        SSEEventType::CustomToolCallInputDone,
        state.output_index,
        [
            ("input".to_owned(), Value::String(input)),
            ("item_id".to_owned(), Value::String(state.public_item_id.clone())),
        ],
    )?);
    Ok(frames)
}

fn custom_done_frame(state: &CustomCallState, call: &AccumulatedFunctionCall<'_>) -> ExecutorResult<EventFrame> {
    custom_frame(
        SSEEventType::OutputItemDone,
        state.output_index,
        [(
            "item".to_owned(),
            serde_json::json!({
                "id": state.public_item_id,
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": call.item.call_id,
                "input": state.emitted_input,
                "name": call.item.name,
            }),
        )],
    )
}

fn custom_frame(
    event_type: SSEEventType,
    output_index: u32,
    fields: impl IntoIterator<Item = (String, Value)>,
) -> ExecutorResult<EventFrame> {
    let mut frame = synthetic_event(event_type, fields)?;
    frame.wire.output_index = Some(u64::from(output_index));
    Ok(frame)
}

fn ensure_function_call_size(arguments: &str) -> ExecutorResult<()> {
    if arguments.len() > MAX_PENDING_FUNCTION_BYTES {
        return Err(ExecutorError::StreamError(format!(
            "function-call SSE exceeded {MAX_PENDING_FUNCTION_BYTES} buffered bytes"
        )));
    }
    Ok(())
}

fn partial_custom_input(state: &mut CustomCallState, arguments: &str) -> ExecutorResult<Option<String>> {
    let input_start = if let Some(input_start) = state.input_start {
        input_start
    } else {
        let Some(input_start) = custom_input_start(arguments) else {
            return Ok(None);
        };
        state.input_start = Some(input_start);
        state.input_cursor = input_start;
        input_start
    };
    if state.input_cursor < input_start || state.input_cursor > arguments.len() {
        return Ok(None);
    }
    let encoded = &arguments[state.input_cursor..];
    let end = complete_json_string_prefix(encoded);
    if end == 0 {
        return Ok(None);
    }
    let candidate = format!("\"{}\"", &encoded[..end]);
    let delta = serde_json::from_str::<String>(&candidate)
        .map_err(|error| ExecutorError::StreamError(format!("invalid custom tool input string: {error}")))?;
    state.input_cursor = state.input_cursor.saturating_add(end);
    Ok((!delta.is_empty()).then_some(delta))
}

fn complete_json_string_prefix(value: &str) -> usize {
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => return index,
            b'\\' => {
                let Some(escape) = bytes.get(index + 1) else {
                    return index;
                };
                if *escape == b'u' {
                    let unicode_end = index.saturating_add(6);
                    if unicode_end > bytes.len() {
                        return index;
                    }
                    let Some(code_unit) = json_hex_quad(&bytes[index + 2..unicode_end]) else {
                        index = unicode_end;
                        continue;
                    };
                    if (0xD800..=0xDBFF).contains(&code_unit) {
                        let pair_end = index.saturating_add(12);
                        if pair_end > bytes.len() {
                            return index;
                        }
                        index = pair_end;
                    } else {
                        index = unicode_end;
                    }
                } else {
                    index = index.saturating_add(2);
                }
            }
            _ => index = index.saturating_add(1),
        }
    }
    index
}

fn json_hex_quad(bytes: &[u8]) -> Option<u16> {
    if bytes.len() != 4 {
        return None;
    }
    bytes.iter().try_fold(0_u16, |value, byte| {
        let digit = byte.to_ascii_lowercase();
        let digit = match digit {
            b'0'..=b'9' => u16::from(digit - b'0'),
            b'a'..=b'f' => u16::from(digit - b'a' + 10),
            _ => return None,
        };
        value.checked_mul(16)?.checked_add(digit)
    })
}

fn custom_input_start(arguments: &str) -> Option<usize> {
    let original_len = arguments.len();
    let arguments = arguments.trim_start();
    let arguments = arguments.strip_prefix("{}").unwrap_or(arguments).trim_start();
    let encoded = arguments
        .strip_prefix('{')?
        .trim_start()
        .strip_prefix("\"input\"")?
        .trim_start()
        .strip_prefix(':')?
        .trim_start()
        .strip_prefix('"')?;
    Some(original_len.saturating_sub(encoded.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::accumulator::ResponseAccumulator;

    fn test_registry(tool_types: HashMap<String, ToolType>) -> ToolRegistry {
        ToolRegistry::from_tool_types(tool_types)
    }

    fn sse(value: &Value) -> String {
        format!("data: {value}")
    }

    fn translate(
        accumulator: &mut ResponseAccumulator,
        translator: &mut FunctionSseTranslator,
        value: &Value,
    ) -> FunctionSseTranslation {
        accumulator
            .process_sse_line_with_translator(&sse(value), translator)
            .expect("translation succeeds")
            .expect("SSE event")
    }

    #[test]
    fn custom_function_arguments_are_emitted_incrementally() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let mut frames = Vec::new();

        for event in [
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_custom",
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": "call_custom",
                    "name": "raw_echo",
                    "arguments": ""
                }
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "fc_custom",
                "call_id": "call_custom",
                "delta": "{\"in"
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "fc_custom",
                "call_id": "call_custom",
                "delta": "put\":\"hello "
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "fc_custom",
                "call_id": "call_custom",
                "delta": "world\"}"
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "item_id": "fc_custom",
                "call_id": "call_custom",
                "name": "raw_echo",
                "arguments": "{\"input\":\"hello world\"}"
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "fc_custom",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_custom",
                    "name": "raw_echo",
                    "arguments": "{\"input\":\"hello world\"}"
                }
            }),
        ] {
            frames.extend(translate(&mut accumulator, &mut translator, &event).frames);
        }

        assert_eq!(
            frames.iter().map(|frame| frame.event_type).collect::<Vec<_>>(),
            [
                SSEEventType::OutputItemAdded,
                SSEEventType::CustomToolCallInputDelta,
                SSEEventType::CustomToolCallInputDelta,
                SSEEventType::CustomToolCallInputDone,
                SSEEventType::OutputItemDone,
            ]
        );
        assert_eq!(frames[0].wire.rest["item"]["type"], "custom_tool_call");
        assert_eq!(frames[1].wire.rest["delta"], "hello ");
        assert_eq!(frames[2].wire.rest["delta"], "world");
        assert_eq!(frames[3].wire.rest["input"], "hello world");
        assert_eq!(frames[4].wire.rest["item"]["input"], "hello world");
    }

    #[test]
    fn custom_input_deltas_match_authoritative_done_input() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let events = [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1",
                    "name": "raw_echo", "arguments": "", "status": "in_progress"}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": "fc_1", "call_id": "call_1", "delta": "{\"input\":\"hello\""
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.done", "output_index": 0,
                "item_id": "fc_1", "call_id": "call_1", "name": "raw_echo",
                "arguments": "{\"input\":\"hello\",\"extra\":true}"
            }),
            serde_json::json!({
                "type": "response.output_item.done", "output_index": 0,
                "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1",
                    "name": "raw_echo", "arguments": "{\"input\":\"hello\",\"extra\":true}", "status": "completed"}
            }),
        ];

        let mut frames = Vec::new();
        for event in events {
            frames.extend(translate(&mut accumulator, &mut translator, &event).frames);
        }
        let deltas = frames
            .iter()
            .filter(|frame| frame.event_type == SSEEventType::CustomToolCallInputDelta)
            .filter_map(|frame| frame.wire.rest["delta"].as_str())
            .collect::<String>();
        let done = frames
            .iter()
            .find(|frame| frame.event_type == SSEEventType::CustomToolCallInputDone)
            .and_then(|frame| frame.wire.rest["input"].as_str())
            .expect("input.done");

        assert_eq!(deltas, done);
    }

    #[test]
    fn custom_input_rejects_authoritative_value_that_contradicts_deltas() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let events = [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1",
                    "name": "raw_echo", "arguments": "", "status": "in_progress"}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": "fc_1", "call_id": "call_1", "delta": "{\"input\":\"hello\"}"
            }),
        ];
        for event in events {
            translate(&mut accumulator, &mut translator, &event);
        }
        let done = serde_json::json!({
            "type": "response.function_call_arguments.done", "output_index": 0,
            "item_id": "fc_1", "call_id": "call_1", "name": "raw_echo",
            "arguments": "{\"input\":\"bye\"}"
        });

        let error = accumulator
            .process_sse_line_with_translator(&sse(&done), &mut translator)
            .expect_err("contradictory final input must fail");
        assert!(error.to_string().contains("contradicts streamed custom tool input"));
    }

    #[test]
    fn malformed_custom_input_escape_is_rejected() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let added = serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1",
                "name": "raw_echo", "arguments": "", "status": "in_progress"}
        });
        translate(&mut accumulator, &mut translator, &added);
        let delta = serde_json::json!({
            "type": "response.function_call_arguments.delta", "output_index": 0,
            "item_id": "fc_1", "call_id": "call_1", "delta": r#"{"input":"\q"#
        });

        let error = accumulator
            .process_sse_line_with_translator(&sse(&delta), &mut translator)
            .expect_err("invalid JSON string escape must fail");
        assert!(error.to_string().contains("invalid custom tool input"));
    }

    #[test]
    fn custom_input_waits_for_split_unicode_surrogate_pair() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let events = [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1",
                    "name": "raw_echo", "arguments": "", "status": "in_progress"}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": "fc_1", "call_id": "call_1", "delta": r#"{"input":"hi \uD83D"#
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": "fc_1", "call_id": "call_1", "delta": r#"\uDE00"}"#
            }),
        ];

        let frames = events
            .iter()
            .flat_map(|event| translate(&mut accumulator, &mut translator, event).frames)
            .collect::<Vec<_>>();
        let input = frames
            .iter()
            .filter(|frame| frame.event_type == SSEEventType::CustomToolCallInputDelta)
            .filter_map(|frame| frame.wire.rest["delta"].as_str())
            .collect::<String>();

        assert_eq!(input, "hi 😀");
    }

    #[test]
    fn custom_input_over_limit_is_rejected() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let added = serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1",
                "name": "raw_echo", "arguments": "", "status": "in_progress"}
        });
        translate(&mut accumulator, &mut translator, &added);
        let oversized = serde_json::json!({
            "type": "response.function_call_arguments.delta", "output_index": 0,
            "item_id": "fc_1", "call_id": "call_1",
            "delta": format!("{{\"input\":\"{}", "x".repeat(MAX_PENDING_FUNCTION_BYTES + 1))
        });

        let error = accumulator
            .process_sse_line_with_translator(&sse(&oversized), &mut translator)
            .expect_err("oversized custom input must fail");
        assert!(error.to_string().contains("function-call SSE exceeded"));
    }

    #[test]
    fn ordinary_functions_pass_through_unchanged() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("echo".to_owned(), ToolType::Function)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let event = serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 3,
            "item": {
                "id": "fc_echo",
                "type": "function_call",
                "call_id": "call_echo",
                "name": "echo",
                "arguments": ""
            }
        });

        let translated = translate(&mut accumulator, &mut translator, &event);

        assert_eq!(translated.frames.len(), 1);
        assert_eq!(translated.frames[0].event_type, SSEEventType::OutputItemAdded);
        assert_eq!(translated.frames[0].wire.rest["item"]["type"], "function_call");
        assert_eq!(translated.defer_from_output_index, None);
    }

    #[test]
    fn unnamed_function_frames_are_recovered_by_output_index_when_done_changes_id() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("echo".to_owned(), ToolType::Function)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let mut frames = Vec::new();
        let mut defer_boundaries = Vec::new();

        for event in [
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "id": "fc_transient",
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": "call_echo",
                    "arguments": ""
                }
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "item_id": "fc_transient",
                "call_id": "call_echo",
                "delta": "{\"value\":1}"
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "id": "fc_stable",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_echo",
                    "name": "echo",
                    "arguments": "{\"value\":1}"
                }
            }),
        ] {
            let translated = translate(&mut accumulator, &mut translator, &event);
            defer_boundaries.push(translated.defer_from_output_index);
            frames.extend(translated.frames);
        }

        assert_eq!(
            frames.iter().map(|frame| frame.event_type).collect::<Vec<_>>(),
            [
                SSEEventType::OutputItemAdded,
                SSEEventType::FunctionCallArgumentsDelta,
                SSEEventType::OutputItemDone,
            ]
        );
        assert_eq!(frames[0].wire.rest["item"]["id"], "fc_transient");
        assert_eq!(frames[1].wire.rest["item_id"], "fc_transient");
        assert_eq!(frames[2].wire.rest["item"]["id"], "fc_stable");
        assert_eq!(defer_boundaries, [Some(1), Some(1), None]);
    }

    #[test]
    fn unnamed_custom_function_is_recovered_by_output_index_when_done_changes_id() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let events = [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 1,
                "item": {"id": "fc_transient", "type": "function_call", "status": "in_progress",
                    "call_id": "call_echo", "arguments": ""}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 1,
                "item_id": "fc_transient", "call_id": "call_echo", "delta": "{\"input\":\"hello\"}"
            }),
            serde_json::json!({
                "type": "response.output_item.done", "output_index": 1,
                "item": {"id": "fc_stable", "type": "function_call", "status": "completed",
                    "call_id": "call_echo", "name": "raw_echo", "arguments": "{\"input\":\"hello\"}"}
            }),
        ];

        let frames = events
            .iter()
            .flat_map(|event| translate(&mut accumulator, &mut translator, event).frames)
            .collect::<Vec<_>>();

        assert_eq!(
            frames.iter().map(|frame| frame.event_type).collect::<Vec<_>>(),
            [
                SSEEventType::OutputItemAdded,
                SSEEventType::CustomToolCallInputDelta,
                SSEEventType::CustomToolCallInputDone,
                SSEEventType::OutputItemDone,
            ]
        );
        assert_eq!(frames[0].wire.rest["item"]["id"], "ctc_stable");
        assert_eq!(frames[1].wire.rest["item_id"], "ctc_stable");
        assert_eq!(frames[2].wire.rest["item_id"], "ctc_stable");
        assert_eq!(frames[3].wire.rest["item"]["id"], "ctc_stable");
    }

    #[test]
    fn parallel_unnamed_functions_with_empty_ids_remain_distinct() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([
            ("first".to_owned(), ToolType::Function),
            ("second".to_owned(), ToolType::Function),
        ]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let events = [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": "", "type": "function_call", "arguments": ""}
            }),
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 1,
                "item": {"id": "", "type": "function_call", "arguments": ""}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": "", "delta": "{\"value\":\"a\"}"
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 1,
                "item_id": "", "delta": "{\"value\":\"b\"}"
            }),
            serde_json::json!({
                "type": "response.output_item.done", "output_index": 0,
                "item": {"id": "fc_first", "type": "function_call", "call_id": "call_first",
                    "name": "first", "arguments": "{\"value\":\"a\"}", "status": "completed"}
            }),
            serde_json::json!({
                "type": "response.output_item.done", "output_index": 1,
                "item": {"id": "fc_second", "type": "function_call", "call_id": "call_second",
                    "name": "second", "arguments": "{\"value\":\"b\"}", "status": "completed"}
            }),
        ];

        let mut frames = Vec::new();
        for event in events {
            frames.extend(translate(&mut accumulator, &mut translator, &event).frames);
        }

        assert_eq!(
            frames
                .iter()
                .filter(|frame| frame.event_type == SSEEventType::OutputItemAdded)
                .count(),
            2
        );
        assert_eq!(
            frames
                .iter()
                .filter(|frame| frame.event_type == SSEEventType::OutputItemDone)
                .count(),
            2
        );
    }

    #[test]
    fn parallel_named_custom_functions_with_empty_ids_remain_distinct() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([
            ("first".to_owned(), ToolType::Custom),
            ("second".to_owned(), ToolType::Custom),
        ]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let events = [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": "", "type": "function_call", "call_id": "call_first",
                    "name": "first", "arguments": "", "status": "in_progress"}
            }),
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 1,
                "item": {"id": "", "type": "function_call", "call_id": "call_second",
                    "name": "second", "arguments": "", "status": "in_progress"}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": "", "call_id": "call_first", "delta": "{\"input\":\"a\"}"
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 1,
                "item_id": "", "call_id": "call_second", "delta": "{\"input\":\"b\"}"
            }),
        ];

        let frames = events
            .iter()
            .flat_map(|event| translate(&mut accumulator, &mut translator, event).frames)
            .collect::<Vec<_>>();
        let deltas = frames
            .iter()
            .filter(|frame| frame.event_type == SSEEventType::CustomToolCallInputDelta)
            .map(|frame| (frame.wire.output_index, frame.wire.rest["delta"].as_str()))
            .collect::<Vec<_>>();

        assert_eq!(deltas, [(Some(0), Some("a")), (Some(1), Some("b"))]);
    }

    #[test]
    fn unnamed_custom_function_with_empty_id_uses_one_public_id() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let events = [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": "", "type": "function_call", "call_id": "call_1", "arguments": ""}
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta", "output_index": 0,
                "item_id": "", "call_id": "call_1", "delta": "{\"input\":\"hello\"}"
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.done", "output_index": 0,
                "item_id": "", "call_id": "call_1", "name": "raw_echo",
                "arguments": "{\"input\":\"hello\"}"
            }),
        ];

        let frames = events
            .iter()
            .flat_map(|event| translate(&mut accumulator, &mut translator, event).frames)
            .collect::<Vec<_>>();
        let added_id = frames
            .iter()
            .find(|frame| frame.event_type == SSEEventType::OutputItemAdded)
            .and_then(|frame| frame.wire.rest["item"]["id"].as_str())
            .expect("custom item id");
        let lifecycle_ids = frames.iter().filter_map(|frame| {
            matches!(
                frame.event_type,
                SSEEventType::CustomToolCallInputDelta | SSEEventType::CustomToolCallInputDone
            )
            .then(|| frame.wire.rest["item_id"].as_str())
            .flatten()
        });

        assert!(lifecycle_ids.eq(std::iter::repeat_n(added_id, 2)));
    }

    #[test]
    fn gateway_owned_functions_are_suppressed_and_mark_the_defer_boundary() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("web_search".to_owned(), ToolType::WebSearch)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let added = serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 2,
            "item": {
                "id": "fc_search",
                "type": "function_call",
                "call_id": "call_search",
                "name": "web_search",
                "arguments": ""
            }
        });
        let delta = serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 2,
            "item_id": "fc_search",
            "call_id": "call_search",
            "delta": "{}"
        });

        let added = translate(&mut accumulator, &mut translator, &added);
        let delta = translate(&mut accumulator, &mut translator, &delta);

        assert!(added.frames.is_empty());
        assert_eq!(added.defer_from_output_index, Some(2));
        assert!(delta.frames.is_empty());
        assert_eq!(delta.defer_from_output_index, Some(2));
    }

    #[test]
    fn synthetic_tool_search_emits_public_frames_but_accumulates_function_call() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let events = [
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_search",
                    "type": "function_call",
                    "call_id": "call_search",
                    "name": "tool_search",
                    "arguments": "",
                    "status": "in_progress"
                }
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "fc_search",
                "call_id": "call_search",
                "delta": "[\"weather\",\"timezone\"]"
            }),
            serde_json::json!({
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "item_id": "fc_search",
                "call_id": "call_search",
                "name": "tool_search",
                "arguments": "[\"weather\",\"timezone\"]"
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "fc_search",
                    "type": "function_call",
                    "call_id": "call_search",
                    "name": "tool_search",
                    "arguments": "[\"weather\",\"timezone\"]",
                    "status": "completed"
                }
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": []}
            }),
        ];

        let frames = events
            .iter()
            .flat_map(|event| translate(&mut accumulator, &mut translator, event).frames)
            .collect::<Vec<_>>();
        let outcome = translator.finish().expect("completed lifecycle");
        assert!(outcome.unfinished_tool_search_item_ids.is_empty());
        assert_eq!(
            frames
                .iter()
                .filter(|frame| {
                    matches!(
                        frame.event_type,
                        SSEEventType::OutputItemAdded | SSEEventType::OutputItemDone
                    ) && frame.wire.rest["item"]["type"] == "tool_search_call"
                })
                .count(),
            2
        );
        assert!(
            frames
                .iter()
                .all(|frame| !matches!(frame.event_type, SSEEventType::FunctionCallArgumentsDelta))
        );
        let completed = frames
            .iter()
            .find(|frame| frame.event_type == SSEEventType::OutputItemDone)
            .expect("synthetic search emits a completed public item");
        assert_eq!(
            completed.wire.rest["item"]["arguments"],
            serde_json::json!(["weather", "timezone"])
        );

        let payload = accumulator.finalize("test", None, None);
        assert!(matches!(payload.output.as_slice(), [OutputItem::FunctionCall(_)]));
    }

    #[test]
    fn second_tool_search_call_is_rejected_across_native_and_synthetic_shapes() {
        let synthetic = |output_index: u32, suffix: &str| {
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    "id": format!("fc_{suffix}"),
                    "type": "function_call",
                    "call_id": format!("call_{suffix}"),
                    "name": "tool_search",
                    "arguments": "",
                    "status": "in_progress"
                }
            })
        };
        let native = |output_index: u32, suffix: &str| {
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    "id": format!("tsc_{suffix}"),
                    "type": "tool_search_call",
                    "call_id": format!("call_{suffix}"),
                    "execution": "client",
                    "arguments": {},
                    "status": "in_progress"
                }
            })
        };
        let cases = [
            (
                "synthetic then synthetic",
                synthetic(0, "first"),
                synthetic(1, "second"),
            ),
            ("native then native", native(0, "first"), native(1, "second")),
            ("synthetic then native", synthetic(0, "first"), native(1, "second")),
            ("native then synthetic", native(0, "first"), synthetic(1, "second")),
        ];

        for (case, first, second) in cases {
            let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
            let registry = test_registry(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
            let mut translator = FunctionSseTranslator::new(&registry);
            let first = accumulator
                .process_sse_line_with_translator(&sse(&first), &mut translator)
                .expect(case)
                .expect("first search event");
            assert_eq!(first.frames.len(), 1, "{case}: first added frame remains public");

            let error = accumulator
                .process_sse_line_with_translator(&sse(&second), &mut translator)
                .expect_err(case);
            assert!(
                matches!(
                    error,
                    ExecutorError::Tool(crate::tool::ToolError::InvalidUpstreamToolSearch)
                ),
                "{case}"
            );
        }
    }

    #[test]
    fn terminal_output_rejects_multiple_tool_search_calls() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let terminal = serde_json::json!({
            "type": "response.incomplete",
            "response": {
                "id": "resp_1",
                "status": "incomplete",
                "output": [
                    {
                        "id": "tsc_native",
                        "type": "tool_search_call",
                        "call_id": "call_native",
                        "execution": "client",
                        "arguments": {},
                        "status": "incomplete"
                    },
                    {
                        "id": "fc_synthetic",
                        "type": "function_call",
                        "call_id": "call_synthetic",
                        "name": "tool_search",
                        "arguments": "",
                        "status": "in_progress"
                    }
                ]
            }
        });

        let error = accumulator
            .process_sse_line_with_translator(&sse(&terminal), &mut translator)
            .expect_err("terminal response must not contain two search calls");
        assert!(matches!(
            error,
            ExecutorError::Tool(crate::tool::ToolError::InvalidUpstreamToolSearch)
        ));
    }

    #[test]
    fn native_done_cannot_reuse_synthetic_identity_after_terminal_event() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        for event in [
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_search", "type": "function_call", "call_id": "call_search",
                    "name": "tool_search", "arguments": "", "status": "in_progress"
                }
            }),
            serde_json::json!({
                "type": "response.incomplete",
                "response": {
                    "id": "resp_1", "status": "incomplete",
                    "output": [{
                        "id": "fc_search", "type": "function_call", "call_id": "call_search",
                        "name": "tool_search", "arguments": "", "status": "in_progress"
                    }]
                }
            }),
        ] {
            translate(&mut accumulator, &mut translator, &event);
        }
        let native_done = serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "tsc_search", "type": "tool_search_call", "call_id": "call_search",
                "execution": "client", "arguments": {}, "status": "completed"
            }
        });

        let error = accumulator
            .process_sse_line_with_translator(&sse(&native_done), &mut translator)
            .expect_err("native done must not reuse a synthetic call identity");
        assert!(matches!(
            error,
            ExecutorError::Tool(crate::tool::ToolError::InvalidUpstreamToolSearch)
        ));
    }

    #[test]
    fn synthetic_tool_search_rejects_non_string_name_in_buffered_added_item() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        translate(
            &mut accumulator,
            &mut translator,
            &serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_search", "type": "function_call", "call_id": "call_search",
                    "name": 7, "arguments": "", "status": "in_progress"
                }
            }),
        );
        let arguments_done = serde_json::json!({
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "item_id": "fc_search",
            "call_id": "call_search",
            "name": "tool_search",
            "arguments": "{}"
        });

        let error = accumulator
            .process_sse_line_with_translator(&sse(&arguments_done), &mut translator)
            .expect_err("a malformed buffered name must not be overwritten");
        assert!(matches!(
            error,
            ExecutorError::Tool(crate::tool::ToolError::InvalidUpstreamToolSearch)
        ));
    }

    #[test]
    fn native_tool_search_frames_pass_through_and_accumulate_natively() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::new());
        let mut translator = FunctionSseTranslator::new(&registry);
        let events = [
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "tsc_native",
                    "type": "tool_search_call",
                    "call_id": "call_search",
                    "execution": "client",
                    "arguments": {},
                    "status": "in_progress"
                }
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "tsc_native",
                    "type": "tool_search_call",
                    "call_id": "call_search",
                    "execution": "client",
                    "arguments": ["weather", "timezone"],
                    "status": "completed"
                }
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": []}
            }),
        ];

        let frames = events
            .iter()
            .flat_map(|event| translate(&mut accumulator, &mut translator, event).frames)
            .collect::<Vec<_>>();
        translator.finish().expect("completed native lifecycle");
        assert_eq!(frames[0].wire.rest["item"]["type"], "tool_search_call");
        assert_eq!(frames[1].wire.rest["item"]["type"], "tool_search_call");

        let payload = accumulator.finalize("test", None, None);
        let [OutputItem::ToolSearchCall(call)] = payload.output.as_slice() else {
            panic!("native tool_search_call must remain typed");
        };
        assert_eq!(call.arguments, serde_json::json!(["weather", "timezone"]));
    }

    #[test]
    fn oversized_native_tool_search_done_arguments_are_rejected() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::new());
        let mut translator = FunctionSseTranslator::new(&registry);
        let done = serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "tsc_native",
                "type": "tool_search_call",
                "call_id": "call_search",
                "execution": "client",
                "arguments": {"query": "x".repeat(MAX_PENDING_FUNCTION_BYTES)},
                "status": "completed"
            }
        });

        let error = accumulator
            .process_sse_line_with_translator(&sse(&done), &mut translator)
            .expect_err("oversized native arguments must fail");
        assert!(error.to_string().contains("function-call SSE exceeded"));
    }

    #[test]
    fn oversized_synthetic_tool_search_done_arguments_are_rejected() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let done = serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "fc_search",
                "type": "function_call",
                "call_id": "call_search",
                "name": "tool_search",
                "arguments": format!("{{\"query\":\"{}\"}}", "x".repeat(MAX_PENDING_FUNCTION_BYTES)),
                "status": "completed"
            }
        });

        let error = accumulator
            .process_sse_line_with_translator(&sse(&done), &mut translator)
            .expect_err("oversized synthetic arguments must fail");
        assert!(error.to_string().contains("function-call SSE exceeded"));
    }

    #[test]
    fn successful_eof_rejects_unfinished_synthetic_and_native_searches() {
        let synthetic_registry = test_registry(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
        let mut synthetic_accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut synthetic = FunctionSseTranslator::new(&synthetic_registry);
        translate(
            &mut synthetic_accumulator,
            &mut synthetic,
            &serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_search", "type": "function_call", "call_id": "call_search",
                    "name": "tool_search", "arguments": "", "status": "in_progress"
                }
            }),
        );
        assert!(synthetic.finish().is_err());

        let native_registry = test_registry(HashMap::new());
        let mut native_accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let mut native = FunctionSseTranslator::new(&native_registry);
        translate(
            &mut native_accumulator,
            &mut native,
            &serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "tsc_native", "type": "tool_search_call", "call_id": "call_search",
                    "execution": "client", "arguments": {}, "status": "in_progress"
                }
            }),
        );
        assert!(native.finish().is_err());
    }

    #[test]
    fn incomplete_stream_preserves_unfinished_synthetic_search() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        let mut public_frames = Vec::new();
        for event in [
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_search", "type": "function_call", "call_id": "call_search",
                    "name": "tool_search", "arguments": "", "status": "in_progress"
                }
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "fc_search", "type": "function_call", "call_id": "call_search",
                    "name": "tool_search", "arguments": "{\"query\":", "status": "in_progress"
                }
            }),
            serde_json::json!({
                "type": "response.incomplete",
                "response": {"id": "resp_1", "status": "incomplete", "output": []}
            }),
        ] {
            public_frames.extend(translate(&mut accumulator, &mut translator, &event).frames);
        }

        let outcome = translator.finish().expect("aborted lifecycle may remain unfinished");
        assert_eq!(
            outcome.unfinished_tool_search_item_ids,
            HashSet::from(["fc_search".to_owned()])
        );
        let mut payload = accumulator.finalize("test", None, None);
        crate::tool::ToolSearchHandler::normalize_response_output(
            &registry,
            &mut payload.output,
            crate::types::event::ResponseStatus::Incomplete,
            &outcome.unfinished_tool_search_item_ids,
        )
        .expect("unfinished synthetic call is preserved as incomplete");
        let [OutputItem::ToolSearchCall(call)] = payload.output.as_slice() else {
            panic!("unfinished synthetic call must use the public tool-search shape");
        };
        let added = public_frames
            .iter()
            .find(|frame| frame.event_type == SSEEventType::OutputItemAdded)
            .expect("public added frame");
        assert_eq!(call.id, added.wire.rest["item"]["id"]);
        assert_eq!(call.call_id, added.wire.rest["item"]["call_id"]);
        assert_eq!(call.arguments, serde_json::json!({}));
        assert_eq!(call.status, crate::types::tools::ToolSearchStatus::Incomplete);
        assert!(payload.output[0].to_input_item().is_none());
    }

    #[test]
    fn incomplete_stream_preserves_native_search_and_terminal_item_parity() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::new());
        let mut translator = FunctionSseTranslator::new(&registry);
        let mut public_frames = Vec::new();
        for event in [
            serde_json::json!({
                "type": "response.output_item.added", "output_index": 0,
                "item": {
                    "id": "tsc_native", "type": "tool_search_call", "call_id": "call_search",
                    "execution": "client", "arguments": {}, "status": "in_progress"
                }
            }),
            serde_json::json!({
                "type": "response.output_item.done", "output_index": 0,
                "item": {
                    "id": "tsc_native", "type": "tool_search_call", "call_id": "call_search",
                    "execution": "client", "arguments": {"query": "weather"}, "status": "incomplete"
                }
            }),
            serde_json::json!({
                "type": "response.incomplete",
                "response": {
                    "id": "resp_1", "status": "incomplete",
                    "output": [{
                        "id": "tsc_native", "type": "tool_search_call", "call_id": "call_search",
                        "execution": "client", "arguments": {"query": "weather"}, "status": "incomplete"
                    }]
                }
            }),
        ] {
            public_frames.extend(translate(&mut accumulator, &mut translator, &event).frames);
        }

        let outcome = translator.finish().expect("incomplete native lifecycle");
        let mut payload = accumulator.finalize("test", None, None);
        crate::tool::ToolSearchHandler::normalize_response_output(
            &registry,
            &mut payload.output,
            crate::types::event::ResponseStatus::Incomplete,
            &outcome.unfinished_tool_search_item_ids,
        )
        .expect("incomplete native call remains public");
        let [OutputItem::ToolSearchCall(call)] = payload.output.as_slice() else {
            panic!("native call must remain typed");
        };
        let done = public_frames
            .iter()
            .find(|frame| frame.event_type == SSEEventType::OutputItemDone)
            .expect("public done frame");
        assert_eq!(
            serde_json::to_value(&payload.output[0]).unwrap(),
            done.wire.rest["item"]
        );
        assert_eq!(call.status, crate::types::tools::ToolSearchStatus::Incomplete);
        assert!(payload.output[0].to_input_item().is_none());
    }

    #[test]
    fn aborted_stream_discards_unnamed_search_candidate_with_empty_raw_id() {
        for (terminal_event, terminal_status, response_status) in [
            (
                "response.incomplete",
                "incomplete",
                crate::types::event::ResponseStatus::Incomplete,
            ),
            ("response.failed", "failed", crate::types::event::ResponseStatus::Error),
        ] {
            let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
            let registry = test_registry(HashMap::from([("tool_search".to_owned(), ToolType::ToolSearch)]));
            let mut translator = FunctionSseTranslator::new(&registry);
            for event in [
                serde_json::json!({
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "id": "", "type": "function_call", "call_id": "call_search",
                        "arguments": "", "status": "in_progress"
                    }
                }),
                serde_json::json!({
                    "type": terminal_event,
                    "response": {"id": "resp_1", "status": terminal_status, "output": []}
                }),
            ] {
                translate(&mut accumulator, &mut translator, &event);
            }

            let outcome = translator
                .finish()
                .expect("aborted lifecycle may leave the call unnamed");
            assert_eq!(outcome.unfinished_tool_search_item_ids.len(), 1);
            let internal_item_id = outcome
                .unfinished_tool_search_item_ids
                .iter()
                .next()
                .expect("accumulator-generated item id");
            assert!(internal_item_id.starts_with("fc_"));

            let mut payload = accumulator.finalize("test", None, None);
            let [OutputItem::FunctionCall(call)] = payload.output.as_slice() else {
                panic!("unfinished unnamed call must be accumulated as a function call");
            };
            assert!(call.name.is_empty());
            assert_eq!(&call.id, internal_item_id);
            crate::tool::ToolSearchHandler::normalize_response_output(
                &registry,
                &mut payload.output,
                response_status,
                &outcome.unfinished_tool_search_item_ids,
            )
            .expect("unfinished unnamed search candidate is discarded");
            assert!(payload.output.is_empty());
        }
    }

    #[test]
    fn aborted_stream_discards_unfinished_native_search() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::new());
        let mut translator = FunctionSseTranslator::new(&registry);
        for event in [
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "tsc_native", "type": "tool_search_call", "call_id": "call_search",
                    "execution": "client", "arguments": {}, "status": "in_progress"
                }
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "tsc_native", "type": "tool_search_call", "call_id": "call_search",
                    "execution": "client", "arguments": {}, "status": "incomplete"
                }
            }),
            serde_json::json!({
                "type": "response.failed",
                "response": {"id": "resp_1", "status": "failed", "output": []}
            }),
        ] {
            translate(&mut accumulator, &mut translator, &event);
        }

        let outcome = translator
            .finish()
            .expect("aborted native lifecycle may remain unfinished");
        assert!(outcome.unfinished_tool_search_item_ids.is_empty());
        let mut payload = accumulator.finalize("test", None, None);
        crate::tool::ToolSearchHandler::normalize_response_output(
            &registry,
            &mut payload.output,
            crate::types::event::ResponseStatus::Error,
            &outcome.unfinished_tool_search_item_ids,
        )
        .expect("unfinished native call is discarded");
        assert!(payload.output.is_empty());
    }

    #[test]
    fn unresolved_ordinary_function_does_not_become_tool_search_failure() {
        let mut accumulator = ResponseAccumulator::new("resp_1".to_owned(), None);
        let registry = test_registry(HashMap::from([("echo".to_owned(), ToolType::Function)]));
        let mut translator = FunctionSseTranslator::new(&registry);
        translate(
            &mut accumulator,
            &mut translator,
            &serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_ordinary", "type": "function_call", "call_id": "call_ordinary",
                    "arguments": "", "status": "in_progress"
                }
            }),
        );

        translator
            .finish()
            .expect("ordinary unnamed call is not a search candidate");
    }
}
