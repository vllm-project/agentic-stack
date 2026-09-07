use std::collections::HashMap;

use serde_json::Value;

use crate::events::{EventFrame, EventPayload, SSEEventType, SSEItemType};
use crate::executor::accumulator::AccumulatedFunctionCall;
use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::gateway_accumulator::synthetic_event;
use crate::tool::ToolType;
use crate::utils::common::serialize_to_string;

const MAX_PENDING_FUNCTION_BYTES: usize = 256 * 1024;

#[derive(Debug)]
enum FunctionCallShape {
    PublicFunction,
    GatewayOwned,
    Custom(CustomCallState),
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
    frames: Vec<EventFrame>,
    bytes: usize,
}

#[derive(Debug, Default)]
pub(super) struct FunctionSseTranslation {
    pub(super) frames: Vec<EventFrame>,
    pub(super) defer_from_output_index: Option<u32>,
}

/// Restores normalized upstream function-call SSE to the public call shape.
/// Tool routing remains outside this type; it receives only the request's
/// model-visible name-to-type mapping.
#[derive(Debug, Default)]
pub(super) struct FunctionSseTranslator {
    tool_types: HashMap<String, ToolType>,
    active: HashMap<u32, FunctionCallShape>,
    pending_unnamed: HashMap<u32, PendingFunctionCall>,
    pending_bytes: usize,
    first_gateway_output_index: Option<u32>,
}

impl FunctionSseTranslator {
    pub(super) fn new(tool_types: HashMap<String, ToolType>) -> Self {
        Self {
            tool_types,
            ..Self::default()
        }
    }

    pub(super) fn translate(
        &mut self,
        frame: EventFrame,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        let mut translated = match &frame.payload {
            EventPayload::OutputItemAdded {
                item_id,
                item_type: SSEItemType::FunctionCall,
                output_index,
                name: Some(name),
                ..
            } => self.start_call(item_id, name, *output_index, Some(frame.clone()), call),
            EventPayload::OutputItemAdded {
                item_id: _,
                item_type: SSEItemType::FunctionCall,
                output_index,
                name: None,
                ..
            } => self.buffer_unnamed(*output_index, frame),
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

    fn start_call(
        &mut self,
        item_id: &str,
        name: &str,
        output_index: u32,
        original: Option<EventFrame>,
        call: Option<AccumulatedFunctionCall<'_>>,
    ) -> ExecutorResult<FunctionSseTranslation> {
        match self.tool_type(name) {
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
        _item_id: &str,
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
            None => self.buffer_unnamed(output_index, original),
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

    fn tool_type(&self, name: &str) -> ToolType {
        self.tool_types.get(name).copied().unwrap_or(ToolType::Function)
    }

    fn defer_from_output_index(&self) -> Option<u32> {
        self.first_gateway_output_index
            .into_iter()
            .chain(self.pending_unnamed.values().map(|pending| pending.output_index))
            .min()
    }

    fn buffer_unnamed(&mut self, output_index: u32, frame: EventFrame) -> ExecutorResult<FunctionSseTranslation> {
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
        let mut translator = FunctionSseTranslator::new(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
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
        let mut translator = FunctionSseTranslator::new(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
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
        let mut translator = FunctionSseTranslator::new(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
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
        let mut translator = FunctionSseTranslator::new(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
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
        let mut translator = FunctionSseTranslator::new(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
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
        let mut translator = FunctionSseTranslator::new(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
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
        let mut translator = FunctionSseTranslator::new(HashMap::from([("echo".to_owned(), ToolType::Function)]));
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
        let mut translator = FunctionSseTranslator::new(HashMap::from([("echo".to_owned(), ToolType::Function)]));
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
        let mut translator = FunctionSseTranslator::new(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
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
        let mut translator = FunctionSseTranslator::new(HashMap::from([
            ("first".to_owned(), ToolType::Function),
            ("second".to_owned(), ToolType::Function),
        ]));
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
        let mut translator = FunctionSseTranslator::new(HashMap::from([
            ("first".to_owned(), ToolType::Custom),
            ("second".to_owned(), ToolType::Custom),
        ]));
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
        let mut translator = FunctionSseTranslator::new(HashMap::from([("raw_echo".to_owned(), ToolType::Custom)]));
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
        let mut translator =
            FunctionSseTranslator::new(HashMap::from([("web_search".to_owned(), ToolType::WebSearch)]));
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
}
