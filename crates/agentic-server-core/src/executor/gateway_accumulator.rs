use crate::events::{EventFrame, EventPayload, SSEEventType, WireEvent, normalize_sse_line};
use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::types::request_response::ResponsePayload;
use crate::utils::common::{serialize_to_string, serialize_to_value};
use serde_json::Value;

#[derive(Clone)]
pub struct GatewayStreamAccumulator {
    next_sequence_number: u64,
    emitted_created: bool,
    emitted_in_progress: bool,
}

pub(super) struct StreamEvent {
    pub(super) content: String,
    pub(super) sequence_number: u64,
}

impl GatewayStreamAccumulator {
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_sequence_number: 0,
            emitted_created: false,
            emitted_in_progress: false,
        }
    }

    pub fn process_sse_line(&mut self, line: &str, output_offset: usize) -> Option<EventFrame> {
        let mut frame = normalize_sse_line(line)?;
        self.process_event(&mut frame, output_offset).then_some(frame)
    }

    #[must_use]
    pub fn process_event(&mut self, frame: &mut EventFrame, output_offset: usize) -> bool {
        if !self.should_emit_lifecycle(frame.event_type) {
            return false;
        }
        self.stamp_event(frame, output_offset);
        true
    }

    fn stamp_event(&mut self, frame: &mut EventFrame, output_offset: usize) {
        frame.wire.sequence_number = Some(self.take_sequence_number());
        rebase_output_index(&mut frame.wire, output_offset);
    }

    pub(crate) fn terminal_response_chunk(&mut self, payload: &ResponsePayload) -> ExecutorResult<String> {
        let mut frame = terminal_response_frame(payload)?;
        self.stamp_event(&mut frame, 0);
        serialize_sse_frame(&frame)
    }

    pub(crate) fn executor_error_chunk(&mut self, error: &ExecutorError) -> String {
        let mut frame = executor_error_frame(error);
        self.stamp_event(&mut frame, 0);
        serialize_sse_frame(&frame)
            .unwrap_or_else(|_| error_sse_chunk(&error.to_string(), frame.sequence_number().unwrap_or(0)))
    }

    fn should_emit_lifecycle(&mut self, event_type: SSEEventType) -> bool {
        match event_type {
            SSEEventType::ResponseCreated => take_once(&mut self.emitted_created),
            SSEEventType::ResponseInProgress => take_once(&mut self.emitted_in_progress),
            _ => true,
        }
    }

    fn take_sequence_number(&mut self) -> u64 {
        let sequence_number = self.next_sequence_number;
        self.next_sequence_number = self.next_sequence_number.saturating_add(1);
        sequence_number
    }
}

impl Default for GatewayStreamAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

fn take_once(already_taken: &mut bool) -> bool {
    if *already_taken {
        false
    } else {
        *already_taken = true;
        true
    }
}

fn rebase_output_index(wire: &mut WireEvent, output_offset: usize) {
    let Some(offset) = u64::try_from(output_offset).ok().filter(|offset| *offset > 0) else {
        return;
    };
    if let Some(index) = wire.output_index {
        wire.output_index = Some(index.saturating_add(offset));
    }
}

fn terminal_response_frame(payload: &ResponsePayload) -> ExecutorResult<EventFrame> {
    let event_type = match payload.terminal_event_type() {
        "response.incomplete" => SSEEventType::ResponseIncomplete,
        "response.failed" => SSEEventType::ResponseFailed,
        "response.in_progress" => SSEEventType::ResponseInProgress,
        _ => SSEEventType::ResponseCompleted,
    };
    let mut rest = serde_json::Map::new();
    rest.insert(
        "response".to_owned(),
        serialize_to_value(payload).map_err(ExecutorError::JsonError)?,
    );
    EventFrame::synthetic(event_type, rest)
        .ok_or_else(|| ExecutorError::StreamError("terminal response event has no wire representation".to_owned()))
}

fn executor_error_frame(error: &ExecutorError) -> EventFrame {
    let mut wire = WireEvent::new("error");
    wire.rest
        .insert("status".to_owned(), serde_json::json!(error.http_status().as_u16()));
    wire.rest.insert("error".to_owned(), error.response_error());
    EventFrame {
        event_type: SSEEventType::Other,
        payload: EventPayload::None,
        wire,
    }
}

pub(super) fn error_sse_chunk(message: &str, sequence_number: u64) -> String {
    let code = "server_error";
    let mut wire = WireEvent::new("error");
    wire.sequence_number = Some(sequence_number);
    wire.rest.insert("status".to_owned(), serde_json::json!(500));
    wire.rest.insert(
        "error".to_owned(),
        serde_json::json!({
            "message": message,
            "type": code,
            "code": code,
        }),
    );
    let frame = EventFrame {
        event_type: SSEEventType::Other,
        payload: EventPayload::None,
        wire,
    };
    serialize_sse_frame(&frame).unwrap_or_else(|_| {
        format!("event: error\ndata: {{\"type\":\"error\",\"status\":500,\"sequence_number\":{sequence_number}}}\n\n")
    })
}

pub(super) fn synthetic_event(
    event_type: SSEEventType,
    rest: impl IntoIterator<Item = (String, Value)>,
) -> ExecutorResult<EventFrame> {
    EventFrame::synthetic(event_type, rest.into_iter().collect())
        .ok_or_else(|| ExecutorError::StreamError("synthetic event has no wire representation".to_owned()))
}

pub(super) fn emit_sse_frame(
    sender: &tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    frame: &EventFrame,
) -> ExecutorResult<()> {
    let sequence_number = frame
        .sequence_number()
        .ok_or_else(|| ExecutorError::StreamError("stream event has no sequence number".to_owned()))?;
    sender
        .send(StreamEvent {
            content: serialize_sse_frame(frame)?,
            sequence_number,
        })
        .map_err(|_| ExecutorError::StreamError("stream receiver closed while emitting gateway event".to_owned()))
}

fn serialize_sse_frame(frame: &EventFrame) -> ExecutorResult<String> {
    let event_json = serialize_to_string(&frame.wire).map_err(ExecutorError::JsonError)?;
    let event_name =
        frame.wire.event_type.as_deref().filter(|event_name| {
            !event_name.is_empty() && !event_name.bytes().any(|byte| matches!(byte, b'\r' | b'\n'))
        });
    let event_name_len = event_name.map_or(0, str::len);
    let mut chunk = String::with_capacity(event_name_len + event_json.len() + 16);
    if let Some(event_name) = event_name {
        chunk.push_str("event: ");
        chunk.push_str(event_name);
        chunk.push('\n');
    }
    chunk.push_str("data: ");
    chunk.push_str(&event_json);
    chunk.push_str("\n\n");
    Ok(chunk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StorageError;
    use crate::tool::ToolError;

    fn parse_named_sse_event(chunk: &str) -> (&str, serde_json::Value) {
        let body = chunk.strip_suffix("\n\n").expect("SSE event terminator");
        let (event_line, data_line) = body.split_once('\n').expect("named SSE event and data lines");
        let event_name = event_line.strip_prefix("event: ").expect("SSE event prefix");
        let data = data_line.strip_prefix("data: ").expect("SSE data prefix");
        let event = serde_json::from_str(data).expect("valid event JSON");
        (event_name, event)
    }

    #[test]
    fn process_sse_line_numbers_and_rebases_output_index() {
        let mut accumulator = GatewayStreamAccumulator::new();
        let frame = accumulator
            .process_sse_line(
                r#"data: {"type":"response.output_text.delta","output_index":2,"delta":"hi"}"#,
                3,
            )
            .expect("line should normalize");

        assert_eq!(frame.sequence_number(), Some(0));
        assert_eq!(frame.wire.sequence_number, Some(0));
        assert_eq!(frame.wire.output_index, Some(5));
        assert_eq!(frame.wire.rest["delta"], "hi");
    }

    #[test]
    fn error_sse_chunk_escapes_error_messages() {
        let chunk = error_sse_chunk("task failed: \"unexpected\"\nretry", 7);
        let (event_name, event) = parse_named_sse_event(&chunk);

        assert_eq!(event_name, "error");
        assert_eq!(event["type"], "error");
        assert_eq!(event["sequence_number"], 7);
        assert_eq!(event["error"]["message"], "task failed: \"unexpected\"\nretry");
    }

    #[test]
    fn executor_conflict_sse_chunk_uses_client_conflict_contract() {
        let mut accumulator = GatewayStreamAccumulator::new();
        let error = ExecutorError::Persistence(Box::new(ExecutorError::ConversationLocked {
            source: StorageError::ConversationConflict {
                conversation_id: "conv_test".to_owned(),
            },
        }));
        let chunk = accumulator.executor_error_chunk(&error);
        let (event_name, event) = parse_named_sse_event(&chunk);

        assert_eq!(event_name, "error");
        assert_eq!(event["type"], "error");
        assert_eq!(event["status"], 400);
        assert_eq!(
            event["error"],
            serde_json::json!({
                "message": "conversation changed while the response was being generated; retry the request",
                "type": "invalid_request_error",
                "code": "conversation_locked",
                "param": "conversation"
            })
        );
    }

    #[test]
    fn missing_tool_output_sse_chunk_matches_openai_error_contract() {
        let mut accumulator = GatewayStreamAccumulator::new();
        let error = ExecutorError::Tool(ToolError::MissingOutput {
            call_id: "call_test".to_owned(),
        });
        let chunk = accumulator.executor_error_chunk(&error);
        let (event_name, event) = parse_named_sse_event(&chunk);

        assert_eq!(event_name, "error");
        assert_eq!(event["status"], 400);
        assert_eq!(
            event["error"],
            serde_json::json!({
                "message": "No tool output found for function call call_test.",
                "type": "invalid_request_error",
                "param": "input",
                "code": null
            })
        );
    }

    #[test]
    fn emits_in_progress_terminal_event_after_lifecycle_event() {
        let mut accumulator = GatewayStreamAccumulator::new();
        accumulator
            .process_sse_line(r#"data: {"type":"response.in_progress"}"#, 0)
            .expect("first lifecycle event should be emitted");

        let payload: ResponsePayload = serde_json::from_value(serde_json::json!({
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "test",
            "status": "in_progress",
            "output": [],
            "usage": null,
            "incomplete_details": null,
            "error": null,
            "previous_response_id": null,
            "conversation_id": null,
            "instructions": null
        }))
        .expect("valid response payload");

        let chunk = accumulator
            .terminal_response_chunk(&payload)
            .expect("terminal event serializes");
        assert!(chunk.starts_with("event: response.in_progress\n"));
        assert!(chunk.contains("\"type\":\"response.in_progress\""));
        assert!(chunk.contains("\"sequence_number\":1"));
    }

    #[test]
    fn serialize_sse_frame_uses_wire_event_type_as_event_name() {
        let frame = synthetic_event(SSEEventType::ResponseCreated, []).expect("synthetic event");
        let chunk = serialize_sse_frame(&frame).expect("event serializes");
        let (event_name, event) = parse_named_sse_event(&chunk);

        assert_eq!(event_name, "response.created");
        assert_eq!(event["type"], event_name);
    }

    #[test]
    fn serialize_sse_frame_omits_invalid_event_name() {
        let frame = EventFrame {
            event_type: SSEEventType::Other,
            payload: EventPayload::None,
            wire: WireEvent::new("error\nevent: injected"),
        };
        let chunk = serialize_sse_frame(&frame).expect("event serializes");

        assert!(chunk.starts_with("data: "));
        assert!(!chunk.contains("\nevent: injected\n"));
    }
}
