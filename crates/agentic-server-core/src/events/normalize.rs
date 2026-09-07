use serde_json::Value;

use super::types::{EventFrame, EventPayload, SSEEventType, SSEItemType, WireEvent};
use crate::utils::common::{deserialize_from_str_opt, deserialize_from_value_opt};

/// Normalize a raw SSE data line into a typed [`EventFrame`].
///
/// Expects input in the form `data: {...}` (the `data: ` prefix is required).
/// Returns `None` for non-data lines, empty lines, and the `data: [DONE]`
/// sentinel.
#[must_use]
pub fn normalize_sse_line(line: &str) -> Option<EventFrame> {
    let data_str = line.strip_prefix("data: ")?;
    if data_str == "[DONE]" {
        return None;
    }

    let json: Value = deserialize_from_str_opt(data_str)?;
    normalize_sse_value(json)
}

/// Normalizes an already parsed SSE payload.
pub(crate) fn normalize_sse_value(json: Value) -> Option<EventFrame> {
    let event_type = json
        .get("type")
        .and_then(Value::as_str)
        .map_or(SSEEventType::Other, SSEEventType::from);

    let payload = extract_payload(event_type, &json);
    let wire: WireEvent = deserialize_from_value_opt(json)?;

    Some(EventFrame {
        event_type,
        payload,
        wire,
    })
}

/// Extract a typed payload from the JSON body based on the classified event type.
fn extract_payload(event_type: SSEEventType, json: &Value) -> EventPayload {
    match event_type {
        SSEEventType::ResponseCreated
        | SSEEventType::ResponseInProgress
        | SSEEventType::ResponseCompleted
        | SSEEventType::ResponseFailed
        | SSEEventType::ResponseIncomplete => extract_response_payload(json),

        SSEEventType::OutputItemAdded => extract_output_item_added(json),
        SSEEventType::OutputItemDone => extract_output_item_done(json),

        SSEEventType::OutputTextDelta => extract_text_delta(json),
        SSEEventType::OutputTextDone => extract_text_done(json),

        SSEEventType::FunctionCallArgumentsDelta => extract_fn_call_args_delta(json),
        SSEEventType::FunctionCallArgumentsDone => extract_fn_call_args_done(json),
        SSEEventType::CustomToolCallInputDelta => extract_custom_tool_call_input_delta(json),
        SSEEventType::CustomToolCallInputDone => extract_custom_tool_call_input_done(json),

        SSEEventType::ReasoningTextDelta => extract_reasoning_text_delta(json),
        SSEEventType::ReasoningTextDone => extract_reasoning_text_done(json),
        SSEEventType::ReasoningSummaryTextDelta => extract_reasoning_summary_text_delta(json),
        SSEEventType::ReasoningSummaryTextDone => extract_reasoning_summary_text_done(json),

        SSEEventType::ContentPartAdded
        | SSEEventType::ContentPartDone
        | SSEEventType::ReasoningPartAdded
        | SSEEventType::ReasoningPartDone
        | SSEEventType::FileSearchCallSearching
        | SSEEventType::FileSearchCallCompleted
        | SSEEventType::WebSearchCallInProgress
        | SSEEventType::WebSearchCallSearching
        | SSEEventType::WebSearchCallCompleted
        | SSEEventType::McpCallInProgress
        | SSEEventType::McpCallArgumentsDelta
        | SSEEventType::McpCallArgumentsDone
        | SSEEventType::McpCallCompleted
        | SSEEventType::McpCallFailed
        | SSEEventType::McpListToolsInProgress
        | SSEEventType::McpListToolsCompleted
        | SSEEventType::McpListToolsFailed
        | SSEEventType::Other => EventPayload::Raw(json.clone()),
    }
}

fn json_str(json: &Value, key: &str) -> String {
    json[key].as_str().unwrap_or_default().to_string()
}

fn json_str_opt(json: &Value, key: &str) -> Option<String> {
    json[key].as_str().map(ToString::to_string)
}

fn json_u32(json: &Value, key: &str) -> u32 {
    u32::try_from(json[key].as_u64().unwrap_or(0)).unwrap_or(u32::MAX)
}

fn extract_response_payload(json: &Value) -> EventPayload {
    let response = &json["response"];
    EventPayload::Response {
        id: json_str(response, "id"),
        status: json_str(response, "status"),
        usage: response
            .get("usage")
            .filter(|v| !v.is_null())
            .and_then(|v| deserialize_from_value_opt(v.clone())),
    }
}

fn extract_output_item_added(json: &Value) -> EventPayload {
    let item = &json["item"];
    EventPayload::OutputItemAdded {
        item_id: json_str(item, "id"),
        item_type: SSEItemType::from(json_str(item, "type")),
        output_index: json_u32(json, "output_index"),
        name: json_str_opt(item, "name"),
        namespace: json_str_opt(item, "namespace"),
        call_id: json_str_opt(item, "call_id"),
    }
}

fn extract_output_item_done(json: &Value) -> EventPayload {
    let item = &json["item"];
    EventPayload::OutputItemDone {
        item_id: json_str(item, "id"),
        item_type: SSEItemType::from(json_str(item, "type")),
        output_index: json_u32(json, "output_index"),
        item: item.clone(),
    }
}

fn extract_text_delta(json: &Value) -> EventPayload {
    EventPayload::TextDelta {
        delta: json_str(json, "delta"),
        item_id: json_str(json, "item_id"),
        output_index: json_u32(json, "output_index"),
        content_index: json_u32(json, "content_index"),
    }
}

fn extract_text_done(json: &Value) -> EventPayload {
    EventPayload::TextDone {
        text: json_str(json, "text"),
        item_id: json_str(json, "item_id"),
        output_index: json_u32(json, "output_index"),
    }
}

fn extract_fn_call_args_delta(json: &Value) -> EventPayload {
    EventPayload::FunctionCallArgsDelta {
        delta: json_str(json, "delta"),
        call_id: json_str_opt(json, "call_id"),
        item_id: json_str(json, "item_id"),
        output_index: json_u32(json, "output_index"),
    }
}

fn extract_fn_call_args_done(json: &Value) -> EventPayload {
    EventPayload::FunctionCallArgsDone {
        arguments: json_str(json, "arguments"),
        call_id: json_str_opt(json, "call_id"),
        item_id: json_str(json, "item_id"),
        name: json_str(json, "name"),
        output_index: json_u32(json, "output_index"),
    }
}

fn extract_custom_tool_call_input_delta(json: &Value) -> EventPayload {
    EventPayload::CustomToolCallInputDelta {
        delta: json_str(json, "delta"),
        item_id: json_str(json, "item_id"),
        output_index: json_u32(json, "output_index"),
    }
}

fn extract_custom_tool_call_input_done(json: &Value) -> EventPayload {
    EventPayload::CustomToolCallInputDone {
        input: json_str(json, "input"),
        item_id: json_str(json, "item_id"),
        output_index: json_u32(json, "output_index"),
    }
}

fn extract_reasoning_text_delta(json: &Value) -> EventPayload {
    EventPayload::ReasoningTextDelta {
        delta: json_str(json, "delta"),
        item_id: json_str(json, "item_id"),
        output_index: json_u32(json, "output_index"),
        content_index: json_u32(json, "content_index"),
    }
}

fn extract_reasoning_text_done(json: &Value) -> EventPayload {
    EventPayload::ReasoningTextDone {
        text: json_str(json, "text"),
        item_id: json_str(json, "item_id"),
        output_index: json_u32(json, "output_index"),
        content_index: json_u32(json, "content_index"),
    }
}

fn extract_reasoning_summary_text_delta(json: &Value) -> EventPayload {
    EventPayload::ReasoningSummaryTextDelta {
        delta: json_str(json, "delta"),
        item_id: json_str(json, "item_id"),
        output_index: json_u32(json, "output_index"),
        summary_index: json_u32(json, "summary_index"),
    }
}

fn extract_reasoning_summary_text_done(json: &Value) -> EventPayload {
    EventPayload::ReasoningSummaryTextDone {
        text: json_str(json, "text"),
        item_id: json_str(json, "item_id"),
        output_index: json_u32(json, "output_index"),
        summary_index: json_u32(json, "summary_index"),
    }
}
