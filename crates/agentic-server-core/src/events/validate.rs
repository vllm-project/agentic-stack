use serde::Deserialize;
use serde_json::{Map, Value};
use thiserror::Error;

use super::{EventFrame, SSEEventType, SSEItemType};
use crate::types::io::OutputItem;

#[derive(Debug, Error)]
#[error("{0}")]
pub(crate) struct EventError(String);

#[derive(Debug)]
pub(crate) struct ValidatedFrame<'a> {
    pub(crate) item: Option<ValidatedItem<'a>>,
}

#[derive(Debug)]
pub(crate) struct ValidatedItem<'a> {
    pub(crate) item_id: &'a str,
    pub(crate) output_index: u32,
    pub(crate) item_type: SSEItemType,
    pub(crate) done_item: Option<OutputItem>,
}

/// Validates the stateless wire-format requirements of one normalized frame.
pub(crate) fn validate_frame(frame: &EventFrame) -> Result<ValidatedFrame<'_>, EventError> {
    let event_name = frame
        .wire
        .event_type
        .as_deref()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| invalid("streaming event has no valid 'type'"))?;

    match frame.event_type {
        SSEEventType::ResponseCreated
        | SSEEventType::ResponseInProgress
        | SSEEventType::ResponseCompleted
        | SSEEventType::ResponseFailed
        | SSEEventType::ResponseIncomplete => {
            validate_response_event(frame, event_name)?;
            Ok(ValidatedFrame { item: None })
        }
        SSEEventType::OutputItemAdded => {
            validate_output_item(frame, event_name, false).map(|item| ValidatedFrame { item: Some(item) })
        }
        SSEEventType::OutputItemDone => {
            validate_output_item(frame, event_name, true).map(|item| ValidatedFrame { item: Some(item) })
        }
        SSEEventType::Other => Ok(ValidatedFrame { item: None }),
        event_type => {
            let output_index = required_output_index(frame, event_name)?;
            let item_id = required_str(&frame.wire.rest, "item_id", event_name)?;
            validate_event_fields(&frame.wire.rest, event_type, event_name)?;
            let item_type = expected_item_type(event_type).ok_or_else(|| {
                invalid(format!(
                    "upstream output item type for event '{event_name}' is unsupported"
                ))
            })?;
            Ok(ValidatedFrame {
                item: Some(ValidatedItem {
                    item_id,
                    output_index,
                    item_type,
                    done_item: None,
                }),
            })
        }
    }
}

fn expected_item_type(event_type: SSEEventType) -> Option<SSEItemType> {
    match event_type {
        SSEEventType::OutputTextDelta
        | SSEEventType::OutputTextDone
        | SSEEventType::ContentPartAdded
        | SSEEventType::ContentPartDone => Some(SSEItemType::Message),
        SSEEventType::FunctionCallArgumentsDelta | SSEEventType::FunctionCallArgumentsDone => {
            Some(SSEItemType::FunctionCall)
        }
        SSEEventType::CustomToolCallInputDelta | SSEEventType::CustomToolCallInputDone => {
            Some(SSEItemType::CustomToolCall)
        }
        SSEEventType::ReasoningTextDelta
        | SSEEventType::ReasoningTextDone
        | SSEEventType::ReasoningPartAdded
        | SSEEventType::ReasoningPartDone
        | SSEEventType::ReasoningSummaryTextDelta
        | SSEEventType::ReasoningSummaryTextDone => Some(SSEItemType::Reasoning),
        SSEEventType::WebSearchCallInProgress
        | SSEEventType::WebSearchCallSearching
        | SSEEventType::WebSearchCallCompleted => Some(SSEItemType::WebSearchCall),
        SSEEventType::McpCallInProgress
        | SSEEventType::McpCallArgumentsDelta
        | SSEEventType::McpCallArgumentsDone
        | SSEEventType::McpCallCompleted
        | SSEEventType::McpCallFailed => Some(SSEItemType::McpCall),
        SSEEventType::McpListToolsInProgress
        | SSEEventType::McpListToolsCompleted
        | SSEEventType::McpListToolsFailed => Some(SSEItemType::McpListTools),
        SSEEventType::ResponseCreated
        | SSEEventType::ResponseInProgress
        | SSEEventType::ResponseCompleted
        | SSEEventType::ResponseFailed
        | SSEEventType::ResponseIncomplete
        | SSEEventType::OutputItemAdded
        | SSEEventType::OutputItemDone
        | SSEEventType::FileSearchCallSearching
        | SSEEventType::FileSearchCallCompleted
        | SSEEventType::Other => None,
    }
}

pub(crate) fn output_item_identity<'a>(
    item: &'a Map<String, Value>,
    owner: &str,
) -> Result<(&'a str, SSEItemType), EventError> {
    let item_id = item
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .or_else(|| item.get("item_id").and_then(Value::as_str).filter(|id| !id.is_empty()))
        .ok_or_else(|| missing_field(owner, "id"))?;
    let item_type_name = required_str(item, "type", owner)?;
    let item_type = item_type_name
        .parse()
        .map_err(|()| invalid(format!("upstream output item type '{item_type_name}' is unsupported")))?;
    Ok((item_id, item_type))
}

pub(crate) fn ensure_supported_output_item_type(item_type: &str) -> Result<(), EventError> {
    if item_type.parse::<SSEItemType>().is_ok() {
        return Ok(());
    }
    Err(invalid(format!(
        "upstream output item type '{item_type}' is unsupported"
    )))
}

fn validate_response_event(frame: &EventFrame, event_name: &str) -> Result<(), EventError> {
    let response = required_object(&frame.wire.rest, "response", event_name)?;
    required_str(response, "id", "upstream response")?;
    let status = required_str(response, "status", "upstream response")?;
    let expected_status = match frame.event_type {
        SSEEventType::ResponseCreated | SSEEventType::ResponseInProgress => "in_progress",
        SSEEventType::ResponseCompleted => "completed",
        SSEEventType::ResponseFailed => "failed",
        SSEEventType::ResponseIncomplete => "incomplete",
        _ => return Ok(()),
    };
    if status == expected_status {
        return Ok(());
    }
    Err(invalid(format!(
        "upstream stream event '{event_name}' has status '{status}', expected '{expected_status}'"
    )))
}

fn validate_output_item<'a>(
    frame: &'a EventFrame,
    event_name: &str,
    complete: bool,
) -> Result<ValidatedItem<'a>, EventError> {
    let output_index = required_output_index(frame, event_name)?;
    let item = required_object(&frame.wire.rest, "item", event_name)?;
    let (item_id, item_type) = output_item_identity(item, "output item")?;
    if !complete {
        return Ok(ValidatedItem {
            item_id,
            output_index,
            item_type,
            done_item: None,
        });
    }

    let mut canonical = Value::Object(item.clone());
    if canonical.get("id").and_then(Value::as_str).is_none_or(str::is_empty) {
        canonical["id"] = Value::String(required_str(item, "item_id", "output item")?.to_owned());
    }
    let output = OutputItem::deserialize(canonical)
        .map_err(|error| invalid(format!("upstream stream output item is invalid: {error}")))?;
    if SSEItemType::try_from(&output) != Ok(item_type) {
        return Err(invalid(format!(
            "upstream output item type '{}' is unsupported",
            item_type.as_str()
        )));
    }
    Ok(ValidatedItem {
        item_id,
        output_index,
        item_type,
        done_item: Some(output),
    })
}

fn validate_event_fields(
    event: &Map<String, Value>,
    event_type: SSEEventType,
    event_name: &str,
) -> Result<(), EventError> {
    match event_type {
        SSEEventType::OutputTextDelta
        | SSEEventType::OutputTextDone
        | SSEEventType::ContentPartAdded
        | SSEEventType::ContentPartDone
        | SSEEventType::ReasoningTextDelta
        | SSEEventType::ReasoningTextDone
        | SSEEventType::ReasoningPartAdded
        | SSEEventType::ReasoningPartDone => {
            required_u32(event, "content_index", event_name)?;
        }
        SSEEventType::ReasoningSummaryTextDelta | SSEEventType::ReasoningSummaryTextDone => {
            required_u32(event, "summary_index", event_name)?;
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
        SSEEventType::FunctionCallArgumentsDone | SSEEventType::McpCallArgumentsDone => Some("arguments"),
        SSEEventType::CustomToolCallInputDone => Some("input"),
        SSEEventType::ContentPartAdded
        | SSEEventType::ContentPartDone
        | SSEEventType::ReasoningPartAdded
        | SSEEventType::ReasoningPartDone => {
            required_object(event, "part", event_name)?;
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
        required_string(event, field, event_name)?;
    }
    Ok(())
}

fn required_output_index(frame: &EventFrame, owner: &str) -> Result<u32, EventError> {
    frame
        .wire
        .output_index
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| missing_field(owner, "output_index"))
}

fn required_str<'a>(value: &'a Map<String, Value>, field: &str, owner: &str) -> Result<&'a str, EventError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| missing_field(owner, field))
}

fn required_string<'a>(value: &'a Map<String, Value>, field: &str, owner: &str) -> Result<&'a str, EventError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| missing_field(owner, field))
}

fn required_object<'a>(
    value: &'a Map<String, Value>,
    field: &str,
    owner: &str,
) -> Result<&'a Map<String, Value>, EventError> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| missing_field(owner, field))
}

fn required_u32(value: &Map<String, Value>, field: &str, owner: &str) -> Result<u32, EventError> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| missing_field(owner, field))
}

fn missing_field(owner: &str, field: &str) -> EventError {
    invalid(format!("{owner} has no valid '{field}'"))
}

fn invalid(message: impl Into<String>) -> EventError {
    EventError(message.into())
}
