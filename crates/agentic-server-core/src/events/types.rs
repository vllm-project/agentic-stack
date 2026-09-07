use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::types::io::{OutputItem, ResponseUsage};

/// The type of an output item received during streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SSEItemType {
    Reasoning,
    FunctionCall,
    CustomToolCall,
    WebSearchCall,
    McpCall,
    McpListTools,
    Compaction,
    Message,
}

impl SSEItemType {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reasoning => "reasoning",
            Self::FunctionCall => "function_call",
            Self::CustomToolCall => "custom_tool_call",
            Self::WebSearchCall => "web_search_call",
            Self::McpCall => "mcp_call",
            Self::McpListTools => "mcp_list_tools",
            Self::Compaction => "compaction",
            Self::Message => "message",
        }
    }
}

impl From<&str> for SSEItemType {
    fn from(s: &str) -> Self {
        s.parse().unwrap_or(Self::Message)
    }
}

impl std::str::FromStr for SSEItemType {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "reasoning" => Ok(Self::Reasoning),
            "function_call" => Ok(Self::FunctionCall),
            "custom_tool_call" => Ok(Self::CustomToolCall),
            "web_search_call" => Ok(Self::WebSearchCall),
            "mcp_call" => Ok(Self::McpCall),
            "mcp_list_tools" => Ok(Self::McpListTools),
            "compaction" => Ok(Self::Compaction),
            "message" => Ok(Self::Message),
            _ => Err(()),
        }
    }
}

impl TryFrom<&OutputItem> for SSEItemType {
    type Error = ();

    fn try_from(item: &OutputItem) -> Result<Self, Self::Error> {
        match item {
            OutputItem::Message(_) => Ok(Self::Message),
            OutputItem::FunctionCall(_) => Ok(Self::FunctionCall),
            OutputItem::CustomToolCall(_) => Ok(Self::CustomToolCall),
            OutputItem::WebSearchCall(_) => Ok(Self::WebSearchCall),
            OutputItem::McpCall(_) => Ok(Self::McpCall),
            OutputItem::McpListTools(_) => Ok(Self::McpListTools),
            OutputItem::Reasoning(_) => Ok(Self::Reasoning),
            OutputItem::Compaction(_) => Ok(Self::Compaction),
            OutputItem::Unknown => Err(()),
        }
    }
}

impl From<String> for SSEItemType {
    fn from(s: String) -> Self {
        Self::from(s.as_str())
    }
}

impl PartialEq<str> for SSEItemType {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for SSEItemType {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// Classification of SSE event types from the Responses API.
///
/// Covers both the `OpenAI` and vLLM wire formats (e.g. `response.done` vs
/// `response.completed`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SSEEventType {
    // Response lifecycle
    ResponseCreated,
    ResponseInProgress,
    ResponseCompleted,
    ResponseFailed,
    ResponseIncomplete,

    // Output item lifecycle
    OutputItemAdded,
    OutputItemDone,

    // Text content
    OutputTextDelta,
    OutputTextDone,
    ContentPartAdded,
    ContentPartDone,

    // Function calls
    FunctionCallArgumentsDelta,
    FunctionCallArgumentsDone,
    CustomToolCallInputDelta,
    CustomToolCallInputDone,

    // Reasoning
    ReasoningTextDelta,
    ReasoningTextDone,
    ReasoningPartAdded,
    ReasoningPartDone,
    ReasoningSummaryTextDelta,
    ReasoningSummaryTextDone,

    // Built-in tool calls
    FileSearchCallSearching,
    FileSearchCallCompleted,
    WebSearchCallInProgress,
    WebSearchCallSearching,
    WebSearchCallCompleted,
    McpCallInProgress,
    McpCallArgumentsDelta,
    McpCallArgumentsDone,
    McpCallCompleted,
    McpCallFailed,
    McpListToolsInProgress,
    McpListToolsCompleted,
    McpListToolsFailed,

    // Catch-all for unrecognized events
    Other,
}

impl From<&str> for SSEEventType {
    fn from(value: &str) -> Self {
        match value {
            "response.created" => Self::ResponseCreated,
            "response.in_progress" => Self::ResponseInProgress,
            "response.completed" | "response.done" => Self::ResponseCompleted,
            "response.failed" => Self::ResponseFailed,
            "response.incomplete" => Self::ResponseIncomplete,
            "response.output_item.added" => Self::OutputItemAdded,
            "response.output_item.done" => Self::OutputItemDone,
            "response.output_text.delta" => Self::OutputTextDelta,
            "response.output_text.done" => Self::OutputTextDone,
            "response.content_part.added" => Self::ContentPartAdded,
            "response.content_part.done" => Self::ContentPartDone,
            "response.function_call_arguments.delta" => Self::FunctionCallArgumentsDelta,
            "response.function_call_arguments.done" => Self::FunctionCallArgumentsDone,
            "response.custom_tool_call_input.delta" => Self::CustomToolCallInputDelta,
            "response.custom_tool_call_input.done" => Self::CustomToolCallInputDone,
            "response.reasoning_text.delta" => Self::ReasoningTextDelta,
            "response.reasoning_text.done" => Self::ReasoningTextDone,
            "response.reasoning_part.added" => Self::ReasoningPartAdded,
            "response.reasoning_part.done" => Self::ReasoningPartDone,
            "response.reasoning_summary_text.delta" => Self::ReasoningSummaryTextDelta,
            "response.reasoning_summary_text.done" => Self::ReasoningSummaryTextDone,
            "response.file_search_call.searching" => Self::FileSearchCallSearching,
            "response.file_search_call.completed" => Self::FileSearchCallCompleted,
            "response.web_search_call.in_progress" => Self::WebSearchCallInProgress,
            "response.web_search_call.searching" => Self::WebSearchCallSearching,
            "response.web_search_call.completed" => Self::WebSearchCallCompleted,
            "response.mcp_call.in_progress" => SSEEventType::McpCallInProgress,
            "response.mcp_call_arguments.delta" => SSEEventType::McpCallArgumentsDelta,
            "response.mcp_call_arguments.done" => SSEEventType::McpCallArgumentsDone,
            "response.mcp_call.completed" => SSEEventType::McpCallCompleted,
            "response.mcp_call.failed" => SSEEventType::McpCallFailed,
            "response.mcp_list_tools.in_progress" => SSEEventType::McpListToolsInProgress,
            "response.mcp_list_tools.completed" => SSEEventType::McpListToolsCompleted,
            "response.mcp_list_tools.failed" => SSEEventType::McpListToolsFailed,
            _ => Self::Other,
        }
    }
}

impl TryFrom<SSEEventType> for &'static str {
    type Error = ();

    fn try_from(value: SSEEventType) -> Result<Self, Self::Error> {
        match value {
            SSEEventType::ResponseCreated => Ok("response.created"),
            SSEEventType::ResponseInProgress => Ok("response.in_progress"),
            SSEEventType::ResponseCompleted => Ok("response.completed"),
            SSEEventType::ResponseFailed => Ok("response.failed"),
            SSEEventType::ResponseIncomplete => Ok("response.incomplete"),
            SSEEventType::OutputItemAdded => Ok("response.output_item.added"),
            SSEEventType::OutputItemDone => Ok("response.output_item.done"),
            SSEEventType::OutputTextDelta => Ok("response.output_text.delta"),
            SSEEventType::OutputTextDone => Ok("response.output_text.done"),
            SSEEventType::ContentPartAdded => Ok("response.content_part.added"),
            SSEEventType::ContentPartDone => Ok("response.content_part.done"),
            SSEEventType::FunctionCallArgumentsDelta => Ok("response.function_call_arguments.delta"),
            SSEEventType::FunctionCallArgumentsDone => Ok("response.function_call_arguments.done"),
            SSEEventType::CustomToolCallInputDelta => Ok("response.custom_tool_call_input.delta"),
            SSEEventType::CustomToolCallInputDone => Ok("response.custom_tool_call_input.done"),
            SSEEventType::ReasoningTextDelta => Ok("response.reasoning_text.delta"),
            SSEEventType::ReasoningTextDone => Ok("response.reasoning_text.done"),
            SSEEventType::ReasoningPartAdded => Ok("response.reasoning_part.added"),
            SSEEventType::ReasoningPartDone => Ok("response.reasoning_part.done"),
            SSEEventType::ReasoningSummaryTextDelta => Ok("response.reasoning_summary_text.delta"),
            SSEEventType::ReasoningSummaryTextDone => Ok("response.reasoning_summary_text.done"),
            SSEEventType::FileSearchCallSearching => Ok("response.file_search_call.searching"),
            SSEEventType::FileSearchCallCompleted => Ok("response.file_search_call.completed"),
            SSEEventType::WebSearchCallInProgress => Ok("response.web_search_call.in_progress"),
            SSEEventType::WebSearchCallSearching => Ok("response.web_search_call.searching"),
            SSEEventType::WebSearchCallCompleted => Ok("response.web_search_call.completed"),
            SSEEventType::McpCallInProgress => Ok("response.mcp_call.in_progress"),
            SSEEventType::McpCallArgumentsDelta => Ok("response.mcp_call_arguments.delta"),
            SSEEventType::McpCallArgumentsDone => Ok("response.mcp_call_arguments.done"),
            SSEEventType::McpCallCompleted => Ok("response.mcp_call.completed"),
            SSEEventType::McpCallFailed => Ok("response.mcp_call.failed"),
            SSEEventType::McpListToolsInProgress => Ok("response.mcp_list_tools.in_progress"),
            SSEEventType::McpListToolsCompleted => Ok("response.mcp_list_tools.completed"),
            SSEEventType::McpListToolsFailed => Ok("response.mcp_list_tools.failed"),
            SSEEventType::Other => Err(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireEvent {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub event_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence_number: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_index: Option<u64>,
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

impl WireEvent {
    #[must_use]
    pub fn new(event_type: impl Into<String>) -> Self {
        Self {
            event_type: Some(event_type.into()),
            sequence_number: None,
            output_index: None,
            rest: Map::new(),
        }
    }
}

/// Typed payload extracted from an SSE event's JSON data.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum EventPayload {
    /// `response.created` / `response.completed` / `response.failed` /
    /// `response.incomplete` / `response.in_progress`
    Response {
        id: String,
        status: String,
        usage: Option<ResponseUsage>,
    },

    /// `response.output_item.added`
    OutputItemAdded {
        item_id: String,
        item_type: SSEItemType,
        output_index: u32,
        name: Option<String>,
        namespace: Option<String>,
        call_id: Option<String>,
    },

    /// `response.output_item.done`
    OutputItemDone {
        item_id: String,
        item_type: SSEItemType,
        output_index: u32,
        item: Value,
    },

    /// `response.output_text.delta`
    TextDelta {
        delta: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
    },

    /// `response.output_text.done`
    TextDone {
        text: String,
        item_id: String,
        output_index: u32,
    },

    /// `response.function_call_arguments.delta`
    FunctionCallArgsDelta {
        delta: String,
        call_id: Option<String>,
        item_id: String,
        output_index: u32,
    },

    /// `response.function_call_arguments.done`
    FunctionCallArgsDone {
        arguments: String,
        call_id: Option<String>,
        item_id: String,
        name: String,
        output_index: u32,
    },

    /// `response.custom_tool_call_input.delta`
    CustomToolCallInputDelta {
        delta: String,
        item_id: String,
        output_index: u32,
    },

    /// `response.custom_tool_call_input.done`
    CustomToolCallInputDone {
        input: String,
        item_id: String,
        output_index: u32,
    },

    /// `response.reasoning_text.delta`
    ReasoningTextDelta {
        delta: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
    },

    /// `response.reasoning_text.done`
    ReasoningTextDone {
        text: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
    },

    /// `response.reasoning_summary_text.delta`
    ReasoningSummaryTextDelta {
        delta: String,
        item_id: String,
        output_index: u32,
        summary_index: u32,
    },

    /// `response.reasoning_summary_text.done`
    ReasoningSummaryTextDone {
        text: String,
        item_id: String,
        output_index: u32,
        summary_index: u32,
    },

    /// Events we classify but don't deeply parse yet.
    Raw(Value),

    /// No meaningful payload (e.g. unparseable content).
    None,
}

/// A normalized SSE event frame — the output of [`normalize_sse_line`].
///
/// [`normalize_sse_line`]: crate::events::normalize::normalize_sse_line
#[derive(Debug, Clone)]
pub struct EventFrame {
    pub event_type: SSEEventType,
    pub payload: EventPayload,
    pub wire: WireEvent,
}

impl EventFrame {
    #[must_use]
    pub fn synthetic(event_type: SSEEventType, rest: Map<String, Value>) -> Option<Self> {
        let event_type_name = <&str>::try_from(event_type).ok()?;
        Some(Self {
            event_type,
            payload: EventPayload::None,
            wire: WireEvent {
                event_type: Some(event_type_name.to_owned()),
                sequence_number: None,
                output_index: None,
                rest,
            },
        })
    }

    #[must_use]
    pub fn sequence_number(&self) -> Option<u64> {
        self.wire.sequence_number
    }
}

#[cfg(test)]
mod tests {
    use super::{SSEEventType, SSEItemType};
    use crate::types::event::MessageStatus;
    use crate::types::io::{FunctionToolCall, OutputItem};

    #[test]
    fn sse_item_type_strictly_parses_known_wire_types() {
        assert_eq!("function_call".parse(), Ok(SSEItemType::FunctionCall));
        assert!("unsupported_item".parse::<SSEItemType>().is_err());
    }

    #[test]
    fn sse_item_type_is_derived_from_typed_output_items() {
        let item = OutputItem::FunctionCall(FunctionToolCall {
            id: "fc_1".to_owned(),
            call_id: "call_1".to_owned(),
            name: "lookup".to_owned(),
            namespace: None,
            arguments: "{}".to_owned(),
            status: MessageStatus::Completed,
        });

        assert_eq!(SSEItemType::try_from(&item), Ok(SSEItemType::FunctionCall));
        assert!(SSEItemType::try_from(&OutputItem::Unknown).is_err());
    }

    #[test]
    fn sse_event_type_wire_names_round_trip() {
        for event_type in [
            SSEEventType::ResponseCreated,
            SSEEventType::ResponseInProgress,
            SSEEventType::ResponseCompleted,
            SSEEventType::ResponseFailed,
            SSEEventType::ResponseIncomplete,
            SSEEventType::OutputItemAdded,
            SSEEventType::OutputItemDone,
            SSEEventType::OutputTextDelta,
            SSEEventType::OutputTextDone,
            SSEEventType::ContentPartAdded,
            SSEEventType::ContentPartDone,
            SSEEventType::FunctionCallArgumentsDelta,
            SSEEventType::FunctionCallArgumentsDone,
            SSEEventType::CustomToolCallInputDelta,
            SSEEventType::CustomToolCallInputDone,
            SSEEventType::ReasoningTextDelta,
            SSEEventType::ReasoningTextDone,
            SSEEventType::ReasoningPartAdded,
            SSEEventType::ReasoningPartDone,
            SSEEventType::ReasoningSummaryTextDelta,
            SSEEventType::ReasoningSummaryTextDone,
            SSEEventType::FileSearchCallSearching,
            SSEEventType::FileSearchCallCompleted,
            SSEEventType::WebSearchCallInProgress,
            SSEEventType::WebSearchCallSearching,
            SSEEventType::WebSearchCallCompleted,
            SSEEventType::McpCallInProgress,
            SSEEventType::McpCallArgumentsDelta,
            SSEEventType::McpCallArgumentsDone,
            SSEEventType::McpCallCompleted,
            SSEEventType::McpCallFailed,
            SSEEventType::McpListToolsInProgress,
            SSEEventType::McpListToolsCompleted,
            SSEEventType::McpListToolsFailed,
        ] {
            let wire_name = <&str>::try_from(event_type).expect("known event type has a wire name");
            assert_eq!(SSEEventType::from(wire_name), event_type);
        }
        assert!(<&str>::try_from(SSEEventType::Other).is_err());
    }
}
