//! Anthropic Messages API (`/v1/messages`) wire types.
//!
//! The gateway serves `/v1/messages` natively: requests that declare a
//! gateway-owned tool are driven through a Messages-native tool loop that talks
//! to vLLM `/v1/messages` upstream (see `executor::messages_loop`); everything
//! else is a transparent proxy. These are the Anthropic wire types shared by
//! the handler, the loop, and the tool seam.

pub mod request;
pub mod tool_seam;

pub use request::{
    ContentBlock, MessageContent, MessageParam, MessagesRequest, OutputConfig, ReasoningEffort, ReasoningEffortLevel,
    SystemBlock, SystemPrompt, ToolParam, ToolResultBlock, ToolResultContent,
};
pub use tool_seam::{
    GatewayToolMap, adapt_web_search_input, call_to_tool_use_block, has_gateway_tool, parse_tool_input, registry_tools,
    strip_gateway_tool_use, tool_result_block, tool_use_to_call,
};
