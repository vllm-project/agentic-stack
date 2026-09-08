//! Wire format types for the tool framework.
//!
//! This module contains only serde shapes (serialization/deserialization types).
//! Behavioral logic (registry, handler trait, normalization) lives in [`crate::tool`].

pub mod params;

pub use params::{
    CodeInterpreterToolParam, CodexNamespaceMember, CodexNamespaceToolParam, CustomToolParam, EmptyToolNameError,
    FileSearchToolParam, FunctionToolParam, McpDiscoveredToolParam, McpToolParam, NonEmptyToolName, ResponsesTool,
    ToolSearchExecution, ToolSearchStatus, ToolSearchToolParam, WebSearchContextSize, WebSearchFilters,
    WebSearchToolParam, WebSearchUserLocation,
};
