//! Domain type for response storage.

use std::convert::TryFrom;

use serde::{Deserialize, Serialize};

use super::super::models::Response as StorageDbResponse;
use super::errors::StorageError;
use crate::types::io::ToolChoice;
use crate::types::tools::ResponsesTool;
use crate::utils::common::serialize_to_string;

/// Response metadata with effective configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponseMetadata {
    pub model: String,
    pub previous_response_id: Option<String>,
    pub effective_tools: Option<Vec<ResponsesTool>>,
    /// Public definitions whose deferred availability was resolved by tool search.
    ///
    /// This is separate from `effective_tools` so public `defer_loading` stays
    /// unchanged while compaction may remove the call/output pair that loaded it.
    pub tool_search_loaded_tools: Option<Vec<ResponsesTool>>,
    pub effective_tool_choice: ToolChoice,
    pub effective_instructions: Option<String>,
}

/// Domain entity for a stored LLM response.
#[derive(Debug, Clone)]
pub struct ResponseData {
    /// Unique response identifier
    pub response_id: String,
    /// Optional conversation this response belongs to
    pub conversation_id: Option<String>,
    /// Optional reference to previous response for chaining
    pub previous_response_id: Option<String>,
    /// Creation timestamp as Unix timestamp in seconds
    pub created_at: i64,
    /// Deserialized history item IDs (vec of item IDs)
    pub history_item_ids: Vec<String>,
    /// Response metadata with effective configuration (fully typed)
    pub metadata: ResponseMetadata,
}

impl From<StorageDbResponse> for ResponseData {
    fn from(row: StorageDbResponse) -> Self {
        let history_item_ids = row.history_item_ids_vec();
        let metadata = row.metadata_as::<ResponseMetadata>().unwrap_or_default();

        Self {
            response_id: row.id,
            conversation_id: row.conversation_id,
            previous_response_id: row.previous_response_id,
            created_at: row.created_at,
            history_item_ids,
            metadata,
        }
    }
}

impl TryFrom<&ResponseMetadata> for String {
    type Error = StorageError;

    fn try_from(metadata: &ResponseMetadata) -> Result<Self, Self::Error> {
        let mut persisted = metadata.clone();
        if let Some(tools) = persisted.effective_tools.as_mut() {
            for tool in tools {
                tool.sanitize_for_persistence();
            }
        }
        if let Some(tools) = persisted.tool_search_loaded_tools.as_mut() {
            for tool in tools {
                tool.sanitize_for_persistence();
            }
        }
        serialize_to_string(&persisted).map_err(StorageError::Serialization)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_data_from_db_response() {
        let db_row = StorageDbResponse {
            id: "resp_123".to_string(),
            conversation_id: Some("conv_456".to_string()),
            previous_response_id: None,
            history_item_ids: Some(r#"["item_1"]"#.to_string()),
            metadata: Some(
                r#"{"model":"gpt-4","previous_response_id":null,"effective_tools":null,"effective_tool_choice":"auto","effective_instructions":null}"#
                    .to_string(),
            ),
            created_at: 1_704_067_200,
        };

        let response: ResponseData = db_row.into();
        assert_eq!(response.response_id, "resp_123");
        assert_eq!(response.conversation_id, Some("conv_456".to_string()));
        assert_eq!(response.created_at, 1_704_067_200);
        assert_eq!(response.history_item_ids, vec!["item_1".to_string()]);
        assert_eq!(response.metadata.model, "gpt-4");
    }

    #[test]
    fn test_response_data_from_db_response_optional_fields() {
        let db_row = StorageDbResponse {
            id: "resp_789".to_string(),
            conversation_id: None,
            previous_response_id: None,
            history_item_ids: None,
            metadata: None,
            created_at: 1_704_067_200,
        };

        let response: ResponseData = db_row.into();
        assert_eq!(response.response_id, "resp_789");
        assert!(response.conversation_id.is_none());
        assert!(response.history_item_ids.is_empty());
        assert_eq!(response.metadata.model, "");
    }

    #[test]
    fn test_response_metadata_serialization() {
        let metadata = ResponseMetadata {
            model: "gpt-4".to_string(),
            previous_response_id: Some("resp_1".to_string()),
            effective_tools: None,
            tool_search_loaded_tools: None,
            effective_tool_choice: ToolChoice::Auto,
            effective_instructions: Some("be helpful".to_string()),
        };

        let json_str = String::try_from(&metadata).expect("serialization failed");
        assert!(json_str.contains("gpt-4"));
        assert!(json_str.contains("resp_1"));
        assert!(json_str.contains("be helpful"));
    }

    #[test]
    fn test_response_metadata_serialization_removes_request_scoped_mcp_state() {
        let mut tool = serde_json::from_value(serde_json::json!({
            "type": "mcp",
            "server_label": "counter",
            "server_url": "https://mcp.example.com/mcp",
            "headers": {"X-API-Key": "secret"},
            "authorization": "bearer-secret",
            "require_approval": "never"
        }))
        .expect("valid MCP tool");
        let ResponsesTool::Mcp(param) = &mut tool else {
            panic!("expected MCP tool");
        };
        param
            .discovered_tools
            .push(crate::types::tools::McpDiscoveredToolParam {
                server_label: "counter".to_owned(),
                tool_name: "increment".to_owned(),
                internal_name: "mcp__counter__increment".to_owned(),
                tool: serde_json::from_value(serde_json::json!({
                    "name": "increment",
                    "inputSchema": {"type": "object"}
                }))
                .expect("discovered MCP tool"),
            });
        let metadata = ResponseMetadata {
            effective_tools: Some(vec![tool]),
            tool_search_loaded_tools: None,
            ..ResponseMetadata::default()
        };

        let serialized = String::try_from(&metadata).expect("serialization failed");
        let serialized_value: serde_json::Value =
            serde_json::from_str(&serialized).expect("serialized response metadata");
        assert!(
            serialized_value["effective_tools"][0]
                .get("_agentic_discovered_tools")
                .is_none()
        );

        let persisted: ResponseMetadata = serde_json::from_str(&serialized).expect("persisted metadata");
        let tools = persisted.effective_tools.expect("persisted tools");
        let ResponsesTool::Mcp(tool) = &tools[0] else {
            panic!("expected MCP tool");
        };

        assert!(tool.headers.is_none());
        assert!(tool.authorization.is_none());
    }

    #[test]
    fn test_response_metadata_default() {
        let metadata = ResponseMetadata::default();
        assert_eq!(metadata.model, "");
        assert!(metadata.previous_response_id.is_none());
        assert!(metadata.effective_tools.is_none());
        assert!(metadata.tool_search_loaded_tools.is_none());
        assert!(metadata.effective_instructions.is_none());
    }

    #[test]
    fn test_response_data_multiple_history_items() {
        let db_row = StorageDbResponse {
            id: "resp_multi".to_string(),
            conversation_id: Some("conv_1".to_string()),
            previous_response_id: Some("resp_prev".to_string()),
            history_item_ids: Some(r#"["item_1","item_2","item_3"]"#.to_string()),
            metadata: Some(r#"{"model":"gpt-3.5"}"#.to_string()),
            created_at: 1_704_067_200,
        };

        let response: ResponseData = db_row.into();
        assert_eq!(response.history_item_ids.len(), 3);
        assert_eq!(response.history_item_ids[0], "item_1");
        assert_eq!(response.history_item_ids[2], "item_3");
        assert_eq!(response.previous_response_id, Some("resp_prev".to_string()));
    }
}
