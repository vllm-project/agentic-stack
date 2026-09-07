//! Domain types for conversation items.

use std::convert::TryFrom;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::storage::StorageError;
use crate::types::io::{InputItem, OutputItem};
use crate::utils::common::serialize_to_value;

pub(crate) const STORED_ITEM_KIND_KEY: &str = "_agentic_item_kind";

/// Item kind (input vs output) for storage and retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemKind {
    Input,
    Output,
}

impl ItemKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
        }
    }

    pub(crate) fn from_stored_str(value: &str) -> Option<Self> {
        match value {
            "input" => Some(Self::Input),
            "output" => Some(Self::Output),
            _ => None,
        }
    }
}

/// Union type for conversation items (input or output).
#[derive(Debug, Clone)]
pub enum InOutItem {
    Input(InputItem),
    Output(OutputItem),
}

fn serialized_values_equal<T: Serialize>(left: &T, right: &T) -> bool {
    let Ok(left) = serialize_to_value(left) else {
        return false;
    };
    let Ok(right) = serialize_to_value(right) else {
        return false;
    };
    left == right
}

impl PartialEq for InOutItem {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Input(left), Self::Input(right)) => serialized_values_equal(left, right),
            (Self::Output(left), Self::Output(right)) => serialized_values_equal(left, right),
            _ => false,
        }
    }
}

impl From<InputItem> for InOutItem {
    fn from(item: InputItem) -> Self {
        Self::Input(item)
    }
}

impl From<OutputItem> for InOutItem {
    fn from(item: OutputItem) -> Self {
        Self::Output(item)
    }
}

impl TryFrom<&InOutItem> for String {
    type Error = StorageError;

    fn try_from(item: &InOutItem) -> Result<Self, Self::Error> {
        let (mut value, kind) = match item {
            InOutItem::Input(input) => (
                serde_json::to_value(input).map_err(StorageError::Serialization)?,
                ItemKind::Input,
            ),
            InOutItem::Output(output) => (
                serde_json::to_value(output).map_err(StorageError::Serialization)?,
                ItemKind::Output,
            ),
        };

        if let Some(obj) = value.as_object_mut() {
            obj.insert(
                STORED_ITEM_KIND_KEY.to_string(),
                Value::String(kind.as_str().to_string()),
            );
        }

        serde_json::to_string(&value).map_err(StorageError::Serialization)
    }
}

impl InOutItem {
    /// Converts stored history into input items for continuation processing.
    /// Internal items are removed later by `ResponsesInput::model_input`.
    #[must_use]
    pub fn into_input_items(history: Vec<InOutItem>) -> Vec<InputItem> {
        history
            .into_iter()
            .filter_map(|item| match item {
                InOutItem::Input(item) if item.is_unknown() => None,
                InOutItem::Input(item) => Some(item),
                InOutItem::Output(output) => output.to_input_item(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::event::MessageStatus;
    use crate::types::io::output::McpListTools;
    use crate::types::io::{
        FunctionToolCall, InputContent, InputMessage, InputMessageContent, OutputMessage, OutputTextContent,
        ReasoningOutput, ReasoningTextContent, ResponsesInput, ShellCall, ShellCallAction, ShellCallStatus,
    };

    #[test]
    fn test_inout_item_from_input() {
        let input = InputItem::Message(InputMessage {
            id: None,
            role: "user".to_string(),
            status: None,
            content: InputMessageContent::Text("hello".to_string()),
        });
        let item: InOutItem = input.into();
        assert!(matches!(item, InOutItem::Input(_)));
    }

    #[test]
    fn test_inout_item_from_output() {
        let output = OutputItem::Message(OutputMessage::new("msg_1", MessageStatus::Completed));
        let item: InOutItem = output.into();
        assert!(matches!(item, InOutItem::Output(_)));
    }

    #[test]
    fn test_inout_item_to_string() {
        let input = InputItem::Message(InputMessage {
            id: None,
            role: "user".to_string(),
            status: None,
            content: InputMessageContent::Text("test".to_string()),
        });
        let item = InOutItem::Input(input);
        let json = String::try_from(&item).expect("serialization failed");
        assert!(json.contains(STORED_ITEM_KIND_KEY));
        assert!(json.contains("input"));
        assert!(json.contains("user"));
        assert!(json.contains("test"));
    }

    #[test]
    fn test_into_input_items_converts_output_messages() {
        let mut output = OutputMessage::new("out1", MessageStatus::Completed);
        output.content.push(OutputTextContent::new("answer"));
        let items = vec![
            InOutItem::Input(InputItem::Message(InputMessage {
                id: None,
                role: "user".to_string(),
                status: None,
                content: InputMessageContent::Text("msg1".to_string()),
            })),
            InOutItem::Output(OutputItem::Message(output)),
            InOutItem::Input(InputItem::Message(InputMessage {
                id: None,
                role: "user".to_string(),
                status: None,
                content: InputMessageContent::Text("msg2".to_string()),
            })),
        ];

        let inputs = InOutItem::into_input_items(items);
        assert_eq!(inputs.len(), 3);
        match &inputs[1] {
            InputItem::Message(message) => {
                assert_eq!(message.role, "assistant");
                match &message.content {
                    InputMessageContent::Parts(parts) => {
                        assert_eq!(parts.len(), 1);
                        match &parts[0] {
                            InputContent::OutputText(t) => {
                                assert_eq!(t.text, "answer");
                            }
                            _ => panic!("expected OutputText part"),
                        }
                    }
                    InputMessageContent::Text(_) => panic!("expected parts content"),
                }
            }
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn test_into_input_items_includes_reasoning() {
        let mut reasoning = ReasoningOutput::new("rs_1");
        reasoning.content.push(ReasoningTextContent::new("thinking..."));
        let items = vec![
            InOutItem::Output(OutputItem::Reasoning(reasoning)),
            InOutItem::Output(OutputItem::Message(OutputMessage::new(
                "msg_1",
                MessageStatus::Completed,
            ))),
        ];

        let inputs = InOutItem::into_input_items(items);
        assert_eq!(inputs.len(), 2);
        assert!(matches!(inputs[0], InputItem::Reasoning(_)));
        if let InputItem::Reasoning(r) = &inputs[0] {
            assert_eq!(r.id, "rs_1");
            assert_eq!(r.content[0].text, "thinking...");
        }
    }

    #[test]
    fn test_into_input_items_preserves_function_calls() {
        use crate::types::event::MessageStatus;
        let fc = FunctionToolCall {
            id: "fc_1".to_string(),
            call_id: "call_abc".to_string(),
            name: "my_tool".to_string(),
            namespace: None,
            arguments: "{}".to_string(),
            status: MessageStatus::Completed,
        };
        let items = vec![InOutItem::Output(OutputItem::FunctionCall(fc))];
        let inputs = InOutItem::into_input_items(items);
        assert_eq!(inputs.len(), 1);
        assert!(matches!(inputs[0], InputItem::FunctionCall(_)));
        if let InputItem::FunctionCall(f) = &inputs[0] {
            assert_eq!(f.name, "my_tool");
        }
    }

    #[test]
    fn test_into_input_items_preserves_shell_calls() {
        let call = ShellCall {
            id: Some("sh_1".to_owned()),
            call_id: "call_shell".to_owned(),
            action: ShellCallAction {
                commands: vec!["pwd".to_owned()],
                timeout_ms: Some(1_000),
                max_output_length: Some(4_096),
                extra: std::collections::HashMap::new(),
            },
            status: Some(ShellCallStatus::Completed),
            extra: std::collections::HashMap::new(),
        };

        let inputs = InOutItem::into_input_items(vec![InOutItem::Output(OutputItem::ShellCall(call))]);
        assert!(matches!(inputs.as_slice(), [InputItem::ShellCall(call)] if call.call_id == "call_shell"));
    }

    #[test]
    fn input_items_preserve_mcp_list_tools_until_model_input() {
        let history = vec![InOutItem::Output(OutputItem::McpListTools(McpListTools::new(
            "mcpl_1",
            "counter",
            Vec::new(),
        )))];

        let input_items = InOutItem::into_input_items(history);

        assert!(matches!(
            input_items.as_slice(),
            [InputItem::McpListTools(list_tools)] if list_tools.server_label == "counter"
        ));
        let model_input = ResponsesInput::Items(input_items).model_input().into_owned();
        let ResponsesInput::Items(model_items) = model_input else {
            panic!("model input should contain items");
        };
        assert!(model_items.is_empty());
    }

    #[test]
    fn test_item_kind_serialization() {
        let kind = ItemKind::Input;
        let json = serde_json::to_string(&kind).expect("serialization failed");
        assert_eq!(json, "\"input\"");

        let kind2 = ItemKind::Output;
        let json2 = serde_json::to_string(&kind2).expect("serialization failed");
        assert_eq!(json2, "\"output\"");
    }
}
