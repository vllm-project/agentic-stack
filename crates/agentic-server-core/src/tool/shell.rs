use std::collections::HashMap;

use crate::types::io::{FunctionTool, FunctionToolCall, OutputItem, ShellCall, ShellCallAction, ShellCallStatus};
use crate::types::tools::{ShellEnvironment, ShellToolParam};
use crate::utils::common::deserialize_from_str;

use super::{ToolEntry, ToolError, ToolHandler, ToolType};

pub(crate) const SHELL_FUNCTION_NAME: &str = "shell";

/// Handler for the client-executed local `type: "shell"` tool.
///
/// The declaration is normalized to a function for inference, then restored
/// to a typed `shell_call` before it is returned to the client. This handler
/// deliberately does not implement `GatewayExecutor`: declaring a shell tool
/// never grants the gateway permission to execute arbitrary commands.
#[derive(Debug)]
pub struct ShellHandler;

impl ShellHandler {
    #[must_use]
    pub(crate) fn output_item(call: &FunctionToolCall) -> Option<OutputItem> {
        Self::output_item_with_status(call, call.status.into())
    }

    #[must_use]
    pub(crate) fn output_item_with_status(call: &FunctionToolCall, status: ShellCallStatus) -> Option<OutputItem> {
        let action = deserialize_from_str::<ShellCallAction>(&call.arguments).ok()?;
        Some(OutputItem::ShellCall(ShellCall {
            id: Some(public_item_id(&call.id)),
            call_id: call.call_id.clone(),
            action,
            status: Some(status),
            extra: HashMap::new(),
        }))
    }
}

impl ToolHandler for ShellHandler {
    type ToolParams = ShellToolParam;

    fn tool_type(&self) -> ToolType {
        ToolType::Shell
    }

    fn validate(&self, params: &ShellToolParam) -> Result<(), ToolError> {
        if !matches!(params.environment, ShellEnvironment::Local(_)) {
            return Err(ToolError::Config(
                "shell tool currently supports only environment.type='local'".to_owned(),
            ));
        }
        Ok(())
    }

    fn normalize(&self, _params: &ShellToolParam) -> Vec<FunctionTool> {
        vec![FunctionTool {
            type_: "function".to_owned(),
            name: SHELL_FUNCTION_NAME.to_owned(),
            description: Some(
                "Run one or more commands in the caller-provided local shell environment. The caller executes the commands and returns their outputs."
                    .to_owned(),
            ),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "commands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Commands to execute in order."
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional timeout in milliseconds."
                    },
                    "max_output_length": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional maximum captured output length."
                    }
                },
                "required": ["commands"],
                "additionalProperties": false
            })),
            strict: Some(false),
        }]
    }
}

pub(crate) fn insert_shell_entry(entries: &mut HashMap<String, ToolEntry>) {
    entries.insert(SHELL_FUNCTION_NAME.to_owned(), ToolEntry::client(ToolType::Shell, None));
}

#[must_use]
pub(crate) fn public_item_id(item_id: &str) -> String {
    if item_id.starts_with("sh_") {
        return item_id.to_owned();
    }
    if let Some(suffix) = item_id.strip_prefix("fc_").filter(|suffix| !suffix.is_empty()) {
        return format!("sh_{suffix}");
    }
    format!("sh_{:016x}", stable_name_hash(item_id))
}

fn stable_name_hash(value: &str) -> u64 {
    value.as_bytes().iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::event::MessageStatus;

    #[test]
    fn shell_entry_uses_client_execution() {
        let mut entries = HashMap::new();
        insert_shell_entry(&mut entries);
        let entry = &entries[SHELL_FUNCTION_NAME];
        assert_eq!(entry.tool_type, ToolType::Shell);
        assert!(!entry.ownership.is_gateway());
    }

    #[test]
    fn public_shell_ids_preserve_native_ids_and_translate_function_ids() {
        let native_id = "sh_6d300b80c049eb3c";
        assert_eq!(public_item_id(native_id), native_id);
        assert_eq!(public_item_id("fc_6d300b80c049eb3c"), native_id);
    }
    fn local_shell() -> ShellToolParam {
        serde_json::from_value(serde_json::json!({
            "environment": {"type": "local"}
        }))
        .expect("local shell declaration")
    }

    #[test]
    fn local_shell_normalizes_to_a_function() {
        let [tool] = ShellHandler.normalize(&local_shell()).try_into().expect("one tool");
        assert_eq!(tool.type_, "function");
        assert_eq!(tool.name, SHELL_FUNCTION_NAME);
        assert_eq!(tool.parameters.unwrap()["required"], serde_json::json!(["commands"]));
        assert_eq!(tool.strict, Some(false));
    }

    #[test]
    fn unknown_shell_environment_is_rejected() {
        let param = serde_json::from_value::<ShellToolParam>(serde_json::json!({
            "environment": {"type": "container_auto"}
        }))
        .expect("preserved unknown environment");
        assert!(ShellHandler.validate(&param).is_err());
    }

    #[test]
    fn normalized_function_call_restores_shell_call() {
        let call = FunctionToolCall {
            id: "fc_123".to_owned(),
            call_id: "call_123".to_owned(),
            name: SHELL_FUNCTION_NAME.to_owned(),
            namespace: None,
            arguments: r#"{"commands":["pwd"],"timeout_ms":1000}"#.to_owned(),
            status: MessageStatus::Completed,
        };

        let Some(OutputItem::ShellCall(shell)) = ShellHandler::output_item(&call) else {
            panic!("expected shell call");
        };
        assert_eq!(shell.id.as_deref(), Some("sh_123"));
        assert_eq!(shell.call_id, "call_123");
        assert_eq!(shell.action.commands, ["pwd"]);
        assert_eq!(shell.action.timeout_ms, Some(1000));
        assert_eq!(shell.status, Some(ShellCallStatus::Completed));
    }

    #[test]
    fn malformed_function_arguments_are_not_restored() {
        let call = FunctionToolCall {
            id: "fc_123".to_owned(),
            call_id: "call_123".to_owned(),
            name: SHELL_FUNCTION_NAME.to_owned(),
            namespace: None,
            arguments: "not-json".to_owned(),
            status: MessageStatus::Completed,
        };
        assert!(ShellHandler::output_item(&call).is_none());
    }
}
