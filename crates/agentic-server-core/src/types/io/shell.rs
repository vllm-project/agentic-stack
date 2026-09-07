use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Lifecycle status for a shell call or shell call output item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShellCallStatus {
    InProgress,
    Completed,
    Incomplete,
}

/// Commands and execution limits requested by a model-generated shell call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellCallAction {
    pub commands: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_length: Option<u64>,
    #[serde(default, flatten)]
    pub extra: HashMap<String, Value>,
}

/// A model-generated request to execute one or more shell commands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub call_id: String,
    pub action: ShellCallAction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<ShellCallStatus>,
    #[serde(default, flatten)]
    pub extra: HashMap<String, Value>,
}

/// Outcome of one command in a shell call output.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ShellCallOutcome {
    Exit {
        exit_code: i32,
    },
    Timeout,
    #[serde(other)]
    Unknown,
}

/// Captured output and outcome for one command in a shell call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellCallOutputContent {
    #[serde(default)]
    pub stdout: String,
    #[serde(default)]
    pub stderr: String,
    pub outcome: ShellCallOutcome,
    #[serde(default, flatten)]
    pub extra: HashMap<String, Value>,
}

/// Output supplied for a previously emitted shell call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellCallOutputMessage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub call_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_length: Option<u64>,
    pub output: Vec<ShellCallOutputContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<ShellCallStatus>,
    #[serde(default, flatten)]
    pub extra: HashMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_call_round_trips_with_limits_and_extra_fields() {
        let value = serde_json::json!({
            "id": "sh_1",
            "call_id": "call_1",
            "action": {
                "commands": ["pwd", "cargo test"],
                "timeout_ms": 120_000,
                "max_output_length": 4096,
                "future_action_field": true
            },
            "status": "in_progress",
            "future_item_field": "kept"
        });

        let call: ShellCall = serde_json::from_value(value).unwrap();
        assert_eq!(call.action.commands, ["pwd", "cargo test"]);
        assert_eq!(call.action.timeout_ms, Some(120_000));
        assert_eq!(call.status, Some(ShellCallStatus::InProgress));

        let serialized = serde_json::to_value(call).unwrap();
        assert_eq!(serialized["future_item_field"], "kept");
        assert_eq!(serialized["action"]["future_action_field"], true);
    }

    #[test]
    fn shell_call_output_round_trips_exit_and_timeout_outcomes() {
        let value = serde_json::json!({
            "id": "sho_1",
            "call_id": "call_1",
            "max_output_length": 4096,
            "output": [
                {
                    "stdout": "ok\n",
                    "stderr": "",
                    "outcome": {"type": "exit", "exit_code": 0}
                },
                {
                    "stdout": "",
                    "stderr": "timed out",
                    "outcome": {"type": "timeout"}
                }
            ],
            "status": "completed"
        });

        let output: ShellCallOutputMessage = serde_json::from_value(value).unwrap();
        assert_eq!(output.output.len(), 2);
        assert_eq!(output.output[0].outcome, ShellCallOutcome::Exit { exit_code: 0 });
        assert_eq!(output.output[1].outcome, ShellCallOutcome::Timeout);
        assert_eq!(output.status, Some(ShellCallStatus::Completed));

        let serialized = serde_json::to_value(output).unwrap();
        assert_eq!(serialized["output"][0]["outcome"]["type"], "exit");
        assert_eq!(serialized["output"][1]["outcome"]["type"], "timeout");
    }
}
