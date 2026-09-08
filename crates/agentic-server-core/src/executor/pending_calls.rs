//! Detects calls in an item sequence that never received a resolving output.
//!
//! Gateway-owned calls are always resolved within the same turn (their
//! output is appended before the turn ends), so anything still unresolved
//! after scanning a full item sequence is, by construction, something the
//! *client* owed a resolution for.

use std::collections::HashSet;

use indexmap::IndexMap;

use super::{ExecutorError, ExecutorResult};
use crate::types::io::InputItem;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallKind {
    Function,
    Custom,
    Shell,
}

impl CallKind {
    const fn call_item_name(self) -> &'static str {
        match self {
            Self::Function => "function_call",
            Self::Custom => "custom_tool_call",
            Self::Shell => "shell_call",
        }
    }

    const fn output_item_name(self) -> &'static str {
        match self {
            Self::Function => "function_call_output",
            Self::Custom => "custom_tool_call_output",
            Self::Shell => "shell_call_output",
        }
    }
}

/// A client-owned call (plain `function`, Codex `namespace` member, or
/// `custom` tool) with no later matching output in the same item sequence.
#[derive(Debug)]
pub(super) struct PendingCall {
    pub(super) call_id: String,
}

/// Scans `items` in order and returns every call left unresolved, in emission
/// order. Calls and outputs must have non-empty IDs and form a one-to-one,
/// same-kind relationship. Namespace member calls are represented as
/// `InputItem::FunctionCall`, so they're covered by the plain function check.
pub(super) fn pending_calls(items: &[InputItem]) -> ExecutorResult<Vec<PendingCall>> {
    let mut seen_call_ids = HashSet::new();
    let mut pending = IndexMap::new();
    for item in items {
        match item {
            InputItem::FunctionCall(call) => {
                add_call(&call.call_id, CallKind::Function, &mut seen_call_ids, &mut pending)?;
            }
            InputItem::CustomToolCall(call) => {
                add_call(&call.call_id, CallKind::Custom, &mut seen_call_ids, &mut pending)?;
            }
            InputItem::FunctionCallOutput(output) => {
                resolve_call(&output.call_id, CallKind::Function, &mut pending)?;
            }
            InputItem::CustomToolCallOutput(output) => {
                resolve_call(&output.call_id, CallKind::Custom, &mut pending)?;
            }
            InputItem::ShellCall(call) => {
                add_call(&call.call_id, CallKind::Shell, &mut seen_call_ids, &mut pending)?;
            }
            InputItem::ShellCallOutput(output) => {
                resolve_call(&output.call_id, CallKind::Shell, &mut pending)?;
            }
            InputItem::Message(_)
            | InputItem::Reasoning(_)
            | InputItem::McpListTools(_)
            | InputItem::Compaction(_)
            | InputItem::CompactionTrigger
            | InputItem::Unknown => {}
        }
    }
    Ok(pending.into_keys().map(|call_id| PendingCall { call_id }).collect())
}

fn add_call(
    call_id: &str,
    kind: CallKind,
    seen_call_ids: &mut HashSet<String>,
    pending: &mut IndexMap<String, CallKind>,
) -> ExecutorResult<()> {
    if call_id.is_empty() {
        return Err(ExecutorError::InvalidRequest(format!(
            "{} call_id must not be empty",
            kind.call_item_name()
        )));
    }
    if !seen_call_ids.insert(call_id.to_owned()) {
        return Err(ExecutorError::InvalidRequest(format!(
            "duplicate call_id '{call_id}' in {}",
            kind.call_item_name()
        )));
    }
    pending.insert(call_id.to_owned(), kind);
    Ok(())
}

fn resolve_call(call_id: &str, output_kind: CallKind, pending: &mut IndexMap<String, CallKind>) -> ExecutorResult<()> {
    if call_id.is_empty() {
        return Err(ExecutorError::InvalidRequest(format!(
            "{} call_id must not be empty",
            output_kind.output_item_name()
        )));
    }
    let Some(call_kind) = pending.shift_remove(call_id) else {
        return Err(ExecutorError::InvalidRequest(format!(
            "{} references call_id '{call_id}' without a pending call",
            output_kind.output_item_name()
        )));
    };
    if call_kind != output_kind {
        return Err(ExecutorError::InvalidRequest(format!(
            "{} cannot resolve {} call_id '{call_id}'",
            output_kind.output_item_name(),
            call_kind.call_item_name()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::io::{
        CustomToolCall, CustomToolCallOutputMessage, FunctionToolResultMessage, InputFunctionToolCall, ShellCall,
        ShellCallAction, ShellCallOutputMessage, ToolCallOutput,
    };

    fn function_call(call_id: &str) -> InputItem {
        InputItem::FunctionCall(InputFunctionToolCall {
            id: None,
            call_id: call_id.to_owned(),
            name: "get_weather".to_owned(),
            namespace: None,
            arguments: "{}".to_owned(),
            status: None,
        })
    }

    fn function_call_output(call_id: &str) -> InputItem {
        InputItem::FunctionCallOutput(FunctionToolResultMessage {
            call_id: call_id.to_owned(),
            output: ToolCallOutput::Text(String::new()),
        })
    }

    fn custom_tool_call(call_id: &str) -> InputItem {
        InputItem::CustomToolCall(CustomToolCall {
            id: String::new(),
            status: None,
            call_id: call_id.to_owned(),
            name: "freeform".to_owned(),
            input: String::new(),
        })
    }

    fn custom_tool_call_output(call_id: &str) -> InputItem {
        InputItem::CustomToolCallOutput(CustomToolCallOutputMessage {
            call_id: call_id.to_owned(),
            name: None,
            output: ToolCallOutput::Text(String::new()),
        })
    }

    fn shell_call(call_id: &str) -> InputItem {
        InputItem::ShellCall(ShellCall {
            id: None,
            call_id: call_id.to_owned(),
            action: ShellCallAction {
                commands: vec!["pwd".to_owned()],
                timeout_ms: None,
                max_output_length: None,
                extra: std::collections::HashMap::new(),
            },
            status: None,
            extra: std::collections::HashMap::new(),
        })
    }

    fn shell_call_output(call_id: &str) -> InputItem {
        InputItem::ShellCallOutput(ShellCallOutputMessage {
            id: None,
            call_id: call_id.to_owned(),
            max_output_length: None,
            output: Vec::new(),
            status: None,
            extra: std::collections::HashMap::new(),
        })
    }

    #[test]
    fn resolved_calls_are_not_pending() {
        let items = vec![function_call("call_1"), function_call_output("call_1")];
        assert!(pending_calls(&items).expect("valid call/output pair").is_empty());
    }

    #[test]
    fn unresolved_function_call_is_reported_in_order() {
        let items = vec![
            function_call("call_1"),
            function_call("call_2"),
            function_call_output("call_1"),
        ];
        let pending = pending_calls(&items).expect("valid partial call/output sequence");
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].call_id, "call_2");
    }

    #[test]
    fn unresolved_custom_tool_call_is_reported() {
        let items = vec![custom_tool_call("call_1")];
        let pending = pending_calls(&items).expect("valid unresolved custom call");
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].call_id, "call_1");
    }

    #[test]
    fn custom_tool_call_output_resolves_custom_tool_call() {
        let items = vec![custom_tool_call("call_1"), custom_tool_call_output("call_1")];
        assert!(pending_calls(&items).expect("valid custom call/output pair").is_empty());
    }

    #[test]
    fn shell_call_output_resolves_shell_call() {
        let items = vec![shell_call("call_1"), shell_call_output("call_1")];
        assert!(pending_calls(&items).expect("valid shell call/output pair").is_empty());
    }

    #[test]
    fn empty_items_have_no_pending_calls() {
        assert!(pending_calls(&[]).expect("empty history is valid").is_empty());
    }

    fn assert_invalid(items: &[InputItem], expected: &str) {
        let error = pending_calls(items).expect_err("invalid call/output sequence must be rejected");
        assert!(error.to_string().contains(expected), "unexpected error: {error}");
    }

    #[test]
    fn empty_and_duplicate_call_ids_are_rejected() {
        assert_invalid(&[function_call("")], "function_call call_id must not be empty");
        assert_invalid(&[custom_tool_call("")], "custom_tool_call call_id must not be empty");
        assert_invalid(&[shell_call("")], "shell_call call_id must not be empty");
        assert_invalid(
            &[
                function_call("call_1"),
                function_call("call_1"),
                function_call_output("call_1"),
            ],
            "duplicate call_id 'call_1'",
        );
        assert_invalid(
            &[
                function_call("call_1"),
                function_call_output("call_1"),
                custom_tool_call("call_1"),
            ],
            "duplicate call_id 'call_1'",
        );
    }

    #[test]
    fn outputs_must_resolve_exactly_one_call_of_the_same_kind() {
        assert_invalid(
            &[function_call_output("")],
            "function_call_output call_id must not be empty",
        );
        assert_invalid(&[function_call_output("call_1")], "without a pending call");
        assert_invalid(
            &[
                function_call("call_1"),
                function_call_output("call_1"),
                function_call_output("call_1"),
            ],
            "without a pending call",
        );
        assert_invalid(
            &[function_call("call_1"), custom_tool_call_output("call_1")],
            "cannot resolve function_call call_id 'call_1'",
        );
        assert_invalid(
            &[shell_call("call_1"), function_call_output("call_1")],
            "cannot resolve shell_call call_id 'call_1'",
        );
    }
}
