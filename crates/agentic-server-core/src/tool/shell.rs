use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub use tokio_util::sync::CancellationToken;

use crate::types::io::{FunctionTool, FunctionToolCall, OutputItem, ShellCall, ShellCallAction, ShellCallStatus};
use crate::types::tools::{ShellEnvironment, ShellToolParam};

use super::{ToolEntry, ToolError, ToolHandler, ToolType};

pub(crate) const SHELL_FUNCTION_NAME: &str = "shell";

/// Handler for the client-owned local `type: "shell"` tool.
///
/// The declaration is normalized to a function for inference, then restored
/// to a typed `shell_call` before it is returned to the client. This handler
/// deliberately does not implement `GatewayExecutor`: declaring a shell tool
/// never grants the gateway permission to execute arbitrary commands.
#[derive(Debug)]
pub struct ShellHandler;

impl ShellHandler {
    /// Lower only the inference copy; storage keeps the public shell history.
    pub(crate) fn model_input(
        input: std::borrow::Cow<'_, crate::types::io::ResponsesInput>,
    ) -> Result<std::borrow::Cow<'_, crate::types::io::ResponsesInput>, ToolError> {
        use crate::types::io::{FunctionToolResultMessage, InputFunctionToolCall, InputItem, ResponsesInput};
        use crate::utils::common::serialize_to_string;
        let ResponsesInput::Items(items) = input.as_ref() else {
            return Ok(input);
        };
        if !items
            .iter()
            .any(|item| matches!(item, InputItem::ShellCall(_) | InputItem::ShellCallOutput(_)))
        {
            return Ok(input);
        }
        let items = items
            .iter()
            .map(|item| {
                Ok(match item {
                    InputItem::ShellCall(call) => InputItem::FunctionCall(InputFunctionToolCall {
                        id: call.id.as_ref().map(|id| {
                            id.strip_prefix("sh_")
                                .map_or_else(|| id.clone(), |suffix| format!("fc_{suffix}"))
                        }),
                        call_id: call.call_id.clone(),
                        name: SHELL_FUNCTION_NAME.to_owned(),
                        namespace: None,
                        arguments: serialize_to_string(&call.action)
                            .map_err(|error| ToolError::Config(error.to_string()))?,
                        status: match call.status {
                            Some(ShellCallStatus::Completed) => Some(crate::types::event::MessageStatus::Completed),
                            Some(ShellCallStatus::InProgress) => Some(crate::types::event::MessageStatus::InProgress),
                            _ => None,
                        },
                    }),
                    InputItem::ShellCallOutput(output) => InputItem::FunctionCallOutput(FunctionToolResultMessage {
                        call_id: output.call_id.clone(),
                        output: serialize_to_string(&output.output)
                            .map_err(|error| ToolError::Config(error.to_string()))?
                            .into(),
                    }),
                    other => other.clone(),
                })
            })
            .collect::<Result<Vec<_>, ToolError>>()?;
        Ok(std::borrow::Cow::Owned(ResponsesInput::Items(items)))
    }

    #[must_use]
    pub(crate) fn output_item(call: &FunctionToolCall) -> Option<OutputItem> {
        Self::output_item_with_status(call, call.status.into())
    }

    #[must_use]
    pub(crate) fn output_item_with_status(call: &FunctionToolCall, status: ShellCallStatus) -> Option<OutputItem> {
        let action = crate::utils::common::deserialize_from_str::<ShellCallAction>(&call.arguments).ok()?;
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

pub(crate) fn insert_shell_entry(
    entries: &mut HashMap<String, ToolEntry>,
    params: &ShellToolParam,
    executor: Option<Arc<dyn ShellExecutor>>,
) {
    let entry = executor.map_or_else(
        || ToolEntry::client(ToolType::Shell, None),
        |executor| {
            ToolEntry::gateway(
                ToolType::Shell,
                None,
                Some(super::GatewayBinding::new(
                    Arc::new(ShellGatewayExecutor(executor)),
                    params.clone(),
                )),
            )
        },
    );
    entries.insert(SHELL_FUNCTION_NAME.to_owned(), entry);
}

/// Opt-in adapter implemented by an application's sandbox, never by the model.
///
/// Implementations must enforce the action's limits, stop subprocesses when the
/// token is cancelled (including when this future is dropped), and apply their
/// own filesystem/network/command policy. The core never starts a local shell.
pub trait ShellExecutor: Send + Sync + 'static {
    /// Execute one typed call, returning exactly one result per command.
    /// `timeout_ms` is capped at 60 seconds and `max_output_length` at 1 MiB
    /// of UTF-8 stdout/stderr bytes in total; omitted limits use those ceilings.
    ///
    /// # Errors
    /// Return [`ToolError::Execution`] when policy denies execution or the sandbox fails.
    fn execute(
        &self,
        call: ShellCall,
        cancellation: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<crate::types::io::ShellCallOutputContent>, ToolError>> + Send + '_>>;
}

struct ShellGatewayExecutor(Arc<dyn ShellExecutor>);

impl ToolHandler for ShellGatewayExecutor {
    type ToolParams = ShellToolParam;
    fn tool_type(&self) -> ToolType {
        ToolType::Shell
    }
    fn validate(&self, params: &ShellToolParam) -> Result<(), ToolError> {
        ShellHandler.validate(params)
    }
    fn normalize(&self, params: &ShellToolParam) -> Vec<FunctionTool> {
        ShellHandler.normalize(params)
    }
}

impl super::GatewayExecutor for ShellGatewayExecutor {
    type ExecutionParams = ShellToolParam;

    fn execute(
        &self,
        call_id: &str,
        _tool_name: &str,
        arguments: &str,
        _params: &ShellToolParam,
    ) -> Pin<Box<dyn Future<Output = Result<super::ToolOutput, ToolError>> + Send + '_>> {
        let call_id = call_id.to_owned();
        let action = crate::utils::common::deserialize_from_str::<ShellCallAction>(arguments);
        Box::pin(async move {
            let mut action = action.map_err(|error| ToolError::Execution(error.to_string()))?;
            if action.commands.is_empty() {
                return Err(ToolError::Execution("shell requires at least one command".to_owned()));
            }
            // Hard ceilings also apply to non-strict model-generated arguments.
            action.timeout_ms = Some(action.timeout_ms.unwrap_or(60_000).min(60_000));
            let output_limit = action.max_output_length.unwrap_or(1_048_576).min(1_048_576);
            action.max_output_length = Some(output_limit);
            let timeout = std::time::Duration::from_millis(action.timeout_ms.unwrap_or(60_000));
            let command_count = action.commands.len();
            let cancellation = CancellationToken::new();
            let _cancel_on_drop = cancellation.clone().drop_guard();
            let call = ShellCall {
                id: None,
                call_id: call_id.clone(),
                action,
                status: Some(ShellCallStatus::InProgress),
                extra: HashMap::new(),
            };
            let outputs = tokio::time::timeout(timeout, self.0.execute(call, cancellation))
                .await
                .map_err(|_| ToolError::Execution("shell execution timed out".to_owned()))??;
            if outputs.len() != command_count {
                return Err(ToolError::Execution(
                    "shell executor must return one output per command".to_owned(),
                ));
            }
            if outputs
                .iter()
                .map(|output| output.stdout.len().saturating_add(output.stderr.len()))
                .sum::<usize>()
                > usize::try_from(output_limit).unwrap_or(usize::MAX)
            {
                return Err(ToolError::Execution(
                    "shell executor exceeded max_output_length".to_owned(),
                ));
            }
            let output = crate::utils::common::serialize_to_string(&outputs)
                .map_err(|error| ToolError::Execution(error.to_string()))?;
            Ok(super::ToolOutput { call_id, output })
        })
    }

    fn started_output(&self, call: &FunctionToolCall, _params: &ShellToolParam) -> Option<OutputItem> {
        ShellHandler::output_item_with_status(call, ShellCallStatus::InProgress)
    }

    fn public_output(
        &self,
        call: &FunctionToolCall,
        _output: &super::ToolOutput,
        status: crate::types::io::output::GatewayCallStatus,
        _params: &ShellToolParam,
    ) -> Option<OutputItem> {
        ShellHandler::output_item_with_status(
            call,
            match status {
                crate::types::io::output::GatewayCallStatus::InProgress => ShellCallStatus::InProgress,
                crate::types::io::output::GatewayCallStatus::Completed => ShellCallStatus::Completed,
                crate::types::io::output::GatewayCallStatus::Failed => ShellCallStatus::Incomplete,
            },
        )
    }
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
    use crate::tool::GatewayExecutor;
    use crate::types::event::MessageStatus;

    #[derive(Default)]
    struct PendingSandbox(std::sync::Mutex<Option<(ShellCall, CancellationToken)>>);

    impl ShellExecutor for PendingSandbox {
        fn execute(
            &self,
            call: ShellCall,
            cancellation: CancellationToken,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<crate::types::io::ShellCallOutputContent>, ToolError>> + Send + '_>>
        {
            *self.0.lock().unwrap() = Some((call, cancellation));
            Box::pin(std::future::pending())
        }
    }

    #[tokio::test]
    async fn shell_future_drop_cancels_sandbox_and_caps_limits() {
        let sandbox = Arc::new(PendingSandbox::default());
        let executor = ShellGatewayExecutor(sandbox.clone());
        let params = local_shell();
        let mut future = executor.execute(
            "call_1",
            "shell",
            r#"{"commands":["pwd"],"timeout_ms":999999,"max_output_length":9999999}"#,
            &params,
        );
        assert!(futures::poll!(future.as_mut()).is_pending());
        let token = {
            let seen = sandbox.0.lock().unwrap();
            let (call, token) = seen.as_ref().unwrap();
            assert_eq!(call.call_id, "call_1");
            assert_eq!(call.action.timeout_ms, Some(60_000));
            assert_eq!(call.action.max_output_length, Some(1_048_576));
            assert!(!token.is_cancelled());
            token.clone()
        };
        drop(future);
        assert!(
            token.is_cancelled(),
            "disconnect/outer timeout must cancel sandbox work"
        );
    }

    #[tokio::test]
    async fn shell_timeout_cancels_sandbox() {
        let sandbox = Arc::new(PendingSandbox::default());
        let executor = ShellGatewayExecutor(sandbox.clone());
        let error = executor
            .execute(
                "call_1",
                "shell",
                r#"{"commands":["pwd"],"timeout_ms":1}"#,
                &local_shell(),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("timed out"));
        assert!(sandbox.0.lock().unwrap().as_ref().unwrap().1.is_cancelled());
    }

    struct OversizedSandbox;
    impl ShellExecutor for OversizedSandbox {
        fn execute(
            &self,
            _call: ShellCall,
            _cancellation: CancellationToken,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<crate::types::io::ShellCallOutputContent>, ToolError>> + Send + '_>>
        {
            Box::pin(async {
                Ok(vec![crate::types::io::ShellCallOutputContent {
                    stdout: "oversized".to_owned(),
                    stderr: String::new(),
                    outcome: crate::types::io::ShellCallOutcome::Exit { exit_code: 0 },
                    extra: HashMap::new(),
                }])
            })
        }
    }

    #[tokio::test]
    async fn shell_rejects_oversized_outputs_and_invalid_calls() {
        let executor = ShellGatewayExecutor(Arc::new(OversizedSandbox));
        let params = local_shell();
        for (arguments, expected) in [
            (
                r#"{"commands":["pwd"],"max_output_length":1}"#,
                "exceeded max_output_length",
            ),
            (r#"{"commands":["pwd","pwd"]}"#, "one output per command"),
            (r#"{"commands":[]}"#, "at least one command"),
            ("not-json", "execution failed"),
        ] {
            let error = executor
                .execute("call_1", "shell", arguments, &params)
                .await
                .unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
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
