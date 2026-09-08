use http::StatusCode;
use thiserror::Error;

use crate::StorageError;
use crate::tool::ToolError;
use crate::utils::common::serialize_to_vec_or_default;

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ExecutorError {
    /// A storage layer operation failed.
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    /// Persistence failed after inference completed.
    ///
    /// The source is retained for internal diagnostics while the display
    /// message remains safe to send to API clients.
    #[error("failed to persist response")]
    Persistence(#[source] Box<ExecutorError>),

    /// A persisted conversation changed after its history was read.
    ///
    /// The storage source is retained for internal diagnostics while the
    /// display message remains safe to send to API clients.
    #[error("conversation changed while the response was being generated; retry the request")]
    ConversationLocked {
        #[source]
        source: StorageError,
    },

    /// The LLM backend returned a non-2xx HTTP response.
    #[error("LLM request failed ({status}): {body}")]
    LLMRequest {
        status: StatusCode,
        body: String,
        headers: http::HeaderMap,
    },

    /// The LLM backend could not be reached or timed out before responding.
    #[error("{message}")]
    LLMTransport { status: StatusCode, message: &'static str },

    /// A network error occurred reading from the LLM response stream.
    ///
    /// The original `reqwest::Error` is preserved as the error source so
    /// callers can inspect the underlying network failure.
    #[error("network error: {0}")]
    NetworkError(
        #[from]
        #[source]
        reqwest::Error,
    ),

    /// JSON deserialisation failed.
    ///
    /// The original `serde_json::Error` is preserved as the error source so
    /// callers can inspect the exact parse failure location and kind.
    #[error("json error: {0}")]
    JsonError(
        #[from]
        #[source]
        serde_json::Error,
    ),

    /// A general stream processing error with a human-readable message.
    ///
    /// Used for non-network stream failures (e.g. worker thread panic).
    #[error("stream error: {0}")]
    StreamError(String),

    /// A validation error on the request payload with a human-readable message.
    ///
    /// Used when required fields are missing or structurally invalid.
    #[error("parse error: {0}")]
    ParseError(String),

    #[error("{entity} not found: {id}")]
    NotFound { entity: String, id: String },

    /// A response session cannot resolve the requested continuation checkpoint.
    #[error("Previous response with id '{id}' not found.")]
    PreviousResponseNotFound { id: String },

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// The request exceeds a documented transport or component size budget.
    #[error("{0}")]
    PayloadTooLarge(String),

    /// The request conflicts with state already stored.
    #[error("conflict: {0}")]
    Conflict(String),

    #[error("compaction summarization failed with status '{status}': {details}")]
    CompactionFailed { status: String, details: String },

    #[error("tool error: {0}")]
    Tool(#[from] ToolError),
}

impl ExecutorError {
    pub(crate) fn is_invalid_upstream_tool_search(&self) -> bool {
        matches!(
            self,
            Self::Tool(ToolError::InvalidUpstreamToolSearch | ToolError::UpstreamWithheldFunctionCall)
        )
    }

    fn client_visible_error(&self) -> &Self {
        match self {
            Self::Persistence(source) if source.contains_conversation_locked() => source.client_visible_error(),
            _ => self,
        }
    }

    fn contains_conversation_locked(&self) -> bool {
        match self {
            Self::ConversationLocked { .. } => true,
            Self::Persistence(source) => source.contains_conversation_locked(),
            _ => false,
        }
    }

    /// HTTP status code that best represents this error to an API caller.
    #[must_use]
    pub fn http_status(&self) -> StatusCode {
        match self.client_visible_error() {
            Self::Storage(e) if e.is_not_found() => StatusCode::NOT_FOUND,
            Self::LLMRequest { status, .. } | Self::LLMTransport { status, .. } => *status,
            Self::ConversationLocked { .. }
            | Self::Tool(ToolError::Config(_) | ToolError::MissingOutput { .. })
            | Self::InvalidRequest(_)
            | Self::PreviousResponseNotFound { .. }
            | Self::JsonError(_) => StatusCode::BAD_REQUEST,
            Self::Tool(
                ToolError::Execution(_)
                | ToolError::InvalidUpstreamToolSearch
                | ToolError::UpstreamWithheldFunctionCall,
            )
            | Self::CompactionFailed { .. } => StatusCode::BAD_GATEWAY,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::PayloadTooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            Self::ParseError(_) => StatusCode::UNPROCESSABLE_ENTITY,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Machine-readable error type for the API error envelope.
    #[must_use]
    pub fn error_type(&self) -> &'static str {
        match self.client_visible_error() {
            Self::ConversationLocked { .. }
            | Self::Tool(ToolError::Config(_) | ToolError::MissingOutput { .. })
            | Self::InvalidRequest(_)
            | Self::PreviousResponseNotFound { .. }
            | Self::ParseError(_)
            | Self::JsonError(_)
            | Self::PayloadTooLarge(_) => "invalid_request_error",
            Self::Storage(e) if e.is_not_found() => "not_found",
            Self::Conflict(_) => "conflict_error",
            Self::LLMRequest { .. } | Self::LLMTransport { .. } | Self::CompactionFailed { .. } => "upstream_error",
            Self::Tool(
                ToolError::Execution(_)
                | ToolError::InvalidUpstreamToolSearch
                | ToolError::UpstreamWithheldFunctionCall,
            ) => "tool_error",
            _ => "server_error",
        }
    }

    /// Short machine-readable error code for the API error envelope.
    #[must_use]
    pub fn error_code(&self) -> &'static str {
        match self.client_visible_error() {
            Self::ConversationLocked { .. } => "conversation_locked",
            Self::PreviousResponseNotFound { .. } => "previous_response_not_found",
            Self::Conflict(_) => "response_already_stored",
            Self::PayloadTooLarge(_) => "body_too_large",
            other => other.error_type(),
        }
    }

    /// Request parameter associated with the API error, when applicable.
    #[must_use]
    pub fn error_param(&self) -> Option<&'static str> {
        match self.client_visible_error() {
            Self::ConversationLocked { .. } => Some("conversation"),
            Self::PreviousResponseNotFound { .. } => Some("previous_response_id"),
            Self::Tool(ToolError::MissingOutput { .. }) => Some("input"),
            _ => None,
        }
    }

    /// Client-safe message for the API error envelope.
    #[must_use]
    pub fn error_message(&self) -> String {
        match self.client_visible_error() {
            Self::Tool(error @ ToolError::MissingOutput { .. }) => error.to_string(),
            other => other.to_string(),
        }
    }

    /// Builds the OpenAI-compatible error object shared by HTTP and SSE.
    pub(crate) fn response_error(&self) -> serde_json::Value {
        let code = if matches!(self.client_visible_error(), Self::Tool(ToolError::MissingOutput { .. })) {
            serde_json::Value::Null
        } else {
            serde_json::json!(self.error_code())
        };
        let mut error = serde_json::Map::new();
        error.insert("message".to_owned(), serde_json::json!(self.error_message()));
        error.insert("type".to_owned(), serde_json::json!(self.error_type()));
        error.insert("code".to_owned(), code);
        if let Some(param) = self.error_param() {
            error.insert("param".to_owned(), serde_json::json!(param));
        }
        serde_json::Value::Object(error)
    }

    /// Serialise the error into the HTTP response body bytes.
    ///
    /// `LLMRequest` bodies are forwarded verbatim; all other variants are
    /// wrapped in the standard `{"error": {"message", "type", "code"}}` envelope.
    #[must_use]
    pub fn into_response_body(self) -> Vec<u8> {
        match self {
            Self::LLMRequest { body, .. } => body.into_bytes(),
            other => serialize_to_vec_or_default(&serde_json::json!({ "error": other.response_error() })),
        }
    }
}

pub type ExecutorResult<T> = Result<T, ExecutorError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_error_display() {
        let err = ExecutorError::InvalidRequest("test message".into());
        assert!(err.to_string().contains("invalid request"));
        assert!(err.to_string().contains("test message"));
    }

    #[test]
    fn test_executor_error_stream() {
        let err = ExecutorError::StreamError("connection lost".into());
        assert!(err.to_string().contains("stream error"));
    }

    #[test]
    fn test_executor_error_not_found() {
        let err = ExecutorError::NotFound {
            entity: "Conversation".into(),
            id: "conv_123".into(),
        };
        assert!(err.to_string().contains("Conversation"));
        assert!(err.to_string().contains("conv_123"));
    }

    #[test]
    fn test_executor_error_from_storage() {
        let storage_err = StorageError::NotConfigured;
        let exec_err = ExecutorError::from(storage_err);
        assert!(exec_err.to_string().contains("storage error"));
    }

    #[test]
    fn previous_response_error_has_the_continuation_envelope() {
        let error = ExecutorError::PreviousResponseNotFound {
            id: "resp_missing".to_owned(),
        };
        assert_eq!(error.http_status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&error.into_response_body()).unwrap(),
            serde_json::json!({"error": {
                "message": "Previous response with id 'resp_missing' not found.",
                "type": "invalid_request_error", "code": "previous_response_not_found",
                "param": "previous_response_id"
            }})
        );
    }

    #[test]
    fn tool_search_configuration_errors_are_bad_requests() {
        let error = ExecutorError::from(ToolError::Config("invalid tool_search request".to_owned()));

        assert_eq!(error.http_status(), StatusCode::BAD_REQUEST);
        assert_eq!(error.error_type(), "invalid_request_error");
    }

    #[test]
    fn test_executor_error_json_preserves_source() {
        use std::error::Error;
        let json_err: serde_json::Error = serde_json::from_str::<serde_json::Value>("{bad}").unwrap_err();
        let exec_err = ExecutorError::from(json_err);
        assert!(exec_err.source().is_some(), "source should be chained");
        assert!(exec_err.to_string().contains("json error"));
    }

    #[test]
    fn conversation_locked_response_preserves_conflict_through_persistence() {
        use std::error::Error;

        let error = ExecutorError::Persistence(Box::new(ExecutorError::ConversationLocked {
            source: StorageError::ConversationConflict {
                conversation_id: "conv_internal".to_owned(),
            },
        }));

        let conversation_locked = error.source().expect("persistence source must be retained");
        let conflict = conversation_locked
            .source()
            .expect("conversation conflict source must be retained");
        assert!(matches!(
            conflict.downcast_ref::<StorageError>(),
            Some(StorageError::ConversationConflict { conversation_id })
                if conversation_id == "conv_internal"
        ));

        assert_eq!(error.http_status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&error.into_response_body())
                .expect("valid error response JSON"),
            serde_json::json!({
                "error": {
                    "message": "conversation changed while the response was being generated; retry the request",
                    "type": "invalid_request_error",
                    "code": "conversation_locked",
                    "param": "conversation"
                }
            })
        );
    }

    #[test]
    fn non_conflict_response_omits_param() {
        let body = ExecutorError::InvalidRequest("invalid input".to_owned()).into_response_body();
        let value: serde_json::Value = serde_json::from_slice(&body).expect("valid error response JSON");

        assert!(!value["error"].as_object().expect("error object").contains_key("param"));
    }

    #[test]
    fn response_id_conflict_has_a_machine_readable_conflict_envelope() {
        let error = ExecutorError::Conflict("a turn is already stored under 'resp_1'".to_owned());

        assert_eq!(error.http_status(), StatusCode::CONFLICT);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&error.into_response_body())
                .expect("valid error response JSON"),
            serde_json::json!({
                "error": {
                    "message": "conflict: a turn is already stored under 'resp_1'",
                    "type": "conflict_error",
                    "code": "response_already_stored"
                }
            })
        );
    }

    #[test]
    fn missing_tool_output_matches_openai_error_envelope() {
        let error = ExecutorError::Tool(ToolError::MissingOutput {
            call_id: "call_test".to_owned(),
        });

        assert_eq!(error.http_status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&error.into_response_body())
                .expect("valid error response JSON"),
            serde_json::json!({
                "error": {
                    "message": "No tool output found for function call call_test.",
                    "type": "invalid_request_error",
                    "param": "input",
                    "code": null
                }
            })
        );
    }
}
