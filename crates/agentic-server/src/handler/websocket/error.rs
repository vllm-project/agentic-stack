use http::StatusCode;
use serde_json::{Value, json};
use thiserror::Error;

use agentic_core::executor::ExecutorError;

#[derive(Debug, Error)]
pub(super) enum WsError {
    #[error(transparent)]
    Executor(Box<ExecutorError>),

    #[error("invalid JSON: {0}")]
    InvalidJson(#[source] serde_json::Error),

    #[error("failed to serialize websocket event: {0}")]
    SerializeJson(#[source] serde_json::Error),

    #[error("websocket message type must be response.create")]
    UnexpectedType,

    #[error("websocket messages must be JSON text frames")]
    BinaryFrame,

    #[error("too many outstanding websocket response.create requests")]
    TooManyRequests,

    #[error("websocket send failed")]
    SendFailed,
}

impl From<ExecutorError> for WsError {
    fn from(error: ExecutorError) -> Self {
        Self::Executor(Box::new(error))
    }
}

impl WsError {
    pub(super) fn status(&self) -> StatusCode {
        match self {
            Self::Executor(err) => err.http_status(),
            Self::InvalidJson(_) | Self::UnexpectedType | Self::BinaryFrame => StatusCode::BAD_REQUEST,
            Self::TooManyRequests => StatusCode::TOO_MANY_REQUESTS,
            Self::SerializeJson(_) | Self::SendFailed => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub(super) fn code(&self) -> &'static str {
        match self {
            Self::Executor(err) => err.error_code(),
            Self::InvalidJson(_) => "invalid_json",
            Self::UnexpectedType | Self::BinaryFrame => "invalid_request_error",
            Self::TooManyRequests => "rate_limit_exceeded",
            Self::SerializeJson(_) | Self::SendFailed => "server_error",
        }
    }

    fn error_type(&self) -> &'static str {
        match self {
            Self::Executor(err) => err.error_type(),
            Self::InvalidJson(_) => "invalid_json",
            Self::UnexpectedType | Self::BinaryFrame => "invalid_request_error",
            Self::TooManyRequests => "rate_limit_error",
            Self::SerializeJson(_) | Self::SendFailed => "server_error",
        }
    }

    fn param(&self) -> Option<&'static str> {
        match self {
            Self::Executor(err) => err.error_param(),
            _ => None,
        }
    }

    fn message(&self) -> String {
        match self {
            Self::Executor(err) => err.error_message(),
            _ => self.to_string(),
        }
    }

    pub(super) fn to_ws_frame(&self) -> Option<Value> {
        if matches!(self, Self::SerializeJson(_) | Self::SendFailed) {
            return None;
        }

        let mut error = serde_json::Map::new();
        error.insert("message".to_owned(), Value::String(self.message()));
        error.insert("type".to_owned(), Value::String(self.error_type().to_owned()));
        error.insert("code".to_owned(), Value::String(self.code().to_owned()));
        if let Some(param) = self.param() {
            error.insert("param".to_owned(), Value::String(param.to_owned()));
        }
        Some(json!({
            "type": "error",
            "status": self.status().as_u16(),
            "error": error
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentic_core::StorageError;

    #[test]
    fn executor_conflict_ws_frame_uses_client_conflict_contract() {
        let error = WsError::Executor(Box::new(ExecutorError::Persistence(Box::new(
            ExecutorError::ConversationLocked {
                source: StorageError::ConversationConflict {
                    conversation_id: "conv_test".to_owned(),
                },
            },
        ))));

        assert_eq!(
            error.to_ws_frame().expect("client-visible websocket error"),
            json!({
                "type": "error",
                "status": 400,
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
    fn non_conflict_ws_frame_omits_param() {
        let frame = WsError::UnexpectedType
            .to_ws_frame()
            .expect("client-visible websocket error");

        assert!(!frame["error"].as_object().expect("error object").contains_key("param"));
    }
}
