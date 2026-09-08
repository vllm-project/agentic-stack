use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};
use serde_json::json;

use agentic_core::executor::{ExecutorError, create_conversation};

use super::super::common::{executor_error_response, extract_store, read_bytes};
use crate::app::AppState;

pub async fn conversations(State(state): State<AppState>, req: Request) -> Response {
    let (_, body) = req.into_parts();
    let bytes = match read_bytes(body, state.max_request_body_size).await {
        Ok(b) => b,
        Err(e) => return e,
    };

    if !extract_store(&bytes) {
        return executor_error_response(ExecutorError::InvalidRequest("conversations require store=true".into()));
    }

    match create_conversation(&state.exec_ctx).await {
        Ok(data) => axum::Json(json!({
            "id": data.conversation_id,
            "created_at": data.created_at,
            "object": "conversation",
            "metadata": {}
        }))
        .into_response(),
        Err(e) => executor_error_response(e),
    }
}
