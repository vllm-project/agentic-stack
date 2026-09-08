use axum::extract::{Request, State};
use axum::http::request::Parts;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use either::Either;
use serde_json::value::RawValue;
use tracing::debug;

use std::sync::Arc;

use agentic_core::executor::{ExecuteRequest, compact_response as execute_compaction};
use agentic_core::proxy::{ProxyRequest, proxy_request};
use agentic_core::tool::ToolSearchHandler;
use agentic_core::types::request_response::{CompactRequest, RequestPayload, ResponseTextConfig};

use super::super::common::{
    convert_response, executor_error_response, extract_bearer, read_bytes, read_json, sse_response,
};
use crate::app::AppState;

type RoutingPayload = RequestPayload<RawValue>;

async fn proxy_responses(state: &AppState, parts: Parts, body: Bytes) -> Response {
    let proxy_req = ProxyRequest {
        headers: parts.headers,
        body,
        query: parts.uri.query().map(str::to_string),
    };
    convert_response(proxy_request(proxy_req, &state.proxy_state).await)
}

async fn execute_responses(state: &AppState, parts: Parts, payload: RequestPayload) -> Response {
    let auth = extract_bearer(&parts.headers, state.openai_api_key.as_deref());
    match ExecuteRequest::new(payload, Arc::clone(&state.exec_ctx))
        .with_auth(auth)
        .run()
        .await
    {
        Ok(Either::Left(response_payload)) => axum::Json(response_payload).into_response(),
        Ok(Either::Right(stream)) => sse_response(stream),
        Err(e) => executor_error_response(e),
    }
}

pub async fn responses(State(state): State<AppState>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let bytes = match read_bytes(body, state.max_request_body_size).await {
        Ok(bytes) => bytes,
        Err(response) => return response,
    };
    let routing_payload = match serde_json::from_slice::<RoutingPayload>(&bytes) {
        Ok(payload) => payload,
        Err(error) => return executor_error_response(error.into()),
    };

    let has_tool_search_state = ToolSearchHandler::request_has_state(&routing_payload);
    let should_execute = routing_payload.store
        || routing_payload.previous_response_id.is_some()
        || has_tool_search_state
        || routing_payload.in_process_feature().is_some();
    debug!(
        route = if should_execute { "executor" } else { "proxy" },
        store = routing_payload.store,
        stream = routing_payload.stream,
        has_previous_response_id = routing_payload.previous_response_id.is_some(),
        has_conversation_id = routing_payload.conversation_id.is_some(),
        has_compaction = routing_payload.input.contains_compaction(),
        has_compaction_trigger = routing_payload.input.has_compaction_trigger(),
        has_tool_search_state,
        context_management = routing_payload.context_management.as_ref().map_or(0, Vec::len),
        tools = routing_payload.tools.as_ref().map_or(0, Vec::len),
        "routing HTTP responses request"
    );

    if should_execute {
        let payload = match routing_payload
            .try_map_text(|text| serde_json::from_str::<ResponseTextConfig>(text.get()).map(Box::new))
        {
            Ok(payload) => payload,
            Err(error) => return executor_error_response(error.into()),
        };
        execute_responses(&state, parts, payload).await
    } else {
        proxy_responses(&state, parts, bytes).await
    }
}

pub async fn compact_response(State(state): State<AppState>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let request: CompactRequest = match read_json(body, state.max_request_body_size).await {
        Ok(request) => request,
        Err(response) => return response,
    };
    let auth = extract_bearer(&parts.headers, state.openai_api_key.as_deref());
    match execute_compaction(request, state.exec_ctx.as_ref(), auth.as_deref()).await {
        Ok(response) => axum::Json(response).into_response(),
        Err(error) => executor_error_response(error),
    }
}
