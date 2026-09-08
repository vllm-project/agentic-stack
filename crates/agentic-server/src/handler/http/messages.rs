use std::sync::Arc;

use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use http::HeaderMap;
use tracing::debug;

use agentic_core::executor::{
    ExecutorError, MessagesUpstream, normalize_native_web_search_for_upstream, run_messages_loop, run_messages_stream,
    validate_native_web_search_request,
};
use agentic_core::proxy::{
    ProxyAuth, ProxyBody, ProxyRequest, ProxyResponse, error_response_for_auth, proxy_request_with_path,
    upstream_request_headers,
};
use agentic_core::tool::ToolRegistry;
use agentic_core::types::messages::{MessagesRequest, has_gateway_tool, registry_tools};

use super::super::common::{convert_response, read_bytes_with_auth, sse_response_with_headers};
use crate::app::AppState;

async fn proxy_messages(
    state: &AppState,
    parts: axum::http::request::Parts,
    body: Bytes,
    path: &'static str,
) -> Response {
    convert_response(
        proxy_request_with_path(
            ProxyRequest {
                headers: parts.headers,
                body,
                query: parts.uri.query().map(str::to_owned),
            },
            path,
            ProxyAuth::Anthropic,
            &state.proxy_state,
        )
        .await,
    )
}

/// Preserve upstream Messages errors verbatim; render local executor failures
/// as an Anthropic error envelope, consistent with the proxy path (E14).
fn messages_error_response(err: ExecutorError) -> Response {
    if let ExecutorError::LLMRequest {
        status,
        body,
        mut headers,
    } = err
    {
        headers
            .entry(http::header::CONTENT_TYPE)
            .or_insert(http::HeaderValue::from_static("application/json"));
        return convert_response(ProxyResponse {
            status,
            headers,
            body: ProxyBody::Full(Bytes::from(body)),
        });
    }
    convert_response(error_response_for_auth(
        err.http_status(),
        err.error_code(),
        &err.to_string(),
        ProxyAuth::Anthropic,
    ))
}

/// Drive the Messages-native gateway tool loop (non-streaming or streaming) for
/// a request that declares a gateway-owned tool.
async fn execute_messages(
    state: &AppState,
    headers: &HeaderMap,
    query: Option<&str>,
    req: &MessagesRequest,
    body: &Bytes,
) -> Response {
    // Build the request-scoped registry from the declared tools (M6). Gateway
    // ownership (incl. configured aliases like Claude Code's `WebSearch`) is
    // resolved against the operator-configured map.
    let gateway_map = &state.exec_ctx.messages_gateway_tools;
    let mut tools = registry_tools(req.tools.as_ref(), gateway_map);
    let mut executors = state.exec_ctx.gateway_executors.clone();
    let registry = match ToolRegistry::build_with_handlers(&mut tools, &mut executors).await {
        Ok(r) => r,
        Err(e) => return messages_error_response(ExecutorError::from(e)),
    };

    // Parse the raw body to a JSON Value the loop forwards upstream untouched —
    // preserving every Anthropic field (tool_choice, stop_sequences, …).
    let request_json: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return messages_error_response(ExecutorError::from(e)),
    };
    if let Err(error) = validate_native_web_search_request(&request_json) {
        return messages_error_response(error);
    }

    let upstream = MessagesUpstream::new(
        &state.exec_ctx.llm_base_url,
        query,
        upstream_request_headers(headers, &state.proxy_state.config, ProxyAuth::Anthropic),
    );
    if req.stream {
        match run_messages_stream(request_json, Arc::new(registry), Arc::clone(&state.exec_ctx), upstream).await {
            Ok(response) => sse_response_with_headers(response.body, response.headers),
            Err(e) => messages_error_response(e),
        }
    } else {
        match run_messages_loop(request_json, &registry, &state.exec_ctx, &upstream).await {
            Ok(message) => {
                let mut response = axum::Json(message.body).into_response();
                response.headers_mut().extend(message.headers);
                response.headers_mut().insert(
                    http::header::CONTENT_TYPE,
                    http::HeaderValue::from_static("application/json"),
                );
                response
            }
            Err(e) => messages_error_response(e),
        }
    }
}

pub async fn messages(State(state): State<AppState>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let bytes: Bytes = match read_bytes_with_auth(body, ProxyAuth::Anthropic, state.max_request_body_size).await {
        Ok(bytes) => bytes,
        Err(response) => return response,
    };

    // Route to the loop only when a gateway-owned tool is declared; everything
    // else keeps the transparent proxy path.
    if let Ok(req) = serde_json::from_slice::<MessagesRequest>(&bytes) {
        let route_to_loop = has_gateway_tool(req.tools.as_ref(), &state.exec_ctx.messages_gateway_tools);
        debug!(
            route = if route_to_loop { "messages_loop" } else { "proxy" },
            stream = req.stream,
            tools = req.tools.as_ref().map_or(0, Vec::len),
            "routing HTTP messages request"
        );
        if route_to_loop {
            return execute_messages(&state, &parts.headers, parts.uri.query(), &req, &bytes).await;
        }
    }

    proxy_messages(&state, parts, bytes, "/v1/messages").await
}

pub async fn count_tokens(State(state): State<AppState>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let mut bytes: Bytes = match read_bytes_with_auth(body, ProxyAuth::Anthropic, state.max_request_body_size).await {
        Ok(bytes) => bytes,
        Err(response) => return response,
    };
    if let Ok(mut request_json) = serde_json::from_slice::<serde_json::Value>(&bytes) {
        match normalize_native_web_search_for_upstream(&mut request_json) {
            Ok(true) => match serde_json::to_vec(&request_json) {
                Ok(body) => bytes = Bytes::from(body),
                Err(error) => return messages_error_response(ExecutorError::from(error)),
            },
            Ok(false) => {}
            Err(error) => return messages_error_response(error),
        }
    }
    proxy_messages(&state, parts, bytes, "/v1/messages/count_tokens").await
}
