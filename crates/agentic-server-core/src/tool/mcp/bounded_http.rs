use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use futures::{StreamExt, stream::BoxStream};
use http::header::{HeaderName, HeaderValue, WWW_AUTHENTICATE};
use rmcp::model::{ClientJsonRpcMessage, JsonRpcMessage, ServerJsonRpcMessage};
use rmcp::transport::streamable_http_client::{
    AuthRequiredError, InsufficientScopeError, SseError, StreamableHttpClient, StreamableHttpError,
    StreamableHttpPostResponse,
};
use rmcp_reqwest as http_client;
use sse_stream::{Sse, SseStream};

use crate::tool::handler::MAX_GATEWAY_TOOL_OUTPUT_BYTES;

const EVENT_STREAM_MIME_TYPE: &str = "text/event-stream";
const JSON_MIME_TYPE: &str = "application/json";
const HEADER_SESSION_ID: &str = "Mcp-Session-Id";
const HEADER_LAST_EVENT_ID: &str = "Last-Event-Id";
const HEADER_MCP_PROTOCOL_VERSION: &str = "MCP-Protocol-Version";
const MAX_MCP_HTTP_MESSAGE_BYTES: usize = MAX_GATEWAY_TOOL_OUTPUT_BYTES;

/// Reqwest adapter for rmcp 1.x that rejects oversized JSON bodies and SSE
/// events while they are being read, before protocol deserialization allocates
/// a complete untrusted response.
#[derive(Clone)]
pub(super) struct BoundedMcpHttpClient {
    inner: http_client::Client,
}

impl BoundedMcpHttpClient {
    pub(super) const fn new(inner: http_client::Client) -> Self {
        Self { inner }
    }
}

impl StreamableHttpClient for BoundedMcpHttpClient {
    type Error = http_client::Error;

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        last_event_id: Option<String>,
        auth_token: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, StreamableHttpError<Self::Error>> {
        let mut request = self
            .inner
            .get(uri.as_ref())
            .header(http_client::header::ACCEPT, accept_header())
            .header(HEADER_SESSION_ID, session_id.as_ref());
        if let Some(last_event_id) = last_event_id {
            request = request.header(HEADER_LAST_EVENT_ID, last_event_id);
        }
        if let Some(auth_token) = auth_token {
            request = request.bearer_auth(auth_token);
        }
        let response = apply_custom_headers(request, custom_headers)?.send().await?;
        if response.status() == http_client::StatusCode::METHOD_NOT_ALLOWED {
            return Err(StreamableHttpError::ServerDoesNotSupportSse);
        }
        let response = response.error_for_status()?;
        validate_stream_content_type(&response)?;
        Ok(bounded_sse_stream(response))
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        auth_token: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), StreamableHttpError<Self::Error>> {
        self.inner
            .delete_session(uri, session_id, auth_token, custom_headers)
            .await
    }

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_token: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        let mut request = self
            .inner
            .post(uri.as_ref())
            .header(http_client::header::ACCEPT, accept_header());
        if let Some(auth_token) = auth_token {
            request = request.bearer_auth(auth_token);
        }
        request = apply_custom_headers(request, custom_headers)?;
        let session_was_attached = session_id.is_some();
        if let Some(session_id) = session_id {
            request = request.header(HEADER_SESSION_ID, session_id.as_ref());
        }

        let response = request.json(&message).send().await?;
        reject_auth_response(&response)?;
        let status = response.status();
        if matches!(
            status,
            http_client::StatusCode::ACCEPTED | http_client::StatusCode::NO_CONTENT
        ) {
            return Ok(StreamableHttpPostResponse::Accepted);
        }
        if status == http_client::StatusCode::NOT_FOUND && session_was_attached {
            return Err(StreamableHttpError::SessionExpired);
        }

        let content_type = response_content_type(&response);
        let content_length = response.content_length();
        let response_session_id = response
            .headers()
            .get(HEADER_SESSION_ID)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        if status.is_success()
            && content_length == Some(0)
            && matches!(
                message,
                ClientJsonRpcMessage::Notification(_)
                    | ClientJsonRpcMessage::Response(_)
                    | ClientJsonRpcMessage::Error(_)
            )
        {
            return Ok(StreamableHttpPostResponse::Accepted);
        }

        if !status.is_success() {
            let body = response_body_limited(response).await?;
            if is_content_type(content_type.as_deref(), JSON_MIME_TYPE) {
                if let Ok(message @ JsonRpcMessage::Error(_)) = serde_json::from_slice(&body) {
                    return Ok(StreamableHttpPostResponse::Json(message, response_session_id));
                }
            }
            return Err(StreamableHttpError::UnexpectedServerResponse(Cow::Owned(format!(
                "HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            ))));
        }

        match content_type.as_deref() {
            Some(content_type) if content_type.as_bytes().starts_with(EVENT_STREAM_MIME_TYPE.as_bytes()) => Ok(
                StreamableHttpPostResponse::Sse(bounded_sse_stream(response), response_session_id),
            ),
            Some(content_type) if content_type.as_bytes().starts_with(JSON_MIME_TYPE.as_bytes()) => {
                let body = response_body_limited(response).await?;
                match serde_json::from_slice::<ServerJsonRpcMessage>(&body) {
                    Ok(message) => Ok(StreamableHttpPostResponse::Json(message, response_session_id)),
                    Err(error) => {
                        tracing::warn!(%error, "could not parse JSON response as an MCP message; treating it as accepted");
                        Ok(StreamableHttpPostResponse::Accepted)
                    }
                }
            }
            _ => Err(StreamableHttpError::UnexpectedContentType(content_type)),
        }
    }
}

fn accept_header() -> String {
    [EVENT_STREAM_MIME_TYPE, JSON_MIME_TYPE].join(", ")
}

fn apply_custom_headers(
    mut request: http_client::RequestBuilder,
    headers: HashMap<HeaderName, HeaderValue>,
) -> Result<http_client::RequestBuilder, StreamableHttpError<http_client::Error>> {
    for (name, value) in headers {
        if ["accept", HEADER_SESSION_ID, HEADER_LAST_EVENT_ID]
            .iter()
            .any(|reserved| name.as_str().eq_ignore_ascii_case(reserved))
            && !name.as_str().eq_ignore_ascii_case(HEADER_MCP_PROTOCOL_VERSION)
        {
            return Err(StreamableHttpError::ReservedHeaderConflict(name.to_string()));
        }
        request = request.header(name, value);
    }
    Ok(request)
}

fn reject_auth_response(response: &http_client::Response) -> Result<(), StreamableHttpError<http_client::Error>> {
    let Some(header) = response.headers().get(WWW_AUTHENTICATE) else {
        return Ok(());
    };
    let header = header.to_str().map_err(|_| {
        StreamableHttpError::UnexpectedServerResponse(Cow::Borrowed("invalid www-authenticate header value"))
    })?;
    match response.status() {
        http_client::StatusCode::UNAUTHORIZED => Err(StreamableHttpError::AuthRequired(AuthRequiredError::new(
            header.to_owned(),
        ))),
        http_client::StatusCode::FORBIDDEN => Err(StreamableHttpError::InsufficientScope(InsufficientScopeError::new(
            header.to_owned(),
            scope_from_auth_header(header),
        ))),
        _ => Ok(()),
    }
}

fn scope_from_auth_header(header: &str) -> Option<String> {
    let start = header.to_ascii_lowercase().find("scope=")? + "scope=".len();
    let value = &header[start..];
    if let Some(value) = value.strip_prefix('"') {
        return value.find('"').map(|end| value[..end].to_owned());
    }
    let end = value
        .find(|character: char| character == ',' || character == ';' || character.is_whitespace())
        .unwrap_or(value.len());
    (end > 0).then(|| value[..end].to_owned())
}

fn response_content_type(response: &http_client::Response) -> Option<String> {
    response
        .headers()
        .get(http_client::header::CONTENT_TYPE)
        .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned())
}

fn is_content_type(actual: Option<&str>, expected: &str) -> bool {
    actual.is_some_and(|actual| actual.as_bytes().starts_with(expected.as_bytes()))
}

fn validate_stream_content_type(
    response: &http_client::Response,
) -> Result<(), StreamableHttpError<http_client::Error>> {
    let content_type = response_content_type(response);
    if is_content_type(content_type.as_deref(), EVENT_STREAM_MIME_TYPE)
        || is_content_type(content_type.as_deref(), JSON_MIME_TYPE)
    {
        Ok(())
    } else {
        Err(StreamableHttpError::UnexpectedContentType(content_type))
    }
}

async fn response_body_limited(
    response: http_client::Response,
) -> Result<Vec<u8>, StreamableHttpError<http_client::Error>> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_MCP_HTTP_MESSAGE_BYTES as u64)
    {
        return Err(oversized_message_error());
    }
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if chunk.len() > MAX_MCP_HTTP_MESSAGE_BYTES.saturating_sub(body.len()) {
            return Err(oversized_message_error());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn bounded_sse_stream(response: http_client::Response) -> BoxStream<'static, Result<Sse, SseError>> {
    let bytes = async_stream::stream! {
        let mut source = response.bytes_stream();
        let mut event_size = SseEventSize::default();
        while let Some(chunk) = source.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    yield Err::<bytes::Bytes, _>(std::io::Error::other(error));
                    return;
                }
            };
            if let Err(error) = event_size.observe(&chunk) {
                yield Err::<bytes::Bytes, _>(error);
                return;
            }
            yield Ok::<_, std::io::Error>(chunk);
        }
    };
    SseStream::from_byte_stream(bytes).boxed()
}

#[derive(Default)]
struct SseEventSize {
    bytes: usize,
    at_line_start: bool,
    last_was_carriage_return: bool,
}

impl SseEventSize {
    fn observe(&mut self, chunk: &[u8]) -> std::io::Result<()> {
        for byte in chunk {
            self.bytes = self.bytes.saturating_add(1);
            match *byte {
                b'\r' => {
                    if self.at_line_start {
                        self.bytes = 0;
                    }
                    self.at_line_start = true;
                    self.last_was_carriage_return = true;
                }
                b'\n' if self.last_was_carriage_return => {
                    self.last_was_carriage_return = false;
                }
                b'\n' => {
                    if self.at_line_start {
                        self.bytes = 0;
                    }
                    self.at_line_start = true;
                }
                _ => {
                    self.at_line_start = false;
                    self.last_was_carriage_return = false;
                }
            }
            if self.bytes > MAX_MCP_HTTP_MESSAGE_BYTES {
                return Err(std::io::Error::other(format!(
                    "MCP SSE event exceeded {MAX_MCP_HTTP_MESSAGE_BYTES} bytes"
                )));
            }
        }
        Ok(())
    }
}

fn oversized_message_error() -> StreamableHttpError<http_client::Error> {
    StreamableHttpError::UnexpectedServerResponse(Cow::Owned(format!(
        "MCP HTTP response exceeded {MAX_MCP_HTTP_MESSAGE_BYTES} bytes"
    )))
}

#[cfg(test)]
mod tests {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::{MAX_MCP_HTTP_MESSAGE_BYTES, SseEventSize, response_body_limited};

    #[test]
    fn sse_event_limit_is_cumulative_across_chunks_and_resets_between_events() {
        let mut size = SseEventSize::default();
        size.observe(b"data: ok\n\n").expect("first event fits");
        size.observe(&vec![b'x'; MAX_MCP_HTTP_MESSAGE_BYTES / 2])
            .expect("first partial chunk fits");
        let error = size
            .observe(&vec![b'x'; MAX_MCP_HTTP_MESSAGE_BYTES / 2 + 1])
            .expect_err("cumulative oversized event must fail");
        assert!(error.to_string().contains("MCP SSE event exceeded"));
    }

    #[tokio::test]
    async fn json_body_limit_is_cumulative_across_http_chunks() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind bounded MCP response server");
        let address = listener.local_addr().expect("bounded MCP response server address");
        let chunk = vec![b'x'; MAX_MCP_HTTP_MESSAGE_BYTES / 2 + 1];
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept MCP response request");
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).await.expect("read MCP response request");
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n")
                .await
                .expect("write MCP response headers");
            for _ in 0..2 {
                stream
                    .write_all(format!("{:x}\r\n", chunk.len()).as_bytes())
                    .await
                    .expect("write MCP response chunk size");
                stream.write_all(&chunk).await.expect("write MCP response chunk");
                stream.write_all(b"\r\n").await.expect("finish MCP response chunk");
            }
            stream.write_all(b"0\r\n\r\n").await.expect("finish MCP response");
        });
        let response = rmcp_reqwest::Client::new()
            .get(format!("http://{address}/mcp"))
            .send()
            .await
            .expect("fetch chunked MCP response");

        let error = response_body_limited(response)
            .await
            .expect_err("cumulative oversized MCP body must fail");

        assert!(error.to_string().contains("MCP HTTP response exceeded"));
        server.await.expect("bounded MCP response server completes");
    }
}
