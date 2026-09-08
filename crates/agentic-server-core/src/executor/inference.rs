//! HTTP transport layer for LLM backend communication.
//!
//! Handles sending requests, reading streaming chunks, and mapping network
//! and HTTP errors to [`ExecutorError`].

use std::sync::Arc;
use std::time::Duration;

use async_stream::stream;
use futures::{Stream, StreamExt};

use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::response_budget::MAX_EXECUTOR_RESPONSE_BYTES;
use crate::proxy::processed_response_headers;

/// SSE stream of raw lines sent to the client (`data: …\n\n` per event).
pub type BoxStream = std::pin::Pin<Box<dyn Stream<Item = String> + Send>>;

/// Wire-format marker signalling end-of-stream to the client.
pub(super) const DONE_MARKER: &str = "data: [DONE]\n\n";
const MAX_SSE_LINE_BYTES: usize = 256 * 1024;

/// Fetch the next raw bytes chunk from a streaming response.
///
/// Returns `Ok(Some(bytes))` on data, `Ok(None)` when the stream ends cleanly,
/// and `Err` on a network failure or chunk timeout.
pub(super) async fn next_chunk<S>(stream: &mut S, timeout: Duration) -> ExecutorResult<Option<bytes::Bytes>>
where
    S: futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    let item = if timeout.is_zero() {
        stream.next().await
    } else {
        tokio::time::timeout(timeout, stream.next()).await.map_err(|_| {
            ExecutorError::StreamError("chunk timeout: no data received within the configured window".into())
        })?
    };
    item.transpose().map_err(ExecutorError::NetworkError)
}

fn drain_complete_utf8_lines(buffer: &mut Vec<u8>) -> ExecutorResult<Vec<String>> {
    let mut lines = Vec::new();
    while let Some(pos) = buffer.iter().position(|byte| *byte == b'\n') {
        if pos > MAX_SSE_LINE_BYTES {
            return Err(ExecutorError::StreamError(format!(
                "upstream SSE line exceeded {MAX_SSE_LINE_BYTES} bytes"
            )));
        }
        let line = buffer.drain(..=pos).collect::<Vec<_>>();
        let line_end = if pos > 0 && line.get(pos - 1) == Some(&b'\r') {
            pos - 1
        } else {
            pos
        };
        if let Ok(line) = std::str::from_utf8(&line[..line_end]) {
            lines.push(line.to_string());
        }
    }
    if buffer.len() > MAX_SSE_LINE_BYTES {
        return Err(ExecutorError::StreamError(format!(
            "upstream SSE line exceeded {MAX_SSE_LINE_BYTES} bytes"
        )));
    }
    Ok(lines)
}

async fn response_text_limited(resp: reqwest::Response) -> ExecutorResult<String> {
    let mut stream = resp.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(ExecutorError::NetworkError)?;
        if chunk.len() > MAX_EXECUTOR_RESPONSE_BYTES.saturating_sub(body.len()) {
            return Err(ExecutorError::StreamError(format!(
                "upstream response exceeded {MAX_EXECUTOR_RESPONSE_BYTES} bytes"
            )));
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body)
        .map_err(|_| ExecutorError::StreamError("upstream response body was not valid UTF-8".to_owned()))
}

/// Build, send, and validate an HTTP POST to the LLM backend.
///
/// Shared by both the blocking path (caller consumes a bounded byte stream) and
/// the streaming path (caller reads `.bytes_stream()`). Maps connect/timeout failures and
/// non-2xx status codes to [`ExecutorError::LLMRequest`] and connection
/// failures to [`ExecutorError::LLMTransport`].
pub(super) async fn send_request(
    client: &reqwest::Client,
    url: &str,
    body: String,
    auth: Option<&str>,
    forwarded_headers: Option<&reqwest::header::HeaderMap>,
) -> ExecutorResult<reqwest::Response> {
    let mut headers = forwarded_headers.cloned().unwrap_or_default();
    headers
        .entry(reqwest::header::CONTENT_TYPE)
        .or_insert(reqwest::header::HeaderValue::from_static("application/json"));
    let mut req = client.post(url).headers(headers).body(body);
    if let Some(key) = auth {
        req = req.bearer_auth(key);
    }

    let resp = req.send().await.map_err(|e| ExecutorError::LLMTransport {
        status: if e.is_timeout() {
            http::StatusCode::GATEWAY_TIMEOUT
        } else {
            http::StatusCode::BAD_GATEWAY
        },
        message: if e.is_timeout() {
            "LLM timeout"
        } else {
            "LLM unavailable"
        },
    })?;

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let headers = processed_response_headers(resp.headers());
        // Log and discard any error reading the error body — the status code
        // is the primary signal; an empty body is acceptable here.
        let body = response_text_limited(resp)
            .await
            .inspect_err(|error| tracing::debug!(%error, "failed to read bounded error response body"))
            .unwrap_or_default();
        return Err(ExecutorError::LLMRequest {
            status: http::StatusCode::from_u16(status).unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR),
            body,
            headers,
        });
    }

    Ok(resp)
}

/// Makes a non-streaming HTTP POST to the LLM backend and returns the full JSON body.
///
/// Used by `run_blocking` so it can pass the result to [`ResponseAccumulator::from_json`](crate::executor::accumulator::ResponseAccumulator::from_json).
pub(super) async fn fetch_response_json(
    upstream_json: String,
    url: &str,
    client: &reqwest::Client,
    auth: Option<&str>,
) -> ExecutorResult<String> {
    let resp = send_request(client, url, upstream_json, auth, None).await?;
    // Preserve the reqwest::Error as the typed source (NetworkError).
    response_text_limited(resp).await
}

/// Makes a non-streaming HTTP POST with caller-supplied upstream headers.
pub(super) async fn fetch_response_json_with_headers(
    upstream_json: String,
    url: &str,
    client: &reqwest::Client,
    headers: &reqwest::header::HeaderMap,
) -> ExecutorResult<(String, http::HeaderMap)> {
    let resp = send_request(client, url, upstream_json, None, Some(headers)).await?;
    let response_headers = processed_response_headers(resp.headers());
    let body = response_text_limited(resp).await?;
    Ok((body, response_headers))
}

/// Step 2 — Call the LLM inference backend; yields raw SSE lines (`data: …`).
///
/// Always requests `stream=true` upstream. Stops on `[DONE]`.
///
/// # Errors
/// Each stream item is `Result<String, ExecutorError>`. The stream yields `Err` on:
/// - [`ExecutorError::LLMTransport`] — connect timeout (504) or connection failure (502)
/// - [`ExecutorError::LLMRequest`] — non-2xx HTTP status from the backend
/// - [`ExecutorError::NetworkError`] — network failure while reading the response body
pub fn call_inference(
    upstream_json: String,
    url: String,
    client: Arc<reqwest::Client>,
    auth: Option<String>,
    chunk_timeout: Duration,
) -> impl Stream<Item = Result<String, ExecutorError>> + Send + 'static {
    stream! {
        let resp = match send_request(&client, &url, upstream_json, auth.as_deref(), None).await {
            Ok(r) => r,
            Err(e) => { yield Err(e); return; }
        };

        let mut lines = Box::pin(response_lines(resp, chunk_timeout));
        while let Some(line) = lines.next().await {
            yield line;
        }
    }
}

/// Convert a successful upstream response body into normalized SSE data lines.
pub(super) fn response_lines(
    resp: reqwest::Response,
    chunk_timeout: Duration,
) -> impl Stream<Item = Result<String, ExecutorError>> + Send + 'static {
    stream! {
        let mut bytes = resp.bytes_stream();
        let mut buf = Vec::with_capacity(8192);

        loop {
            let chunk = match next_chunk(&mut bytes, chunk_timeout).await {
                Ok(Some(c)) => c,
                Ok(None) => break,
                Err(e) => { yield Err(e); return; }
            };

            buf.extend_from_slice(&chunk);

            let lines = match drain_complete_utf8_lines(&mut buf) {
                Ok(lines) => lines,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            for line in lines {
                match line.as_str() {
                    "data: [DONE]" => return,
                    l if l.starts_with("data: ") => yield Ok(line),
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use axum::body::Body;
    use axum::http::StatusCode;
    use axum::response::Response;
    use axum::routing::post;
    use bytes::Bytes;
    use futures::stream;

    use super::*;

    async fn oversized_body_server(status: StatusCode) -> (String, tokio::task::JoinHandle<()>) {
        let app = axum::Router::new().route(
            "/v1/responses",
            post(move || async move {
                let half = MAX_EXECUTOR_RESPONSE_BYTES / 2;
                let chunks = stream::iter([
                    Ok::<_, Infallible>(Bytes::from(vec![b'x'; half + 1])),
                    Ok(Bytes::from(vec![b'x'; MAX_EXECUTOR_RESPONSE_BYTES - half])),
                ]);
                Response::builder()
                    .status(status)
                    .body(Body::from_stream(chunks))
                    .unwrap()
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind response limit server");
        let address = listener.local_addr().expect("response limit server address");
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        (format!("http://{address}/v1/responses"), server)
    }

    #[test]
    fn utf8_line_reader_preserves_split_multibyte_characters() {
        let snowman = "\u{2603}";
        let line = format!(r#"data: {{"delta":"snow {snowman}"}}"#);
        let bytes = format!("{line}\n").into_bytes();
        let split_at = bytes
            .windows(snowman.len())
            .position(|window| window == snowman.as_bytes())
            .expect("snowman bytes present")
            + 1;
        let mut buffer = bytes[..split_at].to_vec();

        assert!(
            drain_complete_utf8_lines(&mut buffer)
                .expect("partial UTF-8 line")
                .is_empty()
        );

        buffer.extend_from_slice(&bytes[split_at..]);
        let lines = drain_complete_utf8_lines(&mut buffer).expect("complete UTF-8 line");

        assert!(buffer.is_empty());
        assert_eq!(lines, vec![line]);
        assert!(!lines[0].contains('\u{FFFD}'));
    }

    #[test]
    fn utf8_line_reader_rejects_an_oversized_unterminated_line() {
        let mut buffer = vec![b'x'; MAX_SSE_LINE_BYTES + 1];
        let error = drain_complete_utf8_lines(&mut buffer).expect_err("oversized SSE line must fail");
        assert!(error.to_string().contains("upstream SSE line exceeded"));
    }

    #[tokio::test]
    async fn blocking_response_reader_rejects_a_cumulative_oversized_body() {
        let (url, server) = oversized_body_server(StatusCode::OK).await;
        let error = fetch_response_json("{}".to_owned(), &url, &reqwest::Client::new(), None)
            .await
            .expect_err("oversized successful response must fail");

        assert!(error.to_string().contains("upstream response exceeded"));
        server.abort();
    }

    #[tokio::test]
    async fn non_success_response_discards_a_cumulative_oversized_body() {
        let (url, server) = oversized_body_server(StatusCode::BAD_GATEWAY).await;
        let error = send_request(&reqwest::Client::new(), &url, "{}".to_owned(), None, None)
            .await
            .expect_err("non-success response must fail");

        let ExecutorError::LLMRequest { status, body, .. } = error else {
            panic!("expected upstream request error");
        };
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert!(body.is_empty());
        server.abort();
    }
}
