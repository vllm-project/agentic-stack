use std::collections::VecDeque;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Extension, State};
use axum::http::HeaderMap;
use axum::response::Response;
use either::Either;
use futures::stream::{SplitSink, SplitStream};
use futures::{Sink, SinkExt, Stream, StreamExt};
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use agentic_core::ResponseUsage;
use agentic_core::executor::{
    BoxStream, ExecuteRequest, ExecutorError, RequestContext, persist_turn, rehydrate_conversation,
};
use agentic_core::types::request_response::RequestPayload;
use agentic_core::utils::common::utcnow_str;

use super::super::common::{MAX_BODY_SIZE, extract_bearer};
use super::error::WsError;
use crate::app::AppState;
use crate::auth::AuthenticatedPrincipal;

type WsSender = SplitSink<WebSocket, Message>;
type WsReceiver = SplitStream<WebSocket>;

pub async fn responses_ws(State(state): State<AppState>, headers: HeaderMap, ws: WebSocketUpgrade) -> Response {
    upgrade_responses_ws(state, headers, ws, None)
}

pub(crate) async fn responses_ws_with_auth(
    State(state): State<AppState>,
    principal: Option<Extension<AuthenticatedPrincipal>>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    upgrade_responses_ws(state, headers, ws, principal.map(|Extension(principal)| principal))
}

fn upgrade_responses_ws(
    state: AppState,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
    principal: Option<AuthenticatedPrincipal>,
) -> Response {
    let websocket_guard = state.websocket_tracker.track();
    ws.max_message_size(MAX_BODY_SIZE)
        .max_frame_size(MAX_BODY_SIZE)
        .on_upgrade(move |socket| async move {
            let _websocket_guard = websocket_guard;
            Box::pin(responses_ws_loop(socket, state, headers, principal)).await;
        })
}

async fn responses_ws_loop(
    socket: WebSocket,
    state: AppState,
    headers: HeaderMap,
    principal: Option<AuthenticatedPrincipal>,
) {
    debug!("responses websocket session opened");
    let shutdown_token = state.shutdown_token.clone();
    let (mut sender, mut receiver) = socket.split();

    // Requests received while a stream is active, processed in order after it completes.
    let mut queue: VecDeque<String> = VecDeque::new();

    loop {
        if shutdown_token.is_cancelled() {
            break;
        }
        let text = if let Some(buffered) = queue.pop_front() {
            buffered
        } else {
            let message = next_ws_message(&shutdown_token, &mut receiver).await;

            let Some(message) = message else {
                break;
            };

            match message {
                Ok(Message::Text(text)) => text.to_string(),
                Ok(Message::Binary(_)) => {
                    if !handle_ws_error(&mut sender, WsError::BinaryFrame).await {
                        break;
                    }
                    continue;
                }
                Ok(Message::Close(_)) => break,
                Ok(Message::Ping(payload)) => {
                    if sender.send(Message::Pong(payload)).await.is_err() {
                        break;
                    }
                    continue;
                }
                Ok(Message::Pong(_)) => continue,
                Err(e) => {
                    warn!("responses websocket receive error: {e}");
                    break;
                }
            }
        };

        if let Some(event) = websocket_identity_error_event(principal.as_ref()) {
            let _ = send_ws_json(&mut sender, event).await;
            break;
        }

        match handle_ws_text(
            &mut sender,
            &mut receiver,
            &state,
            &headers,
            &text,
            &shutdown_token,
            &mut queue,
        )
        .await
        {
            Ok(()) => {}
            Err(err) => {
                if !handle_ws_error(&mut sender, err).await {
                    break;
                }
            }
        }
    }
    close_ws(&mut sender, &mut receiver).await;
    debug!("responses websocket session closed");
}

fn websocket_identity_error_event(principal: Option<&AuthenticatedPrincipal>) -> Option<Value> {
    principal.is_some_and(AuthenticatedPrincipal::is_expired).then(|| {
        serde_json::json!({
            "type": "error",
            "code": "invalid_token",
            "message": "OIDC bearer token expired",
            "param": null,
            "sequence_number": 0,
        })
    })
}

async fn next_ws_message<Receiver>(
    shutdown_token: &CancellationToken,
    receiver: &mut Receiver,
) -> Option<Receiver::Item>
where
    Receiver: Stream + Unpin,
{
    tokio::select! {
        biased;
        () = shutdown_token.cancelled() => None,
        message = receiver.next() => {
            if shutdown_token.is_cancelled() {
                None
            } else {
                message
            }
        },
    }
}

fn keep_if_running<T>(shutdown_token: &CancellationToken, value: T) -> Option<T> {
    (!shutdown_token.is_cancelled()).then_some(value)
}

async fn close_ws<Sender, Receiver, SendError, ReceiveError>(sender: &mut Sender, receiver: &mut Receiver)
where
    Sender: Sink<Message, Error = SendError> + Unpin,
    Receiver: Stream<Item = Result<Message, ReceiveError>> + Unpin,
    SendError: std::fmt::Display,
    ReceiveError: std::fmt::Display,
{
    if let Err(error) = sender.close().await {
        debug!(%error, "failed to send responses websocket close frame");
        return;
    }

    while let Some(message) = receiver.next().await {
        match message {
            Ok(Message::Close(_)) => break,
            Ok(Message::Text(_) | Message::Binary(_) | Message::Ping(_) | Message::Pong(_)) => {}
            Err(error) => {
                debug!(%error, "responses websocket close handshake receive failed");
                break;
            }
        }
    }
}

/// Process one `response.create` message.
///
/// Any requests received from the client while the stream is active are
/// pushed onto `queue` and processed by the caller in order after this returns.
async fn handle_ws_text(
    sender: &mut WsSender,
    receiver: &mut WsReceiver,
    state: &AppState,
    headers: &HeaderMap,
    text: &str,
    shutdown_token: &CancellationToken,
    queue: &mut VecDeque<String>,
) -> Result<(), WsError> {
    let value = serde_json::from_str::<Value>(text).map_err(WsError::InvalidJson)?;

    if value.get("type").and_then(Value::as_str) != Some("response.create") {
        return Err(WsError::UnexpectedType);
    }

    let generate = value.get("generate").and_then(Value::as_bool);
    let mut payload = serde_json::from_value::<RequestPayload>(value).map_err(ExecutorError::from)?;
    let requested_stream = payload.stream;
    let requested_store = payload.store;
    payload.stream = true;
    payload.store = true;
    debug!(
        requested_stream,
        requested_store,
        forced_stream = payload.stream,
        forced_store = payload.store,
        has_previous_response_id = payload.previous_response_id.is_some(),
        has_conversation_id = payload.conversation_id.is_some(),
        ?generate,
        tools = payload.tools.as_ref().map_or(0, Vec::len),
        "accepted websocket response.create"
    );

    if generate == Some(false) {
        debug!("handling non-generating websocket request locally");
        return complete_without_inference(sender, state, payload).await;
    }

    let auth = extract_bearer(headers, state.openai_api_key.as_deref());
    let result = ExecuteRequest::new(payload, Arc::clone(&state.exec_ctx))
        .with_auth(auth)
        .run()
        .await?;
    let Some(result) = keep_if_running(shutdown_token, result) else {
        debug!("discarded websocket response initialized during shutdown");
        return Ok(());
    };
    let Either::Right(stream) = result else {
        return Err(WsError::Executor(Box::new(ExecutorError::InvalidRequest(
            "websocket response.create must produce a stream".to_owned(),
        ))));
    };

    stream_ws_response(sender, receiver, stream, shutdown_token, queue).await
}

async fn complete_without_inference(
    sender: &mut WsSender,
    state: &AppState,
    payload: RequestPayload,
) -> Result<(), WsError> {
    let ctx = rehydrate_conversation(payload, &state.exec_ctx).await?;
    let created_at = utcnow_str();
    let created_event = empty_response_event(&ctx, created_at, "response.created", "in_progress", 0, None);
    let completed_event = empty_response_event(
        &ctx,
        created_at,
        "response.completed",
        "completed",
        1,
        Some(ResponseUsage::default()),
    );

    #[cfg(debug_assertions)]
    state.websocket_tracker.pause_local_completion_after_rehydration().await;
    persist_turn(
        ctx,
        Vec::new(),
        &state.exec_ctx.conv_handler,
        &state.exec_ctx.resp_handler,
    )
    .await?;

    send_ws_json(sender, created_event).await?;
    send_ws_json(sender, completed_event).await
}

fn empty_response_event(
    ctx: &RequestContext,
    created_at: i64,
    event_type: &str,
    status: &str,
    sequence_number: u32,
    usage: Option<ResponseUsage>,
) -> Value {
    serde_json::json!({
        "type": event_type,
        "sequence_number": sequence_number,
        "response": {
            "id": &ctx.response_id,
            "object": "response",
            "created_at": created_at,
            "model": &ctx.enriched_request.model,
            "status": status,
            "output": [],
            "usage": usage,
            "incomplete_details": null,
            "error": null,
            "previous_response_id": &ctx.original_request.previous_response_id,
            "conversation_id": &ctx.conversation_id,
            "instructions": &ctx.enriched_request.instructions,
        },
    })
}

enum ShutdownInput<ReceiverItem, UpstreamItem> {
    Receiver(Option<ReceiverItem>),
    Upstream(Option<UpstreamItem>),
}

async fn next_shutdown_input<Receiver, Upstream>(
    receiver: &mut Receiver,
    upstream: &mut Upstream,
    prefer_receiver: bool,
) -> ShutdownInput<Receiver::Item, Upstream::Item>
where
    Receiver: Stream + Unpin,
    Upstream: Stream + Unpin,
{
    if prefer_receiver {
        tokio::select! {
            biased;
            message = receiver.next() => ShutdownInput::Receiver(message),
            line = upstream.next() => ShutdownInput::Upstream(line),
        }
    } else {
        tokio::select! {
            biased;
            line = upstream.next() => ShutdownInput::Upstream(line),
            message = receiver.next() => ShutdownInput::Receiver(message),
        }
    }
}

/// Stream a response from the executor to the client.
///
/// Requests arriving from the client while the stream is active are pushed
/// onto `queue` so the caller can process them in order after this returns.
async fn stream_ws_response(
    sender: &mut WsSender,
    receiver: &mut WsReceiver,
    mut stream: BoxStream,
    shutdown_token: &CancellationToken,
    queue: &mut VecDeque<String>,
) -> Result<(), WsError> {
    let mut prefer_shutdown_receiver = true;
    'stream: loop {
        if shutdown_token.is_cancelled() {
            match next_shutdown_input(receiver, &mut stream, prefer_shutdown_receiver).await {
                ShutdownInput::Receiver(message) => {
                    prefer_shutdown_receiver = false;
                    match message {
                        None | Some(Ok(Message::Close(_))) => return Err(WsError::ClientDisconnected),
                        Some(Ok(Message::Ping(payload))) => {
                            sender
                                .send(Message::Pong(payload))
                                .await
                                .map_err(|_| WsError::SendFailed)?;
                        }
                        Some(Ok(Message::Text(_) | Message::Binary(_) | Message::Pong(_))) => {}
                        Some(Err(error)) => return Err(WsError::Receive(error.to_string())),
                    }
                    continue 'stream;
                }
                ShutdownInput::Upstream(line) => {
                    prefer_shutdown_receiver = true;
                    let Some(line) = line else {
                        break;
                    };
                    forward_ws_stream_chunk(sender, &line).await?;
                }
            }
            continue;
        }

        let next_line = tokio::select! {
            () = shutdown_token.cancelled() => continue 'stream,
            message = receiver.next() => {
                match message {
                    None | Some(Ok(Message::Close(_))) => return Err(WsError::ClientDisconnected),
                    Some(Ok(Message::Ping(payload))) => {
                        sender.send(Message::Pong(payload)).await.map_err(|_| WsError::SendFailed)?;
                        continue 'stream;
                    }
                    Some(Ok(Message::Pong(_))) => continue 'stream,
                    Some(Ok(Message::Binary(_))) => return Err(WsError::BinaryFrame),
                    Some(Ok(Message::Text(text))) => {
                        // Client pipelined the next request while we are still streaming.
                        // Enqueue it and keep draining the current stream.
                        queue.push_back(text.to_string());
                        debug!(
                            queued_requests = queue.len(),
                            "queued pipelined websocket response.create while stream is active"
                        );
                        continue 'stream;
                    }
                    Some(Err(e)) => return Err(WsError::Receive(e.to_string())),
                }
            }
            line = stream.next() => line,
        };
        let Some(line) = next_line else {
            break;
        };
        forward_ws_stream_chunk(sender, &line).await?;
    }

    Ok(())
}

fn sse_json_data_lines(chunk: &str) -> impl Iterator<Item = &str> {
    chunk
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(str::trim)
        .filter(|data| *data != "[DONE]")
}

async fn forward_ws_stream_chunk(sender: &mut WsSender, chunk: &str) -> Result<(), WsError> {
    for data in sse_json_data_lines(chunk) {
        let value = serde_json::from_str::<Value>(data)
            .map_err(ExecutorError::from)
            .map_err(WsError::from)?;
        send_ws_json(sender, value).await?;
    }
    Ok(())
}

async fn handle_ws_error(sender: &mut WsSender, err: WsError) -> bool {
    match err {
        WsError::ClientDisconnected | WsError::SendFailed => false,
        WsError::Receive(message) => {
            warn!("responses websocket receive error: {message}");
            false
        }
        err => send_ws_error(sender, &err).await.is_ok(),
    }
}

async fn send_ws_error(sender: &mut WsSender, err: &WsError) -> Result<(), WsError> {
    let Some(frame) = err.to_ws_frame() else {
        return Err(WsError::SendFailed);
    };
    send_ws_json(sender, frame).await
}

async fn send_ws_json(sender: &mut WsSender, value: Value) -> Result<(), WsError> {
    let text = serde_json::to_string(&value).map_err(WsError::SerializeJson)?;
    sender
        .send(Message::Text(text.into()))
        .await
        .map_err(|_| WsError::SendFailed)
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use axum::extract::ws::Message;
    use futures::{Sink, Stream, StreamExt, sink, stream};
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use super::{
        ShutdownInput, WsError, close_ws, keep_if_running, next_shutdown_input, next_ws_message, sse_json_data_lines,
        websocket_identity_error_event,
    };
    use crate::auth::AuthenticatedPrincipal;

    struct CloseErrorSink;

    struct CancellingStream {
        shutdown_token: CancellationToken,
        item: Option<&'static str>,
    }

    #[test]
    fn sse_json_data_lines_accept_named_and_data_only_frames() {
        let chunk = concat!(
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\"}\n\n",
            "data: [DONE]\n\n",
        );

        assert_eq!(
            sse_json_data_lines(chunk).collect::<Vec<_>>(),
            [r#"{"type":"response.completed"}"#]
        );
    }

    impl Stream for CancellingStream {
        type Item = &'static str;

        fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            self.shutdown_token.cancel();
            Poll::Ready(self.item.take())
        }
    }

    impl Sink<Message> for CloseErrorSink {
        type Error = &'static str;

        fn poll_ready(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn start_send(self: Pin<&mut Self>, _item: Message) -> Result<(), Self::Error> {
            Ok(())
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Err("close failed"))
        }
    }

    #[tokio::test]
    async fn cancelled_shutdown_wins_over_ready_websocket_message() {
        let shutdown_token = CancellationToken::new();
        shutdown_token.cancel();
        let mut receiver = stream::iter(["must remain unread"]);

        assert!(next_ws_message(&shutdown_token, &mut receiver).await.is_none());
        assert_eq!(receiver.next().await, Some("must remain unread"));
    }

    #[tokio::test]
    async fn cancellation_during_receive_discards_websocket_message() {
        let shutdown_token = CancellationToken::new();
        let mut receiver = CancellingStream {
            shutdown_token: shutdown_token.clone(),
            item: Some("must be discarded"),
        };

        assert!(next_ws_message(&shutdown_token, &mut receiver).await.is_none());
        assert!(shutdown_token.is_cancelled());
        assert_eq!(receiver.next().await, None);
    }

    #[test]
    fn cancellation_after_request_setup_discards_unpolled_stream() {
        let shutdown_token = CancellationToken::new();
        shutdown_token.cancel();

        assert_eq!(keep_if_running(&shutdown_token, "unpolled stream"), None);
    }

    #[test]
    fn websocket_identity_expiry_uses_responses_error_event() {
        assert!(websocket_identity_error_event(None).is_none());
        let frame = websocket_identity_error_event(Some(&AuthenticatedPrincipal::expired_for_test()))
            .expect("expired-token error event");

        assert_eq!(
            frame,
            json!({
                "type": "error",
                "code": "invalid_token",
                "message": "OIDC bearer token expired",
                "param": null,
                "sequence_number": 0,
            })
        );

        let generic_frame = WsError::UnexpectedType
            .to_ws_frame()
            .expect("generic client-visible error frame");
        assert_eq!(generic_frame["status"], 400);
        assert_eq!(generic_frame["error"]["code"], "invalid_request_error");
    }

    #[tokio::test]
    async fn close_ws_ignores_late_frames_until_peer_close() {
        let mut sender = sink::drain();
        let mut receiver = stream::iter([
            Ok::<_, &'static str>(Message::Text("late request".into())),
            Ok(Message::Binary(vec![1].into())),
            Ok(Message::Close(None)),
            Err("must remain unread"),
        ]);

        close_ws(&mut sender, &mut receiver).await;

        assert!(matches!(receiver.next().await, Some(Err("must remain unread"))));
    }

    #[tokio::test]
    async fn close_ws_returns_without_reading_when_close_send_fails() {
        let mut sender = CloseErrorSink;
        let mut receiver = stream::iter([Ok::<_, &'static str>(Message::Close(None))]);

        close_ws(&mut sender, &mut receiver).await;

        assert!(matches!(receiver.next().await, Some(Ok(Message::Close(None)))));
    }

    #[tokio::test]
    async fn close_ws_stops_reading_after_receive_error() {
        let mut sender = sink::drain();
        let mut receiver = stream::iter([Err::<Message, _>("receive failed"), Ok(Message::Close(None))]);

        close_ws(&mut sender, &mut receiver).await;

        assert!(matches!(receiver.next().await, Some(Ok(Message::Close(None)))));
    }

    #[tokio::test]
    async fn shutdown_input_priority_alternates_when_both_streams_are_ready() {
        let mut receiver = stream::repeat(());
        let mut upstream = stream::repeat(());

        assert!(matches!(
            next_shutdown_input(&mut receiver, &mut upstream, true).await,
            ShutdownInput::Receiver(Some(()))
        ));
        assert!(matches!(
            next_shutdown_input(&mut receiver, &mut upstream, false).await,
            ShutdownInput::Upstream(Some(()))
        ));
    }
}
