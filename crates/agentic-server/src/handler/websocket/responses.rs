use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Extension, State};
use axum::http::HeaderMap;
use axum::response::Response;
use either::Either;
use futures::stream::SplitSink;
use futures::{Sink, SinkExt, Stream, StreamExt};
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
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

const WS_OUTBOUND_BUFFER: usize = 64;
const WS_MAX_EVENT_BYTES: usize = 1024 * 1024;
const WS_MAX_OUTSTANDING_REQUESTS: usize = 64;
const WS_MAX_OUTSTANDING_BYTES: usize = 12 * 1024 * 1024;
const WS_MAX_STREAM_ID_CHARS: usize = 256;

/// Serialized and size-checked before entering the bounded outbound queue.
struct WsOutboundEvent(String);

impl WsOutboundEvent {
    fn new(value: Value, stream_id: Option<&StreamId>) -> Result<Self, WsError> {
        let value = attach_stream_id(value, stream_id)?;
        let text = serde_json::to_string(&value).map_err(WsError::SerializeJson)?;
        if text.len() > WS_MAX_EVENT_BYTES {
            return Err(WsError::from(ExecutorError::StreamError(format!(
                "websocket event exceeded {WS_MAX_EVENT_BYTES} bytes"
            ))));
        }
        Ok(Self(text))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(try_from = "String")]
struct StreamId(String);

impl StreamId {
    fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for StreamId {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        let character_count = value.chars().count();
        if (1..=WS_MAX_STREAM_ID_CHARS).contains(&character_count) {
            Ok(Self(value))
        } else {
            Err(format!(
                "stream_id must contain between 1 and {WS_MAX_STREAM_ID_CHARS} characters"
            ))
        }
    }
}

impl TryFrom<&str> for StreamId {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::try_from(value.to_owned())
    }
}

struct WsRequest {
    payload: RequestPayload,
    stream_id: Option<StreamId>,
    generate: Option<bool>,
}

#[derive(Debug)]
struct WsRequestParseError {
    error: WsError,
    stream_id: Option<StreamId>,
}

enum WsWorkItem {
    Execute {
        request: Box<WsRequest>,
        input_bytes: usize,
    },
    Reject {
        error: WsRequestParseError,
        input_bytes: usize,
    },
}

impl WsWorkItem {
    fn stream_id(&self) -> Option<&StreamId> {
        match self {
            Self::Execute { request, .. } => request.stream_id.as_ref(),
            Self::Reject { error, .. } => error.stream_id.as_ref(),
        }
    }

    fn lane(&self) -> Option<StreamId> {
        self.stream_id().cloned()
    }

    fn input_bytes(&self) -> usize {
        match self {
            Self::Execute { input_bytes, .. } | Self::Reject { input_bytes, .. } => *input_bytes,
        }
    }
}

struct WsAdmissionError {
    stream_id: Option<StreamId>,
}

struct RequestCompletion {
    lane: Option<StreamId>,
    input_bytes: usize,
    result: Result<(), WsError>,
}

#[derive(Default)]
struct WsByteBudget {
    used: usize,
}

impl WsByteBudget {
    fn can_reserve(&self, input_bytes: usize) -> bool {
        input_bytes <= WS_MAX_OUTSTANDING_BYTES.saturating_sub(self.used)
    }

    fn reserve(&mut self, input_bytes: usize) {
        debug_assert!(self.can_reserve(input_bytes));
        self.used += input_bytes;
    }

    fn release(&mut self, input_bytes: usize) {
        self.used = self.used.saturating_sub(input_bytes);
    }
}

struct WsMultiplexer {
    state: Arc<AppState>,
    auth: Option<String>,
    principal: Option<Arc<AuthenticatedPrincipal>>,
    outbound_tx: mpsc::Sender<WsOutboundEvent>,
    lanes: HashMap<Option<StreamId>, VecDeque<WsWorkItem>>,
    request_tasks: JoinSet<RequestCompletion>,
    queued_requests: usize,
    byte_budget: WsByteBudget,
    shutdown_token: CancellationToken,
}

impl WsMultiplexer {
    fn new(
        state: Arc<AppState>,
        auth: Option<String>,
        principal: Option<AuthenticatedPrincipal>,
        outbound_tx: mpsc::Sender<WsOutboundEvent>,
        shutdown_token: CancellationToken,
    ) -> Self {
        Self {
            state,
            auth,
            principal: principal.map(Arc::new),
            outbound_tx,
            lanes: HashMap::new(),
            request_tasks: JoinSet::new(),
            queued_requests: 0,
            byte_budget: WsByteBudget::default(),
            shutdown_token,
        }
    }

    fn has_capacity_for(&self, input_bytes: usize) -> bool {
        self.request_tasks.len() + self.queued_requests < WS_MAX_OUTSTANDING_REQUESTS
            && self.byte_budget.can_reserve(input_bytes)
    }

    fn schedule(&mut self, work: WsWorkItem) -> Result<(), WsAdmissionError> {
        if !self.has_capacity_for(work.input_bytes()) {
            return Err(WsAdmissionError {
                stream_id: work.stream_id().cloned(),
            });
        }

        self.byte_budget.reserve(work.input_bytes());
        let lane = work.lane();
        if let Some(queue) = self.lanes.get_mut(&lane) {
            queue.push_back(work);
            self.queued_requests += 1;
            debug!(stream_id = ?lane, queued_requests = queue.len(), "queued websocket request on active lane");
            return Ok(());
        }

        self.lanes.insert(lane.clone(), VecDeque::new());
        self.spawn(lane, work);
        Ok(())
    }

    fn schedule_next(&mut self, lane: Option<StreamId>) {
        let next = self.lanes.get_mut(&lane).and_then(VecDeque::pop_front);
        if let Some(work) = next {
            self.queued_requests -= 1;
            self.spawn(lane, work);
        } else {
            self.lanes.remove(&lane);
        }
    }

    fn discard_queued(&mut self) {
        let discarded_bytes = self
            .lanes
            .values()
            .flat_map(|queue| queue.iter())
            .map(WsWorkItem::input_bytes)
            .sum::<usize>();
        self.byte_budget.release(discarded_bytes);
        self.lanes.clear();
        self.queued_requests = 0;
    }

    fn finish(&mut self, completion: Result<RequestCompletion, tokio::task::JoinError>, draining: bool) -> bool {
        let completion = match completion {
            Ok(completion) => completion,
            Err(error) => {
                warn!(%error, "responses websocket request task failed");
                return false;
            }
        };
        self.byte_budget.release(completion.input_bytes);
        if let Err(error) = completion.result {
            warn!(%error, "responses websocket request failed without a client-visible event");
            return false;
        }
        if !draining && !self.shutdown_token.is_cancelled() {
            self.schedule_next(completion.lane);
        }
        true
    }

    fn reap_ready(&mut self, draining: bool) -> bool {
        while let Some(completion) = self.request_tasks.try_join_next() {
            if !self.finish(completion, draining) {
                return false;
            }
        }
        true
    }

    fn spawn(&mut self, lane: Option<StreamId>, work: WsWorkItem) {
        let state = Arc::clone(&self.state);
        let auth = self.auth.clone();
        let principal = self.principal.clone();
        let outbound_tx = self.outbound_tx.clone();
        let shutdown_token = self.shutdown_token.clone();
        let stream_id = work.stream_id().cloned();
        let input_bytes = work.input_bytes();
        self.request_tasks.spawn(async move {
            // Admission may precede dispatch by an entire inference/tool round.
            // Recheck here for both new lanes and work dequeued by schedule_next.
            if let Some(event) = websocket_identity_error_event(principal.as_deref()) {
                return RequestCompletion {
                    lane,
                    input_bytes,
                    result: queue_ws_json(&outbound_tx, event, stream_id.as_ref()).await,
                };
            }
            let result = match work {
                WsWorkItem::Execute { request, .. } => {
                    handle_ws_request(*request, &state, auth, &outbound_tx, &shutdown_token).await
                }
                WsWorkItem::Reject { error, .. } => Err(error.error),
            };
            let result = match result {
                Ok(()) => Ok(()),
                Err(error) => queue_ws_error(&outbound_tx, error, stream_id.as_ref()).await,
            };
            RequestCompletion {
                lane,
                input_bytes,
                result,
            }
        });
    }
}

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

fn begin_ws_draining(multiplexer: &mut WsMultiplexer, draining: &mut bool) {
    *draining = true;
    multiplexer.discard_queued();
    debug!(
        active_streams = multiplexer.request_tasks.len(),
        "draining responses websocket session"
    );
}

async fn responses_ws_loop(
    socket: WebSocket,
    state: AppState,
    headers: HeaderMap,
    principal: Option<AuthenticatedPrincipal>,
) {
    debug!("responses websocket session opened");
    let shutdown_token = state.shutdown_token.clone();
    let state = Arc::new(state);
    let (mut sender, mut receiver) = socket.split();
    let auth = extract_bearer(&headers, state.openai_api_key.as_deref());
    let (outbound_tx, mut outbound_rx) = mpsc::channel(WS_OUTBOUND_BUFFER);
    let mut multiplexer = WsMultiplexer::new(state, auth, principal, outbound_tx, shutdown_token.clone());
    let mut draining = false;
    let mut client_disconnected = false;

    loop {
        if shutdown_token.is_cancelled() && !draining {
            begin_ws_draining(&mut multiplexer, &mut draining);
        }
        if draining && multiplexer.request_tasks.is_empty() && outbound_rx.is_empty() {
            break;
        }

        tokio::select! {
            () = shutdown_token.cancelled(), if !draining => {
                begin_ws_draining(&mut multiplexer, &mut draining);
            }
            outbound = outbound_rx.recv() => {
                let Some(value) = outbound else {
                    continue;
                };
                if send_ws_event(&mut sender, value).await.is_err() {
                    client_disconnected = true;
                    break;
                }
            }
            completion = multiplexer.request_tasks.join_next(), if !multiplexer.request_tasks.is_empty() => {
                let Some(completion) = completion else {
                    continue;
                };
                if !multiplexer.finish(completion, draining) {
                    client_disconnected = true;
                    break;
                }
                if shutdown_token.is_cancelled() && !draining {
                    begin_ws_draining(&mut multiplexer, &mut draining);
                }
            }
            message = receiver.next() => {
                if shutdown_token.is_cancelled() && !draining {
                    begin_ws_draining(&mut multiplexer, &mut draining);
                    debug!("discarded websocket message received during shutdown");
                    continue;
                }
                let Some(message) = message else {
                    client_disconnected = true;
                    break;
                };
                match message {
                    Ok(message) => {
                        if !handle_ws_client_message(
                            message,
                            &mut sender,
                            &mut multiplexer,
                            &mut draining,
                        )
                        .await
                        {
                            client_disconnected = true;
                            break;
                        }
                    }
                    Err(error) => {
                        warn!(%error, "responses websocket receive error");
                        client_disconnected = true;
                        break;
                    }
                }
            }
        }
    }

    if client_disconnected {
        multiplexer.request_tasks.abort_all();
        while multiplexer.request_tasks.join_next().await.is_some() {}
    }
    drop(multiplexer.outbound_tx);
    close_ws(&mut sender, &mut receiver).await;
    debug!("responses websocket session closed");
}

async fn handle_ws_client_message(
    message: Message,
    sender: &mut WsSender,
    multiplexer: &mut WsMultiplexer,
    draining: &mut bool,
) -> bool {
    match message {
        Message::Text(_) if *draining => {
            debug!("discarded websocket response.create during shutdown");
            true
        }
        Message::Text(text) => {
            if let Some(event) = websocket_identity_error_event(multiplexer.principal.as_deref()) {
                let stream_id = stream_id_from_text(&text);
                let send_succeeded = match WsOutboundEvent::new(event, stream_id.as_ref()) {
                    Ok(event) => send_ws_event(sender, event).await.is_ok(),
                    Err(error) => {
                        warn!(%error, "failed to build websocket identity error event");
                        false
                    }
                };
                *draining = true;
                multiplexer.discard_queued();
                return send_succeeded;
            }

            if !multiplexer.reap_ready(*draining) {
                return false;
            }
            if multiplexer.shutdown_token.is_cancelled() {
                begin_ws_draining(multiplexer, draining);
                debug!("discarded websocket response.create during shutdown");
                return true;
            }
            let input_bytes = text.len();
            if !multiplexer.has_capacity_for(input_bytes) {
                let stream_id = stream_id_from_text(&text);
                return handle_ws_error(sender, WsError::TooManyRequests, stream_id.as_ref()).await;
            }
            let work = match parse_ws_request(&text) {
                Ok(request) => WsWorkItem::Execute {
                    request: Box::new(request),
                    input_bytes,
                },
                Err(error) => WsWorkItem::Reject { error, input_bytes },
            };
            if multiplexer.shutdown_token.is_cancelled() {
                begin_ws_draining(multiplexer, draining);
                debug!("discarded websocket response.create during shutdown");
                return true;
            }
            match multiplexer.schedule(work) {
                Ok(()) => true,
                Err(rejected) => handle_ws_error(sender, WsError::TooManyRequests, rejected.stream_id.as_ref()).await,
            }
        }
        Message::Binary(_) if *draining => true,
        Message::Binary(_) => handle_ws_error(sender, WsError::BinaryFrame, None).await,
        Message::Close(_) => false,
        Message::Ping(payload) => sender.send(Message::Pong(payload)).await.is_ok(),
        Message::Pong(_) => true,
    }
}

fn stream_id_from_text(text: &str) -> Option<StreamId> {
    #[derive(Deserialize)]
    struct StreamIdEnvelope {
        stream_id: Option<StreamId>,
    }

    serde_json::from_str::<StreamIdEnvelope>(text).ok()?.stream_id
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

fn parse_ws_request(text: &str) -> Result<WsRequest, WsRequestParseError> {
    let value = serde_json::from_str::<Value>(text).map_err(|error| WsRequestParseError {
        error: WsError::InvalidJson(error),
        stream_id: None,
    })?;
    let stream_id = value
        .get("stream_id")
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| "stream_id must be a string".to_owned())
                .and_then(StreamId::try_from)
        })
        .transpose()
        .map_err(|error| WsRequestParseError {
            error: WsError::from(ExecutorError::InvalidRequest(error)),
            stream_id: None,
        })?;

    if value.get("type").and_then(Value::as_str) != Some("response.create") {
        return Err(WsRequestParseError {
            error: WsError::UnexpectedType,
            stream_id,
        });
    }

    let generate = value.get("generate").and_then(Value::as_bool);
    let mut payload = serde_json::from_value::<RequestPayload>(value).map_err(|error| WsRequestParseError {
        error: WsError::from(ExecutorError::from(error)),
        stream_id: stream_id.clone(),
    })?;
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
        stream_id = stream_id.as_ref().map(StreamId::as_str),
        ?generate,
        tools = payload.tools.as_ref().map_or(0, Vec::len),
        "accepted websocket response.create"
    );

    Ok(WsRequest {
        payload,
        stream_id,
        generate,
    })
}

async fn handle_ws_request(
    request: WsRequest,
    state: &AppState,
    auth: Option<String>,
    outbound_tx: &mpsc::Sender<WsOutboundEvent>,
    shutdown_token: &CancellationToken,
) -> Result<(), WsError> {
    let WsRequest {
        payload,
        stream_id,
        generate,
    } = request;

    if generate == Some(false) {
        debug!("handling non-generating websocket request locally");
        return complete_without_inference(outbound_tx, state, payload, stream_id.as_ref()).await;
    }

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

    stream_ws_response(outbound_tx, stream, stream_id.as_ref()).await
}

async fn complete_without_inference(
    outbound_tx: &mpsc::Sender<WsOutboundEvent>,
    state: &AppState,
    payload: RequestPayload,
    stream_id: Option<&StreamId>,
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

    // Validate both lifecycle events, including routing metadata, before any
    // persistence or delivery. Completion metadata can exceed the limit even
    // when the created event fits.
    let created_event = WsOutboundEvent::new(created_event, stream_id)?;
    let completed_event = WsOutboundEvent::new(completed_event, stream_id)?;

    #[cfg(debug_assertions)]
    state.websocket_tracker.pause_local_completion_after_rehydration().await;
    persist_turn(
        ctx,
        Vec::new(),
        &state.exec_ctx.conv_handler,
        &state.exec_ctx.resp_handler,
    )
    .await?;

    outbound_tx.send(created_event).await.map_err(|_| WsError::SendFailed)?;
    outbound_tx.send(completed_event).await.map_err(|_| WsError::SendFailed)
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

async fn stream_ws_response(
    outbound_tx: &mpsc::Sender<WsOutboundEvent>,
    mut stream: BoxStream,
    stream_id: Option<&StreamId>,
) -> Result<(), WsError> {
    while let Some(line) = stream.next().await {
        forward_ws_stream_chunk(outbound_tx, &line, stream_id).await?;
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

async fn forward_ws_stream_chunk(
    outbound_tx: &mpsc::Sender<WsOutboundEvent>,
    chunk: &str,
    stream_id: Option<&StreamId>,
) -> Result<(), WsError> {
    for data in sse_json_data_lines(chunk) {
        let value = serde_json::from_str::<Value>(data)
            .map_err(ExecutorError::from)
            .map_err(WsError::from)?;
        queue_ws_json(outbound_tx, value, stream_id).await?;
    }
    Ok(())
}

fn attach_stream_id(mut value: Value, stream_id: Option<&StreamId>) -> Result<Value, WsError> {
    let event = value.as_object_mut().ok_or_else(|| {
        WsError::from(ExecutorError::StreamError(
            "upstream WebSocket event must be a JSON object".to_owned(),
        ))
    })?;
    if let Some(stream_id) = stream_id {
        event.insert("stream_id".to_owned(), Value::String(stream_id.as_str().to_owned()));
    } else {
        event.remove("stream_id");
    }
    Ok(value)
}

async fn queue_ws_json(
    outbound_tx: &mpsc::Sender<WsOutboundEvent>,
    value: Value,
    stream_id: Option<&StreamId>,
) -> Result<(), WsError> {
    outbound_tx
        .send(WsOutboundEvent::new(value, stream_id)?)
        .await
        .map_err(|_| WsError::SendFailed)
}

async fn queue_ws_error(
    outbound_tx: &mpsc::Sender<WsOutboundEvent>,
    err: WsError,
    stream_id: Option<&StreamId>,
) -> Result<(), WsError> {
    let Some(frame) = err.to_ws_frame() else {
        return Err(err);
    };
    queue_ws_json(outbound_tx, frame, stream_id).await
}

async fn handle_ws_error(sender: &mut WsSender, err: WsError, stream_id: Option<&StreamId>) -> bool {
    match err {
        WsError::SendFailed => false,
        err => send_ws_error(sender, &err, stream_id).await.is_ok(),
    }
}

async fn send_ws_error(sender: &mut WsSender, err: &WsError, stream_id: Option<&StreamId>) -> Result<(), WsError> {
    let Some(frame) = err.to_ws_frame() else {
        return Err(WsError::SendFailed);
    };
    send_ws_event(sender, WsOutboundEvent::new(frame, stream_id)?).await
}

async fn send_ws_event(sender: &mut WsSender, event: WsOutboundEvent) -> Result<(), WsError> {
    sender
        .send(Message::Text(event.0.into()))
        .await
        .map_err(|_| WsError::SendFailed)
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use axum::extract::ws::Message;
    use futures::{Sink, StreamExt, sink, stream};
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use super::{
        StreamId, WS_MAX_OUTSTANDING_BYTES, WS_MAX_OUTSTANDING_REQUESTS, WS_MAX_STREAM_ID_CHARS, WsByteBudget, WsError,
        attach_stream_id, close_ws, forward_ws_stream_chunk, keep_if_running, parse_ws_request, queue_ws_json,
        sse_json_data_lines, websocket_identity_error_event,
    };
    use crate::auth::AuthenticatedPrincipal;

    struct CloseErrorSink;

    #[tokio::test]
    async fn outbound_event_limit_counts_routing_metadata_and_json_escaping() {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
        // The JSON envelope {"text":"","type":"test"} occupies 25 bytes.
        let event = json!({"text": "x".repeat(1024 * 1024 - 25), "type": "test"});
        queue_ws_json(&sender, event.clone(), None)
            .await
            .expect("exact limit is accepted");
        assert_eq!(receiver.recv().await.unwrap().0.len(), 1024 * 1024);

        let stream_id = StreamId::try_from("🦀").unwrap();
        let chunk = format!("data: {event}\n\n");
        assert!(
            forward_ws_stream_chunk(&sender, &chunk, Some(&stream_id))
                .await
                .is_err()
        );
        assert!(
            receiver.try_recv().is_err(),
            "routing metadata must count toward the limit"
        );

        let escaped = json!({"type": "test", "text": "\n".repeat(512 * 1024)});
        assert!(queue_ws_json(&sender, escaped, None).await.is_err());
        assert!(
            receiver.try_recv().is_err(),
            "escaped JSON bytes must count toward the limit"
        );
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

    #[test]
    fn stream_id_must_contain_between_one_and_256_characters() {
        for invalid in [String::new(), "a".repeat(WS_MAX_STREAM_ID_CHARS + 1)] {
            let request = json!({
                "type": "response.create",
                "stream_id": invalid,
                "model": "test-model",
                "input": "hello"
            });
            assert!(parse_ws_request(&request.to_string()).is_err());
        }
        let null_request = json!({
            "type": "response.create",
            "stream_id": null,
            "model": "test-model",
            "input": "hello"
        });
        assert!(parse_ws_request(&null_request.to_string()).is_err());

        let maximum = "🦀".repeat(WS_MAX_STREAM_ID_CHARS);
        let request = json!({
            "type": "response.create",
            "stream_id": maximum,
            "model": "test-model",
            "input": "hello"
        });
        let parsed = parse_ws_request(&request.to_string()).expect("256-character stream_id should be valid");
        assert_eq!(parsed.stream_id.expect("stream_id").as_str(), maximum);
    }

    #[test]
    fn websocket_capacity_is_bounded_by_count_and_input_bytes() {
        assert_eq!(WS_MAX_OUTSTANDING_REQUESTS, 64);
        let mut budget = WsByteBudget::default();
        budget.reserve(WS_MAX_OUTSTANDING_BYTES - 1);
        assert!(budget.can_reserve(1));
        budget.reserve(1);
        assert!(!budget.can_reserve(1));
        budget.release(WS_MAX_OUTSTANDING_BYTES);
        assert!(budget.can_reserve(WS_MAX_OUTSTANDING_BYTES));
    }

    #[test]
    fn stream_id_attachment_normalizes_routing_metadata() {
        let requested = StreamId::try_from("requested".to_owned()).expect("valid stream ID");
        let tagged = attach_stream_id(
            json!({"type": "response.created", "stream_id": "spoofed"}),
            Some(&requested),
        )
        .expect("object event");
        assert_eq!(tagged["stream_id"], "requested");

        let untagged =
            attach_stream_id(json!({"type": "response.created", "stream_id": "spoofed"}), None).expect("object event");
        assert!(untagged.get("stream_id").is_none());
        assert!(attach_stream_id(json!(["response.created"]), Some(&requested)).is_err());
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
}
