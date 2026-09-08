//! Streaming Messages-native gateway tool loop.
//!
//! Consumes vLLM's per-round Anthropic SSE and presents the client **one**
//! logical message across all gateway rounds:
//!   * `message_start` emitted once (first round only);
//!   * surfaced `content_block_*` forwarded with client-visible indices rebased
//!     contiguously across rounds;
//!   * gateway-owned `tool_use` blocks suppressed (and their `input_json_delta`
//!     buffered to reconstruct the call for dispatch);
//!   * intermediate `message_delta`/`message_stop` (the per-round terminals)
//!     suppressed; the final round's terminal is forwarded once.
//!
//! Structurally the Anthropic-native analogue of the Responses `GatewayStreamAccumulator`
//! (#119/#132); kept deliberately parallel for a future consolidation. Reuses
//! only the neutral tool layer via [`crate::types::messages::tool_seam`].

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use async_stream::stream;
use futures::StreamExt;
use serde_json::{Value, json};

use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::inference::{BoxStream, response_lines, send_request};
use crate::executor::messages_context::MessagesRequestContext;
use crate::executor::messages_request::web_search_budget_exhausted_result;
use crate::executor::request::ExecutionContext;
use crate::proxy::processed_response_headers;
use crate::tool::ToolRegistry;
use crate::types::messages::tool_seam;
use crate::utils::common::{deserialize_from_str, serialize_to_string};

// Shared with the non-streaming loop so the two Messages loops can't drift.
use crate::executor::messages_loop::{
    GATEWAY_TOOL_TIMEOUT, MAX_GATEWAY_TOOL_ROUNDS, MessagesResponse, MessagesUpstream,
};

/// Drive the streaming Messages-native loop, yielding Anthropic SSE lines for
/// the client. Owns the multi-round → single-message accumulation.
///
/// # Errors
///
/// Returns an executor error when the initial request cannot be serialized or
/// when the upstream rejects it before streaming begins.
pub async fn run_messages_stream(
    mut ctx: MessagesRequestContext,
    registry: Arc<ToolRegistry>,
    exec_ctx: Arc<ExecutionContext>,
    upstream: MessagesUpstream,
) -> ExecutorResult<MessagesResponse<BoxStream>> {
    ctx.force_stream(true);

    // Prime the first upstream request before the handler commits an HTTP 200.
    // This lets initial vLLM errors retain their original status and body.
    let first_body = ctx.upstream_body()?;
    let first_response = send_request(
        &exec_ctx.client,
        upstream.url(),
        first_body,
        None,
        Some(upstream.headers()),
    )
    .await?;
    let response_headers = processed_response_headers(first_response.headers());

    let body: BoxStream = Box::pin(stream! {
        let mut acc = MessagesStreamAccumulator::new(exec_ctx.messages_gateway_tools.clone());
        let mut prepared_response = Some(first_response);

        for _round in 0..MAX_GATEWAY_TOOL_ROUNDS {
            let response = if let Some(response) = prepared_response.take() {
                response
            } else {
                let body = match ctx.upstream_body() {
                    Ok(b) => b,
                    Err(e) => { yield executor_error_sse(&e); return; }
                };
                match send_request(
                    &exec_ctx.client,
                    upstream.url(),
                    body,
                    None,
                    Some(upstream.headers()),
                )
                .await
                {
                    Ok(response) => response,
                    Err(e) => { yield executor_error_sse(&e); return; }
                }
            };
            let mut response_stream = Box::pin(response_lines(response, exec_ctx.streaming_timeout));

            acc.begin_round();
            while let Some(line) = response_stream.next().await {
                let line = match line {
                    Ok(l) => l,
                    Err(e) => { yield error_sse(&e.to_string()); return; }
                };
                for out in acc.push(&line) {
                    yield out;
                }
                if acc.has_upstream_error() {
                    return;
                }
            }

            // Round finished. Continue only for a pure gateway-tool round; a
            // client-owned tool_use (or any non-tool_use stop) is terminal.
            if !acc.should_continue_loop() {
                for out in acc.finish() {
                    yield out;
                }
                return;
            }
            // Reconstruct the FULL assistant turn (thinking/text/signature +
            // gateway tool_use, in order) for the next round's history — not just
            // the gateway tool_use (F3, streaming half). The gateway calls are
            // derived from the same buffered blocks for dispatch.
            let (assistant_content, calls) = acc.take_round();
            let allowed_searches = ctx.reserve_searches(calls.len());
            let tool_results = execute_gateway_calls(
                &calls,
                &registry,
                &exec_ctx.messages_gateway_tools,
                allowed_searches,
            ).await;
            if let Err(e) = ctx.append_round(&assistant_content, tool_results) {
                yield executor_error_sse(&e);
                return;
            }
        }

        // Round budget exhausted.
        yield error_sse(&format!("gateway tool loop exceeded {MAX_GATEWAY_TOOL_ROUNDS} rounds"));
    });
    Ok(MessagesResponse {
        body,
        headers: response_headers,
    })
}

/// A gateway `tool_use` reconstructed from the stream, ready to dispatch.
struct StreamedCall {
    id: String,
    name: String,
    input_json: String,
}

/// One assistant content block buffered across a round, so the full turn
/// (`thinking`/`text`/`signature`/`tool_use`, in order) can be reconstructed for
/// the next round's history — F3. The client-facing SSE is still forwarded live;
/// this is a parallel record for the fed-back conversation state.
struct BufferedBlock {
    /// The `content_block` skeleton from `content_block_start`, mutated by deltas.
    block: Value,
    /// Accumulated `input_json_delta` fragments for a `tool_use` block.
    input_json: String,
    /// Gateway-owned `tool_use` (drives the loop; suppressed from the client).
    is_gateway_tool: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RoundState {
    Active,
    UpstreamError,
}

impl BufferedBlock {
    fn apply_delta(&mut self, delta: &Value) {
        match delta.get("type").and_then(Value::as_str) {
            Some("text_delta") => append_str(&mut self.block, "text", delta.get("text")),
            Some("thinking_delta") => append_str(&mut self.block, "thinking", delta.get("thinking")),
            Some("signature_delta") => append_str(&mut self.block, "signature", delta.get("signature")),
            Some("input_json_delta") => {
                if let Some(partial) = delta.get("partial_json").and_then(Value::as_str) {
                    self.input_json.push_str(partial);
                }
            }
            _ => {}
        }
    }

    /// The finished assistant content block. For `tool_use`, parse the
    /// accumulated arguments (best-effort — a malformed fragment falls back to
    /// `{}`; the paired error `tool_result` records the failure).
    fn to_block(&self) -> Value {
        let mut block = self.block.clone();
        if block.get("type").and_then(Value::as_str) == Some("tool_use") {
            block["input"] = tool_seam::parse_tool_input(&self.input_json).unwrap_or_else(|_| json!({}));
        }
        block
    }
}

/// Append a streamed string fragment onto a string field of `block`, creating it
/// if absent.
fn append_str(block: &mut Value, field: &str, fragment: Option<&Value>) {
    let Some(fragment) = fragment.and_then(Value::as_str) else {
        return;
    };
    let combined = match block.get(field).and_then(Value::as_str) {
        Some(existing) => format!("{existing}{fragment}"),
        None => fragment.to_owned(),
    };
    block[field] = Value::from(combined);
}

/// State machine that turns per-round Anthropic SSE into one client-visible
/// message. Fed line-by-line via [`Self::push`].
struct MessagesStreamAccumulator {
    message_started: bool,
    /// Next client-visible block index (contiguous across rounds).
    next_index: u32,
    /// Map upstream (per-round) block index → client index, for the blocks we
    /// forward this round. Cleared each round.
    index_map: HashMap<u64, u32>,
    /// Upstream indices belonging to a suppressed gateway `tool_use` this round.
    suppressed_indices: HashSet<u64>,
    /// Every assistant block this round, keyed by upstream index (ordered), so
    /// the full turn — `thinking`/`text`/`signature` + gateway `tool_use` — can
    /// be reconstructed for the next round's history (F3). Cleared each round.
    blocks: BTreeMap<u64, BufferedBlock>,
    /// Did this round end with `stop_reason: tool_use`?
    ended_on_tool_use: bool,
    /// Did this round surface a client-owned `tool_use`? If so the loop cannot
    /// continue server-side (the client must supply that tool's result), so it
    /// is terminal — matching the non-streaming path's E7 handling.
    has_client_tool_use: bool,
    /// Buffered terminal `message_delta` from the final round (emitted by `finish`).
    final_message_delta: Option<Value>,
    /// Whether this round is still active or terminated with an upstream error.
    round_state: RoundState,
    /// Operator-configured client-tool → gateway-executor aliases, so a client
    /// tool like Claude Code's `WebSearch` is classified gateway-owned (and
    /// suppressed) the same way the built-in `web_search` is.
    gateway_map: tool_seam::GatewayToolMap,
}

impl MessagesStreamAccumulator {
    fn new(gateway_map: tool_seam::GatewayToolMap) -> Self {
        Self {
            message_started: false,
            next_index: 0,
            index_map: HashMap::new(),
            suppressed_indices: HashSet::new(),
            blocks: BTreeMap::new(),
            ended_on_tool_use: false,
            has_client_tool_use: false,
            final_message_delta: None,
            round_state: RoundState::Active,
            gateway_map,
        }
    }

    fn begin_round(&mut self) {
        self.index_map.clear();
        self.suppressed_indices.clear();
        self.blocks.clear();
        self.ended_on_tool_use = false;
        self.has_client_tool_use = false;
        self.round_state = RoundState::Active;
        // F6: clear the previous round's terminal so a clean-EOF round can't
        // re-emit a stale stop_reason.
        self.final_message_delta = None;
    }

    /// Number of gateway `tool_use` blocks buffered this round.
    fn gateway_call_count(&self) -> usize {
        self.blocks.values().filter(|b| b.is_gateway_tool).count()
    }

    /// Consume this round's buffered blocks, returning (full assistant content in
    /// order, gateway calls to dispatch). The assistant content preserves
    /// `thinking`/`text`/`signature` and the gateway `tool_use` blocks (F3); the
    /// calls are the gateway `tool_use` blocks reconstructed for dispatch.
    fn take_round(&mut self) -> (Vec<Value>, Vec<StreamedCall>) {
        let blocks = std::mem::take(&mut self.blocks);
        let mut assistant_content = Vec::with_capacity(blocks.len());
        let mut calls = Vec::new();
        for buffered in blocks.values() {
            assistant_content.push(buffered.to_block());
            if buffered.is_gateway_tool {
                calls.push(StreamedCall {
                    id: buffered.block["id"].as_str().unwrap_or_default().to_owned(),
                    name: buffered.block["name"].as_str().unwrap_or_default().to_owned(),
                    input_json: buffered.input_json.clone(),
                });
            }
        }
        (assistant_content, calls)
    }

    /// The loop should continue only when the round asked for a gateway tool AND
    /// did not also surface a client-owned tool (which the client must handle,
    /// making the round terminal — E7).
    fn should_continue_loop(&self) -> bool {
        self.ended_on_tool_use && self.gateway_call_count() > 0 && !self.has_client_tool_use
    }

    fn has_upstream_error(&self) -> bool {
        self.round_state == RoundState::UpstreamError
    }

    /// Translate one upstream SSE line into zero or more client SSE lines.
    fn push(&mut self, line: &str) -> Vec<String> {
        let Some(data) = line.strip_prefix("data: ") else {
            return Vec::new();
        };
        let data = data.trim();
        if data == "[DONE]" {
            return Vec::new();
        }
        let Ok(mut event) = serde_json::from_str::<Value>(data) else {
            return Vec::new();
        };
        match event.get("type").and_then(Value::as_str) {
            Some("message_start") => self.on_message_start(&event),
            Some("content_block_start") => self.on_block_start(&mut event),
            Some("content_block_delta") => self.on_block_delta(&mut event),
            Some("content_block_stop") => self.on_block_stop(&mut event),
            Some("message_delta") => {
                // Buffer as the (possibly) final terminal; suppress mid-loop.
                self.ended_on_tool_use = event["delta"]["stop_reason"].as_str() == Some("tool_use");
                self.final_message_delta = Some(event);
                Vec::new()
            }
            Some("error") => {
                self.round_state = RoundState::UpstreamError;
                vec![sse("error", &event)]
            }
            // `message_stop` (per-round terminal) is suppressed; `finish` emits
            // the single client-visible terminal. Everything else is dropped.
            _ => Vec::new(),
        }
    }

    fn on_message_start(&mut self, event: &Value) -> Vec<String> {
        if self.message_started {
            return Vec::new();
        }
        self.message_started = true;
        vec![sse("message_start", event)]
    }

    fn on_block_start(&mut self, event: &mut Value) -> Vec<String> {
        let up_index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
        let block_type = event["content_block"]["type"].as_str().unwrap_or_default();
        let name = event["content_block"]["name"].as_str().unwrap_or_default();

        // Buffer every block for history reconstruction (F3), preserving order.
        let is_gateway_tool = block_type == "tool_use" && self.gateway_map.is_gateway_owned(name);
        self.blocks.insert(
            up_index,
            BufferedBlock {
                block: event["content_block"].clone(),
                input_json: String::new(),
                is_gateway_tool,
            },
        );

        if block_type == "tool_use" {
            if is_gateway_tool {
                // Suppress gateway-owned tool_use from the client; it stays in the
                // buffered history only and drives the loop.
                self.suppressed_indices.insert(up_index);
                return Vec::new();
            }
            // A client-owned tool_use: the client must execute it, so this round
            // is terminal (E7). Forward it (below) and stop the loop.
            self.has_client_tool_use = true;
        }

        // Forward with a rebased contiguous client index.
        let client_index = self.next_index;
        self.next_index += 1;
        self.index_map.insert(up_index, client_index);
        event["index"] = Value::from(client_index);
        vec![sse("content_block_start", event)]
    }

    fn on_block_delta(&mut self, event: &mut Value) -> Vec<String> {
        let up_index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
        // Accumulate the delta into the buffered block (for history — F3),
        // regardless of whether it is forwarded to the client.
        if let Some(buffered) = self.blocks.get_mut(&up_index) {
            buffered.apply_delta(&event["delta"]);
        }
        // A suppressed gateway tool_use is not forwarded to the client (its
        // input_json_delta was just buffered above).
        if self.suppressed_indices.contains(&up_index) {
            return Vec::new();
        }
        let Some(&client_index) = self.index_map.get(&up_index) else {
            return Vec::new();
        };
        event["index"] = Value::from(client_index);
        vec![sse("content_block_delta", event)]
    }

    fn on_block_stop(&mut self, event: &mut Value) -> Vec<String> {
        let up_index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
        if self.suppressed_indices.contains(&up_index) {
            return Vec::new();
        }
        let Some(&client_index) = self.index_map.get(&up_index) else {
            return Vec::new();
        };
        event["index"] = Value::from(client_index);
        vec![sse("content_block_stop", event)]
    }

    /// Emit the terminal `message_delta` + `message_stop` once, at loop end.
    fn finish(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(delta) = self.final_message_delta.take() {
            out.push(sse("message_delta", &delta));
        }
        out.push(sse("message_stop", &json!({"type": "message_stop"})));
        out
    }
}

fn sse(event: &str, value: &Value) -> String {
    let json = serialize_to_string(value).unwrap_or_default();
    format!("event: {event}\ndata: {json}\n\n")
}

fn error_sse(message: &str) -> String {
    let event = json!({"type": "error", "error": {"type": "api_error", "message": message}});
    let json = serialize_to_string(&event).unwrap_or_default();
    format!("event: error\ndata: {json}\n\n")
}

fn executor_error_sse(error: &ExecutorError) -> String {
    if let ExecutorError::LLMRequest { body, .. } = error
        && let Ok(value) = deserialize_from_str::<Value>(body)
        && value.get("type").and_then(Value::as_str) == Some("error")
    {
        let data = if body.contains(['\r', '\n']) {
            serialize_to_string(&value).unwrap_or_else(|_| body.clone())
        } else {
            body.clone()
        };
        return format!("event: error\ndata: {data}\n\n");
    }
    error_sse(&error.to_string())
}

/// Execute reconstructed gateway calls (concurrent, per-call timeout). Errors
/// become error `tool_result`s (E5).
///
/// Returns one `tool_result` block per call, fed back next round. (The assistant
/// turn — including each call's `tool_use` block — is reconstructed from the
/// accumulator's buffered blocks in [`MessagesStreamAccumulator::take_round`].)
async fn execute_gateway_calls(
    calls: &[StreamedCall],
    registry: &ToolRegistry,
    gateway_map: &tool_seam::GatewayToolMap,
    allowed_searches: usize,
) -> Vec<Value> {
    let futures = calls.iter().enumerate().map(|(index, c)| async move {
        if index >= allowed_searches {
            return web_search_budget_exhausted_result(&c.id);
        }
        // F4: reject a malformed/incomplete reconstructed input rather than
        // coercing to {} and dispatching the tool with args the model never sent.
        let (output, is_error) = match tool_seam::parse_tool_input(&c.input_json) {
            Ok(input) => {
                let call = tool_seam::tool_use_to_call(&c.id, &c.name, &input, gateway_map);
                match tokio::time::timeout(GATEWAY_TOOL_TIMEOUT, registry.dispatch(&call)).await {
                    Ok(Some(result)) => match result.output {
                        Ok(o) => (o.output, false),
                        Err(e) => (format!("tool execution failed: {e}"), true),
                    },
                    Ok(None) => (format!("no handler for tool '{}'", c.name), true),
                    Err(_) => (
                        format!("gateway tool '{}' timed out after {GATEWAY_TOOL_TIMEOUT:?}", c.name),
                        true,
                    ),
                }
            }
            Err(reason) => (format!("{reason}; tool was not run"), true),
        };
        tool_seam::tool_result_block(&c.id, &output, is_error)
    });
    futures::future::join_all(futures).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(v: &Value) -> String {
        format!("data: {v}")
    }

    /// Accumulator with the default gateway map (built-in `web_search` only).
    fn acc() -> MessagesStreamAccumulator {
        MessagesStreamAccumulator::new(tool_seam::GatewayToolMap::default())
    }

    // A single non-tool round: message_start forwarded once, blocks pass through
    // with contiguous indices, terminal emitted by finish().
    #[test]
    fn single_round_text_passes_through() {
        let mut acc = acc();
        acc.begin_round();
        let mut out = Vec::new();
        out.extend(acc.push(&line(&json!({"type": "message_start", "message": {"id": "m"}}))));
        out.extend(acc.push(&line(
            &json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        )));
        out.extend(acc.push(&line(
            &json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
        )));
        out.extend(acc.push(&line(&json!({"type": "content_block_stop", "index": 0}))));
        out.extend(acc.push(&line(
            &json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
        )));
        out.extend(acc.push(&line(&json!({"type": "message_stop"}))));
        assert!(!acc.should_continue_loop(), "text-only round is terminal");
        out.extend(acc.finish());
        let s = out.join("");
        assert_eq!(s.matches("event: message_start").count(), 1);
        assert_eq!(s.matches("event: message_stop").count(), 1);
        assert!(s.contains("text_delta"));
        assert!(s.contains("end_turn"));
    }

    // A gateway tool round: the tool_use block (start/delta/stop) is suppressed,
    // its input reconstructed, thinking/text forwarded, and no terminal leaks.
    #[test]
    fn gateway_tool_round_suppresses_tool_use_and_reconstructs_call() {
        let mut acc = acc();
        acc.begin_round();
        let mut out = Vec::new();
        out.extend(acc.push(&line(&json!({"type": "message_start", "message": {"id": "m"}}))));
        // thinking idx0 (forward)
        out.extend(acc.push(&line(
            &json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
        )));
        out.extend(acc.push(&line(&json!({"type": "content_block_stop", "index": 0}))));
        // gateway tool_use idx1 (suppress + reconstruct)
        out.extend(acc.push(&line(&json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tid", "name": "web_search", "input": {}}}))));
        out.extend(acc.push(&line(&json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"query\":"}}))));
        out.extend(acc.push(&line(&json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"rust\"}"}}))));
        out.extend(acc.push(&line(&json!({"type": "content_block_stop", "index": 1}))));
        out.extend(acc.push(&line(
            &json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}}),
        )));
        out.extend(acc.push(&line(&json!({"type": "message_stop"}))));

        let s = out.join("");
        assert!(acc.should_continue_loop(), "pure gateway-tool round continues the loop");
        assert!(!s.contains("tool_use"), "gateway tool_use must not surface: {s}");
        assert!(!s.contains("message_stop"), "intermediate terminal suppressed");
        assert!(s.contains("thinking"), "thinking forwarded");
        let (_assistant, calls) = acc.take_round();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].input_json, "{\"query\":\"rust\"}");
    }

    // Across two rounds, client-visible block indices stay contiguous (round 1
    // thinking=0, round 2 text=1) — no reset/collision.
    #[test]
    fn indices_are_contiguous_across_rounds() {
        let mut acc = acc();
        // round 1: thinking (idx0) + suppressed tool_use (idx1)
        acc.begin_round();
        acc.push(&line(&json!({"type": "message_start", "message": {"id": "m"}})));
        acc.push(&line(
            &json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}),
        ));
        acc.push(&line(&json!({"type": "content_block_stop", "index": 0})));
        acc.push(&line(&json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "name": "web_search", "id": "t"}})));
        acc.push(&line(&json!({"type": "content_block_stop", "index": 1})));
        // round 2: text (upstream idx0) must map to client idx1
        acc.begin_round();
        let out = acc.push(&line(
            &json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
        ));
        let started: Value =
            serde_json::from_str(out[0].lines().nth(1).unwrap().strip_prefix("data: ").unwrap()).unwrap();
        assert_eq!(started["index"], 1, "round-2 text rebased to contiguous client index 1");
    }

    // E7 (streaming): a round with a gateway tool_use AND a client-owned tool_use
    // is terminal — the loop must NOT continue (the client owns the second tool).
    // The client-owned tool_use is forwarded; the gateway one is suppressed.
    #[test]
    fn mixed_client_and_gateway_tool_use_stops_the_loop() {
        let mut acc = acc();
        acc.begin_round();
        acc.push(&line(&json!({"type": "message_start", "message": {"id": "m"}})));
        // gateway tool_use (idx0) — suppressed
        acc.push(&line(&json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "name": "web_search", "id": "g"}})));
        acc.push(&line(&json!({"type": "content_block_stop", "index": 0})));
        // client tool_use (idx1) — forwarded
        let out = acc.push(&line(&json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "name": "get_weather", "id": "c"}})));
        acc.push(&line(&json!({"type": "content_block_stop", "index": 1})));
        acc.push(&line(
            &json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}}),
        ));

        // Client tool_use surfaces; gateway one does not.
        let started: Value =
            serde_json::from_str(out[0].lines().nth(1).unwrap().strip_prefix("data: ").unwrap()).unwrap();
        assert_eq!(
            started["content_block"]["name"], "get_weather",
            "client tool_use forwarded"
        );
        // The loop must terminate despite a gateway call being present.
        assert!(
            !acc.should_continue_loop(),
            "mixed round is terminal — loop must not continue"
        );
    }

    // F6 (repro): begin_round() must reset final_message_delta. Round 1 ends on a
    // tool_use terminal; round 2 ends WITHOUT a message_delta (clean EOF). finish()
    // must NOT emit round 1's stale stop_reason: tool_use.
    #[test]
    fn repro_f6_begin_round_resets_stale_terminal() {
        let mut acc = acc();
        // Round 1: a gateway tool round → sets final_message_delta = tool_use terminal.
        acc.begin_round();
        acc.push(&line(&json!({"type": "message_start", "message": {"id": "m"}})));
        acc.push(&line(&json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "name": "web_search", "id": "t"}})));
        acc.push(&line(&json!({"type": "content_block_stop", "index": 0})));
        acc.push(&line(
            &json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}}),
        ));
        // Round 2: text, but upstream ends with NO message_delta (cut short).
        acc.begin_round();
        acc.push(&line(
            &json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
        ));
        acc.push(&line(
            &json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
        ));
        acc.push(&line(&json!({"type": "content_block_stop", "index": 0})));
        let out = acc.finish().join("");
        assert!(
            !out.contains(r#""stop_reason":"tool_use""#),
            "must not emit round 1's stale tool_use terminal: {out}"
        );
    }

    // F3 (repro): the assistant turn fed into the next round's history must
    // preserve the model's thinking/text/signature blocks in order, not just the
    // gateway tool_use. (This is the streaming half of Maral's F3 — "also repeated
    // in messages_stream.rs".)
    #[test]
    fn repro_f3_stream_history_preserves_thinking_text_and_signature() {
        let mut acc = acc();
        acc.begin_round();
        acc.push(&line(&json!({"type": "message_start", "message": {"id": "m"}})));
        // thinking idx0 (with a signature delta)
        acc.push(&line(
            &json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
        ));
        acc.push(&line(&json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "let me search"}})));
        acc.push(&line(&json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "SIG=="}})));
        acc.push(&line(&json!({"type": "content_block_stop", "index": 0})));
        // text idx1
        acc.push(&line(
            &json!({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
        ));
        acc.push(&line(&json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Searching..."}})));
        acc.push(&line(&json!({"type": "content_block_stop", "index": 1})));
        // gateway tool_use idx2 (suppressed from client, but must appear in history)
        acc.push(&line(&json!({"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "tid", "name": "web_search", "input": {}}})));
        acc.push(&line(&json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"query\":\"rust\"}"}})));
        acc.push(&line(&json!({"type": "content_block_stop", "index": 2})));
        acc.push(&line(
            &json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}}),
        ));

        let (assistant, _calls) = acc.take_round();
        let types: Vec<&str> = assistant.iter().filter_map(|b| b["type"].as_str()).collect();
        assert_eq!(
            types,
            vec!["thinking", "text", "tool_use"],
            "full assistant turn preserved in order, not just the gateway tool_use: {assistant:?}"
        );
        assert_eq!(assistant[0]["thinking"], "let me search", "thinking text reconstructed");
        assert_eq!(
            assistant[0]["signature"], "SIG==",
            "signature preserved for the next round"
        );
        assert_eq!(assistant[1]["text"], "Searching...", "text reconstructed");
        assert_eq!(
            assistant[2]["input"]["query"], "rust",
            "gateway call input reconstructed"
        );
    }

    // F4 (repro): a malformed/incomplete input_json for a gateway call must NOT
    // silently become `{}` and dispatch the tool with args the model never sent.
    #[tokio::test]
    async fn repro_f4_malformed_partial_json_is_not_dispatched_with_empty_args() {
        let mut acc = acc();
        acc.begin_round();
        acc.push(&line(&json!({"type": "message_start", "message": {"id": "m"}})));
        acc.push(&line(&json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "name": "web_search", "id": "t"}})));
        // Incomplete partial_json (stream cut mid-arguments).
        acc.push(&line(&json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"query\":"}})));
        let (_assistant, calls) = acc.take_round();
        assert_eq!(calls.len(), 1);
        // The reconstructed input is invalid JSON.
        assert!(
            serde_json::from_str::<serde_json::Value>(&calls[0].input_json).is_err(),
            "incomplete partial_json is invalid JSON"
        );
        // After the fix, execute_gateway_calls must NOT coerce invalid input to
        // {} and dispatch — it must produce an error tool_result. Assert the
        // reconstructed call is flagged invalid rather than silently dispatchable.
        let resolved = execute_gateway_calls(
            &calls,
            &no_op_registry().await,
            &tool_seam::GatewayToolMap::default(),
            calls.len(),
        )
        .await;
        let content = resolved[0]["content"].as_str().unwrap_or_default();
        assert!(
            content.contains("invalid") || content.contains("malformed") || content.contains("could not"),
            "malformed args must yield an error tool_result, not an empty-arg dispatch: {content:?}"
        );
    }

    /// Registry with no gateway executors — dispatch of any call returns None, so
    /// the ONLY way `execute_gateway_calls` can produce a non-"no handler" result
    /// for a malformed input is by rejecting the args before dispatch (the fix).
    async fn no_op_registry() -> ToolRegistry {
        let mut tools = [];
        let mut executors = crate::tool::GatewayExecutors::from_env(std::sync::Arc::new(reqwest::Client::new()));
        ToolRegistry::build_with_handlers(&mut tools, &mut executors)
            .await
            .unwrap()
    }
}
