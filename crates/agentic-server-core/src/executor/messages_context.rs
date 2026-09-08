//! Per-request context for the Anthropic Messages gateway tool loops.
//!
//! The Messages loops are a pass-through, not a transform: the client's request
//! is forwarded to vLLM `/v1/messages` essentially untouched, and only `tools`
//! and `stream`/`messages` are read or rewritten. That rules out a typed
//! round-trip through [`MessagesRequest`] as the upstream body — `ContentBlock`
//! carries a `#[serde(other)] Unknown` catch-all, and several block types model
//! only the fields the gateway reads, so re-serializing would silently drop
//! `cache_control`, `is_error`, and every unmodeled block (`image`,
//! `redacted_thinking`, future provider extensions).
//!
//! So this context carries **two views of one request**, built once per request:
//!
//! * `raw` — the JSON body actually sent upstream. It
//!   is the single source of truth for `messages` and `system`, and the only
//!   thing the loops mutate.
//! * `typed` — the client's request as received, for safe field access to
//!   `tools`, `stream`, and `model` in routing and the loops.
//!
//! The two are **not** kept byte-identical, and must not be confused: `typed` is
//! what the client sent, `raw` is what the gateway sends upstream. They diverge
//! wherever the gateway rewrites the body for upstream — today
//! `normalize_native_web_search` rewriting a native `web_search_20250305`
//! declaration into the ordinary function-tool shape vLLM accepts. Accordingly,
//! only the fields the loops never mutate are exposed off `typed`
//! ([`tools`](MessagesRequestContext::tools),
//! [`stream`](MessagesRequestContext::stream),
//! [`model`](MessagesRequestContext::model)); `messages` and `system` are
//! deliberately unreachable through it, so a stale typed view can never be read
//! back after a round is appended.

use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::messages_request::{WebSearchBudget, normalize_native_web_search};
use crate::types::messages::{MessagesRequest, ToolParam};
use crate::utils::common::serialize_to_string;

/// One `/v1/messages` request, in both the typed and raw views the gateway tool
/// loops need. See the module docs for why both exist.
#[derive(Debug)]
pub struct MessagesRequestContext {
    /// The client's request as received. Read-only: routing and registry
    /// construction only.
    typed: MessagesRequest,
    /// The upstream body. Mutated by the loops; the source of truth for
    /// `messages` and `system`.
    raw: Value,
    /// Request-wide native web-search budget, derived while normalizing `raw`.
    web_search_budget: WebSearchBudget,
}

impl MessagesRequestContext {
    /// Build the context from a request the caller has already parsed for
    /// routing, plus the original body bytes it was parsed from.
    ///
    /// Reusing the caller's `typed` keeps each body parsed exactly once per
    /// view: the routing parse is carried into the loop instead of being
    /// discarded, and the raw parse only happens on the loop path, so a proxied
    /// request never pays for a view it does not use.
    ///
    /// Native web-search declarations are validated and normalized here, before
    /// a streaming handler commits its HTTP status — an invalid declaration must
    /// surface as an error response, not as a mid-stream event.
    ///
    /// # Errors
    /// Returns [`ExecutorError::JsonError`] if `body` is not valid JSON, or
    /// [`ExecutorError::InvalidRequest`] if it carries an unsupported or invalid
    /// native web-search declaration.
    pub fn new(typed: MessagesRequest, body: &[u8]) -> ExecutorResult<Self> {
        let raw = serde_json::from_slice(body).map_err(ExecutorError::JsonError)?;
        Self::from_parts(typed, raw)
    }

    /// Build the context from a raw JSON body alone, deriving the typed view
    /// from it.
    ///
    /// Prefer [`new`](Self::new) when the caller has already parsed the request
    /// for routing; this exists for callers that only hold a [`Value`].
    ///
    /// # Errors
    /// Returns [`ExecutorError::JsonError`] if `raw` is not a well-formed
    /// Messages request, or [`ExecutorError::InvalidRequest`] if it carries an
    /// unsupported or invalid native web-search declaration.
    pub fn from_value(raw: Value) -> ExecutorResult<Self> {
        // Deserializing from the parsed tree avoids re-lexing the body text.
        let typed = MessagesRequest::deserialize(&raw).map_err(ExecutorError::JsonError)?;
        Self::from_parts(typed, raw)
    }

    fn from_parts(typed: MessagesRequest, mut raw: Value) -> ExecutorResult<Self> {
        let web_search_budget = normalize_native_web_search(&mut raw)?;
        Ok(Self {
            typed,
            raw,
            web_search_budget,
        })
    }

    /// The tools the client declared, for routing and registry construction.
    ///
    /// These are the client's declarations as received — before the upstream
    /// normalization applied to `raw` — which is what the tool seam
    /// needs to recognise a native server-tool declaration.
    #[must_use]
    pub fn tools(&self) -> Option<&Vec<ToolParam>> {
        self.typed.tools.as_ref()
    }

    /// Whether the client asked for a streaming response.
    #[must_use]
    pub fn stream(&self) -> bool {
        self.typed.stream
    }

    /// The model the client requested.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.typed.model
    }

    /// The body to POST upstream for the next round.
    ///
    /// # Errors
    /// Returns [`ExecutorError::JsonError`] if the body cannot be serialized.
    pub(super) fn upstream_body(&self) -> ExecutorResult<String> {
        serialize_to_string(&self.raw).map_err(ExecutorError::JsonError)
    }

    /// Force the upstream streaming mode, regardless of what the client asked.
    ///
    /// Each loop drives its own rounds and so pins `stream` to what it can
    /// consume; the client-facing mode is [`stream`](Self::stream), decided by
    /// the handler before the loop starts.
    pub(super) fn force_stream(&mut self, streaming: bool) {
        self.raw["stream"] = Value::Bool(streaming);
    }

    /// Reserve up to `requested` native web searches, returning how many may run.
    pub(super) fn reserve_searches(&mut self, requested: usize) -> usize {
        self.web_search_budget.reserve(requested)
    }

    /// Append the model's assistant turn (preserving its `thinking`/`text`/
    /// `tool_use` blocks in order — F3) and a following user turn of
    /// `tool_result`s, so the next upstream round sees the full conversation
    /// state. These stay internal — the client never sees them (hide-the-call).
    ///
    /// # Errors
    /// Returns [`ExecutorError::InvalidRequest`] if the body has no `messages`
    /// array to append to. Unreachable for a context built through either
    /// constructor, since `MessagesRequest::messages` is a required array —
    /// erroring keeps it from silently no-opping into a loop that re-POSTs an
    /// unchanged body until the round cap.
    pub(super) fn append_round(&mut self, assistant_content: &[Value], tool_results: Vec<Value>) -> ExecutorResult<()> {
        let messages = self
            .raw
            .get_mut("messages")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| ExecutorError::InvalidRequest("request has no messages array".to_owned()))?;
        messages.push(json!({ "role": "assistant", "content": assistant_content }));
        // Built by hand rather than with `json!` so the tool outputs move in
        // instead of being deep-copied — a web-search result runs to kilobytes.
        let mut user = Map::new();
        user.insert("role".to_owned(), Value::String("user".to_owned()));
        user.insert("content".to_owned(), Value::Array(tool_results));
        messages.push(Value::Object(user));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> Value {
        json!({
            "model": "qwen3", "max_tokens": 1024, "stream": true,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "web_search", "type": "web_search_20250305", "max_uses": 2}]
        })
    }

    #[test]
    fn typed_view_reads_client_fields_and_raw_carries_upstream_normalization() {
        let ctx = MessagesRequestContext::from_value(request()).unwrap();

        assert_eq!(ctx.model(), "qwen3");
        assert!(ctx.stream());
        // The typed view keeps the client's native declaration, which is what
        // the tool seam classifies on...
        let tools = ctx.tools().expect("tools");
        assert_eq!(tools[0].name, "web_search");
        assert_eq!(tools[0].type_.as_deref(), Some("web_search_20250305"));
        // ...while the raw body carries the function-tool shape vLLM accepts.
        assert_eq!(ctx.raw["tools"][0]["name"], "web_search");
        assert!(ctx.raw["tools"][0].get("type").is_none());
        assert!(ctx.raw["tools"][0].get("input_schema").is_some());
    }

    #[test]
    fn force_stream_overrides_the_client_mode_without_touching_the_typed_view() {
        let mut ctx = MessagesRequestContext::from_value(request()).unwrap();
        ctx.force_stream(false);

        assert_eq!(ctx.raw["stream"], json!(false));
        assert!(ctx.stream(), "the client's requested mode is still readable");
    }

    #[test]
    fn append_round_extends_the_raw_history_only() {
        let mut ctx = MessagesRequestContext::from_value(request()).unwrap();
        let assistant = vec![json!({"type": "tool_use", "id": "t1", "name": "web_search", "input": {}})];
        ctx.append_round(&assistant, vec![json!({"type": "tool_result", "tool_use_id": "t1"})])
            .unwrap();

        let messages = ctx.raw["messages"].as_array().expect("messages");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], json!(assistant));
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"][0]["tool_use_id"], "t1");
    }

    #[test]
    fn budget_is_shared_across_rounds() {
        let mut ctx = MessagesRequestContext::from_value(request()).unwrap();
        assert_eq!(ctx.reserve_searches(1), 1);
        assert_eq!(ctx.reserve_searches(3), 1, "max_uses caps the request-wide total");
        assert_eq!(ctx.reserve_searches(1), 0);
    }

    #[test]
    fn invalid_native_web_search_declaration_is_rejected_at_construction() {
        let mut body = request();
        body["tools"][0]["max_uses"] = json!(0);
        let error = MessagesRequestContext::from_value(body).unwrap_err();
        assert!(matches!(error, ExecutorError::InvalidRequest(_)), "{error:?}");
    }

    #[test]
    fn unmodeled_blocks_and_cache_control_survive_in_the_raw_body() {
        // The reason the raw view exists: a typed round-trip would drop these.
        let body = json!({
            "model": "m", "max_tokens": 8,
            "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
                {"type": "redacted_thinking", "data": "enc"}
            ]}]
        });
        let ctx = MessagesRequestContext::from_value(body.clone()).unwrap();
        assert_eq!(ctx.raw, body);
    }

    #[test]
    fn new_reuses_the_routing_parse() {
        let body = serde_json::to_vec(&request()).unwrap();
        let typed: MessagesRequest = serde_json::from_slice(&body).unwrap();
        let ctx = MessagesRequestContext::new(typed, &body).unwrap();

        assert_eq!(ctx.model(), "qwen3");
        assert_eq!(ctx.raw["messages"][0]["content"], "hi");
    }
}
