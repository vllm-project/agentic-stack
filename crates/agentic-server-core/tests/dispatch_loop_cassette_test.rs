//! Cassette-driven integration tests for the reworked gateway tool loop
//! (`run_until_gateway_tools_complete` via `ExecuteRequest::run`).
//!
//! These drive real recorded `OpenAI` Responses wire bodies through the loop to
//! confirm the `LoopDecision` classification behaves end-to-end:
//!   * client-owned (function) tool calls terminate the loop in one model call
//!     and are handed back to the caller (`RequiresClientAction`);
//!   * parallel function calls in a single turn are preserved on output;
//!   * a gateway-owned call drives another round and comes back `completed`.
//!
//! The synthetic-payload edge cases (usage accumulation, fanout cap, error
//! feedback, the round-cap `incomplete` path) live in `web_search_tool_test.rs`;
//! this file focuses on coverage against recorded `OpenAI` wire bodies.

use std::sync::Arc;

use agentic_core::executor::{ConversationHandler, ExecuteRequest, ExecutionContext, ResponseHandler};
use agentic_core::storage::{ConversationStore, ResponseStore};
use agentic_core::tool::WebSearchHandler;
use agentic_core::types::io::{OutputItem, ResponsesInput};
use agentic_core::types::request_response::RequestPayload;
use agentic_core::types::tools::ResponsesTool;
use either::Either;
use futures::StreamExt;

mod support;

const CASSETTE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes");

/// Load turn N (0-based) response from a multi-turn cassette under
/// `tests/cassettes/tool_calls/multi_turn` as a mock response for the LLM
/// `MockServer`.
///
/// Uses a permissive local parser (input typed as `Value`) rather than the
/// shared strict `Cassette` — these `OpenAI` cassettes carry array-valued `input`
/// on later turns, which the shared string-typed loader rejects. We only need
/// the response, so the request shape is irrelevant here.
fn cassette_turn(filename: &str, turn: usize) -> support::MockResponse {
    cassette_turn_at(&format!("tool_calls/multi_turn/{filename}"), turn)
}

/// Same as [`cassette_turn`] but takes a path relative to `tests/cassettes`, so
/// recordings outside `tool_calls/multi_turn` (e.g. `codex/…`) can be replayed
/// through the loop too.
fn cassette_turn_at(rel_path: &str, turn: usize) -> support::MockResponse {
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Doc {
        turns: Vec<DocTurn>,
    }
    #[derive(Deserialize)]
    struct DocTurn {
        response: DocResponse,
    }
    #[derive(Deserialize)]
    struct DocResponse {
        #[serde(default)]
        body: Option<serde_json::Value>,
        #[serde(default)]
        sse: Option<Vec<String>>,
    }

    let path = format!("{CASSETTE_DIR}/{rel_path}");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let doc: Doc = serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"));
    let resp = doc
        .turns
        .into_iter()
        .nth(turn)
        .unwrap_or_else(|| panic!("cassette {rel_path} missing turn {turn}"))
        .response;

    if let Some(body) = resp.body {
        support::MockResponse::Json(serde_json::to_string(&body).expect("cassette body serializes"))
    } else if let Some(sse) = resp.sse {
        support::MockResponse::Sse(sse.join(""))
    } else {
        panic!("cassette {rel_path} turn {turn} has neither body nor sse");
    }
}

async fn build_exec_ctx(llm_url: &str) -> Arc<ExecutionContext> {
    let pool = support::setup_pool().await;
    let conv_handler = ConversationHandler::new(ConversationStore::new(Arc::clone(&pool)));
    let resp_handler = ResponseHandler::new(ResponseStore::new(pool));
    let client = Arc::new(reqwest::Client::new());
    Arc::new(ExecutionContext::new(
        conv_handler,
        resp_handler,
        client,
        llm_url.to_owned(),
    ))
}

/// Same as `build_exec_ctx` but registers a `WebSearchHandler` gateway executor
/// backed by the given You.com mock URL.
async fn build_exec_ctx_with_web_search(llm_url: &str, you_url: &str) -> Arc<ExecutionContext> {
    let pool = support::setup_pool().await;
    let conv_handler = ConversationHandler::new(ConversationStore::new(Arc::clone(&pool)));
    let resp_handler = ResponseHandler::new(ResponseStore::new(pool));
    let client = Arc::new(reqwest::Client::new());
    Arc::new(
        ExecutionContext::new(conv_handler, resp_handler, Arc::clone(&client), llm_url.to_owned())
            .with_gateway_executor(Arc::new(WebSearchHandler::with_api_key(
                client,
                "secret-you-key".to_owned(),
                you_url,
            ))),
    )
}

fn request(text: &str, tools: Option<Vec<ResponsesTool>>) -> RequestPayload {
    RequestPayload {
        model: "test-model".to_owned(),
        input: ResponsesInput::Text(text.to_owned()),
        instructions: None,
        previous_response_id: None,
        conversation_id: None,
        tools,
        tool_choice: None,
        stream: false,
        store: true,
        include: None,
        reasoning: None,
        text: None,
        temperature: None,
        top_p: None,
        max_output_tokens: Some(1024),
        truncation: None,
        metadata: None,
        parallel_tool_calls: None,
        cache_salt: None,
        context_management: None,
    }
}

fn function_call_names(output: &[OutputItem]) -> Vec<&str> {
    output
        .iter()
        .filter_map(|item| match item {
            OutputItem::FunctionCall(fc) => Some(fc.name.as_str()),
            _ => None,
        })
        .collect()
}

/// `OpenAI` cassette, turn 1 emits a single client-owned `get_job_status`
/// function call. With no gateway executor registered, the loop must classify
/// this as `RequiresClientAction`: exactly one model call, the call handed back
/// on `output`, `status: "completed"`.
#[tokio::test]
async fn openai_single_client_owned_call_terminates_in_one_round() {
    let llm = support::MockServer::start_deque(vec![cassette_turn("openai_responses_tool_calls_3turn.yaml", 0)]).await;
    let exec_ctx = build_exec_ctx(llm.url()).await;

    let result = ExecuteRequest::new(request("check job status", None), exec_ctx)
        .run()
        .await
        .expect("execute should succeed");
    let Either::Left(response) = result else {
        panic!("non-streaming request should return a payload");
    };

    // Client-owned function call: no gateway execution, exactly one LLM call.
    assert_eq!(
        llm.request_bodies().await.len(),
        1,
        "client-owned tools take one model call"
    );
    assert_eq!(response.status, "completed");
    assert_eq!(
        function_call_names(&response.output),
        vec!["get_job_status"],
        "the client-owned call must be handed back on output"
    );
}

/// `OpenAI` cassette, turn 1 emits two parallel client-owned function calls
/// (`get_job_status` + `web_search`). With no gateway executor, both are handed
/// back unexecuted in a single round.
#[tokio::test]
async fn openai_parallel_client_owned_calls_preserved_on_output() {
    let llm =
        support::MockServer::start_deque(vec![cassette_turn("openai_responses_tool_calls_parallel.yaml", 0)]).await;
    let exec_ctx = build_exec_ctx(llm.url()).await;

    let result = ExecuteRequest::new(request("check status and search", None), exec_ctx)
        .run()
        .await
        .expect("execute should succeed");
    let Either::Left(response) = result else {
        panic!("non-streaming request should return a payload");
    };

    assert_eq!(llm.request_bodies().await.len(), 1);
    let names = function_call_names(&response.output);
    assert_eq!(names.len(), 2, "both parallel calls preserved, got {names:?}");
    assert!(
        names.contains(&"get_job_status"),
        "expected get_job_status in {names:?}"
    );
    assert!(names.contains(&"web_search"), "expected web_search in {names:?}");
}

/// When `web_search` is declared as a gateway tool AND a `WebSearchHandler` is
/// registered, the same first-turn call is gateway-owned: the loop executes it,
/// feeds the output back, and the recorded second turn returns text. The
/// `get_job_status` call in the parallel cassette stays client-owned, so the
/// turn is `RequiresClientAction` and terminates after one model call — the
/// gateway `web_search` output is still recorded alongside it.
#[tokio::test]
async fn openai_mixed_gateway_and_client_owned_hands_back_after_gateway_exec() {
    let (you_url, mut captured_you, _you_handle) = spawn_mock_you().await;
    let llm =
        support::MockServer::start_deque(vec![cassette_turn("openai_responses_tool_calls_parallel.yaml", 0)]).await;
    let exec_ctx = build_exec_ctx_with_web_search(llm.url(), &you_url).await;
    let web_search: ResponsesTool = serde_json::from_value(serde_json::json!({"type": "web_search_preview"})).unwrap();

    let result = ExecuteRequest::new(request("status and search", Some(vec![web_search])), exec_ctx)
        .run()
        .await
        .expect("execute should succeed");
    let Either::Left(response) = result else {
        panic!("non-streaming request should return a payload");
    };

    // web_search is gateway-owned → executed against You.com mock.
    let search = recv_search(&mut captured_you, "web_search should hit You.com").await;
    assert!(search.body.get("query").is_some(), "web_search executed with a query");

    // get_job_status is still client-owned → RequiresClientAction, one model call.
    assert_eq!(llm.request_bodies().await.len(), 1);
    assert!(
        function_call_names(&response.output).contains(&"get_job_status"),
        "client-owned call handed back alongside the executed gateway call"
    );
}

/// Drive a real recorded 2-turn cassette (gateway `web_search` turn then a Codex
/// `namespace` turn) through the streaming loop and assert the end-to-end
/// behavior — shared by the `OpenAI` and `vLLM` backend recordings.
///
/// Round 0: the model emits a gateway-owned `web_search` call → the loop
/// executes it against the You.com mock and continues. Round 1: the model emits
/// the flat, model-visible `agentic_ns__mcp__shell__run` → the loop restores it
/// to `{namespace: "mcp__shell", name: "run"}`, classifies it client-owned, and
/// hands the turn back (`RequiresClientAction`). Two model calls total; the
/// namespace call is returned restored, never flattened.
async fn assert_web_search_then_namespace(cassette_rel_path: &str) {
    let (you_url, mut captured_you, _you_handle) = spawn_mock_you().await;
    let llm = support::MockServer::start_deque(vec![
        cassette_turn_at(cassette_rel_path, 0),
        cassette_turn_at(cassette_rel_path, 1),
    ])
    .await;
    let exec_ctx = build_exec_ctx_with_web_search(llm.url(), &you_url).await;

    let web_search: ResponsesTool = serde_json::from_value(serde_json::json!({"type": "web_search_preview"})).unwrap();
    let codex_namespace: ResponsesTool = serde_json::from_value(serde_json::json!({
        "type": "namespace",
        "name": "mcp__shell",
        "tools": [{"type": "function", "name": "run", "parameters": {"type": "object"}}]
    }))
    .unwrap();

    let mut payload = request("search then run pwd", Some(vec![web_search, codex_namespace]));
    payload.stream = true;
    let result = ExecuteRequest::new(payload, exec_ctx)
        .run()
        .await
        .expect("execute should succeed");
    let Either::Right(stream) = result else {
        panic!("streaming request should return a stream");
    };
    let chunks: Vec<String> = stream.collect().await;

    // Round 0 executed the gateway web_search against the You.com mock, then
    // round 1 emitted the namespace call — two model calls in one conversation.
    let search = recv_search(&mut captured_you, "web_search should hit You.com").await;
    assert!(search.body.get("query").is_some(), "web_search executed with a query");
    assert_eq!(
        llm.request_bodies().await.len(),
        2,
        "gateway round + client-action round"
    );

    let events = support::streamed_sse_events(&chunks);

    // The gateway web_search surfaced as public web_search_call lifecycle events;
    // its raw function call never leaked to the client.
    let event_types: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();
    assert!(
        event_types.contains(&"response.web_search_call.completed"),
        "gateway web_search should emit its public lifecycle events; got {event_types:?}"
    );
    assert!(
        !events
            .iter()
            .any(|e| e["item"]["type"] == "function_call" && e["item"]["name"] == "web_search"),
        "internal web_search function call must not be forwarded"
    );

    // The terminal response hands back the namespace call restored to
    // {namespace, name}, never the flat model-visible name.
    let terminal = events
        .iter()
        .rev()
        .find_map(|e| e.get("response").filter(|r| r.get("output").is_some()))
        .expect("stream should include a terminal response payload");
    let items = terminal["output"].as_array().unwrap();
    let ns_call = items
        .iter()
        .find(|it| it["type"] == "function_call" && it["namespace"] == "mcp__shell")
        .expect("namespace call handed back to the client");
    assert_eq!(ns_call["name"], "run", "flat name restored to member name");
    assert!(
        !items.iter().any(|it| it["name"] == "agentic_ns__mcp__shell__run"),
        "flat namespaced name must not leak to the client"
    );
    assert!(
        items.iter().any(|it| it["type"] == "web_search_call"),
        "web_search resolved into a public web_search_call: {terminal:#}"
    );
}

/// Real recorded **`OpenAI`** (`gpt-4o`) traffic through the mixed
/// gateway-plus-namespace flow. Both turns are typed incremental SSE.
#[tokio::test]
async fn openai_gateway_web_search_then_codex_namespace_across_turns() {
    assert_web_search_then_namespace("codex/codex-openai-web-search-then-namespace-gpt-4o-streaming.yaml").await;
}

/// Real recorded **vLLM** (`openai/gpt-oss-20b`) traffic through the same flow —
/// backend parity with the `OpenAI` recording (mirrors the dual-backend approach
/// in PR #77).
#[tokio::test]
async fn vllm_gateway_web_search_then_codex_namespace_across_turns() {
    assert_web_search_then_namespace("codex/codex-vllm-web-search-then-namespace-gpt-oss-20b-streaming.yaml").await;
}

/// Drive a real recorded 3-turn cassette where the model calls the gateway
/// `web_search` tool on two consecutive turns before answering. This is the only
/// coverage of the loop's `Continue` decision firing **more than once**: round 0
/// and round 1 both dispatch a gateway call and loop, round 2 returns a message
/// so the loop terminates `Done`. Three model calls, two You.com executions.
async fn assert_multi_round_web_search(cassette_rel_path: &str, expected_item_types: &[&str]) {
    let (you_url, mut captured_you, _you_handle) = spawn_mock_you().await;
    let llm = support::MockServer::start_deque(vec![
        cassette_turn_at(cassette_rel_path, 0),
        cassette_turn_at(cassette_rel_path, 1),
        cassette_turn_at(cassette_rel_path, 2),
    ])
    .await;
    let exec_ctx = build_exec_ctx_with_web_search(llm.url(), &you_url).await;

    let web_search: ResponsesTool = serde_json::from_value(serde_json::json!({"type": "web_search_preview"})).unwrap();
    let mut payload = request("research rust async", Some(vec![web_search]));
    payload.stream = true;
    let result = ExecuteRequest::new(payload, exec_ctx)
        .run()
        .await
        .expect("execute should succeed");
    let Either::Right(stream) = result else {
        panic!("streaming request should return a stream");
    };
    let chunks: Vec<String> = stream.collect().await;

    // Two gateway web_search executions (rounds 0 and 1), three model calls total
    // (the loop continued twice, then the model answered).
    let first = recv_search(&mut captured_you, "first web_search should hit You.com").await;
    assert!(first.body.get("query").is_some(), "round-0 web_search executed");
    let second = recv_search(&mut captured_you, "second web_search should hit You.com").await;
    assert!(second.body.get("query").is_some(), "round-1 web_search executed");
    assert_eq!(
        llm.request_bodies().await.len(),
        3,
        "two Continue rounds + a final answering round"
    );

    let events = support::streamed_sse_events(&chunks);
    assert_multi_round_public_stream(&events, expected_item_types);
    // Both gateway calls surfaced as public web_search_call items in the terminal
    // response; the final turn produced a message.
    let terminal = &events.last().expect("stream should end with a terminal event")["response"];
    let items = terminal["output"].as_array().unwrap();
    let searches = items.iter().filter(|it| it["type"] == "web_search_call").count();
    assert_eq!(searches, 2, "both gateway searches recorded on output: {terminal:#}");
    assert!(
        items.iter().any(|it| it["type"] == "message"),
        "final turn produced a message"
    );
    assert!(
        !items
            .iter()
            .any(|it| it["type"] == "function_call" && it["name"] == "web_search"),
        "raw web_search function calls must stay internal"
    );
}

fn assert_multi_round_public_stream(events: &[serde_json::Value], expected_item_types: &[&str]) {
    let lifecycle: Vec<&str> = events
        .iter()
        .map(|event| event["type"].as_str().expect("event should have a type"))
        .filter(|kind| {
            matches!(
                *kind,
                "response.created"
                    | "response.in_progress"
                    | "response.completed"
                    | "response.incomplete"
                    | "response.failed"
                    | "error"
            )
        })
        .collect();
    assert_eq!(
        lifecycle,
        ["response.created", "response.in_progress", "response.completed"],
        "all inference rounds should share one successful public lifecycle"
    );
    assert_eq!(events.first().unwrap()["type"], "response.created");
    let terminal = events.last().unwrap();
    assert_eq!(
        terminal["type"], "response.completed",
        "completion must be the last event"
    );

    for (sequence, event) in events.iter().enumerate() {
        assert_eq!(
            event["sequence_number"].as_u64(),
            Some(u64::try_from(sequence).unwrap()),
            "public sequence must include synthetic and terminal events: {event}"
        );
    }

    let mut expected_boundaries = Vec::new();
    for (index, item_type) in expected_item_types.iter().enumerate() {
        let index = u64::try_from(index).unwrap();
        expected_boundaries.push(("response.output_item.added", index));
        if *item_type == "web_search_call" {
            expected_boundaries.extend([
                ("response.web_search_call.in_progress", index),
                ("response.web_search_call.searching", index),
                ("response.web_search_call.completed", index),
            ]);
        }
        expected_boundaries.push(("response.output_item.done", index));
    }
    let actual_boundaries: Vec<(&str, u64)> = events
        .iter()
        .filter(|event| {
            let kind = event["type"].as_str().unwrap();
            kind.starts_with("response.output_item.") || kind.starts_with("response.web_search_call.")
        })
        .map(|event| {
            (
                event["type"].as_str().unwrap(),
                event["output_index"].as_u64().expect("item event should have an index"),
            )
        })
        .collect();
    assert_eq!(
        actual_boundaries, expected_boundaries,
        "recorded reasoning, synthesized searches, and the answer must retain their public item order"
    );

    let output = terminal["response"]["output"].as_array().unwrap();
    let actual_item_types: Vec<&str> = output.iter().map(|item| item["type"].as_str().unwrap()).collect();
    assert_eq!(actual_item_types, expected_item_types);
    assert_recorded_text_channels(events, output, expected_item_types);
}

fn assert_recorded_text_channels(
    events: &[serde_json::Value],
    output: &[serde_json::Value],
    expected_item_types: &[&str],
) {
    for (index, item_type) in expected_item_types.iter().enumerate() {
        let delta_type = match *item_type {
            "reasoning" => "response.reasoning_text.delta",
            "message" => "response.output_text.delta",
            _ => continue,
        };
        let index = u64::try_from(index).unwrap();
        assert!(
            events.iter().any(|event| {
                event["type"] == delta_type
                    && event["output_index"].as_u64() == Some(index)
                    && event["delta"].as_str().is_some_and(|text| !text.is_empty())
            }),
            "expected nonempty {delta_type} for public item {index}"
        );
    }

    let visible_text: String = events
        .iter()
        .filter(|event| event["type"] == "response.output_text.delta")
        .map(|event| {
            let index = usize::try_from(event["output_index"].as_u64().unwrap()).unwrap();
            assert_eq!(
                expected_item_types[index], "message",
                "answer delta must target a message"
            );
            event["delta"].as_str().expect("text delta should be a string")
        })
        .collect();
    let terminal_text: String = output
        .iter()
        .filter(|item| item["type"] == "message")
        .flat_map(|item| item["content"].as_array().unwrap())
        .filter(|part| part["type"] == "output_text")
        .map(|part| part["text"].as_str().unwrap())
        .collect();
    assert_eq!(
        visible_text, terminal_text,
        "streamed answer should match the terminal answer"
    );
    assert!(!visible_text.is_empty(), "recorded scenario must produce visible text");
    // Inspect the assembled answer so a tag split across delta events is still detected.
    assert!(!visible_text.contains("<think>") && !visible_text.contains("</think>"));
}

/// Real recorded **`OpenAI`** (`gpt-4o`) multi-round gateway loop.
#[tokio::test]
async fn openai_multi_round_web_search_loops_then_answers() {
    assert_multi_round_web_search(
        "codex/codex-openai-multi-round-web-search-gpt-4o-streaming.yaml",
        &["web_search_call", "web_search_call", "message"],
    )
    .await;
}

/// Real recorded **vLLM** (`openai/gpt-oss-20b`) multi-round gateway loop —
/// backend parity.
#[tokio::test]
async fn vllm_multi_round_web_search_loops_then_answers() {
    assert_multi_round_web_search(
        "codex/codex-vllm-multi-round-web-search-gpt-oss-20b-streaming.yaml",
        &[
            "reasoning",
            "web_search_call",
            "reasoning",
            "web_search_call",
            "message",
        ],
    )
    .await;
}

/// Real recorded **OpenAI** (`gpt-4o`) turn that emits a gateway `web_search`
/// call **and** a Codex namespace call in a *single* model output. The loop
/// executes the gateway call and, because a client-owned call is also present,
/// hands the turn back (`RequiresClientAction`) in one round — the gateway
/// result is recorded alongside the restored namespace call. gpt-oss does not
/// emit both tools in one turn, so this path is `OpenAI`-only.
#[tokio::test]
async fn openai_web_search_and_namespace_same_turn_hands_back_in_one_round() {
    let (you_url, mut captured_you, _you_handle) = spawn_mock_you().await;
    let llm = support::MockServer::start_deque(vec![cassette_turn_at(
        "codex/codex-openai-web-search-and-namespace-same-turn-gpt-4o-streaming.yaml",
        0,
    )])
    .await;
    let exec_ctx = build_exec_ctx_with_web_search(llm.url(), &you_url).await;

    let web_search: ResponsesTool = serde_json::from_value(serde_json::json!({"type": "web_search_preview"})).unwrap();
    let codex_namespace: ResponsesTool = serde_json::from_value(serde_json::json!({
        "type": "namespace",
        "name": "mcp__shell",
        "tools": [{"type": "function", "name": "run", "parameters": {"type": "object"}}]
    }))
    .unwrap();
    let mut payload = request("search and run pwd together", Some(vec![web_search, codex_namespace]));
    payload.stream = true;
    let result = ExecuteRequest::new(payload, exec_ctx)
        .run()
        .await
        .expect("execute should succeed");
    let Either::Right(stream) = result else {
        panic!("streaming request should return a stream");
    };
    let chunks: Vec<String> = stream.collect().await;

    // The gateway web_search executed this turn, and the turn ended in ONE model
    // call (no loop-back) because a client-owned namespace call was also present.
    let search = recv_search(&mut captured_you, "web_search should hit You.com").await;
    assert!(search.body.get("query").is_some(), "web_search executed with a query");
    assert_eq!(
        llm.request_bodies().await.len(),
        1,
        "mixed turn hands back immediately — no second model call"
    );

    let events = support::streamed_sse_events(&chunks);
    let terminal = events
        .iter()
        .rev()
        .find_map(|e| e.get("response").filter(|r| r.get("output").is_some()))
        .expect("stream should include a terminal response payload");
    let items = terminal["output"].as_array().unwrap();
    // Gateway call resolved to a public web_search_call...
    assert!(
        items.iter().any(|it| it["type"] == "web_search_call"),
        "web_search resolved into a public web_search_call: {terminal:#}"
    );
    // ...and the namespace call handed back restored to {namespace, name}.
    let ns_call = items
        .iter()
        .find(|it| it["type"] == "function_call" && it["namespace"] == "mcp__shell")
        .expect("namespace call handed back to the client");
    assert_eq!(ns_call["name"], "run", "flat name restored to member name");
    assert!(
        !items.iter().any(|it| it["name"] == "agentic_ns__mcp__shell__run"),
        "flat namespaced name must not leak to the client"
    );
}

// --- You.com mock (mirrors web_search_tool_test.rs) -------------------------

struct CapturedSearch {
    body: serde_json::Value,
}

async fn recv_search(captured: &mut tokio::sync::mpsc::Receiver<CapturedSearch>, expectation: &str) -> CapturedSearch {
    tokio::time::timeout(std::time::Duration::from_secs(5), captured.recv())
        .await
        .unwrap_or_else(|_| panic!("{expectation}: timed out after 5 seconds"))
        .unwrap_or_else(|| panic!("{expectation}: mock server stopped before receiving a request"))
}

fn query_params_as_json(uri: &axum::http::Uri) -> serde_json::Value {
    let params = url::form_urlencoded::parse(uri.query().unwrap_or_default().as_bytes())
        .map(|(key, value)| (key.into_owned(), serde_json::Value::String(value.into_owned())))
        .collect();
    serde_json::Value::Object(params)
}

async fn spawn_mock_you() -> (
    String,
    tokio::sync::mpsc::Receiver<CapturedSearch>,
    tokio::task::JoinHandle<()>,
) {
    use axum::extract::State;
    use axum::http::Uri;
    use axum::routing::get;
    use axum::{Json, Router};
    use tokio::net::TcpListener;
    use tokio::sync::mpsc;

    let (tx, rx) = mpsc::channel(16);
    let app = Router::new()
        .route(
            "/v1/search",
            get(
                move |State(tx): State<mpsc::Sender<CapturedSearch>>, uri: Uri| async move {
                    let body = query_params_as_json(&uri);
                    tx.send(CapturedSearch { body }).await.unwrap();
                    (
                        axum::http::StatusCode::OK,
                        Json(serde_json::json!({
                            "results": {"web": [{"url": "https://example.com", "title": "R"}], "news": []},
                            "metadata": {"query": "q"}
                        })),
                    )
                },
            ),
        )
        .with_state(tx);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), rx, handle)
}
