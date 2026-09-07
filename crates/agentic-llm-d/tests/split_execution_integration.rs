//! Split execution at the library level: what the handlers compose, and the
//! cases HTTP cannot reach. The round trip over a socket is in `backend_mode_test`.
#![allow(clippy::result_large_err)] // `ExecutorError` is core's; boxing it is not ours to decide

use std::sync::Arc;

use agentic_core::executor::request::RequestContext;
use agentic_core::executor::{
    ConversationHandler, ExecutionContext, ExecutorError, ExecutorResult, ResponseHandler, UpstreamBody, commit,
    decode_upstream, persist_turn, rehydrate_conversation, upstream_request,
};
use agentic_core::storage::{ConversationStore, ResponseStore, create_pool_with_schema};
use agentic_core::types::request_response::{RequestPayload, ResponsePayload};
use agentic_llm_d::SigningKey;
use agentic_llm_d::context::{Hydration, SplitContext, ensure_splittable, seal, unseal};
use serde_json::value::RawValue;
use serde_json::{Value, json};

async fn exec_ctx() -> ExecutionContext {
    let pool = create_pool_with_schema(Some("sqlite://?mode=memory"))
        .await
        .expect("pool");
    ExecutionContext::new(
        ConversationHandler::new(ConversationStore::new(Arc::clone(&pool))),
        ResponseHandler::new(ResponseStore::new(pool)),
        Arc::new(reqwest::Client::new()),
        "http://localhost:8000".to_owned(),
    )
}

/// Built the way a request actually arrives — through deserialization.
fn request(input: &str, previous: Option<&str>) -> RequestPayload {
    let mut body = json!({"model": "test-model", "input": input, "store": true});
    if let Some(previous) = previous {
        body["previous_response_id"] = json!(previous);
    }
    serde_json::from_value(body).expect("valid request")
}

fn answer(text: &str) -> Value {
    json!([{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}])
}

/// A complete upstream body, as the model backend returns it.
fn upstream_json(text: &str) -> String {
    json!({"id": "resp_upstream", "object": "response", "created_at": 1_700_000_000,
           "model": "test-model", "status": "completed", "output": answer(text)})
    .to_string()
}

/// The same turn as SSE, the way a streaming caller would have relayed it.
fn upstream_sse(text: &str) -> String {
    [
        json!({"type": "response.created", "response": {"id": "resp_upstream", "status": "in_progress"}}),
        json!({"type": "response.in_progress", "response": {"id": "resp_upstream", "status": "in_progress"}}),
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []}}),
        json!({"type": "response.content_part.added", "output_index": 0, "content_index": 0,
               "item_id": "msg_1", "part": {"type": "output_text", "text": "", "annotations": []}}),
        json!({"type": "response.output_text.delta", "output_index": 0, "content_index": 0,
               "item_id": "msg_1", "delta": text}),
        json!({"type": "response.output_text.done", "output_index": 0, "content_index": 0,
               "item_id": "msg_1", "text": text}),
        json!({"type": "response.content_part.done", "output_index": 0, "content_index": 0,
               "item_id": "msg_1", "part": answer(text)[0]["content"][0]}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": answer(text)[0]}),
        json!({"type": "response.completed",
               "response": {"id": "resp_upstream", "status": "completed", "output": answer(text)}}),
    ]
    .iter()
    .map(|frame| format!("data: {frame}\n\n"))
    .collect::<Vec<_>>()
    .concat()
}

fn signing_key() -> SigningKey {
    SigningKey::new(b"test-signing-key-32-bytes-minimum!".to_vec()).expect("valid test key")
}

/// What the hydrate handler composes.
async fn hydrate(request: RequestPayload, ctx: &ExecutionContext) -> ExecutorResult<Hydration> {
    ensure_splittable(&request)?;
    let live = rehydrate_conversation(request, ctx).await?;
    ensure_splittable(&live.enriched_request)?;
    let stream = live.original_request.stream;
    let body = RawValue::from_string(upstream_request(&live, stream)?).map_err(ExecutorError::JsonError)?;
    Ok(Hydration {
        request: body,
        context: seal(live.into(), &signing_key())?,
    })
}

/// What the persist handler composes.
async fn persist(
    context: String,
    upstream: UpstreamBody<'_>,
    ctx: &ExecutionContext,
) -> ExecutorResult<ResponsePayload> {
    let live = RequestContext::from(unseal(&context, &signing_key())?);
    let payload = decode_upstream(&live, upstream)?;
    commit(live, payload, ctx).await
}

async fn turn(input: &str, previous: Option<&str>, ctx: &ExecutionContext) -> Hydration {
    hydrate(request(input, previous), ctx).await.expect("hydrate")
}

/// How many input items the upstream request replays, and their combined text.
fn replayed(turn: &Hydration) -> (usize, String) {
    let sent: Value = serde_json::from_str(turn.request.get()).expect("valid request");
    let items = sent["input"].as_array().expect("input items").clone();
    (items.len(), items.iter().map(ToString::to_string).collect())
}

fn status_of(error: &ExecutorError) -> u16 {
    error.http_status().as_u16()
}

#[tokio::test]
async fn a_streamed_turn_persists_from_the_relayed_frames() {
    let ctx = exec_ctx().await;
    let mut streaming = request("What is 2+2?", None);
    streaming.stream = true;
    let streamed = hydrate(streaming, &ctx).await.expect("hydrate");

    let stored = persist(streamed.context, UpstreamBody::Sse(&upstream_sse("4")), &ctx)
        .await
        .expect("persist from SSE");
    let next = turn("What did I ask?", Some(&stored.id), &ctx).await;
    assert_eq!(replayed(&next).0, 3, "the streamed turn is continuable");
}

#[tokio::test]
async fn output_item_done_is_authoritative_when_delta_is_missing() {
    let ctx = exec_ctx().await;
    let streamed = turn("What is 2+2?", None, &ctx).await;
    let sse = [
        json!({"type": "response.created", "response": {"id": "resp_upstream", "status": "in_progress"}}),
        json!({"type": "response.in_progress", "response": {"id": "resp_upstream", "status": "in_progress"}}),
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "message", "id": "msg_1", "role": "assistant",
                        "status": "in_progress", "content": []}}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": answer("4")[0]}),
        json!({"type": "response.completed",
               "response": {"id": "resp_upstream", "status": "completed", "output": answer("4")}}),
    ]
    .iter()
    .map(|frame| format!("data: {frame}\n\n"))
    .collect::<Vec<_>>()
    .concat();

    let stored = persist(streamed.context, UpstreamBody::Sse(&sse), &ctx)
        .await
        .expect("persist from authoritative done item");
    let output = serde_json::to_value(&stored.output).expect("serialize output");
    assert_eq!(output[0]["content"][0]["text"], "4");
}

#[tokio::test]
async fn a_function_call_stream_passes_strict_validation() {
    let ctx = exec_ctx().await;
    let streamed = turn("Call lookup", None, &ctx).await;
    let sse = [
        r#"data: {"type":"response.created","response":{"id":"resp_upstream","status":"in_progress"}}"#,
        r#"data: {"type":"response.in_progress","response":{"id":"resp_upstream","status":"in_progress"}}"#,
        r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"","status":"in_progress"}}"#,
        r#"data: {"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":"{\"q\":"}"#,
        r#"data: {"type":"response.function_call_arguments.done","output_index":0,"item_id":"fc_1","name":"lookup","arguments":"{\"q\":\"rust\"}"}"#,
        r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"rust\"}","status":"completed"}}"#,
        r#"data: {"type":"response.completed","response":{"id":"resp_upstream","status":"completed","output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"rust\"}","status":"completed"}]}}"#,
    ]
    .join("\n\n");

    let stored = persist(streamed.context, UpstreamBody::Sse(&sse), &ctx)
        .await
        .expect("persist function-call SSE");
    assert!(matches!(
        stored.output.as_slice(),
        [agentic_core::types::io::OutputItem::FunctionCall(_)]
    ));
}

/// Every way a turn is refused, and the status the caller sees.
#[tokio::test]
async fn a_turn_that_cannot_be_stored_is_refused() {
    let ctx = exec_ctx().await;

    let unknown = hydrate(request("hi", Some("resp_missing")), &ctx).await;
    assert_eq!(status_of(&unknown.expect_err("unknown id")), 404);

    // A caller-supplied body gets none of the in-process parser's defaults,
    // including the catch-all used for compatibility on trusted internal paths.
    let mut in_progress: Value = serde_json::from_str(&upstream_json("partial")).expect("json");
    in_progress["status"] = json!("in_progress");
    for body in [
        in_progress.to_string(),
        r#"{"id":"resp_upstream"}"#.to_owned(),
        r#"{"id":"resp_upstream","status":"completed"}"#.to_owned(),
        r#"{"id":"resp_upstream","status":"completed","output":[123]}"#.to_owned(),
        r#"{"id":"resp_upstream","status":"queued","output":[]}"#.to_owned(),
        r#"{"id":"resp_upstream","status":"potato","output":[]}"#.to_owned(),
        r#"{"id":"resp_upstream","status":"completed","output":[{"type":"future_item","id":"future_1"}]}"#.to_owned(),
        r#"{"id":"resp_upstream","status":"completed","output":[{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"}]}"#.to_owned(),
        r#"{"id":"resp_upstream","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[]},{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[]}]}"#.to_owned(),
    ] {
        let refused = persist(turn("hi", None, &ctx).await.context, UpstreamBody::Json(&body), &ctx).await;
        assert_eq!(status_of(&refused.expect_err("not storable")), 400, "accepted: {body}");
    }

    let completed = r#"data: {"type":"response.completed","response":{"id":"resp_upstream","status":"completed"}}"#;
    let created = r#"data: {"type":"response.created","response":{"id":"resp_upstream","status":"in_progress"}}"#;
    let in_progress =
        r#"data: {"type":"response.in_progress","response":{"id":"resp_upstream","status":"in_progress"}}"#;
    for sse in [
        // A relay that died mid-stream: `finish_stream` would call this complete.
        r#"data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"4"}"#.to_owned(),
        // A frame the accumulator could not read must not be skipped past.
        format!("data: not-json\n\n{completed}"),
        // A terminal event without its required response object must not turn
        // a truncated stream into an empty completed response.
        r#"data: {"type":"response.completed"}"#.to_owned(),
        // A terminal event cannot stand in for the lifecycle that preceded it.
        completed.to_owned(),
        // The response can be created only once.
        [created, created, in_progress, completed].join("\n\n"),
        // Every lifecycle event must describe the same upstream response.
        [
            created,
            in_progress,
            r#"data: {"type":"response.completed","response":{"id":"resp_other","status":"completed"}}"#,
        ]
        .join("\n\n"),
        // A valid terminal event must not hide an unmatched content delta.
        format!(
            "{created}\n\n{in_progress}\n\ndata: {{\"type\":\"response.output_text.delta\",\"output_index\":0,\"item_id\":\"msg_1\",\"delta\":\"4\"}}\n\n{completed}"
        ),
        // Terminal output cannot contain an item that was omitted from the relayed item events.
        format!(
            "{created}\n\n{in_progress}\n\ndata: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"resp_upstream\",\"status\":\"completed\",\"output\":{}}}}}",
            answer("4")
        ),
        // A lifecycle-correct frame must still carry its event-specific data.
        [
            created,
            in_progress,
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}"#,
            r#"data: {"type":"response.output_text.delta","output_index":0,"item_id":"msg_1"}"#,
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[]}}"#,
            completed,
        ]
        .join("\n\n"),
        // Known item events must identify the item they mutate.
        [
            created,
            in_progress,
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"","status":"in_progress"}}"#,
            r#"data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{}"}"#,
        ]
        .join("\n\n"),
        // An event cannot mutate an active item of a different type.
        [
            created,
            in_progress,
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}"#,
            r#"data: {"type":"response.function_call_arguments.delta","output_index":0,"item_id":"msg_1","delta":"{}"}"#,
        ]
        .join("\n\n"),
        // Completed item identifiers and indexes cannot be reused later in the stream.
        [
            created,
            in_progress,
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}"#,
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[]}}"#,
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}"#,
        ]
        .join("\n\n"),
        // Unsupported output item types must not be coerced into empty messages.
        [
            created,
            in_progress,
            r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"file_search_call","id":"fs_1","status":"in_progress","queries":["rust"]}}"#,
            r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"file_search_call","id":"fs_1","status":"completed","queries":["rust"]}}"#,
            completed,
        ]
        .join("\n\n"),
    ] {
        let refused = persist(turn("hi", None, &ctx).await.context, UpstreamBody::Sse(&sse), &ctx).await;
        assert_eq!(status_of(&refused.expect_err("bad stream")), 400, "accepted: {sse}");
    }

    let opened = unseal(&turn("hi", None, &ctx).await.context, &signing_key()).expect("unseal");
    let no_id = seal(
        SplitContext {
            response_id: String::new(),
            ..opened
        },
        &signing_key(),
    )
    .expect("seal");
    let refused = persist(no_id, UpstreamBody::Json(&upstream_json("4")), &ctx).await;
    assert_eq!(status_of(&refused.expect_err("no reserved id")), 400);
}

/// A model that failed is not written, but is not a boundary error either.
#[tokio::test]
async fn a_turn_that_is_not_written_still_returns() {
    let ctx = exec_ctx().await;

    let mut failed: Value = serde_json::from_str(&upstream_json("")).expect("json");
    failed["status"] = json!("failed");
    failed["output"] = json!([]);
    let attempt = turn("hi", None, &ctx).await;
    let id = unseal(&attempt.context, &signing_key()).expect("unseal").response_id;
    let payload = persist(attempt.context, UpstreamBody::Json(&failed.to_string()), &ctx)
        .await
        .expect("a failed turn is not a boundary error");
    assert_eq!(payload.status, "error", "`failed` normalizes to the error status");
    let orphan = hydrate(request("and then?", Some(&id)), &ctx).await;
    assert_eq!(status_of(&orphan.expect_err("never stored")), 404);
}

/// Until an identical retry can be proven identical, a reused id is refused.
#[tokio::test]
async fn reusing_a_reserved_id_is_refused() {
    let ctx = exec_ctx().await;
    let stored = turn("What is 2+2?", None, &ctx).await;
    let context = stored.context.clone();
    let first = persist(stored.context, UpstreamBody::Json(&upstream_json("4")), &ctx)
        .await
        .expect("persist");

    // Different content under the same id must never come back as if stored.
    let reused = persist(context, UpstreamBody::Json(&upstream_json("5")), &ctx).await;
    assert_eq!(status_of(&reused.expect_err("id already used")), 409);

    // And the turn that was stored is untouched.
    let (count, text) = replayed(&turn("and?", Some(&first.id), &ctx).await);
    assert_eq!(count, 3, "stored once");
    assert!(text.contains('4') && !text.contains('5'));
}

/// Concurrent delivery must have the same externally visible retry contract as
/// a sequential retry: one write wins and the duplicate is a 409 conflict.
#[tokio::test]
async fn concurrent_reuse_of_a_reserved_id_is_a_conflict() {
    let ctx = exec_ctx().await;
    let hydrated = turn("What is 2+2?", None, &ctx).await;
    let first_context = hydrated.context.clone();
    let second_context = hydrated.context;
    let first_body = upstream_json("4");
    let second_body = first_body.clone();

    let (first, second) = tokio::join!(
        persist(first_context, UpstreamBody::Json(&first_body), &ctx),
        persist(second_context, UpstreamBody::Json(&second_body), &ctx),
    );

    let statuses = [first, second].map(|result| result.map_or_else(|error| status_of(&error), |_| 200));
    assert_eq!(statuses.iter().filter(|status| **status == 200).count(), 1);
    assert_eq!(statuses.iter().filter(|status| **status == 409).count(), 1);
}

/// A gateway-owned tool reaches a split turn only by inheritance: the gateway
/// stored it, and a continuation sending no `tools` picks it up during
/// rehydration — after the request itself has already passed the check.
#[tokio::test]
async fn an_inherited_gateway_tool_is_refused() {
    let ctx = exec_ctx().await;
    let mut gateway_turn =
        RequestContext::from(unseal(&turn("hi", None, &ctx).await.context, &signing_key()).expect("unseal"));
    gateway_turn.enriched_request.tools = Some(vec![
        serde_json::from_value(json!({"type": "web_search_preview"})).expect("gateway tool"),
    ]);
    let id = gateway_turn.response_id.clone();
    persist_turn(
        gateway_turn,
        serde_json::from_value(answer("sure")).expect("output items"),
        &ctx.conv_handler,
        &ctx.resp_handler,
    )
    .await
    .expect("the gateway stores turns the split path would refuse");

    let error = hydrate(request("and then?", Some(&id)), &ctx)
        .await
        .expect_err("the stored tool is inherited during rehydration");
    assert_eq!(status_of(&error), 400);
    assert!(error.to_string().contains("tools"), "got: {error}");
}

#[test]
fn the_boundary_check_names_what_cannot_be_split() {
    let gateway_tool: RequestPayload = serde_json::from_value(json!({
        "model": "test-model", "input": "hi", "store": true, "tools": [{"type": "web_search_preview"}]
    }))
    .expect("valid request");
    let error = ensure_splittable(&gateway_tool).expect_err("the loop needs a caller");
    assert!(error.to_string().contains("tools"), "got: {error}");

    let mut conversational = request("hi", None);
    conversational.conversation_id = Some("conv_1".into());
    let error = ensure_splittable(&conversational).expect_err("its version cannot cross");
    assert!(error.to_string().contains("conversation_id"), "got: {error}");

    let mut streaming = request("hi", None);
    streaming.stream = true;
    ensure_splittable(&streaming).expect("the caller relays the frames, then replays them");
    ensure_splittable(&request("hi", None)).expect("a plain turn is splittable");
}

/// The wire form drops what it can rebuild, and the rebuild must agree.
#[tokio::test]
async fn the_wire_context_round_trips_into_an_equal_context() {
    let ctx = exec_ctx().await;
    let live = rehydrate_conversation(request("What is 2+2?", None), &ctx)
        .await
        .expect("rehydrate");
    let (id, items) = (live.response_id.clone(), live.new_input_items.len());

    let wire = serde_json::to_string(&SplitContext::from(live)).expect("serialize");
    assert!(!wire.contains("enriched_request"), "already in flight as the request");
    assert!(
        !wire.contains("conversation_version"),
        "conversation mode does not split"
    );

    let back = RequestContext::from(serde_json::from_str::<SplitContext>(&wire).expect("deserialize"));
    assert_eq!(back.response_id, id);
    assert_eq!(back.new_input_items.len(), items, "derived items match");
    assert!(
        back.enriched_request.previous_response_id.is_none(),
        "upstream stays stateless"
    );
    assert!(
        back.conversation_version.is_none(),
        "never resumed with a stale version"
    );
}

/// A context `hydrate` did not issue must not be usable.
#[tokio::test]
async fn a_context_this_service_did_not_seal_is_rejected() {
    let ctx = exec_ctx().await;
    let sealed = turn("hi", None, &ctx).await.context;

    for forged in [
        seal(
            unseal(&sealed, &signing_key()).expect("unseal"),
            &SigningKey::new(b"a-different-signing-key-32-bytes!".to_vec()).expect("alternate test key"),
        )
        .expect("seal"),
        format!("{sealed}tampered"),
    ] {
        let refused = persist(forged, UpstreamBody::Json(&upstream_json("4")), &ctx).await;
        assert_eq!(status_of(&refused.expect_err("not ours")), 400);
    }

    // The one we did issue still works.
    persist(sealed, UpstreamBody::Json(&upstream_json("4")), &ctx)
        .await
        .expect("our own context");
}
