//! Public core continuation/commit and lifecycle seams.
//! Tests use disabled storage or isolated SQLite and local protocol stubs, never a live model.
//! Set `AGENTIC_SESSION_TEST_POSTGRES_URL` to an isolated PostgreSQL database with
//! CREATE DATABASE privileges to rerun persistence cases using fresh databases.

use std::collections::VecDeque;
use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use agentic_core::executor::{
    ConversationHandler, ExecuteRequest, ExecutionContext, ExecutorError, RequestContext, ResponseHandler,
    ResponseSession, ResponseSessionGroup, commit, rehydrate_in_session,
};
use agentic_core::storage::{
    ConversationStore, InOutItem, ResponseMetadata, ResponseStore, StorageError, create_pool_with_schema,
};
use agentic_core::tool::{GatewayExecutor, ToolError, ToolHandler, ToolOutput, ToolType};
use agentic_core::types::io::FunctionTool;
use agentic_core::types::io::output::{FunctionToolCall, GatewayCallStatus, OutputItem, WebSearchCall};
use agentic_core::types::request_response::{RequestPayload, ResponsePayload};
use agentic_core::types::tools::WebSearchToolParam;
use axum::{Json, Router, routing::post};
use serde_json::{Value, json};
use tokio::sync::Mutex;

struct LocalSearch {
    calls: Arc<AtomicUsize>,
}

impl ToolHandler for LocalSearch {
    type ToolParams = WebSearchToolParam;

    fn tool_type(&self) -> ToolType {
        ToolType::WebSearch
    }

    fn validate(&self, _params: &WebSearchToolParam) -> Result<(), ToolError> {
        Ok(())
    }

    fn normalize(&self, _params: &WebSearchToolParam) -> Vec<FunctionTool> {
        vec![FunctionTool {
            type_: "function".to_owned(),
            name: "web_search".to_owned(),
            description: None,
            parameters: Some(json!({"type":"object", "properties":{"query":{"type":"string"}}})),
            strict: Some(false),
        }]
    }
}

impl GatewayExecutor for LocalSearch {
    type ExecutionParams = WebSearchToolParam;

    fn execute(
        &self,
        call_id: &str,
        _tool_name: &str,
        _arguments: &str,
        _params: &WebSearchToolParam,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let call_id = call_id.to_owned();
        Box::pin(async move {
            Ok(ToolOutput {
                call_id,
                output: "local search result".to_owned(),
            })
        })
    }

    fn public_output(
        &self,
        call: &FunctionToolCall,
        _output: &ToolOutput,
        status: GatewayCallStatus,
        _params: &WebSearchToolParam,
    ) -> Option<OutputItem> {
        Some(OutputItem::WebSearchCall(
            WebSearchCall::try_new(&call.id, status, vec!["local".to_owned()], Vec::new()).unwrap(),
        ))
    }
}

struct LocalModel {
    exec: ExecutionContext,
    requests: Arc<Mutex<Vec<Value>>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl LocalModel {
    async fn start(responses: Vec<Value>) -> Self {
        let responses = Arc::new(Mutex::new(VecDeque::from(responses)));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let app = Router::new().route(
            "/v1/responses",
            post(move |Json(request): Json<Value>| {
                let responses = Arc::clone(&responses);
                let captured = Arc::clone(&captured);
                async move {
                    captured.lock().await.push(request);
                    Json(
                        responses
                            .lock()
                            .await
                            .pop_front()
                            .expect("planned local model response"),
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let exec = ExecutionContext::new(
            ConversationHandler::new(ConversationStore::disabled()),
            ResponseHandler::new(ResponseStore::disabled()),
            Arc::new(reqwest::Client::new()),
            format!("http://{address}"),
        );
        Self {
            exec,
            requests,
            handle: Some(handle),
        }
    }

    async fn close(mut self) {
        let handle = self.handle.take().unwrap();
        handle.abort();
        assert!(handle.await.unwrap_err().is_cancelled());
    }
}

impl Drop for LocalModel {
    fn drop(&mut self) {
        if let Some(handle) = &self.handle {
            handle.abort();
        }
    }
}

fn model_message(id: &str, text: &str) -> Value {
    json!({"id": id, "object": "response", "created_at": 0, "model": "test-model", "status": "completed",
        "output": [{"type":"message", "id":format!("msg_{id}"), "role":"assistant", "status":"completed",
            "content":[{"type":"output_text", "text":text, "annotations":[]}]}]})
}

fn search_round(id: &str, text: &str) -> Value {
    let mut response = model_message(id, text);
    response["output"].as_array_mut().unwrap().push(json!({
        "type":"function_call", "id":format!("fc_{id}"), "call_id":format!("call_{id}"),
        "name":"web_search", "arguments":"{\"query\":\"local\"}", "status":"completed"
    }));
    response
}

fn with_local_search(model: &mut LocalModel) -> Arc<AtomicUsize> {
    let calls = Arc::new(AtomicUsize::new(0));
    model.exec = model.exec.clone().with_gateway_executor(Arc::new(LocalSearch {
        calls: Arc::clone(&calls),
    }));
    calls
}

fn search_request() -> RequestPayload {
    let mut payload = request(None, json!("short"));
    payload.tools = Some(serde_json::from_value(json!([{"type":"web_search_preview"}])).unwrap());
    payload
}

async fn execute_local(model: &LocalModel, session: &ResponseSession, payload: RequestPayload) -> ResponsePayload {
    let result = ExecuteRequest::new(payload, Arc::new(model.exec.clone()))
        .with_session(session)
        .unwrap()
        .run()
        .await
        .unwrap();
    let either::Either::Left(payload) = result else {
        panic!("expected blocking response")
    };
    payload
}

async fn storage_pool() -> Arc<agentic_core::storage::DbPool> {
    let base_url = match std::env::var("AGENTIC_SESSION_TEST_POSTGRES_URL") {
        Ok(url) => url,
        Err(std::env::VarError::NotPresent) => {
            return create_pool_with_schema(Some("sqlite::memory:")).await.unwrap();
        }
        Err(error) => panic!("invalid PostgreSQL test configuration: {error}"),
    };
    let mut database_url = reqwest::Url::parse(&base_url).expect("valid PostgreSQL fixture URL");
    assert!(matches!(database_url.scheme(), "postgres" | "postgresql"));
    let admin = agentic_core::storage::create_pool(Some(&base_url)).await.unwrap();
    // UUID-only identifiers are safe here; each case owns a new database and
    // never truncates or drops any database supplied by the caller.
    let database = format!("session_{}", uuid::Uuid::now_v7().simple());
    sqlx::query(&format!("CREATE DATABASE {database}"))
        .execute(admin.as_ref())
        .await
        .unwrap();
    admin.close().await;
    database_url.set_path(&database);
    let pool = agentic_core::storage::create_pool_with_schema_and_configs(
        Some(database_url.as_str()),
        agentic_core::config::SqliteConfig::default(),
        agentic_core::config::PostgresConfig {
            max_connections: 1,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    eprintln!("PostgreSQL session fixture ready: {database}");
    pool
}

fn execution() -> ExecutionContext {
    ExecutionContext::new(
        ConversationHandler::new(ConversationStore::disabled()),
        ResponseHandler::new(ResponseStore::disabled()),
        Arc::new(reqwest::Client::new()),
        "http://127.0.0.1:1".to_owned(),
    )
}

fn session() -> ResponseSession {
    ResponseSession::new(NonZeroUsize::new(100).unwrap(), NonZeroUsize::new(100_000).unwrap())
}

fn request(previous: Option<&str>, input: Value) -> RequestPayload {
    let mut value = json!({"model": "test-model", "store": false, "previous_response_id": previous});
    value["input"] = input;
    serde_json::from_value(value).unwrap()
}

fn response(ctx: &RequestContext, output: Value) -> ResponsePayload {
    let mut value = json!({
        "id": ctx.response_id, "object": "response", "created_at": 0,
        "model": "test-model", "status": "completed"
    });
    value["output"] = output;
    serde_json::from_value(value).unwrap()
}

fn function_call(id: &str) -> Value {
    json!({"type": "function_call", "id": format!("fc_{id}"), "call_id": id,
        "name": "weather", "arguments": "{}", "status": "completed"})
}

#[tokio::test]
async fn session_explicit_conversation_generated_turns_match_no_session_history() {
    for use_session in [false, true] {
        for store in [false, true] {
            let mut model = LocalModel::start(vec![
                model_message("resp_first", "first answer"),
                model_message("resp_second", "second answer"),
            ])
            .await;
            let pool = storage_pool().await;
            let conversations = ConversationStore::new(pool.clone());
            model.exec.conv_handler = ConversationHandler::new(conversations.clone());
            model.exec.resp_handler = ResponseHandler::new(ResponseStore::new(Arc::clone(&pool)));
            let conversation = model.exec.conv_handler.create().await.unwrap();
            let session = session();
            for input in ["first question", "second question"] {
                let mut payload = request(None, json!(input));
                payload.conversation_id = Some(conversation.conversation_id.clone());
                payload.store = store;
                let execution = ExecuteRequest::new(payload, Arc::new(model.exec.clone()));
                let execution = if use_session {
                    execution.with_session(&session).unwrap()
                } else {
                    execution
                };
                let result = execution.run().await.unwrap();
                assert!(matches!(result, either::Either::Left(_)));
                session.wait_until_idle().await.unwrap();
            }
            let requests = model.requests.lock().await;
            let input = requests[1]["input"].as_array().unwrap();
            assert_eq!(input.len(), 3, "session={use_session}, store={store}: {input:?}");
            assert!(input[0].to_string().contains("first question"));
            assert!(input[1].to_string().contains("first answer"));
            assert!(input[2].to_string().contains("second question"));
            drop(requests);
            let history = conversations.rehydrate(&conversation.conversation_id).await.unwrap();
            assert_eq!(history.len(), 4, "session={use_session}, store={store}");
            model.close().await;
            pool.close().await;
        }
    }
}

async fn explicit_conversation_tool_history(use_session: bool, gateway_tool: bool, store: bool) -> (Value, Value) {
    let first_response = if gateway_tool {
        search_round("resp_search", "before search")
    } else {
        json!({"id":"resp_call", "object":"response", "created_at":0, "model":"test-model",
            "status":"completed", "output":[function_call("client")]})
    };
    let mut responses = vec![first_response];
    if gateway_tool {
        responses.push(model_message("resp_first", "first answer"));
    }
    responses.push(model_message("resp_second", "second answer"));
    let mut model = LocalModel::start(responses).await;
    let calls = with_local_search(&mut model);
    let pool = storage_pool().await;
    let conversations = ConversationStore::new(Arc::clone(&pool));
    model.exec.conv_handler = ConversationHandler::new(conversations.clone());
    model.exec.resp_handler = ResponseHandler::new(ResponseStore::new(Arc::clone(&pool)));
    let conversation = model.exec.conv_handler.create().await.unwrap();
    let session = session();
    let next_input = if gateway_tool {
        json!("second question")
    } else {
        json!([{"type":"function_call_output", "call_id":"client", "output":"sunny"}])
    };
    for (index, input) in [json!("first question"), next_input].into_iter().enumerate() {
        let mut payload = request(None, input);
        payload.conversation_id = Some(conversation.conversation_id.clone());
        payload.store = store;
        if index == 0 {
            payload.tools = Some(if gateway_tool {
                serde_json::from_value(json!([{"type":"web_search_preview"}])).unwrap()
            } else {
                serde_json::from_value(json!([{
                    "type":"function", "name":"weather", "parameters":{"type":"object", "properties":{}}
                }]))
                .unwrap()
            });
        }
        let execution = ExecuteRequest::new(payload, Arc::new(model.exec.clone()));
        let execution = if use_session {
            execution.with_session(&session).unwrap()
        } else {
            execution
        };
        let result = execution.run().await.unwrap();
        assert!(matches!(result, either::Either::Left(_)));
        session.wait_until_idle().await.unwrap();
    }
    assert_eq!(calls.load(Ordering::Relaxed), usize::from(gateway_tool));
    let requests = model.requests.lock().await;
    assert_eq!(requests.len(), if gateway_tool { 3 } else { 2 });
    let model_history = requests.last().unwrap()["input"].clone();
    drop(requests);
    let history = conversations.rehydrate(&conversation.conversation_id).await.unwrap();
    let stored_history = serde_json::to_value(InOutItem::into_input_items(history)).unwrap();
    model.close().await;
    pool.close().await;
    (model_history, stored_history)
}

#[tokio::test]
async fn session_explicit_conversation_tool_turns_match_no_session_history() {
    for gateway_tool in [false, true] {
        for store in [false, true] {
            let baseline = explicit_conversation_tool_history(false, gateway_tool, store).await;
            let session = explicit_conversation_tool_history(true, gateway_tool, store).await;
            assert_eq!(session, baseline, "gateway_tool={gateway_tool}, store={store}");
        }
    }
}

#[tokio::test]
async fn grouped_session_fork_pins_parent_while_source_advances() {
    let exec = execution();
    let group = ResponseSessionGroup::new(
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(100).unwrap(),
        NonZeroUsize::new(100_000).unwrap(),
        NonZeroUsize::new(1_000_000).unwrap(),
    );
    let source = group.new_session().unwrap();
    let fork = group.new_session().unwrap();
    let ctx = rehydrate_in_session(request(None, json!("source prompt")), &exec, &source)
        .await
        .unwrap();
    let parent_id = ctx.response_id.clone();
    let output = response(&ctx, json!([]));
    commit(ctx, output, &exec).await.unwrap();

    let fork_ctx = rehydrate_in_session(request(Some(&parent_id), json!("fork prompt")), &exec, &fork)
        .await
        .expect("a grouped session can fork the cached source parent");
    let fork_id = fork_ctx.response_id.clone();
    let source_ctx = rehydrate_in_session(request(Some(&parent_id), json!("source advanced")), &exec, &source)
        .await
        .expect("source and fork can execute concurrently");
    let source_id = source_ctx.response_id.clone();
    let output = response(&source_ctx, json!([]));
    commit(source_ctx, output, &exec).await.unwrap();
    let output = response(&fork_ctx, json!([]));
    commit(fork_ctx, output, &exec).await.unwrap();

    for (session, id, own_text, other_text) in [
        (&source, &source_id, "source advanced", "fork prompt"),
        (&fork, &fork_id, "fork prompt", "source advanced"),
    ] {
        let ctx = rehydrate_in_session(request(Some(id), json!("next")), &exec, session)
            .await
            .unwrap();
        let history = serde_json::to_value(&ctx.enriched_request.input).unwrap();
        assert_eq!(history.as_array().unwrap().len(), 3);
        assert!(history.to_string().contains("source prompt"));
        assert!(history.to_string().contains(own_text));
        assert!(!history.to_string().contains(other_text));
    }
}

#[tokio::test]
async fn grouped_session_failed_fork_preserves_source_tool_continuation() {
    let exec = execution();
    let group = ResponseSessionGroup::new(
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(100).unwrap(),
        NonZeroUsize::new(100_000).unwrap(),
        NonZeroUsize::new(1_000_000).unwrap(),
    );
    let source = group.new_session().unwrap();
    let fork = group.new_session().unwrap();
    let parent = parent_with_call(&exec, &source).await;
    let error = rehydrate_in_session(request(Some(&parent), json!("missing output")), &exec, &fork)
        .await
        .unwrap_err();
    assert!(matches!(error, ExecutorError::Tool(_)));
    let ctx = resolved_continuation(&exec, &source, &parent).await;
    let output = response(&ctx, json!([]));
    commit(ctx, output, &exec).await.unwrap();
}

#[tokio::test]
async fn grouped_session_late_fork_does_not_restore_an_evicted_transient_parent() {
    let exec = execution();
    let group = ResponseSessionGroup::new(
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(100).unwrap(),
        NonZeroUsize::new(100_000).unwrap(),
        NonZeroUsize::new(1_000_000).unwrap(),
    );
    let source = group.new_session().unwrap();
    let fork = group.new_session().unwrap();
    let parent = parent_with_call(&exec, &source).await;
    rehydrate_in_session(request(Some(&parent), json!("missing output")), &exec, &source)
        .await
        .unwrap_err();
    let error = rehydrate_in_session(
        request(
            Some(&parent),
            json!([{"type":"function_call_output", "call_id":"call_first", "output":"sunny"}]),
        ),
        &exec,
        &fork,
    )
    .await
    .unwrap_err();
    assert!(matches!(error, ExecutorError::PreviousResponseNotFound { .. }));
}

#[tokio::test]
async fn grouped_session_budget_failure_does_not_evict_the_source_parent() {
    let exec = execution();
    let group = ResponseSessionGroup::new(
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(3).unwrap(),
        NonZeroUsize::new(100_000).unwrap(),
        NonZeroUsize::new(1_000_000).unwrap(),
    );
    let source = group.new_session().unwrap();
    let fork = group.new_session().unwrap();
    let parent = parent_with_call(&exec, &source).await;
    let ctx = resolved_continuation(&exec, &fork, &parent).await;
    let output = response(
        &ctx,
        json!([{
            "type":"message", "id":"msg_fork", "role":"assistant", "status":"completed",
            "content":[{"type":"output_text", "text":"fork answer", "annotations":[]}]
        }]),
    );
    let error = commit(ctx, output, &exec).await.unwrap_err();
    assert!(matches!(error, ExecutorError::PayloadTooLarge(_)));
    let ctx = resolved_continuation(&exec, &source, &parent).await;
    let output = response(&ctx, json!([]));
    commit(ctx, output, &exec).await.unwrap();
    rehydrate_in_session(request(None, json!("fresh fork")), &exec, &fork)
        .await
        .expect("failed publication must release the fork's execution slot");
}

#[tokio::test]
async fn session_cache_miss_does_not_mask_disabled_durable_storage() {
    let exec = execution();
    let session = session();
    let mut payload = request(Some("resp_missing"), json!("next"));
    let missing = rehydrate_in_session(payload.clone(), &exec, &session)
        .await
        .unwrap_err();
    assert!(matches!(missing, ExecutorError::PreviousResponseNotFound { .. }));
    payload.store = true;
    let unavailable = rehydrate_in_session(payload, &exec, &session).await.unwrap_err();
    assert!(matches!(
        unavailable,
        ExecutorError::Storage(StorageError::NotConfigured)
    ));
}

async fn parent_with_call(exec: &ExecutionContext, session: &ResponseSession) -> String {
    let ctx = rehydrate_in_session(request(None, json!("first")), exec, session)
        .await
        .unwrap();
    let payload = response(&ctx, json!([function_call("call_first")]));
    let id = ctx.response_id.clone();
    commit(ctx, payload, exec)
        .await
        .expect("initial unstored call checkpoint");
    id
}

async fn resolved_continuation(exec: &ExecutionContext, session: &ResponseSession, parent: &str) -> RequestContext {
    rehydrate_in_session(
        request(
            Some(parent),
            json!([
                {"type": "function_call_output", "call_id": "call_first", "output": "sunny"}
            ]),
        ),
        exec,
        session,
    )
    .await
    .expect("cached call output resolves its parent")
}

#[tokio::test]
async fn session_commit_accepts_a_new_call_without_reading_disabled_storage() {
    let exec = execution();
    let session = session();
    let parent = parent_with_call(&exec, &session).await;
    let ctx = resolved_continuation(&exec, &session, &parent).await;
    let payload = response(&ctx, json!([function_call("call_second")]));
    let result = commit(ctx, payload, &exec).await;
    assert!(
        result.is_ok(),
        "a cached history must not require durable storage: {result:?}"
    );
    let second = result.unwrap().id;
    let next = rehydrate_in_session(
        request(
            Some(&second),
            json!([
                {"type": "function_call_output", "call_id": "call_second", "output": "cloudy"}
            ]),
        ),
        &exec,
        &session,
    )
    .await
    .expect("second call is also retained in the checkpoint");
    let history = serde_json::to_value(&next.enriched_request.input).unwrap();
    assert_eq!(history.as_array().unwrap().len(), 5);
}

#[tokio::test]
async fn session_commit_rejects_reused_call_id_against_pinned_history() {
    let exec = execution();
    let session = session();
    let parent = parent_with_call(&exec, &session).await;
    let ctx = resolved_continuation(&exec, &session, &parent).await;
    let payload = response(&ctx, json!([function_call("call_first")]));
    let error = commit(ctx, payload, &exec).await.unwrap_err();
    assert!(
        matches!(&error, ExecutorError::InvalidRequest(message) if message.contains("continued history")),
        "must reject the repeated call, not fail on a database lookup: {error:?}"
    );
}

#[tokio::test]
async fn session_commit_custom_call_uses_the_same_cached_history_validation() {
    let exec = execution();
    let session = session();
    let parent = parent_with_call(&exec, &session).await;
    let ctx = resolved_continuation(&exec, &session, &parent).await;
    let payload = response(
        &ctx,
        json!([{"type":"custom_tool_call", "id":"ctc_second",
        "call_id":"call_second", "name":"apply_patch", "input":"patch text", "status":"completed"}]),
    );
    commit(ctx, payload, &exec)
        .await
        .expect("custom call validation uses the pinned parent too");
}

#[tokio::test]
async fn session_commit_message_only_continuation_is_a_positive_control() {
    let exec = execution();
    let session = session();
    let parent = parent_with_call(&exec, &session).await;
    let ctx = resolved_continuation(&exec, &session, &parent).await;
    let payload = response(
        &ctx,
        json!([{"type":"message", "id":"msg_done", "role":"assistant",
        "status":"completed", "content":[{"type":"output_text", "text":"done", "annotations":[]}]}]),
    );
    commit(ctx, payload, &exec)
        .await
        .expect("message-only completion does not inspect output call IDs");
}

#[tokio::test]
async fn session_automatic_compaction_fits_the_replaced_history_budget() {
    let model = LocalModel::start(vec![
        model_message("resp_old", "obsolete assistant detail"),
        model_message("resp_summary", "compact summary"),
        model_message("resp_compacted", "new answer"),
    ])
    .await;
    let session = ResponseSession::new(NonZeroUsize::new(4).unwrap(), NonZeroUsize::new(100_000).unwrap());
    let first = execute_local(&model, &session, request(None, json!("first user"))).await;
    let mut payload = request(Some(&first.id), json!("second user"));
    payload.context_management =
        Some(serde_json::from_value(json!([{"type":"compaction", "compact_threshold":1}])).unwrap());
    let second = execute_local(&model, &session, payload).await;
    let next = rehydrate_in_session(request(Some(&second.id), json!("third user")), &model.exec, &session)
        .await
        .unwrap();
    let history = serde_json::to_value(&next.enriched_request.input).unwrap();
    assert_eq!(
        history.as_array().unwrap().len(),
        5,
        "four retained items plus the new user item"
    );
    assert!(!history.to_string().contains("obsolete assistant detail"));
    assert_eq!(
        history[2]["type"], "compaction",
        "retain canonical items, not model-normalized summaries"
    );
    assert_eq!(model.requests.lock().await.len(), 3);
    model.close().await;
}

#[tokio::test]
async fn session_compaction_trigger_replaces_history_before_retention() {
    let model = LocalModel::start(vec![
        model_message("resp_old", "obsolete assistant detail"),
        model_message("resp_summary", "compact summary"),
    ])
    .await;
    let session = ResponseSession::new(NonZeroUsize::new(2).unwrap(), NonZeroUsize::new(100_000).unwrap());
    let first = execute_local(&model, &session, request(None, json!("first user"))).await;
    let second = execute_local(
        &model,
        &session,
        request(Some(&first.id), json!([{"type":"compaction_trigger"}])),
    )
    .await;
    assert_eq!(serde_json::to_value(&second.output).unwrap()[0]["type"], "compaction");
    let next = rehydrate_in_session(request(Some(&second.id), json!("second user")), &model.exec, &session)
        .await
        .unwrap();
    let history = serde_json::to_value(&next.enriched_request.input).unwrap();
    assert_eq!(history.as_array().unwrap().len(), 3);
    assert!(!history.to_string().contains("obsolete assistant detail"));
    assert_eq!(model.requests.lock().await.len(), 2);
    model.close().await;
}

#[tokio::test]
async fn session_compacted_transient_parent_promotes_only_the_canonical_window() {
    let mut model = LocalModel::start(vec![
        model_message("resp_old", "obsolete assistant detail"),
        model_message("resp_summary", "compact summary"),
        model_message("resp_compacted", "new answer"),
    ])
    .await;
    let pool = storage_pool().await;
    let store = ResponseStore::new(Arc::clone(&pool));
    model.exec.resp_handler = ResponseHandler::new(store.clone());
    let session = ResponseSession::new(NonZeroUsize::new(4).unwrap(), NonZeroUsize::new(100_000).unwrap());
    let first = execute_local(&model, &session, request(None, json!("first user"))).await;
    let mut payload = request(Some(&first.id), json!("second user"));
    payload.store = true;
    payload.context_management =
        Some(serde_json::from_value(json!([{"type":"compaction", "compact_threshold":1}])).unwrap());
    let second = execute_local(&model, &session, payload).await;
    assert!(store.get(&first.id).await.unwrap_err().is_not_found());
    let durable =
        serde_json::to_value(InOutItem::into_input_items(store.rehydrate(&second.id).await.unwrap())).unwrap();
    assert_eq!(durable.as_array().unwrap().len(), 4);
    assert!(!durable.to_string().contains("obsolete assistant detail"));
    let fresh = ResponseSession::new(NonZeroUsize::new(4).unwrap(), NonZeroUsize::new(100_000).unwrap());
    let mut continuation = request(Some(&second.id), json!("third user"));
    continuation.store = true;
    let restored = rehydrate_in_session(continuation, &model.exec, &fresh).await.unwrap();
    assert_eq!(
        serde_json::to_value(&restored.enriched_request.input)
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        5
    );
    drop(restored);
    model.close().await;
    pool.close().await;
}

#[tokio::test]
async fn session_failed_response_evicts_only_a_referenced_parent() {
    for references_parent in [false, true] {
        let failed = json!({"id":"resp_failed", "object":"response", "created_at":0, "model":"test-model",
            "status":"failed", "output":[], "error":{"code":"upstream_error", "message":"failed"}});
        let model = LocalModel::start(vec![model_message("resp_first", "first answer"), failed]).await;
        let session = session();
        let first = execute_local(&model, &session, request(None, json!("first user"))).await;
        let parent = references_parent.then_some(first.id.as_str());
        let failed = execute_local(&model, &session, request(parent, json!("failed request"))).await;
        // The existing core status enum normalizes upstream "failed" to "error".
        assert_eq!(failed.status, "error");
        assert_eq!(failed.error.as_ref().unwrap()["code"], "upstream_error");
        let result = rehydrate_in_session(request(Some(&first.id), json!("retry")), &model.exec, &session).await;
        if references_parent {
            assert!(matches!(result, Err(ExecutorError::PreviousResponseNotFound { .. })));
        } else {
            assert!(result.is_ok(), "an unrelated failed turn must not erase the checkpoint");
        }
        drop(result);
        assert!(matches!(
            rehydrate_in_session(
                request(Some(&failed.id), json!("retry failed id")),
                &model.exec,
                &session
            )
            .await,
            Err(ExecutorError::PreviousResponseNotFound { .. })
        ));
        model.close().await;
    }
}

#[tokio::test]
async fn session_incomplete_response_keeps_partial_output_for_continuation() {
    let mut incomplete = model_message("resp_partial", "partial answer");
    incomplete["status"] = json!("incomplete");
    incomplete["incomplete_details"] = json!({"reason":"max_output_tokens"});
    let model = LocalModel::start(vec![incomplete]).await;
    let session = session();
    let first = execute_local(&model, &session, request(None, json!("first user"))).await;
    assert_eq!(first.status, "incomplete");
    let next = rehydrate_in_session(request(Some(&first.id), json!("continue")), &model.exec, &session)
        .await
        .unwrap();
    let history = serde_json::to_value(&next.enriched_request.input).unwrap();
    assert_eq!(history.as_array().unwrap().len(), 3);
    assert!(history[1].to_string().contains("partial answer"));
    model.close().await;
}

#[tokio::test]
async fn session_tool_round_compaction_does_not_restore_superseded_output() {
    for store_durable in [false, true] {
        let obsolete = "obsolete intermediate assistant detail ".repeat(20);
        let mut first_round = model_message("resp_tool", &obsolete);
        first_round["output"].as_array_mut().unwrap().push(json!({
            "type":"function_call", "id":"fc_search", "call_id":"call_search",
            "name":"web_search", "arguments":"{\"query\":\"local\"}", "status":"completed"
        }));
        let mut model = LocalModel::start(vec![
            first_round,
            model_message("resp_summary", "compact summary"),
            model_message("resp_answer", "final answer"),
        ])
        .await;
        let calls = Arc::new(AtomicUsize::new(0));
        model.exec = model.exec.clone().with_gateway_executor(Arc::new(LocalSearch {
            calls: Arc::clone(&calls),
        }));
        let pool = storage_pool().await;
        let store = ResponseStore::new(pool);
        model.exec.resp_handler = ResponseHandler::new(store.clone());
        let session = session();
        let mut payload = request(None, json!("short"));
        payload.store = store_durable;
        payload.tools = Some(serde_json::from_value(json!([{"type":"web_search_preview"}])).unwrap());
        payload.context_management =
            Some(serde_json::from_value(json!([{"type":"compaction", "compact_threshold":50}])).unwrap());
        let result = execute_local(&model, &session, payload).await;
        let public_output = serde_json::to_value(&result.output).unwrap();
        assert!(public_output.to_string().contains(&obsolete));
        assert!(public_output.to_string().contains("final answer"));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        {
            let requests = model.requests.lock().await;
            assert_eq!(requests.len(), 3);
            assert_eq!(requests[0]["input"].as_array().unwrap().len(), 1);
            assert!(requests[1]["input"].to_string().contains(&obsolete));
            assert_eq!(requests[2]["input"].as_array().unwrap().len(), 2);
            assert!(!requests[2]["input"].to_string().contains(&obsolete));
        }
        let next = rehydrate_in_session(request(Some(&result.id), json!("next")), &model.exec, &session)
            .await
            .unwrap();
        let history = serde_json::to_value(&next.enriched_request.input).unwrap();
        assert!(
            !history.to_string().contains(&obsolete),
            "pre-compaction output must remain public without returning to retained history: {history}"
        );
        assert_eq!(history.as_array().unwrap().len(), 4);
        assert_eq!(history[1]["type"], "compaction");
        if store_durable {
            let durable = InOutItem::into_input_items(store.rehydrate(&result.id).await.unwrap());
            let durable = serde_json::to_value(durable).unwrap();
            assert_eq!(durable.as_array().unwrap().len(), 3);
            assert!(!durable.to_string().contains(&obsolete));
        } else {
            assert!(matches!(
                store.get(&result.id).await,
                Err(StorageError::NotFound { .. })
            ));
        }
        model.close().await;
    }
}

#[tokio::test]
async fn grouped_aggregate_rejection_precedes_durable_child_writes_and_preserves_other_lanes() {
    for durable_parent in [false, true] {
        let mut model = LocalModel::start(vec![
            model_message("resp_parent", "parent answer"),
            model_message("resp_large", &"large output ".repeat(400)),
            model_message("resp_recovered", "recovered"),
        ])
        .await;
        let pool = storage_pool().await;
        model.exec.resp_handler = ResponseHandler::new(ResponseStore::new(Arc::clone(&pool)));
        let group = ResponseSessionGroup::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(100).unwrap(),
            NonZeroUsize::new(100_000).unwrap(),
            NonZeroUsize::new(3_000).unwrap(),
        );
        let source = group.new_session().unwrap();
        let target = group.new_session().unwrap();
        let mut payload = request(None, json!("source"));
        payload.store = durable_parent;
        let parent = execute_local(&model, &source, payload).await;
        let target_ctx = rehydrate_in_session(request(None, json!("unrelated target")), &model.exec, &target)
            .await
            .unwrap();
        let target_id = target_ctx.response_id.clone();
        let output = response(&target_ctx, json!([]));
        commit(target_ctx, output, &model.exec).await.unwrap();
        let before: (i64, i64) =
            sqlx::query_as("SELECT (SELECT COUNT(*) FROM responses), (SELECT COUNT(*) FROM items)")
                .fetch_one(pool.as_ref())
                .await
                .unwrap();
        let mut payload = request(Some(&parent.id), json!("fork"));
        payload.store = true;
        let result = ExecuteRequest::new(payload, Arc::new(model.exec.clone()))
            .with_session(&target)
            .unwrap()
            .run()
            .await;
        assert!(matches!(result, Err(ExecutorError::PayloadTooLarge(ref message)) if message.contains("aggregate")));
        let after: (i64, i64) = sqlx::query_as("SELECT (SELECT COUNT(*) FROM responses), (SELECT COUNT(*) FROM items)")
            .fetch_one(pool.as_ref())
            .await
            .unwrap();
        assert_eq!(after, before, "aggregate admission must precede the durable write");
        let recovered = execute_local(&model, &target, request(Some(&target_id), json!("retry own target"))).await;
        assert_eq!(recovered.status, "completed");
        rehydrate_in_session(
            request(Some(&parent.id), json!("source still available")),
            &model.exec,
            &source,
        )
        .await
        .unwrap();
        assert_eq!(model.requests.lock().await.len(), 3);
        model.close().await;
    }
}

#[tokio::test]
async fn grouped_aggregate_reservation_is_rolled_back_when_durable_storage_fails() {
    let exec = execution(); // Both stores intentionally disabled.
    let group = ResponseSessionGroup::new(
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(100).unwrap(),
        NonZeroUsize::new(100_000).unwrap(),
        NonZeroUsize::new(6_000).unwrap(),
    );
    let source = group.new_session().unwrap();
    let target = group.new_session().unwrap();
    let source_ctx = rehydrate_in_session(request(None, json!("s".repeat(2_000))), &exec, &source)
        .await
        .unwrap();
    let source_id = source_ctx.response_id.clone();
    let output = response(&source_ctx, json!([]));
    commit(source_ctx, output, &exec).await.unwrap();
    let mut payload = request(None, json!("t".repeat(2_000)));
    payload.store = true;
    let ctx = rehydrate_in_session(payload, &exec, &target).await.unwrap();
    let output = response(&ctx, json!([]));
    let result = commit(ctx, output, &exec).await;
    let Err(ExecutorError::Persistence(cause)) = result else {
        panic!("expected wrapped durable-storage failure, got {result:?}");
    };
    assert!(matches!(*cause, ExecutorError::Storage(StorageError::NotConfigured)));
    // Two such checkpoints fit, but a third does not. Recovery proves that the
    // failed persistence did not leak its already-reserved candidate charge.
    let ctx = rehydrate_in_session(request(None, json!("t".repeat(2_000))), &exec, &target)
        .await
        .unwrap();
    let output = response(&ctx, json!([]));
    commit(ctx, output, &exec)
        .await
        .expect("failed durable write must return its reservation");
    rehydrate_in_session(request(Some(&source_id), json!("source survives")), &exec, &source)
        .await
        .unwrap();
}

#[tokio::test]
async fn grouped_aggregate_sql_failure_rolls_back_items_and_releases_capacity() {
    for fork in [false, true] {
        let pool = storage_pool().await;
        let store = ResponseStore::new(Arc::clone(&pool));
        let mut exec = execution();
        exec.resp_handler = ResponseHandler::new(store.clone());
        let group = ResponseSessionGroup::new(
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(100).unwrap(),
            NonZeroUsize::new(100_000).unwrap(),
            NonZeroUsize::new(6_000).unwrap(),
        );
        let source = group.new_session().unwrap();
        let destination = group.new_session().unwrap();
        let probe = group.new_session().unwrap();
        let source_ctx = rehydrate_in_session(request(None, json!("s".repeat(2_000))), &exec, &source)
            .await
            .unwrap();
        let source_id = source_ctx.response_id.clone();
        let output = response(&source_ctx, json!([]));
        commit(source_ctx, output, &exec).await.unwrap();

        let target = if fork { &destination } else { &source };
        let mut payload = request(Some(&source_id), json!("continue"));
        payload.store = true;
        let ctx = rehydrate_in_session(payload, &exec, target).await.unwrap();
        let failed_id = ctx.response_id.clone();
        // Seed only the response ID, not its items. The real persist path inserts
        // the candidate's items before hitting this duplicate response ID.
        // This tests a SQL failure after writes, not a disabled store/pool wait.
        store
            .persist(&failed_id, None, vec![], &ResponseMetadata::default())
            .await
            .unwrap();
        let existing = store.get(&failed_id).await.unwrap();
        let output = response(&ctx, json!([]));
        let result = commit(ctx, output, &exec).await;
        // The public handler translates this unique violation to Conflict.
        assert!(
            matches!(result, Err(ExecutorError::Conflict(ref message))
                if message == &format!("a turn is already stored under '{failed_id}'")),
            "must reach the duplicate-response insert: {result:?}"
        );
        tokio::time::timeout(std::time::Duration::from_secs(1), target.wait_until_idle())
            .await
            .expect("SQL failure must release the execution slot")
            .unwrap();
        let counts: (i64, i64) =
            sqlx::query_as("SELECT (SELECT COUNT(*) FROM responses), (SELECT COUNT(*) FROM items)")
                .fetch_one(pool.as_ref())
                .await
                .unwrap();
        assert_eq!(
            counts,
            (1, 0),
            "candidate items must roll back; only the seeded response remains"
        );
        let after = store.get(&failed_id).await.unwrap();
        assert_eq!(after.history_item_ids, existing.history_item_ids);
        assert_eq!(after.created_at, existing.created_at);
        assert!(matches!(
            rehydrate_in_session(request(Some(&failed_id), json!("no failed checkpoint")), &exec, target).await,
            Err(ExecutorError::PreviousResponseNotFound { .. })
        ));

        // With a leaked prepared checkpoint neither size fits. A failed fork
        // retains its source charge; a failed own continuation evicts it.
        let recovery_bytes = if fork { 2_000 } else { 4_000 };
        let recovered = rehydrate_in_session(request(None, json!("r".repeat(recovery_bytes))), &exec, &probe)
            .await
            .unwrap();
        let output = response(&recovered, json!([]));
        commit(recovered, output, &exec)
            .await
            .expect("SQL failure must return its prepared-checkpoint reservation");
        let source_result =
            rehydrate_in_session(request(Some(&source_id), json!("source check")), &exec, &source).await;
        if fork {
            let source_ctx = source_result.expect("a failed fork must preserve its source checkpoint");
            assert!(
                serde_json::to_string(&source_ctx.enriched_request.input)
                    .unwrap()
                    .contains(&"s".repeat(2_000))
            );
        } else {
            assert!(matches!(
                source_result,
                Err(ExecutorError::PreviousResponseNotFound { .. })
            ));
        }
        pool.close().await;
    }
}

#[tokio::test]
async fn grouped_aggregate_cancellation_while_waiting_for_storage_releases_capacity() {
    for fork in [false, true] {
        let pool = storage_pool().await;
        assert_eq!(pool.options().get_max_connections(), 1);
        let mut exec = execution();
        exec.resp_handler = ResponseHandler::new(ResponseStore::new(Arc::clone(&pool)));
        let group = ResponseSessionGroup::new(
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(100).unwrap(),
            NonZeroUsize::new(100_000).unwrap(),
            NonZeroUsize::new(6_000).unwrap(),
        );
        let source = group.new_session().unwrap();
        let destination = group.new_session().unwrap();
        let probe = group.new_session().unwrap();
        let source_ctx = rehydrate_in_session(request(None, json!("s".repeat(2_000))), &exec, &source)
            .await
            .unwrap();
        let source_id = source_ctx.response_id.clone();
        let output = response(&source_ctx, json!([]));
        commit(source_ctx, output, &exec).await.unwrap();

        let target = if fork { &destination } else { &source };
        let mut payload = request(Some(&source_id), json!("continue"));
        payload.store = true;
        let ctx = rehydrate_in_session(payload, &exec, target).await.unwrap();
        let output = response(&ctx, json!([]));
        // Holding the pool's only connection stops persistence before any write.
        // One poll reaches that wait after the candidate's budget reservation.
        let held_connection = pool.acquire().await.unwrap();
        let mut pending = Box::pin(commit(ctx, output, &exec));
        assert!(futures::poll!(&mut pending).is_pending());
        assert!(
            ExecuteRequest::new(request(None, json!("busy")), Arc::new(exec.clone()))
                .with_session(target)
                .is_err()
        );
        let probe_ctx = rehydrate_in_session(request(None, json!("p".repeat(2_000))), &exec, &probe)
            .await
            .unwrap();
        let output = response(&probe_ctx, json!([]));
        let pressured = commit(probe_ctx, output, &exec).await;
        assert!(
            matches!(pressured, Err(ExecutorError::PayloadTooLarge(ref message)) if message.contains("aggregate")),
            "the pending durable candidate must already hold a reservation: fork={fork}"
        );

        // Drop the actual public commit future while the pool remains occupied.
        // This is cancellation, not a storage error or a completed transaction.
        drop(pending);
        tokio::time::timeout(std::time::Duration::from_secs(1), target.wait_until_idle())
            .await
            .expect("cancelled persistence must release the execution slot")
            .unwrap();
        drop(held_connection);
        let counts: (i64, i64) =
            sqlx::query_as("SELECT (SELECT COUNT(*) FROM responses), (SELECT COUNT(*) FROM items)")
                .fetch_one(pool.as_ref())
                .await
                .unwrap();
        assert_eq!(counts, (0, 0), "cancelled persistence must not leave durable rows");

        // For a fork the source stays charged; for an own continuation it is
        // evicted. Either recovery would exceed 6000 if the candidate leaked.
        let recovery_bytes = if fork { 2_000 } else { 4_000 };
        let recovered = rehydrate_in_session(request(None, json!("r".repeat(recovery_bytes))), &exec, &probe)
            .await
            .unwrap();
        let output = response(&recovered, json!([]));
        commit(recovered, output, &exec)
            .await
            .expect("cancelled persistence must return its reserved capacity");
        let source_result =
            rehydrate_in_session(request(Some(&source_id), json!("source check")), &exec, &source).await;
        if fork {
            let source_ctx = source_result.expect("cancelling a fork must preserve its source checkpoint");
            assert!(
                serde_json::to_string(&source_ctx.enriched_request.input)
                    .unwrap()
                    .contains(&"s".repeat(2_000))
            );
        } else {
            assert!(matches!(
                source_result,
                Err(ExecutorError::PreviousResponseNotFound { .. })
            ));
        }
        pool.close().await;
    }
}

#[tokio::test]
async fn grouped_aggregate_durable_fallback_is_rejected_before_inference_and_can_recover() {
    let mut model = LocalModel::start(vec![
        model_message("resp_parent", "stored"),
        model_message("resp_new", "fresh"),
    ])
    .await;
    let pool = storage_pool().await;
    let store = ResponseStore::new(pool);
    model.exec.resp_handler = ResponseHandler::new(store.clone());
    let original = session();
    let mut payload = request(None, json!("durable input ".repeat(1_000)));
    payload.store = true;
    let parent = execute_local(&model, &original, payload).await;
    drop(original);
    let group = ResponseSessionGroup::new(
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(100).unwrap(),
        NonZeroUsize::new(100_000).unwrap(),
        NonZeroUsize::new(1_500).unwrap(),
    );
    let restored = group.new_session().unwrap();
    let mut payload = request(Some(&parent.id), json!("continue"));
    payload.store = true;
    let result = ExecuteRequest::new(payload, Arc::new(model.exec.clone()))
        .with_session(&restored)
        .unwrap()
        .run()
        .await;
    assert!(matches!(result, Err(ExecutorError::PayloadTooLarge(ref message)) if message.contains("aggregate")));
    assert_eq!(
        model.requests.lock().await.len(),
        1,
        "over-budget durable parent must not reach inference"
    );
    assert!(
        store.get(&parent.id).await.is_ok(),
        "retention failure must not delete durable history"
    );
    let recovered = execute_local(&model, &restored, request(None, json!("fresh root"))).await;
    assert_eq!(recovered.status, "completed");
    model.close().await;
}

#[tokio::test]
async fn session_budget_rejection_does_not_partially_store_a_child() {
    for durable_parent in [false, true] {
        for byte_budget in [false, true] {
            let mut model = LocalModel::start(vec![
                model_message("resp_parent", "parent answer"),
                model_message("resp_too_large", &"oversized answer ".repeat(200)),
                model_message("resp_recovery", "recovered"),
            ])
            .await;
            let pool = storage_pool().await;
            model.exec.resp_handler = ResponseHandler::new(ResponseStore::new(Arc::clone(&pool)));
            let session = ResponseSession::new(
                NonZeroUsize::new(if byte_budget { 100 } else { 3 }).unwrap(),
                NonZeroUsize::new(if byte_budget { 2_000 } else { 100_000 }).unwrap(),
            );
            let mut payload = request(None, json!("parent"));
            payload.store = durable_parent;
            let parent = execute_local(&model, &session, payload).await;
            let before: (i64, i64, i64) = sqlx::query_as(
                "SELECT (SELECT COUNT(*) FROM responses), (SELECT COUNT(*) FROM items), (SELECT COUNT(*) FROM conversations)",
            )
            .fetch_one(pool.as_ref())
            .await
            .unwrap();
            assert_eq!(before, if durable_parent { (1, 2, 0) } else { (0, 0, 0) });
            let mut payload = request(Some(&parent.id), json!("child"));
            payload.store = true;
            let result = ExecuteRequest::new(payload, Arc::new(model.exec.clone()))
                .with_session(&session)
                .unwrap()
                .run()
                .await;
            assert!(matches!(result, Err(ExecutorError::PayloadTooLarge(_))));
            let after: (i64, i64, i64) = sqlx::query_as(
                "SELECT (SELECT COUNT(*) FROM responses), (SELECT COUNT(*) FROM items), (SELECT COUNT(*) FROM conversations)",
            )
            .fetch_one(pool.as_ref())
            .await
            .unwrap();
            assert_eq!(after, before, "a failed budget check must precede durable writes");
            assert!(matches!(
                rehydrate_in_session(request(Some(&parent.id), json!("next")), &model.exec, &session).await,
                Err(ExecutorError::PreviousResponseNotFound { .. })
            ));
            let recovery = execute_local(&model, &session, request(None, json!("fresh"))).await;
            assert_eq!(
                recovery.status, "completed",
                "budget failure releases the execution slot"
            );
            model.close().await;
        }
    }
}

#[tokio::test]
async fn session_tool_round_preserves_reasoning_message_call_output_order() {
    for store_durable in [false, true] {
        let mut first_round = model_message("resp_tool", "intermediate answer");
        let output = first_round["output"].as_array_mut().unwrap();
        output.insert(
            0,
            json!({"type":"reasoning", "id":"rs_round", "summary":[],
            "content":[{"type":"reasoning_text", "text":"intermediate reasoning"}]}),
        );
        output.push(
            json!({"type":"function_call", "id":"fc_search", "call_id":"call_search",
            "name":"web_search", "arguments":"{\"query\":\"local\"}", "status":"completed"}),
        );
        let mut model = LocalModel::start(vec![first_round, model_message("resp_answer", "final answer")]).await;
        model.exec = model.exec.clone().with_gateway_executor(Arc::new(LocalSearch {
            calls: Arc::new(AtomicUsize::new(0)),
        }));
        let pool = storage_pool().await;
        let store = ResponseStore::new(pool);
        model.exec.resp_handler = ResponseHandler::new(store.clone());
        let session = session();
        let mut payload = request(None, json!("question"));
        payload.store = store_durable;
        payload.tools = Some(serde_json::from_value(json!([{"type":"web_search_preview"}])).unwrap());
        let result = execute_local(&model, &session, payload).await;
        let next = rehydrate_in_session(request(Some(&result.id), json!("next")), &model.exec, &session)
            .await
            .unwrap();
        let history = serde_json::to_value(&next.enriched_request.input).unwrap();
        let kinds = history
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["type"].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            vec![
                json!("message"),
                json!("reasoning"),
                json!("message"),
                json!("function_call"),
                json!("function_call_output"),
                json!("message"),
                json!("message")
            ],
            "canonical continuation must preserve inference-round order: {history}"
        );
        assert!(history[2].to_string().contains("intermediate answer"));
        assert!(history[5].to_string().contains("final answer"));
        if store_durable {
            let durable = InOutItem::into_input_items(store.rehydrate(&result.id).await.unwrap());
            let durable = serde_json::to_value(durable).unwrap();
            assert_eq!(durable, Value::Array(history.as_array().unwrap()[..6].to_vec()));
        }
        model.close().await;
    }
}

#[tokio::test]
async fn session_mixed_tool_round_records_calls_once_before_client_continuation() {
    let mut first_round = search_round("search", "before the calls");
    first_round["output"]
        .as_array_mut()
        .unwrap()
        .push(function_call("client"));
    let mut model = LocalModel::start(vec![first_round]).await;
    let calls = with_local_search(&mut model);
    let session = session();
    let mut payload = search_request();
    payload.tools.as_mut().unwrap().push(
        serde_json::from_value(json!({
            "type":"function", "name":"weather", "parameters":{"type":"object", "properties":{}}
        }))
        .unwrap(),
    );
    let result = execute_local(&model, &session, payload).await;
    assert_eq!(result.output.len(), 3);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    let next = rehydrate_in_session(
        request(
            Some(&result.id),
            json!([{"type":"function_call_output", "call_id":"client", "output":"sunny"}]),
        ),
        &model.exec,
        &session,
    )
    .await
    .expect("the client call is retained exactly once despite the mixed round");
    let history = serde_json::to_value(&next.enriched_request.input).unwrap();
    assert_eq!(history.as_array().unwrap().len(), 6);
    assert!(history[1].to_string().contains("before the calls"));
    assert_eq!(history[2]["call_id"], "call_search");
    assert_eq!(history[3]["call_id"], "client");
    assert_eq!(history[4]["type"], "function_call_output");
    assert_eq!(history[4]["call_id"], "call_search");
    assert_eq!(history[5]["call_id"], "client");
    model.close().await;
}

#[tokio::test]
async fn session_round_limit_retains_the_final_call_output_in_order() {
    let mut model = LocalModel::start(
        (0..10)
            .map(|round| search_round(&format!("round_{round}"), "intermediate"))
            .collect(),
    )
    .await;
    let calls = with_local_search(&mut model);
    let session = session();
    let result = execute_local(&model, &session, search_request()).await;
    assert_eq!(result.status, "incomplete");
    assert_eq!(calls.load(Ordering::Relaxed), 10);
    let next = rehydrate_in_session(request(Some(&result.id), json!("continue")), &model.exec, &session)
        .await
        .expect("round exhaustion must not leave an unanswered built-in call");
    let history = serde_json::to_value(&next.enriched_request.input).unwrap();
    let items = history.as_array().unwrap();
    assert_eq!(items.len(), 32);
    for (round, items) in items[1..31].chunks_exact(3).enumerate() {
        assert_eq!(items[0]["type"], "message");
        assert_eq!(items[1]["type"], "function_call");
        assert_eq!(items[2]["type"], "function_call_output");
        assert_eq!(items[1]["call_id"], format!("call_round_{round}"));
        assert_eq!(items[2]["call_id"], items[1]["call_id"]);
    }
    model.close().await;
}

#[tokio::test]
async fn session_repeated_in_round_compaction_retains_only_the_latest_window() {
    let mut model = LocalModel::start(vec![
        search_round("first", &"obsolete first round ".repeat(30)),
        model_message("resp_summary_one", "first summary"),
        search_round("second", &"obsolete second round ".repeat(30)),
        model_message("resp_summary_two", "second summary"),
        model_message("resp_final", "final answer"),
    ])
    .await;
    let calls = with_local_search(&mut model);
    let session = ResponseSession::new(NonZeroUsize::new(3).unwrap(), NonZeroUsize::new(100_000).unwrap());
    let mut payload = search_request();
    payload.context_management =
        Some(serde_json::from_value(json!([{"type":"compaction", "compact_threshold":50}])).unwrap());
    let result = execute_local(&model, &session, payload).await;
    assert_eq!(
        result.output.len(),
        5,
        "both intermediate messages and search calls remain public"
    );
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(model.requests.lock().await.len(), 5);
    let next = rehydrate_in_session(request(Some(&result.id), json!("next")), &model.exec, &session)
        .await
        .unwrap();
    let history = serde_json::to_value(&next.enriched_request.input).unwrap();
    assert_eq!(history.as_array().unwrap().len(), 4);
    assert_eq!(history[1]["type"], "compaction");
    assert_eq!(history[1]["encrypted_content"], "second summary");
    assert!(history[2].to_string().contains("final answer"));
    assert!(!history.to_string().contains("obsolete"));
    assert!(!history.to_string().contains("first summary"));
    model.close().await;
}

#[tokio::test]
async fn session_compacted_durable_parent_restores_within_the_canonical_budget() {
    for (max_items, max_bytes, aggregate) in [(6, 100_000, None), (100, 3_000, None), (100, 100_000, Some(6_000))] {
        let mut model = LocalModel::start(vec![
            model_message("resp_old", &"obsolete durable assistant detail".repeat(1_000)),
            model_message("resp_summary", "compact durable summary"),
            model_message("resp_compacted", "compacted answer"),
            model_message("resp_child", "new answer"),
        ])
        .await;
        let pool = storage_pool().await;
        let store = ResponseStore::new(pool);
        model.exec.resp_handler = ResponseHandler::new(store.clone());
        let original_session = session();
        let mut payload = request(None, json!("first user"));
        payload.store = true;
        let first = execute_local(&model, &original_session, payload).await;
        let mut payload = request(Some(&first.id), json!("second user"));
        payload.store = true;
        payload.context_management =
            Some(serde_json::from_value(json!([{"type":"compaction", "compact_threshold":1}])).unwrap());
        let compacted = execute_local(&model, &original_session, payload).await;
        assert_eq!(
            store.rehydrate(&compacted.id).await.unwrap().len(),
            6,
            "durable history still references its pre-compaction parent rows"
        );
        drop(original_session);
        let group = aggregate.map(|limit| {
            ResponseSessionGroup::new(
                NonZeroUsize::new(1).unwrap(),
                NonZeroUsize::new(max_items).unwrap(),
                NonZeroUsize::new(max_bytes).unwrap(),
                NonZeroUsize::new(limit).unwrap(),
            )
        });
        let restored = match &group {
            Some(group) => group.new_session().unwrap(),
            None => ResponseSession::new(
                NonZeroUsize::new(max_items).unwrap(),
                NonZeroUsize::new(max_bytes).unwrap(),
            ),
        };
        let mut payload = request(Some(&compacted.id), json!("third user"));
        payload.store = true;
        let child = execute_local(&model, &restored, payload).await;
        let ctx = rehydrate_in_session(request(Some(&child.id), json!("next")), &model.exec, &restored)
            .await
            .unwrap();
        let history = serde_json::to_value(&ctx.enriched_request.input).unwrap();
        assert_eq!(history.as_array().unwrap().len(), 7);
        assert_eq!(history[2]["type"], "compaction");
        assert!(!history.to_string().contains("obsolete durable assistant detail"));
        assert!(
            store.get(&first.id).await.is_ok(),
            "cache canonicalization must not delete durable history"
        );
        model.close().await;
    }
}
