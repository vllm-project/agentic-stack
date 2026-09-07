mod support;

use std::collections::HashMap;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use agentic_core::RequestPayload;
use agentic_core::tool::{ToolSearchState, model_visible_namespace_member_name};
use serde_json::Value;

#[derive(Clone, Copy)]
enum Projection {
    Public,
    Normalized,
}

#[derive(Debug, PartialEq, Eq)]
struct SemanticFlow {
    execution: &'static str,
    status: &'static str,
    returned_tools: Value,
    loaded_calls: Vec<LoadedCall>,
    final_text: String,
}

#[derive(Debug, PartialEq, Eq)]
struct LoadedCall {
    namespace: Option<String>,
    name: String,
    function_output: Value,
}

#[derive(Debug, PartialEq, Eq)]
struct RelevantCallLifecycle {
    event_type: String,
    status: Option<String>,
    arguments: Option<Value>,
    execution: Option<String>,
    item_id: Option<String>,
    call_id: Option<String>,
}

const OPENAI_BLOCKING_CASSETTE: &str = "tool-search-openai-reference-gpt-5.6-nonstreaming.yaml";
const OPENAI_STREAMING_CASSETTE: &str = "tool-search-openai-reference-gpt-5.6-streaming.yaml";
const GATEWAY_BLOCKING_CASSETTE: &str = "tool-search-gateway-Qwen-Qwen3.6-35B-A3B-FP8-nonstreaming.yaml";
const GATEWAY_STREAMING_CASSETTE: &str = "tool-search-gateway-Qwen-Qwen3.6-35B-A3B-FP8-streaming.yaml";
const GATEWAY_WEBSOCKET_CASSETTE: &str = "tool-search-gateway-Qwen-Qwen3.6-35B-A3B-FP8-websocket.yaml";

fn tool_search_cassette_directory() -> PathBuf {
    std::env::var_os("TOOL_SEARCH_CASSETTE_DIR").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cassettes/tool_search"),
        PathBuf::from,
    )
}

fn one_gateway_stream_cassette(directory: &Path, suffix: &str) -> PathBuf {
    let matches = fs::read_dir(directory)
        .expect("tool-search cassette directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name().and_then(|name| name.to_str()).is_some_and(|name| {
                name.starts_with("tool-search-gateway-")
                    && !name.ends_with("-nonstreaming.yaml")
                    && name.ends_with(suffix)
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        matches.len(),
        1,
        "expected one recorder-generated gateway {suffix} cassette, found {matches:?}"
    );
    matches.into_iter().next().expect("one gateway stream cassette")
}

fn relevant_call_lifecycle_from_named_sse<'a>(lines: impl IntoIterator<Item = &'a str>) -> Vec<RelevantCallLifecycle> {
    let mut call_ids_by_item_id = HashMap::<String, String>::new();

    support::named_sse_events(lines)
        .into_iter()
        .filter_map(|event| {
            let item = event.get("item").and_then(Value::as_object);
            let is_call_item = item.is_some_and(|item| {
                matches!(
                    item.get("type").and_then(Value::as_str),
                    Some("tool_search_call" | "function_call" | "custom_tool_call")
                )
            });
            if is_call_item {
                let item_id = item.and_then(|item| item.get("id")).and_then(Value::as_str);
                let call_id = item.and_then(|item| item.get("call_id")).and_then(Value::as_str);
                if let (Some(item_id), Some(call_id)) = (item_id, call_id) {
                    call_ids_by_item_id.insert(item_id.to_string(), call_id.to_string());
                }
            }

            let linked_item_id = event.get("item_id").and_then(Value::as_str);
            if !is_call_item && !linked_item_id.is_some_and(|item_id| call_ids_by_item_id.contains_key(item_id)) {
                return None;
            }

            let status = item
                .and_then(|item| item.get("status"))
                .or_else(|| event.get("status"))
                .and_then(Value::as_str)
                .map(str::to_string);
            let arguments = event
                .get("arguments")
                .or_else(|| item.and_then(|item| item.get("arguments")))
                .or_else(|| event.get("delta"))
                .cloned();
            let execution = item
                .and_then(|item| item.get("execution"))
                .or_else(|| event.get("execution"))
                .and_then(Value::as_str)
                .map(str::to_string);
            let call_id = item
                .and_then(|item| item.get("call_id"))
                .and_then(Value::as_str)
                .or_else(|| event.get("call_id").and_then(Value::as_str))
                .map(str::to_string)
                .or_else(|| linked_item_id.and_then(|item_id| call_ids_by_item_id.get(item_id).cloned()));
            let item_id = item
                .and_then(|item| item.get("id"))
                .and_then(Value::as_str)
                .or(linked_item_id)
                .map(str::to_string);

            Some(RelevantCallLifecycle {
                event_type: event["type"]
                    .as_str()
                    .expect("named SSE event should have a type")
                    .to_string(),
                status,
                arguments,
                execution,
                item_id,
                call_id,
            })
        })
        .collect()
}

fn non_empty_call_id(item: &Value) -> &str {
    let call_id = item["call_id"].as_str().expect("tool item should have a call_id");
    assert!(!call_id.trim().is_empty(), "tool item call_id should not be empty");
    call_id
}

fn terminal_response(turn: &support::Turn) -> Value {
    if let Some(body) = &turn.response.body {
        return body.clone();
    }
    support::recorded_named_sse_events(turn)
        .into_iter()
        .find(|event| event["type"] == "response.completed")
        .and_then(|event| event.get("response").cloned())
        .expect("streaming characterization should contain response.completed")
}

fn terminal_response_from_sse_chunks(chunks: &[&str]) -> Value {
    support::named_sse_events(chunks.iter().flat_map(|chunk| chunk.lines()))
        .into_iter()
        .find(|event| event["type"] == "response.completed")
        .and_then(|event| event.get("response").cloned())
        .expect("offline SSE should contain response.completed")
}

fn assert_stream_completed(events: &[Value]) {
    assert!(
        events
            .iter()
            .all(|event| !matches!(event["type"].as_str(), Some("error" | "response.failed"))),
        "recorded stream must not contain failure events"
    );
    let completed = events
        .iter()
        .filter(|event| event["type"] == "response.completed")
        .collect::<Vec<_>>();
    assert_eq!(completed.len(), 1, "recorded stream should complete exactly once");
    assert_eq!(completed[0]["response"]["status"], "completed");
}

fn response_lifecycle_metadata(turn: &support::Turn) -> Vec<Value> {
    support::recorded_named_sse_events(turn)
        .into_iter()
        .filter(|event| {
            matches!(
                event["type"].as_str(),
                Some("response.created" | "response.in_progress" | "response.completed")
            )
        })
        .collect()
}

fn assert_post_search_response_metadata(turn: &support::Turn, expected_tools: &Value) {
    let events = response_lifecycle_metadata(turn);
    assert_eq!(
        events
            .iter()
            .map(|event| event["type"].as_str().expect("lifecycle event type"))
            .collect::<Vec<_>>(),
        ["response.created", "response.in_progress", "response.completed"]
    );
    let expected_choice = turn
        .request
        .body
        .tool_choice
        .as_ref()
        .expect("tool-search flow turn should specify tool_choice");
    for event in events {
        assert_eq!(
            canonical_response_tools(&event["response"]["tools"]),
            *expected_tools,
            "{} must expose only public callable tools",
            event["type"]
        );
        assert_eq!(
            &event["response"]["tool_choice"], expected_choice,
            "{} must preserve the public tool choice",
            event["type"]
        );
    }
}

fn canonical_response_tools(tools: &Value) -> Value {
    let mut tools = tools.clone();
    let Some(tool_array) = tools.as_array_mut() else {
        return tools;
    };
    for tool in tool_array {
        canonical_response_tool(tool, false);
    }
    tools
}

fn canonical_response_tool(tool: &mut Value, make_callable: bool) {
    let Some(tool) = tool.as_object_mut() else {
        return;
    };
    if make_callable {
        tool.remove("defer_loading");
    }
    if tool.get("output_schema").is_some_and(Value::is_null) {
        tool.remove("output_schema");
    }
    if let Some(members) = tool.get_mut("tools").and_then(Value::as_array_mut) {
        for member in members {
            canonical_response_tool(member, make_callable);
        }
    }
}

fn expected_loaded_response_tools(directory: &Path) -> Value {
    let mut tools = fixture_json(directory, "returned_tools.json");
    for tool in tools.as_array_mut().expect("returned tools fixture") {
        canonical_response_tool(tool, true);
    }
    tools
}

fn assert_loaded_response_tool_shape(tools: &Value) {
    let tools = tools.as_array().expect("loaded response tools should be an array");
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["name"], "get_weather");
    assert!(tools[0].get("defer_loading").is_none());
    assert_eq!(tools[1]["type"], "namespace");
    assert_eq!(tools[1]["name"], "travel");
    let members = tools[1]["tools"].as_array().expect("travel namespace members");
    assert_eq!(members.len(), 1);
    assert_eq!(members[0]["type"], "function");
    assert_eq!(members[0]["name"], "get_timezone");
    assert!(members[0].get("defer_loading").is_none());
}

fn assert_observed_call_lifecycle(turn: &support::Turn) -> (String, String, Value) {
    let events = support::recorded_named_sse_events(turn);
    assert_stream_completed(&events);

    let lifecycle = relevant_call_lifecycle_from_named_sse(
        turn.response
            .sse
            .as_ref()
            .expect("streaming cassette should contain SSE")
            .iter()
            .flat_map(|entry| entry.lines()),
    );
    assert!(
        lifecycle.len() >= 4,
        "call lifecycle should include added, deltas, and done events"
    );
    assert_eq!(lifecycle[0].event_type, "response.output_item.added");
    assert_eq!(
        lifecycle.last().expect("lifecycle should not be empty").event_type,
        "response.output_item.done"
    );
    assert_eq!(
        lifecycle.last().and_then(|event| event.status.as_deref()),
        Some("completed")
    );

    let arguments_done_indices = lifecycle
        .iter()
        .enumerate()
        .filter_map(|(index, event)| (event.event_type == "response.function_call_arguments.done").then_some(index))
        .collect::<Vec<_>>();
    assert_eq!(
        arguments_done_indices.len(),
        1,
        "call lifecycle should contain exactly one arguments.done event"
    );
    let arguments_done_index = arguments_done_indices[0];
    assert!(
        arguments_done_index > 1,
        "call lifecycle should contain at least one argument delta"
    );
    assert_eq!(arguments_done_index + 1, lifecycle.len() - 1);
    assert!(
        lifecycle[1..arguments_done_index]
            .iter()
            .all(|event| event.event_type == "response.function_call_arguments.delta"),
        "only argument deltas should occur between output_item.added and arguments.done"
    );

    let aggregated_arguments = lifecycle[1..arguments_done_index]
        .iter()
        .map(|event| {
            event
                .arguments
                .as_ref()
                .and_then(Value::as_str)
                .expect("argument delta should be text")
        })
        .collect::<String>();
    let done_arguments = lifecycle[arguments_done_index]
        .arguments
        .as_ref()
        .and_then(Value::as_str)
        .expect("arguments.done should contain final arguments");
    assert_eq!(
        aggregated_arguments, done_arguments,
        "aggregated deltas should be invariant to provider chunk boundaries"
    );
    assert_eq!(
        lifecycle.last().and_then(|event| event.arguments.as_ref()),
        lifecycle[arguments_done_index].arguments.as_ref(),
        "output_item.done should repeat the completed arguments"
    );

    let item_id = lifecycle[0]
        .item_id
        .as_deref()
        .expect("output_item.added should contain an item ID");
    let call_id = lifecycle[0]
        .call_id
        .as_deref()
        .expect("output_item.added should contain a call ID");
    assert!(!item_id.trim().is_empty());
    assert!(!call_id.trim().is_empty());
    assert!(
        lifecycle.iter().all(|event| event.item_id.as_deref() == Some(item_id)),
        "all call lifecycle events should link to one item ID"
    );
    assert!(
        lifecycle.iter().all(|event| event.call_id.as_deref() == Some(call_id)),
        "all call lifecycle events should link to one call ID"
    );

    (
        item_id.to_string(),
        call_id.to_string(),
        serde_json::from_str(done_arguments).expect("completed arguments should be valid JSON"),
    )
}

fn client_calls(output: &[Value]) -> Vec<&Value> {
    output
        .iter()
        .filter(|item| {
            matches!(
                item["type"].as_str(),
                Some("tool_search_call" | "function_call" | "custom_tool_call")
            )
        })
        .collect()
}

fn normalize_search_step(response: &Value, continuation: &Value, projection: Projection) -> Value {
    let output = response["output"]
        .as_array()
        .expect("first response output should be an array");
    assert_eq!(
        client_calls(output).len(),
        1,
        "first response should contain exactly one client call"
    );
    let search_calls = output
        .iter()
        .filter(|item| match projection {
            Projection::Public => item["type"] == "tool_search_call",
            Projection::Normalized => item["type"] == "function_call" && item["name"] == "tool_search",
        })
        .collect::<Vec<_>>();
    assert_eq!(
        search_calls.len(),
        1,
        "first response should contain exactly one search call"
    );
    let search_call = search_calls[0];
    let search_call_id = non_empty_call_id(search_call);
    assert_eq!(search_call["status"], "completed");
    match projection {
        Projection::Public => {
            assert_eq!(search_call["execution"], "client");
            assert!(search_call["arguments"].as_object().is_some_and(|arguments| {
                arguments
                    .get("query")
                    .and_then(Value::as_str)
                    .is_some_and(|query| !query.is_empty())
            }));
        }
        Projection::Normalized => {
            let arguments = search_call["arguments"]
                .as_str()
                .and_then(|arguments| serde_json::from_str::<Value>(arguments).ok())
                .expect("normalized search arguments should be valid JSON");
            assert!(
                arguments["query"].as_str().is_some_and(|query| !query.is_empty()),
                "normalized search arguments should contain a query"
            );
        }
    }

    let search_outputs = continuation
        .as_array()
        .expect("search continuation should be an input array")
        .iter()
        .filter(|item| {
            item["call_id"] == search_call_id
                && match projection {
                    Projection::Public => item["type"] == "tool_search_output",
                    Projection::Normalized => item["type"] == "function_call_output",
                }
        })
        .collect::<Vec<_>>();
    assert_eq!(
        search_outputs.len(),
        1,
        "exactly one search output should link to the search call"
    );
    let search_output = search_outputs[0];
    match projection {
        Projection::Public => {
            assert_eq!(search_output["execution"], "client");
            assert_eq!(search_output["status"], "completed");
            search_output["tools"].clone()
        }
        Projection::Normalized => {
            let output = search_output["output"]
                .as_str()
                .expect("normalized search output should be JSON text");
            let decoded = serde_json::from_str::<Value>(output).expect("normalized search output should be valid JSON");
            assert_eq!(
                output,
                serde_json::to_string(&decoded).expect("normalized search output should serialize canonically"),
                "normalized search output should use canonical compact JSON"
            );
            decoded["tools"].clone()
        }
    }
}

fn returned_identity_for_call(
    loaded_call: &Value,
    returned_tools: &Value,
    projection: Projection,
) -> (Option<String>, String) {
    let raw_name = loaded_call["name"]
        .as_str()
        .expect("loaded function call should have a name");
    let raw_namespace = loaded_call.get("namespace").and_then(Value::as_str);
    let returned_tools = returned_tools
        .as_array()
        .expect("search output tools should be an array");

    if let Some(namespace) = raw_namespace {
        assert!(
            matches!(projection, Projection::Public),
            "direct-vLLM calls should use a flattened model-visible name"
        );
        let namespace_tool = returned_tools
            .iter()
            .find(|tool| tool["type"] == "namespace" && tool["name"] == namespace)
            .expect("public namespace call should come from the search output");
        assert!(
            namespace_tool["tools"]
                .as_array()
                .is_some_and(|members| members.iter().any(|member| member["name"] == raw_name)),
            "public namespace member should come from the search output"
        );
        return (Some(namespace.to_owned()), raw_name.to_owned());
    }

    if returned_tools
        .iter()
        .any(|tool| tool["type"] == "function" && tool["name"] == raw_name)
    {
        return (None, raw_name.to_owned());
    }

    assert!(
        matches!(projection, Projection::Normalized),
        "public namespace calls should preserve their namespace"
    );
    for namespace_tool in returned_tools.iter().filter(|tool| tool["type"] == "namespace") {
        let namespace = namespace_tool["name"]
            .as_str()
            .expect("returned namespace should have a name");
        for member in namespace_tool["tools"]
            .as_array()
            .expect("returned namespace should contain members")
        {
            let member_name = member["name"]
                .as_str()
                .expect("returned namespace member should have a name");
            if model_visible_namespace_member_name(namespace, member_name) == raw_name {
                return (Some(namespace.to_owned()), member_name.to_owned());
            }
        }
    }
    panic!("called function should come from the search output: {raw_name}");
}

fn normalize_loaded_step(
    response: &Value,
    continuation: &Value,
    returned_tools: &Value,
    projection: Projection,
) -> LoadedCall {
    let output = response["output"]
        .as_array()
        .expect("loaded-tool response output should be an array");
    assert_eq!(
        client_calls(output).len(),
        1,
        "each loaded-tool response should contain exactly one client call"
    );
    let loaded_calls = output
        .iter()
        .filter(|item| item["type"] == "function_call" && item["name"] != "tool_search")
        .collect::<Vec<_>>();
    assert_eq!(
        loaded_calls.len(),
        1,
        "loaded-tool response should call exactly one loaded function"
    );
    let loaded_call = loaded_calls[0];
    let loaded_call_id = non_empty_call_id(loaded_call);
    assert_eq!(loaded_call["status"], "completed");
    let (namespace, name) = returned_identity_for_call(loaded_call, returned_tools, projection);
    let loaded_arguments = loaded_call["arguments"]
        .as_str()
        .and_then(|arguments| serde_json::from_str::<Value>(arguments).ok())
        .expect("loaded function arguments should be valid JSON");
    assert_eq!(loaded_arguments, serde_json::json!({"city": "Paris"}));

    let function_outputs = continuation
        .as_array()
        .expect("function continuation should be an input array")
        .iter()
        .filter(|item| item["type"] == "function_call_output" && item["call_id"] == loaded_call_id)
        .collect::<Vec<_>>();
    assert_eq!(
        function_outputs.len(),
        1,
        "exactly one function output should link to the loaded call"
    );
    LoadedCall {
        namespace,
        name,
        function_output: function_outputs[0]["output"].clone(),
    }
}

fn normalized_final_text(response: &Value) -> String {
    let output = response["output"]
        .as_array()
        .expect("final response output should be an array");
    assert!(
        client_calls(output).is_empty(),
        "final response must not contain tool calls"
    );
    output
        .iter()
        .filter(|item| item["type"] == "message")
        .flat_map(|message| message["content"].as_array().into_iter().flatten())
        .filter(|part| part["type"] == "output_text")
        .filter_map(|part| part["text"].as_str())
        .collect::<String>()
        .trim()
        .to_owned()
}

fn normalize_flow(responses: &[Value], continuation_inputs: &[Value], projection: Projection) -> SemanticFlow {
    assert_eq!(responses.len(), 4, "tool-search characterization needs four responses");
    assert_eq!(
        continuation_inputs.len(),
        3,
        "tool-search characterization needs three continuations"
    );
    let returned_tools = normalize_search_step(&responses[0], &continuation_inputs[0], projection);
    let loaded_calls = vec![
        normalize_loaded_step(&responses[1], &continuation_inputs[1], &returned_tools, projection),
        normalize_loaded_step(&responses[2], &continuation_inputs[2], &returned_tools, projection),
    ];
    assert_eq!(
        loaded_calls
            .iter()
            .map(|call| (call.namespace.as_deref(), call.name.as_str()))
            .collect::<Vec<_>>(),
        [(None, "get_weather"), (Some("travel"), "get_timezone")],
        "the flow must call only the selected ordinary function and selected namespace member"
    );
    SemanticFlow {
        execution: "client",
        status: "completed",
        returned_tools,
        loaded_calls,
        final_text: normalized_final_text(&responses[3]),
    }
}

fn mixed_loaded_tool_definitions() -> Value {
    serde_json::json!([
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": false
            },
            "strict": true,
            "defer_loading": true
        },
        {
            "type": "namespace",
            "name": "travel",
            "description": "Travel tools",
            "tools": [{
                "type": "function",
                "name": "get_timezone",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": false
                },
                "strict": true,
                "defer_loading": true
            }]
        }
    ])
}

fn fixture_json(directory: &Path, filename: &str) -> Value {
    serde_json::from_str(
        &fs::read_to_string(directory.join(filename))
            .unwrap_or_else(|error| panic!("{filename} should be readable: {error}")),
    )
    .unwrap_or_else(|error| panic!("{filename} should be valid JSON: {error}"))
}

fn lowered_fixture_tools(tools: &Value, input: &Value) -> Value {
    let mut request: RequestPayload = serde_json::from_value(serde_json::json!({
        "model": "fixture-model",
        "input": input,
        "tools": tools,
        "store": false,
        "stream": false,
        "parallel_tool_calls": false
    }))
    .expect("fixture should deserialize as a public request");
    let mut state = ToolSearchState::build(&request).expect("fixture should build tool-search state");
    state
        .prepare_inference_request(&mut request)
        .expect("fixture should prepare private inference state");
    let upstream = request
        .to_upstream_request(false)
        .expect("fixture should lower into an upstream request");
    serde_json::to_value(upstream).expect("upstream fixture should serialize")["tools"].clone()
}

#[test]
fn mixed_catalog_fixtures_match_private_tool_search_lowering() {
    let directory = tool_search_cassette_directory();
    let public_tools = fixture_json(&directory, "openai_tools.json");
    let returned_tools = fixture_json(&directory, "returned_tools.json");
    let expected_initial = fixture_json(&directory, "vllm_initial_tools.json");
    let expected_loaded = fixture_json(&directory, "vllm_tools_after_search.json");
    let openai_tool_choices = fixture_json(&directory, "openai_tool_choice_sequence.json");
    let gateway_tool_choices = fixture_json(&directory, "gateway_tool_choice_sequence.json");

    for choice in [openai_tool_choices, gateway_tool_choices].iter().flat_map(|choices| {
        choices
            .as_array()
            .expect("public tool-choice sequence should be an array")
    }) {
        serde_json::from_value::<RequestPayload>(serde_json::json!({
            "model": "fixture-model",
            "input": "fixture input",
            "tools": public_tools,
            "tool_choice": choice,
            "parallel_tool_calls": false
        }))
        .expect("every public tool-choice fixture should match the typed request model");
    }

    assert_eq!(
        lowered_fixture_tools(&public_tools, &serde_json::json!("find weather and timezone tools")),
        expected_initial
    );
    assert_eq!(
        lowered_fixture_tools(
            &public_tools,
            &serde_json::json!([
                {
                    "type": "tool_search_call",
                    "id": "tsc_fixture",
                    "call_id": "call_fixture",
                    "execution": "client",
                    "status": "completed",
                    "arguments": {"query": "weather and timezone"}
                },
                {
                    "type": "tool_search_output",
                    "call_id": "call_fixture",
                    "execution": "client",
                    "status": "completed",
                    "tools": returned_tools
                }
            ])
        ),
        expected_loaded
    );
}

fn public_semantic_fixture(returned_tools: &Value) -> (Vec<Value>, Vec<Value>) {
    let responses = vec![
        serde_json::json!({
            "id": "resp_public_search",
            "created_at": 10,
            "usage": {"total_tokens": 50},
            "output": [{
                "id": "tsc_public",
                "type": "tool_search_call",
                "call_id": "call_public_search",
                "execution": "client",
                "status": "completed",
                "arguments": {"query": "weather tool"}
            }]
        }),
        serde_json::json!({
            "id": "resp_public_weather",
            "reasoning": {"summary": "provider noise"},
            "output": [{
                "id": "fc_public",
                "type": "function_call",
                "name": "get_weather",
                "call_id": "call_public_function",
                "status": "completed",
                "arguments": "{\"city\":\"Paris\"}"
            }]
        }),
        serde_json::json!({
            "id": "resp_public_timezone",
            "output": [{
                "id": "fc_public_timezone",
                "type": "function_call",
                "namespace": "travel",
                "name": "get_timezone",
                "call_id": "call_public_timezone",
                "status": "completed",
                "arguments": "{\"city\":\"Paris\"}"
            }]
        }),
        serde_json::json!({
            "id": "resp_public_final",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "\n\nPARIS_MIXED_TOOLS_OK"}]
            }]
        }),
    ];
    let inputs = vec![
        serde_json::json!([{
            "type": "tool_search_output",
            "call_id": "call_public_search",
            "execution": "client",
            "status": "completed",
            "tools": returned_tools
        }]),
        serde_json::json!([{
            "type": "function_call_output",
            "call_id": "call_public_function",
            "output": "weather result"
        }]),
        serde_json::json!([{
            "type": "function_call_output",
            "call_id": "call_public_timezone",
            "output": "timezone result"
        }]),
    ];
    (responses, inputs)
}

fn normalized_semantic_fixture(returned_tools: &Value) -> (Vec<Value>, Vec<Value>) {
    let responses = vec![
        serde_json::json!({
            "id": "resp_vllm_search",
            "created_at": 999,
            "usage": null,
            "output": [{
                "id": "fc_vllm_search",
                "type": "function_call",
                "name": "tool_search",
                "call_id": "call_vllm_search",
                "status": "completed",
                "arguments": "{\"query\":\"weather tool\"}"
            }]
        }),
        serde_json::json!({
            "id": "resp_vllm_weather",
            "output": [{
                "id": "fc_vllm",
                "type": "function_call",
                "name": "get_weather",
                "call_id": "call_vllm_function",
                "status": "completed",
                "arguments": "{\"city\":\"Paris\"}"
            }]
        }),
        serde_json::json!({
            "id": "resp_vllm_timezone",
            "output": [{
                "id": "fc_vllm_timezone",
                "type": "function_call",
                "name": "agentic_ns__travel__get_timezone",
                "call_id": "call_vllm_timezone",
                "status": "completed",
                "arguments": "{\"city\":\"Paris\"}"
            }]
        }),
        serde_json::json!({
            "id": "resp_vllm_final",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "PARIS_MIXED_TOOLS_OK"}]
            }]
        }),
    ];
    let inputs = vec![
        serde_json::json!([{
            "type": "function_call_output",
            "call_id": "call_vllm_search",
            "output": serde_json::to_string(&serde_json::json!({"tools": returned_tools})).unwrap()
        }]),
        serde_json::json!([{
            "type": "function_call_output",
            "call_id": "call_vllm_function",
            "output": "weather result"
        }]),
        serde_json::json!([{
            "type": "function_call_output",
            "call_id": "call_vllm_timezone",
            "output": "timezone result"
        }]),
    ];
    (responses, inputs)
}

fn normalized_manual_inputs(responses: &[Value], inputs: &[Value]) -> Vec<Value> {
    let prompts = ["call weather", "call timezone", "finish"];
    let mut history = vec![serde_json::json!({
        "type": "message",
        "role": "user",
        "content": "find weather and timezone tools"
    })];
    responses
        .iter()
        .zip(inputs)
        .zip(prompts)
        .map(|((response, input), prompt)| {
            history.extend(
                response["output"]
                    .as_array()
                    .expect("fixture response output")
                    .iter()
                    .cloned(),
            );
            history.extend(input.as_array().expect("fixture continuation input").iter().cloned());
            history.push(serde_json::json!({
                "type": "message",
                "role": "user",
                "content": prompt
            }));
            Value::Array(history.clone())
        })
        .collect()
}

fn assert_semantic_mutations_are_visible(
    public: &SemanticFlow,
    public_responses: &[Value],
    public_inputs: &[Value],
    normalized_responses: &[Value],
    normalized_inputs: &[Value],
) {
    let mut schema_mutation = normalized_inputs.to_vec();
    let mut decoded = serde_json::from_str::<Value>(schema_mutation[0][0]["output"].as_str().unwrap()).unwrap();
    decoded["tools"][0]["parameters"]["properties"]["city"]["type"] = Value::String("number".to_string());
    schema_mutation[0][0]["output"] = Value::String(serde_json::to_string(&decoded).unwrap());
    let mutated = normalize_flow(normalized_responses, &schema_mutation, Projection::Normalized);
    assert_ne!(public, &mutated, "schema mutations must remain semantically visible");

    let mut function_output_mutation = normalized_inputs.to_vec();
    function_output_mutation[1][0]["output"] = Value::String("different weather".to_string());
    let mutated = normalize_flow(normalized_responses, &function_output_mutation, Projection::Normalized);
    assert_ne!(
        public, &mutated,
        "client function-output mutations must remain semantically visible"
    );

    let mut namespace_output_mutation = normalized_inputs.to_vec();
    namespace_output_mutation[2][0]["output"] = Value::String("different timezone".to_string());
    let mutated = normalize_flow(normalized_responses, &namespace_output_mutation, Projection::Normalized);
    assert_ne!(
        public, &mutated,
        "client namespace-member output mutations must remain semantically visible"
    );

    let mut status_mutation = public_responses.to_vec();
    status_mutation[0]["output"][0]["status"] = Value::String("in_progress".to_string());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            normalize_flow(&status_mutation, public_inputs, Projection::Public)
        }))
        .is_err(),
        "public status mutation should be rejected"
    );

    for (output_index, label) in [
        (0, "normalized search"),
        (1, "normalized loaded function"),
        (2, "normalized loaded namespace member"),
    ] {
        let mut status_mutation = normalized_responses.to_vec();
        status_mutation[output_index]["output"][0]["status"] = Value::String("in_progress".to_string());
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                normalize_flow(&status_mutation, normalized_inputs, Projection::Normalized)
            }))
            .is_err(),
            "{label} status mutation should be rejected"
        );
    }

    let mut linkage_mutation = public_inputs.to_vec();
    linkage_mutation[0][0]["call_id"] = Value::String("call_wrong".to_string());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            normalize_flow(public_responses, &linkage_mutation, Projection::Public)
        }))
        .is_err(),
        "call linkage mutation should be rejected"
    );

    let mut namespace_linkage_mutation = public_inputs.to_vec();
    namespace_linkage_mutation[2][0]["call_id"] = Value::String("call_wrong".to_string());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            normalize_flow(public_responses, &namespace_linkage_mutation, Projection::Public)
        }))
        .is_err(),
        "namespace-member call linkage mutation should be rejected"
    );
}

#[test]
fn raw_semantic_normalization_ignores_provider_ids_usage_and_wire_projection() {
    let returned_tools = mixed_loaded_tool_definitions();
    let (public_responses, public_inputs) = public_semantic_fixture(&returned_tools);
    let (normalized_responses, normalized_inputs) = normalized_semantic_fixture(&returned_tools);
    let public = normalize_flow(&public_responses, &public_inputs, Projection::Public);
    let normalized = normalize_flow(&normalized_responses, &normalized_inputs, Projection::Normalized);
    assert_eq!(public, normalized);

    assert_eq!(
        normalize_flow(
            &normalized_responses,
            &normalized_manual_inputs(&normalized_responses, &normalized_inputs),
            Projection::Normalized,
        ),
        public,
        "manual full-history replay should normalize to the same semantics"
    );
    assert_semantic_mutations_are_visible(
        &public,
        &public_responses,
        &public_inputs,
        &normalized_responses,
        &normalized_inputs,
    );
}

#[test]
fn offline_sse_terminal_normalization_ignores_event_chunk_grouping() {
    let completed = serde_json::json!({
        "type": "response.completed",
        "response": {
            "id": "resp_terminal",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "PARIS_MIXED_TOOLS_OK"}]
            }]
        }
    });
    let completed_data = format!("data: {completed}");
    let grouped = format!(
        "event: response.created\ndata: {{\"type\":\"response.created\"}}\nevent: response.completed\n{completed_data}\n"
    );
    let split = [
        "event: response.created",
        "data: {\"type\":\"response.created\"}",
        "event: response.completed",
        completed_data.as_str(),
    ];

    assert_eq!(
        terminal_response_from_sse_chunks(&[grouped.as_str()]),
        terminal_response_from_sse_chunks(&split)
    );
}

#[test]
fn named_sse_call_lifecycle_projection_preserves_order_and_linkage() {
    let sse = [
        "event: response.output_item.added",
        r#"data: {"type":"response.output_item.added","item":{"id":"reasoning_1","type":"reasoning","status":"in_progress"}}"#,
        "event: response.output_item.added",
        r#"data: {"type":"response.output_item.added","item":{"id":"function_1","type":"function_call","name":"fixture_call","call_id":"call_1","status":"in_progress","arguments":""}}"#,
        "event: response.function_call_arguments.delta",
        r#"data: {"type":"response.function_call_arguments.delta","item_id":"function_1","delta":"{\"city\":"}"#,
        "event: response.function_call_arguments.done",
        r#"data: {"type":"response.function_call_arguments.done","item_id":"function_1","arguments":"{\"city\":\"Paris\"}"}"#,
        "event: response.output_item.done",
        r#"data: {"type":"response.output_item.done","item":{"id":"function_1","type":"function_call","name":"fixture_call","call_id":"call_1","status":"completed","arguments":"{\"city\":\"Paris\"}"}}"#,
    ];

    assert_eq!(
        relevant_call_lifecycle_from_named_sse(sse),
        vec![
            RelevantCallLifecycle {
                event_type: "response.output_item.added".to_string(),
                status: Some("in_progress".to_string()),
                arguments: Some(Value::String(String::new())),
                execution: None,
                item_id: Some("function_1".to_string()),
                call_id: Some("call_1".to_string()),
            },
            RelevantCallLifecycle {
                event_type: "response.function_call_arguments.delta".to_string(),
                status: None,
                arguments: Some(Value::String("{\"city\":".to_string())),
                execution: None,
                item_id: Some("function_1".to_string()),
                call_id: Some("call_1".to_string()),
            },
            RelevantCallLifecycle {
                event_type: "response.function_call_arguments.done".to_string(),
                status: None,
                arguments: Some(Value::String("{\"city\":\"Paris\"}".to_string())),
                execution: None,
                item_id: Some("function_1".to_string()),
                call_id: Some("call_1".to_string()),
            },
            RelevantCallLifecycle {
                event_type: "response.output_item.done".to_string(),
                status: Some("completed".to_string()),
                arguments: Some(Value::String("{\"city\":\"Paris\"}".to_string())),
                execution: None,
                item_id: Some("function_1".to_string()),
                call_id: Some("call_1".to_string()),
            },
        ]
    );
}

fn assert_reference_and_blocking_response_metadata(directory: &Path) -> (Value, SemanticFlow) {
    let openai_streaming = support::load_cassette(
        directory
            .join(OPENAI_STREAMING_CASSETTE)
            .to_str()
            .expect("OpenAI streaming cassette path"),
    );
    let expected_loaded_tools = expected_loaded_response_tools(directory);
    assert_loaded_response_tool_shape(&expected_loaded_tools);
    for turn in &openai_streaming.turns[1..] {
        assert_post_search_response_metadata(turn, &expected_loaded_tools);
    }

    let openai_blocking = support::load_cassette(
        directory
            .join(OPENAI_BLOCKING_CASSETTE)
            .to_str()
            .expect("OpenAI blocking cassette path"),
    );
    for turn in &openai_blocking.turns[1..] {
        let response = terminal_response(turn);
        assert_eq!(canonical_response_tools(&response["tools"]), expected_loaded_tools);
        assert_eq!(response["tool_choice"], turn.request.body.tool_choice.clone().unwrap());
    }

    let blocking = support::load_cassette(
        directory
            .join(GATEWAY_BLOCKING_CASSETTE)
            .to_str()
            .expect("blocking gateway cassette path"),
    );
    let blocking_responses = blocking.turns.iter().map(terminal_response).collect::<Vec<_>>();
    for (turn, response) in blocking.turns[1..].iter().zip(&blocking_responses[1..]) {
        assert_eq!(canonical_response_tools(&response["tools"]), expected_loaded_tools);
        assert_eq!(response["tool_choice"], turn.request.body.tool_choice.clone().unwrap());
    }
    let blocking_inputs = blocking.turns[1..]
        .iter()
        .map(|turn| turn.request.body.input.clone())
        .collect::<Vec<_>>();
    let blocking_flow = normalize_flow(&blocking_responses, &blocking_inputs, Projection::Public);
    (expected_loaded_tools, blocking_flow)
}

#[test]
fn gateway_http_sse_and_websocket_cassettes_replay_the_public_lifecycle() {
    let directory = tool_search_cassette_directory();
    let (expected_loaded_tools, blocking_flow) = assert_reference_and_blocking_response_metadata(&directory);

    for suffix in ["-streaming.yaml", "-websocket.yaml"] {
        let path = one_gateway_stream_cassette(&directory, suffix);
        let cassette = support::load_cassette(path.to_str().expect("gateway stream cassette path"));
        assert_eq!(cassette.turns.len(), 4);

        for turn in &cassette.turns {
            let events = support::recorded_named_sse_events(turn);
            assert_eq!(
                events
                    .iter()
                    .map(|event| event["sequence_number"].as_u64())
                    .collect::<Vec<_>>(),
                (0..u64::try_from(events.len()).unwrap()).map(Some).collect::<Vec<_>>()
            );
            assert!(
                events
                    .iter()
                    .all(|event| event["type"] != "error" && event["type"] != "response.failed")
            );
            assert!(events.iter().all(|event| {
                event["response"]["tools"].as_array().is_none_or(|tools| {
                    tools
                        .iter()
                        .all(|tool| !(tool["type"] == "function" && tool["name"] == "tool_search"))
                })
            }));
        }
        for turn in &cassette.turns[1..] {
            assert_post_search_response_metadata(turn, &expected_loaded_tools);
        }

        let first_events = support::recorded_named_sse_events(&cassette.turns[0]);
        assert!(first_events.iter().any(|event| {
            event["response"]["tools"].as_array().is_some_and(|tools| {
                tools
                    .iter()
                    .any(|tool| tool["type"] == "tool_search" && tool["execution"] == "client")
                    && tools
                        .iter()
                        .any(|tool| tool["name"] == "get_weather" && tool["defer_loading"] == true)
                    && tools
                        .iter()
                        .filter(|tool| tool["type"] == "function" && tool["defer_loading"] == true)
                        .count()
                        == 3
                    && tools.iter().any(|tool| {
                        tool["type"] == "namespace"
                            && tool["name"] == "travel"
                            && tool["tools"].as_array().is_some_and(|members| members.len() == 3)
                    })
            })
        }));
        assert!(first_events.iter().all(|event| {
            !(matches!(
                event["type"].as_str(),
                Some("response.function_call_arguments.delta" | "response.function_call_arguments.done")
            ) || matches!(
                event["type"].as_str(),
                Some("response.output_item.added" | "response.output_item.done")
            ) && event["item"]["type"] == "function_call"
                && event["item"]["name"] == "tool_search")
        }));
        let lifecycle = first_events
            .iter()
            .filter(|event| {
                matches!(
                    event["type"].as_str(),
                    Some("response.output_item.added" | "response.output_item.done")
                ) && event["item"]["type"] == "tool_search_call"
            })
            .collect::<Vec<_>>();
        assert_eq!(lifecycle.len(), 2);
        assert_eq!(lifecycle[0]["item"]["status"], "in_progress");
        assert_eq!(lifecycle[0]["item"]["arguments"], serde_json::json!({}));
        assert_eq!(lifecycle[1]["item"]["status"], "completed");
        assert_eq!(lifecycle[0]["item"]["id"], lifecycle[1]["item"]["id"]);
        assert_eq!(lifecycle[0]["item"]["call_id"], lifecycle[1]["item"]["call_id"]);
        assert_eq!(lifecycle[0]["output_index"], lifecycle[1]["output_index"]);

        let responses = cassette.turns.iter().map(terminal_response).collect::<Vec<_>>();
        let terminal_search = responses[0]["output"]
            .as_array()
            .expect("first gateway output")
            .iter()
            .find(|item| item["type"] == "tool_search_call")
            .expect("terminal public search call");
        assert_eq!(terminal_search, &lifecycle[1]["item"]);
        let inputs = cassette.turns[1..]
            .iter()
            .map(|turn| turn.request.body.input.clone())
            .collect::<Vec<_>>();
        assert_eq!(normalize_flow(&responses, &inputs, Projection::Public), blocking_flow);
    }
}

#[test]
fn openai_streaming_cassette_preserves_public_lifecycle_and_terminal_identity() {
    let path = tool_search_cassette_directory().join(OPENAI_STREAMING_CASSETTE);
    let cassette = support::load_cassette(path.to_str().expect("cassette path should be UTF-8"));
    assert_eq!(cassette.turns.len(), 4);

    let search_events = support::recorded_named_sse_events(&cassette.turns[0]);
    assert!(
        search_events
            .iter()
            .all(|event| !matches!(event["type"].as_str(), Some("error" | "response.failed")))
    );
    let search_lifecycle = relevant_call_lifecycle_from_named_sse(
        cassette.turns[0]
            .response
            .sse
            .as_ref()
            .expect("streaming cassette should contain SSE")
            .iter()
            .flat_map(|entry| entry.lines()),
    );
    assert_eq!(
        search_lifecycle
            .iter()
            .map(|event| event.event_type.as_str())
            .collect::<Vec<_>>(),
        ["response.output_item.added", "response.output_item.done"]
    );
    assert_eq!(search_lifecycle[0].status.as_deref(), Some("in_progress"));
    assert_eq!(search_lifecycle[1].status.as_deref(), Some("completed"));
    assert_eq!(search_lifecycle[0].execution.as_deref(), Some("client"));
    assert_eq!(search_lifecycle[1].execution.as_deref(), Some("client"));
    assert_eq!(search_lifecycle[0].arguments, Some(serde_json::json!({})));
    assert!(
        search_lifecycle[1]
            .arguments
            .as_ref()
            .and_then(|arguments| arguments["query"].as_str())
            .is_some_and(|query| !query.trim().is_empty())
    );
    assert_eq!(search_lifecycle[0].item_id, search_lifecycle[1].item_id);
    assert_eq!(search_lifecycle[0].call_id, search_lifecycle[1].call_id);

    let (loaded_item_id, loaded_call_id, loaded_arguments) = assert_observed_call_lifecycle(&cassette.turns[1]);
    assert_eq!(loaded_arguments, serde_json::json!({"city": "Paris"}));
    let (namespace_item_id, namespace_call_id, namespace_arguments) =
        assert_observed_call_lifecycle(&cassette.turns[2]);
    assert_eq!(namespace_arguments, serde_json::json!({"city": "Paris"}));

    let responses = cassette.turns.iter().map(terminal_response).collect::<Vec<_>>();
    let terminal_search_call = responses[0]["output"]
        .as_array()
        .expect("turn one terminal output should be an array")
        .iter()
        .find(|item| item["type"] == "tool_search_call")
        .expect("turn one terminal response should contain tool_search_call");
    assert_eq!(
        terminal_search_call["id"].as_str(),
        search_lifecycle[1].item_id.as_deref(),
        "OpenAI preserves the search item ID into terminal output"
    );
    assert_eq!(
        terminal_search_call["call_id"].as_str(),
        search_lifecycle[1].call_id.as_deref(),
        "OpenAI preserves the search call ID into terminal output"
    );
    assert_eq!(terminal_search_call["execution"], "client");
    assert_eq!(terminal_search_call["status"], "completed");
    assert_eq!(
        Some(&terminal_search_call["arguments"]),
        search_lifecycle[1].arguments.as_ref()
    );

    let terminal_loaded_call = responses[1]["output"]
        .as_array()
        .expect("turn two terminal output should be an array")
        .iter()
        .find(|item| item["type"] == "function_call" && item["name"] == "get_weather")
        .expect("turn two terminal response should contain get_weather");
    assert_eq!(terminal_loaded_call["id"].as_str(), Some(loaded_item_id.as_str()));
    assert_eq!(terminal_loaded_call["call_id"].as_str(), Some(loaded_call_id.as_str()));

    let terminal_namespace_call = responses[2]["output"]
        .as_array()
        .expect("turn three terminal output should be an array")
        .iter()
        .find(|item| item["type"] == "function_call" && item["namespace"] == "travel" && item["name"] == "get_timezone")
        .expect("turn three terminal response should contain travel.get_timezone");
    assert_eq!(terminal_namespace_call["id"].as_str(), Some(namespace_item_id.as_str()));
    assert_eq!(
        terminal_namespace_call["call_id"].as_str(),
        Some(namespace_call_id.as_str())
    );

    let final_events = support::recorded_named_sse_events(&cassette.turns[3]);
    assert!(
        final_events
            .iter()
            .all(|event| !matches!(event["type"].as_str(), Some("error" | "response.failed")))
    );
    let continuation_inputs = cassette.turns[1..]
        .iter()
        .map(|turn| turn.request.body.input.clone())
        .collect::<Vec<_>>();
    let semantic = normalize_flow(&responses, &continuation_inputs, Projection::Public);
    assert_eq!(semantic.final_text.trim(), "PARIS_MIXED_TOOLS_OK");
}

const PROVIDER_PARITY_CASSETTES: [&str; 5] = [
    OPENAI_BLOCKING_CASSETTE,
    OPENAI_STREAMING_CASSETTE,
    GATEWAY_BLOCKING_CASSETTE,
    GATEWAY_STREAMING_CASSETTE,
    GATEWAY_WEBSOCKET_CASSETTE,
];

#[cfg(unix)]
fn assert_cassette_is_not_executable(path: &Path, filename: &str) {
    use std::os::unix::fs::PermissionsExt as _;
    assert_eq!(
        fs::metadata(path)
            .expect("characterization cassette metadata")
            .permissions()
            .mode()
            & 0o111,
        0,
        "checked-in cassette must not be executable: {filename}"
    );
}

#[cfg(not(unix))]
fn assert_cassette_is_not_executable(_path: &Path, _filename: &str) {}

fn assert_public_request_projection(
    directory: &Path,
    filename: &str,
    request_bodies: &[&serde_json::Map<String, Value>],
    responses: &[Value],
) {
    let expected_initial = serde_json::from_str::<Value>(
        &fs::read_to_string(directory.join("openai_tools.json")).expect("OpenAI tool fixture should be readable"),
    )
    .expect("OpenAI tool fixture should be valid JSON");
    assert_eq!(request_bodies[0].get("tools"), Some(&expected_initial));
    let tool_choice_fixture = if filename.contains("openai-reference") {
        "openai_tool_choice_sequence.json"
    } else {
        "gateway_tool_choice_sequence.json"
    };
    assert_tool_choice_sequence(directory, tool_choice_fixture, request_bodies);
    if filename == GATEWAY_BLOCKING_CASSETTE {
        assert!(
            request_bodies
                .iter()
                .all(|body| body.get("store") == Some(&Value::Bool(false)))
        );
        assert!(
            request_bodies
                .iter()
                .all(|body| !body.contains_key("previous_response_id"))
        );
    } else {
        assert!(
            request_bodies
                .iter()
                .all(|body| body.get("store") == Some(&Value::Bool(true)))
        );
        for index in 1..request_bodies.len() {
            assert_eq!(
                request_bodies[index].get("previous_response_id"),
                responses[index - 1].get("id")
            );
        }
    }
    assert!(request_bodies[1..].iter().all(|body| !body.contains_key("tools")));
}

fn assert_tool_choice_sequence(directory: &Path, filename: &str, request_bodies: &[&serde_json::Map<String, Value>]) {
    let expected = fixture_json(directory, filename);
    let expected = expected
        .as_array()
        .expect("tool-choice sequence fixture should be an array");
    assert_eq!(request_bodies.len(), expected.len());
    for (body, choice) in request_bodies.iter().zip(expected) {
        assert_eq!(body.get("tool_choice"), Some(choice));
        assert_eq!(body.get("parallel_tool_calls"), Some(&Value::Bool(false)));
    }
}

fn assert_request_projection(directory: &Path, filename: &str, raw_document: &Value, responses: &[Value]) {
    let request_bodies = raw_document["turns"]
        .as_array()
        .expect("raw cassette should contain turns")
        .iter()
        .map(|turn| {
            turn["request"]["body"]
                .as_object()
                .expect("recorded request body should be an object")
        })
        .collect::<Vec<_>>();
    assert_public_request_projection(directory, filename, &request_bodies, responses);
}

fn normalize_provider_cassette(directory: &Path, filename: &str) -> SemanticFlow {
    let path = directory.join(filename);
    assert!(
        path.is_file(),
        "required provider parity cassette is missing: {filename}"
    );
    assert_cassette_is_not_executable(&path, filename);
    let raw_text = fs::read_to_string(&path).expect("characterization cassette should be readable");
    let raw_document = serde_yaml::from_str::<Value>(&raw_text).expect("characterization YAML should be valid");
    let cassette = support::load_cassette(path.to_str().expect("cassette path should be UTF-8"));
    assert_eq!(cassette.turns.len(), 4, "{filename} should contain four turns");
    let responses = cassette.turns.iter().map(terminal_response).collect::<Vec<_>>();
    let inputs = cassette.turns[1..]
        .iter()
        .map(|turn| turn.request.body.input.clone())
        .collect::<Vec<_>>();
    assert_request_projection(directory, filename, &raw_document, &responses);
    normalize_flow(&responses, &inputs, Projection::Public)
}

#[test]
fn provider_parity_recorder_generated_matrix_has_one_semantic_flow() {
    let directory = tool_search_cassette_directory();
    let expected_tools = serde_json::from_str::<Value>(
        &fs::read_to_string(directory.join("returned_tools.json")).expect("returned tool fixture should be readable"),
    )
    .expect("returned tool fixture should be valid JSON");
    let expected_outputs = serde_json::from_str::<Value>(
        &fs::read_to_string(directory.join("function_outputs.json"))
            .expect("function-output fixture should be readable"),
    )
    .expect("function-output fixture should be valid JSON");
    let mut reference = None;
    for filename in PROVIDER_PARITY_CASSETTES {
        let semantic = normalize_provider_cassette(&directory, filename);
        assert_eq!(
            semantic.returned_tools, expected_tools,
            "returned tool drift in {filename}"
        );
        assert_eq!(
            semantic.loaded_calls[0].function_output, expected_outputs["get_weather"],
            "ordinary function-output drift in {filename}"
        );
        assert_eq!(
            semantic.loaded_calls[1].function_output, expected_outputs["get_timezone"],
            "namespace-member function-output drift in {filename}"
        );
        assert_eq!(semantic.final_text.trim(), "PARIS_MIXED_TOOLS_OK");
        if let Some(expected) = &reference {
            assert_eq!(&semantic, expected, "semantic provider drift in {filename}");
        } else {
            reference = Some(semantic);
        }
    }
}
