use agentic_core::executor::accumulator::ResponseAccumulator;
use agentic_core::tool::{GatewayExecutors, ToolOwnership, ToolRegistry, ToolType};
use agentic_core::types::io::output::McpListTools;
use agentic_core::types::io::{McpCall, OutputItem};
use agentic_core::types::tools::ResponsesTool;
use serde_json::{Value, json};
use std::collections::HashMap;

mod support;

const MODEL: &str = "Qwen/Qwen3.5-35B-A3B-FP8";
const MCP_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/mcp");
const GATEWAY_MODEL_SLUG: &str = "Qwen-Qwen3.5-35B-A3B-FP8";
const OPENAI_MODEL_SLUG: &str = "gpt-4o";

fn load_mcp_cassette(filename: &str) -> support::Cassette {
    support::load_cassette(&format!("{MCP_DIR}/{filename}"))
}

fn load_scenario_pair(scenario: &str, streaming: bool) -> (support::Cassette, support::Cassette) {
    let mode = if streaming { "streaming" } else { "nonstreaming" };
    let openai = load_mcp_cassette(&format!(
        "mcp-openai-reference-counter-{scenario}-{OPENAI_MODEL_SLUG}-{mode}.yaml"
    ));
    let gateway = load_mcp_cassette(&format!(
        "mcp-gateway-counter-{scenario}-{GATEWAY_MODEL_SLUG}-{mode}.yaml"
    ));
    (openai, gateway)
}

fn native_mcp_declaration() -> ResponsesTool {
    serde_json::from_value(serde_json::json!({
        "type": "mcp",
        "server_label": "counter",
        "server_url": "http://127.0.0.1:8000/mcp",
        "allowed_tools": ["increment"],
        "require_approval": "never"
    }))
    .expect("native MCP declaration")
}

#[test]
fn native_mcp_declaration_uses_server_identity_without_a_tool_name() {
    let ResponsesTool::Mcp(param) = native_mcp_declaration() else {
        panic!("expected MCP declaration");
    };

    assert_eq!(param.server_label, "counter");
    assert_eq!(param.server_url.as_deref(), Some("http://127.0.0.1:8000/mcp"));
    assert_eq!(
        param.allowed_tools.as_deref(),
        Some(["increment".to_owned()].as_slice())
    );
    assert_eq!(param.require_approval.as_deref(), Some("never"));
}

#[test]
fn native_mcp_declaration_ignores_a_client_supplied_tool_name() {
    let tool = serde_json::from_value::<ResponsesTool>(serde_json::json!({
        "type": "mcp",
        "name": "increment",
        "server_label": "counter",
        "server_url": "http://127.0.0.1:8000/mcp"
    }))
    .expect("MCP declaration with an unknown field");

    let serialized = serde_json::to_value(tool).expect("serialized MCP declaration");
    assert_eq!(serialized["server_label"], "counter");
    assert!(serialized.get("name").is_none());
}

#[tokio::test]
async fn read_mcp_resource_function_is_client_owned() {
    let mut tools = vec![
        serde_json::from_value::<ResponsesTool>(serde_json::json!({
            "type": "function",
            "name": "read_mcp_resource",
            "description": "A client-owned function with no gateway MCP semantics",
            "parameters": {"type": "object"},
            "metadata": {
                "server_label": "repo",
                "server_url": "http://127.0.0.1:8000/mcp"
            }
        }))
        .expect("function declaration"),
    ];
    let mut executors = GatewayExecutors::default();

    let registry = ToolRegistry::build_with_handlers(&mut tools, &mut executors)
        .await
        .expect("function registry");
    let entry = registry.lookup("read_mcp_resource").expect("function registry entry");

    assert_eq!(entry.tool_type, ToolType::Function);
    assert!(matches!(entry.ownership, ToolOwnership::Client));
}

fn assert_matching_native_mcp_requests(
    openai: &support::Cassette,
    gateway: &support::Cassette,
    streaming: bool,
    allowed_tools: Option<&[&str]>,
) {
    assert_eq!(openai.turns.len(), 1);
    assert_eq!(gateway.turns.len(), 1);

    let openai_request = &openai.turns[0].request;
    let gateway_request = &gateway.turns[0].request;
    let openai_server_url = openai_request.body.tools[0]["server_url"]
        .as_str()
        .expect("OpenAI MCP server_url");
    let gateway_server_url = gateway_request.body.tools[0]["server_url"]
        .as_str()
        .expect("gateway MCP server_url");
    assert!(
        openai_server_url.starts_with("https://"),
        "OpenAI MCP cassette requires a public HTTPS server_url"
    );
    assert!(
        gateway_server_url.starts_with("http://") || gateway_server_url.starts_with("https://"),
        "gateway MCP cassette requires an HTTP(S) server_url"
    );

    for request in [openai_request, gateway_request] {
        assert_eq!(request.path, "/v1/responses");
        assert_eq!(request.body.stream, streaming);
        assert_eq!(request.body.tools.len(), 1);

        let declaration = &request.body.tools[0];
        assert_eq!(declaration["type"], "mcp");
        assert_eq!(declaration["server_label"], "counter");
        assert_eq!(declaration["require_approval"], "never");
        assert!(declaration.get("name").is_none());

        match allowed_tools {
            Some(expected) => {
                let actual = declaration["allowed_tools"]
                    .as_array()
                    .expect("allowed_tools array")
                    .iter()
                    .map(|name| name.as_str().expect("allowed tool name"))
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected);
            }
            None => assert!(declaration.get("allowed_tools").is_none()),
        }
    }

    // OpenAI requires a public HTTPS endpoint, while the gateway can use a local
    // or otherwise separately reachable endpoint. Compare the MCP declarations
    // without coupling the reference cassettes to the same recorded URL.
    let mut openai_declaration = openai_request.body.tools[0].clone();
    let mut gateway_declaration = gateway_request.body.tools[0].clone();
    openai_declaration
        .as_object_mut()
        .expect("OpenAI MCP declaration")
        .remove("server_url");
    gateway_declaration
        .as_object_mut()
        .expect("gateway MCP declaration")
        .remove("server_url");
    assert_eq!(openai_declaration, gateway_declaration);
    assert_eq!(openai_request.body.tool_choice, gateway_request.body.tool_choice);
}

fn streaming_events(turn: &support::Turn) -> Vec<Value> {
    support::recorded_named_sse_events(turn)
}

fn response_output(turn: &support::Turn) -> Vec<OutputItem> {
    let response = if let Some(body) = &turn.response.body {
        body.clone()
    } else {
        streaming_events(turn)
            .into_iter()
            .rev()
            .filter_map(|event| event.get("response").cloned())
            .find(|response| response["status"] == "completed" && response["output"].is_array())
            .expect("completed streaming response payload")
    };
    let accumulator = ResponseAccumulator::from_json(&response.to_string(), None).expect("valid completed response");
    let payload = accumulator.finalize(MODEL, None, None);
    assert_eq!(payload.status, "completed");
    payload.output
}

fn output_text(output: &[OutputItem]) -> String {
    output
        .iter()
        .filter_map(|item| match item {
            OutputItem::Message(message) => Some(
                message
                    .content
                    .iter()
                    .map(|content| content.text.as_str())
                    .collect::<String>(),
            ),
            _ => None,
        })
        .collect::<String>()
        .trim()
        .to_owned()
}

fn mcp_calls(output: &[OutputItem]) -> Vec<&McpCall> {
    assert!(
        !output.iter().any(|item| matches!(item, OutputItem::FunctionCall(_))),
        "internal function calls must not leak from the gateway"
    );
    output
        .iter()
        .filter_map(|item| match item {
            OutputItem::McpCall(call) => Some(call),
            _ => None,
        })
        .collect()
}

fn mcp_list_tools(output: &[OutputItem]) -> &McpListTools {
    let items = output
        .iter()
        .filter_map(|item| match item {
            OutputItem::McpListTools(list_tools) => Some(list_tools),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(items.len(), 1, "response should contain one mcp_list_tools item");
    items[0]
}

fn normalized_mcp_list_tools_item(item: &Value) -> Value {
    json!({
        "server_label": item["server_label"],
        "tools": item["tools"],
        "error": item.get("error").cloned().unwrap_or(Value::Null),
    })
}

fn assert_list_tools_output_matches_openai(openai: &[OutputItem], gateway: &[OutputItem]) {
    let expected = mcp_list_tools(openai);
    let actual = mcp_list_tools(gateway);
    assert!(expected.id.starts_with("mcpl_"));
    assert!(actual.id.starts_with("mcpl_"));

    let expected = serde_json::to_value(expected).expect("OpenAI mcp_list_tools JSON");
    let actual = serde_json::to_value(actual).expect("gateway mcp_list_tools JSON");
    assert_eq!(
        normalized_mcp_list_tools_item(&actual),
        normalized_mcp_list_tools_item(&expected)
    );
}

fn normalized_json_string(value: &str) -> Value {
    serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_owned()))
}

fn normalized_optional_output(output: Option<&str>) -> Value {
    output.map_or(Value::Null, normalized_json_string)
}

fn assert_calls_match_openai(openai: &[OutputItem], gateway: &[OutputItem], compare_arguments: bool) {
    assert_list_tools_output_matches_openai(openai, gateway);

    let expected = mcp_calls(openai);
    let actual = mcp_calls(gateway);
    assert_eq!(actual.len(), expected.len());

    for (expected, actual) in expected.into_iter().zip(actual) {
        assert_eq!(actual.server_label, expected.server_label);
        assert_eq!(actual.name, expected.name);
        assert_eq!(actual.status, expected.status);
        assert_eq!(
            normalized_optional_output(actual.output.as_deref()),
            normalized_optional_output(expected.output.as_deref())
        );
        assert_eq!(
            serde_json::to_value(&actual.error).expect("gateway MCP error JSON"),
            serde_json::to_value(&expected.error).expect("OpenAI MCP error JSON")
        );
        if compare_arguments {
            assert_eq!(
                normalized_json_string(&actual.arguments),
                normalized_json_string(&expected.arguments)
            );
        } else {
            assert!(normalized_json_string(&actual.arguments).is_object());
            assert!(normalized_json_string(&expected.arguments).is_object());
        }
    }
}

fn assert_mcp_list_tools_lifecycle(events: &[Value]) -> Value {
    let sequence_numbers = events
        .iter()
        .map(|event| {
            event["sequence_number"]
                .as_u64()
                .unwrap_or_else(|| panic!("event missing sequence_number: {event}"))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sequence_numbers,
        (0..u64::try_from(events.len()).expect("event count fits in u64")).collect::<Vec<_>>()
    );

    let lifecycle = events
        .iter()
        .filter(|event| {
            event["type"]
                .as_str()
                .is_some_and(|event_type| event_type.starts_with("response.mcp_list_tools."))
                || matches!(
                    event["type"].as_str(),
                    Some("response.output_item.added" | "response.output_item.done")
                ) && event["item"]["type"] == "mcp_list_tools"
        })
        .collect::<Vec<_>>();
    assert_eq!(
        lifecycle
            .iter()
            .filter_map(|event| event["type"].as_str())
            .collect::<Vec<_>>(),
        [
            "response.output_item.added",
            "response.mcp_list_tools.in_progress",
            "response.mcp_list_tools.completed",
            "response.output_item.done",
        ]
    );

    let item_id = lifecycle[0]["item"]["id"]
        .as_str()
        .expect("mcp_list_tools added item id");
    assert!(item_id.starts_with("mcpl_"));
    assert!(
        lifecycle
            .iter()
            .all(|event| { event["item"]["id"].as_str().or_else(|| event["item_id"].as_str()) == Some(item_id) })
    );

    let output_index = &lifecycle[0]["output_index"];
    assert!(lifecycle.iter().all(|event| &event["output_index"] == output_index));
    assert_eq!(lifecycle[0]["item"]["tools"], json!([]));
    assert_eq!(lifecycle[0]["item"]["server_label"], "counter");

    let done_item = &lifecycle[3]["item"];
    assert_eq!(done_item["server_label"], "counter");
    assert!(!done_item["tools"].as_array().expect("discovered tools").is_empty());

    let terminal_item = events
        .iter()
        .rev()
        .find(|event| event["type"] == "response.completed")
        .and_then(|event| event["response"]["output"].as_array())
        .and_then(|output| output.iter().find(|item| item["type"] == "mcp_list_tools"))
        .expect("terminal response mcp_list_tools item");
    assert_eq!(terminal_item, done_item);

    done_item.clone()
}

fn mcp_call_event_traces(events: &[Value]) -> Vec<(String, Vec<String>)> {
    let mut traces: HashMap<String, (String, Vec<String>)> = HashMap::new();

    for event in events {
        let Some(event_type) = event["type"].as_str() else {
            continue;
        };
        let item = event.get("item");
        let mcp_item = item.filter(|item| item["type"] == "mcp_call");
        let item_id = mcp_item
            .and_then(|item| item["id"].as_str())
            .or_else(|| event["item_id"].as_str());
        let Some(item_id) = item_id else {
            continue;
        };

        if let Some(item) = mcp_item {
            let name = item["name"].as_str().expect("mcp_call name").to_owned();
            traces.entry(item_id.to_owned()).or_insert_with(|| (name, Vec::new()));
        }
        if event_type.starts_with("response.mcp_call")
            || matches!(event_type, "response.output_item.added" | "response.output_item.done") && mcp_item.is_some()
        {
            traces
                .get_mut(item_id)
                .expect("mcp_call added before lifecycle events")
                .1
                .push(event_type.to_owned());
        }
    }

    let mut traces = traces.into_values().collect::<Vec<_>>();
    traces.sort_by(|left, right| left.0.cmp(&right.0));
    traces
}

fn normalized_mcp_item_transitions(events: &[Value]) -> HashMap<String, Vec<Value>> {
    let mut transitions: HashMap<String, Vec<Value>> = HashMap::new();
    for event in events {
        let Some(event_type) = event["type"].as_str() else {
            continue;
        };
        if !matches!(event_type, "response.output_item.added" | "response.output_item.done") {
            continue;
        }
        let Some(item) = event.get("item") else {
            continue;
        };
        if item["type"] != "mcp_call" {
            continue;
        }
        let name = item["name"].as_str().expect("mcp_call name").to_owned();
        transitions.entry(name).or_default().push(json!({
            "event_type": event_type,
            "type": item["type"],
            "server_label": item["server_label"],
            "name": item["name"],
            "status": item["status"],
            "arguments": item["arguments"].as_str().map(normalized_json_string),
            "output": item["output"].as_str().map(normalized_json_string),
            "error": item["error"],
            "approval_request_id": item["approval_request_id"],
        }));
    }
    transitions
}

fn assert_streaming_contract_matches_openai(openai: &support::Turn, gateway: &support::Turn) {
    let expected = streaming_events(openai);
    let actual = streaming_events(gateway);
    let expected_list_tools = assert_mcp_list_tools_lifecycle(&expected);
    let actual_list_tools = assert_mcp_list_tools_lifecycle(&actual);
    assert_eq!(
        normalized_mcp_list_tools_item(&actual_list_tools),
        normalized_mcp_list_tools_item(&expected_list_tools)
    );
    assert_eq!(mcp_call_event_traces(&actual), mcp_call_event_traces(&expected));
    assert_eq!(
        normalized_mcp_item_transitions(&actual),
        normalized_mcp_item_transitions(&expected)
    );
    assert!(actual.iter().all(|event| {
        !event["type"]
            .as_str()
            .is_some_and(|kind| kind.contains("mcp_tool_call"))
    }));
}

#[test]
fn mcp_tool_listing_matches_openai_without_calls() {
    let (openai, gateway) = load_scenario_pair("list-tools", true);
    assert_matching_native_mcp_requests(&openai, &gateway, true, None);

    let openai_output = response_output(&openai.turns[0]);
    let gateway_output = response_output(&gateway.turns[0]);
    assert_calls_match_openai(&openai_output, &gateway_output, true);
    assert_streaming_contract_matches_openai(&openai.turns[0], &gateway.turns[0]);

    for text in [output_text(&openai_output), output_text(&gateway_output)] {
        for tool_name in ["increment", "get_value", "sum"] {
            assert!(
                text.contains(tool_name),
                "tool listing should contain {tool_name}: {text}"
            );
        }
    }
}

#[test]
fn successful_streaming_mcp_calls_match_openai() {
    let (openai, gateway) = load_scenario_pair("call-sum-and-echo", true);
    assert_matching_native_mcp_requests(&openai, &gateway, true, Some(&["sum", "echo"]));

    let openai_output = response_output(&openai.turns[0]);
    let gateway_output = response_output(&gateway.turns[0]);
    assert_calls_match_openai(&openai_output, &gateway_output, true);
    assert_streaming_contract_matches_openai(&openai.turns[0], &gateway.turns[0]);
    assert_eq!(output_text(&gateway_output), output_text(&openai_output));
}

#[test]
fn missing_argument_mcp_failure_matches_openai() {
    let (openai, gateway) = load_scenario_pair("sum-missing-argument", true);
    assert_matching_native_mcp_requests(&openai, &gateway, true, Some(&["sum"]));

    let openai_output = response_output(&openai.turns[0]);
    let gateway_output = response_output(&gateway.turns[0]);
    assert_calls_match_openai(&openai_output, &gateway_output, true);
    assert_streaming_contract_matches_openai(&openai.turns[0], &gateway.turns[0]);
}

#[test]
fn invalid_argument_type_mcp_failure_matches_openai() {
    let (openai, gateway) = load_scenario_pair("sum-invalid-argument-type", true);
    assert_matching_native_mcp_requests(&openai, &gateway, true, Some(&["sum"]));

    let openai_output = response_output(&openai.turns[0]);
    let gateway_output = response_output(&gateway.turns[0]);
    assert_calls_match_openai(&openai_output, &gateway_output, true);
    assert_streaming_contract_matches_openai(&openai.turns[0], &gateway.turns[0]);
}

#[test]
fn successful_blocking_mcp_call_matches_openai() {
    let (openai, gateway) = load_scenario_pair("say-hello", false);
    assert_matching_native_mcp_requests(&openai, &gateway, false, Some(&["say_hello"]));

    let openai_output = response_output(&openai.turns[0]);
    let gateway_output = response_output(&gateway.turns[0]);
    // Arguments are model-generated. Qwen adds an ignored placeholder field
    // while GPT-4o sends `{}`; both execute the same zero-argument MCP tool.
    assert_calls_match_openai(&openai_output, &gateway_output, false);
    assert_eq!(output_text(&gateway_output), output_text(&openai_output));
}
