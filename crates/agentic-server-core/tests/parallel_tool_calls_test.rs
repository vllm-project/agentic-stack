use serde_json::{Value, json};

mod support;

const CASSETTE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/parallel_tool_calls");
const GATEWAY_MODEL_SLUG: &str = "Qwen-Qwen3.5-35B-A3B-FP8";
const OPENAI_MODEL_SLUG: &str = "gpt-5.6";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ToolCallContract {
    call_type: String,
    name: String,
    input: String,
}

fn load_builtin_pair(streaming: bool) -> (support::Cassette, support::Cassette) {
    let mode = if streaming { "streaming" } else { "nonstreaming" };
    let openai = support::load_cassette(&format!(
        "{CASSETTE_DIR}/parallel-builtin-only-openai-reference-{OPENAI_MODEL_SLUG}-{mode}.yaml"
    ));
    let gateway = support::load_cassette(&format!(
        "{CASSETTE_DIR}/parallel-builtin-only-gateway-{GATEWAY_MODEL_SLUG}-{mode}.yaml"
    ));
    (openai, gateway)
}

fn terminal_response(turn: &support::Turn) -> Value {
    if let Some(body) = &turn.response.body {
        return body.clone();
    }

    support::recorded_named_sse_events(turn)
        .into_iter()
        .find(|event| event["type"] == "response.completed")
        .and_then(|event| event.get("response").cloned())
        .expect("streaming turn should contain response.completed")
}

fn terminal_output(turn: &support::Turn) -> Vec<Value> {
    let response = terminal_response(turn);
    assert_eq!(response["status"], "completed");
    response["output"]
        .as_array()
        .expect("completed response should contain output")
        .clone()
}

fn assert_request_contract(openai: &support::Cassette, gateway: &support::Cassette, streaming: bool) {
    assert_eq!(openai.turns.len(), 3);
    assert_eq!(gateway.turns.len(), openai.turns.len());

    for (turn_index, (expected, actual)) in openai.turns.iter().zip(&gateway.turns).enumerate() {
        let expected = &expected.request;
        let actual = &actual.request;

        assert_eq!(expected.path, "/v1/responses");
        assert_eq!(actual.path, expected.path);
        assert_eq!(expected.body.input, actual.body.input, "turn {} input", turn_index + 1);
        assert_eq!(expected.body.tools, actual.body.tools, "turn {} tools", turn_index + 1);
        assert_eq!(expected.body.tool_choice, actual.body.tool_choice);
        assert_eq!(expected.body.max_output_tokens, actual.body.max_output_tokens);
        assert_eq!(expected.body.store, actual.body.store);
        assert_eq!(expected.body.stream, streaming);
        assert_eq!(actual.body.stream, streaming);
        assert_eq!(expected.body.parallel_tool_calls, Some(true));
        assert_eq!(actual.body.parallel_tool_calls, Some(true));
    }
}

fn assert_previous_response_chain(cassette: &support::Cassette) {
    let response_ids = cassette
        .turns
        .iter()
        .map(|turn| {
            terminal_response(turn)["id"]
                .as_str()
                .expect("completed response should have an ID")
                .to_owned()
        })
        .collect::<Vec<_>>();

    for (turn_index, turn) in cassette.turns.iter().enumerate() {
        let expected = turn_index.checked_sub(1).map(|index| response_ids[index].as_str());
        assert_eq!(
            turn.request.body.previous_response_id.as_deref(),
            expected,
            "turn {} should continue the immediately preceding response",
            turn_index + 1
        );
    }
}

fn canonical_json(raw: &str) -> String {
    serde_json::from_str::<Value>(raw).map_or_else(
        |_| raw.trim().to_owned(),
        |value| serde_json::to_string(&value).expect("JSON value should serialize"),
    )
}

fn tool_call_contract(item: &Value) -> Option<ToolCallContract> {
    match item["type"].as_str()? {
        "web_search_call" => {
            assert_eq!(item["status"], "completed");
            Some(ToolCallContract {
                call_type: "web_search_call".to_owned(),
                name: item["action"]["type"].as_str().unwrap_or_default().to_owned(),
                // Search sources are provider-specific. The requested query batch is
                // the stable execution contract.
                input: serde_json::to_string(&item["action"]["queries"]).expect("web-search queries should serialize"),
            })
        }
        "mcp_call" => {
            assert_eq!(item["status"], "completed");
            assert!(item["error"].is_null());
            assert!(item["output"].as_str().is_some_and(|output| !output.is_empty()));
            Some(ToolCallContract {
                call_type: "mcp_call".to_owned(),
                name: format!(
                    "{}/{}",
                    item["server_label"].as_str().unwrap_or_default(),
                    item["name"].as_str().unwrap_or_default()
                ),
                input: canonical_json(item["arguments"].as_str().unwrap_or_default()),
            })
        }
        _ => None,
    }
}

fn tool_call_contracts(output: &[Value]) -> Vec<ToolCallContract> {
    let mut calls = output.iter().filter_map(tool_call_contract).collect::<Vec<_>>();
    calls.sort();
    calls
}

fn public_output_types(output: &[Value]) -> Vec<&str> {
    output
        .iter()
        .filter_map(|item| item["type"].as_str())
        .filter(|item_type| *item_type != "reasoning")
        .collect()
}

fn mcp_list_tools_contract(item: &Value) -> Value {
    let tools = item["tools"]
        .as_array()
        .expect("mcp_list_tools should contain tools")
        .iter()
        .map(|tool| {
            json!({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"],
                "annotations": tool["annotations"],
            })
        })
        .collect::<Vec<_>>();

    json!({
        "server_label": item["server_label"],
        "tools": tools,
    })
}

fn sole_mcp_list_tools(cassette: &support::Cassette) -> Value {
    let discovered = cassette
        .turns
        .iter()
        .enumerate()
        .flat_map(|(turn_index, turn)| {
            terminal_output(turn)
                .into_iter()
                .filter(|item| item["type"] == "mcp_list_tools")
                .map(move |item| (turn_index, item))
        })
        .collect::<Vec<_>>();

    assert_eq!(
        discovered.len(),
        1,
        "MCP discovery must be emitted once for the whole stored response chain"
    );
    assert_eq!(discovered[0].0, 0, "MCP discovery belongs to the first turn only");
    mcp_list_tools_contract(&discovered[0].1)
}

fn assert_streaming_lifecycle(turn: &support::Turn, output: &[Value]) {
    let events = support::recorded_named_sse_events(turn);
    let sequence_numbers = events
        .iter()
        .map(|event| event["sequence_number"].as_u64().expect("SSE event sequence number"))
        .collect::<Vec<_>>();
    assert!(
        sequence_numbers.windows(2).all(|pair| pair[1] == pair[0] + 1),
        "SSE sequence numbers should be contiguous"
    );

    for (output_type, completed_event) in [
        ("mcp_list_tools", "response.mcp_list_tools.completed"),
        ("mcp_call", "response.mcp_call.completed"),
        ("web_search_call", "response.web_search_call.completed"),
    ] {
        let output_count = output.iter().filter(|item| item["type"] == output_type).count();
        let completed_count = events.iter().filter(|event| event["type"] == completed_event).count();
        let done_count = events
            .iter()
            .filter(|event| event["type"] == "response.output_item.done" && event["item"]["type"] == output_type)
            .count();
        assert_eq!(completed_count, output_count, "{output_type} completed lifecycle count");
        assert_eq!(done_count, output_count, "{output_type} output-item lifecycle count");
    }
}

#[test]
fn multi_turn_parallel_builtin_calls_match_openai_reference() {
    let expected_types = [
        vec!["mcp_list_tools", "web_search_call", "web_search_call", "message"],
        vec!["mcp_call", "mcp_call", "message"],
        vec!["web_search_call", "mcp_call", "message"],
    ];

    for streaming in [false, true] {
        let (openai, gateway) = load_builtin_pair(streaming);
        assert_request_contract(&openai, &gateway, streaming);
        assert_previous_response_chain(&openai);
        assert_previous_response_chain(&gateway);

        for (turn_index, ((expected_turn, actual_turn), expected_types)) in
            openai.turns.iter().zip(&gateway.turns).zip(&expected_types).enumerate()
        {
            let expected_output = terminal_output(expected_turn);
            let actual_output = terminal_output(actual_turn);
            assert_eq!(public_output_types(&expected_output), *expected_types);
            assert_eq!(
                public_output_types(&actual_output),
                *expected_types,
                "gateway turn {} public output types",
                turn_index + 1
            );

            let expected_calls = tool_call_contracts(&expected_output);
            let actual_calls = tool_call_contracts(&actual_output);
            assert_eq!(expected_calls.len(), 2, "reference turn {} call count", turn_index + 1);
            assert_eq!(
                actual_calls,
                expected_calls,
                "gateway turn {} tool calls should match the OpenAI contract",
                turn_index + 1
            );

            if streaming {
                assert_streaming_lifecycle(expected_turn, &expected_output);
                assert_streaming_lifecycle(actual_turn, &actual_output);
            }
        }

        assert_eq!(
            sole_mcp_list_tools(&gateway),
            sole_mcp_list_tools(&openai),
            "gateway MCP discovery should match OpenAI and should not repeat on later turns"
        );
    }
}
