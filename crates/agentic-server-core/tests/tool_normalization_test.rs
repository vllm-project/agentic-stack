//! Cassette-driven validation of the tool framework wire types and normalization pipeline.
//!
//! Validates that real cassette request bodies — the exact JSON the gateway receives —
//! parse correctly into `Vec<ResponsesTool>` and normalize through the full pipeline.

use serde::Deserialize;
use serde_json::Value;

use agentic_core::events::WireEvent;
use agentic_core::executor::RequestContext;
use agentic_core::tool::{
    CodexNamespaceHandler, GatewayExecutors, ToolRegistry, ToolType, model_visible_namespace_member_name,
};
use agentic_core::types::request_response::RequestPayload;
use agentic_core::types::tools::ResponsesTool;
use agentic_core::utils::common::serialize_to_string;

const MULTI_TURN_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/tool_calls/multi_turn");
const CODEX_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes/codex");

const CODEX_CASSETTES: &[&str] = &[
    "codex-direct-vllm-http-custom-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-direct-vllm-http-flat-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-direct-vllm-http-function-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-gateway-http-custom-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-gateway-http-function-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-gateway-http-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-gateway-websocket-custom-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-gateway-websocket-function-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-gateway-websocket-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-openai-https-custom-tool-gpt-5.6-streaming.yaml",
    "codex-openai-https-function-tool-gpt-4o-streaming.yaml",
    "codex-openai-https-namespace-tool-gpt-4o-streaming.yaml",
    "codex-openai-websocket-custom-tool-gpt-5.6-streaming.yaml",
    "codex-openai-websocket-function-tool-gpt-4o-streaming.yaml",
    "codex-openai-websocket-namespace-tool-gpt-4o-streaming.yaml",
];

const CODEX_CUSTOM_CASSETTES: &[&str] = &[
    "codex-direct-vllm-http-custom-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-gateway-http-custom-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-gateway-websocket-custom-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-openai-https-custom-tool-gpt-5.6-streaming.yaml",
    "codex-openai-websocket-custom-tool-gpt-5.6-streaming.yaml",
];

const CODEX_NAMESPACE_CASSETTES: &[&str] = &[
    "codex-gateway-http-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-gateway-websocket-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml",
    "codex-openai-https-namespace-tool-gpt-4o-streaming.yaml",
    "codex-openai-websocket-namespace-tool-gpt-4o-streaming.yaml",
];

#[derive(Deserialize)]
struct TurnCassette {
    turns: Vec<Turn>,
}

#[derive(Deserialize)]
struct Turn {
    request: serde_yml::Value,
    #[allow(dead_code)]
    response: serde_yml::Value,
}

fn load_cassette_from(dir: &str, filename: &str) -> TurnCassette {
    let path = format!("{dir}/{filename}");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_yml::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

fn load_cassette(filename: &str) -> TurnCassette {
    load_cassette_from(MULTI_TURN_DIR, filename)
}

fn load_codex_cassette(filename: &str) -> TurnCassette {
    load_cassette_from(CODEX_DIR, filename)
}

fn tools_from_turn(turn: &Turn) -> Option<serde_json::Value> {
    let body = turn.request.get("body")?;
    let json: serde_json::Value = serde_json::to_value(body).ok()?;
    json.get("tools").cloned()
}

fn request_body_from_turn(turn: &Turn) -> serde_json::Value {
    let body = turn.request.get("body").expect("turn has body");
    serde_json::to_value(body).expect("body to json")
}

fn parse_tools_from_turn(cassette_file: &str, turn_idx: usize, turn: &Turn) -> Vec<ResponsesTool> {
    let tools_val =
        tools_from_turn(turn).unwrap_or_else(|| panic!("{cassette_file} turn {turn_idx}: expected tools array"));
    serde_json::from_value(tools_val.clone())
        .unwrap_or_else(|e| panic!("{cassette_file} turn {turn_idx}: tools parse failed: {e}\nJSON: {tools_val}"))
}

fn upstream_request_value(payload: RequestPayload, stream: bool) -> Value {
    let ctx = RequestContext {
        original_request: payload.clone(),
        enriched_request: payload,
        new_input_items: Vec::new(),
        response_id: "resp_test".to_string(),
        conversation_id: None,
        conversation_version: None,
        continuation: None,
    };
    let upstream_request = ctx
        .enriched_request
        .to_upstream_request(stream)
        .expect("valid upstream request");
    let body = serialize_to_string(&upstream_request).expect("serialize upstream request");
    serde_json::from_str(&body).expect("upstream request should be JSON")
}

/// Parse `request.body.tools` from every turn of a cassette into `Vec<ResponsesTool>`.
fn assert_tools_parse(cassette_file: &str) {
    let cassette = load_cassette(cassette_file);
    for (i, turn) in cassette.turns.iter().enumerate() {
        let Some(tools_val) = tools_from_turn(turn) else {
            continue;
        };
        let tools: Vec<ResponsesTool> = serde_json::from_value(tools_val.clone())
            .unwrap_or_else(|e| panic!("{cassette_file} turn {i}: tools parse failed: {e}\nJSON: {tools_val}"));
        assert!(
            !tools.is_empty(),
            "{cassette_file} turn {i}: expected non-empty tools array"
        );
    }
}

/// For every turn that has tools, verify normalization produces only `FunctionTool` entries.
fn assert_tools_normalize(cassette_file: &str) {
    let cassette = load_cassette(cassette_file);
    for (i, turn) in cassette.turns.iter().enumerate() {
        let Some(tools_val) = tools_from_turn(turn) else {
            continue;
        };
        let tools: Vec<ResponsesTool> = serde_json::from_value(tools_val).expect("tools parse");
        let resolved = CodexNamespaceHandler
            .resolve_namespace_members(&tools)
            .unwrap_or_else(|err| panic!("{cassette_file} turn {i}: namespace resolution failed: {err}"));
        let normalized: Vec<_> = resolved.iter().flat_map(ResponsesTool::to_function_tools).collect();
        for ft in &normalized {
            assert_eq!(
                ft.type_, "function",
                "{cassette_file} turn {i}: normalized type must be 'function'"
            );
            assert!(
                !ft.name.is_empty(),
                "{cassette_file} turn {i}: normalized name must not be empty"
            );
        }
        // Every function visible after namespace member renaming must normalize —
        // plain Function tools, plus each Namespace's Function members.
        let function_count = resolved
            .iter()
            .map(|t| match t {
                ResponsesTool::Function(_) => 1,
                ResponsesTool::Namespace(namespace) => namespace
                    .tools
                    .iter()
                    .filter(|member| matches!(member, agentic_core::types::CodexNamespaceMember::Function(_)))
                    .count(),
                _ => 0,
            })
            .sum::<usize>();
        assert_eq!(
            normalized.len(),
            function_count,
            "{cassette_file} turn {i}: each Function tool must produce exactly one FunctionTool (got {} of {})",
            normalized.len(),
            function_count
        );
    }
}

/// For every turn that has tools, verify registry construction produces correct entries.
async fn assert_registry_lookup(cassette_file: &str) {
    let cassette = load_cassette(cassette_file);
    for (i, turn) in cassette.turns.iter().enumerate() {
        let Some(tools_val) = tools_from_turn(turn) else {
            continue;
        };
        let mut tools: Vec<ResponsesTool> = serde_json::from_value(tools_val).expect("tools parse");
        let declared_tools = tools.clone();
        let registry = ToolRegistry::build_with_handlers(&mut tools, &mut GatewayExecutors::default())
            .await
            .unwrap_or_else(|err| panic!("{cassette_file} turn {i}: registry failed: {err}"));
        for tool in &declared_tools {
            if let ResponsesTool::Function(p) = tool {
                let entry = registry
                    .lookup(p.name.as_str())
                    .unwrap_or_else(|| panic!("{cassette_file} turn {i}: tool '{}' not found in registry", p.name));
                assert_eq!(
                    entry.tool_type,
                    ToolType::Function,
                    "{cassette_file} turn {i}: tool '{}' must be Function type",
                    p.name
                );
            }
        }
    }
}

/// Full round-trip: deserialize `request.body` → `RequestPayload` → `ToolRegistry`
/// → `to_upstream_request()` → assert upstream tools only contains function entries.
fn assert_full_roundtrip(cassette_file: &str) {
    let cassette = load_cassette(cassette_file);
    for (i, turn) in cassette.turns.iter().enumerate() {
        let body = turn.request.get("body").expect("turn has body");
        let json: serde_json::Value = serde_json::to_value(body).expect("body to json");
        let payload: RequestPayload = serde_json::from_value(json.clone())
            .unwrap_or_else(|e| panic!("{cassette_file} turn {i}: RequestPayload parse failed: {e}"));
        let upstream = upstream_request_value(payload, false);
        if let Some(tools) = upstream.get("tools").and_then(Value::as_array) {
            for ft in tools {
                assert_eq!(
                    ft.get("type").and_then(Value::as_str),
                    Some("function"),
                    "{cassette_file} turn {i}: upstream tools must only contain FunctionTool"
                );
                assert!(
                    ft.get("name")
                        .and_then(Value::as_str)
                        .is_some_and(|name| !name.is_empty()),
                    "{cassette_file} turn {i}: upstream tool name must not be empty"
                );
            }
        }
    }
}

#[test]
fn tools_parse_3turn() {
    assert_tools_parse("openai_responses_tool_calls_3turn.yaml");
}

#[test]
fn tools_parse_5turn() {
    assert_tools_parse("openai_responses_tool_calls_5turn.yaml");
}

#[test]
fn tools_parse_parallel() {
    assert_tools_parse("openai_responses_tool_calls_parallel.yaml");
}

#[test]
fn tools_normalize_3turn() {
    assert_tools_normalize("openai_responses_tool_calls_3turn.yaml");
}

#[test]
fn tools_normalize_5turn() {
    assert_tools_normalize("openai_responses_tool_calls_5turn.yaml");
}

#[test]
fn tools_normalize_parallel() {
    assert_tools_normalize("openai_responses_tool_calls_parallel.yaml");
}

#[tokio::test]
async fn registry_lookup_3turn() {
    assert_registry_lookup("openai_responses_tool_calls_3turn.yaml").await;
}

#[tokio::test]
async fn registry_lookup_5turn() {
    assert_registry_lookup("openai_responses_tool_calls_5turn.yaml").await;
}

#[tokio::test]
async fn registry_lookup_parallel() {
    assert_registry_lookup("openai_responses_tool_calls_parallel.yaml").await;
}

#[test]
fn roundtrip_3turn() {
    assert_full_roundtrip("openai_responses_tool_calls_3turn.yaml");
}

#[test]
fn roundtrip_5turn() {
    assert_full_roundtrip("openai_responses_tool_calls_5turn.yaml");
}

#[test]
fn roundtrip_parallel() {
    assert_full_roundtrip("openai_responses_tool_calls_parallel.yaml");
}

#[test]
fn codex_request_payloads_parse_all_recorded_shapes() {
    for filename in CODEX_CASSETTES {
        let cassette = load_codex_cassette(filename);
        assert_eq!(cassette.turns.len(), 2, "{filename} should have two turns");

        for (i, turn) in cassette.turns.iter().enumerate() {
            let json = request_body_from_turn(turn);
            let payload: RequestPayload = serde_json::from_value(json.clone())
                .unwrap_or_else(|e| panic!("{filename} turn {i}: RequestPayload parse failed: {e}\nJSON: {json}"));
            assert!(
                payload.tools.as_ref().is_some_and(|tools| !tools.is_empty()),
                "{filename} turn {i}: expected tools"
            );
            if filename.contains("websocket") {
                assert!(
                    json.get("stream").is_none(),
                    "{filename} turn {i}: websocket cassette should not contain HTTP-only stream field"
                );
            }
        }
    }
}

#[test]
fn codex_custom_grammar_cassettes_fail_closed_before_upstream_normalization() {
    for filename in CODEX_CUSTOM_CASSETTES {
        let cassette = load_codex_cassette(filename);
        assert_eq!(cassette.turns.len(), 2, "{filename} should have two turns");

        for (i, turn) in cassette.turns.iter().enumerate() {
            let json = request_body_from_turn(turn);
            let payload: RequestPayload = serde_json::from_value(json.clone())
                .unwrap_or_else(|e| panic!("{filename} turn {i}: RequestPayload parse failed: {e}\nJSON: {json}"));
            let tools = payload.tools.as_ref().expect("custom cassette should declare tools");
            assert!(
                tools.iter().any(|tool| matches!(
                    tool,
                    ResponsesTool::Custom(custom)
                        if custom.name.as_str() == "agentic_raw_echo"
                            && custom.format.as_ref().and_then(|format| format.get("syntax")).and_then(Value::as_str)
                                == Some("lark")
                )),
                "{filename} turn {i}: expected native custom grammar declaration"
            );

            // The historical gateway and OpenAI cassettes both returned the
            // only value allowed by this Lark grammar, but the prompt requested
            // that exact value too. OpenAI constrained generation; the old
            // gateway only exposed the grammar as model guidance, so matching
            // output did not prove equivalent enforcement. Reject the format
            // rather than silently normalizing it to an unconstrained function.
            let error = payload
                .to_upstream_request(false)
                .expect_err("custom grammar must be rejected before normalization");
            assert!(
                error.to_string().contains("cannot preserve constrained decoding"),
                "{filename} turn {i}: unexpected validation error: {error}"
            );
        }
    }
}

#[test]
fn codex_namespace_cassettes_flatten_to_safe_upstream_function_name() {
    let expected_flat_name = model_visible_namespace_member_name("mcp__agentic_fixture", "add_numbers");

    for filename in CODEX_NAMESPACE_CASSETTES {
        let cassette = load_codex_cassette(filename);
        for (i, turn) in cassette.turns.iter().enumerate() {
            let tools = parse_tools_from_turn(filename, i, turn);
            assert!(
                tools.iter().any(|tool| {
                    matches!(
                        tool,
                        ResponsesTool::Namespace(namespace)
                            if namespace.name == "mcp__agentic_fixture"
                                && namespace.tools.iter().any(|member| {
                                    matches!(
                                        member,
                                        agentic_core::types::CodexNamespaceMember::Function(function)
                                            if function.name.as_str() == "add_numbers"
                                    )
                                })
                    )
                }),
                "{filename} turn {i}: expected raw namespace tool"
            );

            let resolved = CodexNamespaceHandler
                .resolve_namespace_members(&tools)
                .unwrap_or_else(|err| panic!("{filename} turn {i}: namespace resolution failed: {err}"));
            assert!(
                resolved.iter().any(|tool| {
                    matches!(tool, ResponsesTool::Namespace(namespace)
                    if namespace.tools.iter().any(|member| matches!(
                        member,
                        agentic_core::types::CodexNamespaceMember::Function(function)
                            if function.name.as_str() == expected_flat_name
                    )))
                }),
                "{filename} turn {i}: expected renamed namespace member {expected_flat_name}"
            );
            let upstream_tools: Vec<_> = resolved.iter().flat_map(ResponsesTool::to_function_tools).collect();
            assert!(
                upstream_tools.iter().any(|tool| tool.name == expected_flat_name),
                "{filename} turn {i}: expected upstream FunctionTool {expected_flat_name}"
            );
        }
    }
}

#[tokio::test]
async fn tool_registry_restores_wire_event_namespace_losslessly() {
    let mut tools: Vec<ResponsesTool> = serde_json::from_value(serde_json::json!([
        {
            "type": "namespace",
            "name": "mcp__agentic_fixture",
            "tools": [{"type": "function", "name": "add_numbers"}]
        }
    ]))
    .unwrap();
    let registry = ToolRegistry::build_with_handlers(&mut tools, &mut GatewayExecutors::default())
        .await
        .expect("valid registry");
    let mut wire = WireEvent::new("response.output_item.done");
    wire.output_index = Some(0);
    wire.rest.insert(
        "item".to_owned(),
        serde_json::json!({
            "type": "function_call",
            "name": "agentic_ns__mcp__agentic_fixture__add_numbers",
            "call_id": "call_1",
            "arguments": "{\"numbers\":[8,0]}",
            "provider_extra": {"kept": true}
        }),
    );

    assert!(registry.restore_stream_event_wire(&mut wire));

    let item = &wire.rest["item"];
    assert_eq!(item["namespace"], "mcp__agentic_fixture");
    assert_eq!(item["name"], "add_numbers");
    assert_eq!(item["arguments"], "{\"numbers\":[8,0]}");
    assert_eq!(item["provider_extra"]["kept"], true);
}

#[test]
fn codex_direct_vllm_flat_namespace_cassette_is_plain_function_tool() {
    let filename = "codex-direct-vllm-http-flat-namespace-tool-Qwen-Qwen3.6-35B-A3B-streaming.yaml";
    let expected_flat_name = model_visible_namespace_member_name("mcp__agentic_fixture", "add_numbers");
    let cassette = load_codex_cassette(filename);

    for (i, turn) in cassette.turns.iter().enumerate() {
        let tools = parse_tools_from_turn(filename, i, turn);
        assert_eq!(
            tools.len(),
            1,
            "{filename} turn {i}: direct vLLM flat namespace fixture should declare one tool"
        );
        assert!(
            matches!(
                &tools[0],
                ResponsesTool::Function(function) if function.name.as_str() == expected_flat_name
            ),
            "{filename} turn {i}: expected direct vLLM to see a plain function tool named {expected_flat_name}"
        );

        let flattened = CodexNamespaceHandler
            .resolve_namespace_members(&tools)
            .unwrap_or_else(|err| panic!("{filename} turn {i}: namespace resolution failed: {err}"));
        assert_eq!(flattened.len(), 1);
        assert!(
            matches!(
                &flattened[0],
                ResponsesTool::Function(function) if function.name.as_str() == expected_flat_name
            ),
            "{filename} turn {i}: flattening should preserve already-flat direct vLLM function"
        );
    }
}

#[test]
fn web_search_preview_normalizes_to_gateway_function() {
    let payload: RequestPayload = serde_json::from_value(serde_json::json!({
        "model": "test",
        "input": "what changed today?",
        "tools": [{"type": "web_search_preview"}]
    }))
    .unwrap();

    let upstream = upstream_request_value(payload, false);
    let tools = upstream
        .get("tools")
        .and_then(Value::as_array)
        .expect("web_search should normalize to a function tool");

    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].get("type").and_then(Value::as_str), Some("function"));
    assert_eq!(tools[0].get("name").and_then(Value::as_str), Some("web_search"));
    assert!(tools[0]["parameters"]["properties"]["query"].is_object());
    assert!(tools[0]["parameters"]["properties"]["queries"].is_object());
    assert_eq!(
        tools[0]["parameters"]["anyOf"],
        serde_json::json!([{"required": ["query"]}, {"required": ["queries"]}])
    );
}
