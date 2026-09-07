use agentic_core::tool::{ToolSearchState, model_visible_namespace_member_name};
use agentic_core::{InputItem, RequestPayload, ResponsesInput};
use serde_json::{Value, json};

fn request(tools: Value, input: Value) -> RequestPayload {
    let mut value = json!({
        "model": "test-model",
        "store": false,
        "parallel_tool_calls": false
    });
    value["input"] = input;
    value["tools"] = tools;
    serde_json::from_value(value).expect("test request must match the public wire model")
}

fn search_declaration() -> Value {
    json!({
        "type": "tool_search",
        "execution": "client",
        "description": "Search the client tool catalog",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": false
        }
    })
}

fn function(name: &str, description: &str, city_type: &str, deferred: bool) -> Value {
    json!({
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": city_type}},
            "required": ["city"],
            "additionalProperties": false
        },
        "strict": true,
        "defer_loading": deferred
    })
}

fn search_call(id: &str) -> Value {
    json!({
        "type": "tool_search_call",
        "id": format!("tsc_{id}"),
        "call_id": id,
        "arguments": {"query": "weather"}
    })
}

fn search_output(id: &str, tools: Vec<Value>) -> Value {
    let mut value = json!({
        "type": "tool_search_output",
        "call_id": id
    });
    value["tools"] = Value::Array(tools);
    value
}

fn tool_values(tools: Option<&[agentic_core::ResponsesTool]>) -> Value {
    serde_json::to_value(tools).expect("prepared tools serialize")
}

fn private_request(state: &mut ToolSearchState, public: &RequestPayload) -> RequestPayload {
    let mut private = public.clone();
    state
        .prepare_inference_request(&mut private)
        .expect("prepared state materializes a private inference request");
    private
}

fn private_tool_values(state: &mut ToolSearchState, public: &RequestPayload) -> Value {
    let private = private_request(state, public);
    tool_values(private.tools.as_deref())
}

fn private_input_value(state: &mut ToolSearchState, public: &RequestPayload) -> Value {
    serde_json::to_value(private_request(state, public).input).expect("private input serializes")
}

fn synthetic_description(state: &ToolSearchState) -> &str {
    state
        .synthetic_tool_search()
        .and_then(|function| function.description.as_deref())
        .expect("active tool search has a synthetic description")
}

#[test]
fn fresh_and_sequential_state_has_distinct_deterministic_views() {
    let deferred = function("get_weather", "Get weather", "string", true);
    let dynamic = function("get_uv", "Get UV index", "string", true);
    let tools = json!([
        search_declaration(),
        {
            "type": "function",
            "name": "current_time",
            "description": "Get current time",
            "parameters": {"type": "object"}
        },
        deferred
    ]);
    let input = json!([
        search_call("call_search_1"),
        search_output("call_search_1", vec![dynamic.clone()]),
        search_call("call_search_2"),
        search_output("call_search_2", vec![dynamic.clone()])
    ]);
    let request = request(tools, input);

    let mut state = ToolSearchState::build(&request).expect("valid ordered history");
    let mut rebuilt = ToolSearchState::build(&request).expect("same request builds again");
    let private = private_tool_values(&mut state, &request);
    let rebuilt_private = private_tool_values(&mut rebuilt, &request);

    assert!(state.is_active());
    assert_eq!(
        serde_json::to_string(&(
            tool_values(state.public_effective_tools()),
            &private,
            tool_values(Some(state.loaded_public_tools())),
            serde_json::to_value(state.synthetic_tool_search()).expect("synthetic declaration serializes")
        ))
        .expect("state snapshot serializes"),
        serde_json::to_string(&(
            tool_values(rebuilt.public_effective_tools()),
            &rebuilt_private,
            tool_values(Some(rebuilt.loaded_public_tools())),
            serde_json::to_value(rebuilt.synthetic_tool_search()).expect("synthetic declaration serializes")
        ))
        .expect("rebuilt snapshot serializes")
    );

    let public = tool_values(state.public_effective_tools());
    assert_eq!(public.as_array().map(Vec::len), Some(4));
    assert_eq!(public[2]["defer_loading"], true, "public deferral must be preserved");
    assert_eq!(
        public[3], dynamic,
        "a dynamic definition absent initially is appended once"
    );

    let loaded = tool_values(Some(state.loaded_public_tools()));
    assert_eq!(loaded, json!([dynamic]), "an exact repeated definition is idempotent");

    assert_eq!(private.as_array().map(Vec::len), Some(3));
    assert_eq!(private[0]["type"], "tool_search");
    assert_eq!(private[0]["execution"], "client");
    assert_eq!(private[1]["name"], "current_time");
    assert_eq!(private[2]["name"], "get_uv");
    assert!(private[2].get("defer_loading").is_none());
    assert!(
        private
            .as_array()
            .expect("private tools")
            .iter()
            .all(|tool| tool["name"] != "get_weather"),
        "unloaded deferred schemas must not enter the private view"
    );

    assert_eq!(
        serde_json::to_value(state.synthetic_tool_search()).expect("synthetic declaration serializes"),
        json!({
            "execution": "client",
            "description": "Search the client tool catalog. Available catalog entry: get_weather — Get weather.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": false
            }
        })
    );
}

#[test]
fn optional_declaration_fields_get_private_defaults_and_supplied_values_are_preserved() {
    let default_parameters = json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A concise description of the needed capabilities."
            }
        },
        "required": ["query"],
        "additionalProperties": false
    });
    let minimal = request(
        json!([{"type": "tool_search", "execution": "client"}]),
        json!("find a weather tool"),
    );
    let minimal_state = ToolSearchState::build(&minimal).expect("minimal declaration reaches state preparation");
    let minimal_synthetic = serde_json::to_value(minimal_state.synthetic_tool_search()).expect("synthetic serializes");
    assert_eq!(minimal_synthetic["description"], "Search the client tool catalog");
    assert_eq!(minimal_synthetic["parameters"], default_parameters);

    let supplied_declaration = search_declaration();
    let supplied = request(json!([supplied_declaration.clone()]), json!("find a weather tool"));
    let supplied_state = ToolSearchState::build(&supplied).expect("supplied declaration reaches state preparation");
    let supplied_synthetic =
        serde_json::to_value(supplied_state.synthetic_tool_search()).expect("synthetic serializes");
    assert_eq!(supplied_synthetic["description"], supplied_declaration["description"]);
    assert_eq!(supplied_synthetic["parameters"], supplied_declaration["parameters"]);

    let supplied_parameters = json!({"type": "object", "properties": {"term": {"type": "string"}}});
    let blank_description = request(
        json!([{
            "type": "tool_search",
            "execution": "client",
            "description": "   ",
            "parameters": supplied_parameters.clone()
        }]),
        json!("find a weather tool"),
    );
    let state = ToolSearchState::build(&blank_description).expect("blank description is normalized privately");
    let synthetic = serde_json::to_value(state.synthetic_tool_search()).expect("synthetic serializes");
    assert_eq!(synthetic["description"], "Search the client tool catalog");
    assert_eq!(synthetic["parameters"], supplied_parameters);

    for invalid_parameters in [json!({}), json!({"type": "array"})] {
        let invalid_schema = request(
            json!([{
                "type": "tool_search",
                "execution": "client",
                "description": "Find exactly the needed tool",
                "parameters": invalid_parameters
            }]),
            json!("find a weather tool"),
        );
        let state = ToolSearchState::build(&invalid_schema).expect("invalid schema is normalized privately");
        let synthetic = serde_json::to_value(state.synthetic_tool_search()).expect("synthetic serializes");
        assert_eq!(synthetic["description"], "Find exactly the needed tool");
        assert_eq!(synthetic["parameters"], default_parameters);
    }
}

#[test]
fn only_completed_search_outputs_load_definitions() {
    let deferred = function("get_weather", "Get weather", "string", true);

    for status in ["in_progress", "incomplete"] {
        let mut output = search_output("call_search_1", vec![deferred.clone()]);
        output["status"] = Value::String(status.to_owned());
        let public = request(
            json!([search_declaration(), deferred.clone()]),
            json!([search_call("call_search_1"), output]),
        );
        let error = ToolSearchState::build(&public).expect_err("a non-completed output must not load definitions");
        assert!(
            error
                .to_string()
                .contains("must be completed before it may load tool definitions")
        );
    }

    let mut output = search_output("call_search_1", vec![deferred.clone()]);
    output["status"] = json!("completed");
    let completed = request(
        json!([search_declaration(), deferred.clone()]),
        json!([search_call("call_search_1"), output]),
    );
    let state = ToolSearchState::build(&completed).expect("a completed output loads definitions");
    assert_eq!(tool_values(Some(state.loaded_public_tools())), json!([deferred]));
}

#[test]
fn replayed_search_calls_accept_documented_statuses() {
    let deferred = function("get_weather", "Get weather", "string", true);

    for status in ["in_progress", "completed", "incomplete"] {
        let mut call = search_call("call_search_1");
        call["status"] = Value::String(status.to_owned());
        let public = request(
            json!([search_declaration(), deferred.clone()]),
            json!([call, search_output("call_search_1", vec![deferred.clone()])]),
        );

        let state = ToolSearchState::build(&public).expect("documented replayed call status is accepted");
        assert_eq!(tool_values(Some(state.loaded_public_tools())), json!([deferred]));
    }
}

#[test]
fn one_history_pass_prepares_canonical_private_input_without_mutating_public_input() {
    let returned = function("get_weather", "Get weather", "string", true);
    let request = request(
        json!([search_declaration()]),
        json!([
            {"role": "user", "content": "find weather"},
            {
                "type": "tool_search_call",
                "id": "tsc_1",
                "call_id": "call_search_1",
                "arguments": {"z": 2, "a": 1}
            },
            search_output("call_search_1", vec![returned.clone()])
        ]),
    );
    let public_before = serde_json::to_value(&request.input).expect("public input serializes");

    let mut state = ToolSearchState::build(&request).expect("matching history prepares once");

    assert_eq!(
        serde_json::to_value(&request.input).expect("public input remains serializable"),
        public_before,
        "the public request must not be rewritten"
    );
    assert_eq!(
        private_input_value(&mut state, &request),
        json!([
            {"type": "message", "role": "user", "content": "find weather"},
            {
                "type": "function_call",
                "id": "tsc_1",
                "call_id": "call_search_1",
                "name": "tool_search",
                "arguments": "{\"z\":2,\"a\":1}",
                "status": "completed"
            },
            {
                "type": "function_call_output",
                "call_id": "call_search_1",
                "output": format!("{{\"tools\":[{}]}}", serde_json::to_string(&returned).unwrap())
            }
        ])
    );
}

#[test]
fn invalid_history_order_and_linkage_are_rejected() {
    let deferred = function("get_weather", "Get weather", "string", true);
    let base_tools = json!([search_declaration(), deferred.clone()]);
    let mut blank_call = request(base_tools.clone(), json!([search_call("call_search_1")]));
    let ResponsesInput::Items(items) = &mut blank_call.input else {
        panic!("test request uses item input")
    };
    let InputItem::ToolSearchCall(call) = &mut items[0] else {
        panic!("test request starts with search call")
    };
    call.call_id.clear();
    let mut blank_output = request(
        base_tools.clone(),
        json!([search_call("call_search_1"), search_output("call_search_1", vec![])]),
    );
    let ResponsesInput::Items(items) = &mut blank_output.input else {
        panic!("test request uses item input")
    };
    let InputItem::ToolSearchOutput(output) = &mut items[1] else {
        panic!("test request ends with search output")
    };
    output.call_id.clear();

    let cases = [
        (
            "orphan output",
            request(base_tools.clone(), json!([search_output("call_search_1", vec![])])),
        ),
        (
            "output before call",
            request(
                base_tools.clone(),
                json!([search_output("call_search_1", vec![]), search_call("call_search_1")]),
            ),
        ),
        (
            "mismatched call id",
            request(
                base_tools.clone(),
                json!([search_call("call_search_1"), search_output("call_search_2", vec![])]),
            ),
        ),
        (
            "ambiguous nested call",
            request(
                base_tools.clone(),
                json!([search_call("call_search_1"), search_call("call_search_2")]),
            ),
        ),
        (
            "duplicate output",
            request(
                base_tools.clone(),
                json!([
                    search_call("call_search_1"),
                    search_output("call_search_1", vec![]),
                    search_output("call_search_1", vec![])
                ]),
            ),
        ),
        (
            "unresolved call",
            request(base_tools.clone(), json!([search_call("call_search_1")])),
        ),
        ("empty call id", blank_call),
        ("empty output call id", blank_output),
    ];

    for (case, request) in cases {
        assert!(ToolSearchState::build(&request).is_err(), "{case} must be rejected");
    }
}

#[test]
fn invalid_loaded_definitions_and_normalized_collisions_are_rejected() {
    let deferred = function("get_weather", "Get weather", "string", true);
    let changed_schema = function("get_weather", "Get weather", "integer", true);
    let base_tools = json!([search_declaration(), deferred]);
    let cases = [
        (
            "schema conflict",
            request(
                base_tools.clone(),
                json!([
                    search_call("call_search_1"),
                    search_output("call_search_1", vec![changed_schema])
                ]),
            ),
        ),
        (
            "cross-kind identity conflict",
            request(
                base_tools.clone(),
                json!([
                    search_call("call_search_1"),
                    search_output(
                        "call_search_1",
                        vec![json!({
                            "type": "mcp",
                            "server_label": "get_weather",
                            "server_url": "https://mcp.example.test/mcp",
                            "defer_loading": true
                        })]
                    )
                ]),
            ),
        ),
        (
            "reserved synthetic name",
            request(
                base_tools,
                json!([
                    search_call("call_search_1"),
                    search_output(
                        "call_search_1",
                        vec![function("tool_search", "Conflict", "string", true)]
                    )
                ]),
            ),
        ),
        (
            "normalized namespace member collision",
            request(
                json!([
                    search_declaration(),
                    {"type": "function", "name": "agentic_ns__weather__forecast"}
                ]),
                json!([
                    search_call("call_search_1"),
                    search_output(
                        "call_search_1",
                        vec![json!({
                            "type": "namespace",
                            "name": "weather",
                            "tools": [{
                                "type": "function",
                                "name": "forecast",
                                "defer_loading": true
                            }]
                        })]
                    )
                ]),
            ),
        ),
        (
            "unsupported dynamically loaded custom tool",
            request(
                json!([search_declaration()]),
                json!([
                    search_call("call_search_1"),
                    search_output("call_search_1", vec![json!({"type": "custom", "name": "unsupported"})])
                ]),
            ),
        ),
    ];

    for (case, request) in cases {
        assert!(ToolSearchState::build(&request).is_err(), "{case} must be rejected");
    }
}

#[test]
fn initial_and_dynamic_namespace_collisions_are_rejected() {
    let initial = json!([
        search_declaration(),
        {"type": "namespace", "name": "a__b", "tools": [{"type": "function", "name": "c"}]},
        {"type": "namespace", "name": "a", "tools": [{"type": "function", "name": "b__c"}]}
    ]);
    assert!(
        ToolSearchState::build(&request(initial, json!("find a tool"))).is_err(),
        "initial namespace collisions must fail"
    );

    let tools = json!([{
        "type": "namespace",
        "name": "weather",
        "tools": [{"type": "function", "name": "forecast"}]
    }, search_declaration()]);
    let input = json!([
        search_call("call_search_1"),
        search_output(
            "call_search_1",
            vec![function(
                "agentic_ns__weather__forecast",
                "Dynamic function",
                "string",
                true
            )]
        )
    ]);
    assert!(
        ToolSearchState::build(&request(tools, input)).is_err(),
        "dynamic namespace collisions must fail"
    );
}

#[test]
fn loaded_namespace_model_output_is_identity_only_while_private_tools_retain_members() {
    let namespace = json!({
        "type": "namespace",
        "name": "weather",
        "description": "Weather tools",
        "namespace_private_extra": "namespace-extra-sentinel",
        "tools": [{
            "type": "function",
            "name": "forecast",
            "description": "Forecast member sentinel",
            "parameters": {
                "type": "object",
                "properties": {"member-schema-sentinel": {"type": "string"}}
            },
            "defer_loading": true
        }]
    });
    let public = request(
        json!([search_declaration()]),
        json!([
            search_call("call_search_namespace"),
            search_output("call_search_namespace", vec![namespace])
        ]),
    );
    let mut state = ToolSearchState::build(&public).expect("loaded namespace prepares without transport behavior");

    let private = private_request(&mut state, &public);
    let private_input = serde_json::to_value(&private.input).expect("private input serializes");
    assert_eq!(
        private_input[1]["output"],
        r#"{"tools":[{"type":"namespace","name":"weather","description":"Weather tools"}]}"#
    );
    let private_tools = serde_json::to_string(&private.tools).expect("private tools serialize");
    assert!(
        private_tools.contains("forecast"),
        "loaded member must remain available to request lowering"
    );
    assert!(private_tools.contains("member-schema-sentinel"));
    let private_input = private_input.to_string();
    assert!(!private_input.contains("forecast"));
    assert!(!private_input.contains("member-schema-sentinel"));
    assert!(!private_input.contains("namespace-extra-sentinel"));
}

#[test]
fn namespace_partial_load_merges_members_and_is_idempotent() {
    let namespace = json!({
        "type": "namespace",
        "name": "weather",
        "description": "Weather tools",
        "tools": [
            {
                "type": "function",
                "name": "current",
                "description": "Current conditions",
                "parameters": {"type": "object"}
            },
            {
                "type": "function",
                "name": "forecast",
                "description": "Weather forecast",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                },
                "defer_loading": true
            },
            {
                "type": "function",
                "name": "alerts",
                "description": "Weather alerts",
                "parameters": {"type": "object"},
                "defer_loading": true
            }
        ]
    });
    let loaded_subset = json!({
        "type": "namespace",
        "name": "weather",
        "description": "Weather tools",
        "tools": [namespace["tools"][1].clone()]
    });
    let public_request = request(
        json!([search_declaration(), namespace.clone()]),
        json!([
            search_call("call_namespace_1"),
            search_output("call_namespace_1", vec![loaded_subset.clone()]),
            search_call("call_namespace_2"),
            search_output("call_namespace_2", vec![loaded_subset.clone()])
        ]),
    );
    let mut state =
        ToolSearchState::build(&public_request).expect("same-name partial namespace output merges exact members");

    let public = tool_values(state.public_effective_tools());
    assert_eq!(
        public[1], namespace,
        "public declaration and member deferral are preserved"
    );
    let loaded = tool_values(Some(state.loaded_public_tools()));
    assert_eq!(loaded, json!([loaded_subset]), "exact member reload is idempotent");
    let private = private_tool_values(&mut state, &public_request);
    assert_eq!(private[1]["tools"].as_array().map(Vec::len), Some(2));
    assert!(private[1]["tools"][0].get("defer_loading").is_none());
    assert!(private[1]["tools"][1].get("defer_loading").is_none());
    assert_eq!(
        synthetic_description(&state),
        "Search the client tool catalog. Available catalog entry: weather — Weather tools.",
        "the remaining deferred member keeps only namespace identity in the catalog"
    );
    assert!(!synthetic_description(&state).contains("alerts"));

    assert_namespace_availability_counter_transitions(&namespace, &loaded_subset);

    let mut conflicting_subset = loaded_subset;
    conflicting_subset["tools"][0]["parameters"]["properties"]["city"]["type"] = json!("integer");
    assert!(
        ToolSearchState::build(&request(
            json!([search_declaration(), namespace]),
            json!([
                search_call("call_namespace_conflict"),
                search_output("call_namespace_conflict", vec![conflicting_subset])
            ]),
        ))
        .is_err(),
        "same member identity with changed schema must conflict"
    );
}

fn assert_namespace_availability_counter_transitions(namespace: &Value, loaded_subset: &Value) {
    let alerts_subset = json!({
        "type": "namespace",
        "name": "weather",
        "description": "Weather tools",
        "tools": [namespace["tools"][2].clone()]
    });
    let fully_loaded = ToolSearchState::build(&request(
        json!([search_declaration(), namespace]),
        json!([
            search_call("call_forecast"),
            search_output("call_forecast", vec![(*loaded_subset).clone()]),
            search_call("call_forecast_repeat"),
            search_output("call_forecast_repeat", vec![(*loaded_subset).clone()]),
            search_call("call_alerts"),
            search_output("call_alerts", vec![alerts_subset.clone()]),
            search_call("call_alerts_repeat"),
            search_output("call_alerts_repeat", vec![alerts_subset])
        ]),
    ))
    .expect("each deferred member transition is counted once");
    assert_eq!(synthetic_description(&fully_loaded), "Search the client tool catalog");
    assert_eq!(
        fully_loaded.loaded_public_tools().len(),
        2,
        "an exact repeat neither decrements availability twice nor duplicates the loaded subset"
    );
}

#[test]
fn active_namespace_validation_reserves_catalog_identity_and_rejects_unsupported_shapes() {
    let invalid = [
        json!({
            "type": "namespace",
            "name": "tool_search",
            "description": "Reserved catalog identity",
            "tools": [{"type": "function", "name": "run", "defer_loading": true}]
        }),
        json!({"type": "namespace", "name": "empty", "tools": []}),
        json!({
            "type": "namespace",
            "name": "unknown_member",
            "tools": [{"type": "future_member", "opaque": true}]
        }),
    ];
    for namespace in invalid {
        assert!(
            ToolSearchState::build(&request(json!([search_declaration(), namespace]), json!("find a tool"),)).is_err()
        );
    }

    assert!(
        ToolSearchState::build(&request(
            json!([search_declaration()]),
            json!([
                search_call("call_reserved_namespace"),
                search_output(
                    "call_reserved_namespace",
                    vec![json!({
                        "type": "namespace",
                        "name": "tool_search",
                        "tools": [{"type": "function", "name": "run"}]
                    })]
                )
            ]),
        ))
        .is_err(),
        "a dynamically returned namespace catalog identity is also reserved"
    );

    let ordinary = request(
        json!([{
            "type": "namespace",
            "name": "tool_search",
            "tools": [{"type": "function", "name": "run"}]
        }]),
        json!("ordinary namespace"),
    );
    assert!(
        ToolSearchState::build(&ordinary).is_ok(),
        "the namespace identity is reserved only while tool search is active"
    );
}

#[test]
fn namespace_tool_choice_rejects_withheld_member_and_accepts_loaded_member() {
    let namespace = json!({
        "type": "namespace",
        "name": "weather",
        "tools": [{
            "type": "function",
            "name": "forecast",
            "parameters": {"type": "object"},
            "defer_loading": true
        }]
    });
    let mut withheld = request(json!([search_declaration(), namespace.clone()]), json!("find weather"));
    withheld.tool_choice = Some(
        serde_json::from_value(json!({"type": "function", "namespace": "weather", "name": "forecast"}))
            .expect("namespaced choice"),
    );
    let mut state = ToolSearchState::build(&withheld).expect("state preparation succeeds before readiness check");
    let mut private = withheld.clone();
    let error = state
        .prepare_inference_request(&mut private)
        .expect_err("withheld namespace member cannot be forced");
    assert!(error.to_string().contains("before its definition is loaded"));

    let mut loaded = request(
        json!([search_declaration(), namespace.clone()]),
        json!([
            search_call("call_namespace"),
            search_output("call_namespace", vec![namespace])
        ]),
    );
    loaded.tool_choice.clone_from(&withheld.tool_choice);
    let mut state = ToolSearchState::build(&loaded).expect("exact namespace member loads");
    let private = private_request(&mut state, &loaded);
    let upstream = serde_json::to_value(private.to_upstream_request(false).expect("private request lowers"))
        .expect("upstream request serializes");
    assert_eq!(
        upstream["tool_choice"]["name"],
        model_visible_namespace_member_name("weather", "forecast")
    );
    assert!(upstream["tool_choice"].get("namespace").is_none());
}

#[test]
fn namespace_history_rejects_exact_withheld_calls_and_lowers_loaded_calls() {
    let namespace = json!({
        "type": "namespace",
        "name": "weather",
        "tools": [{
            "type": "function",
            "name": "forecast",
            "parameters": {"type": "object"},
            "defer_loading": true
        }]
    });
    let flat_name = model_visible_namespace_member_name("weather", "forecast");
    for call in [
        json!({
            "type": "function_call", "id": "fc_public", "call_id": "call_public",
            "namespace": "weather", "name": "forecast", "arguments": "{}", "status": "completed"
        }),
        json!({
            "type": "function_call", "id": "fc_flat", "call_id": "call_flat",
            "name": flat_name, "arguments": "{}", "status": "completed"
        }),
    ] {
        assert!(
            ToolSearchState::build(&request(
                json!([search_declaration(), namespace.clone()]),
                json!([call]),
            ))
            .is_err(),
            "an exact known-but-withheld history call must fail state preparation"
        );
    }

    let loaded_call = json!({
        "type": "function_call", "id": "fc_loaded", "call_id": "call_loaded",
        "namespace": "weather", "name": "forecast", "arguments": "{}", "status": "completed"
    });
    let loaded_request = request(
        json!([search_declaration(), namespace.clone()]),
        json!([
            search_call("call_load_namespace"),
            search_output("call_load_namespace", vec![namespace.clone()]),
            loaded_call
        ]),
    );
    let mut state = ToolSearchState::build(&loaded_request).expect("loaded known history call remains valid");
    let private = private_request(&mut state, &loaded_request);
    let upstream = serde_json::to_value(private.to_upstream_request(false).expect("loaded history lowers"))
        .expect("upstream request serializes");
    assert_eq!(upstream["input"][2]["name"], flat_name);
    assert!(upstream["input"][2].get("namespace").is_none());

    let unknown = request(
        json!([search_declaration(), namespace]),
        json!([{
            "type": "function_call", "id": "fc_unknown", "call_id": "call_unknown",
            "namespace": "other", "name": "ordinary", "arguments": "{}", "status": "completed"
        }]),
    );
    ToolSearchState::build(&unknown).expect("unknown ordinary calls retain existing behavior");
}

#[test]
fn top_level_function_availability_follows_ordered_search_history() {
    let deferred = function("get_weather", "Get weather", "string", true);
    let call = json!({
        "type": "function_call", "id": "fc_weather", "call_id": "call_weather",
        "name": "get_weather", "arguments": "{}", "status": "completed"
    });

    assert!(
        ToolSearchState::build(&request(
            json!([search_declaration(), deferred.clone()]),
            json!([call.clone()]),
        ))
        .is_err(),
        "an initially deferred function cannot be called before it is loaded"
    );

    assert!(
        ToolSearchState::build(&request(
            json!([search_declaration()]),
            json!([
                call.clone(),
                search_call("call_dynamic"),
                search_output("call_dynamic", vec![deferred.clone()]),
            ]),
        ))
        .is_err(),
        "a dynamically returned function cannot resolve an earlier call"
    );

    ToolSearchState::build(&request(
        json!([search_declaration(), deferred.clone()]),
        json!([
            search_call("call_load"),
            search_output("call_load", vec![deferred]),
            call,
        ]),
    ))
    .expect("the same function is available after its ordered load point");

    ToolSearchState::build(&request(
        json!([search_declaration()]),
        json!([{
            "type": "function_call", "id": "fc_unknown", "call_id": "call_unknown",
            "name": "ordinary_client_function", "arguments": "{}", "status": "completed"
        }]),
    ))
    .expect("unrelated unknown client functions retain existing behavior");

    let immediate = function("get_weather", "Get weather", "string", false);
    ToolSearchState::build(&request(
        json!([search_declaration(), immediate.clone()]),
        json!([
            {
                "type": "function_call", "id": "fc_immediate", "call_id": "call_immediate",
                "name": "get_weather", "arguments": "{}", "status": "completed"
            },
            search_call("call_identical"),
            search_output("call_identical", vec![immediate]),
        ]),
    ))
    .expect("an immediate function stays available before an identical search result");
}

#[test]
fn top_level_function_tool_choices_require_the_definition_to_be_loaded() {
    let deferred = function("get_weather", "Get weather", "string", true);
    for choice in [
        json!({"type": "function", "name": "get_weather"}),
        json!({
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "function", "name": "get_weather"}]
        }),
    ] {
        let mut withheld = request(json!([search_declaration(), deferred.clone()]), json!("find weather"));
        withheld.tool_choice = Some(serde_json::from_value(choice).expect("function tool choice"));
        let mut state = ToolSearchState::build(&withheld).expect("state preparation succeeds before choice validation");
        let mut private = withheld.clone();
        state
            .prepare_inference_request(&mut private)
            .expect_err("a withheld function cannot be selected");
    }

    let mut loaded = request(
        json!([search_declaration(), deferred.clone()]),
        json!([
            search_call("call_load_choice"),
            search_output("call_load_choice", vec![deferred]),
        ]),
    );
    loaded.tool_choice =
        Some(serde_json::from_value(json!({"type": "function", "name": "get_weather"})).expect("function tool choice"));
    let mut state = ToolSearchState::build(&loaded).expect("loaded state");
    private_request(&mut state, &loaded);
}

#[test]
fn dynamically_returned_namespace_members_start_loaded_without_catalog_debt() {
    let dynamic_namespace = json!({
        "type": "namespace",
        "name": "dynamic_weather",
        "description": "Dynamically returned weather tools",
        "tools": [{
            "type": "function",
            "name": "forecast",
            "parameters": {"type": "object"},
            "defer_loading": true
        }]
    });
    let public = request(
        json!([search_declaration()]),
        json!([
            search_call("call_dynamic_namespace"),
            search_output("call_dynamic_namespace", vec![dynamic_namespace])
        ]),
    );
    let mut state = ToolSearchState::build(&public).expect("dynamically returned namespace members are already loaded");

    assert_eq!(synthetic_description(&state), "Search the client tool catalog");
    let private = private_tool_values(&mut state, &public);
    assert_eq!(private[1]["tools"].as_array().map(Vec::len), Some(1));
    assert!(private[1]["tools"][0].get("defer_loading").is_none());
}

#[test]
fn dynamic_namespace_history_rejects_forward_references_but_accepts_valid_order_and_unknowns() {
    let dynamic_namespace = json!({
        "type": "namespace",
        "name": "dynamic_weather",
        "tools": [{
            "type": "function", "name": "forecast", "parameters": {"type": "object"},
            "defer_loading": true
        }]
    });
    let flat_name = model_visible_namespace_member_name("dynamic_weather", "forecast");
    let public_call = json!({
        "type": "function_call", "id": "fc_public", "call_id": "call_public",
        "namespace": "dynamic_weather", "name": "forecast", "arguments": "{}", "status": "completed"
    });
    let flat_call = json!({
        "type": "function_call", "id": "fc_flat", "call_id": "call_flat",
        "name": flat_name, "arguments": "{}", "status": "completed"
    });
    for call in [public_call.clone(), flat_call.clone()] {
        assert!(
            ToolSearchState::build(&request(
                json!([search_declaration()]),
                json!([
                    call,
                    search_call("call_dynamic"),
                    search_output("call_dynamic", vec![dynamic_namespace.clone()])
                ]),
            ))
            .is_err(),
            "a call cannot forward-reference a namespace member loaded later in ordered history"
        );
    }

    for call in [public_call, flat_call] {
        let valid = request(
            json!([search_declaration()]),
            json!([
                search_call("call_dynamic"),
                search_output("call_dynamic", vec![dynamic_namespace.clone()]),
                call
            ]),
        );
        let mut state = ToolSearchState::build(&valid).expect("output-before-call order is valid");
        let private = private_request(&mut state, &valid);
        let upstream = serde_json::to_value(private.to_upstream_request(false).expect("valid history lowers"))
            .expect("upstream request serializes");
        assert_eq!(upstream["input"][2]["name"], flat_name);
        assert!(upstream["input"][2].get("namespace").is_none());
    }

    for unknown in [
        json!({
            "type": "function_call", "id": "fc_unknown_public", "call_id": "call_unknown_public",
            "namespace": "never_loaded", "name": "ordinary", "arguments": "{}", "status": "completed"
        }),
        json!({
            "type": "function_call", "id": "fc_unknown_flat", "call_id": "call_unknown_flat",
            "name": "agentic_ns__never_loaded__ordinary", "arguments": "{}", "status": "completed"
        }),
    ] {
        ToolSearchState::build(&request(json!([search_declaration()]), json!([unknown])))
            .expect("a call that remains unknown retains existing behavior");
    }
}

#[test]
fn deferred_declarations_do_not_implicitly_synthesize_search() {
    let request = request(
        json!([function("get_weather", "Get weather", "string", true)]),
        json!("find weather"),
    );

    assert!(
        ToolSearchState::build(&request).is_err(),
        "defer_loading activates routing safety but requires a declaration or replayed search state"
    );
}

#[test]
fn declaration_free_manual_replay_builds_loaded_views_without_a_synthetic_declaration() {
    let dynamic = function("get_weather", "Get weather", "string", true);
    let public_request = request(
        json!([]),
        json!([
            search_call("call_search_1"),
            search_output("call_search_1", vec![dynamic.clone()])
        ]),
    );
    let mut state =
        ToolSearchState::build(&public_request).expect("manual public replay is valid without redeclaring tool_search");

    assert!(state.is_active());
    assert!(state.synthetic_tool_search().is_none());
    assert_eq!(tool_values(Some(state.loaded_public_tools())), json!([dynamic]));
    let public = tool_values(state.public_effective_tools());
    let private = private_tool_values(&mut state, &public_request);
    assert_eq!(public[0]["defer_loading"], true);
    assert!(private[0].get("defer_loading").is_none());
}

#[test]
fn prepare_inference_request_consumes_prepared_views_without_mutating_public_source() {
    let deferred = function("get_weather", "Get weather", "string", true);
    let public = request(
        json!([search_declaration(), deferred.clone()]),
        json!([
            {"type": "message", "role": "user", "content": "find weather"},
            search_call("call_search_1"),
            search_output("call_search_1", vec![deferred.clone()])
        ]),
    );
    let public_before = serde_json::to_value(&public).expect("public request serializes");
    let mut state = ToolSearchState::build(&public).expect("valid function-only state");
    assert_eq!(tool_values(Some(state.loaded_public_tools())), json!([deferred]));
    assert_eq!(synthetic_description(&state), "Search the client tool catalog");

    let private = private_request(&mut state, &public);

    assert_eq!(
        serde_json::to_value(&public).expect("public request still serializes"),
        public_before,
        "private lowering must not mutate the public request"
    );
    let private_value = serde_json::to_value(&private).expect("private request serializes");
    assert_eq!(private_value["input"][1]["type"], "function_call");
    assert_eq!(private_value["input"][1]["name"], "tool_search");
    assert_eq!(private_value["input"][1]["call_id"], "call_search_1");
    assert_eq!(private_value["input"][2]["type"], "function_call_output");
    assert_eq!(private_value["input"][2]["call_id"], "call_search_1");
    assert_eq!(private_value["tools"].as_array().map(Vec::len), Some(2));
    assert_eq!(private_value["tools"][0]["type"], "tool_search");
    assert_eq!(private_value["tools"][0]["execution"], "client");
    assert_eq!(private_value["tools"][1]["name"], "get_weather");
    assert!(private_value["tools"][1].get("defer_loading").is_none());

    let upstream = serde_json::to_value(private.to_upstream_request(false).expect("prepared request lowers"))
        .expect("upstream request serializes");
    assert_eq!(upstream["tools"][0]["type"], "function");
    assert_eq!(upstream["tools"][0]["name"], "tool_search");
}

#[test]
fn safe_catalog_is_minimal_and_never_exposes_deferred_configuration() {
    let request = request(
        json!([
            search_declaration(),
            {
                "type": "function",
                "name": "hidden_function",
                "description": "Safe function description",
                "parameters": {
                    "type": "object",
                    "properties": {"secret_parameter": {"type": "string"}}
                },
                "defer_loading": true
            },
            {
                "type": "namespace",
                "name": "hidden_namespace",
                "description": "Safe namespace description",
                "tools": [{
                    "type": "function",
                    "name": "secret_member",
                    "description": "secret member description",
                    "parameters": {"type": "object", "properties": {"secret": {"type": "string"}}},
                    "defer_loading": true
                }]
            }
        ]),
        json!("find a tool"),
    );

    let public_before = serde_json::to_value(&request).expect("public request serializes");
    let mut state = ToolSearchState::build(&request).expect("catalog construction is pure");
    drop(private_request(&mut state, &request));
    assert_eq!(
        serde_json::to_value(&request).expect("public request still serializes"),
        public_before,
        "private materialization must preserve public deferred configuration"
    );
    assert_eq!(
        synthetic_description(&state),
        "Search the client tool catalog. Available catalog entries: hidden_function — Safe function description; \
hidden_namespace — Safe namespace description."
    );

    let model_visible =
        serde_json::to_string(&state.synthetic_tool_search()).expect("synthetic declaration serializes");
    for secret in ["secret_parameter", "secret_member", "secret member description"] {
        assert!(!model_visible.contains(secret), "catalog leaked {secret}");
    }
}

#[test]
fn replay_restores_loaded_deferred_tool_after_compaction_removed_search_pair() {
    let deferred = json!({
        "type": "function",
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object"},
        "defer_loading": true
    });
    let request = request(
        json!([search_declaration(), deferred]),
        json!([{
            "type": "compaction",
            "encrypted_content": "The weather tool was loaded earlier."
        }]),
    );
    let restored: Vec<agentic_core::types::tools::ResponsesTool> = serde_json::from_value(json!([{
        "type": "function",
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object"},
        "defer_loading": true
    }]))
    .expect("valid restored loaded definitions");

    let mut state = ToolSearchState::build_with_loaded_tools(&request, &restored, false)
        .expect("typed metadata restores compaction-lost loaded state");

    assert_eq!(state.loaded_public_tools().len(), 1);
    assert_eq!(synthetic_description(&state), "Search the client tool catalog");
    let private_request = private_request(&mut state, &request);
    let private = private_request.tools.as_deref().expect("private tools");
    let loaded = private
        .iter()
        .find_map(|tool| match tool {
            agentic_core::types::tools::ResponsesTool::Function(function)
                if function.name.as_str() == "get_weather" =>
            {
                Some(function)
            }
            _ => None,
        })
        .expect("loaded function is effective upstream");
    assert_eq!(loaded.defer_loading, None);
}

#[test]
fn compacted_replay_does_not_reload_definition_omitted_by_explicit_tools() {
    let request = request(
        json!([search_declaration()]),
        json!([
            {
                "type": "tool_search_call",
                "id": "tsc_obsolete",
                "call_id": "call_obsolete",
                "arguments": {"query": "weather"}
            },
            {
                "type": "tool_search_output",
                "call_id": "call_obsolete",
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                    "defer_loading": true
                }]
            },
            {
                "type": "compaction",
                "encrypted_content": "The old search pair is superseded."
            },
            {
                "type": "message",
                "role": "user",
                "content": "Continue without weather"
            }
        ]),
    );
    let restored: Vec<agentic_core::types::tools::ResponsesTool> = serde_json::from_value(json!([{
        "type": "function",
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object"},
        "defer_loading": true
    }]))
    .expect("stored loaded definition");

    let mut state = ToolSearchState::build_with_loaded_tools(&request, &restored, true)
        .expect("explicit tools define the post-compaction catalog");
    assert!(state.loaded_public_tools().is_empty());
    assert!(state
        .public_effective_tools()
        .unwrap()
        .iter()
        .all(|tool| !matches!(tool, agentic_core::types::tools::ResponsesTool::Function(function) if function.name.as_str() == "get_weather")));
    assert!(private_request(&mut state, &request)
        .tools
        .as_deref()
        .expect("private tools")
        .iter()
        .all(|tool| !matches!(tool, agentic_core::types::tools::ResponsesTool::Function(function) if function.name.as_str() == "get_weather")));
}

#[test]
fn replayed_loaded_marker_rejects_explicit_cross_kind_identity_collision() {
    let request = request(
        json!([
            search_declaration(),
            {
                "type": "namespace",
                "name": "shared_identity",
                "description": "Replacement namespace",
                "tools": [{
                    "type": "function",
                    "name": "member",
                    "parameters": {"type": "object"}
                }]
            }
        ]),
        json!([{
            "type": "compaction",
            "encrypted_content": "A function with this name was loaded earlier."
        }]),
    );
    let restored: Vec<agentic_core::types::tools::ResponsesTool> = serde_json::from_value(json!([{
        "type": "function",
        "name": "shared_identity",
        "description": "Original function",
        "parameters": {"type": "object"},
        "defer_loading": true
    }]))
    .expect("stored function marker");

    assert!(ToolSearchState::build_with_loaded_tools(&request, &restored, true).is_err());
}
