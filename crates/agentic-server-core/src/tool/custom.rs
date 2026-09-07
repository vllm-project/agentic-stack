use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::events::WireEvent;
use crate::types::io::{CustomToolCall, FunctionTool, FunctionToolCall, OutputItem, ToolChoice};
use crate::types::tools::{CustomToolParam, ResponsesTool};
use crate::utils::common::serialize_to_value_or_custom_default;

use super::{ToolEntry, ToolError, ToolHandler, ToolType};

/// Request-scoped mapping from normalized function names to their original
/// public custom-tool declarations.
#[derive(Debug, Default)]
pub(crate) struct CustomToolMap {
    declarations: HashMap<String, CustomToolParam>,
}

impl CustomToolMap {
    fn from_tools(tools: &[ResponsesTool]) -> Option<Self> {
        let declarations = tools
            .iter()
            .filter_map(|tool| match tool {
                ResponsesTool::Custom(param) => Some((param.name.as_str().to_owned(), param.clone())),
                _ => None,
            })
            .collect::<HashMap<_, _>>();
        (!declarations.is_empty()).then_some(Self { declarations })
    }

    fn declaration(&self, name: &str) -> Option<&CustomToolParam> {
        self.declarations.get(name)
    }
}

/// Handler for client-owned `type: "custom"` tools.
///
/// Custom tools are normalized for the model but are executed by the client,
/// so this intentionally implements [`ToolHandler`] without
/// [`super::GatewayExecutor`].
#[derive(Debug)]
pub struct CustomHandler;

impl CustomHandler {
    #[must_use]
    pub(crate) fn build_tool_map(tools: &[ResponsesTool]) -> Option<CustomToolMap> {
        CustomToolMap::from_tools(tools)
    }

    pub(crate) fn validate_tool_choice(
        tools: Option<&[ResponsesTool]>,
        tool_choice: &ToolChoice,
    ) -> Result<(), ToolError> {
        let map = tools.and_then(CustomToolMap::from_tools);
        match tool_choice {
            ToolChoice::Custom { name } => validate_custom_selector(map.as_ref(), name.as_str()),
            ToolChoice::AllowedTools { tools, .. } => {
                for tool in tools {
                    if tool.type_.as_str() == "custom" {
                        validate_custom_selector(map.as_ref(), tool.name.as_str())?;
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    #[must_use]
    pub fn to_function_call(param: &CustomToolParam) -> FunctionTool {
        FunctionTool {
            type_: "function".to_owned(),
            name: param.name.as_str().to_owned(),
            description: Some(model_visible_description(param)),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Raw custom tool input. Follow the tool description and declared format exactly."
                    }
                },
                "required": ["input"],
                "additionalProperties": false
            })),
            strict: Some(true),
        }
    }

    #[must_use]
    pub(crate) fn output_item(call: &FunctionToolCall) -> OutputItem {
        OutputItem::CustomToolCall(CustomToolCall {
            id: public_item_id(&call.id),
            status: Some(call.status),
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            input: input_from_arguments(&call.arguments),
        })
    }

    /// Restores normalized custom-tool declarations in response lifecycle
    /// metadata before the event is emitted to the client.
    pub(crate) fn restore_response_wire(wire: &mut WireEvent, map: Option<&CustomToolMap>) -> bool {
        let Some(map) = map else {
            return false;
        };
        restore_response_map(&mut wire.rest, map)
    }
}

fn validate_custom_selector(map: Option<&CustomToolMap>, name: &str) -> Result<(), ToolError> {
    if map.and_then(|map| map.declaration(name)).is_some() {
        return Ok(());
    }
    Err(ToolError::Config(format!(
        "tool_choice selects custom tool '{name}', but no matching custom tool is declared"
    )))
}

fn restore_response_map(object: &mut Map<String, Value>, map: &CustomToolMap) -> bool {
    let mut changed = restore_response_metadata(object, map);
    for key in ["response", "payload"] {
        if let Some(nested) = object.get_mut(key).and_then(Value::as_object_mut) {
            changed |= restore_response_map(nested, map);
        }
    }
    changed
}

fn restore_response_metadata(object: &mut Map<String, Value>, map: &CustomToolMap) -> bool {
    let mut changed = false;
    if let Some(tools) = object.get_mut("tools").and_then(Value::as_array_mut) {
        for tool in tools {
            changed |= restore_custom_declaration(tool, map);
        }
    }
    if let Some(tool_choice) = object.get_mut("tool_choice") {
        changed |= restore_custom_tool_choice(tool_choice, map);
    }
    changed
}

fn restore_custom_declaration(tool: &mut Value, map: &CustomToolMap) -> bool {
    let Some(name) = normalized_custom_name(tool, map) else {
        return false;
    };
    let Some(param) = map.declaration(&name) else {
        return false;
    };
    let Some(mut declaration) =
        serialize_to_value_or_custom_default(param, "custom tool metadata serialization failed", Some, None)
    else {
        return false;
    };
    let Some(object) = declaration.as_object_mut() else {
        return false;
    };
    object.insert("type".to_owned(), Value::String("custom".to_owned()));
    *tool = declaration;
    true
}

fn restore_custom_tool_choice(choice: &mut Value, map: &CustomToolMap) -> bool {
    let Some(object) = choice.as_object_mut() else {
        return false;
    };
    if object.get("type").and_then(Value::as_str) == Some("allowed_tools") {
        let Some(tools) = object.get_mut("tools").and_then(Value::as_array_mut) else {
            return false;
        };
        return tools
            .iter_mut()
            .map(|tool| restore_custom_choice_type(tool, map))
            .fold(false, |changed, restored| changed | restored);
    }
    restore_custom_choice_type(choice, map)
}

fn restore_custom_choice_type(choice: &mut Value, map: &CustomToolMap) -> bool {
    if normalized_custom_name(choice, map).is_none() {
        return false;
    }
    let Some(object) = choice.as_object_mut() else {
        return false;
    };
    object.insert("type".to_owned(), Value::String("custom".to_owned()));
    object.remove("namespace");
    true
}

fn normalized_custom_name(value: &Value, map: &CustomToolMap) -> Option<String> {
    let object = value.as_object()?;
    if object.get("type").and_then(Value::as_str) != Some("function") {
        return None;
    }
    let name = object.get("name")?.as_str()?;
    map.declaration(name).map(|_| name.to_owned())
}

fn model_visible_description(param: &CustomToolParam) -> String {
    let mut fragments = Vec::new();
    if let Some(description) = param
        .description
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        fragments.push(description.to_owned());
    }

    fragments.push("Provide the raw tool input in the `input` string field.".to_owned());

    if !param.extra.is_empty()
        && let Ok(extra) = serde_json::to_string(&param.extra)
    {
        fragments.push(format!(
            "Additional custom tool declaration fields that must be respected:\n{extra}"
        ));
    }

    fragments.join("\n\n")
}

impl ToolHandler for CustomHandler {
    type ToolParams = CustomToolParam;

    fn tool_type(&self) -> ToolType {
        ToolType::Custom
    }

    fn validate(&self, params: &CustomToolParam) -> Result<(), ToolError> {
        if params
            .format
            .as_ref()
            .is_some_and(|format| format.get("type").and_then(Value::as_str) != Some("text"))
        {
            return Err(ToolError::Config(format!(
                "custom tool '{}' uses an unsupported format; gateway normalization cannot preserve constrained decoding",
                params.name
            )));
        }
        Ok(())
    }

    fn normalize(&self, params: &CustomToolParam) -> Vec<FunctionTool> {
        vec![Self::to_function_call(params)]
    }
}

pub(crate) fn insert_custom_entry(entries: &mut HashMap<String, ToolEntry>, param: &CustomToolParam) {
    entries.insert(
        param.name.as_str().to_owned(),
        ToolEntry::client(ToolType::Custom, None),
    );
}

pub(crate) fn public_item_id(item_id: &str) -> String {
    if item_id.starts_with("ctc_") {
        return item_id.to_owned();
    }
    if let Some(suffix) = item_id.strip_prefix("fc_").filter(|suffix| !suffix.is_empty()) {
        return format!("ctc_{suffix}");
    }
    format!("ctc_{:016x}", stable_name_hash(item_id))
}

fn stable_name_hash(value: &str) -> u64 {
    value.as_bytes().iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

pub(crate) fn input_from_arguments(arguments: &str) -> String {
    try_input_from_arguments(arguments).unwrap_or_else(|| {
        tracing::debug!(
            argument_bytes = arguments.len(),
            "custom tool arguments did not match the normalized input envelope; forwarding raw arguments"
        );
        arguments.to_owned()
    })
}

pub(crate) fn try_input_from_arguments(arguments: &str) -> Option<String> {
    match serde_json::from_str::<serde_json::Value>(arguments).ok()? {
        serde_json::Value::String(input) => Some(input),
        serde_json::Value::Object(fields) => fields
            .get("input")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::event::MessageStatus;

    #[test]
    fn function_fallback_uses_public_custom_tool_shape() {
        let call = FunctionToolCall {
            id: "fc_1".to_owned(),
            call_id: "call_1".to_owned(),
            name: "raw_echo".to_owned(),
            namespace: None,
            arguments: r#"{"input":"hello"}"#.to_owned(),
            status: MessageStatus::Completed,
        };

        let OutputItem::CustomToolCall(completed) = CustomHandler::output_item(&call) else {
            panic!("expected custom output item");
        };
        assert_eq!(completed.id, "ctc_1");
        assert_eq!(completed.input, "hello");
        assert_eq!(completed.status, Some(MessageStatus::Completed));
    }

    #[test]
    fn custom_call_id_is_stable_for_every_source_item_id() {
        assert_eq!(public_item_id("fc_item"), "ctc_item");
        assert_eq!(public_item_id("ctc_item"), "ctc_item");
        assert_eq!(public_item_id("provider_item"), public_item_id("provider_item"));
    }

    #[test]
    fn custom_declaration_normalizes_to_function_with_raw_input() {
        let param = serde_json::from_value::<CustomToolParam>(serde_json::json!({
            "name": "raw_echo",
            "description": "Echo raw input.",
            "x-provider-field": {"mode": "strict"}
        }))
        .expect("custom tool");

        let mut tools = CustomHandler.normalize(&param);
        let tool = tools.pop().expect("normalized custom tool");

        assert_eq!(tool.type_, "function");
        assert_eq!(tool.name, "raw_echo");
        assert_eq!(
            tool.parameters.as_ref().unwrap()["properties"]["input"]["type"],
            "string"
        );
        assert_eq!(tool.parameters.as_ref().unwrap()["required"][0], "input");
        let description = tool.description.as_deref().expect("model-visible description");
        assert!(description.contains("Echo raw input."));
        assert!(description.contains("raw tool input in the `input` string field"));
        assert!(description.contains("x-provider-field"));
        assert!(description.contains("strict"));
    }

    #[test]
    fn grammar_formats_are_rejected() {
        for syntax in ["lark", "regex"] {
            let param = serde_json::from_value::<CustomToolParam>(serde_json::json!({
                "name": "constrained_input",
                "format": {
                    "type": "grammar",
                    "syntax": syntax,
                    "definition": "start: value"
                }
            }))
            .expect("custom tool");

            let error = CustomHandler.validate(&param).expect_err("grammar must be rejected");
            assert!(error.to_string().contains("cannot preserve constrained decoding"));
        }
    }

    #[test]
    fn explicit_text_format_is_supported() {
        let param = serde_json::from_value::<CustomToolParam>(serde_json::json!({
            "name": "freeform",
            "format": {"type": "text"}
        }))
        .expect("custom tool");

        CustomHandler
            .validate(&param)
            .expect("unconstrained text is representable");
    }

    #[test]
    fn response_lifecycle_metadata_restores_public_custom_tool_shape() {
        let param = serde_json::from_value::<CustomToolParam>(serde_json::json!({
            "name": "raw_echo",
            "description": "Echo raw input."
        }))
        .expect("custom tool");
        let tools = vec![ResponsesTool::Custom(param)];
        let map = CustomHandler::build_tool_map(&tools);
        let mut wire = WireEvent::new("response.created");
        wire.rest.insert(
            "response".to_owned(),
            serde_json::json!({
                "tools": [{
                    "type": "function",
                    "name": "raw_echo",
                    "description": "normalized description",
                    "parameters": {"type": "object"}
                }],
                "tool_choice": {"type": "function", "name": "raw_echo"}
            }),
        );

        assert!(CustomHandler::restore_response_wire(&mut wire, map.as_ref()));
        let response = &wire.rest["response"];
        assert_eq!(response["tools"][0]["type"], "custom");
        assert_eq!(response["tools"][0]["description"], "Echo raw input.");
        assert!(response["tools"][0].get("parameters").is_none());
        assert_eq!(response["tool_choice"]["type"], "custom");
        assert_eq!(response["tool_choice"]["name"], "raw_echo");
    }

    #[test]
    fn allowed_tools_metadata_restores_custom_selector_type() {
        let param = serde_json::from_value::<CustomToolParam>(serde_json::json!({
            "name": "raw_echo"
        }))
        .expect("custom tool");
        let tools = vec![ResponsesTool::Custom(param)];
        let map = CustomHandler::build_tool_map(&tools);
        let mut wire = WireEvent::new("response.in_progress");
        wire.rest.insert(
            "response".to_owned(),
            serde_json::json!({
                "tool_choice": {
                    "type": "allowed_tools",
                    "mode": "required",
                    "tools": [
                        {"type": "function", "name": "ordinary"},
                        {"type": "function", "name": "raw_echo"}
                    ]
                }
            }),
        );

        assert!(CustomHandler::restore_response_wire(&mut wire, map.as_ref()));
        let tools = &wire.rest["response"]["tool_choice"]["tools"];
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[1]["type"], "custom");
    }
}
