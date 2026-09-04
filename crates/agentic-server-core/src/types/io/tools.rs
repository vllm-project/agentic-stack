use serde::{Deserialize, Deserializer, Serialize, Serializer, de, ser::SerializeMap};
use serde_json::Value;

use crate::types::tools::{NonEmptyToolName, ResponsesTool};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct FunctionTool {
    #[serde(rename = "type")]
    pub type_: String,
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Function {
        namespace: Option<String>,
        name: NonEmptyToolName,
    },
    Custom {
        name: NonEmptyToolName,
    },
    AllowedTools {
        mode: AllowedToolsMode,
        tools: Vec<AllowedTool>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct AllowedTool {
    #[serde(rename = "type")]
    pub type_: NonEmptyToolName,
    pub name: NonEmptyToolName,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum AllowedToolsMode {
    Auto,
    Required,
}

#[cfg(feature = "openapi")]
impl utoipa::PartialSchema for ToolChoice {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::{
            ObjectBuilder, Ref,
            schema::{ArrayBuilder, OneOfBuilder, SchemaType, Type},
        };

        let str_type = || ObjectBuilder::new().schema_type(SchemaType::new(Type::String));

        OneOfBuilder::new()
            .item(str_type().enum_values(Some(["auto", "none", "required"])))
            .item(
                ObjectBuilder::new()
                    .property("type", str_type().enum_values(Some(["function"])))
                    .required("type")
                    .property("name", str_type())
                    .required("name")
                    .property("namespace", str_type()),
            )
            .item(
                ObjectBuilder::new()
                    .property("type", str_type().enum_values(Some(["custom"])))
                    .required("type")
                    .property("name", str_type())
                    .required("name"),
            )
            .item(
                ObjectBuilder::new()
                    .property("type", str_type().enum_values(Some(["allowed_tools"])))
                    .required("type")
                    .property("mode", Ref::from_schema_name("AllowedToolsMode"))
                    .required("mode")
                    .property("tools", ArrayBuilder::new().items(Ref::from_schema_name("AllowedTool")))
                    .required("tools"),
            )
            .into()
    }
}
#[cfg(feature = "openapi")]
impl utoipa::ToSchema for ToolChoice {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("ToolChoice")
    }
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::None => serializer.serialize_str("none"),
            Self::Required => serializer.serialize_str("required"),
            Self::Function { namespace, name } => {
                let mut map = serializer.serialize_map(Some(2 + usize::from(namespace.is_some())))?;
                map.serialize_entry("type", "function")?;
                if let Some(namespace) = namespace {
                    map.serialize_entry("namespace", namespace)?;
                }
                map.serialize_entry("name", name.as_str())?;
                map.end()
            }
            Self::Custom { name } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "custom")?;
                map.serialize_entry("name", name.as_str())?;
                map.end()
            }
            Self::AllowedTools { mode, tools } => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("type", "allowed_tools")?;
                map.serialize_entry("mode", mode)?;
                map.serialize_entry("tools", tools)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(choice) => match choice.as_str() {
                "auto" => Ok(Self::Auto),
                "none" => Ok(Self::None),
                "required" => Ok(Self::Required),
                other => Err(de::Error::unknown_variant(
                    other,
                    &["auto", "none", "required", "function", "custom"],
                )),
            },
            Value::Object(object) => {
                if object.get("type").and_then(Value::as_str) == Some("function") {
                    let namespace = object.get("namespace").and_then(Value::as_str).map(str::to_string);
                    let name = object
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| de::Error::missing_field("name"))?;
                    let name = NonEmptyToolName::try_from(name).map_err(de::Error::custom)?;
                    return Ok(Self::Function { namespace, name });
                }

                if object.get("type").and_then(Value::as_str) == Some("custom") {
                    let name = object
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| de::Error::missing_field("name"))?;
                    let name = NonEmptyToolName::try_from(name).map_err(de::Error::custom)?;
                    return Ok(Self::Custom { name });
                }

                if object.get("type").and_then(Value::as_str) == Some("allowed_tools") {
                    let mode = object
                        .get("mode")
                        .cloned()
                        .ok_or_else(|| de::Error::missing_field("mode"))?;
                    let mode = serde_json::from_value(mode).map_err(de::Error::custom)?;
                    let tools = object
                        .get("tools")
                        .cloned()
                        .ok_or_else(|| de::Error::missing_field("tools"))?;
                    let tools = serde_json::from_value(tools).map_err(de::Error::custom)?;
                    return Ok(Self::AllowedTools { mode, tools });
                }

                if let Some(function) = object.get("function").and_then(Value::as_object) {
                    let namespace = function.get("namespace").and_then(Value::as_str).map(str::to_string);
                    let name = function
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| de::Error::missing_field("name"))?;
                    let name = NonEmptyToolName::try_from(name).map_err(de::Error::custom)?;
                    return Ok(Self::Function { namespace, name });
                }

                Err(de::Error::custom(
                    "expected tool_choice string, named tool object, or allowed_tools object",
                ))
            }
            _ => Err(de::Error::custom(
                "expected tool_choice string, named tool object, or allowed_tools object",
            )),
        }
    }
}

impl ToolChoice {
    /// Converts client-facing custom-tool selectors to the function-tool shape
    /// used by the normalized upstream tool declarations.
    #[must_use]
    pub(crate) fn normalized_for_upstream(&self) -> Self {
        match self {
            Self::Custom { name } => Self::Function {
                namespace: None,
                name: name.clone(),
            },
            Self::AllowedTools { mode, tools } => Self::AllowedTools {
                mode: *mode,
                tools: tools.iter().cloned().map(normalize_allowed_tool).collect(),
            },
            choice => choice.clone(),
        }
    }
}

fn normalize_allowed_tool(mut tool: AllowedTool) -> AllowedTool {
    if tool.type_.as_str() == "custom" {
        tool.type_ = NonEmptyToolName::try_from("function").expect("function is a non-empty tool type");
    }
    tool
}

/// Returns the effective tool list, preferring `request_tools` when explicitly
/// set by the caller, otherwise falling back to the stored configuration.
#[inline]
pub(crate) fn resolve_tools(
    request_tools: Option<&[ResponsesTool]>,
    stored_tools: Option<&[ResponsesTool]>,
    tools_explicitly_set: bool,
) -> Option<Vec<ResponsesTool>> {
    if tools_explicitly_set {
        request_tools
    } else {
        stored_tools
    }
    .map(<[_]>::to_vec)
}

/// Returns the effective tool choice using the same precedence as [`resolve_tools`].
#[inline]
pub(crate) fn resolve_tool_choice(
    request_choice: Option<&ToolChoice>,
    stored_choice: &ToolChoice,
    explicitly_set: bool,
) -> ToolChoice {
    if explicitly_set {
        request_choice.cloned().unwrap_or_default()
    } else {
        stored_choice.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_tool_choice_rejects_empty_name() {
        assert!(
            serde_json::from_value::<ToolChoice>(serde_json::json!({
                "type": "function",
                "name": ""
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<ToolChoice>(serde_json::json!({
                "function": {
                    "name": ""
                }
            }))
            .is_err()
        );
    }

    #[test]
    fn custom_tool_choice_round_trips() {
        let custom = serde_json::json!({
            "type": "custom",
            "name": "apply_patch"
        });

        let choice: ToolChoice = serde_json::from_value(custom.clone()).unwrap();
        assert_eq!(
            choice,
            ToolChoice::Custom {
                name: NonEmptyToolName::try_from("apply_patch").unwrap()
            }
        );
        assert_eq!(serde_json::to_value(choice).unwrap(), custom);
    }

    #[test]
    fn custom_tool_choice_rejects_empty_name() {
        assert!(
            serde_json::from_value::<ToolChoice>(serde_json::json!({
                "type": "custom",
                "name": ""
            }))
            .is_err()
        );
    }

    #[test]
    fn allowed_tools_round_trip() {
        let expected = serde_json::json!({
            "type": "allowed_tools",
            "mode": "required",
            "tools": [
                {"type": "function", "name": "get_weather"},
                {"type": "custom", "name": "code_exec"}
            ]
        });

        let choice: ToolChoice = serde_json::from_value(expected.clone()).unwrap();
        assert_eq!(serde_json::to_value(choice).unwrap(), expected);
    }

    #[test]
    fn allowed_tools_require_non_empty_type_and_name() {
        for invalid_tool in [
            serde_json::json!({"type": "function"}),
            serde_json::json!({"type": "", "name": "get_weather"}),
            serde_json::json!({"type": "function", "name": ""}),
        ] {
            assert!(
                serde_json::from_value::<ToolChoice>(serde_json::json!({
                    "type": "allowed_tools",
                    "mode": "auto",
                    "tools": [invalid_tool]
                }))
                .is_err()
            );
        }
    }
}
