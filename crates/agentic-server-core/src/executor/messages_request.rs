//! Request preparation shared by the native Anthropic Messages tool loops.

use serde_json::{Value, json};

use crate::executor::{ExecutorError, ExecutorResult};
use crate::tool::web_search::web_search_function_tool;
use crate::types::messages::tool_seam::{NATIVE_WEB_SEARCH_TYPE, WEB_SEARCH_EXECUTOR};

/// Request-wide native web-search execution budget.
#[derive(Debug)]
pub(super) struct WebSearchBudget {
    remaining: Option<usize>,
}

impl WebSearchBudget {
    /// Reserve up to `requested` searches and return how many may execute.
    pub(super) fn reserve(&mut self, requested: usize) -> usize {
        match &mut self.remaining {
            Some(remaining) => {
                let allowed = requested.min(*remaining);
                *remaining -= allowed;
                allowed
            }
            None => requested,
        }
    }
}

fn validate_domain_list(tool: &Value, field: &str) -> ExecutorResult<()> {
    let Some(value) = tool.get(field) else {
        return Ok(());
    };
    let valid = value.as_array().is_some_and(|domains| {
        domains.iter().all(|domain| {
            domain.as_str().is_some_and(|domain| {
                let domain = domain.trim();
                !domain.is_empty()
                    && !domain.contains("://")
                    && !domain.starts_with('/')
                    && !domain.chars().any(char::is_whitespace)
            })
        })
    });
    if valid {
        Ok(())
    } else {
        Err(ExecutorError::InvalidRequest(format!(
            "web_search {field} must be an array of non-empty strings"
        )))
    }
}

fn validate_allowed_callers(tool: &Value) -> ExecutorResult<()> {
    let Some(value) = tool.get("allowed_callers") else {
        return Ok(());
    };
    let Some(callers) = value.as_array() else {
        return Err(ExecutorError::InvalidRequest(
            "web_search allowed_callers must be an array of strings".to_owned(),
        ));
    };
    if !callers.iter().all(Value::is_string) {
        return Err(ExecutorError::InvalidRequest(
            "web_search allowed_callers must be an array of strings".to_owned(),
        ));
    }
    if !callers.iter().any(|caller| caller.as_str() == Some("direct")) {
        return Err(ExecutorError::InvalidRequest(
            "web_search allowed_callers must permit direct invocation".to_owned(),
        ));
    }
    Ok(())
}

fn validate_user_location(tool: &Value) -> ExecutorResult<()> {
    let Some(value) = tool.get("user_location") else {
        return Ok(());
    };
    let Some(location) = value.as_object() else {
        return Err(ExecutorError::InvalidRequest(
            "web_search user_location must be an object".to_owned(),
        ));
    };
    if location.get("type").and_then(Value::as_str) != Some("approximate") {
        return Err(ExecutorError::InvalidRequest(
            "web_search user_location.type must be approximate".to_owned(),
        ));
    }
    let fields = ["city", "region", "country", "timezone"];
    let mut has_location = false;
    for field in fields {
        if let Some(value) = location.get(field) {
            let valid = value.as_str().is_some_and(|value| !value.trim().is_empty());
            if !valid {
                return Err(ExecutorError::InvalidRequest(format!(
                    "web_search user_location.{field} must be a non-empty string"
                )));
            }
            has_location = true;
        }
    }
    if !has_location {
        return Err(ExecutorError::InvalidRequest(
            "web_search user_location must include city, region, country, or timezone".to_owned(),
        ));
    }
    Ok(())
}

fn native_web_search_max_uses(request: &Value) -> ExecutorResult<Option<usize>> {
    let Some(tools) = request.get("tools").and_then(Value::as_array) else {
        return Ok(None);
    };
    let mut max_uses = None;

    for tool in tools {
        let tool_type = tool.get("type").and_then(Value::as_str);
        let is_web_search = tool.get("name").and_then(Value::as_str) == Some(WEB_SEARCH_EXECUTOR);
        if is_web_search
            && tool_type
                .is_some_and(|tool_type| tool_type.starts_with("web_search_") && tool_type != NATIVE_WEB_SEARCH_TYPE)
        {
            return Err(ExecutorError::InvalidRequest(format!(
                "unsupported web_search tool type '{}'",
                tool_type.unwrap_or_default()
            )));
        }
        if tool_type != Some(NATIVE_WEB_SEARCH_TYPE) || !is_web_search {
            continue;
        }
        validate_domain_list(tool, "allowed_domains")?;
        validate_domain_list(tool, "blocked_domains")?;
        let has_allowed_domains = tool
            .get("allowed_domains")
            .and_then(Value::as_array)
            .is_some_and(|domains| !domains.is_empty());
        let has_blocked_domains = tool
            .get("blocked_domains")
            .and_then(Value::as_array)
            .is_some_and(|domains| !domains.is_empty());
        if has_allowed_domains && has_blocked_domains {
            return Err(ExecutorError::InvalidRequest(
                "web_search allowed_domains and blocked_domains cannot be used together".to_owned(),
            ));
        }
        validate_user_location(tool)?;
        validate_allowed_callers(tool)?;
        if let Some(value) = tool.get("max_uses") {
            let parsed = value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    ExecutorError::InvalidRequest("web_search max_uses must be a positive integer".to_owned())
                })?;
            max_uses = Some(max_uses.map_or(parsed, |current: usize| current.min(parsed)));
        }
    }

    Ok(max_uses)
}

/// Normalize native web-search declarations for an upstream endpoint that
/// validates ordinary function-tool schemas, returning whether the body changed.
///
/// # Errors
/// Returns [`ExecutorError::InvalidRequest`] for unsupported or invalid native
/// declarations.
pub fn normalize_native_web_search_for_upstream(request: &mut Value) -> ExecutorResult<bool> {
    let had_native = request.get("tools").and_then(Value::as_array).is_some_and(|tools| {
        tools.iter().any(|tool| {
            tool.get("type").and_then(Value::as_str) == Some(NATIVE_WEB_SEARCH_TYPE)
                && tool.get("name").and_then(Value::as_str) == Some(WEB_SEARCH_EXECUTOR)
        })
    });
    normalize_native_web_search(request).map(|_| had_native)
}

pub(super) fn web_search_budget_exhausted_result(tool_use_id: &str) -> Value {
    crate::types::messages::tool_seam::tool_result_block(
        tool_use_id,
        "web_search max_uses exceeded; search was not run",
        true,
    )
}

/// Translate Claude's native server-tool declaration into the ordinary
/// Anthropic function-tool shape accepted by vLLM. The gateway executes the
/// resulting `tool_use` internally, so this is an upstream-only representation
/// change; the client request itself remains Anthropic-native.
pub(super) fn normalize_native_web_search(request: &mut Value) -> ExecutorResult<WebSearchBudget> {
    let max_uses = native_web_search_max_uses(request)?;
    let Some(tools) = request.get_mut("tools").and_then(Value::as_array_mut) else {
        return Ok(WebSearchBudget { remaining: None });
    };
    let function_tool = web_search_function_tool();

    for tool in tools {
        let tool_type = tool.get("type").and_then(Value::as_str);
        let is_web_search = tool.get("name").and_then(Value::as_str) == Some(WEB_SEARCH_EXECUTOR);
        let is_native_web_search = tool_type == Some(NATIVE_WEB_SEARCH_TYPE) && is_web_search;
        if is_native_web_search {
            *tool = json!({
                "name": function_tool.name.clone(),
                "description": function_tool.description.clone(),
                "input_schema": function_tool.parameters.clone(),
            });
        }
    }

    Ok(WebSearchBudget { remaining: max_uses })
}
