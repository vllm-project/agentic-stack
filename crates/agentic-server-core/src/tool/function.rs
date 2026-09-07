use std::collections::HashMap;

use crate::types::io::FunctionTool;
use crate::types::tools::FunctionToolParam;

use super::handler::{ToolError, ToolHandler};
use super::registry::{ToolEntry, ToolType};

impl From<&FunctionToolParam> for FunctionTool {
    fn from(p: &FunctionToolParam) -> Self {
        Self {
            type_: "function".to_owned(),
            name: p.name.as_str().to_owned(),
            description: p.description.clone(),
            parameters: p.parameters.clone(),
            strict: p.strict,
        }
    }
}

/// Handler for `type: "function"` tools.
///
/// Function tools are client-owned: the gateway normalises them for vLLM but
/// never executes them. `FunctionHandler` intentionally implements only
/// [`ToolHandler`], not [`super::handler::GatewayExecutor`] — the type system
/// makes it impossible to call `execute()` on a client-owned tool.
#[derive(Debug)]
pub struct FunctionHandler;

impl ToolHandler for FunctionHandler {
    type ToolParams = FunctionToolParam;

    fn tool_type(&self) -> ToolType {
        ToolType::Function
    }

    fn validate(&self, params: &FunctionToolParam) -> Result<(), ToolError> {
        if params.name.as_str().is_empty() {
            Err(ToolError::Config("function tool must have a non-empty name".into()))
        } else {
            Ok(())
        }
    }

    fn normalize(&self, params: &FunctionToolParam) -> Vec<FunctionTool> {
        vec![FunctionTool::from(params)]
    }
}

pub(crate) fn insert_function_entry(entries: &mut HashMap<String, ToolEntry>, p: &FunctionToolParam) {
    // p.name is NonEmptyToolName — empty names are impossible here
    // (serde rejects them at deserialization time).
    if entries
        .insert(p.name.as_str().to_owned(), ToolEntry::client(ToolType::Function, None))
        .is_some()
    {
        tracing::warn!(name = %p.name, "duplicate tool name — previous definition overwritten");
    }
}
