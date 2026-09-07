use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::types::event::{MessageStatus, ResponseStatus};
use crate::types::io::output::FunctionToolCall;
use crate::types::io::{
    FunctionTool, FunctionToolResultMessage, InputFunctionToolCall, InputItem, InputToolSearchCall, OutputItem,
    ResponsesInput, ToolCallOutput, ToolChoice, ToolSearchCall, ToolSearchOutputMessage,
};
use crate::types::request_response::RequestPayload;
use crate::types::tools::{
    CodexNamespaceMember, CodexNamespaceToolParam, FunctionToolParam, ResponsesTool, ToolSearchStatus,
    ToolSearchToolParam,
};
use crate::utils::common::{deserialize_from_str, deserialize_from_value, serialize_to_string, serialize_to_value};

use super::CodexNamespaceHandler;
use super::handler::{ToolError, ToolHandler};
use super::registry::{ToolEntry, ToolRegistry, ToolType};

pub(crate) const TOOL_SEARCH_NAME: &str = "tool_search";
const DEFAULT_DESCRIPTION: &str = "Search the client tool catalog";
const DEFAULT_QUERY_DESCRIPTION: &str = "A concise description of the needed capabilities.";

/// Handler for client-executed `type: "tool_search"` declarations.
///
/// The declaration remains a first-class tool-search type throughout request
/// preparation and registry construction. This handler performs the one
/// provider-specific lowering step to the ordinary function shape understood
/// by upstreams without native tool-search support.
#[derive(Debug)]
pub struct ToolSearchHandler;

impl ToolSearchHandler {
    /// Whether this request carries a tool-search declaration, deferred
    /// definition, or replayed tool-search item and therefore needs the
    /// executor's preparation path.
    #[must_use]
    pub fn request_has_state<T: ?Sized>(request: &RequestPayload<T>) -> bool {
        request_contains_tool_search_state(request, &request.input)
    }

    /// Prepare the private inference view from fully rehydrated public state.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] when the public tool-search history or
    /// effective tool selection is invalid.
    pub(crate) fn prepare_request(
        request: &mut RequestPayload,
        restored_loaded_tools: &[ResponsesTool],
        restore_only_declared: bool,
    ) -> Result<Option<ToolSearchState>, ToolError> {
        let mut state =
            ToolSearchState::build_with_loaded_tools(request, restored_loaded_tools, restore_only_declared)?;
        if !state.is_active() {
            return Ok(None);
        }
        state.prepare_inference_request(request)?;
        Ok(Some(state))
    }

    /// Normalize native and synthetic upstream tool-search calls into the
    /// canonical public output item.
    pub(crate) fn normalize_response_output(
        registry: &ToolRegistry,
        output: &mut Vec<OutputItem>,
        status: ResponseStatus,
        unfinished_stream_item_ids: &HashSet<String>,
    ) -> Result<(), ToolError> {
        let discard_unfinished = matches!(status, ResponseStatus::Error | ResponseStatus::Incomplete);
        let mut normalized = Vec::with_capacity(output.len());
        for item in std::mem::take(output) {
            match item {
                OutputItem::FunctionCall(call)
                    if discard_unfinished && unfinished_stream_item_ids.contains(&call.id) => {}
                OutputItem::FunctionCall(call) => {
                    ensure_function_is_available(registry.is_withheld_function(&call.name))?;
                    if registry.tool_type(&call.name) == ToolType::ToolSearch {
                        if let Some(public) = project_synthetic_call(
                            &call,
                            discard_unfinished,
                            unfinished_stream_item_ids.contains(&call.id),
                        )? {
                            normalized.push(OutputItem::ToolSearchCall(public));
                        }
                    } else {
                        normalized.push(OutputItem::FunctionCall(call));
                    }
                }
                OutputItem::ToolSearchCall(call) => {
                    if let Some(public) = project_native_call(&call, discard_unfinished)? {
                        normalized.push(OutputItem::ToolSearchCall(public));
                    }
                }
                item => normalized.push(item),
            }
        }
        *output = normalized;
        Ok(())
    }

    #[must_use]
    pub(crate) fn normalized_param(param: &ToolSearchToolParam) -> ToolSearchToolParam {
        let mut normalized = param.clone();
        normalized.description = Some(
            param
                .description
                .as_deref()
                .filter(|description| !description.trim().is_empty())
                .unwrap_or(DEFAULT_DESCRIPTION)
                .to_owned(),
        );
        normalized.parameters = Some(
            param
                .parameters
                .clone()
                .unwrap_or_else(|| Value::Object(default_parameters())),
        );
        normalized
    }

    #[must_use]
    fn function_tool(param: &ToolSearchToolParam) -> FunctionTool {
        let normalized = Self::normalized_param(param);
        FunctionTool {
            type_: "function".to_owned(),
            name: TOOL_SEARCH_NAME.to_owned(),
            description: normalized.description,
            parameters: normalized.parameters,
            strict: Some(false),
        }
    }
}

impl ToolHandler for ToolSearchHandler {
    type ToolParams = ToolSearchToolParam;

    fn tool_type(&self) -> ToolType {
        ToolType::ToolSearch
    }

    fn validate(&self, param: &ToolSearchToolParam) -> Result<(), ToolError> {
        if param
            .parameters
            .as_ref()
            .is_some_and(|parameters| !parameters.is_object())
        {
            return Err(ToolError::Config(
                "tool_search parameters must be a JSON object for private function lowering".to_owned(),
            ));
        }
        Ok(())
    }

    fn normalize(&self, param: &ToolSearchToolParam) -> Vec<FunctionTool> {
        vec![Self::function_tool(param)]
    }
}

pub(crate) fn insert_tool_search_entry(entries: &mut HashMap<String, ToolEntry>, _param: &ToolSearchToolParam) {
    if entries
        .insert(
            TOOL_SEARCH_NAME.to_owned(),
            ToolEntry::client(ToolType::ToolSearch, None),
        )
        .is_some()
    {
        tracing::warn!(
            name = TOOL_SEARCH_NAME,
            "duplicate tool name — previous definition overwritten"
        );
    }
}

fn default_parameters() -> Map<String, Value> {
    let query = Map::from_iter([
        ("type".to_owned(), Value::String("string".to_owned())),
        (
            "description".to_owned(),
            Value::String(DEFAULT_QUERY_DESCRIPTION.to_owned()),
        ),
    ]);
    let properties = Map::from_iter([("query".to_owned(), Value::Object(query))]);
    Map::from_iter([
        ("type".to_owned(), Value::String("object".to_owned())),
        ("properties".to_owned(), Value::Object(properties)),
        (
            "required".to_owned(),
            Value::Array(vec![Value::String("query".to_owned())]),
        ),
        ("additionalProperties".to_owned(), Value::Bool(false)),
    ])
}

/// Stable public identity used to compare definitions accumulated from search outputs.
///
/// Equality remains type-aware while the state builder also indexes the visible name
/// separately, so returning the same name under a different supported kind is rejected.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum LoadedToolIdentity {
    Function(String),
    Namespace(String),
}

impl LoadedToolIdentity {
    fn name(&self) -> &str {
        match self {
            Self::Function(name) | Self::Namespace(name) => name,
        }
    }

    const fn kind(&self) -> &'static str {
        match self {
            Self::Function(_) => "function",
            Self::Namespace(_) => "namespace",
        }
    }
}

struct DefinitionRecord {
    identity: LoadedToolIdentity,
    canonical: Value,
    public_index: usize,
    loaded: bool,
    namespace_members: Option<NamespaceMemberRecords>,
}

struct NamespaceMemberRecord {
    canonical: Value,
    public_member_index: usize,
    loaded: bool,
}

struct NamespaceMemberRecords {
    ordered: Vec<NamespaceMemberRecord>,
    indexes: HashMap<String, usize>,
    unloaded_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToolSearchActivity {
    Inactive,
    Active,
}

struct PendingSearchCall {
    call_id: String,
}

struct DefinitionAccumulator<'a> {
    public_tools: &'a mut Vec<ResponsesTool>,
    definitions: &'a mut Vec<DefinitionRecord>,
    definition_indexes: &'a mut HashMap<String, usize>,
    loaded_public_tools: &'a mut Vec<ResponsesTool>,
    withheld_function_names: &'a mut HashSet<String>,
    prior_unknown_namespace_calls: HashMap<String, HashSet<String>>,
    unqualified_call_positions: HashMap<String, usize>,
    current_history_position: Option<usize>,
}

struct DefinitionViews<'a> {
    public_tools: &'a mut Vec<ResponsesTool>,
    definitions: &'a mut Vec<DefinitionRecord>,
    definition_indexes: &'a mut HashMap<String, usize>,
    loaded_public_tools: &'a mut Vec<ResponsesTool>,
    withheld_function_names: &'a mut HashSet<String>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum CatalogEntry {
    Function {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    Namespace {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
}

impl CatalogEntry {
    fn display_name(&self) -> &str {
        match self {
            Self::Function { name, .. } | Self::Namespace { name, .. } => name,
        }
    }

    fn description(&self) -> Option<&str> {
        match self {
            Self::Function { description, .. } | Self::Namespace { description, .. } => description.as_deref(),
        }
    }
}

/// Pure, request-scoped state derived from fully rehydrated public history.
///
/// The state deliberately has no `Serialize` implementation and its `Debug`
/// output contains counts only.
pub struct ToolSearchState {
    activity: ToolSearchActivity,
    has_completed_search: bool,
    public_effective_tools: Option<Vec<ResponsesTool>>,
    private_upstream_tools: Option<Vec<ResponsesTool>>,
    private_upstream_input: Option<ResponsesInput>,
    loaded_public_tools: Vec<ResponsesTool>,
    synthetic_tool_search: Option<ToolSearchToolParam>,
    withheld_function_names: HashSet<String>,
    unqualified_call_positions: HashMap<String, usize>,
}

/// Public tool-search state retained only for response persistence.
pub(crate) struct ToolSearchMetadata {
    pub(crate) effective_tools: Option<Vec<ResponsesTool>>,
    pub(crate) loaded_tools: Vec<ResponsesTool>,
}

impl fmt::Debug for ToolSearchState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolSearchState")
            .field("activity", &self.activity)
            .field("active", &self.is_active())
            .field("has_completed_search", &self.has_completed_search)
            .field(
                "public_effective_tool_count",
                &self.public_effective_tools.as_ref().map_or(0, Vec::len),
            )
            .field(
                "private_upstream_tool_count",
                &self.private_upstream_tools.as_ref().map_or(0, Vec::len),
            )
            .field("loaded_public_tool_count", &self.loaded_public_tools.len())
            .field("has_private_upstream_input", &self.private_upstream_input.is_some())
            .field("has_synthetic_tool_search", &self.synthetic_tool_search.is_some())
            .field("withheld_function_count", &self.withheld_function_names.len())
            .field("unqualified_history_call_count", &self.unqualified_call_positions.len())
            .finish()
    }
}

impl Default for ToolSearchState {
    fn default() -> Self {
        Self {
            activity: ToolSearchActivity::Inactive,
            has_completed_search: false,
            public_effective_tools: None,
            private_upstream_tools: None,
            private_upstream_input: None,
            loaded_public_tools: Vec::new(),
            synthetic_tool_search: None,
            withheld_function_names: HashSet::new(),
            unqualified_call_positions: HashMap::new(),
        }
    }
}

impl ToolSearchState {
    /// Build deterministic public/private views from ordered public history.
    ///
    /// This function performs no network, storage, clock, random-ID, or
    /// transport work. It runs in linear time in input items and definitions;
    /// vectors retain declaration/history order and maps are lookup-only.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] for an invalid public declaration,
    /// call/output ordering or linkage error, duplicate/conflicting definition,
    /// or normalized-name collision.
    pub fn build(request: &RequestPayload) -> Result<Self, ToolError> {
        Self::build_with_loaded_tools(request, &[], false)
    }

    /// Build state with public loaded definitions restored from typed response
    /// metadata when compaction has removed the original search pair.
    ///
    /// The restored definitions pass through the same validation and loading
    /// logic as definitions in a public `tool_search_output`.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] under the same conditions as [`Self::build`].
    pub fn build_with_loaded_tools(
        request: &RequestPayload,
        restored_loaded_tools: &[ResponsesTool],
        restore_only_declared: bool,
    ) -> Result<Self, ToolError> {
        let mut active_input = request.input.model_input().into_owned();
        if let (ResponsesInput::Items(public_items), ResponsesInput::Items(active_items)) =
            (&request.input, &mut active_input)
        {
            active_items.extend(
                public_items
                    .iter()
                    .filter(|item| matches!(item, InputItem::McpListTools(_)))
                    .cloned(),
            );
        }
        if !validate_tool_search_request(request, &active_input)? {
            return Ok(Self::default());
        }

        let input_items = match &active_input {
            ResponsesInput::Text(_) => &[][..],
            ResponsesInput::Items(items) => items.as_slice(),
        };
        let has_search_history = input_items
            .iter()
            .any(|item| matches!(item, InputItem::ToolSearchCall(_) | InputItem::ToolSearchOutput(_)));
        let has_completed_search = !restored_loaded_tools.is_empty()
            || input_items
                .iter()
                .any(|item| matches!(item, InputItem::ToolSearchOutput(_)));
        let declaration = request
            .tools
            .as_deref()
            .unwrap_or_default()
            .iter()
            .find_map(|tool| match tool {
                ResponsesTool::ToolSearch(declaration) => Some(declaration),
                _ => None,
            });
        if declaration.is_none() && !has_search_history {
            return Err(ToolError::Config(
                "defer_loading requires a tool_search declaration or replayed tool-search history".to_owned(),
            ));
        }

        let tools_were_present = request.tools.is_some();
        let mut public_tools = request.tools.clone().unwrap_or_default();
        let mut definitions = Vec::with_capacity(public_tools.len());
        let mut definition_indexes = HashMap::with_capacity(public_tools.len());
        index_initial_definitions(&public_tools, &mut definitions, &mut definition_indexes)?;
        let mut withheld_function_names =
            initial_withheld_function_names(&public_tools, &definitions, &definition_indexes)?;
        let mut unqualified_call_positions = HashMap::new();

        let mut loaded_public_tools = Vec::new();
        restore_loaded_definitions(
            restored_loaded_tools,
            DefinitionViews {
                public_tools: &mut public_tools,
                definitions: &mut definitions,
                definition_indexes: &mut definition_indexes,
                loaded_public_tools: &mut loaded_public_tools,
                withheld_function_names: &mut withheld_function_names,
            },
            restore_only_declared,
        )?;
        let private_upstream_input = prepare_history(
            &active_input,
            DefinitionViews {
                public_tools: &mut public_tools,
                definitions: &mut definitions,
                definition_indexes: &mut definition_indexes,
                loaded_public_tools: &mut loaded_public_tools,
                withheld_function_names: &mut withheld_function_names,
            },
            &mut unqualified_call_positions,
        )?;

        CodexNamespaceHandler.validate_namespace_collisions(Some(&public_tools))?;

        let catalog = build_catalog(&public_tools, &definitions, &definition_indexes);
        let synthetic_tool_search = declaration.map(|declaration| synthetic_tool_search(declaration, &catalog));
        let private_tools = build_private_tools(
            &public_tools,
            &definitions,
            &definition_indexes,
            synthetic_tool_search.as_ref(),
        );
        let public_effective_tools = (tools_were_present || !public_tools.is_empty()).then_some(public_tools);
        let private_upstream_tools = (tools_were_present || !private_tools.is_empty()).then_some(private_tools);

        Ok(Self {
            activity: ToolSearchActivity::Active,
            has_completed_search,
            public_effective_tools,
            private_upstream_tools,
            private_upstream_input: Some(private_upstream_input),
            loaded_public_tools,
            synthetic_tool_search,
            withheld_function_names,
            unqualified_call_positions,
        })
    }

    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self.activity, ToolSearchActivity::Active)
    }

    #[must_use]
    pub fn public_effective_tools(&self) -> Option<&[ResponsesTool]> {
        self.public_effective_tools.as_deref()
    }

    /// Public declarations available for selection in response metadata.
    /// Before search completes, the response echoes the declared catalog. Once
    /// search resolves availability, it exposes only initially available and
    /// loaded definitions while preserving their public namespace shape.
    #[must_use]
    pub(crate) fn public_response_tools(&self) -> Vec<ResponsesTool> {
        let public_tools = self.public_effective_tools.as_deref().unwrap_or_default();
        if !self.has_completed_search {
            return public_tools.to_vec();
        }
        available_public_tools(public_tools, &self.loaded_public_tools)
    }

    /// Public definitions resolved by completed search outputs, in first-load order.
    ///
    /// This remains separate from `public_effective_tools`: an initially
    /// deferred definition stays deferred publicly even after becoming loaded.
    #[must_use]
    pub fn loaded_public_tools(&self) -> &[ResponsesTool] {
        &self.loaded_public_tools
    }

    /// Private tool-search declaration used by request-scoped registry and upstream normalization.
    #[must_use]
    pub const fn synthetic_tool_search(&self) -> Option<&ToolSearchToolParam> {
        self.synthetic_tool_search.as_ref()
    }

    #[must_use]
    pub(crate) fn withheld_function_names(&self) -> &HashSet<String> {
        &self.withheld_function_names
    }

    /// Replace the request's public tool-search views with the prepared private
    /// input and tools used for inference. The retained state then contains
    /// only public metadata needed after inference.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError::Config`] when the effective tool choice conflicts
    /// with the prepared private tool set.
    pub fn prepare_inference_request(&mut self, request: &mut RequestPayload) -> Result<(), ToolError> {
        validate_effective_tool_choice(request.tool_choice.as_ref(), &self.withheld_function_names)?;
        let input = self.private_upstream_input.take().ok_or_else(|| {
            ToolError::Config("tool-search private inference input has already been consumed".to_owned())
        })?;
        request.input = input;
        request.tools = self.private_upstream_tools.take();
        Ok(())
    }

    pub(crate) fn into_public_metadata(mut self) -> ToolSearchMetadata {
        ToolSearchMetadata {
            effective_tools: self.public_effective_tools.take(),
            loaded_tools: std::mem::take(&mut self.loaded_public_tools),
        }
    }
}

fn validate_tool_search_request(request: &RequestPayload, input: &ResponsesInput) -> Result<bool, ToolError> {
    if !request_contains_tool_search_state(request, input) {
        return Ok(false);
    }

    let tools = request.tools.as_deref().unwrap_or_default();
    if tools
        .iter()
        .filter(|tool| matches!(tool, ResponsesTool::ToolSearch(_)))
        .count()
        > 1
    {
        return Err(ToolError::Config(
            "tool search accepts at most one tool_search declaration".to_owned(),
        ));
    }
    if request.parallel_tool_calls == Some(true) {
        return Err(ToolError::Config(
            "parallel_tool_calls must be false when tool search is active".to_owned(),
        ));
    }

    for tool in tools {
        tool.validate()?;
        if has_reserved_tool_search_name(tool) {
            return Err(ToolError::Config(
                "model-visible tool name 'tool_search' is reserved while tool search is active".to_owned(),
            ));
        }
    }

    Ok(true)
}

fn request_contains_tool_search_state<T: ?Sized>(request: &RequestPayload<T>, input: &ResponsesInput) -> bool {
    input_contains_tool_search_state(input)
        || request
            .tools
            .as_deref()
            .is_some_and(|tools| tools.iter().any(tool_activates_tool_search))
}

fn input_contains_tool_search_state(input: &ResponsesInput) -> bool {
    matches!(
        input,
        ResponsesInput::Items(items)
            if items
                .iter()
                .any(|item| matches!(item, InputItem::ToolSearchCall(_) | InputItem::ToolSearchOutput(_)))
    )
}

fn tool_activates_tool_search(tool: &ResponsesTool) -> bool {
    matches!(tool, ResponsesTool::ToolSearch(_)) || tool_has_deferred_definition(tool)
}

fn tool_has_deferred_definition(tool: &ResponsesTool) -> bool {
    match tool {
        ResponsesTool::Function(function) => function.defer_loading == Some(true),
        ResponsesTool::Namespace(namespace) => namespace.tools.iter().any(
            |member| matches!(member, CodexNamespaceMember::Function(function) if function.defer_loading == Some(true)),
        ),
        ResponsesTool::Mcp(mcp) => mcp.defer_loading == Some(true),
        ResponsesTool::Custom(custom) => custom.defer_loading == Some(true),
        ResponsesTool::ToolSearch(_)
        | ResponsesTool::WebSearch(_)
        | ResponsesTool::FileSearch(_)
        | ResponsesTool::CodeInterpreter(_)
        | ResponsesTool::Unknown => false,
    }
}

pub(crate) fn ensure_request_prepared(request: &RequestPayload, prepared: bool) -> Result<(), ToolError> {
    if ToolSearchHandler::request_has_state(request) && !prepared {
        return Err(ToolError::Config(
            "tool_search requests require prepared request-scoped state before upstream conversion".to_owned(),
        ));
    }
    Ok(())
}

fn has_reserved_tool_search_name(tool: &ResponsesTool) -> bool {
    match tool {
        ResponsesTool::Function(function) => function.name.as_str() == TOOL_SEARCH_NAME,
        ResponsesTool::Custom(custom) => custom.name.as_str() == TOOL_SEARCH_NAME,
        ResponsesTool::Namespace(namespace) => namespace.name == TOOL_SEARCH_NAME,
        ResponsesTool::ToolSearch(_)
        | ResponsesTool::Mcp(_)
        | ResponsesTool::WebSearch(_)
        | ResponsesTool::FileSearch(_)
        | ResponsesTool::CodeInterpreter(_)
        | ResponsesTool::Unknown => false,
    }
}

fn validate_effective_tool_choice(
    tool_choice: Option<&ToolChoice>,
    withheld_function_names: &HashSet<String>,
) -> Result<(), ToolError> {
    let targets_withheld = match tool_choice {
        Some(ToolChoice::Function { namespace, name }) => {
            let model_name = namespace.as_deref().map_or_else(
                || name.as_str().to_owned(),
                |namespace| super::model_visible_namespace_member_name(namespace, name.as_str()),
            );
            withheld_function_names.contains(&model_name)
        }
        Some(ToolChoice::AllowedTools { tools, .. }) => tools
            .iter()
            .any(|tool| tool.type_.as_str() == "function" && withheld_function_names.contains(tool.name.as_str())),
        _ => false,
    };
    if targets_withheld {
        return Err(ToolError::Config(
            "tool_choice targets a function before its definition is loaded".to_owned(),
        ));
    }
    Ok(())
}

fn restore_loaded_definitions(
    restored_loaded_tools: &[ResponsesTool],
    views: DefinitionViews<'_>,
    restore_only_declared: bool,
) -> Result<(), ToolError> {
    let DefinitionViews {
        public_tools,
        definitions,
        definition_indexes,
        loaded_public_tools,
        withheld_function_names,
    } = views;
    let mut accumulator = DefinitionAccumulator {
        public_tools,
        definitions,
        definition_indexes,
        loaded_public_tools,
        withheld_function_names,
        prior_unknown_namespace_calls: HashMap::new(),
        unqualified_call_positions: HashMap::new(),
        current_history_position: None,
    };
    for tool in restored_loaded_tools {
        let Some(tool) = restored_definition_for_load(tool, accumulator.definition_indexes, restore_only_declared)?
        else {
            continue;
        };
        load_definition(&tool, &mut accumulator)?;
    }
    Ok(())
}

fn restored_definition_for_load(
    restored: &ResponsesTool,
    definition_indexes: &HashMap<String, usize>,
    restore_only_declared: bool,
) -> Result<Option<ResponsesTool>, ToolError> {
    let identity = loaded_tool_identity(restored)?.ok_or_else(|| {
        ToolError::Config("stored tool-search availability contains an unsupported definition".to_owned())
    })?;
    if restore_only_declared && !definition_indexes.contains_key(identity.name()) {
        return Ok(None);
    }
    Ok(Some(restored.clone()))
}

pub(crate) fn public_item_id(item_id: &str) -> String {
    if item_id.strip_prefix("tsc_").is_some_and(|suffix| !suffix.is_empty()) {
        return item_id.to_owned();
    }
    if let Some(suffix) = item_id.strip_prefix("fc_").filter(|suffix| !suffix.is_empty()) {
        return format!("tsc_{suffix}");
    }
    let domain_separated = format!("tool_search_item:{item_id}");
    format!("tsc_{:016x}", stable_hash(&domain_separated))
}

fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

pub(crate) fn invalid_upstream_search_call() -> ToolError {
    ToolError::InvalidUpstreamToolSearch
}

pub(crate) fn invalid_upstream_withheld_function_call() -> ToolError {
    ToolError::UpstreamWithheldFunctionCall
}

pub(crate) fn started_public_call(call: &FunctionToolCall) -> Result<ToolSearchCall, ToolError> {
    if call.id.trim().is_empty()
        || call.call_id.trim().is_empty()
        || call.name != TOOL_SEARCH_NAME
        || call.namespace.is_some()
    {
        return Err(invalid_upstream_search_call());
    }
    Ok(ToolSearchCall {
        id: public_item_id(&call.id),
        call_id: call.call_id.clone(),
        execution: crate::types::tools::ToolSearchExecution::Client,
        arguments: Value::Object(Map::new()),
        status: ToolSearchStatus::InProgress,
    })
}

pub(crate) fn completed_public_call(call: &FunctionToolCall) -> Result<ToolSearchCall, ToolError> {
    if call.status != MessageStatus::Completed {
        return Err(invalid_upstream_search_call());
    }
    let mut public = started_public_call(call)?;
    public.arguments = json_value(&call.arguments)?;
    public.status = ToolSearchStatus::Completed;
    Ok(public)
}

pub(crate) fn project_synthetic_call(
    call: &FunctionToolCall,
    discard_incomplete: bool,
    unfinished_stream_call: bool,
) -> Result<Option<ToolSearchCall>, ToolError> {
    if unfinished_stream_call || call.status != MessageStatus::Completed {
        return if discard_incomplete {
            Ok(None)
        } else {
            Err(invalid_upstream_search_call())
        };
    }
    completed_public_call(call).map(Some)
}

pub(crate) fn project_native_call(
    call: &ToolSearchCall,
    discard_incomplete: bool,
) -> Result<Option<ToolSearchCall>, ToolError> {
    if call.status == ToolSearchStatus::Completed {
        return Ok(Some(call.clone()));
    }
    if discard_incomplete {
        return Ok(None);
    }
    Err(invalid_upstream_search_call())
}

pub(crate) fn ensure_function_is_available(is_withheld: bool) -> Result<(), ToolError> {
    if is_withheld {
        return Err(invalid_upstream_withheld_function_call());
    }
    Ok(())
}

pub(crate) fn validate_public_arguments(arguments: &str) -> Result<(), ToolError> {
    json_value(arguments).map(|_| ())
}

pub(crate) fn strict_started_function(item: &Value) -> Result<FunctionToolCall, ToolError> {
    let function = strict_function_call(item)?;
    if function.status != MessageStatus::InProgress || !function.arguments.is_empty() {
        return Err(invalid_upstream_search_call());
    }
    Ok(function)
}

#[derive(Debug, Deserialize)]
struct StrictFunctionToolCall {
    id: String,
    call_id: String,
    name: String,
    #[serde(default)]
    namespace: Option<Value>,
    arguments: String,
    status: MessageStatus,
}

pub(crate) fn strict_function_call(item: &Value) -> Result<FunctionToolCall, ToolError> {
    let call: StrictFunctionToolCall =
        deserialize_from_value(item.clone()).map_err(|_| invalid_upstream_search_call())?;
    if call.namespace.is_some() {
        return Err(invalid_upstream_search_call());
    }
    let call = FunctionToolCall {
        id: call.id,
        call_id: call.call_id,
        name: call.name,
        namespace: None,
        arguments: call.arguments,
        status: call.status,
    };
    started_public_call(&call)?;
    if call.status == MessageStatus::Completed {
        json_value(&call.arguments)?;
    }
    Ok(call)
}

pub(crate) fn strict_native_call(item: Value) -> Result<ToolSearchCall, ToolError> {
    if item.get("namespace").is_some_and(|namespace| !namespace.is_null()) {
        return Err(invalid_upstream_search_call());
    }
    deserialize_from_value(item).map_err(|_| invalid_upstream_search_call())
}

fn json_value(arguments: &str) -> Result<Value, ToolError> {
    deserialize_from_str(arguments).map_err(|_| invalid_upstream_search_call())
}

pub(crate) fn validate_blocking_response(
    body: &str,
    tool_search_enabled: bool,
    withheld_function_names: &HashSet<String>,
) -> Result<(), ToolError> {
    if !tool_search_enabled && withheld_function_names.is_empty() && !might_contain_tool_search_wire(body) {
        return Ok(());
    }
    let value: Value = match deserialize_from_str(body) {
        Ok(value) => value,
        Err(_) => return Ok(()),
    };
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .map_or(ResponseStatus::Completed, |status| status.parse().unwrap_or_default());
    let discard_unfinished = matches!(status, ResponseStatus::Error | ResponseStatus::Incomplete);
    for item in value.get("output").and_then(Value::as_array).into_iter().flatten() {
        if item.get("type").and_then(Value::as_str) == Some("function_call")
            && item
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| withheld_function_names.contains(name))
        {
            return Err(invalid_upstream_withheld_function_call());
        }
        match item.get("type").and_then(Value::as_str) {
            Some("tool_search_call") => {
                let call = strict_native_call(item.clone())?;
                if !discard_unfinished && call.status != ToolSearchStatus::Completed {
                    return Err(invalid_upstream_search_call());
                }
            }
            Some("function_call")
                if tool_search_enabled && item.get("name").and_then(Value::as_str) == Some(TOOL_SEARCH_NAME) =>
            {
                let call = strict_function_call(item)?;
                if !discard_unfinished && call.status != MessageStatus::Completed {
                    return Err(invalid_upstream_search_call());
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn might_contain_tool_search_wire(wire: &str) -> bool {
    wire.contains(TOOL_SEARCH_NAME)
}

fn index_initial_definitions(
    tools: &[ResponsesTool],
    definitions: &mut Vec<DefinitionRecord>,
    definition_indexes: &mut HashMap<String, usize>,
) -> Result<(), ToolError> {
    for (public_index, tool) in tools.iter().enumerate() {
        let Some(identity) = loaded_tool_identity(tool)? else {
            continue;
        };
        if definition_indexes.contains_key(identity.name()) {
            return Err(ToolError::Config(format!(
                "duplicate tool-search definition identity '{}'",
                identity.name()
            )));
        }
        let index = definitions.len();
        definition_indexes.insert(identity.name().to_owned(), index);
        definitions.push(definition_record(tool, identity, public_index, false)?);
    }
    Ok(())
}

fn definition_record(
    tool: &ResponsesTool,
    identity: LoadedToolIdentity,
    public_index: usize,
    dynamically_loaded: bool,
) -> Result<DefinitionRecord, ToolError> {
    let namespace_members = match tool {
        ResponsesTool::Namespace(namespace) => Some(namespace_member_records(namespace, dynamically_loaded)?),
        ResponsesTool::Function(_) => None,
        ResponsesTool::ToolSearch(_)
        | ResponsesTool::Mcp(_)
        | ResponsesTool::WebSearch(_)
        | ResponsesTool::FileSearch(_)
        | ResponsesTool::CodeInterpreter(_)
        | ResponsesTool::Custom(_)
        | ResponsesTool::Unknown => {
            return Err(ToolError::Config(
                "tool-search definition record received an unsupported tool".to_owned(),
            ));
        }
    };
    let loaded = namespace_members
        .as_ref()
        .map_or(dynamically_loaded, |members| members.unloaded_count == 0);
    Ok(DefinitionRecord {
        identity,
        canonical: canonical_definition(tool)?,
        public_index,
        loaded,
        namespace_members,
    })
}

fn namespace_member_records(
    namespace: &CodexNamespaceToolParam,
    dynamically_loaded: bool,
) -> Result<NamespaceMemberRecords, ToolError> {
    if namespace.tools.is_empty() {
        return Err(ToolError::Config(
            "tool-search namespaces must contain at least one function member".to_owned(),
        ));
    }
    let mut ordered = Vec::with_capacity(namespace.tools.len());
    let mut indexes = HashMap::with_capacity(namespace.tools.len());
    let mut unloaded_count = 0;
    for (public_member_index, member) in namespace.tools.iter().enumerate() {
        let CodexNamespaceMember::Function(function) = member else {
            return Err(ToolError::Config(
                "tool-search namespaces may contain only function members".to_owned(),
            ));
        };
        let name = function.name.as_str();
        if indexes.insert(name.to_owned(), ordered.len()).is_some() {
            return Err(ToolError::Config(format!(
                "duplicate namespace member identity '{}.{name}'",
                namespace.name
            )));
        }
        let loaded = dynamically_loaded || function.defer_loading != Some(true);
        unloaded_count += usize::from(!loaded);
        ordered.push(NamespaceMemberRecord {
            canonical: canonical_namespace_member(function)?,
            public_member_index,
            loaded,
        });
    }
    Ok(NamespaceMemberRecords {
        ordered,
        indexes,
        unloaded_count,
    })
}

fn initial_withheld_function_names(
    public_tools: &[ResponsesTool],
    definitions: &[DefinitionRecord],
    definition_indexes: &HashMap<String, usize>,
) -> Result<HashSet<String>, ToolError> {
    let mut withheld = HashSet::new();
    for tool in public_tools {
        if let ResponsesTool::Function(function) = tool {
            if function.defer_loading == Some(true) {
                withheld.insert(function.name.as_str().to_owned());
            }
            continue;
        }
        let ResponsesTool::Namespace(namespace) = tool else {
            continue;
        };
        let record = definition_indexes
            .get(&namespace.name)
            .and_then(|index| definitions.get(*index))
            .ok_or_else(|| ToolError::Config("namespace availability state is inconsistent".to_owned()))?;
        let members = record
            .namespace_members
            .as_ref()
            .ok_or_else(|| ToolError::Config("namespace availability state is inconsistent".to_owned()))?;
        for member in &namespace.tools {
            let CodexNamespaceMember::Function(function) = member else {
                continue;
            };
            let member_index = members
                .indexes
                .get(function.name.as_str())
                .ok_or_else(|| ToolError::Config("namespace availability state is inconsistent".to_owned()))?;
            let is_withheld = !members.ordered[*member_index].loaded;
            if is_withheld {
                withheld.insert(super::model_visible_namespace_member_name(
                    &namespace.name,
                    function.name.as_str(),
                ));
            }
        }
    }
    Ok(withheld)
}

#[derive(Serialize)]
struct CanonicalToolSearchOutput<'a> {
    tools: &'a [ModelVisibleLoadedTool<'a>],
}

/// Typed model-output projection, deliberately separate from the raw
/// credential-sensitive definition retained for equality and later execution.
#[derive(Serialize)]
#[serde(untagged)]
enum ModelVisibleLoadedTool<'a> {
    Function(ModelVisibleFunction<'a>),
    Namespace(ModelVisibleNamespace<'a>),
}

#[derive(Serialize)]
struct ModelVisibleFunction<'a> {
    #[serde(rename = "type")]
    type_: &'static str,
    #[serde(flatten)]
    definition: &'a FunctionToolParam,
}

#[derive(Serialize)]
struct ModelVisibleNamespace<'a> {
    #[serde(rename = "type")]
    type_: &'static str,
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
}

fn prepare_history(
    input: &ResponsesInput,
    views: DefinitionViews<'_>,
    unqualified_call_positions: &mut HashMap<String, usize>,
) -> Result<ResponsesInput, ToolError> {
    let ResponsesInput::Items(items) = input else {
        return Ok(input.clone());
    };
    let mut private_items = Vec::with_capacity(items.len());
    let mut unresolved_call: Option<PendingSearchCall> = None;
    let mut completed_call_ids = HashSet::new();
    let mut item_ids = HashSet::new();
    let DefinitionViews {
        public_tools,
        definitions,
        definition_indexes,
        loaded_public_tools,
        withheld_function_names,
    } = views;
    let mut definition_accumulator = DefinitionAccumulator {
        public_tools,
        definitions,
        definition_indexes,
        loaded_public_tools,
        withheld_function_names,
        prior_unknown_namespace_calls: HashMap::new(),
        unqualified_call_positions: std::mem::take(unqualified_call_positions),
        current_history_position: None,
    };

    for (position, item) in items.iter().enumerate() {
        definition_accumulator.current_history_position = Some(position);
        match item {
            InputItem::ToolSearchCall(call) => {
                private_items.push(prepare_search_call(
                    call,
                    &mut unresolved_call,
                    &completed_call_ids,
                    &mut item_ids,
                )?);
            }
            InputItem::ToolSearchOutput(output) => {
                private_items.push(prepare_search_output(
                    output,
                    &mut unresolved_call,
                    &mut completed_call_ids,
                    &mut definition_accumulator,
                )?);
            }
            InputItem::FunctionCall(call) => {
                ensure_history_call_is_available(call, &mut definition_accumulator)?;
                private_items.push(item.clone());
            }
            InputItem::CompactionTrigger => {}
            InputItem::Message(_)
            | InputItem::McpListTools(_)
            | InputItem::FunctionCallOutput(_)
            | InputItem::CustomToolCall(_)
            | InputItem::CustomToolCallOutput(_)
            | InputItem::Reasoning(_)
            | InputItem::Compaction(_)
            | InputItem::Unknown => private_items.push(item.clone()),
        }
    }

    if unresolved_call.is_some() {
        return Err(ToolError::Config(
            "unresolved tool_search_call requires a matching completed tool_search_output".to_owned(),
        ));
    }
    *unqualified_call_positions = definition_accumulator.unqualified_call_positions;
    Ok(ResponsesInput::Items(private_items))
}

fn ensure_history_call_is_available(
    call: &InputFunctionToolCall,
    definitions: &mut DefinitionAccumulator<'_>,
) -> Result<(), ToolError> {
    if let Some(namespace) = call.namespace.as_deref() {
        let member = definitions
            .definition_indexes
            .get(namespace)
            .and_then(|index| definitions.definitions.get(*index))
            .filter(|record| matches!(record.identity, LoadedToolIdentity::Namespace(_)))
            .and_then(|record| record.namespace_members.as_ref())
            .and_then(|members| members.indexes.get(&call.name).map(|index| &members.ordered[*index]));
        match member {
            Some(member) if !member.loaded => return Err(withheld_function_history_call()),
            Some(_) => {}
            None => {
                definitions
                    .prior_unknown_namespace_calls
                    .entry(namespace.to_owned())
                    .or_default()
                    .insert(call.name.clone());
            }
        }
    } else {
        if definitions.withheld_function_names.contains(&call.name) {
            return Err(withheld_function_history_call());
        }
        let position = definitions
            .current_history_position
            .ok_or_else(|| ToolError::Config("tool-search history position is unavailable".to_owned()))?;
        definitions
            .unqualified_call_positions
            .entry(call.name.clone())
            .or_insert(position);
    }
    Ok(())
}

fn withheld_function_history_call() -> ToolError {
    ToolError::Config("request history calls a function before its definition is loaded".to_owned())
}

fn prepare_search_call(
    call: &InputToolSearchCall,
    unresolved_call: &mut Option<PendingSearchCall>,
    completed_call_ids: &HashSet<String>,
    item_ids: &mut HashSet<String>,
) -> Result<InputItem, ToolError> {
    if call.id.trim().is_empty() {
        return Err(ToolError::Config("tool_search_call id must not be blank".to_owned()));
    }
    if call.call_id.trim().is_empty() {
        return Err(ToolError::Config(
            "tool_search_call call_id must not be blank".to_owned(),
        ));
    }
    if !item_ids.insert(call.id.clone()) {
        return Err(ToolError::Config("duplicate tool_search_call item id".to_owned()));
    }
    if unresolved_call.is_some() {
        return Err(ToolError::Config(
            "ambiguous tool-search history contains a call before the preceding call is resolved".to_owned(),
        ));
    }
    if completed_call_ids.contains(call.call_id.as_str()) {
        return Err(ToolError::Config("duplicate tool_search_call call_id".to_owned()));
    }
    let canonical_arguments = serialize_to_string(&call.arguments)
        .map_err(|_| ToolError::Config("tool_search_call arguments could not be canonicalized safely".to_owned()))?;
    *unresolved_call = Some(PendingSearchCall {
        call_id: call.call_id.clone(),
    });
    Ok(InputItem::FunctionCall(InputFunctionToolCall {
        id: Some(call.id.clone()),
        call_id: call.call_id.clone(),
        name: TOOL_SEARCH_NAME.to_owned(),
        namespace: None,
        arguments: canonical_arguments,
        status: Some(MessageStatus::Completed),
    }))
}

fn prepare_search_output(
    output: &ToolSearchOutputMessage,
    unresolved_call: &mut Option<PendingSearchCall>,
    completed_call_ids: &mut HashSet<String>,
    definition_accumulator: &mut DefinitionAccumulator<'_>,
) -> Result<InputItem, ToolError> {
    if output.call_id.trim().is_empty() {
        return Err(ToolError::Config(
            "tool_search_output call_id must not be blank".to_owned(),
        ));
    }
    if completed_call_ids.contains(output.call_id.as_str()) {
        return Err(ToolError::Config("duplicate tool_search_output call_id".to_owned()));
    }
    let Some(pending) = unresolved_call.take() else {
        return Err(ToolError::Config(
            "orphan tool_search_output has no unresolved call".to_owned(),
        ));
    };
    if pending.call_id != output.call_id {
        return Err(ToolError::Config(
            "tool_search_output call_id does not match the preceding unresolved call".to_owned(),
        ));
    }
    if output.status != ToolSearchStatus::Completed {
        return Err(ToolError::Config(
            "tool_search_output must be completed before it may load tool definitions".to_owned(),
        ));
    }
    for tool in &output.tools {
        load_definition(tool, definition_accumulator)?;
    }
    let projected_tools = model_visible_output_tools(&output.tools)?;
    let canonical_value = serialize_to_value(&CanonicalToolSearchOutput {
        tools: &projected_tools,
    })
    .map_err(|_| ToolError::Config("tool_search_output could not be canonicalized safely".to_owned()))?;
    let canonical_output = serialize_to_string(&canonical_value)
        .map_err(|_| ToolError::Config("tool_search_output could not be canonicalized safely".to_owned()))?;
    completed_call_ids.insert(output.call_id.clone());
    Ok(InputItem::FunctionCallOutput(FunctionToolResultMessage {
        call_id: output.call_id.clone(),
        output: ToolCallOutput::Text(canonical_output),
    }))
}

fn model_visible_output_tools(tools: &[ResponsesTool]) -> Result<Vec<ModelVisibleLoadedTool<'_>>, ToolError> {
    tools
        .iter()
        .map(|tool| match tool {
            ResponsesTool::Function(definition) => Ok(ModelVisibleLoadedTool::Function(ModelVisibleFunction {
                type_: "function",
                definition,
            })),
            ResponsesTool::Namespace(namespace) => Ok(ModelVisibleLoadedTool::Namespace(ModelVisibleNamespace {
                type_: "namespace",
                name: &namespace.name,
                description: namespace.description.as_deref(),
            })),
            ResponsesTool::ToolSearch(_)
            | ResponsesTool::Mcp(_)
            | ResponsesTool::WebSearch(_)
            | ResponsesTool::FileSearch(_)
            | ResponsesTool::CodeInterpreter(_)
            | ResponsesTool::Custom(_)
            | ResponsesTool::Unknown => Err(ToolError::Config(
                "tool_search_output contains an unsupported model-output definition".to_owned(),
            )),
        })
        .collect()
}

fn load_definition(tool: &ResponsesTool, definitions: &mut DefinitionAccumulator<'_>) -> Result<(), ToolError> {
    let identity = loaded_tool_identity(tool)?
        .ok_or_else(|| ToolError::Config("tool_search_output contains an unsupported tool definition".to_owned()))?;
    let canonical = canonical_definition(tool)?;
    if let Some(index) = definitions.definition_indexes.get(identity.name()).copied() {
        let record = &mut definitions.definitions[index];
        if record.identity != identity || record.canonical != canonical {
            return Err(ToolError::Config(format!(
                "loaded definition for identity '{}' conflicts with its existing type, schema, description, or configuration",
                identity.name()
            )));
        }
        if let ResponsesTool::Namespace(returned) = tool {
            return load_namespace_members(
                returned,
                record,
                definitions.public_tools,
                definitions.loaded_public_tools,
                definitions.withheld_function_names,
                &definitions.prior_unknown_namespace_calls,
                &definitions.unqualified_call_positions,
            );
        }
        if !record.loaded {
            if definitions.withheld_function_names.contains(record.identity.name())
                && definitions
                    .unqualified_call_positions
                    .contains_key(record.identity.name())
            {
                return Err(withheld_function_history_call());
            }
            record.loaded = true;
            definitions.withheld_function_names.remove(record.identity.name());
            definitions
                .loaded_public_tools
                .push(definitions.public_tools[record.public_index].clone());
        }
        return Ok(());
    }

    match tool {
        ResponsesTool::Function(function)
            if definitions
                .unqualified_call_positions
                .contains_key(function.name.as_str()) =>
        {
            return Err(withheld_function_history_call());
        }
        ResponsesTool::Namespace(namespace) => ensure_namespace_members_do_not_resolve_prior_calls(
            namespace,
            namespace.tools.iter(),
            &definitions.prior_unknown_namespace_calls,
            &definitions.unqualified_call_positions,
        )?,
        _ => {}
    }
    let public_index = definitions.public_tools.len();
    definitions.public_tools.push(tool.clone());
    let index = definitions.definitions.len();
    definitions.definition_indexes.insert(identity.name().to_owned(), index);
    definitions
        .definitions
        .push(definition_record(tool, identity, public_index, true)?);
    definitions.loaded_public_tools.push(tool.clone());
    Ok(())
}

fn ensure_namespace_members_do_not_resolve_prior_calls<'a>(
    namespace: &CodexNamespaceToolParam,
    members: impl Iterator<Item = &'a CodexNamespaceMember>,
    prior_unknown_namespace_calls: &HashMap<String, HashSet<String>>,
    unqualified_call_positions: &HashMap<String, usize>,
) -> Result<(), ToolError> {
    let prior_public_members = prior_unknown_namespace_calls.get(&namespace.name);
    for member in members {
        let CodexNamespaceMember::Function(function) = member else {
            continue;
        };
        let public_match = prior_public_members.is_some_and(|members| members.contains(function.name.as_str()));
        let flat_name = super::model_visible_namespace_member_name(&namespace.name, function.name.as_str());
        if public_match || unqualified_call_positions.contains_key(&flat_name) {
            return Err(withheld_function_history_call());
        }
    }
    Ok(())
}

fn load_namespace_members(
    returned: &CodexNamespaceToolParam,
    record: &mut DefinitionRecord,
    public_tools: &mut [ResponsesTool],
    loaded_public_tools: &mut Vec<ResponsesTool>,
    withheld_function_names: &mut HashSet<String>,
    prior_unknown_namespace_calls: &HashMap<String, HashSet<String>>,
    unqualified_call_positions: &HashMap<String, usize>,
) -> Result<(), ToolError> {
    if returned.tools.is_empty() {
        return Err(ToolError::Config(
            "tool_search_output namespaces must contain at least one function member".to_owned(),
        ));
    }
    let ResponsesTool::Namespace(public_namespace) = &mut public_tools[record.public_index] else {
        return Err(ToolError::Config(
            "namespace identity conflicts with an existing non-namespace definition".to_owned(),
        ));
    };
    let members = record
        .namespace_members
        .as_mut()
        .ok_or_else(|| ToolError::Config("namespace definition is missing prepared member state".to_owned()))?;
    ensure_namespace_members_do_not_resolve_prior_calls(
        returned,
        returned.tools.iter().filter(|member| match member {
            CodexNamespaceMember::Function(function) => !members.indexes.contains_key(function.name.as_str()),
            CodexNamespaceMember::Unknown => true,
        }),
        prior_unknown_namespace_calls,
        unqualified_call_positions,
    )?;
    let mut newly_loaded = Vec::new();
    for member in &returned.tools {
        let CodexNamespaceMember::Function(returned_function) = member else {
            return Err(ToolError::Config(
                "tool_search_output namespaces may contain only function members".to_owned(),
            ));
        };
        let member_name = returned_function.name.as_str();
        let canonical = canonical_namespace_member(returned_function)?;
        if let Some(member_index) = members.indexes.get(member_name).copied() {
            let member_record = &mut members.ordered[member_index];
            if member_record.canonical != canonical {
                return Err(ToolError::Config(format!(
                    "loaded namespace member '{}.{member_name}' conflicts with its existing schema, description, or configuration",
                    returned.name
                )));
            }
            if !member_record.loaded {
                let unloaded_count = members.unloaded_count.checked_sub(1).ok_or_else(|| {
                    ToolError::Config("namespace member availability state is inconsistent".to_owned())
                })?;
                member_record.loaded = true;
                members.unloaded_count = unloaded_count;
                withheld_function_names
                    .remove(&super::model_visible_namespace_member_name(&returned.name, member_name));
                newly_loaded.push(public_namespace.tools[member_record.public_member_index].clone());
            }
            continue;
        }

        let public_member_index = public_namespace.tools.len();
        public_namespace.tools.push(member.clone());
        members.indexes.insert(member_name.to_owned(), members.ordered.len());
        members.ordered.push(NamespaceMemberRecord {
            canonical,
            public_member_index,
            loaded: true,
        });
        newly_loaded.push(member.clone());
    }
    if !newly_loaded.is_empty() {
        let mut loaded_subset = public_namespace.clone();
        loaded_subset.tools = newly_loaded;
        loaded_public_tools.push(ResponsesTool::Namespace(loaded_subset));
    }
    record.loaded = members.unloaded_count == 0;
    Ok(())
}

fn loaded_tool_identity(tool: &ResponsesTool) -> Result<Option<LoadedToolIdentity>, ToolError> {
    let identity = match tool {
        ResponsesTool::Function(function) => LoadedToolIdentity::Function(function.name.as_str().to_owned()),
        ResponsesTool::Namespace(namespace) => LoadedToolIdentity::Namespace(namespace.name.clone()),
        ResponsesTool::ToolSearch(_)
        | ResponsesTool::Mcp(_)
        | ResponsesTool::WebSearch(_)
        | ResponsesTool::FileSearch(_)
        | ResponsesTool::CodeInterpreter(_)
        | ResponsesTool::Custom(_)
        | ResponsesTool::Unknown => return Ok(None),
    };
    if identity.name().trim().is_empty() {
        return Err(ToolError::Config(format!(
            "{} definition identity must not be blank",
            identity.kind()
        )));
    }
    if matches!(
        &identity,
        LoadedToolIdentity::Function(name) | LoadedToolIdentity::Namespace(name)
            if name == TOOL_SEARCH_NAME
    ) {
        return Err(ToolError::Config(
            "model-visible tool name 'tool_search' is reserved while tool search is active".to_owned(),
        ));
    }
    Ok(Some(identity))
}

fn canonical_definition(tool: &ResponsesTool) -> Result<Value, ToolError> {
    let projected = match tool {
        ResponsesTool::Namespace(namespace) => {
            let mut namespace = namespace.clone();
            namespace.tools.clear();
            ResponsesTool::Namespace(namespace)
        }
        other => other.clone(),
    };
    serialize_to_value(&projected)
        .map_err(|_| ToolError::Config("tool-search definition could not be compared safely".to_owned()))
}

fn canonical_namespace_member(function: &FunctionToolParam) -> Result<Value, ToolError> {
    serialize_to_value(function)
        .map_err(|_| ToolError::Config("namespace member definition could not be compared safely".to_owned()))
}

fn build_catalog(
    public_tools: &[ResponsesTool],
    definitions: &[DefinitionRecord],
    definition_indexes: &HashMap<String, usize>,
) -> Vec<CatalogEntry> {
    public_tools
        .iter()
        .filter_map(|tool| {
            let identity = loaded_tool_identity(tool).ok().flatten()?;
            let record = &definitions[*definition_indexes.get(identity.name())?];
            if record.loaded {
                return None;
            }
            match tool {
                ResponsesTool::Function(function) if function.defer_loading == Some(true) => {
                    Some(CatalogEntry::Function {
                        name: function.name.as_str().to_owned(),
                        description: function.description.clone(),
                    })
                }
                ResponsesTool::Namespace(namespace)
                    if namespace_has_withheld_member(record.namespace_members.as_ref()) =>
                {
                    Some(CatalogEntry::Namespace {
                        name: namespace.name.clone(),
                        description: namespace.description.clone(),
                    })
                }
                ResponsesTool::Function(_)
                | ResponsesTool::Namespace(_)
                | ResponsesTool::Mcp(_)
                | ResponsesTool::ToolSearch(_)
                | ResponsesTool::WebSearch(_)
                | ResponsesTool::FileSearch(_)
                | ResponsesTool::CodeInterpreter(_)
                | ResponsesTool::Custom(_)
                | ResponsesTool::Unknown => None,
            }
        })
        .collect()
}

/// Catalog prose deliberately follows the provider-characterization shape:
/// declaration text, then one ordered semicolon-delimited list of `name —
/// description` entries. It never uses schemas or execution configuration.
fn synthetic_description(description: &str, catalog: &[CatalogEntry]) -> String {
    if catalog.is_empty() {
        return description.to_owned();
    }
    let entries = catalog
        .iter()
        .map(|entry| {
            entry.description().map_or_else(
                || entry.display_name().to_owned(),
                |description| {
                    let description = description.trim();
                    if description.is_empty() {
                        entry.display_name().to_owned()
                    } else {
                        format!("{} — {description}", entry.display_name())
                    }
                },
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    let noun = if catalog.len() == 1 { "entry" } else { "entries" };
    format!(
        "{}. Available catalog {noun}: {entries}.",
        description.trim().trim_end_matches('.')
    )
}

fn synthetic_tool_search(declaration: &ToolSearchToolParam, catalog: &[CatalogEntry]) -> ToolSearchToolParam {
    let mut normalized = ToolSearchHandler::normalized_param(declaration);
    let description = normalized.description.as_deref().unwrap_or_default();
    normalized.description = Some(synthetic_description(description, catalog));
    normalized
}

fn build_private_tools(
    public_tools: &[ResponsesTool],
    definitions: &[DefinitionRecord],
    definition_indexes: &HashMap<String, usize>,
    synthetic_tool_search: Option<&ToolSearchToolParam>,
) -> Vec<ResponsesTool> {
    public_tools
        .iter()
        .filter_map(|tool| match tool {
            ResponsesTool::ToolSearch(_) => synthetic_tool_search.cloned().map(ResponsesTool::ToolSearch),
            ResponsesTool::Function(_) | ResponsesTool::Namespace(_) => {
                private_definition(tool, definitions, definition_indexes)
            }
            ResponsesTool::Mcp(_)
            | ResponsesTool::WebSearch(_)
            | ResponsesTool::FileSearch(_)
            | ResponsesTool::CodeInterpreter(_)
            | ResponsesTool::Custom(_)
            | ResponsesTool::Unknown => Some(tool.clone()),
        })
        .collect()
}

fn available_public_tools(public_tools: &[ResponsesTool], loaded_tools: &[ResponsesTool]) -> Vec<ResponsesTool> {
    let mut loaded_functions = HashSet::new();
    let mut loaded_namespace_members = HashMap::<&str, HashSet<&str>>::new();
    for tool in loaded_tools {
        match tool {
            ResponsesTool::Function(function) => {
                loaded_functions.insert(function.name.as_str());
            }
            ResponsesTool::Namespace(namespace) => {
                let members = loaded_namespace_members.entry(namespace.name.as_str()).or_default();
                members.extend(namespace.tools.iter().filter_map(|member| match member {
                    CodexNamespaceMember::Function(function) => Some(function.name.as_str()),
                    CodexNamespaceMember::Unknown => None,
                }));
            }
            ResponsesTool::ToolSearch(_)
            | ResponsesTool::Mcp(_)
            | ResponsesTool::WebSearch(_)
            | ResponsesTool::FileSearch(_)
            | ResponsesTool::CodeInterpreter(_)
            | ResponsesTool::Custom(_)
            | ResponsesTool::Unknown => {}
        }
    }

    public_tools
        .iter()
        .filter_map(|tool| match tool {
            ResponsesTool::Function(function) => {
                let loaded = loaded_functions.contains(function.name.as_str());
                if function.defer_loading == Some(true) && !loaded {
                    return None;
                }
                let mut available = function.clone();
                if loaded {
                    available.defer_loading = None;
                }
                Some(ResponsesTool::Function(available))
            }
            ResponsesTool::Namespace(namespace) => {
                let loaded_members = loaded_namespace_members.get(namespace.name.as_str());
                let mut available = namespace.clone();
                available.tools = available
                    .tools
                    .into_iter()
                    .filter_map(|member| match member {
                        CodexNamespaceMember::Function(mut function) => {
                            let loaded = loaded_members.is_some_and(|members| members.contains(function.name.as_str()));
                            if function.defer_loading == Some(true) && !loaded {
                                return None;
                            }
                            if loaded {
                                function.defer_loading = None;
                            }
                            Some(CodexNamespaceMember::Function(function))
                        }
                        CodexNamespaceMember::Unknown => None,
                    })
                    .collect();
                (!available.tools.is_empty()).then_some(ResponsesTool::Namespace(available))
            }
            ResponsesTool::Mcp(_)
            | ResponsesTool::WebSearch(_)
            | ResponsesTool::FileSearch(_)
            | ResponsesTool::CodeInterpreter(_)
            | ResponsesTool::Custom(_)
            | ResponsesTool::Unknown => Some(tool.clone()),
            ResponsesTool::ToolSearch(_) => None,
        })
        .collect()
}

fn private_definition(
    tool: &ResponsesTool,
    definitions: &[DefinitionRecord],
    definition_indexes: &HashMap<String, usize>,
) -> Option<ResponsesTool> {
    let identity = loaded_tool_identity(tool).ok().flatten()?;
    let loaded = definitions[*definition_indexes.get(identity.name())?].loaded;
    match tool {
        ResponsesTool::Function(function) if loaded || function.defer_loading != Some(true) => {
            let mut function = function.clone();
            function.defer_loading = None;
            Some(ResponsesTool::Function(function))
        }
        ResponsesTool::Namespace(namespace) => private_namespace(
            namespace,
            definitions[*definition_indexes.get(identity.name())?]
                .namespace_members
                .as_ref(),
        )
        .map(ResponsesTool::Namespace),
        ResponsesTool::Function(_)
        | ResponsesTool::Mcp(_)
        | ResponsesTool::ToolSearch(_)
        | ResponsesTool::WebSearch(_)
        | ResponsesTool::FileSearch(_)
        | ResponsesTool::CodeInterpreter(_)
        | ResponsesTool::Custom(_)
        | ResponsesTool::Unknown => None,
    }
}

fn private_namespace(
    namespace: &CodexNamespaceToolParam,
    member_records: Option<&NamespaceMemberRecords>,
) -> Option<CodexNamespaceToolParam> {
    let member_records = member_records?;
    let tools = namespace
        .tools
        .iter()
        .filter_map(|member| match member {
            CodexNamespaceMember::Function(function)
                if member_records
                    .indexes
                    .get(function.name.as_str())
                    .is_some_and(|index| member_records.ordered[*index].loaded) =>
            {
                let mut function = function.clone();
                function.defer_loading = None;
                Some(CodexNamespaceMember::Function(function))
            }
            CodexNamespaceMember::Function(_) => None,
            CodexNamespaceMember::Unknown => Some(CodexNamespaceMember::Unknown),
        })
        .collect::<Vec<_>>();
    (!tools.is_empty()).then(|| CodexNamespaceToolParam {
        tools,
        ..namespace.clone()
    })
}

fn namespace_has_withheld_member(member_records: Option<&NamespaceMemberRecords>) -> bool {
    member_records.is_some_and(|members| members.unloaded_count != 0)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tool::ToolRegistry;

    fn param(value: Value) -> ToolSearchToolParam {
        let ResponsesTool::ToolSearch(param) = serde_json::from_value(value).expect("valid tool_search declaration")
        else {
            panic!("expected tool_search");
        };
        param
    }

    fn assert_invalid_blocking_search(registry: &ToolRegistry, case: &str, item: &Value) {
        let body = json!({"status": "completed", "output": [item]}).to_string();
        assert!(
            matches!(
                registry.validate_blocking_response(&body),
                Err(ToolError::InvalidUpstreamToolSearch)
            ),
            "{case}"
        );
    }

    #[test]
    fn handler_validates_and_normalizes_exactly_one_function() {
        let param = param(json!({
            "type": "tool_search",
            "execution": "client",
            "description": "Find matching tools",
            "parameters": {"type": "array", "items": {"type": "string"}}
        }));
        ToolSearchHandler.validate(&param).unwrap();
        assert_eq!(ToolSearchHandler.tool_type(), ToolType::ToolSearch);
        assert_eq!(
            serde_json::to_value(ToolSearchHandler.normalize(&param)).unwrap(),
            json!([{
                "type": "function",
                "name": "tool_search",
                "description": "Find matching tools",
                "parameters": {"type": "array", "items": {"type": "string"}},
                "strict": false
            }])
        );
    }

    #[test]
    fn handler_rejects_non_object_parameter_values() {
        let param = param(json!({
            "type": "tool_search",
            "execution": "client",
            "parameters": ["not", "an", "object"]
        }));

        let error = ToolSearchHandler
            .validate(&param)
            .expect_err("private function parameters require an object");
        assert!(error.to_string().contains("parameters must be a JSON object"));
    }

    #[test]
    fn normalization_uses_safe_defaults() {
        let param = param(json!({"type": "tool_search", "execution": "client", "description": "  "}));
        assert_eq!(
            serde_json::to_value(ToolSearchHandler.normalize(&param)).unwrap(),
            json!([{
                "type": "function",
                "name": "tool_search",
                "description": "Search the client tool catalog",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {
                        "type": "string",
                        "description": "A concise description of the needed capabilities."
                    }},
                    "required": ["query"],
                    "additionalProperties": false
                },
                "strict": false
            }])
        );
    }

    #[test]
    fn synthetic_public_call_construction_is_validated_in_tool_layer() {
        let valid = FunctionToolCall {
            id: "fc_search".to_owned(),
            call_id: "call_search".to_owned(),
            name: TOOL_SEARCH_NAME.to_owned(),
            namespace: None,
            arguments: r#"["weather","timezone"]"#.to_owned(),
            status: MessageStatus::Completed,
        };
        let started = started_public_call(&valid).expect("valid started call");
        assert_eq!(started.id, "tsc_search");
        assert_eq!(started.status, ToolSearchStatus::InProgress);
        let completed = completed_public_call(&valid).expect("valid completed call");
        assert_eq!(completed.arguments, json!(["weather", "timezone"]));

        for invalid in [
            FunctionToolCall {
                name: "ordinary".to_owned(),
                ..valid.clone()
            },
            FunctionToolCall {
                namespace: Some("catalog".to_owned()),
                ..valid.clone()
            },
            FunctionToolCall {
                arguments: "not valid JSON".to_owned(),
                ..valid.clone()
            },
            FunctionToolCall {
                status: MessageStatus::InProgress,
                ..valid.clone()
            },
        ] {
            assert!(completed_public_call(&invalid).is_err());
        }
    }

    #[test]
    fn terminal_projection_rejects_or_discards_unfinished_calls() {
        let synthetic = FunctionToolCall {
            id: "fc_search".to_owned(),
            call_id: "call_search".to_owned(),
            name: TOOL_SEARCH_NAME.to_owned(),
            namespace: None,
            arguments: String::new(),
            status: MessageStatus::InProgress,
        };
        assert!(project_synthetic_call(&synthetic, false, true).is_err());
        assert!(project_synthetic_call(&synthetic, true, true).unwrap().is_none());

        let native = ToolSearchCall {
            id: "tsc_search".to_owned(),
            call_id: "call_search".to_owned(),
            execution: crate::types::tools::ToolSearchExecution::Client,
            arguments: json!({}),
            status: ToolSearchStatus::Incomplete,
        };
        assert!(project_native_call(&native, false).is_err());
        assert!(project_native_call(&native, true).unwrap().is_none());
    }

    #[test]
    fn registry_requires_tool_search_preparation_before_upstream_conversion() {
        let mut request: RequestPayload = serde_json::from_value(json!({
            "model": "test",
            "input": "find weather",
            "parallel_tool_calls": false,
            "tools": [{"type": "tool_search", "execution": "client"}]
        }))
        .expect("request shape");

        assert!(ToolRegistry::default().ensure_request_prepared(&request).is_err());
        let state = ToolSearchHandler::prepare_request(&mut request, &[], false)
            .expect("tool-search preparation")
            .expect("active tool-search state");
        let mut registry =
            ToolRegistry::from_tool_types(HashMap::from([(TOOL_SEARCH_NAME.to_owned(), ToolType::ToolSearch)]));
        registry
            .install_tool_search_state(Some(state))
            .expect("install prepared tool-search state");
        registry
            .ensure_request_prepared(&request)
            .expect("prepared request is ready for upstream conversion");
    }

    #[test]
    fn preparation_retains_mcp_list_metadata_until_upstream_projection() {
        let mut request: RequestPayload = serde_json::from_value(json!({
            "model": "test",
            "input": [
                {
                    "type": "mcp_list_tools",
                    "id": "mcpl_counter",
                    "server_label": "counter",
                    "tools": []
                },
                {"role": "user", "content": "find weather"}
            ],
            "parallel_tool_calls": false,
            "tools": [{"type": "tool_search", "execution": "client"}]
        }))
        .expect("request shape");

        ToolSearchHandler::prepare_request(&mut request, &[], false)
            .expect("tool-search preparation")
            .expect("active tool-search state");

        let ResponsesInput::Items(prepared_items) = &request.input else {
            panic!("expected prepared item input");
        };
        assert!(
            prepared_items
                .iter()
                .any(|item| matches!(item, InputItem::McpListTools(list) if list.server_label == "counter"))
        );

        let model_input = request.input.model_input();
        let ResponsesInput::Items(model_items) = model_input.as_ref() else {
            panic!("expected model item input");
        };
        assert!(
            model_items
                .iter()
                .all(|item| !matches!(item, InputItem::McpListTools(_)))
        );
    }

    #[test]
    fn ordinary_function_named_tool_search_does_not_require_preparation() {
        let request: RequestPayload = serde_json::from_value(json!({
            "model": "test",
            "input": "call the ordinary function",
            "tools": [{"type": "function", "name": "tool_search"}]
        }))
        .expect("ordinary function request");

        ToolRegistry::default()
            .ensure_request_prepared(&request)
            .expect("the reserved name applies only to active tool search");
    }

    #[test]
    fn registry_strictly_validates_blocking_search_without_changing_inactive_functions() {
        let mut request: RequestPayload = serde_json::from_value(json!({
            "model": "test",
            "input": "find weather",
            "parallel_tool_calls": false,
            "tools": [{"type": "tool_search", "execution": "client"}]
        }))
        .expect("request shape");
        let state = ToolSearchHandler::prepare_request(&mut request, &[], false)
            .expect("tool-search preparation")
            .expect("active tool-search state");
        let mut registry =
            ToolRegistry::from_tool_types(HashMap::from([(TOOL_SEARCH_NAME.to_owned(), ToolType::ToolSearch)]));
        registry
            .install_tool_search_state(Some(state))
            .expect("install prepared tool-search state");
        let native = json!({
            "type": "tool_search_call",
            "id": "tsc_1",
            "call_id": "call_search",
            "execution": "client",
            "arguments": ["weather", "timezone"],
            "status": "completed"
        });
        let synthetic = json!({
            "type": "function_call",
            "id": "fc_search",
            "call_id": "call_search",
            "name": "tool_search",
            "arguments": "[\"weather\",\"timezone\"]",
            "status": "completed"
        });

        for item in [&native, &synthetic] {
            let body = json!({"status": "completed", "output": [item]}).to_string();
            registry
                .validate_blocking_response(&body)
                .expect("native and synthetic array arguments are valid");
        }
        let malformed = [
            ("native missing id", {
                let mut item = native.clone();
                item.as_object_mut().unwrap().remove("id");
                item
            }),
            ("native missing call_id", {
                let mut item = native.clone();
                item.as_object_mut().unwrap().remove("call_id");
                item
            }),
            ("native missing arguments", {
                let mut item = native.clone();
                item.as_object_mut().unwrap().remove("arguments");
                item
            }),
            ("native namespace", {
                let mut item = native.clone();
                item["namespace"] = json!("catalog");
                item
            }),
            ("synthetic missing status", {
                let mut item = synthetic.clone();
                item.as_object_mut().unwrap().remove("status");
                item
            }),
            ("synthetic null status", {
                let mut item = synthetic.clone();
                item["status"] = Value::Null;
                item
            }),
            ("synthetic invalid JSON arguments", {
                let mut item = synthetic.clone();
                item["arguments"] = json!("not valid JSON");
                item
            }),
        ];

        for (case, item) in malformed {
            assert_invalid_blocking_search(&registry, case, &item);
        }

        let partial = json!({
            "status": "incomplete",
            "output": [{
                "type": "function_call",
                "id": "fc_partial",
                "call_id": "call_partial",
                "name": "tool_search",
                "arguments": "{\"query\":",
                "status": "in_progress"
            }]
        })
        .to_string();
        registry
            .validate_blocking_response(&partial)
            .expect("unfinished search placeholder is allowed on an incomplete response");

        let ordinary = json!({
            "status": "completed",
            "output": [{"type": "function_call", "name": "tool_search", "arguments": "{}"}]
        })
        .to_string();
        ToolRegistry::default()
            .validate_blocking_response(&ordinary)
            .expect("inactive ordinary function keeps generic compatibility defaults");
    }

    #[test]
    fn prepared_response_tools_remove_request_scoped_mcp_secrets_and_discovery() {
        let mut request: RequestPayload = serde_json::from_value(json!({
            "model": "test",
            "input": "find weather",
            "parallel_tool_calls": false,
            "tools": [
                {"type": "tool_search", "execution": "client"},
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": "https://mcp.example.test/mcp",
                    "headers": {"Authorization": "Bearer header-secret"},
                    "authorization": "field-secret",
                    "_agentic_discovered_tools": [{
                        "server_label": "weather",
                        "tool_name": "forecast",
                        "internal_name": "mcp__weather__forecast",
                        "tool": {"name": "forecast", "inputSchema": {"type": "object"}}
                    }]
                }
            ]
        }))
        .expect("request shape");

        let state = ToolSearchHandler::prepare_request(&mut request, &[], false)
            .expect("tool-search preparation")
            .expect("active tool-search state");
        let mut registry =
            ToolRegistry::from_tool_types(HashMap::from([(TOOL_SEARCH_NAME.to_owned(), ToolType::ToolSearch)]));
        registry
            .install_tool_search_state(Some(state))
            .expect("install prepared tool-search state");
        let serialized = serde_json::to_value(registry.tool_search_response_tools().expect("active public tools"))
            .expect("public tools serialize");
        let serialized = serialized.to_string();

        for secret in [
            "header-secret",
            "field-secret",
            "mcp__weather__forecast",
            "_agentic_discovered_tools",
        ] {
            assert!(!serialized.contains(secret));
        }
    }

    #[test]
    fn public_tool_search_item_ids_are_stable_and_domain_separated() {
        assert_eq!(public_item_id("tsc_existing"), "tsc_existing");
        assert_eq!(public_item_id("fc_search_1"), "tsc_search_1");
        let first = public_item_id("provider-item-1");
        assert_eq!(first, public_item_id("provider-item-1"));
        assert!(first.starts_with("tsc_"));
        assert_ne!(first, crate::tool::custom::public_item_id("provider-item-1"));
    }

    #[test]
    fn response_tools_after_search_keep_immediate_and_loaded_public_availability() {
        let request: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "store": false,
            "tools": [
                {
                    "type": "tool_search",
                    "execution": "client",
                    "description": "Find tools",
                    "parameters": {"type": "object"}
                },
                {"type": "function", "name": "always_ready"},
                {"type": "function", "name": "get_weather", "defer_loading": true},
                {"type": "function", "name": "not_loaded", "defer_loading": true},
                {
                    "type": "namespace",
                    "name": "travel",
                    "tools": [
                        {"type": "function", "name": "always_ready_member"},
                        {"type": "function", "name": "get_timezone", "defer_loading": true},
                        {"type": "function", "name": "not_loaded_member", "defer_loading": true}
                    ]
                }
            ],
            "input": [
                {
                    "type": "tool_search_call",
                    "id": "tsc_1",
                    "call_id": "call_search_1",
                    "arguments": {"query": "weather and timezone"}
                },
                {
                    "type": "tool_search_output",
                    "call_id": "call_search_1",
                    "tools": [
                        {"type": "function", "name": "get_weather", "defer_loading": true},
                        {
                            "type": "namespace",
                            "name": "travel",
                            "tools": [{"type": "function", "name": "get_timezone", "defer_loading": true}]
                        }
                    ]
                }
            ]
        }))
        .expect("valid mixed-availability tool-search request");

        let state = ToolSearchState::build(&request).expect("tool-search state");
        let tools = serialize_to_value(&state.public_response_tools()).expect("response tools serialize");
        assert_eq!(
            tools,
            serde_json::json!([
                {"type": "function", "name": "always_ready"},
                {"type": "function", "name": "get_weather"},
                {
                    "type": "namespace",
                    "name": "travel",
                    "tools": [
                        {"type": "function", "name": "always_ready_member"},
                        {"type": "function", "name": "get_timezone"}
                    ]
                }
            ])
        );
    }

    #[test]
    fn tool_search_output_rejects_mcp_definitions() {
        let request: RequestPayload = serde_json::from_value(serde_json::json!({
            "model": "test",
            "store": false,
            "parallel_tool_calls": false,
            "tools": [{
                "type": "tool_search",
                "execution": "client",
                "description": "Find a tool",
                "parameters": {"type": "object"}
            }],
            "input": [
                {
                    "type": "tool_search_call",
                    "id": "tsc_1",
                    "call_id": "call_search_1",
                    "arguments": {"query": "weather"}
                },
                {
                    "type": "tool_search_output",
                    "call_id": "call_search_1",
                    "tools": [{
                        "type": "mcp",
                        "server_label": "weather",
                        "server_url": "https://mcp.example.test/mcp"
                    }]
                }
            ]
        }))
        .expect("typed request");

        let error = ToolSearchState::build(&request).expect_err("MCP is not a client-loaded tool definition");

        assert!(matches!(
            error,
            ToolError::Config(message) if message.contains("unsupported tool definition")
        ));
    }
}
