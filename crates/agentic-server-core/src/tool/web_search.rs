use std::collections::HashMap;
use std::future::Future;
use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Semaphore;

use super::handler::MAX_GATEWAY_TOOL_OUTPUT_BYTES;
use super::handler::{GatewayExecutor, GatewayToolEventPlan, ToolError, ToolHandler, ToolOutput};
use super::ownership::GatewayBinding;
use super::registry::{ToolEntry, ToolType};
use crate::config::DEFAULT_MAX_CONCURRENT_GATEWAY_CALLS;
use crate::types::io::output::{FunctionToolCall, WebSearchCall, WebSearchCallStatus, WebSearchSource};
use crate::types::io::{FunctionTool, OutputItem};
use crate::types::tools::{WebSearchContextSize, WebSearchToolParam};

const YOU_API_KEY: &str = "YOU_API_KEY";
const YOU_API_BASE_URL: &str = "YOU_API_BASE_URL";
const MAX_WEB_SEARCH_QUERIES: usize = 5;

#[derive(Default)]
struct CountingWriter {
    bytes: usize,
}

impl Write for CountingWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(buffer.len())
            .ok_or_else(|| io::Error::other("serialized JSON size overflow"))?;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) type WebSearchExecutor =
    dyn GatewayExecutor<ToolParams = WebSearchToolParam, ExecutionParams = WebSearchToolParam>;

pub(crate) fn insert_web_search_entry(
    entries: &mut HashMap<String, ToolEntry>,
    params: &WebSearchToolParam,
    executor: Arc<WebSearchExecutor>,
) {
    entries.insert(
        "web_search".to_owned(),
        ToolEntry::gateway(
            ToolType::WebSearch,
            None,
            Some(GatewayBinding::new(executor, params.clone())),
        ),
    );
}

#[must_use]
pub(crate) fn web_search_function_tool() -> FunctionTool {
    FunctionTool {
        type_: "function".to_owned(),
        name: "web_search".to_owned(),
        description: Some(
            "Search the public web for current information and return structured web and news results.".to_owned(),
        ),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language web search query."
                },
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": MAX_WEB_SEARCH_QUERIES,
                    "description": "Multiple independent search queries to run in parallel, instead of a single query."
                },
                "count": {
                    "type": "integer",
                    "description": "Maximum results per section, from 1 to 100."
                },
                "freshness": {
                    "type": "string",
                    "description": "Optional recency filter: day, week, month, year, or YYYY-MM-DDtoYYYY-MM-DD."
                },
                "country": {
                    "type": "string",
                    "description": "Optional ISO 3166-1 alpha-2 country code."
                },
                "language": {
                    "type": "string",
                    "description": "Optional BCP 47 language code."
                },
                "include_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional strict allowlist of domains."
                },
                "exclude_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional domain blocklist."
                }
            },
            "anyOf": [
                {"required": ["query"]},
                {"required": ["queries"]}
            ]
        })),
        strict: Some(false),
    }
}

#[must_use]
pub(crate) fn output_item(
    call: &FunctionToolCall,
    output: &ToolOutput,
    status: WebSearchCallStatus,
) -> Option<OutputItem> {
    let parsed_output = serde_json::from_str::<Value>(&output.output).ok();
    let queries = parsed_output
        .as_ref()
        .and_then(queries_from_value)
        .or_else(|| queries_from_arguments(&call.arguments))
        .unwrap_or_else(|| vec![String::new()]);
    let sources = parsed_output.as_ref().map(sources_from_output).unwrap_or_default();
    WebSearchCall::try_new(call_output_id(call), status, queries, sources)
        .map(OutputItem::WebSearchCall)
        .ok()
}

#[must_use]
pub(crate) fn started_output_item(call: &FunctionToolCall) -> Option<OutputItem> {
    WebSearchCall::try_new(
        call_output_id(call),
        WebSearchCallStatus::InProgress,
        queries_from_arguments(&call.arguments).unwrap_or_else(|| vec![String::new()]),
        Vec::new(),
    )
    .map(OutputItem::WebSearchCall)
    .ok()
}

#[derive(Debug, Clone)]
pub struct WebSearchHandler {
    provider: Option<Arc<dyn WebSearchProvider>>,
    max_concurrent_queries: NonZeroUsize,
    query_permits: Arc<Semaphore>,
}

impl WebSearchHandler {
    #[must_use]
    pub fn from_env(client: Arc<reqwest::Client>) -> Self {
        Self::from_values(
            client,
            std::env::var(YOU_API_KEY).ok(),
            std::env::var(YOU_API_BASE_URL).ok(),
            DEFAULT_MAX_CONCURRENT_GATEWAY_CALLS,
        )
    }

    #[must_use]
    pub fn from_values(
        client: Arc<reqwest::Client>,
        api_key: Option<String>,
        base_url: Option<String>,
        max_concurrent_queries: NonZeroUsize,
    ) -> Self {
        Self::with_provider_and_query_concurrency(
            Arc::new(YouSearchProvider::from_values(client, api_key, base_url)),
            max_concurrent_queries,
        )
    }

    #[must_use]
    pub fn with_api_key(client: Arc<reqwest::Client>, api_key: String, base_url: &str) -> Self {
        Self::with_provider_and_query_concurrency(
            Arc::new(YouSearchProvider::with_api_key(client, api_key, base_url)),
            DEFAULT_MAX_CONCURRENT_GATEWAY_CALLS,
        )
    }

    /// Builds a handler usable only for shaping placeholder/error output
    /// (`ToolHandler::normalize`, `GatewayExecutor::started_output`/`public_output`)
    /// when no real provider is configured — `execute()` always fails.
    #[must_use]
    pub fn spec_only() -> Self {
        Self {
            provider: None,
            max_concurrent_queries: DEFAULT_MAX_CONCURRENT_GATEWAY_CALLS,
            query_permits: Arc::new(Semaphore::new(DEFAULT_MAX_CONCURRENT_GATEWAY_CALLS.get())),
        }
    }

    #[cfg(test)]
    fn with_provider(provider: Arc<dyn WebSearchProvider>) -> Self {
        Self::with_provider_and_query_concurrency(provider, DEFAULT_MAX_CONCURRENT_GATEWAY_CALLS)
    }

    fn with_provider_and_query_concurrency(
        provider: Arc<dyn WebSearchProvider>,
        max_concurrent_queries: NonZeroUsize,
    ) -> Self {
        Self {
            provider: Some(provider),
            max_concurrent_queries,
            query_permits: Arc::new(Semaphore::new(max_concurrent_queries.get())),
        }
    }

    async fn execute_search(
        &self,
        call_id: &str,
        arguments: &str,
        params: &WebSearchToolParam,
    ) -> Result<ToolOutput, ToolError> {
        let provider = self
            .provider
            .as_ref()
            .ok_or_else(|| ToolError::Config("web_search spec-only handler cannot execute tools".to_owned()))?;
        let args = WebSearchArguments::from_json(arguments)?;
        let queries = args.all_queries();
        let args_ref = &args;
        let mut responses = Box::pin(
            futures::stream::iter(queries.iter().cloned())
                .map(|query| {
                    let provider = Arc::clone(provider);
                    let query_permits = Arc::clone(&self.query_permits);
                    async move {
                        let _permit = query_permits.acquire_owned().await.map_err(|error| {
                            ToolError::Execution(format!("web_search query scheduler closed: {error}"))
                        })?;
                        provider.search(&query, args_ref, params).await
                    }
                })
                .buffered(self.max_concurrent_queries.get()),
        );

        let mut web = Vec::new();
        let mut news = Vec::new();
        let mut metadata = Vec::new();
        let mut accumulated_bytes = 0usize;
        while let Some(mut response) = responses.try_next().await? {
            let mut counter = CountingWriter::default();
            serde_json::to_writer(&mut counter, &response.results)
                .and_then(|()| serde_json::to_writer(&mut counter, &response.metadata))
                .map_err(|error| ToolError::Execution(format!("failed to size web_search output: {error}")))?;
            accumulated_bytes = accumulated_bytes.saturating_add(counter.bytes);
            if accumulated_bytes > MAX_GATEWAY_TOOL_OUTPUT_BYTES {
                return Err(ToolError::Execution(format!(
                    "web_search output exceeded {MAX_GATEWAY_TOOL_OUTPUT_BYTES} bytes"
                )));
            }
            if let Some(results) = response.results.get_mut("web").and_then(Value::as_array_mut) {
                web.append(results);
            }
            if let Some(results) = response.results.get_mut("news").and_then(Value::as_array_mut) {
                news.append(results);
            }
            metadata.push(response.metadata);
        }
        let output = serde_json::to_string(&serde_json::json!({
            "query": queries[0],
            "queries": queries,
            "results": {"web": web, "news": news},
            "metadata": metadata
        }))
        .map_err(|e| ToolError::Execution(format!("failed to serialize web_search output: {e}")))?;
        if output.len() > MAX_GATEWAY_TOOL_OUTPUT_BYTES {
            return Err(ToolError::Execution(format!(
                "web_search output exceeded {MAX_GATEWAY_TOOL_OUTPUT_BYTES} bytes"
            )));
        }

        Ok(ToolOutput {
            call_id: call_id.to_owned(),
            output,
        })
    }
}

trait WebSearchProvider: std::fmt::Debug + Send + Sync {
    fn search<'a>(
        &'a self,
        query: &'a str,
        args: &'a WebSearchArguments,
        config: &'a WebSearchToolParam,
    ) -> Pin<Box<dyn Future<Output = Result<WebSearchProviderResponse, ToolError>> + Send + 'a>>;
}

struct WebSearchProviderResponse {
    results: Value,
    metadata: Value,
}

#[derive(Debug, Clone)]
struct YouSearchProvider {
    client: Arc<reqwest::Client>,
    api_key: Option<String>,
    base_url: Option<String>,
}

impl YouSearchProvider {
    fn from_values(client: Arc<reqwest::Client>, api_key: Option<String>, base_url: Option<String>) -> Self {
        let api_key = api_key
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
        let base_url = base_url.and_then(|value| clean_base_url(&value));
        Self {
            client,
            api_key,
            base_url,
        }
    }

    fn with_api_key(client: Arc<reqwest::Client>, api_key: String, base_url: &str) -> Self {
        Self {
            client,
            api_key: Some(api_key),
            base_url: clean_base_url(base_url),
        }
    }
}

impl WebSearchProvider for YouSearchProvider {
    fn search<'a>(
        &'a self,
        query: &'a str,
        args: &'a WebSearchArguments,
        config: &'a WebSearchToolParam,
    ) -> Pin<Box<dyn Future<Output = Result<WebSearchProviderResponse, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            let api_key = self
                .api_key
                .as_deref()
                .ok_or_else(|| ToolError::Config(format!("{YOU_API_KEY} must be set to use the web_search tool")))?;
            let base_url = self.base_url.as_deref().ok_or_else(|| {
                ToolError::Config(format!("{YOU_API_BASE_URL} must be set to use the web_search tool"))
            })?;
            let request = YouSearchRequest::from_args_and_config(query, args, config)?;
            let resp = self
                .client
                .get(format!("{base_url}/v1/search"))
                .query(&request.query_params())
                .header("X-API-Key", api_key)
                .send()
                .await
                .map_err(|e| ToolError::Execution(format!("You.com search request failed: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = read_search_response_limited(resp).await.unwrap_or_default();
                return Err(ToolError::Execution(format!(
                    "You.com search returned {status}: {body}"
                )));
            }

            let response_text = read_search_response_limited(resp).await?;
            let response: Value = serde_json::from_str(&response_text)
                .map_err(|e| ToolError::Execution(format!("You.com search returned invalid JSON: {e}")))?;
            Ok(WebSearchProviderResponse {
                results: response
                    .get("results")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"web": [], "news": []})),
                metadata: response.get("metadata").cloned().unwrap_or(Value::Null),
            })
        })
    }
}

async fn read_search_response_limited(resp: reqwest::Response) -> Result<String, ToolError> {
    let mut stream = resp.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|error| ToolError::Execution(format!("failed to read You.com search response: {error}")))?;
        if chunk.len() > MAX_GATEWAY_TOOL_OUTPUT_BYTES.saturating_sub(body.len()) {
            return Err(ToolError::Execution(format!(
                "You.com search response exceeded {MAX_GATEWAY_TOOL_OUTPUT_BYTES} bytes"
            )));
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body).map_err(|_| ToolError::Execution("You.com search response was not valid UTF-8".to_owned()))
}

impl ToolHandler for WebSearchHandler {
    type ToolParams = WebSearchToolParam;

    fn tool_type(&self) -> ToolType {
        ToolType::WebSearch
    }

    fn validate(&self, _params: &WebSearchToolParam) -> Result<(), ToolError> {
        Ok(())
    }

    fn normalize(&self, _params: &WebSearchToolParam) -> Vec<FunctionTool> {
        vec![web_search_function_tool()]
    }
}

impl GatewayExecutor for WebSearchHandler {
    type ExecutionParams = WebSearchToolParam;

    fn execute(
        &self,
        call_id: &str,
        tool_name: &str,
        arguments: &str,
        params: &WebSearchToolParam,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + '_>> {
        let call_id = call_id.to_owned();
        let tool_name = tool_name.to_owned();
        let arguments = arguments.to_owned();
        let params = params.clone();
        Box::pin(async move {
            if tool_name != "web_search" {
                return Err(ToolError::Config(format!(
                    "web_search handler cannot execute tool '{tool_name}'"
                )));
            }
            self.execute_search(&call_id, &arguments, &params).await
        })
    }

    fn supports_parallel_execution(&self) -> bool {
        true
    }

    fn plan_gateway_events(&self, call: &FunctionToolCall, _params: &WebSearchToolParam) -> GatewayToolEventPlan {
        GatewayToolEventPlan::new(started_output_item(call))
    }

    fn public_output(
        &self,
        call: &FunctionToolCall,
        output: &ToolOutput,
        status: WebSearchCallStatus,
        _params: &WebSearchToolParam,
    ) -> Option<OutputItem> {
        output_item(call, output, status)
    }
}

#[derive(Debug, Deserialize)]
struct WebSearchArguments {
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    queries: Option<Vec<String>>,
    count: Option<u16>,
    freshness: Option<String>,
    country: Option<String>,
    language: Option<String>,
    safesearch: Option<String>,
    livecrawl: Option<String>,
    livecrawl_formats: Option<Vec<String>>,
    crawl_timeout: Option<u16>,
    include_domains: Option<Vec<String>>,
    exclude_domains: Option<Vec<String>>,
    boost_domains: Option<Vec<String>>,
}

impl WebSearchArguments {
    fn from_json(arguments: &str) -> Result<Self, ToolError> {
        let args = serde_json::from_str::<Self>(arguments)
            .map_err(|e| ToolError::Config(format!("web_search arguments must be valid JSON: {e}")))?;
        let query_count = args.all_queries().len();
        if query_count == 0 {
            return Err(ToolError::Config(
                "web_search requires a non-empty query or queries".to_owned(),
            ));
        }
        if query_count > MAX_WEB_SEARCH_QUERIES {
            return Err(ToolError::Config(format!(
                "web_search accepts at most {MAX_WEB_SEARCH_QUERIES} queries per call"
            )));
        }
        Ok(args)
    }

    fn all_queries(&self) -> Vec<String> {
        let queries = clean_vec(self.queries.as_deref()).unwrap_or_default();
        if !queries.is_empty() {
            return queries;
        }
        clean_string(self.query.as_deref()).into_iter().collect()
    }
}

#[derive(Debug, Serialize)]
struct YouSearchRequest {
    query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    count: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    freshness: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safesearch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    livecrawl: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    livecrawl_formats: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    crawl_timeout: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exclude_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    boost_domains: Option<Vec<String>>,
}

impl YouSearchRequest {
    fn query_params(&self) -> Vec<(String, String)> {
        let mut params = vec![("query".to_owned(), self.query.clone())];
        if let Some(count) = self.count {
            params.push(("count".to_owned(), count.to_string()));
        }
        if let Some(freshness) = &self.freshness {
            params.push(("freshness".to_owned(), freshness.clone()));
        }
        if let Some(country) = &self.country {
            params.push(("country".to_owned(), country.clone()));
        }
        if let Some(language) = &self.language {
            params.push(("language".to_owned(), language.clone()));
        }
        if let Some(safesearch) = &self.safesearch {
            params.push(("safesearch".to_owned(), safesearch.clone()));
        }
        if let Some(livecrawl) = &self.livecrawl {
            params.push(("livecrawl".to_owned(), livecrawl.clone()));
        }
        for format in self.livecrawl_formats.iter().flatten() {
            params.push(("livecrawl_formats".to_owned(), format.clone()));
        }
        if let Some(crawl_timeout) = self.crawl_timeout {
            params.push(("crawl_timeout".to_owned(), crawl_timeout.to_string()));
        }
        for domain in self.include_domains.iter().flatten() {
            params.push(("include_domains".to_owned(), domain.clone()));
        }
        for domain in self.exclude_domains.iter().flatten() {
            params.push(("exclude_domains".to_owned(), domain.clone()));
        }
        for domain in self.boost_domains.iter().flatten() {
            params.push(("boost_domains".to_owned(), domain.clone()));
        }
        params
    }

    fn from_args_and_config(
        query: &str,
        args: &WebSearchArguments,
        config: &WebSearchToolParam,
    ) -> Result<Self, ToolError> {
        let count = args
            .count
            .or_else(|| {
                config
                    .search_context_size
                    .map(WebSearchContextSize::default_count)
                    .map(u16::from)
            })
            .map(validate_count)
            .transpose()?;
        let crawl_timeout = args.crawl_timeout.map(validate_crawl_timeout).transpose()?;
        let config_domains = config
            .filters
            .as_ref()
            .and_then(|filters| clean_vec(filters.allowed_domains.as_deref()));
        let config_blocked_domains = config
            .filters
            .as_ref()
            .and_then(|filters| clean_vec(filters.blocked_domains.as_deref()));
        let include_domains = config_domains.or_else(|| clean_vec(args.include_domains.as_deref()));
        let exclude_domains = config_blocked_domains.or_else(|| clean_vec(args.exclude_domains.as_deref()));
        let boost_domains = clean_vec(args.boost_domains.as_deref());
        if include_domains.is_some() && (exclude_domains.is_some() || boost_domains.is_some()) {
            return Err(ToolError::Config(
                "include_domains cannot be combined with exclude_domains or boost_domains".to_owned(),
            ));
        }
        let country = config
            .user_location
            .as_ref()
            .and_then(|location| clean_string(location.country.as_deref()))
            .or_else(|| clean_string(args.country.as_deref()))
            .map(|value| value.to_ascii_uppercase());

        Ok(Self {
            query: query.trim().to_owned(),
            count,
            freshness: clean_string(args.freshness.as_deref()),
            country,
            language: clean_string(args.language.as_deref()),
            safesearch: clean_string(args.safesearch.as_deref()),
            livecrawl: clean_string(args.livecrawl.as_deref()),
            livecrawl_formats: clean_vec(args.livecrawl_formats.as_deref()),
            crawl_timeout,
            include_domains,
            exclude_domains,
            boost_domains,
        })
    }
}

fn validate_count(count: u16) -> Result<u8, ToolError> {
    if (1..=100).contains(&count) {
        Ok(u8::try_from(count).expect("validated web_search count must fit in u8"))
    } else {
        Err(ToolError::Config(
            "web_search count must be between 1 and 100".to_owned(),
        ))
    }
}

fn validate_crawl_timeout(timeout: u16) -> Result<u8, ToolError> {
    if (1..=60).contains(&timeout) {
        u8::try_from(timeout).map_err(|e| ToolError::Config(format!("invalid crawl_timeout: {e}")))
    } else {
        Err(ToolError::Config(
            "web_search crawl_timeout must be between 1 and 60".to_owned(),
        ))
    }
}

fn clean_string(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn clean_json_str(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn call_output_id(call: &FunctionToolCall) -> String {
    if let Some(suffix) = call.id.strip_prefix("fc_").filter(|suffix| !suffix.is_empty()) {
        return format!("ws_{suffix}");
    }
    if let Some(suffix) = call.call_id.strip_prefix("call_").filter(|suffix| !suffix.is_empty()) {
        return format!("ws_{suffix}");
    }
    crate::utils::uuid7_str("ws_")
}

fn queries_from_value(value: &Value) -> Option<Vec<String>> {
    let queries: Vec<String> = value
        .get("queries")?
        .as_array()?
        .iter()
        .filter_map(|item| clean_json_str(Some(item)))
        .collect();
    (!queries.is_empty()).then_some(queries)
}

fn queries_from_arguments(arguments: &str) -> Option<Vec<String>> {
    let args = serde_json::from_str::<Value>(arguments).ok()?;
    queries_from_value(&args).or_else(|| clean_json_str(args.get("query")).map(|query| vec![query]))
}

fn sources_from_output(output: &Value) -> Vec<WebSearchSource> {
    ["web", "news"]
        .into_iter()
        .filter_map(|section| output.get("results")?.get(section)?.as_array())
        .flat_map(|results| results.iter())
        .filter_map(source_from_result)
        .collect()
}

fn source_from_result(result: &Value) -> Option<WebSearchSource> {
    let url = clean_json_str(result.get("url"))?;
    Some(WebSearchSource {
        url,
        title: clean_json_str(result.get("title")),
    })
}

fn clean_base_url(value: &str) -> Option<String> {
    let trimmed = value.trim().trim_end_matches('/');
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

fn clean_vec(values: Option<&[String]>) -> Option<Vec<String>> {
    let cleaned: Vec<String> = values
        .unwrap_or_default()
        .iter()
        .filter_map(|value| clean_string(Some(value.as_str())))
        .collect();
    (!cleaned.is_empty()).then_some(cleaned)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use axum::body::Body;
    use bytes::Bytes;

    use super::*;

    #[derive(Debug)]
    struct MockSearchProvider;

    impl WebSearchProvider for MockSearchProvider {
        fn search<'a>(
            &'a self,
            _query: &'a str,
            _args: &'a WebSearchArguments,
            _config: &'a WebSearchToolParam,
        ) -> Pin<Box<dyn Future<Output = Result<WebSearchProviderResponse, ToolError>> + Send + 'a>> {
            Box::pin(async move {
                Ok(WebSearchProviderResponse {
                    results: serde_json::json!({
                        "web": [
                            {
                                "url": "https://example.com/potato",
                                "title": "Potato"
                            }
                        ],
                        "news": []
                    }),
                    metadata: serde_json::json!({"provider": "mock"}),
                })
            })
        }
    }

    #[derive(Debug)]
    struct LargeSearchProvider;

    impl WebSearchProvider for LargeSearchProvider {
        fn search<'a>(
            &'a self,
            _query: &'a str,
            _args: &'a WebSearchArguments,
            _config: &'a WebSearchToolParam,
        ) -> Pin<Box<dyn Future<Output = Result<WebSearchProviderResponse, ToolError>> + Send + 'a>> {
            Box::pin(async move {
                Ok(WebSearchProviderResponse {
                    results: serde_json::json!({
                        "web": [{"snippet": "x".repeat(600 * 1024)}],
                        "news": []
                    }),
                    metadata: Value::Null,
                })
            })
        }
    }

    #[derive(Debug, Default)]
    struct ConcurrencyTrackingProvider {
        active: AtomicUsize,
        max_active: AtomicUsize,
    }

    impl WebSearchProvider for ConcurrencyTrackingProvider {
        fn search<'a>(
            &'a self,
            _query: &'a str,
            _args: &'a WebSearchArguments,
            _config: &'a WebSearchToolParam,
        ) -> Pin<Box<dyn Future<Output = Result<WebSearchProviderResponse, ToolError>> + Send + 'a>> {
            Box::pin(async move {
                let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                self.max_active.fetch_max(active, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                self.active.fetch_sub(1, Ordering::SeqCst);
                Ok(WebSearchProviderResponse {
                    results: serde_json::json!({"web": [], "news": []}),
                    metadata: serde_json::json!({"provider": "tracking"}),
                })
            })
        }
    }

    #[test]
    fn web_search_schema_caps_batched_queries() {
        let parameters = web_search_function_tool().parameters.expect("web_search parameters");
        assert_eq!(parameters["properties"]["queries"]["maxItems"], MAX_WEB_SEARCH_QUERIES);
    }

    #[tokio::test]
    async fn you_search_response_body_is_bounded_while_reading() {
        let chunk = Bytes::from(vec![b'x'; MAX_GATEWAY_TOOL_OUTPUT_BYTES / 2 + 1]);
        let app = axum::Router::new().route(
            "/v1/search",
            axum::routing::get(move || {
                let chunks = [Ok::<_, std::convert::Infallible>(chunk.clone()), Ok(chunk.clone())];
                async move { Body::from_stream(futures::stream::iter(chunks)) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind search response limit server");
        let address = listener.local_addr().expect("search response limit server address");
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        let response = reqwest::Client::new()
            .get(format!("http://{address}/v1/search"))
            .send()
            .await
            .expect("fetch oversized search response");

        let error = read_search_response_limited(response)
            .await
            .expect_err("oversized search response must fail");
        assert!(error.to_string().contains("search response exceeded"));
        server.abort();
    }

    #[tokio::test]
    async fn web_search_handler_delegates_to_provider() {
        let handler = WebSearchHandler::with_provider(Arc::new(MockSearchProvider));
        let output = handler
            .execute(
                "call_search",
                "web_search",
                r#"{"query":" potato "}"#,
                &WebSearchToolParam::default(),
            )
            .await
            .unwrap();
        let body: Value = serde_json::from_str(&output.output).unwrap();
        assert_eq!(output.call_id, "call_search");
        assert_eq!(body["query"], "potato");
        assert_eq!(body["queries"], serde_json::json!(["potato"]));
        assert_eq!(body["metadata"][0]["provider"], "mock");
        assert_eq!(body["results"]["web"][0]["url"], "https://example.com/potato");
    }

    #[tokio::test]
    async fn web_search_handler_fans_out_multiple_queries() {
        let handler = WebSearchHandler::with_provider(Arc::new(MockSearchProvider));
        let output = handler
            .execute(
                "call_search",
                "web_search",
                r#"{"queries":["potato","tomato"]}"#,
                &WebSearchToolParam::default(),
            )
            .await
            .unwrap();
        let body: Value = serde_json::from_str(&output.output).unwrap();
        assert_eq!(body["query"], "potato");
        assert_eq!(body["queries"], serde_json::json!(["potato", "tomato"]));
        assert_eq!(body["results"]["web"].as_array().unwrap().len(), 2);
        assert_eq!(body["metadata"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn web_search_handler_bounds_aggregate_query_results() {
        let handler = WebSearchHandler::with_provider(Arc::new(LargeSearchProvider));
        let error = handler
            .execute(
                "call_search",
                "web_search",
                r#"{"queries":["potato","tomato"]}"#,
                &WebSearchToolParam::default(),
            )
            .await
            .expect_err("aggregate query results must be bounded");

        assert!(error.to_string().contains("web_search output exceeded"));
    }

    #[tokio::test]
    async fn web_search_handler_rejects_oversized_query_batches() {
        let handler = WebSearchHandler::with_provider(Arc::new(MockSearchProvider));
        let error = handler
            .execute(
                "call_search",
                "web_search",
                r#"{"queries":["one","two","three","four","five","six"]}"#,
                &WebSearchToolParam::default(),
            )
            .await
            .expect_err("oversized query batch must fail");

        assert_eq!(
            error.to_string(),
            format!("invalid tool config: web_search accepts at most {MAX_WEB_SEARCH_QUERIES} queries per call")
        );
    }

    #[tokio::test]
    async fn web_search_handler_shares_query_concurrency_across_calls() {
        let provider = Arc::new(ConcurrencyTrackingProvider::default());
        let handler = WebSearchHandler::with_provider_and_query_concurrency(
            provider.clone(),
            NonZeroUsize::new(2).expect("nonzero test limit"),
        );
        let params = WebSearchToolParam::default();
        let arguments = r#"{"queries":["one","two","three","four","five"]}"#;

        let (first, second) = tokio::join!(
            handler.execute("call_one", "web_search", arguments, &params),
            handler.execute("call_two", "web_search", arguments, &params),
        );

        first.expect("first batched call");
        second.expect("second batched call");
        assert_eq!(provider.max_active.load(Ordering::SeqCst), 2);
    }
}
