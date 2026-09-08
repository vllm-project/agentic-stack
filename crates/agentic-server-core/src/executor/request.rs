use std::sync::Arc;
use std::time::Duration;

use crate::config::{Config, default_database_url};
use crate::error::Error;
use crate::executor::gateway::GatewaySchedulerPolicy;
use crate::executor::modes::{ConversationHandler, ResponseHandler};
use crate::storage::backend::redact_database_urls;
use crate::storage::{
    ConversationStore, ConversationVersion, DatabaseBackend, ResponseStore, create_pool_with_schema_and_configs,
};
use crate::tool::{GatewayExecutorRegistration, GatewayExecutors};
use crate::types::io::InputItem;
use crate::types::messages::GatewayToolMap;
use crate::types::request_response::{RequestPayload, ResponsePayload};

/// Env var configuring client-tool → gateway-executor aliases for `/v1/messages`
/// (e.g. `WebSearch=web_search`). Empty/unset means no aliases — client
/// functions stay client-owned (the ownership doctrine's default).
const GATEWAY_TOOL_ALIASES_ENV: &str = "MESSAGES_GATEWAY_TOOL_ALIASES";

/// Context built by `rehydrate_conversation`, threaded through the execute pipeline.
#[derive(Debug)]
pub struct RequestContext {
    /// Untouched original request from the client.
    pub original_request: RequestPayload,
    /// Enriched request with rehydrated conversation history injected into `.input`.
    /// This is the request forwarded to the LLM.
    pub enriched_request: RequestPayload,
    /// This turn's canonical input items for persistence, including built-in calls
    /// and call outputs. Session execution also records each round's other output
    /// items here in inference order; compaction can replace them with a new window.
    pub new_input_items: Vec<InputItem>,
    /// Our generated response ID (uuid7 with "resp_" prefix).
    pub response_id: String,
    /// Resolved conversation ID. `None` when `store=false` or non-conversational.
    pub conversation_id: Option<String>,
    /// Conversation version captured with rehydrated history.
    /// `None` for non-conversation and `previous_response_id` execution.
    pub conversation_version: Option<ConversationVersion>,
    /// Optional transient-session lease and canonical parent snapshot. Never serialized.
    pub continuation: Option<super::session::ResponseContinuation>,
}

impl RequestContext {
    /// Inject our `response_id` and `conversation_id` into a `ResponsePayload`
    /// received from the LLM (which carries the upstream's own IDs).
    pub(crate) fn inject_ids(&self, payload: &mut ResponsePayload) {
        payload.id.clone_from(&self.response_id);
        payload.conversation_id.clone_from(&self.conversation_id);
        payload
            .previous_response_id
            .clone_from(&self.original_request.previous_response_id);
    }
}

/// Runtime dependencies passed into `execute()`.
///
/// Owns the storage handlers, HTTP client, and LLM endpoint configuration.
/// Per-request auth is supplied via [`crate::executor::engine::ExecuteRequest::with_auth`]
/// rather than stored here, keeping this context purely shared and immutable.
#[derive(Clone, Debug)]
pub struct ExecutionContext {
    pub conv_handler: ConversationHandler,
    pub resp_handler: ResponseHandler,
    pub client: Arc<reqwest::Client>,
    pub gateway_executors: GatewayExecutors,
    /// Client-tool → gateway-executor aliases for the `/v1/messages` loop
    /// (e.g. Claude Code's `WebSearch` → `web_search`). Empty unless configured.
    pub messages_gateway_tools: GatewayToolMap,
    /// Base URL for the LLM backend, e.g. `"http://localhost:8000"`.
    pub llm_base_url: String,
    /// Maximum wait time for the next SSE chunk.  `Duration::ZERO` disables the timeout.
    /// Sourced from the `STREAMING_CHUNK_TIMEOUT_S` environment variable, defaulting to
    /// [`DEFAULT_STREAMING_TIMEOUT`] when unset or unparseable.
    pub streaming_timeout: Duration,
    /// Bounded-concurrency policy applied to gateway-owned calls in each round.
    pub(crate) gateway_scheduler_policy: GatewaySchedulerPolicy,
    storage_pool: Option<Arc<crate::storage::DbPool>>,
}

impl ExecutionContext {
    /// Returns the full URL for the `/v1/responses` endpoint.
    #[must_use]
    pub fn responses_url(&self) -> String {
        format!("{}/v1/responses", self.llm_base_url)
    }

    /// Returns the full URL for the `/v1/conversations` endpoint.
    #[must_use]
    pub fn conversations_url(&self) -> String {
        format!("{}/v1/conversations", self.llm_base_url)
    }

    #[must_use]
    pub fn new(
        conv_handler: ConversationHandler,
        resp_handler: ResponseHandler,
        client: Arc<reqwest::Client>,
        llm_base_url: String,
    ) -> Self {
        let gateway_executors = GatewayExecutors::from_env(Arc::clone(&client));
        Self {
            conv_handler,
            resp_handler,
            client,
            gateway_executors,
            messages_gateway_tools: messages_gateway_tools_from_env(),
            llm_base_url,
            streaming_timeout: streaming_timeout_from_env(),
            gateway_scheduler_policy: GatewaySchedulerPolicy::default(),
            storage_pool: None,
        }
    }

    #[must_use]
    pub fn with_gateway_executor(mut self, executor: impl Into<GatewayExecutorRegistration>) -> Self {
        self.gateway_executors.insert(executor);
        self
    }

    /// Checks whether configured persistence can execute a bounded query.
    pub async fn storage_ready(&self, timeout: Duration) -> bool {
        let Some(pool) = &self.storage_pool else {
            return true;
        };
        matches!(
            tokio::time::timeout(timeout, crate::storage::schema::verify_persistence_ready(pool.as_ref())).await,
            Ok(Ok(()))
        )
    }

    /// Returns the configured persistence pool, if storage is enabled.
    #[must_use]
    pub fn storage_pool(&self) -> Option<&crate::storage::DbPool> {
        self.storage_pool.as_deref()
    }

    /// Build an `ExecutionContext` directly from [`Config`](crate::config::Config).
    ///
    /// Creates the database pool, both storage handlers, and an HTTP client
    /// internally so callers don't need to depend on the storage layer.
    ///
    /// # Errors
    ///
    /// Returns an error if the database pool cannot be opened or the schema
    /// migration fails.
    pub async fn from_config(cfg: &Config) -> Result<Self, Error> {
        let default_db_url = cfg.db_url.is_none().then(default_database_url).transpose()?;
        let db_url = cfg
            .db_url
            .as_deref()
            .or(default_db_url.as_deref())
            .ok_or_else(|| Error::Config("default database URL was not resolved".to_owned()))?;
        let database_backend = DatabaseBackend::from_url(db_url)
            .map_err(|error| Error::Config(format!("invalid DATABASE_URL: {error}")))?;
        let pool = create_pool_with_schema_and_configs(Some(db_url), cfg.sqlite, cfg.postgres)
            .await
            .map_err(|error| database_open_error(database_backend, &error))?;
        crate::storage::schema::verify_persistence_writable(pool.as_ref())
            .await
            .map_err(|error| database_open_error(database_backend, &error))?;

        let conv_handler = ConversationHandler::new(ConversationStore::new(pool.clone()));
        let resp_handler = ResponseHandler::new(ResponseStore::new(pool.clone()));
        let client = Arc::new(reqwest::Client::new());
        let gateway_executors = GatewayExecutors::from_config(Arc::clone(&client), &cfg.tools)
            .map_err(|error| Error::Config(format!("failed to validate configured MCP server policies: {error}")))?;
        Ok(Self {
            conv_handler,
            resp_handler,
            client,
            gateway_executors,
            messages_gateway_tools: std::env::var(GATEWAY_TOOL_ALIASES_ENV)
                .ok()
                .as_deref()
                .or(cfg.tools.messages_gateway_tool_aliases.as_deref())
                .map(GatewayToolMap::from_env_str)
                .unwrap_or_default(),
            llm_base_url: cfg.llm_api_base.clone(),
            streaming_timeout: streaming_timeout_from_env(),
            gateway_scheduler_policy: GatewaySchedulerPolicy::new(cfg.tools.max_concurrent_gateway_calls),
            storage_pool: Some(pool),
        })
    }
}

/// Default streaming chunk timeout when `STREAMING_CHUNK_TIMEOUT_S` is unset or
/// unparseable. Generous to accommodate long-prefill / high-TTFT workloads.
pub const DEFAULT_STREAMING_TIMEOUT: Duration = Duration::from_secs(600);

fn streaming_timeout_from_env() -> Duration {
    parse_streaming_timeout(std::env::var("STREAMING_CHUNK_TIMEOUT_S").ok().as_deref())
}

/// Parse a `STREAMING_CHUNK_TIMEOUT_S` value into a chunk timeout.
///
/// Falls back to [`DEFAULT_STREAMING_TIMEOUT`] when the value is absent or not a
/// valid non-negative integer. A value of `0` yields `Duration::ZERO`, which
/// disables the per-chunk timeout.
fn parse_streaming_timeout(value: Option<&str>) -> Duration {
    value
        .and_then(|v| v.parse::<u64>().ok())
        .map_or(DEFAULT_STREAMING_TIMEOUT, Duration::from_secs)
}

fn database_open_error(database_backend: DatabaseBackend, error: &sqlx::Error) -> Error {
    let category = match error {
        sqlx::Error::Configuration(_) | sqlx::Error::InvalidArgument(_) => "configuration error".to_owned(),
        sqlx::Error::Database(database_error) => database_error
            .code()
            .filter(|code| code.len() <= 5 && code.bytes().all(|byte| byte.is_ascii_alphanumeric()))
            .map_or_else(
                || "database error".to_owned(),
                |code| format!("database error (SQLSTATE {code})"),
            ),
        sqlx::Error::Io(_) => "database I/O error".to_owned(),
        sqlx::Error::Tls(_) => "database TLS error".to_owned(),
        sqlx::Error::Protocol(_) => "database protocol error".to_owned(),
        sqlx::Error::PoolTimedOut => "connection pool timeout".to_owned(),
        sqlx::Error::PoolClosed => "connection pool closed".to_owned(),
        sqlx::Error::WorkerCrashed => "database worker crashed".to_owned(),
        sqlx::Error::Migrate(_) => "database migration error".to_owned(),
        _ => "database error".to_owned(),
    };
    let detail = redact_database_urls(&error.to_string());
    Error::Config(format!(
        "failed to open {} database: {category}: {detail}",
        database_backend.display_name()
    ))
}

/// Load the `/v1/messages` gateway-tool alias map from the environment.
fn messages_gateway_tools_from_env() -> GatewayToolMap {
    std::env::var(GATEWAY_TOOL_ALIASES_ENV)
        .ok()
        .map(|raw| GatewayToolMap::from_env_str(&raw))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::{DEFAULT_STREAMING_TIMEOUT, ExecutionContext, database_open_error, parse_streaming_timeout};
    use crate::executor::{ConversationHandler, ResponseHandler};
    use crate::storage::{ConversationStore, DatabaseBackend, ResponseStore, create_pool_with_schema};

    #[test]
    fn streaming_timeout_parsing_covers_unset_invalid_zero_and_valid() {
        // Unset falls back to the default.
        assert_eq!(parse_streaming_timeout(None), DEFAULT_STREAMING_TIMEOUT);
        // Non-numeric and negative values are unparseable → default.
        assert_eq!(parse_streaming_timeout(Some("")), DEFAULT_STREAMING_TIMEOUT);
        assert_eq!(parse_streaming_timeout(Some("abc")), DEFAULT_STREAMING_TIMEOUT);
        assert_eq!(parse_streaming_timeout(Some("-1")), DEFAULT_STREAMING_TIMEOUT);
        // `0` disables the timeout.
        assert_eq!(parse_streaming_timeout(Some("0")), Duration::ZERO);
        // A valid value is honored.
        assert_eq!(parse_streaming_timeout(Some("30")), Duration::from_secs(30));
    }

    #[test]
    fn database_errors_are_actionable_without_exposing_credentials() {
        let database_url = "postgresql://agentic-api:super'secret@postgres.example.com/agentic_api";
        let backend = DatabaseBackend::from_url(database_url).expect("valid PostgreSQL URL");
        let tls_error = sqlx::Error::Tls(Box::new(std::io::Error::other(format!(
            "{database_url} certificate verify failed"
        ))));
        let pool_error = sqlx::Error::PoolTimedOut;
        let tls_message = database_open_error(backend, &tls_error).to_string();
        let pool_message = database_open_error(backend, &pool_error).to_string();

        assert!(tls_message.contains("database TLS error"));
        assert!(tls_message.contains("certificate verify failed"));
        assert!(tls_message.contains("postgresql://[redacted]"));
        assert_eq!(
            pool_message,
            "failed to open PostgreSQL database: connection pool timeout: \
             pool timed out while waiting for an open connection"
        );
        assert!(!tls_message.contains("super-secret"));
        assert!(!tls_message.contains("secret"));
        assert!(!tls_message.contains("agentic-api"));
        assert_ne!(tls_message, pool_message);

        let mysql_error = sqlx::Error::Io(std::io::Error::other(
            "mysql://gateway:mysql-secret@mysql.example.com/agentic_api refused",
        ));
        let mysql_message = database_open_error(DatabaseBackend::Other, &mysql_error).to_string();
        assert!(mysql_message.contains("mysql://[redacted]"));
        assert!(!mysql_message.contains("mysql-secret"));

        let short_postgres_url = "postgres://gateway:postgres-secret@postgres.example.com/agentic_api";
        let short_postgres_error = sqlx::Error::Io(std::io::Error::other(format!("{short_postgres_url} refused")));
        let short_postgres_backend = DatabaseBackend::from_url(short_postgres_url).expect("valid PostgreSQL URL");
        let short_postgres_message = database_open_error(short_postgres_backend, &short_postgres_error).to_string();
        assert!(short_postgres_message.contains("failed to open PostgreSQL database"));
        assert!(short_postgres_message.contains("postgres://[redacted]"));
        assert!(!short_postgres_message.contains("postgres-secret"));
    }

    #[test]
    fn uppercase_database_urls_are_classified_and_redacted() {
        let database_url = "POSTGRESQL://gateway:postgres-secret@postgres.example.com/agentic_api";
        let error = sqlx::Error::Io(std::io::Error::other(format!("{database_url} refused")));
        let backend = DatabaseBackend::from_url(database_url).expect("valid uppercase PostgreSQL URL");
        let message = database_open_error(backend, &error).to_string();

        assert!(message.contains("failed to open PostgreSQL database"));
        assert!(message.contains("postgresql://[redacted]"));
        assert!(!message.contains("postgres-secret"));
    }

    #[tokio::test]
    async fn storage_readiness_is_bounded_by_the_probe_timeout() {
        let pool = create_pool_with_schema(Some("sqlite://?mode=memory"))
            .await
            .expect("create single-connection pool");
        let held_connection = pool.acquire().await.expect("hold the only connection");
        let mut context = ExecutionContext::new(
            ConversationHandler::new(ConversationStore::disabled()),
            ResponseHandler::new(ResponseStore::disabled()),
            Arc::new(reqwest::Client::new()),
            "http://localhost:8000".to_owned(),
        );
        context.storage_pool = Some(pool.clone());

        assert!(!context.storage_ready(Duration::from_millis(10)).await);
        drop(held_connection);
        assert!(context.storage_ready(Duration::from_secs(1)).await);
    }

    #[tokio::test]
    async fn storage_readiness_rejects_missing_persistence_tables() {
        let pool = create_pool_with_schema(Some("sqlite://?mode=memory"))
            .await
            .expect("create persistence pool");
        let mut context = ExecutionContext::new(
            ConversationHandler::new(ConversationStore::disabled()),
            ResponseHandler::new(ResponseStore::disabled()),
            Arc::new(reqwest::Client::new()),
            "http://localhost:8000".to_owned(),
        );
        context.storage_pool = Some(pool.clone());
        assert!(context.storage_ready(Duration::from_secs(1)).await);

        sqlx::query("DROP TABLE responses")
            .execute(pool.as_ref())
            .await
            .expect("drop persistence table");

        assert!(!context.storage_ready(Duration::from_secs(1)).await);
    }

    #[tokio::test]
    async fn storage_readiness_rejects_read_only_persistence() {
        let pool = create_pool_with_schema(Some("sqlite://?mode=memory"))
            .await
            .expect("create persistence pool");
        let mut context = ExecutionContext::new(
            ConversationHandler::new(ConversationStore::disabled()),
            ResponseHandler::new(ResponseStore::disabled()),
            Arc::new(reqwest::Client::new()),
            "http://localhost:8000".to_owned(),
        );
        context.storage_pool = Some(pool.clone());
        assert!(context.storage_ready(Duration::from_secs(1)).await);

        sqlx::query("PRAGMA query_only = ON")
            .execute(pool.as_ref())
            .await
            .expect("make persistence read-only");

        assert!(!context.storage_ready(Duration::from_secs(1)).await);
    }

    #[tokio::test]
    async fn storage_readiness_rolls_back_probe_rows() {
        let pool = create_pool_with_schema(Some("sqlite://?mode=memory"))
            .await
            .expect("create persistence pool");

        crate::storage::schema::verify_persistence_writable(pool.as_ref())
            .await
            .expect("run functional persistence probe");
        for table in ["conversations", "items", "responses"] {
            let row_count: i64 = sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table}"))
                .fetch_one(pool.as_ref())
                .await
                .expect("count persistence rows");
            assert_eq!(row_count, 0, "readiness probe leaked a row into {table}");
        }
    }
}
