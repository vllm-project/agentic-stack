use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::error::Error;
use crate::tool::McpServerEntry;

pub const AGENTIC_API_HOME_ENV: &str = "AGENTIC_API_HOME";
pub const CONFIG_FILE_NAME: &str = "config.toml";
pub const DATABASE_FILE_NAME: &str = "agentic_api.db";

pub const DEFAULT_POSTGRES_MAX_CONNECTIONS: u32 = 10;
pub const DEFAULT_POSTGRES_ACQUIRE_TIMEOUT_SECONDS: u64 = 30;
pub const DEFAULT_POSTGRES_IDLE_TIMEOUT_SECONDS: u64 = 600;
pub const DEFAULT_POSTGRES_LOCK_TIMEOUT_SECONDS: u64 = 5;
pub const DEFAULT_POSTGRES_MAX_LIFETIME_SECONDS: u64 = 1_800;
pub const DEFAULT_POSTGRES_MIGRATION_TIMEOUT_SECONDS: u64 = 300;
pub const DEFAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS: u64 = 30;
pub const DEFAULT_SQLITE_MAX_CONNECTIONS: u32 = 4;
pub const DEFAULT_SQLITE_JOURNAL_SIZE_LIMIT_BYTES: u64 = 6_144_000;
pub const DEFAULT_SQLITE_MMAP_SIZE_BYTES: u64 = 268_435_456;
pub const DEFAULT_MAX_CONCURRENT_GATEWAY_CALLS: NonZeroUsize = NonZeroUsize::new(5).expect("default is nonzero");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostgresConfig {
    pub max_connections: u32,
    pub acquire_timeout: Duration,
    pub lock_timeout: Duration,
    pub migration_timeout: Duration,
    pub statement_timeout: Duration,
    pub idle_timeout: Option<Duration>,
    pub max_lifetime: Option<Duration>,
}

impl Default for PostgresConfig {
    fn default() -> Self {
        Self {
            max_connections: DEFAULT_POSTGRES_MAX_CONNECTIONS,
            acquire_timeout: Duration::from_secs(DEFAULT_POSTGRES_ACQUIRE_TIMEOUT_SECONDS),
            lock_timeout: Duration::from_secs(DEFAULT_POSTGRES_LOCK_TIMEOUT_SECONDS),
            migration_timeout: Duration::from_secs(DEFAULT_POSTGRES_MIGRATION_TIMEOUT_SECONDS),
            statement_timeout: Duration::from_secs(DEFAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS),
            idle_timeout: Some(Duration::from_secs(DEFAULT_POSTGRES_IDLE_TIMEOUT_SECONDS)),
            max_lifetime: Some(Duration::from_secs(DEFAULT_POSTGRES_MAX_LIFETIME_SECONDS)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SqliteTempStore {
    Default,
    File,
    #[default]
    Memory,
}

impl SqliteTempStore {
    #[must_use]
    pub fn as_pragma_value(self) -> &'static str {
        match self {
            Self::Default => "DEFAULT",
            Self::File => "FILE",
            Self::Memory => "MEMORY",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqliteConfig {
    pub max_connections: u32,
    pub journal_size_limit_bytes: u64,
    pub temp_store: SqliteTempStore,
    pub mmap_size_bytes: u64,
}

impl Default for SqliteConfig {
    fn default() -> Self {
        Self {
            max_connections: DEFAULT_SQLITE_MAX_CONNECTIONS,
            journal_size_limit_bytes: DEFAULT_SQLITE_JOURNAL_SIZE_LIMIT_BYTES,
            temp_store: SqliteTempStore::default(),
            mmap_size_bytes: DEFAULT_SQLITE_MMAP_SIZE_BYTES,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct WebSearchProviderConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ToolRuntimeConfig {
    pub web_search: WebSearchProviderConfig,
    pub mcp_servers: HashMap<String, McpServerEntry>,
    pub mcp_allowed_hosts: Vec<String>,
    pub messages_gateway_tool_aliases: Option<String>,
    /// Upper bound on gateway-owned tool calls executing concurrently within one
    /// round. A sliding window admits another call as one finishes. Handlers with
    /// nested outbound work also use this value as their provider-level concurrency
    /// ceiling; individual handlers may further serialize calls to the same tool
    /// name. The nonzero type prevents constructing a scheduler window that can
    /// never be polled.
    pub max_concurrent_gateway_calls: NonZeroUsize,
}

impl Default for ToolRuntimeConfig {
    fn default() -> Self {
        Self {
            web_search: WebSearchProviderConfig::default(),
            mcp_servers: HashMap::default(),
            mcp_allowed_hosts: Vec::default(),
            messages_gateway_tool_aliases: None,
            max_concurrent_gateway_calls: DEFAULT_MAX_CONCURRENT_GATEWAY_CALLS,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub llm_api_base: String,
    pub openai_api_key: Option<String>,
    pub llm_ready_timeout_s: f64,
    pub llm_ready_interval_s: f64,
    pub skip_llm_ready_check: bool,
    /// Database URL for conversation and response storage.
    /// `None` uses the local database in the Agentic API home directory.
    pub db_url: Option<String>,
    pub postgres: PostgresConfig,
    pub sqlite: SqliteConfig,
    pub tools: ToolRuntimeConfig,
}

/// Resolves the directory used for user configuration and local state.
///
/// `AGENTIC_API_HOME` takes precedence over the default `~/.agentic-api`.
/// The returned path is absolute, but it is not created by this function.
///
/// # Errors
///
/// Returns a configuration error when the home directory cannot be found or
/// `AGENTIC_API_HOME` is not an absolute path.
pub fn agentic_api_home() -> Result<PathBuf, Error> {
    let configured = std::env::var_os(AGENTIC_API_HOME_ENV).filter(|value| !value.is_empty());
    resolve_agentic_api_home(configured.map(PathBuf::from), dirs::home_dir())
}

fn resolve_agentic_api_home(configured: Option<PathBuf>, user_home: Option<PathBuf>) -> Result<PathBuf, Error> {
    if let Some(path) = configured {
        if !path.is_absolute() {
            return Err(Error::Config(format!(
                "{AGENTIC_API_HOME_ENV} must be an absolute path: {}",
                path.display()
            )));
        }
        return Ok(path);
    }

    let user_home = user_home.ok_or_else(|| Error::Config("could not determine the user home directory".to_owned()))?;
    if !user_home.is_absolute() {
        return Err(Error::Config(format!(
            "user home directory must be an absolute path: {}",
            user_home.display()
        )));
    }
    Ok(user_home.join(".agentic-api"))
}

/// Resolves and creates the Agentic API home directory.
///
/// # Errors
///
/// Returns an error when the path cannot be resolved or created, or when an
/// existing path is not a directory.
pub fn ensure_agentic_api_home() -> Result<PathBuf, Error> {
    let path = agentic_api_home()?;
    std::fs::create_dir_all(&path).map_err(|error| {
        Error::Config(format!(
            "failed to create Agentic API home directory {}: {error}",
            path.display()
        ))
    })?;
    if !path.is_dir() {
        return Err(Error::Config(format!(
            "Agentic API home path is not a directory: {}",
            path.display()
        )));
    }
    Ok(path)
}

/// Returns the default `SQLite` URL inside the Agentic API home directory.
///
/// # Errors
///
/// Returns an error when the home directory cannot be resolved or created.
pub fn default_database_url() -> Result<String, Error> {
    default_database_url_in(&ensure_agentic_api_home()?)
}

fn default_database_url_in(home: &Path) -> Result<String, Error> {
    const SQLITE_PATH_ENCODE_SET: &percent_encoding::AsciiSet = &percent_encoding::CONTROLS
        .add(b' ')
        .add(b'"')
        .add(b'#')
        .add(b'<')
        .add(b'>')
        .add(b'?')
        .add(b'%')
        .add(b'`')
        .add(b'{')
        .add(b'}');

    let path = home.join(DATABASE_FILE_NAME);
    let path = path
        .to_str()
        .ok_or_else(|| Error::Config(format!("default database path is not valid UTF-8: {}", path.display())))?;
    #[cfg(windows)]
    let path = path.replace('\\', "/");
    let encoded = percent_encoding::utf8_percent_encode(path, SQLITE_PATH_ENCODE_SET);
    Ok(format!("sqlite://{encoded}"))
}

#[must_use]
pub fn normalize_base_url(url: &str) -> String {
    let mut s = url.trim_end_matches('/').to_owned();
    if s.ends_with("/v1") {
        s.truncate(s.len() - 3);
        s = s.trim_end_matches('/').to_owned();
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn strip_trailing_v1() {
        assert_eq!(normalize_base_url("http://host:8000/v1"), "http://host:8000");
        assert_eq!(normalize_base_url("http://host:8000/v1/"), "http://host:8000");
    }

    #[test]
    fn no_v1_unchanged() {
        assert_eq!(normalize_base_url("http://host:8000"), "http://host:8000");
        assert_eq!(normalize_base_url("http://host:8000/"), "http://host:8000");
    }

    #[test]
    fn home_override_takes_precedence() {
        let configured = if cfg!(windows) {
            PathBuf::from(r"C:\agentic-home")
        } else {
            PathBuf::from("/tmp/agentic-home")
        };
        let resolved = resolve_agentic_api_home(Some(configured.clone()), Some(PathBuf::from("/ignored")))
            .expect("absolute configured home");
        assert_eq!(resolved, configured);
    }

    #[test]
    fn default_home_is_hidden_directory() {
        let user_home = if cfg!(windows) {
            PathBuf::from(r"C:\Users\agentic")
        } else {
            PathBuf::from("/home/agentic")
        };
        let resolved = resolve_agentic_api_home(None, Some(user_home.clone())).expect("user home");
        assert_eq!(resolved, user_home.join(".agentic-api"));
    }

    #[test]
    fn relative_home_override_is_rejected() {
        let error = resolve_agentic_api_home(Some(PathBuf::from("relative")), Some(PathBuf::from("/home/agentic")))
            .expect_err("relative override must fail");
        assert!(error.to_string().contains("must be an absolute path"));
    }

    #[test]
    fn database_url_uses_home_directory() {
        let home = if cfg!(windows) {
            PathBuf::from(r"C:\Users\agentic\.agentic-api")
        } else {
            PathBuf::from("/home/agentic/.agentic-api")
        };
        let url = default_database_url_in(&home).expect("database URL");
        assert!(url.starts_with("sqlite://"));
        assert!(url.ends_with("/.agentic-api/agentic_api.db"));
    }

    #[test]
    fn database_url_encodes_url_delimiters_in_home_path() {
        let home = if cfg!(windows) {
            PathBuf::from(r"C:\Users\agentic api\state?#%")
        } else {
            PathBuf::from("/home/agentic api/state?#%")
        };
        let url = default_database_url_in(&home).expect("database URL");
        assert!(url.contains("agentic%20api"));
        assert!(url.contains("state%3F%23%25"));
    }
}
