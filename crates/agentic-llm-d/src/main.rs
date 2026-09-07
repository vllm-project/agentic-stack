//! `agentic-llm-d` — the coordinator calls `hydrate`, runs inference against its
//! own model fleet, then calls `persist`. This binary does neither.

use clap::Parser;
use tokio_util::sync::CancellationToken;

use agentic_core::config::{Config, PostgresConfig, SqliteConfig, ToolRuntimeConfig};
use agentic_llm_d::runner;

#[derive(Parser)]
#[command(name = "agentic-llm-d", about = "agentic-api backend mode for the llm-d coordinator")]
struct Cli {
    /// Matches `agentic-server`'s default. Keep the listener cluster-internal.
    #[arg(long, env = "AGENTIC_LLM_D_HOST", default_value = "0.0.0.0")]
    host: String,
    #[arg(long, env = "AGENTIC_LLM_D_PORT", default_value_t = 8081)]
    port: u16,
    /// Defaults to the local database under the agentic-api home.
    #[arg(long, env = "DATABASE_URL")]
    db_url: Option<String>,

    /// Seals the context passed between hydrate and persist. Without it a caller
    /// could forge one and write turns under any id. 32+ characters of
    /// independently generated randomness, distinct from the API token.
    #[arg(long, env = "AGENTIC_LLM_D_SIGNING_KEY", hide_env_values = true)]
    signing_key: String,

    /// Shared secret every split-route caller must present; these endpoints read
    /// and write conversation history. Same requirement as the signing key.
    #[arg(long, env = "AGENTIC_LLM_D_API_TOKEN", hide_env_values = true)]
    api_token: String,
}

#[tokio::main]
async fn main() -> Result<(), runner::Error> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    // Every model-facing field is inert here; llm-d owns the fleet.
    let config = Config {
        llm_api_base: String::new(),
        openai_api_key: None,
        llm_ready_timeout_s: 0.0,
        llm_ready_interval_s: 0.0,
        skip_llm_ready_check: true,
        db_url: cli.db_url,
        postgres: PostgresConfig::default(),
        sqlite: SqliteConfig::default(),
        tools: ToolRuntimeConfig::default(),
    };

    let shutdown = CancellationToken::new();
    let on_signal = shutdown.clone();
    tokio::spawn(async move {
        if shutdown_signal().await.is_ok() {
            on_signal.cancel();
        }
    });

    runner::serve(
        &config,
        cli.signing_key.into_bytes(),
        cli.api_token,
        &cli.host,
        cli.port,
        shutdown,
    )
    .await
}

/// SIGTERM's default action would bypass graceful shutdown and cut off in-flight
/// requests.
#[cfg(unix)]
async fn shutdown_signal() -> Result<(), std::io::Error> {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;

    tokio::select! {
        signal = tokio::signal::ctrl_c() => signal,
        _ = terminate.recv() => Ok(()),
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() -> Result<(), std::io::Error> {
    tokio::signal::ctrl_c().await
}
