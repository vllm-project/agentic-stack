use std::num::NonZeroUsize;

use clap::{
    Args, Parser, Subcommand, ValueEnum,
    builder::{Styles, styling::AnsiColor},
};

use crate::agentic_output::redact_url;

const fn brand_styles() -> Styles {
    Styles::styled()
        .header(AnsiColor::BrightCyan.on_default().bold())
        .usage(AnsiColor::BrightBlue.on_default().bold())
        .literal(AnsiColor::BrightYellow.on_default().bold())
        .placeholder(AnsiColor::BrightMagenta.on_default())
        .valid(AnsiColor::BrightGreen.on_default())
}

pub const DEFAULT_DATABASE_URL: &str = "sqlite://./agentic_api.db";

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum Harness {
    Codex,
    Claude,
}

#[derive(Debug, Parser)]
#[command(
    name = "agentic",
    about = "Agentic API — local agent gateway for Claude Code and Codex",
    version,
    styles = brand_styles(),
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Start Agentic API and launch a coding harness
    Run {
        #[command(subcommand)]
        harness: HarnessCommand,
    },
    /// Launch a coding harness against an already-running Agentic API gateway
    Harness {
        #[command(subcommand)]
        harness: AttachedHarnessCommand,
    },
    /// Start Agentic API without launching a harness
    Serve(ServeOptions),
    /// Validate the local Agentic API session prerequisites
    Validate(ValidateOptions),
}

#[derive(Debug, Subcommand)]
pub enum AttachedHarnessCommand {
    /// Launch Codex with an isolated provider configuration
    Codex(AttachedHarnessOptions),
    /// Launch Claude Code with isolated provider and model configuration
    Claude(AttachedHarnessOptions),
}

#[derive(Args, Clone, Debug)]
pub struct AttachedHarnessOptions {
    /// URL of an already-running Agentic API gateway
    #[arg(long, value_parser = parse_upstream_url)]
    pub gateway_url: String,

    /// Model ID served by the gateway
    #[arg(long)]
    pub model: String,

    /// API key forwarded to the gateway and harness when configured
    #[arg(long, env = "AGENTIC_GATEWAY_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,

    /// Suppress lifecycle output
    #[arg(long)]
    pub quiet: bool,

    /// Skip harness permission prompts and sandbox restrictions
    #[arg(long)]
    pub yolo: bool,

    /// Disable ANSI color output
    #[arg(long)]
    pub no_color: bool,

    /// Arguments forwarded to the selected harness after `--`
    #[arg(last = true, allow_hyphen_values = true)]
    pub harness_args: Vec<String>,
}

#[derive(Debug, Subcommand)]
pub enum HarnessCommand {
    /// Launch Codex with an isolated provider configuration
    Codex(HarnessOptions),
    /// Launch Claude Code with an isolated gateway environment
    Claude(HarnessOptions),
}

#[derive(Args, Clone, Debug)]
pub struct HarnessOptions {
    #[command(flatten)]
    pub source: SourceOptions,

    #[command(flatten)]
    pub common: CommonOptions,

    /// Arguments forwarded to the selected harness after `--`
    #[arg(last = true, allow_hyphen_values = true)]
    pub harness_args: Vec<String>,
}

#[derive(Args, Clone, Debug)]
pub struct ServeOptions {
    #[command(flatten)]
    pub source: SourceOptions,

    #[command(flatten)]
    pub common: CommonOptions,
}

#[derive(Args, Clone, Debug)]
pub struct ValidateOptions {
    #[command(flatten)]
    pub source: SourceOptions,

    #[command(flatten)]
    pub common: CommonOptions,

    /// Also verify a harness binary without launching it
    #[arg(long, value_enum)]
    pub harness: Option<Harness>,
}

#[derive(Args, Clone, Debug)]
pub struct SourceOptions {
    /// Connect to an already-running OpenAI-compatible upstream (`http://` or `https://` base URL)
    #[arg(long, required_unless_present = "model", value_parser = parse_upstream_url)]
    pub upstream: Option<String>,

    /// Model to start with vLLM, or the model name to use with `--upstream`.
    /// When omitted alongside `--upstream`, the first model served by the upstream is used.
    #[arg(long, required_unless_present = "upstream")]
    pub model: Option<String>,

    /// vLLM port when starting a model
    #[arg(long, default_value_t = 8000)]
    pub llm_port: u16,
}

#[derive(Args, Clone, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct CommonOptions {
    /// Gateway bind host
    #[arg(long, default_value = "127.0.0.1", env = "GATEWAY_HOST")]
    pub gateway_host: String,

    /// Gateway bind port
    #[arg(long, default_value_t = 3000, env = "GATEWAY_PORT")]
    pub gateway_port: u16,

    /// `SQLite` or `PostgreSQL` storage URL
    #[arg(long, default_value = DEFAULT_DATABASE_URL, env = "DATABASE_URL", hide_env_values = true)]
    pub database_url: String,

    /// Maximum serialized request size in bytes accepted by the gateway.
    /// Left to the gateway's own configuration chain when omitted.
    #[arg(long)]
    pub max_request_body_size_bytes: Option<NonZeroUsize>,

    /// API key forwarded to the gateway and harness when configured
    #[arg(long, env = "OPENAI_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,

    /// Skip the upstream readiness probe
    #[arg(long, default_value_t = false)]
    pub skip_llm_ready_check: bool,

    /// Upstream readiness timeout in seconds
    #[arg(long, default_value_t = 600.0, value_parser = parse_timeout_seconds)]
    pub llm_ready_timeout_s: f64,

    /// Upstream readiness poll interval in seconds
    #[arg(long, default_value_t = 2.0, value_parser = parse_interval_seconds)]
    pub llm_ready_interval_s: f64,

    /// Suppress lifecycle output
    #[arg(long)]
    pub quiet: bool,

    /// Skip harness permission prompts and sandbox restrictions
    #[arg(long)]
    pub yolo: bool,

    /// Disable ANSI color output
    #[arg(long)]
    pub no_color: bool,
}

fn parse_upstream_url(value: &str) -> Result<String, String> {
    let display_value = redact_url(value);
    let parsed = url::Url::parse(value).map_err(|error| format!("invalid upstream URL `{display_value}`: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(format!(
            "invalid upstream URL `{display_value}`: expected an http:// or https:// base URL"
        ));
    }
    if parsed.host_str().is_none_or(str::is_empty) {
        return Err(format!("invalid upstream URL `{display_value}`: missing host"));
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(format!(
            "invalid upstream URL `{display_value}`: query strings and fragments are not supported; pass a base URL such as http://host:port"
        ));
    }
    Ok(value.trim_end_matches('/').to_owned())
}

fn parse_timeout_seconds(value: &str) -> Result<f64, String> {
    let value = value
        .parse::<f64>()
        .map_err(|error| format!("invalid timeout in seconds: {error}"))?;
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err("timeout must be a finite, non-negative number of seconds".to_owned())
    }
}

fn parse_interval_seconds(value: &str) -> Result<f64, String> {
    let value = value
        .parse::<f64>()
        .map_err(|error| format!("invalid interval in seconds: {error}"))?;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err("interval must be a finite, positive number of seconds".to_owned())
    }
}

impl Default for CommonOptions {
    fn default() -> Self {
        Self {
            gateway_host: "127.0.0.1".to_owned(),
            gateway_port: 3000,
            database_url: DEFAULT_DATABASE_URL.to_owned(),
            max_request_body_size_bytes: None,
            api_key: None,
            skip_llm_ready_check: false,
            llm_ready_timeout_s: 600.0,
            llm_ready_interval_s: 2.0,
            quiet: false,
            yolo: false,
            no_color: false,
        }
    }
}

impl HarnessCommand {
    #[must_use]
    pub fn harness(&self) -> Harness {
        match self {
            Self::Codex(_) => Harness::Codex,
            Self::Claude(_) => Harness::Claude,
        }
    }

    #[must_use]
    pub fn options(&self) -> &HarnessOptions {
        match self {
            Self::Codex(options) | Self::Claude(options) => options,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use clap::{CommandFactory, Parser};

    use super::{AttachedHarnessCommand, Cli, Command, DEFAULT_DATABASE_URL, HarnessCommand};

    #[test]
    fn run_codex_uses_sqlite_by_default_and_preserves_arguments() {
        let cli = Cli::try_parse_from([
            "agentic",
            "run",
            "codex",
            "--model",
            "Qwen/test",
            "--",
            "exec",
            "inspect this repo",
        ])
        .expect("valid CLI");

        let Command::Run { harness } = cli.command else {
            panic!("expected run command");
        };
        assert!(matches!(harness, HarnessCommand::Codex(_)));
        let options = harness.options();
        assert_eq!(options.source.model.as_deref(), Some("Qwen/test"));
        assert_eq!(options.common.database_url, DEFAULT_DATABASE_URL);
        assert_eq!(options.harness_args, ["exec", "inspect this repo"]);
    }

    #[test]
    fn run_claude_accepts_an_explicit_postgres_database() {
        let cli = Cli::try_parse_from([
            "agentic",
            "run",
            "claude",
            "--upstream",
            "http://127.0.0.1:8000",
            "--database-url",
            "postgresql://user:secret@localhost/agentic",
        ])
        .expect("valid CLI");

        let Command::Run { harness } = cli.command else {
            panic!("expected run command");
        };
        assert!(matches!(harness, HarnessCommand::Claude(_)));
        let options = harness.options();
        assert_eq!(options.source.upstream.as_deref(), Some("http://127.0.0.1:8000"));
        assert_eq!(
            options.common.database_url,
            "postgresql://user:secret@localhost/agentic"
        );
    }

    #[test]
    fn run_accepts_upstream_with_an_explicit_model_name() {
        let result = Cli::try_parse_from([
            "agentic",
            "run",
            "codex",
            "--model",
            "Qwen/test",
            "--upstream",
            "http://127.0.0.1:8000",
        ]);

        let cli = result.expect("valid CLI");
        let Command::Run { harness } = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(harness.options().source.model.as_deref(), Some("Qwen/test"));
    }

    #[test]
    fn run_rejects_malformed_upstream_urls() {
        for upstream in [
            "http//127.0.0.1:8000",
            "127.0.0.1:8000",
            "ftp://127.0.0.1:8000",
            "http://",
        ] {
            let error = Cli::try_parse_from(["agentic", "run", "claude", "--upstream", upstream])
                .expect_err("malformed upstream URL should be rejected");
            assert!(
                error.to_string().contains("invalid upstream URL"),
                "unexpected error for {upstream}: {error}"
            );
        }
    }

    #[test]
    fn run_normalizes_trailing_slash_on_upstream() {
        let cli = Cli::try_parse_from(["agentic", "run", "claude", "--upstream", "http://127.0.0.1:8000/"])
            .expect("valid CLI");
        let Command::Run { harness } = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(
            harness.options().source.upstream.as_deref(),
            Some("http://127.0.0.1:8000")
        );
    }

    #[test]
    fn run_accepts_yolo_mode() {
        let cli =
            Cli::try_parse_from(["agentic", "run", "claude", "--model", "Qwen/test", "--yolo"]).expect("valid CLI");

        let Command::Run { harness } = cli.command else {
            panic!("expected run command");
        };
        assert!(harness.options().common.yolo);
    }

    #[test]
    fn harness_claude_accepts_gateway_and_namespaced_model() {
        let cli = Cli::try_parse_from([
            "agentic",
            "harness",
            "claude",
            "--gateway-url",
            "http://127.0.0.1:9000/",
            "--model",
            "Qwen/Qwen3-8B",
            "--",
            "--resume",
        ])
        .expect("valid attached Claude CLI");

        let Command::Harness { harness } = cli.command else {
            panic!("expected harness command");
        };
        let AttachedHarnessCommand::Claude(options) = harness else {
            panic!("expected Claude harness");
        };
        assert_eq!(options.gateway_url, "http://127.0.0.1:9000");
        assert_eq!(options.model, "Qwen/Qwen3-8B");
        assert_eq!(options.harness_args, ["--resume"]);
    }

    #[test]
    fn harness_claude_rejects_malformed_gateway_urls() {
        let error = Cli::try_parse_from([
            "agentic",
            "harness",
            "claude",
            "--gateway-url",
            "127.0.0.1:9000",
            "--model",
            "Qwen/Qwen3-8B",
        ])
        .expect_err("malformed gateway URL should be rejected");

        assert!(error.to_string().contains("invalid upstream URL"));
    }

    #[test]
    fn harness_codex_accepts_gateway_model_and_passthrough() {
        let cli = Cli::try_parse_from([
            "agentic",
            "harness",
            "codex",
            "--gateway-url",
            "http://127.0.0.1:9000/",
            "--model",
            "Qwen/Qwen3-8B",
            "--",
            "exec",
            "say hello",
        ])
        .expect("valid attached Codex CLI");

        let Command::Harness { harness } = cli.command else {
            panic!("expected harness command");
        };
        let AttachedHarnessCommand::Codex(options) = harness else {
            panic!("expected Codex harness");
        };
        assert_eq!(options.gateway_url, "http://127.0.0.1:9000");
        assert_eq!(options.model, "Qwen/Qwen3-8B");
        assert_eq!(options.harness_args, ["exec", "say hello"]);
    }

    #[test]
    fn attached_api_key_uses_gateway_specific_environment_variable() {
        let command = Cli::command();
        let harness = command
            .get_subcommands()
            .find(|command| command.get_name() == "harness")
            .expect("harness subcommand");
        let claude = harness
            .get_subcommands()
            .find(|command| command.get_name() == "claude")
            .expect("Claude subcommand");
        let api_key = claude
            .get_arguments()
            .find(|argument| argument.get_id() == "api_key")
            .expect("attached API key argument");

        assert_eq!(api_key.get_env(), Some(OsStr::new("AGENTIC_GATEWAY_API_KEY")));
    }
}
