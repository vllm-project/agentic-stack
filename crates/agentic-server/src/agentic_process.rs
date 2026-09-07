use std::{ffi::OsString, path::Path, time::Duration};

use agentic_core::error::Error;
use reqwest::Client;
use serde::Deserialize;
use tokio::time::{Instant, sleep};

use crate::{
    agentic_cli::{CommonOptions, SourceOptions},
    agentic_output::redact_url,
};

/// Reasoning effort passed to Claude Code unless `AGENTIC_CLAUDE_EFFORT` overrides it.
///
/// Qwen chat templates served by vLLM accept `low`, `medium`, and `xhigh`; Claude Code's
/// default of `high` is rejected by the template, so the CLI always pins a compatible value.
pub const DEFAULT_CLAUDE_EFFORT: &str = "medium";
const CLAUDE_EFFORT_ENV: &str = "AGENTIC_CLAUDE_EFFORT";
const PLACEHOLDER_MODEL: &str = "agentic-api";
const CLAUDE_TOOLS: &str = "Bash,Edit,Read,WebSearch";

#[must_use]
pub fn server_args(source: &SourceOptions, common: &CommonOptions) -> Vec<OsString> {
    let mut args = Vec::new();
    if let Some(upstream) = &source.upstream {
        args.extend([OsString::from("--llm-api-base"), OsString::from(upstream)]);
    } else if let Some(model) = &source.model {
        args.extend([OsString::from("serve"), OsString::from(model)]);
        args.extend([OsString::from("--port"), OsString::from(source.llm_port.to_string())]);
    }
    args.extend([
        OsString::from("--gateway-host"),
        OsString::from(&common.gateway_host),
        OsString::from("--gateway-port"),
        OsString::from(common.gateway_port.to_string()),
        OsString::from("--db-url"),
        OsString::from(&common.database_url),
        OsString::from("--llm-ready-timeout-s"),
        OsString::from(common.llm_ready_timeout_s.to_string()),
        OsString::from("--llm-ready-interval-s"),
        OsString::from(common.llm_ready_interval_s.to_string()),
    ]);
    if let Some(api_key) = &common.api_key {
        args.extend([OsString::from("--openai-api-key"), OsString::from(api_key)]);
    }
    if common.skip_llm_ready_check {
        args.push(OsString::from("--skip-llm-ready-check"));
    }
    args
}

#[must_use]
pub fn server_binary_path(current_exe: &Path) -> std::path::PathBuf {
    current_exe.with_file_name("agentic-server")
}

#[must_use]
pub fn claude_effort() -> String {
    std::env::var(CLAUDE_EFFORT_ENV)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_CLAUDE_EFFORT.to_owned())
}

fn harness_launch_args(
    harness: crate::agentic_cli::Harness,
    yolo: bool,
    claude_effort: &str,
    managed: &[String],
    passthrough: &[String],
) -> Vec<String> {
    let mut args = Vec::with_capacity(managed.len() + passthrough.len() + 3);
    match harness {
        crate::agentic_cli::Harness::Codex => {
            if yolo {
                args.push("--dangerously-bypass-approvals-and-sandbox".to_owned());
            }
        }
        crate::agentic_cli::Harness::Claude => {
            if yolo {
                args.push("--dangerously-skip-permissions".to_owned());
            }
            args.extend([
                "--model".to_owned(),
                crate::agentic_harness::CLAUDE_CANONICAL_MODEL.to_owned(),
                "--tools".to_owned(),
                CLAUDE_TOOLS.to_owned(),
                "--setting-sources".to_owned(),
                "user".to_owned(),
            ]);
            args.extend(["--effort".to_owned(), claude_effort.to_owned()]);
        }
    }
    args.extend_from_slice(managed);
    args.extend_from_slice(passthrough);
    args
}

fn validate_claude_passthrough(passthrough: &[String]) -> Result<(), Error> {
    const MANAGED_FLAGS: [&str; 4] = ["--model", "--settings", "--setting-sources", "--bare"];
    if let Some(argument) = passthrough.iter().find(|argument| {
        MANAGED_FLAGS
            .iter()
            .any(|flag| argument.as_str() == *flag || argument.starts_with(&format!("{flag}=")))
    }) {
        return Err(Error::Config(format!(
            "Claude Code argument {argument} cannot be forwarded because Agentic API manages model and setting isolation"
        )));
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct ModelList {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
}

/// Resolve the harness model: the explicit `--model`, or the first model the upstream serves.
///
/// # Errors
///
/// Returns a configuration error when no model is given and the upstream lists none.
pub async fn resolve_model(client: &Client, source: &SourceOptions, api_key: Option<&str>) -> Result<String, Error> {
    if let Some(model) = &source.model {
        return Ok(model.clone());
    }
    let Some(upstream) = &source.upstream else {
        return Ok(PLACEHOLDER_MODEL.to_owned());
    };
    let models_url = format!("{}/v1/models", agentic_core::config::normalize_base_url(upstream));
    let display_models_url = redact_url(&models_url);
    let display_upstream = redact_url(upstream);
    let mut request = client.get(&models_url);
    if let Some(api_key) = api_key {
        request = request.bearer_auth(api_key);
    }
    let response = request
        .send()
        .await
        .map_err(|error| {
            Error::Config(format!(
                "failed to list upstream models at {display_models_url}: {}",
                error.without_url()
            ))
        })?
        .error_for_status()
        .map_err(|error| {
            Error::Config(format!(
                "upstream model listing at {display_models_url} failed: {}",
                error.without_url()
            ))
        })?;
    let body = response.text().await.map_err(|error| {
        Error::Config(format!(
            "failed to read model listing from {display_models_url}: {}",
            error.without_url()
        ))
    })?;
    let list: ModelList = agentic_core::utils::common::deserialize_from_str(&body)
        .map_err(|error| Error::Config(format!("invalid model listing from {display_models_url}: {error}")))?;
    let mut ids = list.data.into_iter().map(|entry| entry.id);
    let Some(model) = ids.next() else {
        return Err(Error::Config(format!(
            "upstream {display_upstream} serves no models; pass --model explicitly"
        )));
    };
    let remaining = ids.count();
    if remaining > 0 {
        eprintln!(
            "upstream serves {} models; using {model}. Pass --model to choose another.",
            remaining + 1
        );
    }
    Ok(model)
}

/// Wait until the gateway is live and, unless skipped, its upstream is ready.
///
/// # Errors
///
/// Returns a configuration error when the timeout expires.
pub async fn wait_for_gateway(
    client: &Client,
    gateway_url: &str,
    timeout: Duration,
    interval: Duration,
    skip_llm_ready_check: bool,
) -> Result<(), Error> {
    let deadline = Instant::now() + timeout;
    let health_url = format!("{}/health", gateway_url.trim_end_matches('/'));
    let ready_url = format!("{}/ready", gateway_url.trim_end_matches('/'));
    let ready = tokio::time::timeout_at(deadline, async {
        while Instant::now() < deadline {
            let health_ok = client
                .get(&health_url)
                .send()
                .await
                .is_ok_and(|response| response.status().is_success());
            let ready_ok = skip_llm_ready_check
                || client
                    .get(&ready_url)
                    .send()
                    .await
                    .is_ok_and(|response| response.status().is_success());
            if health_ok && ready_ok {
                // A completed future can win timeout polling after the deadline.
                return Instant::now() < deadline;
            }
            sleep(interval).await;
        }
        false
    })
    .await
    .unwrap_or(false);

    if ready {
        Ok(())
    } else {
        Err(Error::Config(format!(
            "gateway did not become ready at {}",
            redact_url(gateway_url)
        )))
    }
}

/// Run one gateway-plus-harness session and return the harness exit status.
///
/// # Errors
///
/// Returns an error when a child cannot start or readiness fails.
pub async fn run_session(
    current_exe: &Path,
    harness: crate::agentic_cli::Harness,
    options: crate::agentic_cli::HarnessOptions,
) -> Result<std::process::ExitStatus, Error> {
    if matches!(harness, crate::agentic_cli::Harness::Claude) {
        validate_claude_passthrough(&options.harness_args)?;
    }
    let gateway_url = format!("http://{}:{}", options.common.gateway_host, options.common.gateway_port);
    let session_root = create_session_root("agentic-api-session")?;
    let claude_state_root = harness_state_root(harness, session_root.path())?;

    let mut server = start_server(current_exe, &options)?;

    let client = match gateway_client() {
        Ok(client) => client,
        Err(error) => {
            cleanup(&mut server, session_root.path()).await;
            return Err(error);
        }
    };
    if let Err(error) = wait_for_gateway(
        &client,
        &gateway_url,
        Duration::from_secs_f64(options.common.llm_ready_timeout_s),
        Duration::from_secs_f64(options.common.llm_ready_interval_s),
        options.common.skip_llm_ready_check,
    )
    .await
    {
        cleanup(&mut server, session_root.path()).await;
        return Err(error);
    }

    let model = match resolve_model(&client, &options.source, options.common.api_key.as_deref()).await {
        Ok(model) => model,
        Err(error) => {
            cleanup(&mut server, session_root.path()).await;
            return Err(error);
        }
    };
    let harness_env = match harness_environment(
        harness,
        &gateway_url,
        &model,
        &options,
        session_root.path(),
        &claude_state_root,
    ) {
        Ok(environment) => environment,
        Err(error) => {
            cleanup(&mut server, session_root.path()).await;
            return Err(error);
        }
    };
    if !options.common.quiet {
        println!("{}", harness_env.summary);
    }

    let mut harness_child = match spawn_harness(harness, options.common.yolo, &options.harness_args, &harness_env) {
        Ok(child) => child,
        Err(error) => {
            cleanup(&mut server, session_root.path()).await;
            return Err(error);
        }
    };

    let harness_status = tokio::select! {
        status = harness_child.wait() => status?,
        signal = tokio::signal::ctrl_c() => {
            signal?;
            let _ = harness_child.kill().await;
            harness_child.wait().await?
        }
    };
    cleanup(&mut server, session_root.path()).await;
    Ok(harness_status)
}

fn start_server(
    current_exe: &Path,
    options: &crate::agentic_cli::HarnessOptions,
) -> Result<tokio::process::Child, Error> {
    let server_path = server_binary_path(current_exe);
    if !server_path.is_file() {
        return Err(Error::Config(format!(
            "agentic-server binary not found beside {}; run cargo build -p agentic-server --bins first",
            current_exe.display()
        )));
    }
    let mut server = tokio::process::Command::new(server_path);
    server.kill_on_drop(true);
    server.args(server_args(&options.source, &options.common));
    server.stdout(std::process::Stdio::inherit());
    server.stderr(std::process::Stdio::inherit());
    Ok(server.spawn()?)
}

fn harness_environment(
    harness: crate::agentic_cli::Harness,
    gateway_url: &str,
    model: &str,
    options: &crate::agentic_cli::HarnessOptions,
    session_root: &Path,
    claude_state_root: &Path,
) -> Result<crate::agentic_harness::HarnessEnv, Error> {
    let inherited_auth_token = std::env::var("ANTHROPIC_AUTH_TOKEN")
        .ok()
        .filter(|value| !value.trim().is_empty());
    let mut environment = match harness {
        crate::agentic_cli::Harness::Codex => crate::agentic_harness::prepare_codex_home(
            session_root,
            gateway_url,
            model,
            options.common.api_key.as_deref(),
        )
        .map_err(Error::from),
        crate::agentic_cli::Harness::Claude => crate::agentic_harness::prepare_claude_home_with_state(
            session_root,
            claude_state_root,
            gateway_url,
            model,
            inherited_auth_token.as_deref(),
            options.common.api_key.as_deref(),
        )
        .map_err(Error::from),
    }?;
    if matches!(harness, crate::agentic_cli::Harness::Claude) {
        // Claude Code gives CLAUDE_CODE_EFFORT_LEVEL precedence over --effort, so set both
        // to keep an inherited `high` from reaching the Qwen chat template.
        environment
            .environment
            .insert("CLAUDE_CODE_EFFORT_LEVEL".to_owned(), claude_effort());
    }
    Ok(environment)
}

fn spawn_harness(
    harness: crate::agentic_cli::Harness,
    yolo: bool,
    passthrough: &[String],
    harness_env: &crate::agentic_harness::HarnessEnv,
) -> Result<tokio::process::Child, Error> {
    let binary_name = match harness {
        crate::agentic_cli::Harness::Codex => "codex",
        crate::agentic_cli::Harness::Claude => "claude",
    };
    let override_name = match harness {
        crate::agentic_cli::Harness::Codex => "AGENTIC_CODEX_BIN",
        crate::agentic_cli::Harness::Claude => "AGENTIC_CLAUDE_BIN",
    };
    let binary = std::env::var_os(override_name).unwrap_or_else(|| binary_name.into());
    let mut harness_command = build_harness_command(&binary, harness, yolo, passthrough, harness_env);
    harness_command
        .spawn()
        .map_err(|error| Error::Config(format!("failed to launch {binary_name} ({override_name}): {error}")))
}

fn build_harness_command(
    binary: &std::ffi::OsStr,
    harness: crate::agentic_cli::Harness,
    yolo: bool,
    passthrough: &[String],
    harness_env: &crate::agentic_harness::HarnessEnv,
) -> tokio::process::Command {
    let mut harness_command = tokio::process::Command::new(binary);
    harness_command.kill_on_drop(true);
    harness_command.args(harness_launch_args(
        harness,
        yolo,
        &claude_effort(),
        &harness_env.arguments,
        passthrough,
    ));
    for name in &harness_env.environment_remove {
        harness_command.env_remove(name);
    }
    harness_command.envs(&harness_env.environment);
    harness_command.stdin(std::process::Stdio::inherit());
    harness_command.stdout(std::process::Stdio::inherit());
    harness_command.stderr(std::process::Stdio::inherit());
    harness_command
}

fn prepare_attached_harness_environment(
    harness: crate::agentic_cli::Harness,
    session_root: &Path,
    claude_state_root: &Path,
    gateway_url: &str,
    model: &str,
    api_key: Option<&str>,
) -> Result<crate::agentic_harness::HarnessEnv, Error> {
    match harness {
        crate::agentic_cli::Harness::Codex => {
            crate::agentic_harness::prepare_codex_home(session_root, gateway_url, model, api_key).map_err(Error::from)
        }
        crate::agentic_cli::Harness::Claude => crate::agentic_harness::prepare_claude_home_with_state(
            session_root,
            claude_state_root,
            gateway_url,
            model,
            None,
            api_key,
        )
        .map_err(Error::from),
    }
}

/// Launch a coding harness against an already-running Agentic API gateway.
///
/// # Errors
///
/// Returns an error when the gateway is not ready, configuration cannot be written, or the harness cannot start.
pub async fn run_attached_harness(
    harness: crate::agentic_cli::Harness,
    options: crate::agentic_cli::AttachedHarnessOptions,
) -> Result<std::process::ExitStatus, Error> {
    if matches!(harness, crate::agentic_cli::Harness::Claude) {
        validate_claude_passthrough(&options.harness_args)?;
    }
    let session_root = create_session_root("agentic-api-harness")?;
    let claude_state_root = harness_state_root(harness, session_root.path())?;

    let result = async {
        let client = gateway_client()?;
        wait_for_gateway(
            &client,
            &options.gateway_url,
            Duration::from_secs(30),
            Duration::from_millis(250),
            false,
        )
        .await?;
        let mut harness_env = prepare_attached_harness_environment(
            harness,
            session_root.path(),
            &claude_state_root,
            &options.gateway_url,
            &options.model,
            options.api_key.as_deref(),
        )?;
        if matches!(harness, crate::agentic_cli::Harness::Claude) {
            harness_env
                .environment
                .insert("CLAUDE_CODE_EFFORT_LEVEL".to_owned(), claude_effort());
        }
        if !options.quiet {
            println!("{}", harness_env.summary);
        }
        let mut child = spawn_harness(harness, options.yolo, &options.harness_args, &harness_env)?;
        let status = tokio::select! {
            status = child.wait() => status?,
            signal = tokio::signal::ctrl_c() => {
                signal?;
                let _ = child.kill().await;
                child.wait().await?
            }
        };
        Ok(status)
    }
    .await;
    let _ = tokio::fs::remove_dir_all(session_root.path()).await;
    result
}

async fn cleanup(server: &mut tokio::process::Child, session_root: &Path) {
    let _ = server.kill().await;
    let _ = server.wait().await;
    let _ = tokio::fs::remove_dir_all(session_root).await;
}

struct SessionRoot {
    directory: tempfile::TempDir,
}

impl SessionRoot {
    fn path(&self) -> &Path {
        self.directory.path()
    }
}

fn create_session_root(prefix: &str) -> Result<SessionRoot, Error> {
    let prefix = format!("{prefix}-");
    let mut builder = tempfile::Builder::new();
    builder.prefix(&prefix);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(std::fs::Permissions::from_mode(0o700));
    }
    Ok(SessionRoot {
        directory: builder.tempdir()?,
    })
}

fn harness_state_root(
    harness: crate::agentic_cli::Harness,
    temporary_root: &Path,
) -> Result<std::path::PathBuf, Error> {
    match harness {
        crate::agentic_cli::Harness::Claude => Ok(agentic_core::config::agentic_api_home()?
            .join("harnesses")
            .join("claude")),
        crate::agentic_cli::Harness::Codex => Ok(temporary_root.to_owned()),
    }
}

fn gateway_client() -> Result<Client, Error> {
    Client::builder()
        .timeout(Duration::from_secs(2))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(Error::HttpClient)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::{DEFAULT_CLAUDE_EFFORT, harness_launch_args, server_args};
    use crate::agentic_cli::{CommonOptions, Harness, SourceOptions};

    #[test]
    fn integrated_mode_builds_server_arguments() {
        let args = server_args(
            &SourceOptions {
                upstream: None,
                model: Some("Qwen/test".to_owned()),
                llm_port: 8000,
            },
            &CommonOptions::default(),
        );
        let args: Vec<_> = args.iter().map(OsString::as_os_str).collect();

        assert_eq!(args[0], "serve");
        assert_eq!(args[1], "Qwen/test");
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--db-url", "sqlite://./agentic_api.db"])
        );
    }

    #[test]
    fn standalone_mode_builds_upstream_arguments() {
        let args = server_args(
            &SourceOptions {
                upstream: Some("http://127.0.0.1:8000".to_owned()),
                model: None,
                llm_port: 8000,
            },
            &CommonOptions::default(),
        );
        let args: Vec<_> = args.iter().map(OsString::as_os_str).collect();

        assert_eq!(args[0], "--llm-api-base");
        assert_eq!(args[1], "http://127.0.0.1:8000");
    }

    #[test]
    fn explicit_upstream_wins_when_model_names_the_harness_model() {
        let args = server_args(
            &SourceOptions {
                upstream: Some("http://127.0.0.1:8000".to_owned()),
                model: Some("Qwen/test".to_owned()),
                llm_port: 8000,
            },
            &CommonOptions::default(),
        );
        let args: Vec<_> = args.iter().map(OsString::as_os_str).collect();

        assert_eq!(args[0], "--llm-api-base");
        assert!(!args.iter().any(|arg| *arg == "serve"));
    }

    #[test]
    fn yolo_mode_uses_native_codex_bypass_flag() {
        assert_eq!(
            harness_launch_args(Harness::Codex, true, DEFAULT_CLAUDE_EFFORT, &[], &["exec".to_owned()]),
            ["--dangerously-bypass-approvals-and-sandbox", "exec"]
        );
    }

    #[test]
    fn yolo_mode_uses_native_claude_bypass_and_compatible_effort() {
        assert_eq!(
            harness_launch_args(Harness::Claude, true, DEFAULT_CLAUDE_EFFORT, &[], &[]),
            [
                "--dangerously-skip-permissions",
                "--model",
                "claude-sonnet-4-5-20250929",
                "--tools",
                "Bash,Edit,Read,WebSearch",
                "--setting-sources",
                "user",
                "--effort",
                "medium"
            ]
        );
    }

    #[test]
    fn claude_always_receives_a_compatible_effort() {
        assert_eq!(
            harness_launch_args(Harness::Claude, false, "low", &[], &["-p".to_owned(), "hi".to_owned()]),
            [
                "--model",
                "claude-sonnet-4-5-20250929",
                "--tools",
                "Bash,Edit,Read,WebSearch",
                "--setting-sources",
                "user",
                "--effort",
                "low",
                "-p",
                "hi"
            ]
        );
        assert_eq!(
            harness_launch_args(Harness::Codex, false, DEFAULT_CLAUDE_EFFORT, &[], &[]),
            Vec::<String>::new()
        );
    }

    #[test]
    fn claude_environment_pins_effort_without_yolo() {
        let options = crate::agentic_cli::HarnessOptions {
            source: SourceOptions {
                upstream: Some("http://127.0.0.1:8000".to_owned()),
                model: None,
                llm_port: 8000,
            },
            common: CommonOptions::default(),
            harness_args: Vec::new(),
        };
        let root = std::env::temp_dir().join(format!("agentic-api-effort-test-{}", std::process::id()));
        let state_root = std::env::temp_dir().join(format!("agentic-api-state-test-{}", std::process::id()));
        let environment = super::harness_environment(
            Harness::Claude,
            "http://127.0.0.1:3000",
            "served-discovered",
            &options,
            &root,
            &state_root,
        )
        .expect("Claude environment");

        assert_eq!(
            environment.environment.get("CLAUDE_CODE_EFFORT_LEVEL"),
            Some(&DEFAULT_CLAUDE_EFFORT.to_owned())
        );
        assert_eq!(
            environment.environment.get("CLAUDE_CONFIG_DIR"),
            Some(&state_root.display().to_string())
        );
        assert!(root.join("settings.json").is_file());
        std::fs::remove_dir_all(root).expect("cleanup");
        std::fs::remove_dir_all(state_root).expect("state cleanup");
    }

    #[test]
    fn integrated_claude_preserves_inherited_auth_token() {
        const CHILD_MARKER: &str = "AGENTIC_TEST_INTEGRATED_CLAUDE_AUTH";
        if std::env::var_os(CHILD_MARKER).is_some() {
            let settings_root = tempfile::tempdir().expect("settings root");
            let state_root = tempfile::tempdir().expect("state root");
            let options = crate::agentic_cli::HarnessOptions {
                source: SourceOptions {
                    upstream: Some("http://127.0.0.1:8000".to_owned()),
                    model: None,
                    llm_port: 8000,
                },
                common: CommonOptions::default(),
                harness_args: Vec::new(),
            };
            let environment = super::harness_environment(
                Harness::Claude,
                "http://127.0.0.1:3000",
                "served-discovered",
                &options,
                settings_root.path(),
                state_root.path(),
            )
            .expect("Claude environment");

            assert_eq!(
                environment.environment.get("ANTHROPIC_AUTH_TOKEN"),
                Some(&"oidc-token".to_owned())
            );
            return;
        }

        let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--exact",
                "agentic_process::tests::integrated_claude_preserves_inherited_auth_token",
            ])
            .env(CHILD_MARKER, "1")
            .env("ANTHROPIC_AUTH_TOKEN", "oidc-token")
            .status()
            .expect("run isolated integrated auth test");

        assert!(status.success(), "isolated integrated auth test failed with {status}");
    }

    #[test]
    fn claude_rejects_passthrough_that_can_replace_isolated_configuration() {
        for argument in [
            "--model",
            "--model=opus",
            "--settings",
            "--settings={}",
            "--setting-sources",
            "--setting-sources=project",
            "--bare",
        ] {
            let error = super::validate_claude_passthrough(&[argument.to_owned()])
                .expect_err("configuration-owning argument should be rejected");
            assert!(error.to_string().contains(argument));
        }
        super::validate_claude_passthrough(&["-p".to_owned(), "hello".to_owned()])
            .expect("normal passthrough arguments");
    }

    #[test]
    fn attached_codex_uses_an_isolated_responses_provider() {
        let root = std::env::temp_dir().join(format!("agentic-api-attached-codex-test-{}", std::process::id()));
        let environment = super::prepare_attached_harness_environment(
            Harness::Codex,
            &root,
            &root,
            "http://127.0.0.1:9000",
            "Qwen/Qwen3-8B",
            None,
        )
        .expect("Codex environment");
        let config = std::fs::read_to_string(root.join("config.toml")).expect("Codex config");

        assert_eq!(
            environment.environment.get("CODEX_HOME"),
            Some(&root.display().to_string())
        );
        assert!(!environment.environment.contains_key("CLAUDE_CONFIG_DIR"));
        assert!(config.contains("model = \"Qwen/Qwen3-8B\""));
        assert!(config.contains("base_url = \"http://127.0.0.1:9000/v1\""));
        assert!(config.contains("wire_api = \"responses\""));

        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn attached_claude_uses_persistent_state_root() {
        let session_root = tempfile::tempdir().expect("session root");
        let state_root = tempfile::tempdir().expect("state root");
        let environment = super::prepare_attached_harness_environment(
            Harness::Claude,
            session_root.path(),
            state_root.path(),
            "http://127.0.0.1:9000",
            "Qwen/Qwen3-8B",
            None,
        )
        .expect("Claude environment");

        assert_eq!(
            environment.environment.get("CLAUDE_CONFIG_DIR"),
            Some(&state_root.path().display().to_string())
        );
    }

    #[test]
    fn attached_claude_ignores_inherited_anthropic_auth_token() {
        const CHILD_MARKER: &str = "AGENTIC_TEST_ATTACHED_CLAUDE_AUTH";
        if std::env::var_os(CHILD_MARKER).is_some() {
            let session_root = tempfile::tempdir().expect("session root");
            let state_root = tempfile::tempdir().expect("state root");
            let environment = super::prepare_attached_harness_environment(
                Harness::Claude,
                session_root.path(),
                state_root.path(),
                "http://127.0.0.1:9000",
                "Qwen/Qwen3-8B",
                None,
            )
            .expect("Claude environment");

            assert_eq!(
                environment.environment.get("ANTHROPIC_AUTH_TOKEN"),
                Some(&"agentic-api-local".to_owned())
            );
            return;
        }

        let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--exact",
                "agentic_process::tests::attached_claude_ignores_inherited_anthropic_auth_token",
            ])
            .env(CHILD_MARKER, "1")
            .env("ANTHROPIC_AUTH_TOKEN", "must-not-be-forwarded")
            .status()
            .expect("run isolated attached auth test");

        assert!(status.success(), "isolated attached auth test failed with {status}");
    }

    #[cfg(unix)]
    #[test]
    fn session_root_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let root = super::create_session_root("agentic-api-private-test").expect("private session root");
        let mode = std::fs::metadata(root.path())
            .expect("session root metadata")
            .permissions()
            .mode()
            & 0o777;

        assert_eq!(mode, 0o700);

        std::fs::remove_dir_all(root.path()).expect("cleanup");
    }

    #[test]
    fn session_root_is_removed_when_its_guard_drops() {
        let path = {
            let root = super::create_session_root("agentic-api-drop-test").expect("session root");
            root.path().to_owned()
        };

        let removed = !path.exists();
        if !removed {
            std::fs::remove_dir_all(&path).expect("cleanup failed session root");
        }
        assert!(removed, "session root survived its guard");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dropping_harness_child_prevents_later_side_effects() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let marker = temporary.path().join("child-survived");
        let environment = crate::agentic_harness::HarnessEnv {
            environment: std::collections::BTreeMap::new(),
            environment_remove: Vec::new(),
            arguments: Vec::new(),
            files: Vec::new(),
            summary: String::new(),
        };
        let passthrough = vec![
            "-c".to_owned(),
            "sleep 0.15; printf survived > \"$1\"".to_owned(),
            "agentic-test".to_owned(),
            marker.display().to_string(),
        ];
        let mut command = super::build_harness_command(
            std::ffi::OsStr::new("/bin/sh"),
            Harness::Codex,
            false,
            &passthrough,
            &environment,
        );
        let child = command.spawn().expect("test child");

        drop(child);
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        assert!(!marker.exists(), "dropped harness child continued running");
    }

    #[tokio::test]
    async fn resolve_model_does_not_double_the_v1_suffix() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0_u8; 1024];
            let read = socket.read(&mut buffer).await.unwrap();
            let request_line = String::from_utf8_lossy(&buffer[..read])
                .lines()
                .next()
                .unwrap_or_default()
                .to_owned();
            let body = r#"{"data":[{"id":"Qwen/served"}]}"#;
            let status = if request_line.starts_with("GET /v1/models ") {
                "200 OK"
            } else {
                "404 Not Found"
            };
            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        let client = reqwest::Client::new();
        let source = SourceOptions {
            upstream: Some(format!("http://{address}/v1")),
            model: None,
            llm_port: 8000,
        };
        assert_eq!(
            super::resolve_model(&client, &source, None).await.unwrap(),
            "Qwen/served"
        );
    }

    #[tokio::test]
    async fn resolve_model_prefers_explicit_model() {
        let client = reqwest::Client::new();
        let source = SourceOptions {
            upstream: Some("http://127.0.0.1:9".to_owned()),
            model: Some("Qwen/test".to_owned()),
            llm_port: 8000,
        };
        assert_eq!(super::resolve_model(&client, &source, None).await.unwrap(), "Qwen/test");
    }

    #[tokio::test]
    async fn readiness_timeout_redacts_gateway_password() {
        let error = super::wait_for_gateway(
            &reqwest::Client::new(),
            "https://agentic:gateway-secret@example.com",
            std::time::Duration::ZERO,
            std::time::Duration::from_millis(1),
            false,
        )
        .await
        .expect_err("readiness should time out");
        let message = error.to_string();

        assert!(!message.contains("gateway-secret"));
        assert!(message.contains("https://[REDACTED]@example.com"));
    }

    #[tokio::test(start_paused = true)]
    async fn readiness_timeout_bounds_retry_sleep() {
        let timeout = std::time::Duration::from_secs(1);
        let started = tokio::time::Instant::now();
        // Make probe failure immediate so this test measures only the retry timer.
        let result = super::wait_for_gateway(
            &super::gateway_client().expect("gateway client"),
            "http://[invalid",
            timeout,
            std::time::Duration::from_secs(60),
            false,
        )
        .await;

        assert!(result.is_err());
        assert_eq!(started.elapsed(), timeout, "retry sleep exceeded the readiness budget");
    }

    async fn check_readiness_timeout_interrupts_probe(stalled_path: &'static str, skip_llm_ready_check: bool) {
        use axum::{Router, http::StatusCode, routing::get};

        let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut application = Router::new();
        for path in ["/health", "/ready"] {
            let started_tx = started_tx.clone();
            application = application.route(
                path,
                get(move || async move {
                    if path == stalled_path {
                        started_tx.send(()).expect("probe observer");
                        std::future::pending::<()>().await;
                    }
                    StatusCode::OK
                }),
            );
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("listener");
        let gateway_url = format!("http://{}", listener.local_addr().expect("listener address"));
        let server = tokio::spawn(async move { axum::serve(listener, application).await.expect("test server") });
        let timeout = std::time::Duration::from_secs(60);
        let client = reqwest::Client::builder()
            .timeout(timeout * 2)
            .build()
            .expect("gateway client");
        let readiness = super::wait_for_gateway(
            &client,
            &gateway_url,
            timeout,
            std::time::Duration::from_secs(1),
            skip_llm_ready_check,
        );
        tokio::pin!(readiness);
        tokio::select! {
            started = started_rx.recv() => started.expect("stalled probe must reach the HTTP server"),
            result = &mut readiness => panic!("readiness finished before the stalled probe: {result:?}"),
        }

        // Start virtual time only after real HTTP I/O reaches the blocked endpoint.
        tokio::time::pause();
        tokio::time::advance(timeout).await;
        // Allow the runtime to process the expired timer without reaching the request timeout.
        let result = tokio::time::timeout(std::time::Duration::from_secs(1), readiness).await;
        server.abort();

        assert!(
            matches!(result, Ok(Err(agentic_core::error::Error::Config(_)))),
            "readiness remained blocked in {stalled_path} after its deadline: {result:?}"
        );
    }

    #[tokio::test]
    async fn readiness_timeout_interrupts_health_probe() {
        check_readiness_timeout_interrupts_probe("/health", false).await;
    }

    #[tokio::test]
    async fn readiness_timeout_interrupts_upstream_probe() {
        check_readiness_timeout_interrupts_probe("/ready", false).await;
    }

    #[tokio::test]
    async fn readiness_timeout_still_bounds_health_when_upstream_probe_is_skipped() {
        check_readiness_timeout_interrupts_probe("/health", true).await;
    }

    #[tokio::test]
    async fn readiness_rejects_success_observed_after_deadline() {
        use axum::{Router, routing::get};

        let (ready_tx, mut ready_rx) = tokio::sync::mpsc::unbounded_channel();
        let application = Router::new().route("/health", get(|| async {})).route(
            "/ready",
            get(move || async move { ready_tx.send(()).expect("probe observer") }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("listener");
        let gateway_url = format!("http://{}", listener.local_addr().expect("listener address"));
        let server = tokio::spawn(async move { axum::serve(listener, application).await.expect("test server") });
        let timeout = std::time::Duration::from_secs(60);
        let client = reqwest::Client::builder()
            .timeout(timeout * 2)
            .build()
            .expect("gateway client");
        let readiness = super::wait_for_gateway(&client, &gateway_url, timeout, timeout, false);
        tokio::pin!(readiness);
        tokio::select! {
            biased;
            ready = ready_rx.recv() => ready.expect("both probes must reach the HTTP server"),
            result = &mut readiness => panic!("readiness finished before the final response: {result:?}"),
        }

        tokio::time::pause();
        tokio::time::advance(timeout).await;
        tokio::time::resume();
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), readiness)
            .await
            .expect("readiness must finish after its deadline");
        server.abort();

        assert!(result.is_err(), "successful probes bypassed the readiness deadline");
    }

    #[tokio::test]
    async fn readiness_accepts_healthy_gateway_and_skips_only_upstream_probe() {
        use axum::{Router, routing::get};

        for skip_llm_ready_check in [false, true] {
            let (probe_tx, mut probe_rx) = tokio::sync::mpsc::unbounded_channel();
            let mut application = Router::new();
            for path in ["/health", "/ready"] {
                let probe_tx = probe_tx.clone();
                application = application.route(
                    path,
                    get(move || async move { probe_tx.send(path).expect("probe observer") }),
                );
            }
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("listener");
            let gateway_url = format!("http://{}", listener.local_addr().expect("listener address"));
            let server = tokio::spawn(async move { axum::serve(listener, application).await.expect("test server") });
            let result = super::wait_for_gateway(
                &super::gateway_client().expect("gateway client"),
                &gateway_url,
                std::time::Duration::from_secs(60),
                std::time::Duration::from_secs(1),
                skip_llm_ready_check,
            )
            .await;
            server.abort();

            assert!(result.is_ok());
            assert_eq!(probe_rx.try_recv(), Ok("/health"));
            if !skip_llm_ready_check {
                assert_eq!(probe_rx.try_recv(), Ok("/ready"));
            }
            assert!(probe_rx.try_recv().is_err(), "unexpected readiness probe");
        }
    }

    #[tokio::test]
    async fn empty_model_listing_error_redacts_upstream_userinfo() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("listener");
        let address = listener.local_addr().expect("listener address");
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};

            let (mut socket, _) = listener.accept().await.expect("connection");
            let mut buffer = [0_u8; 1024];
            let _ = socket.read(&mut buffer).await;
            let body = r#"{"data":[]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.expect("response");
        });
        let source = SourceOptions {
            upstream: Some(format!("http://secret-token@{address}")),
            model: None,
            llm_port: 8000,
        };

        let error = super::resolve_model(&reqwest::Client::new(), &source, None)
            .await
            .expect_err("empty model listing should fail");
        let message = error.to_string();

        assert!(!message.contains("secret-token"));
        assert!(message.contains("http://[REDACTED]@"));
    }

    #[tokio::test]
    async fn readiness_rejects_redirected_login_pages() {
        use axum::{Router, response::Redirect, routing::get};
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("listener");
        let address = listener.local_addr().expect("listener address");
        let login_visits = Arc::new(AtomicUsize::new(0));
        let login_visits_for_route = Arc::clone(&login_visits);
        let application = Router::new()
            .route("/health", get(|| async { Redirect::temporary("/login") }))
            .route("/ready", get(|| async { Redirect::temporary("/login") }))
            .route(
                "/login",
                get(move || {
                    login_visits_for_route.fetch_add(1, Ordering::SeqCst);
                    async { "sign in" }
                }),
            );
        let server = tokio::spawn(async move {
            axum::serve(listener, application).await.expect("test server");
        });

        let result = super::wait_for_gateway(
            &super::gateway_client().expect("gateway client"),
            &format!("http://{address}"),
            std::time::Duration::from_millis(100),
            std::time::Duration::from_millis(1),
            false,
        )
        .await;
        server.abort();

        assert!(result.is_err(), "redirected login page passed readiness");
        assert_eq!(login_visits.load(Ordering::SeqCst), 0, "readiness followed a redirect");
    }

    #[tokio::test]
    async fn resolve_model_discovers_first_upstream_model() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0_u8; 1024];
            let _ = socket.read(&mut buffer).await;
            let body = r#"{"object":"list","data":[{"id":"Qwen/served"},{"id":"other"}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        let client = reqwest::Client::new();
        let source = SourceOptions {
            upstream: Some(format!("http://{address}")),
            model: None,
            llm_port: 8000,
        };
        assert_eq!(
            super::resolve_model(&client, &source, None).await.unwrap(),
            "Qwen/served"
        );
    }

    #[test]
    fn yolo_claude_environment_overrides_inherited_effort() {
        let options = crate::agentic_cli::HarnessOptions {
            source: SourceOptions {
                upstream: Some("http://127.0.0.1:8000".to_owned()),
                model: Some("served-test".to_owned()),
                llm_port: 8000,
            },
            common: CommonOptions {
                yolo: true,
                ..CommonOptions::default()
            },
            harness_args: Vec::new(),
        };
        let root = std::env::temp_dir().join(format!("agentic-api-yolo-test-{}", std::process::id()));
        let environment = super::harness_environment(
            Harness::Claude,
            "http://127.0.0.1:3000",
            "served-test",
            &options,
            &root,
            &root,
        )
        .expect("Claude environment");

        assert_eq!(
            environment.environment.get("CLAUDE_CODE_EFFORT_LEVEL"),
            Some(&"medium".to_owned())
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }
}
