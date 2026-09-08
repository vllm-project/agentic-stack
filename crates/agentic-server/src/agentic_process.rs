use std::{ffi::OsString, path::Path, time::Duration};

use agentic_core::error::Error;
use reqwest::Client;
use serde::Deserialize;
use tokio::time::{Instant, sleep};

use crate::{
    agentic_cli::{CommonOptions, Harness, HarnessOptions, SourceOptions},
    agentic_harness::HarnessEnv,
    agentic_output::redact_url,
    model_capabilities::{CodexCatalogCapabilities, InputModalities},
};

/// Reasoning effort passed to Claude Code unless `AGENTIC_CLAUDE_EFFORT` overrides it.
///
/// Qwen chat templates served by vLLM accept `low`, `medium`, and `xhigh`; Claude Code's
/// default of `high` is rejected by the template, so the CLI always pins a compatible value.
pub const DEFAULT_CLAUDE_EFFORT: &str = "medium";
const CLAUDE_EFFORT_ENV: &str = "AGENTIC_CLAUDE_EFFORT";
const PLACEHOLDER_MODEL: &str = "agentic-api";
const CLAUDE_TOOLS: &str = "Bash,Edit,Read,WebSearch";
/// Operator-provided Codex version, used instead of probing the Codex binary.
const CODEX_CLIENT_VERSION_ENV: &str = "AGENTIC_CODEX_CLIENT_VERSION";
/// Bound on the `codex --version` probe.
const CODEX_VERSION_PROBE_TIMEOUT: Duration = Duration::from_secs(5);
/// Upper bound on the catalog payload the launcher reads from a gateway.
const MAX_CATALOG_BYTES: usize = 1024 * 1024;
/// How long a served, non-empty catalog may keep omitting the selected model.
///
/// A catalog that already lists other models proves the upstream is warm, so a missing model
/// is a configuration error rather than a cold start and must not consume the whole budget.
const CATALOG_MODEL_GRACE: Duration = Duration::from_secs(10);
/// Readiness budget for a gateway the launcher did not start.
const ATTACHED_GATEWAY_TIMEOUT: Duration = Duration::from_secs(30);
/// Readiness poll interval for a gateway the launcher did not start.
const ATTACHED_GATEWAY_INTERVAL: Duration = Duration::from_millis(250);

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

const fn harness_binary_names(harness: Harness) -> (&'static str, &'static str) {
    match harness {
        Harness::Codex => ("codex", "AGENTIC_CODEX_BIN"),
        Harness::Claude => ("claude", "AGENTIC_CLAUDE_BIN"),
    }
}

fn harness_binary(harness: Harness) -> OsString {
    let (binary_name, override_name) = harness_binary_names(harness);
    std::env::var_os(override_name).unwrap_or_else(|| binary_name.into())
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

/// The Codex model the launcher runs and the input modalities the gateway resolved for it.
#[derive(Debug)]
struct CodexModelSelection {
    model: String,
    input_modalities: InputModalities,
}

/// The model a harness will run, with the metadata that harness needs to configure it.
#[derive(Debug)]
enum HarnessModel {
    Codex(CodexModelSelection),
    Claude(String),
}

/// The polling budget for one catalog resolution.
#[derive(Clone, Copy, Debug)]
struct CatalogBudget {
    /// Overall wall-clock budget for resolving the catalog.
    timeout: Duration,
    /// Delay between attempts, unless the gateway asks for a longer one.
    interval: Duration,
    /// How long a served, non-empty catalog may keep omitting the selected model.
    missing_grace: Duration,
}

/// Why one catalog attempt failed, and whether another attempt could succeed.
enum CatalogAttempt {
    Resolved(CodexModelSelection),
    /// The gateway or its upstream may still be warming up.
    Transient(Error, Option<Duration>),
    /// Another attempt cannot change the result.
    Permanent(Error),
    /// The catalog is served and lists models, but not the selected one.
    ModelMissing(Error),
}

enum BodyError {
    TooLarge,
    Transport(reqwest::Error),
}

/// Statuses a warming gateway can return before it can serve its catalog.
fn is_transient_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error()
        || matches!(
            status,
            reqwest::StatusCode::REQUEST_TIMEOUT
                | reqwest::StatusCode::TOO_EARLY
                | reqwest::StatusCode::TOO_MANY_REQUESTS
        )
}

/// `Retry-After` expressed in whole seconds; the HTTP-date form is not honored.
fn retry_after(response: &reqwest::Response) -> Option<Duration> {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

/// Read a catalog response without trusting the gateway to bound it.
async fn read_bounded_body(mut response: reqwest::Response) -> Result<Vec<u8>, BodyError> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_CATALOG_BYTES as u64)
    {
        return Err(BodyError::TooLarge);
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(BodyError::Transport)? {
        if body.len() + chunk.len() > MAX_CATALOG_BYTES {
            return Err(BodyError::TooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// The models a catalog advertises, for an error message that names the alternatives.
fn advertised_models(catalog: &CodexCatalogCapabilities) -> String {
    const LISTED: usize = 5;
    let listed = catalog
        .models
        .iter()
        .take(LISTED)
        .map(|entry| entry.slug.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    if catalog.models.len() > LISTED {
        format!("{listed}, ...")
    } else {
        listed
    }
}

/// Resolve the Codex CLI version the gateway catalog is requested for.
///
/// The gateway only transforms its model list when a client version is present, and Codex
/// reports its own version, so the launcher asks the same binary it is about to run instead of
/// inventing a value. [`CODEX_CLIENT_VERSION_ENV`] skips the probe where it cannot run.
///
/// # Errors
///
/// Returns a configuration error when the Codex binary cannot be run or reports no version.
async fn codex_client_version() -> Result<String, Error> {
    if let Some(version) = std::env::var(CODEX_CLIENT_VERSION_ENV)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
    {
        return Ok(version);
    }
    let binary = harness_binary(Harness::Codex);
    let display_binary = binary.to_string_lossy().into_owned();
    let mut command = tokio::process::Command::new(&binary);
    command
        .arg("--version")
        .kill_on_drop(true)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    let output = tokio::time::timeout(CODEX_VERSION_PROBE_TIMEOUT, command.output())
        .await
        .map_err(|_| {
            Error::Config(format!(
                "{display_binary} --version timed out after {}s; set {CODEX_CLIENT_VERSION_ENV} to skip the probe",
                CODEX_VERSION_PROBE_TIMEOUT.as_secs()
            ))
        })?
        .map_err(|error| {
            Error::Config(format!(
                "failed to run {display_binary} --version: {error}; install Codex or set AGENTIC_CODEX_BIN"
            ))
        })?;
    if !output.status.success() {
        return Err(Error::Config(format!(
            "{display_binary} --version failed with {}; set {CODEX_CLIENT_VERSION_ENV} to skip the probe",
            output.status
        )));
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find(|line| !line.trim().is_empty())
        .and_then(|line| line.split_whitespace().next_back())
        .map(str::to_owned)
        .ok_or_else(|| {
            Error::Config(format!(
                "could not read a version from {display_binary} --version; set {CODEX_CLIENT_VERSION_ENV} to provide it"
            ))
        })
}

/// Ask the gateway once for the model catalog and select the requested model.
async fn catalog_attempt(
    client: &Client,
    catalog_url: &str,
    display_url: &str,
    client_version: &str,
    requested_model: Option<&str>,
    api_key: Option<&str>,
) -> CatalogAttempt {
    let mut request = client.get(catalog_url).query(&[("client_version", client_version)]);
    if let Some(api_key) = api_key {
        request = request.bearer_auth(api_key);
    }
    let response = match request.send().await {
        Ok(response) => response,
        Err(error) => {
            return CatalogAttempt::Transient(
                Error::Config(format!(
                    "failed to reach the gateway model catalog at {display_url}: {}",
                    error.without_url()
                )),
                None,
            );
        }
    };

    let status = response.status();
    if !status.is_success() {
        let retry_after = retry_after(&response);
        let message = format!("the gateway model catalog at {display_url} returned HTTP {status}");
        return if matches!(
            status,
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN
        ) {
            CatalogAttempt::Permanent(Error::Config(format!(
                "{message}; pass --api-key if the gateway requires authentication"
            )))
        } else if is_transient_status(status) {
            CatalogAttempt::Transient(Error::Config(message), retry_after)
        } else {
            CatalogAttempt::Permanent(Error::Config(message))
        };
    }

    let body = match read_bounded_body(response).await {
        Ok(body) => body,
        Err(BodyError::TooLarge) => {
            return CatalogAttempt::Permanent(Error::Config(format!(
                "the gateway model catalog at {display_url} is larger than {MAX_CATALOG_BYTES} bytes"
            )));
        }
        Err(BodyError::Transport(error)) => {
            return CatalogAttempt::Transient(
                Error::Config(format!(
                    "failed to read the gateway model catalog at {display_url}: {}",
                    error.without_url()
                )),
                None,
            );
        }
    };

    let catalog: CodexCatalogCapabilities = match serde_json::from_slice(&body) {
        Ok(catalog) => catalog,
        Err(error) => {
            return CatalogAttempt::Permanent(Error::Config(format!(
                "the gateway model catalog at {display_url} is not a Codex model catalog: {error}"
            )));
        }
    };
    if catalog.models.is_empty() {
        return CatalogAttempt::Transient(
            Error::Config(format!("the gateway model catalog at {display_url} lists no models")),
            None,
        );
    }
    let Some(entry) = catalog.select(requested_model) else {
        return CatalogAttempt::ModelMissing(Error::Config(format!(
            "the gateway model catalog at {display_url} does not list model {:?}; it serves: {}",
            requested_model.unwrap_or_default(),
            advertised_models(&catalog)
        )));
    };
    if requested_model.is_none() && catalog.models.len() > 1 {
        eprintln!(
            "gateway serves {} models; using {}. Pass --model to choose another.",
            catalog.models.len(),
            entry.slug
        );
    }
    CatalogAttempt::Resolved(CodexModelSelection {
        model: entry.slug.clone(),
        input_modalities: entry.input_modalities,
    })
}

/// Resolve the Codex model and its input modalities from one gateway catalog snapshot.
///
/// Selecting the model and reading its capabilities from the same response keeps the isolated
/// Codex catalog consistent with what the gateway serves over HTTP. Transient failures are
/// retried until `timeout` expires, because a gateway can answer `/health` before its upstream
/// can list models; authentication failures and undecodable catalogs are reported immediately.
///
/// # Errors
///
/// Returns a configuration error when the catalog cannot be fetched within `timeout`, the
/// gateway rejects the request, or the catalog does not list the selected model.
async fn resolve_codex_selection(
    client: &Client,
    gateway_url: &str,
    requested_model: Option<&str>,
    api_key: Option<&str>,
    timeout: Duration,
    interval: Duration,
) -> Result<CodexModelSelection, Error> {
    let client_version = codex_client_version().await?;
    catalog_selection(
        client,
        gateway_url,
        &client_version,
        requested_model,
        api_key,
        CatalogBudget {
            timeout,
            interval,
            missing_grace: CATALOG_MODEL_GRACE,
        },
    )
    .await
}

/// Poll the gateway catalog for `requested_model` until it resolves or the budget expires.
async fn catalog_selection(
    client: &Client,
    gateway_url: &str,
    client_version: &str,
    requested_model: Option<&str>,
    api_key: Option<&str>,
    budget: CatalogBudget,
) -> Result<CodexModelSelection, Error> {
    let base = gateway_url.trim_end_matches('/');
    let catalog_url = format!("{base}/v1/models");
    let display_url = redact_url(base);
    let deadline = Instant::now() + budget.timeout;
    // Set on the first miss rather than up front: a slow warm-up must not consume the grace a
    // served catalog is owed once it starts answering.
    let mut missing_deadline = None;
    let mut last_error = None;

    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let Ok(attempt) = tokio::time::timeout(
            remaining,
            catalog_attempt(
                client,
                &catalog_url,
                &display_url,
                client_version,
                requested_model,
                api_key,
            ),
        )
        .await
        else {
            break;
        };

        let delay = match attempt {
            CatalogAttempt::Resolved(selection) => return Ok(selection),
            CatalogAttempt::Permanent(error) => return Err(error),
            CatalogAttempt::ModelMissing(error) => {
                let now = Instant::now();
                if now >= *missing_deadline.get_or_insert((now + budget.missing_grace).min(deadline)) {
                    return Err(error);
                }
                last_error = Some(error);
                budget.interval
            }
            CatalogAttempt::Transient(error, retry_after) => {
                last_error = Some(error);
                retry_after.unwrap_or(budget.interval)
            }
        };

        let now = Instant::now();
        if now >= deadline {
            break;
        }
        sleep(delay.min(deadline - now)).await;
    }

    Err(last_error.unwrap_or_else(|| {
        Error::Config(format!(
            "the gateway model catalog at {display_url} did not become available"
        ))
    }))
}

/// Resolve the model each harness will run.
///
/// Codex reads its model and capabilities from the gateway catalog; Claude Code keeps using the
/// upstream model listing, which needs no capability metadata.
///
/// # Errors
///
/// Returns a configuration error when no model can be resolved.
async fn resolve_harness_model(
    client: &Client,
    harness: Harness,
    gateway_url: &str,
    options: &HarnessOptions,
) -> Result<HarnessModel, Error> {
    match harness {
        Harness::Codex => Ok(HarnessModel::Codex(
            resolve_codex_selection(
                client,
                gateway_url,
                options.source.model.as_deref(),
                options.common.api_key.as_deref(),
                Duration::from_secs_f64(options.common.llm_ready_timeout_s),
                Duration::from_secs_f64(options.common.llm_ready_interval_s),
            )
            .await?,
        )),
        Harness::Claude => Ok(HarnessModel::Claude(
            resolve_model(client, &options.source, options.common.api_key.as_deref()).await?,
        )),
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

    let harness_model = match resolve_harness_model(&client, harness, &gateway_url, &options).await {
        Ok(harness_model) => harness_model,
        Err(error) => {
            cleanup(&mut server, session_root.path()).await;
            return Err(error);
        }
    };
    let harness_env = match harness_environment(
        &harness_model,
        &gateway_url,
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
    harness_model: &HarnessModel,
    gateway_url: &str,
    options: &HarnessOptions,
    session_root: &Path,
    claude_state_root: &Path,
) -> Result<HarnessEnv, Error> {
    let inherited_auth_token = std::env::var("ANTHROPIC_AUTH_TOKEN")
        .ok()
        .filter(|value| !value.trim().is_empty());
    let mut environment = match harness_model {
        HarnessModel::Codex(selection) => crate::agentic_harness::prepare_codex_home(
            session_root,
            gateway_url,
            &selection.model,
            selection.input_modalities,
            options.common.api_key.as_deref(),
        )
        .map_err(Error::from),
        HarnessModel::Claude(model) => crate::agentic_harness::prepare_claude_home_with_state(
            session_root,
            claude_state_root,
            gateway_url,
            model,
            inherited_auth_token.as_deref(),
            options.common.api_key.as_deref(),
        )
        .map_err(Error::from),
    }?;
    if matches!(harness_model, HarnessModel::Claude(_)) {
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
    let (binary_name, override_name) = harness_binary_names(harness);
    let binary = harness_binary(harness);
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
    harness_model: &HarnessModel,
    session_root: &Path,
    claude_state_root: &Path,
    gateway_url: &str,
    api_key: Option<&str>,
) -> Result<HarnessEnv, Error> {
    match harness_model {
        HarnessModel::Codex(selection) => crate::agentic_harness::prepare_codex_home(
            session_root,
            gateway_url,
            &selection.model,
            selection.input_modalities,
            api_key,
        )
        .map_err(Error::from),
        HarnessModel::Claude(model) => crate::agentic_harness::prepare_claude_home_with_state(
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
            ATTACHED_GATEWAY_TIMEOUT,
            ATTACHED_GATEWAY_INTERVAL,
            false,
        )
        .await?;
        let harness_model = match harness {
            Harness::Codex => HarnessModel::Codex(
                resolve_codex_selection(
                    &client,
                    &options.gateway_url,
                    Some(&options.model),
                    options.api_key.as_deref(),
                    ATTACHED_GATEWAY_TIMEOUT,
                    ATTACHED_GATEWAY_INTERVAL,
                )
                .await?,
            ),
            Harness::Claude => HarnessModel::Claude(options.model.clone()),
        };
        let mut harness_env = prepare_attached_harness_environment(
            &harness_model,
            session_root.path(),
            &claude_state_root,
            &options.gateway_url,
            options.api_key.as_deref(),
        )?;
        if matches!(harness, Harness::Claude) {
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

    use super::{CodexModelSelection, DEFAULT_CLAUDE_EFFORT, HarnessModel, harness_launch_args, server_args};
    use crate::agentic_cli::{CommonOptions, Harness, SourceOptions};
    use crate::model_capabilities::InputModalities;

    /// A gateway that answers catalog requests from a scripted queue and records what it was asked.
    struct MockGateway {
        url: String,
        requests: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl MockGateway {
        fn request_count(&self) -> usize {
            self.requests.lock().expect("request log").len()
        }

        fn first_request(&self) -> String {
            self.requests
                .lock()
                .expect("request log")
                .first()
                .cloned()
                .unwrap_or_default()
        }
    }

    /// Serve `responses` in order, repeating the last one once the queue is exhausted.
    async fn spawn_mock_gateway(responses: Vec<(&'static str, String)>) -> MockGateway {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("listener");
        let address = listener.local_addr().expect("listener address");
        let requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = std::sync::Arc::clone(&requests);
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};

            let mut served = 0;
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let mut buffer = [0_u8; 4096];
                let read = socket.read(&mut buffer).await.unwrap_or_default();
                recorded
                    .lock()
                    .expect("request log")
                    .push(String::from_utf8_lossy(&buffer[..read]).into_owned());
                let (status, body) = responses
                    .get(served)
                    .or_else(|| responses.last())
                    .cloned()
                    .unwrap_or(("200 OK", String::new()));
                served += 1;
                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        MockGateway {
            url: format!("http://{address}"),
            requests,
        }
    }

    fn catalog_body() -> String {
        r#"{"models":[
            {"slug":"first-model","input_modalities":["text"]},
            {"slug":"vision-model","input_modalities":["text","image"]}
        ]}"#
        .to_owned()
    }

    async fn select(
        gateway: &MockGateway,
        requested_model: Option<&str>,
        api_key: Option<&str>,
    ) -> Result<super::CodexModelSelection, agentic_core::error::Error> {
        super::catalog_selection(
            &reqwest::Client::new(),
            &gateway.url,
            "9.9.9",
            requested_model,
            api_key,
            super::CatalogBudget {
                timeout: std::time::Duration::from_millis(400),
                interval: std::time::Duration::from_millis(10),
                missing_grace: super::CATALOG_MODEL_GRACE,
            },
        )
        .await
    }

    #[tokio::test]
    async fn catalog_selection_reads_the_resolved_modalities() {
        let gateway = spawn_mock_gateway(vec![("200 OK", catalog_body())]).await;

        let selection = select(&gateway, Some("vision-model"), None)
            .await
            .expect("the catalog lists the requested model");

        assert_eq!(selection.model, "vision-model");
        assert_eq!(selection.input_modalities, InputModalities::TextAndImage);
        let request = gateway.first_request();
        assert!(
            request.starts_with("GET /v1/models?client_version=9.9.9 "),
            "the gateway only transforms its catalog for a client version: {request}"
        );
        assert!(
            !request.to_ascii_lowercase().contains("authorization:"),
            "no credential must be sent when none is configured"
        );
    }

    #[tokio::test]
    async fn catalog_selection_defaults_to_the_first_advertised_model() {
        let gateway = spawn_mock_gateway(vec![("200 OK", catalog_body())]).await;

        let selection = select(&gateway, None, None).await.expect("a catalog entry is selected");

        assert_eq!(selection.model, "first-model");
        assert_eq!(selection.input_modalities, InputModalities::Text);
    }

    #[tokio::test]
    async fn catalog_selection_sends_the_configured_credential() {
        let gateway = spawn_mock_gateway(vec![("200 OK", catalog_body())]).await;

        select(&gateway, Some("first-model"), Some("gateway-key"))
            .await
            .expect("the catalog lists the requested model");

        assert!(
            gateway
                .first_request()
                .to_ascii_lowercase()
                .contains("authorization: bearer gateway-key"),
            "the configured API key must reach a protected gateway"
        );
    }

    #[tokio::test]
    async fn catalog_selection_reports_a_model_the_gateway_does_not_serve() {
        let gateway = spawn_mock_gateway(vec![("200 OK", catalog_body())]).await;

        let error = select(&gateway, Some("absent-model"), None)
            .await
            .expect_err("a model the gateway does not serve must fail");
        let message = error.to_string();

        assert!(message.contains("absent-model"), "{message}");
        assert!(message.contains("first-model, vision-model"), "{message}");
    }

    #[tokio::test]
    async fn catalog_selection_does_not_retry_rejected_credentials() {
        let gateway = spawn_mock_gateway(vec![("401 Unauthorized", "{}".to_owned())]).await;

        let error = select(&gateway, Some("first-model"), None)
            .await
            .expect_err("a rejected credential must fail");
        let message = error.to_string();

        assert!(message.contains("401"), "{message}");
        assert!(message.contains("--api-key"), "{message}");
        assert_eq!(
            gateway.request_count(),
            1,
            "authentication failures must not be retried"
        );
    }

    #[tokio::test]
    async fn catalog_selection_retries_a_warming_gateway() {
        let gateway = spawn_mock_gateway(vec![
            ("503 Service Unavailable", "{}".to_owned()),
            ("200 OK", catalog_body()),
        ])
        .await;

        let selection = select(&gateway, Some("vision-model"), None)
            .await
            .expect("a warming gateway must be retried");

        assert_eq!(selection.input_modalities, InputModalities::TextAndImage);
        assert_eq!(gateway.request_count(), 2);
    }

    #[tokio::test]
    async fn catalog_selection_retries_an_empty_catalog() {
        let gateway = spawn_mock_gateway(vec![
            ("200 OK", r#"{"models":[]}"#.to_owned()),
            ("200 OK", catalog_body()),
        ])
        .await;

        let selection = select(&gateway, None, None)
            .await
            .expect("an upstream that is still loading must be retried");

        assert_eq!(selection.model, "first-model");
        assert_eq!(gateway.request_count(), 2);
    }

    #[tokio::test]
    async fn catalog_selection_starts_the_missing_model_grace_at_the_first_miss() {
        let warming = ("503 Service Unavailable", "{}".to_owned());
        let without_the_model = (
            "200 OK",
            r#"{"models":[{"slug":"other-model","input_modalities":["text"]}]}"#.to_owned(),
        );
        let gateway = spawn_mock_gateway(vec![
            warming.clone(),
            warming.clone(),
            warming.clone(),
            warming.clone(),
            warming.clone(),
            warming,
            without_the_model,
            ("200 OK", catalog_body()),
        ])
        .await;

        // Warm-up alone outlasts the grace: six retries at 30ms exceed the 150ms window, so a
        // grace anchored at the first attempt would already have expired by the first miss.
        let selection = super::catalog_selection(
            &reqwest::Client::new(),
            &gateway.url,
            "9.9.9",
            Some("vision-model"),
            None,
            super::CatalogBudget {
                timeout: std::time::Duration::from_secs(3),
                interval: std::time::Duration::from_millis(30),
                missing_grace: std::time::Duration::from_millis(150),
            },
        )
        .await
        .expect("a slow warm-up must not consume the model-missing grace");

        assert_eq!(selection.model, "vision-model");
        assert_eq!(selection.input_modalities, InputModalities::TextAndImage);
        assert_eq!(
            gateway.request_count(),
            8,
            "every warm-up response, the miss, and the successful catalog must each be requested"
        );
    }

    #[tokio::test]
    async fn catalog_selection_rejects_an_undecodable_catalog() {
        let gateway = spawn_mock_gateway(vec![("200 OK", "not a catalog".to_owned())]).await;

        let error = select(&gateway, Some("first-model"), None)
            .await
            .expect_err("an undecodable catalog must fail");

        assert!(error.to_string().contains("not a Codex model catalog"), "{error}");
        assert_eq!(gateway.request_count(), 1, "an undecodable catalog must not be retried");
    }

    #[tokio::test]
    async fn catalog_selection_rejects_an_oversized_catalog() {
        let oversized = format!(
            r#"{{"models":[{{"slug":"{}","input_modalities":["text"]}}]}}"#,
            "x".repeat(super::MAX_CATALOG_BYTES + 1)
        );
        let gateway = spawn_mock_gateway(vec![("200 OK", oversized)]).await;

        let error = select(&gateway, Some("first-model"), None)
            .await
            .expect_err("an oversized catalog must fail");

        assert!(error.to_string().contains("larger than"), "{error}");
        assert_eq!(gateway.request_count(), 1, "an oversized catalog must not be retried");
    }

    #[tokio::test]
    async fn catalog_selection_redacts_gateway_credentials_when_unreachable() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("listener");
        let address = listener.local_addr().expect("listener address");
        drop(listener);

        let error = super::catalog_selection(
            &reqwest::Client::new(),
            &format!("http://agentic:gateway-secret@{address}"),
            "9.9.9",
            Some("first-model"),
            None,
            super::CatalogBudget {
                timeout: std::time::Duration::from_millis(50),
                interval: std::time::Duration::from_millis(10),
                missing_grace: super::CATALOG_MODEL_GRACE,
            },
        )
        .await
        .expect_err("an unreachable gateway must fail");
        let message = error.to_string();

        assert!(!message.contains("gateway-secret"), "{message}");
        assert!(message.contains("[REDACTED]"), "{message}");
    }

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
            &HarnessModel::Claude("served-discovered".to_owned()),
            "http://127.0.0.1:3000",
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
                &HarnessModel::Claude("served-discovered".to_owned()),
                "http://127.0.0.1:3000",
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
            &HarnessModel::Codex(CodexModelSelection {
                model: "Qwen/Qwen3-8B".to_owned(),
                input_modalities: InputModalities::TextAndImage,
            }),
            &root,
            &root,
            "http://127.0.0.1:9000",
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
        let catalog: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("model_catalog.json")).expect("Codex catalog"))
                .expect("valid catalog JSON");
        assert_eq!(
            catalog["models"][0]["input_modalities"],
            serde_json::json!(["text", "image"]),
            "the attached launcher must write the modalities the gateway resolved"
        );

        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn attached_claude_uses_persistent_state_root() {
        let session_root = tempfile::tempdir().expect("session root");
        let state_root = tempfile::tempdir().expect("state root");
        let environment = super::prepare_attached_harness_environment(
            &HarnessModel::Claude("Qwen/Qwen3-8B".to_owned()),
            session_root.path(),
            state_root.path(),
            "http://127.0.0.1:9000",
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
                &HarnessModel::Claude("Qwen/Qwen3-8B".to_owned()),
                session_root.path(),
                state_root.path(),
                "http://127.0.0.1:9000",
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
            &HarnessModel::Claude("served-test".to_owned()),
            "http://127.0.0.1:3000",
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
