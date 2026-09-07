use std::process::Command;

#[test]
fn missing_llm_api_base_error_mentions_config_environment_and_flag() {
    let home = tempfile::tempdir().expect("temporary Agentic API home");
    let output = Command::new(env!("CARGO_BIN_EXE_agentic-server"))
        .env("AGENTIC_API_HOME", home.path())
        .env_remove("LLM_API_BASE")
        .env_remove("OIDC_ISSUER")
        .env_remove("OIDC_AUDIENCE")
        .output()
        .expect("agentic-server must run");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("stderr must be UTF-8");
    assert!(
        stderr.contains("llm_api_base in config.toml, LLM_API_BASE, or --llm-api-base"),
        "unexpected error message: {stderr}"
    );
}

#[test]
fn help_does_not_expose_database_url_credentials() {
    let database_url = "postgresql://agentic:super-secret@db.example/agentic";
    let output = Command::new(env!("CARGO_BIN_EXE_agentic-server"))
        .env("DATABASE_URL", database_url)
        .arg("--help")
        .output()
        .expect("agentic-server help must run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout must be UTF-8");
    assert!(stdout.contains("DATABASE_URL"));
    assert!(!stdout.contains(database_url));
    assert!(!stdout.contains("super-secret"));
}

#[test]
fn agentic_server_reports_version() {
    let output = Command::new(env!("CARGO_BIN_EXE_agentic-server"))
        .arg("--version")
        .output()
        .expect("agentic-server version must run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout must be UTF-8");
    assert!(stdout.contains(concat!("agentic-server ", env!("CARGO_PKG_VERSION"))));
}

#[test]
fn packaged_agentic_cli_preserves_top_level_commands_and_harness_subcommands() {
    let output = Command::new(env!("CARGO_BIN_EXE_agentic"))
        .arg("--help")
        .output()
        .expect("agentic help must run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout must be UTF-8");
    assert!(stdout.contains("run"));
    assert!(stdout.contains("serve"));
    assert!(stdout.contains("validate"));

    let run_output = Command::new(env!("CARGO_BIN_EXE_agentic"))
        .args(["run", "--help"])
        .output()
        .expect("agentic run help must run");

    assert!(run_output.status.success());
    let run_stdout = String::from_utf8(run_output.stdout).expect("stdout must be UTF-8");
    assert!(run_stdout.contains("codex"));
    assert!(run_stdout.contains("claude"));
}

#[test]
fn agentic_cli_errors_redact_url_userinfo() {
    let output = Command::new(env!("CARGO_BIN_EXE_agentic"))
        .args([
            "run",
            "claude",
            "--upstream",
            "https://secret-token@example.com?unsupported=true",
        ])
        .output()
        .expect("agentic CLI must run");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("stderr must be UTF-8");
    assert!(!stderr.contains("secret-token"), "credential leaked in: {stderr}");
    assert!(stderr.contains("https://[REDACTED]@example.com?unsupported=true"));
}

#[test]
fn agentic_version_exits_successfully() {
    let output = Command::new(env!("CARGO_BIN_EXE_agentic"))
        .arg("--version")
        .output()
        .expect("agentic CLI must run");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("stdout must be UTF-8");
    assert!(stdout.starts_with("agentic "));
}
