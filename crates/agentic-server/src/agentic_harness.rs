use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
};

use crate::agentic_output::redact_url;
use crate::model_capabilities::InputModalities;

#[derive(Debug)]
pub struct HarnessEnv {
    pub environment: BTreeMap<String, String>,
    pub environment_remove: Vec<String>,
    pub arguments: Vec<String>,
    pub files: Vec<PathBuf>,
    pub summary: String,
}

pub const CLAUDE_CANONICAL_MODEL: &str = "claude-sonnet-4-5-20250929";

/// Write an isolated Codex home for an Agentic API session.
///
/// `input_modalities` must be the modalities the gateway resolved for `model`: Codex strips
/// image content from a request when this catalog says the model accepts text only, so the
/// value is required rather than defaulted.
///
/// # Errors
///
/// Returns an I/O error when the temporary home or generated files cannot be written.
pub(crate) fn prepare_codex_home(
    root: &Path,
    gateway_url: &str,
    model: &str,
    input_modalities: InputModalities,
    api_key: Option<&str>,
) -> Result<HarnessEnv, io::Error> {
    fs::create_dir_all(root)?;
    let gateway_url = format!("{}/v1", gateway_url.trim_end_matches('/'));
    let catalog_path = root.join("model_catalog.json");
    let config_path = root.join("config.toml");
    let catalog = serde_json::json!({
        "models": [{
            "slug": model,
            "display_name": model,
            "supported_in_api": true,
            "visibility": "list",
            "priority": 0,
            "input_modalities": input_modalities,
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [
                {"effort": "low", "description": "Fast responses"},
                {"effort": "medium", "description": "Balanced responses"},
                {"effort": "high", "description": "Deep reasoning"}
            ],
            "supports_reasoning_summaries": true,
            "supports_parallel_tool_calls": true,
            // apply_patch_tool_type is intentionally omitted: Codex only supports
            // "freeform", which the gateway cannot normalize while preserving
            // constrained decoding. Codex falls back to editing via the shell tool.
            "web_search_tool_type": "text",
            "shell_type": "local",
            "context_window": 32768,
            "max_context_window": 262_144,
            "base_instructions": "",
            "support_verbosity": false,
            "supports_image_detail_original": false,
            "use_responses_lite": false,
            "supports_search_tool": false,
            "include_skills_usage_instructions": false,
            "truncation_policy": {"limit": 32768, "mode": "tokens"},
            "experimental_supported_tools": []
        }]
    });
    let catalog_bytes = serde_json::to_vec_pretty(&catalog).map_err(io::Error::other)?;
    fs::write(&catalog_path, catalog_bytes).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "failed to write Codex model catalog {}: {error}",
                catalog_path.display()
            ),
        )
    })?;

    let requires_auth = api_key.is_some();
    let config = format!(
        "model = \"{}\"\nmodel_provider = \"agentic-api\"\nmodel_catalog_json = \"{}\"\n\n\
[model_providers.agentic-api]\nname = \"Agentic API\"\nbase_url = \"{}\"\n\
wire_api = \"responses\"\nrequires_openai_auth = {requires_auth}\nsupports_websockets = true\n",
        toml_escape(model),
        toml_escape(&catalog_path.display().to_string()),
        toml_escape(&gateway_url),
    );
    fs::write(&config_path, config).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to write Codex config {}: {error}", config_path.display()),
        )
    })?;

    let mut environment = BTreeMap::new();
    environment.insert("CODEX_HOME".to_owned(), root.display().to_string());
    if let Some(api_key) = api_key {
        environment.insert("OPENAI_API_KEY".to_owned(), api_key.to_owned());
    }

    Ok(HarnessEnv {
        environment,
        environment_remove: vec!["OPENAI_API_KEY".to_owned()],
        arguments: Vec::new(),
        files: vec![config_path, catalog_path],
        summary: format!("Codex home: {} (model: {model})", root.display()),
    })
}

/// Write an isolated Claude Code configuration for an Agentic API session.
///
/// # Errors
///
/// Returns an I/O error when the temporary configuration directory or settings file cannot be written.
pub fn prepare_claude_home(
    root: &Path,
    gateway_url: &str,
    model: &str,
    api_key: Option<&str>,
) -> Result<HarnessEnv, io::Error> {
    prepare_claude_home_with_state(root, root, gateway_url, model, None, api_key)
}

pub(crate) fn prepare_claude_home_with_state(
    root: &Path,
    state_root: &Path,
    gateway_url: &str,
    model: &str,
    auth_token: Option<&str>,
    api_key: Option<&str>,
) -> Result<HarnessEnv, io::Error> {
    fs::create_dir_all(root)?;
    ensure_private_directory(state_root)?;
    let settings_path = root.join("settings.json");
    let settings = serde_json::json!({
        "modelOverrides": {
            CLAUDE_CANONICAL_MODEL: model,
        }
    });
    let settings_bytes = serde_json::to_vec_pretty(&settings).map_err(io::Error::other)?;
    fs::write(&settings_path, settings_bytes).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "failed to write Claude Code settings {}: {error}",
                settings_path.display()
            ),
        )
    })?;

    let api_key_value = api_key.unwrap_or("agentic-api-local");
    let auth_token = auth_token.unwrap_or(api_key_value);
    let mut environment = BTreeMap::from([
        (
            "ANTHROPIC_BASE_URL".to_owned(),
            gateway_url.trim_end_matches('/').to_owned(),
        ),
        ("ANTHROPIC_MODEL".to_owned(), model.to_owned()),
        ("ANTHROPIC_SMALL_FAST_MODEL".to_owned(), model.to_owned()),
        ("ANTHROPIC_DEFAULT_OPUS_MODEL".to_owned(), model.to_owned()),
        ("ANTHROPIC_DEFAULT_SONNET_MODEL".to_owned(), model.to_owned()),
        ("ANTHROPIC_DEFAULT_HAIKU_MODEL".to_owned(), model.to_owned()),
        ("ANTHROPIC_API_KEY".to_owned(), api_key_value.to_owned()),
        ("ANTHROPIC_AUTH_TOKEN".to_owned(), auth_token.to_owned()),
        ("CLAUDE_CONFIG_DIR".to_owned(), state_root.display().to_string()),
        ("CLAUDE_CODE_MAX_CONTEXT_TOKENS".to_owned(), "32768".to_owned()),
        ("CLAUDE_CODE_MAX_OUTPUT_TOKENS".to_owned(), "2048".to_owned()),
        ("MAX_THINKING_TOKENS".to_owned(), "0".to_owned()),
    ]);
    if let Some(api_key) = api_key {
        environment.insert("OPENAI_API_KEY".to_owned(), api_key.to_owned());
    }
    Ok(HarnessEnv {
        environment,
        environment_remove: [
            "OPENAI_API_KEY",
            "CLAUDE_CODE_USE_VERTEX",
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_FOUNDRY",
            "CLAUDE_CODE_USE_ANTHROPIC_AWS",
            "CLAUDE_CODE_USE_MANTLE",
            "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST",
            "ANTHROPIC_CUSTOM_HEADERS",
            "ANTHROPIC_VERTEX_PROJECT_ID",
            "CLOUD_ML_REGION",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        arguments: vec!["--settings".to_owned(), settings_path.display().to_string()],
        files: vec![settings_path],
        summary: format!(
            "Claude Code config: {} (gateway: {}, model: {model})",
            root.display(),
            redact_url(gateway_url.trim_end_matches('/'))
        ),
    })
}

fn toml_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\"', "\\\"")
}

fn ensure_private_directory(path: &Path) -> Result<(), io::Error> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700).create(path)?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    #[cfg(not(unix))]
    fs::create_dir_all(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{prepare_claude_home, prepare_claude_home_with_state, prepare_codex_home};
    use crate::model_capabilities::InputModalities;

    #[test]
    fn codex_config_is_isolated_and_contains_gateway_provider() {
        let root = unique_temp_dir("codex");
        let env = prepare_codex_home(&root, "http://127.0.0.1:3000", "Qwen/test", InputModalities::Text, None)
            .expect("config");
        let config = fs::read_to_string(root.join("config.toml")).expect("config file");

        assert!(config.contains("[model_providers.agentic-api]"));
        assert!(config.contains("base_url = \"http://127.0.0.1:3000/v1\""));
        assert!(config.contains("wire_api = \"responses\""));
        assert!(config.contains("requires_openai_auth = false"));
        assert!(config.contains("model = \"Qwen/test\""));
        assert!(!env.summary.contains("secret"));

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn codex_home_removes_inherited_openai_key_unless_explicitly_configured() {
        let root = unique_temp_dir("codex-credential-isolation");
        let without_key = prepare_codex_home(&root, "http://127.0.0.1:3000", "Qwen/test", InputModalities::Text, None)
            .expect("Codex config without gateway key");

        assert!(without_key.environment_remove.contains(&"OPENAI_API_KEY".to_owned()));
        assert!(!without_key.environment.contains_key("OPENAI_API_KEY"));

        let with_key = prepare_codex_home(
            &root,
            "http://127.0.0.1:3000",
            "Qwen/test",
            InputModalities::Text,
            Some("gateway-key"),
        )
        .expect("Codex config with gateway key");
        assert_eq!(
            with_key.environment.get("OPENAI_API_KEY"),
            Some(&"gateway-key".to_owned())
        );

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn claude_home_is_isolated_and_maps_the_canonical_model() {
        let root = unique_temp_dir("claude");
        let env = prepare_claude_home_with_state(
            &root,
            &root,
            "http://127.0.0.1:3000",
            "Qwen/test",
            None,
            Some("secret-key"),
        )
        .expect("Claude config");
        let settings = fs::read_to_string(root.join("settings.json")).expect("settings file");
        let settings: serde_json::Value = serde_json::from_str(&settings).expect("valid settings JSON");

        assert_eq!(
            env.environment.get("ANTHROPIC_BASE_URL"),
            Some(&"http://127.0.0.1:3000".to_owned())
        );
        assert_eq!(
            env.environment.get("CLAUDE_CONFIG_DIR"),
            Some(&root.display().to_string())
        );
        assert_eq!(
            env.environment.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS"),
            Some(&"32768".to_owned())
        );
        assert_eq!(
            env.environment.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"),
            Some(&"2048".to_owned())
        );
        assert_eq!(env.environment.get("MAX_THINKING_TOKENS"), Some(&"0".to_owned()));
        assert_eq!(env.environment.get("ANTHROPIC_API_KEY"), Some(&"secret-key".to_owned()));
        assert_eq!(
            env.environment.get("ANTHROPIC_AUTH_TOKEN"),
            Some(&"secret-key".to_owned())
        );
        assert_eq!(settings["modelOverrides"]["claude-sonnet-4-5-20250929"], "Qwen/test");
        assert!(env.environment_remove.contains(&"CLAUDE_CODE_USE_VERTEX".to_owned()));
        assert!(
            env.environment_remove
                .contains(&"CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST".to_owned())
        );
        assert!(
            env.environment_remove
                .contains(&"CLAUDE_CODE_USE_ANTHROPIC_AWS".to_owned())
        );
        assert!(env.environment_remove.contains(&"CLAUDE_CODE_USE_MANTLE".to_owned()));
        assert!(env.environment_remove.contains(&"ANTHROPIC_CUSTOM_HEADERS".to_owned()));
        assert!(
            env.environment_remove
                .contains(&"ANTHROPIC_VERTEX_PROJECT_ID".to_owned())
        );
        assert!(env.environment_remove.contains(&"CLOUD_ML_REGION".to_owned()));
        assert!(!env.summary.contains("secret-key"));

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn claude_home_maps_every_model_tier_to_the_served_model() {
        let root = unique_temp_dir("claude-model-tiers");
        let env = prepare_claude_home(&root, "http://127.0.0.1:3000", "Qwen/test", None).expect("Claude config");

        for variable in [
            "ANTHROPIC_MODEL",
            "ANTHROPIC_SMALL_FAST_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        ] {
            assert_eq!(
                env.environment.get(variable),
                Some(&"Qwen/test".to_owned()),
                "missing model mapping for {variable}"
            );
            assert!(!env.environment_remove.contains(&variable.to_owned()));
        }

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn claude_home_uses_a_local_placeholder_without_inherited_auth() {
        let root = unique_temp_dir("claude-no-key");
        let env = prepare_claude_home_with_state(&root, &root, "http://127.0.0.1:3000", "Qwen/test", None, None)
            .expect("Claude config");

        assert_eq!(
            env.environment.get("ANTHROPIC_API_KEY"),
            Some(&"agentic-api-local".to_owned())
        );
        assert_eq!(
            env.environment.get("ANTHROPIC_AUTH_TOKEN"),
            Some(&"agentic-api-local".to_owned())
        );
        assert!(!env.environment.contains_key("OPENAI_API_KEY"));
        assert!(env.environment_remove.contains(&"OPENAI_API_KEY".to_owned()));

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn claude_home_accepts_an_explicit_auth_token() {
        let root = unique_temp_dir("claude-explicit-auth");
        let env = prepare_claude_home_with_state(
            &root,
            &root,
            "http://127.0.0.1:3000",
            "Qwen/test",
            Some("oidc-token"),
            None,
        )
        .expect("Claude config");

        assert_eq!(
            env.environment.get("ANTHROPIC_AUTH_TOKEN"),
            Some(&"oidc-token".to_owned())
        );
        assert_eq!(
            env.environment.get("ANTHROPIC_API_KEY"),
            Some(&"agentic-api-local".to_owned())
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn claude_summary_redacts_gateway_password() {
        let root = unique_temp_dir("claude-redacted-url");
        let env = prepare_claude_home(&root, "https://agentic:gateway-secret@example.com", "Qwen/test", None)
            .expect("Claude config");

        assert!(!env.summary.contains("gateway-secret"));
        assert!(env.summary.contains("https://[REDACTED]@example.com"));

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn claude_uses_persistent_state_with_per_run_settings() {
        let settings_root = tempfile::tempdir().expect("settings root");
        let state_root = tempfile::tempdir().expect("state root");
        let env = prepare_claude_home_with_state(
            settings_root.path(),
            state_root.path(),
            "http://127.0.0.1:3000",
            "Qwen/test",
            None,
            None,
        )
        .expect("Claude config");

        assert_eq!(
            env.environment.get("CLAUDE_CONFIG_DIR"),
            Some(&state_root.path().display().to_string())
        );
        assert_eq!(
            env.arguments,
            [
                "--settings".to_owned(),
                settings_root.path().join("settings.json").display().to_string(),
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn claude_persistent_state_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let settings_root = tempfile::tempdir().expect("settings root");
        let state_parent = tempfile::tempdir().expect("state parent");
        let state_root = state_parent.path().join("claude");
        prepare_claude_home_with_state(
            settings_root.path(),
            &state_root,
            "http://127.0.0.1:3000",
            "Qwen/test",
            None,
            None,
        )
        .expect("Claude config");

        let mode = fs::metadata(state_root).expect("state metadata").permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
    }

    #[test]
    fn codex_catalog_advertises_the_resolved_input_modalities() {
        for (modalities, expected) in [
            (InputModalities::Text, serde_json::json!(["text"])),
            (InputModalities::TextAndImage, serde_json::json!(["text", "image"])),
        ] {
            let root = unique_temp_dir("codex-modalities");
            prepare_codex_home(&root, "http://127.0.0.1:3000", "Qwen/test", modalities, None).expect("config");
            let catalog: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(root.join("model_catalog.json")).expect("catalog file"))
                    .expect("valid catalog JSON");

            assert_eq!(catalog["models"][0]["input_modalities"], expected);
            assert_eq!(catalog["models"][0]["slug"], "Qwen/test");

            fs::remove_dir_all(root).expect("cleanup");
        }
    }

    #[test]
    fn codex_catalog_keeps_its_launcher_specific_tool_settings() {
        let root = unique_temp_dir("codex-tool-settings");
        prepare_codex_home(
            &root,
            "http://127.0.0.1:3000",
            "Qwen/test",
            InputModalities::TextAndImage,
            None,
        )
        .expect("config");
        let catalog: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(root.join("model_catalog.json")).expect("catalog file"))
                .expect("valid catalog JSON");
        let model = &catalog["models"][0];

        assert_eq!(
            model["shell_type"], "local",
            "the launcher runs Codex against a local shell"
        );
        assert!(
            model.get("apply_patch_tool_type").is_none(),
            "the launcher omits apply_patch_tool_type so Codex edits through the shell tool"
        );
        assert_eq!(
            model["truncation_policy"],
            serde_json::json!({"limit": 32768, "mode": "tokens"})
        );
        assert_eq!(model["supports_image_detail_original"], false);
        assert_eq!(model["web_search_tool_type"], "text");
        assert_eq!(model["include_skills_usage_instructions"], false);

        fs::remove_dir_all(root).expect("cleanup");
    }

    fn unique_temp_dir(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("agentic-api-{name}-{}", std::process::id()));
        fs::create_dir_all(&path).expect("temp dir");
        path
    }
}
