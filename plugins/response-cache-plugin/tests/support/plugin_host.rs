// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Loads the built response-cache library into an in-process Relay host.
//!
//! The first activation builds the `cdylib` once per test binary in release
//! mode. Each activation writes a manifest with the library hash and a
//! `plugins.toml` that enables the plugin with the test's configuration.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{LazyLock, Mutex, OnceLock};

use nemo_relay_plugin_host::dynamic::{PluginHostActivation, initialize};
use nemo_relay_plugin_host::{
    ConfigReport, DiagnosticLevel, PluginComponentSpec, PluginConfig, PluginError, Result,
};
use nemo_relay_response_cache_plugin::{PLUGIN_KIND, ResponseCacheConfig, validate_config};
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

static TEST_PLUGIN_HOST: LazyLock<Mutex<Option<(PluginHostActivation, TempDir)>>> =
    LazyLock::new(|| Mutex::new(None));

/// Diagnostics from validating one response-cache configuration.
#[derive(Debug)]
pub struct ValidationReport {
    pub diagnostics: Vec<nemo_relay_plugin_host::ConfigDiagnostic>,
}

impl ValidationReport {
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.level == DiagnosticLevel::Error)
    }
}

/// Validates a configuration with the plugin's own rules.
pub fn validate_cache(config: &ResponseCacheConfig) -> ValidationReport {
    ValidationReport {
        diagnostics: validate_config(&config_object(config)),
    }
}

/// Activates the plugin alone and returns the host's configuration report.
pub async fn activate_cache_report(config: ResponseCacheConfig) -> Result<ConfigReport> {
    activate_plugin(config_object(&config), Vec::new()).await
}

/// Activates the plugin next to static components, such as Relay's adaptive plugin.
pub async fn activate_plugin(
    config: Map<String, Json>,
    components: Vec<PluginComponentSpec>,
) -> Result<ConfigReport> {
    test_close_plugin_host()?;
    let directory = TempDir::new().map_err(|error| PluginError::Internal(error.to_string()))?;
    let manifest = write_manifest(directory.path());
    let plugins_toml = write_plugins_toml(directory.path(), &manifest, &config);
    let activation = initialize(
        PluginConfig {
            components,
            ..PluginConfig::default()
        },
        Some(plugins_toml),
    )
    .await?;
    let report = activation.report().config;
    *TEST_PLUGIN_HOST.lock().map_err(|error| {
        PluginError::Internal(format!("test plugin host lock poisoned: {error}"))
    })? = Some((activation, directory));
    Ok(report)
}

/// Closes the active test plugin host, if any.
pub fn test_close_plugin_host() -> Result<()> {
    let active = TEST_PLUGIN_HOST
        .lock()
        .map_err(|error| PluginError::Internal(format!("test plugin host lock poisoned: {error}")))?
        .take();
    let Some((mut activation, directory)) = active else {
        return Ok(());
    };
    match activation.close() {
        Ok(()) => Ok(()),
        Err(error) => {
            *TEST_PLUGIN_HOST.lock().map_err(|lock_error| {
                PluginError::Internal(format!("test plugin host lock poisoned: {lock_error}"))
            })? = Some((activation, directory));
            Err(error)
        }
    }
}

fn config_object(config: &ResponseCacheConfig) -> Map<String, Json> {
    match serde_json::to_value(config).expect("cache config serializes") {
        Json::Object(object) => object,
        _ => unreachable!("cache config serializes to an object"),
    }
}

/// Builds the plugin library once per test binary, in release mode.
fn plugin_library() -> &'static Path {
    static LIBRARY: OnceLock<PathBuf> = OnceLock::new();
    LIBRARY.get_or_init(|| {
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        // A separate target directory avoids waiting on the outer test build's lock.
        let target = crate_dir.join("target").join("test-cdylib");
        let status = Command::new(env!("CARGO"))
            .args(["build", "--release", "--locked", "--lib", "--manifest-path"])
            .arg(crate_dir.join("Cargo.toml"))
            .arg("--target-dir")
            .arg(&target)
            .status()
            .expect("cargo build should start");
        assert!(
            status.success(),
            "cargo build should produce the plugin library"
        );
        let library = target.join("release").join(library_name());
        assert!(
            library.exists(),
            "missing built library {}",
            library.display()
        );
        library
    })
}

fn write_manifest(directory: &Path) -> PathBuf {
    let library = plugin_library();
    let schema = directory.join("config.schema.json");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("config.schema.json"),
        &schema,
    )
    .expect("config schema should copy");
    let digest = Sha256::digest(std::fs::read(library).expect("read plugin library"));
    let digest: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    let library = format!("{:?}", library.to_string_lossy());
    let manifest = directory.join("relay-plugin.toml");
    std::fs::write(
        &manifest,
        format!(
            r#"manifest_version = 1

[plugin]
id = "{PLUGIN_KIND}"
kind = "rust_dynamic"

[compat]
relay = ">=1.0.0,<2.0"
native_api = "v1.0"

[defaults]
enabled = false

[capabilities]
items = ["plugin_native", "config_schema"]

[config_schema]
path = "config.schema.json"

[source]
artifact = {library}

[integrity]
sha256 = "sha256:{digest}"

[load]
library = {library}
"#
        ),
    )
    .expect("manifest should write");
    manifest
}

fn write_plugins_toml(directory: &Path, manifest: &Path, config: &Map<String, Json>) -> PathBuf {
    let path = directory.join("plugins.toml");
    let config = toml::Value::try_from(config.clone()).expect("plugin config should serialize");
    let document = toml::toml! {
        version = 1
        [plugins.policy.defaults]
        attestation = "integrity_only"
    };
    let mut document = toml::Value::Table(document);
    let plugins = document
        .get_mut("plugins")
        .and_then(toml::Value::as_table_mut)
        .expect("plugins table");
    let mut record = toml::map::Map::new();
    record.insert(
        "manifest".into(),
        toml::Value::String(manifest.to_string_lossy().into_owned()),
    );
    record.insert("config".into(), config);
    plugins.insert(
        "dynamic".into(),
        toml::Value::Array(vec![toml::Value::Table(record)]),
    );
    std::fs::write(
        &path,
        toml::to_string(&document).expect("plugins.toml serializes"),
    )
    .expect("plugins.toml should write");
    path
}

fn library_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "nemo_relay_response_cache_plugin.dll"
    } else if cfg!(target_os = "macos") {
        "libnemo_relay_response_cache_plugin.dylib"
    } else {
        "libnemo_relay_response_cache_plugin.so"
    }
}
