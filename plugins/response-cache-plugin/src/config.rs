// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Component configuration for the response-cache plugin.
//!
//! The component configuration is the [`ResponseCacheConfig`] object. Its keys
//! match the former `response_cache` section of Relay's adaptive plugin.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value as Json};

/// Strategy for deriving an LLM response-cache key.
///
/// The `Unknown` variant preserves an unsupported JSON/TOML value long enough
/// for configuration validation to report it with a field-specific diagnostic.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ResponseCacheKeyStrategy {
    /// Key on the normalized request exactly.
    #[default]
    ExactRequest,
    /// Normalize tool schemas structurally while preserving their interface.
    Logical,
    /// A wire value not supported by this Relay build.
    Unknown(String),
}

impl ResponseCacheKeyStrategy {
    /// Stable JSON/TOML representation of this strategy.
    pub fn as_str(&self) -> &str {
        match self {
            Self::ExactRequest => "exact_request",
            Self::Logical => "logical",
            Self::Unknown(value) => value,
        }
    }
}

impl From<&str> for ResponseCacheKeyStrategy {
    fn from(value: &str) -> Self {
        match value {
            "exact_request" => Self::ExactRequest,
            "logical" => Self::Logical,
            _ => Self::Unknown(value.to_string()),
        }
    }
}

impl From<String> for ResponseCacheKeyStrategy {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}

impl Serialize for ResponseCacheKeyStrategy {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ResponseCacheKeyStrategy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Self::from(value))
    }
}

/// Default in-memory byte budget: 256 MiB.
pub const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;

/// Storage backend selection: a `kind` plus kind-specific options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BackendConfig {
    /// Backend kind: `"in_memory"` or `"redis"`.
    pub kind: String,
    /// Backend-specific options (in_memory: `max_bytes`; redis: `url`/`key_prefix`).
    pub config: Map<String, Json>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            kind: "in_memory".to_string(),
            config: Map::new(),
        }
    }
}

impl BackendConfig {
    /// In-memory total-bytes budget before oldest-first eviction.
    pub fn max_bytes(&self) -> usize {
        self.config
            .get("max_bytes")
            .and_then(Json::as_u64)
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_MAX_BYTES)
    }
}

/// Opt-in tool-result cache configuration.
///
/// Cache only tools that are read-only and stable for their TTL.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolCacheConfig {
    /// Master switch; off by default.
    pub enabled: bool,
    /// Tool execution-intercept priority. Defaults to `150`.
    /// Lower values run first. Give guardrails a lower priority than the
    /// cache so guardrails run on cache hits.
    pub priority: i32,
    /// Whether conventional in-band tool error results may be stored.
    pub cache_errors: bool,
    /// Policy for unclassified tools; not cacheable by default.
    pub default: ToolClass,
    /// Named tool classes.
    pub classes: BTreeMap<String, ToolClass>,
    /// Per-tool refinements keyed by exact name, `prefix*`, `*suffix`, or `*contains*`.
    pub overrides: BTreeMap<String, ToolOverride>,
}

impl Default for ToolCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            priority: 150,
            cache_errors: false,
            default: ToolClass::default(),
            classes: BTreeMap::new(),
            overrides: BTreeMap::new(),
        }
    }
}

/// Policy shared by a class of tools.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolClass {
    /// Whether class members may be served from cache.
    pub cacheable: bool,
    /// TTL in seconds; inherits the response-cache TTL when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
    /// Live-rerun probability; inherits the response-cache rate when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bypass_rate: Option<f64>,
    /// Version string folded into cache keys for members of this class.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_version: Option<String>,
    /// Top-level argument keys dropped before keying.
    pub arg_skip: Vec<String>,
    /// Exact tool names, `prefix*`, `*suffix`, or `*contains*` patterns in this class.
    pub members: Vec<String>,
}

/// Per-tool refinement applied after class resolution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolOverride {
    /// Overrides the class cacheability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cacheable: Option<bool>,
    /// Overrides the class TTL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
    /// Overrides the class bypass rate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bypass_rate: Option<f64>,
    /// Version string folded into the cache key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_version: Option<String>,
    /// Replaces the class argument skip list when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arg_skip: Option<Vec<String>>,
}

/// Configuration for the LLM response and opt-in tool-result cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ResponseCacheConfig {
    /// How long a stored answer stays reusable, in seconds.
    pub ttl_seconds: u64,
    /// Required, non-empty trust-domain partition folded into every key.
    ///
    /// One response-cache namespace must not span mutually untrusted tenants
    /// or upstreams. The empty default is an unconfigured sentinel rejected
    /// by validation.
    pub namespace: String,
    /// Execution-intercept priority. Defaults to `150`.
    /// Lower values run first. Give guardrails a lower priority than the
    /// cache so guardrails run on cache hits.
    pub priority: i32,
    /// Probability in `[0.0, 1.0]` of skipping the cache and running live.
    pub bypass_rate: f64,
    /// Cache nondeterministic requests too. Set `false` to cache only
    /// requests explicitly pinned deterministic (`temperature` = 0) — absent
    /// or unreadable temperatures count as nondeterministic.
    pub cache_nondeterministic: bool,
    /// Typed key-derivation strategy.
    pub key_strategy: ResponseCacheKeyStrategy,
    /// Request headers (case-insensitive) folded into the key; never auth headers.
    pub header_allowlist: Vec<String>,
    /// Storage backend selection.
    pub backend: BackendConfig,
    /// Opt-in tool-result cache configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolCacheConfig>,
}

impl Default for ResponseCacheConfig {
    fn default() -> Self {
        Self {
            ttl_seconds: 3600,
            namespace: String::new(),
            priority: 150,
            bypass_rate: 0.0,
            cache_nondeterministic: false,
            key_strategy: ResponseCacheKeyStrategy::ExactRequest,
            header_allowlist: Vec::new(),
            backend: BackendConfig::default(),
            tools: None,
        }
    }
}

impl ResponseCacheConfig {
    /// TTL as a [`std::time::Duration`].
    pub fn ttl(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.ttl_seconds)
    }
}

#[cfg(test)]
#[path = "../tests/unit/config_tests.rs"]
mod tests;
