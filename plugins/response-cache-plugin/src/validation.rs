// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Semantic validation of the response-cache configuration.
//!
//! `config.schema.json` rejects unknown fields and wrong types; these checks
//! cover value ranges and cross-field rules the schema cannot express.

use std::collections::HashMap;

use nemo_relay_plugin::{ConfigDiagnostic, DiagnosticLevel};
use serde_json::Value as Json;

use crate::config::{ResponseCacheConfig, ResponseCacheKeyStrategy, ToolCacheConfig, ToolClass};
use crate::tool::{is_supported_tool_pattern, wildcard_patterns_overlap};

/// Validates a response-cache component configuration.
///
/// These are hard errors (not policy-driven): an invalid response-cache config
/// fails component activation rather than silently disabling the cache,
/// so a misconfiguration is caught at startup instead of producing surprising
/// runtime behavior. The raw object tells explicit settings from defaults
/// where record/replay forbids them.
pub(crate) fn validate_response_cache_with_raw(
    report: &mut Vec<ConfigDiagnostic>,
    config: &ResponseCacheConfig,
    raw: Option<&serde_json::Map<String, Json>>,
) {
    if config.namespace.trim().is_empty() {
        report.push(response_cache_error(
            "response_cache.missing_namespace",
            Some("namespace"),
            "namespace must identify one trusted cache-sharing domain; do not enable one \
             response cache across mutually untrusted tenants or upstreams"
                .to_string(),
        ));
    }
    if config.ttl_seconds == 0 {
        report.push(response_cache_error(
            "response_cache.invalid_ttl",
            Some("ttl_seconds"),
            "ttl_seconds must be greater than 0".to_string(),
        ));
    }
    if config.record_replay.is_none() && !(0.0..=1.0).contains(&config.bypass_rate) {
        report.push(response_cache_error(
            "response_cache.invalid_bypass_rate",
            Some("bypass_rate"),
            "bypass_rate must be in [0.0, 1.0]".to_string(),
        ));
    }
    if matches!(config.key_strategy, ResponseCacheKeyStrategy::Unknown(_)) {
        report.push(response_cache_error(
            "response_cache.unsupported_key_strategy",
            Some("key_strategy"),
            format!(
                "unsupported key_strategy '{}'; supported: \"exact_request\", \"logical\"",
                config.key_strategy.as_str()
            ),
        ));
    }
    if let Some(record_replay) = &config.record_replay {
        validate_record_replay(report, config, record_replay, raw);
        return;
    }
    // Auth material must never enter the key or the stored entries.
    const AUTH_HEADERS: &[&str] = &[
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "anthropic-api-key",
        "x-goog-api-key",
    ];
    for name in &config.header_allowlist {
        if AUTH_HEADERS.contains(&name.to_ascii_lowercase().as_str()) {
            report.push(response_cache_error(
                "response_cache.auth_header_allowlisted",
                Some("header_allowlist"),
                format!("'{name}' is an auth header and must never be folded into cache keys"),
            ));
        }
    }
    match config.backend.kind.as_str() {
        "in_memory" => {
            // Without this check, a mistyped budget silently falls back to the
            // default and `max_bytes: 0` disables storage; reject both.
            if let Some(value) = config.backend.config.get("max_bytes")
                && value.as_u64().is_none_or(|value| value == 0)
            {
                report.push(response_cache_error(
                    "response_cache.invalid_backend_option",
                    Some("backend.config"),
                    "max_bytes must be a positive integer".to_string(),
                ));
            }
        }
        "redis" => {
            if config
                .backend
                .config
                .get("url")
                .and_then(Json::as_str)
                .is_none_or(|url| url.trim().is_empty())
            {
                report.push(response_cache_error(
                    "response_cache.missing_redis_url",
                    Some("backend.config.url"),
                    "redis backend requires backend.config.url".to_string(),
                ));
            }
            // A mistyped key_prefix would silently fall back to the shared
            // default prefix; reject it like a mistyped max_bytes.
            if let Some(value) = config.backend.config.get("key_prefix")
                && !value.is_string()
            {
                report.push(response_cache_error(
                    "response_cache.invalid_backend_option",
                    Some("backend.config"),
                    "key_prefix must be a string".to_string(),
                ));
            }
        }
        other => report.push(response_cache_error(
            "response_cache.unknown_backend",
            Some("backend.kind"),
            format!("unknown backend kind '{other}'"),
        )),
    }

    if let Some(tools) = &config.tools {
        validate_tool_cache(report, tools);
    }
}

fn validate_tool_cache(report: &mut Vec<ConfigDiagnostic>, tools: &ToolCacheConfig) {
    validate_tool_policy(
        report,
        "default",
        tools.default.ttl_seconds,
        tools.default.bypass_rate,
    );
    if tools.default.cacheable {
        report.push(response_cache_error(
            "response_cache.tool_cacheable_default",
            Some("tools.default.cacheable"),
            "tools.default.cacheable must be false; use a named class or override to classify cacheable tools"
                .to_string(),
        ));
    }
    if !tools.default.members.is_empty() {
        report.push(response_cache_error(
            "response_cache.tool_default_members",
            Some("tools.default"),
            "tools.default.members is never matched (the default bucket applies to every \
             unclassified tool); move these names into a named class"
                .to_string(),
        ));
    }

    let mut owning_class: HashMap<&str, &str> = HashMap::new();
    for (class_name, class) in &tools.classes {
        validate_tool_cache_class(report, class_name, class, &mut owning_class);
    }

    validate_conflicting_tool_class_patterns(report, tools);

    for (tool_name, over) in &tools.overrides {
        if !is_supported_tool_pattern(tool_name) {
            report.push(response_cache_error(
                "response_cache.tool_invalid_pattern",
                Some("tools.overrides"),
                format!(
                    "tool override '{tool_name}' must be an exact name, 'prefix*', '*suffix', or \
                     '*contains*'"
                ),
            ));
        }
        validate_tool_policy(
            report,
            &format!("overrides.{tool_name}"),
            over.ttl_seconds,
            over.bypass_rate,
        );
        if over.cacheable == Some(true) && tool_name == "*" {
            report.push(response_cache_error(
                "response_cache.tool_catch_all_override",
                Some("tools.overrides"),
                format!(
                    "override '{tool_name}' sets cacheable = true for every tool; use a \
                     named override to classify cacheable tools"
                ),
            ));
        }
    }

    validate_conflicting_tool_override_patterns(report, tools);
}

fn validate_tool_cache_class<'a>(
    report: &mut Vec<ConfigDiagnostic>,
    class_name: &'a str,
    class: &'a ToolClass,
    owning_class: &mut HashMap<&'a str, &'a str>,
) {
    validate_tool_policy(
        report,
        &format!("classes.{class_name}"),
        class.ttl_seconds,
        class.bypass_rate,
    );
    for member in &class.members {
        if !is_supported_tool_pattern(member) {
            report.push(response_cache_error(
                "response_cache.tool_invalid_pattern",
                Some("tools.classes"),
                format!(
                    "tool member '{member}' in class '{class_name}' must be an exact name, \
                     'prefix*', '*suffix', or '*contains*'"
                ),
            ));
        }
        if let Some(previous) = owning_class.get(member.as_str()) {
            if *previous != class_name {
                report.push(response_cache_error(
                    "response_cache.tool_multiple_classes",
                    Some("tools.classes"),
                    format!(
                        "tool member '{member}' appears in multiple classes ('{previous}' and \
                         '{class_name}'); a member — exact name or pattern — may appear in at \
                         most one class"
                    ),
                ));
            }
        } else {
            owning_class.insert(member.as_str(), class_name);
        }
        if class.cacheable && member == "*" {
            report.push(response_cache_error(
                "response_cache.tool_catch_all_member",
                Some("tools.classes"),
                format!(
                    "class '{class_name}' lists the catch-all member '{member}' with \
                     cacheable = true, which caches every tool; use a named class to \
                     classify cacheable tools"
                ),
            ));
        }
    }
}

fn validate_conflicting_tool_class_patterns(
    report: &mut Vec<ConfigDiagnostic>,
    tools: &ToolCacheConfig,
) {
    for (index, (left_name, left_class)) in tools.classes.iter().enumerate() {
        for (right_name, right_class) in tools.classes.iter().skip(index + 1) {
            if left_class.cacheable == right_class.cacheable {
                continue;
            }
            let conflict = left_class.members.iter().any(|left_member| {
                left_member.contains('*')
                    && right_class.members.iter().any(|right_member| {
                        left_member != right_member
                            && right_member.contains('*')
                            && wildcard_patterns_overlap(left_member, right_member)
                    })
            });
            if conflict {
                report.push(response_cache_error(
                    "response_cache.tool_conflicting_classes",
                    Some("tools.classes"),
                    format!(
                        "classes '{left_name}' and '{right_name}' contain overlapping wildcard \
                         members with conflicting cacheable settings; split the patterns so one \
                         policy applies to every tool"
                    ),
                ));
            }
        }
    }
}

fn validate_conflicting_tool_override_patterns(
    report: &mut Vec<ConfigDiagnostic>,
    tools: &ToolCacheConfig,
) {
    for (index, (left_name, left_override)) in tools.overrides.iter().enumerate() {
        for (right_name, right_override) in tools.overrides.iter().skip(index + 1) {
            // An omitted value inherits from whichever class wins for the
            // concrete tool name. Because overlapping patterns can select
            // different classes, only identical declarations are safe.
            let conflicting_cacheability = left_override.cacheable != right_override.cacheable;
            if !conflicting_cacheability
                || !left_name.contains('*')
                || !right_name.contains('*')
                || !wildcard_patterns_overlap(left_name, right_name)
            {
                continue;
            }
            report.push(response_cache_error(
                "response_cache.tool_conflicting_overrides",
                Some("tools.overrides"),
                format!(
                    "overrides '{left_name}' and '{right_name}' overlap with conflicting \
                     cacheable settings; split the patterns so one policy applies to every tool"
                ),
            ));
        }
    }
}

fn validate_tool_policy(
    report: &mut Vec<ConfigDiagnostic>,
    location: &str,
    ttl_seconds: Option<u64>,
    bypass_rate: Option<f64>,
) {
    if ttl_seconds == Some(0) {
        report.push(response_cache_error(
            "response_cache.tool_invalid_ttl",
            Some("tools"),
            format!("tools.{location}.ttl_seconds must be greater than 0 when set"),
        ));
    }
    if let Some(rate) = bypass_rate
        && !(0.0..=1.0).contains(&rate)
    {
        report.push(response_cache_error(
            "response_cache.tool_invalid_bypass_rate",
            Some("tools"),
            format!("tools.{location}.bypass_rate must be in [0.0, 1.0] when set"),
        ));
    }
}

fn response_cache_error(code: &str, field: Option<&str>, message: String) -> ConfigDiagnostic {
    response_cache_diag(DiagnosticLevel::Error, code, field, message)
}

fn response_cache_diag(
    level: DiagnosticLevel,
    code: &str,
    field: Option<&str>,
    message: String,
) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level,
        code: code.to_string(),
        component: Some("response_cache".to_string()),
        field: field.map(str::to_string),
        message,
    }
}

/// Record/replay warnings. They never fail activation: an invalid setup
/// installs fail-closed middleware that reports its reason on every call.
fn validate_record_replay(
    report: &mut Vec<ConfigDiagnostic>,
    config: &ResponseCacheConfig,
    record_replay: &Json,
    raw: Option<&serde_json::Map<String, Json>>,
) {
    use crate::record_replay::config::{diagnostic, parse_record_replay};
    if !cfg!(unix) {
        report.push(diagnostic(
            "platform_unsupported",
            "record_replay",
            "record/replay currently requires a Unix platform",
        ));
    }
    report.extend(parse_record_replay(record_replay).diagnostics);
    let explicit = |field: &str| raw.is_some_and(|raw| raw.contains_key(field));
    for (field, forbidden) in [
        ("tools", config.tools.is_some()),
        ("bypass_rate", config.bypass_rate != 0.0),
        ("header_allowlist", !config.header_allowlist.is_empty()),
        ("backend", config.backend.kind != "in_memory"),
        ("priority", explicit("priority")),
    ] {
        if forbidden {
            report.push(diagnostic(
                "forbidden_cache_setting",
                field,
                "cache setting is incompatible with record/replay",
            ));
        }
    }
}
