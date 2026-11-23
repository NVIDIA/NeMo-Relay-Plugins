// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tolerant, warning-only parsing: a malformed mode must still intercept calls.

use std::collections::HashSet;
use std::path::PathBuf;

use nemo_relay_plugin::{ConfigDiagnostic, DiagnosticLevel};
use regex::RegexBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordReplayConfig {
    pub mode: String,
    pub trace_directory: PathBuf,
    pub output_directory: PathBuf,
    pub latency_scale: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_gate: Option<StartGateConfig>,
    pub key_filters: KeyFiltersConfig,
    pub divergence: DivergenceConfig,
}

impl Default for RecordReplayConfig {
    fn default() -> Self {
        Self {
            mode: "replay".into(),
            trace_directory: PathBuf::new(),
            output_directory: PathBuf::new(),
            latency_scale: 1.0,
            start_gate: None,
            key_filters: KeyFiltersConfig::default(),
            divergence: DivergenceConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartGateConfig {
    pub path: PathBuf,
    pub timeout_seconds: u64,
}
impl Default for StartGateConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            timeout_seconds: 3600,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RedactBlock {
    pub start: String,
    pub end: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyFiltersConfig {
    pub drop: Vec<String>,
    pub redact_lines: Vec<String>,
    pub redact_blocks: Vec<RedactBlock>,
    pub tool_results: String,
}
impl Default for KeyFiltersConfig {
    fn default() -> Self {
        Self {
            drop: vec![],
            redact_lines: vec![],
            redact_blocks: vec![],
            tool_results: "keep".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceConfig {
    pub tool_names: Vec<String>,
    pub tool_fields: Vec<String>,
    pub flag_fields: Vec<String>,
    pub output_size_tolerance: f64,
}
impl Default for DivergenceConfig {
    fn default() -> Self {
        Self {
            tool_names: vec![],
            tool_fields: vec!["/exit_code".into(), "/error".into()],
            flag_fields: vec!["/error".into()],
            output_size_tolerance: 0.5,
        }
    }
}

pub struct RecordReplayParse {
    pub config: Option<RecordReplayConfig>,
    pub output_directory: Option<PathBuf>,
    pub diagnostics: Vec<ConfigDiagnostic>,
}

pub fn diagnostic(code: &str, field: &str, message: &str) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level: DiagnosticLevel::Warning,
        code: format!("response_cache.record_replay_{code}"),
        component: Some("response_cache".into()),
        field: Some(field.into()),
        message: message.into(),
    }
}

pub fn valid_pointer(pointer: &str) -> bool {
    if !pointer.is_empty() && !pointer.starts_with('/') {
        return false;
    }
    let mut chars = pointer.chars();
    while let Some(c) = chars.next() {
        if c == '~' && !matches!(chars.next(), Some('0' | '1')) {
            return false;
        }
    }
    true
}

#[derive(Default)]
struct Parser {
    diagnostics: Vec<ConfigDiagnostic>,
}
impl Parser {
    fn error(&mut self, code: &str, field: &str, message: &str) {
        self.diagnostics.push(diagnostic(code, field, message));
    }
    fn table<'a>(
        &mut self,
        value: &'a Json,
        field: &str,
        keys: &[&str],
    ) -> Option<&'a Map<String, Json>> {
        let Some(map) = value.as_object() else {
            self.error("invalid_type", field, "expected a table");
            return None;
        };
        for key in map.keys() {
            if !keys.contains(&key.as_str()) {
                self.error(
                    "unknown_field",
                    &format!("{field}.{key}"),
                    "unknown record/replay field",
                );
            }
        }
        Some(map)
    }
    fn string(
        &mut self,
        map: &Map<String, Json>,
        key: &str,
        field: &str,
        required: bool,
        default: &str,
    ) -> String {
        match map.get(key) {
            None if required => {
                self.error("missing_field", field, "required field is missing");
                String::new()
            }
            None => default.into(),
            Some(Json::String(value)) => value.clone(),
            Some(_) => {
                self.error("invalid_type", field, "expected a string");
                String::new()
            }
        }
    }
    fn strings(
        &mut self,
        map: &Map<String, Json>,
        key: &str,
        field: &str,
        default: Vec<String>,
    ) -> Vec<String> {
        let Some(value) = map.get(key) else {
            return default;
        };
        let Some(items) = value.as_array() else {
            self.error("invalid_type", field, "expected a list");
            return vec![];
        };
        items
            .iter()
            .enumerate()
            .filter_map(|(i, value)| {
                if let Some(value) = value.as_str() {
                    Some(value.into())
                } else {
                    self.error(
                        "invalid_type",
                        &format!("{field}[{i}]"),
                        "expected a string",
                    );
                    None
                }
            })
            .collect()
    }
    fn number(&mut self, map: &Map<String, Json>, key: &str, field: &str, default: f64) -> f64 {
        match map.get(key) {
            None => default,
            Some(value) => match value.as_f64().filter(|v| v.is_finite()) {
                Some(value) => value,
                None => {
                    self.error("invalid_type", field, "expected a finite number");
                    default
                }
            },
        }
    }
    fn pointers(&mut self, values: &[String], field: &str, code: &str) {
        for (i, value) in values.iter().enumerate() {
            if !valid_pointer(value) {
                self.error(
                    code,
                    &format!("{field}[{i}]"),
                    "expected an RFC 6901 pointer",
                );
            }
        }
    }
}

pub fn parse_record_replay(value: &Json) -> RecordReplayParse {
    let mut p = Parser::default();
    let mut config = RecordReplayConfig::default();
    let output_directory = value
        .get("output_directory")
        .and_then(Json::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from);
    if let Some(map) = p.table(
        value,
        "record_replay",
        &[
            "mode",
            "trace_directory",
            "output_directory",
            "latency_scale",
            "start_gate",
            "key_filters",
            "divergence",
        ],
    ) {
        config.mode = p.string(map, "mode", "record_replay.mode", true, "");
        if map.get("mode").is_some_and(Json::is_string)
            && !["record", "verify", "replay"].contains(&config.mode.as_str())
        {
            p.error(
                "unsupported_mode",
                "record_replay.mode",
                "expected record, verify or replay",
            );
        }
        for key in ["trace_directory", "output_directory"] {
            let value = p.string(map, key, &format!("record_replay.{key}"), true, "");
            if map.get(key).is_some_and(Json::is_string) && value.trim().is_empty() {
                p.error(
                    "missing_field",
                    &format!("record_replay.{key}"),
                    "expected a non-empty path",
                );
            }
            if key == "trace_directory" {
                config.trace_directory = value.into();
            } else {
                config.output_directory = value.into();
            }
        }
        config.latency_scale = p.number(map, "latency_scale", "record_replay.latency_scale", 1.0);
        if !(0.0..=1000.0).contains(&config.latency_scale) {
            p.error(
                "invalid_latency_scale",
                "record_replay.latency_scale",
                "scale must be between 0 and 1000",
            );
        }
        parse_gate(&mut p, &mut config, map);
        parse_filters(&mut p, &mut config, map);
        parse_divergence(&mut p, &mut config, map);
    }
    RecordReplayParse {
        config: p.diagnostics.is_empty().then_some(config),
        output_directory,
        diagnostics: p.diagnostics,
    }
}

fn parse_gate(
    p: &mut Parser,
    config: &mut RecordReplayConfig,
    map: &serde_json::Map<String, Json>,
) {
    if let Some(value) = map.get("start_gate")
        && let Some(gate) = p.table(
            value,
            "record_replay.start_gate",
            &["path", "timeout_seconds"],
        )
    {
        let path = p.string(gate, "path", "record_replay.start_gate.path", true, "");
        let timeout = match gate.get("timeout_seconds") {
            None => Some(3600),
            Some(value) => {
                let timeout = value.as_u64();
                if timeout.is_none() {
                    p.error(
                        "invalid_type",
                        "record_replay.start_gate.timeout_seconds",
                        "expected a positive integer",
                    );
                }
                timeout
            }
        };
        // A missing path or a mistyped timeout already has its own code.
        let empty_path = gate.get("path").is_some_and(Json::is_string) && path.trim().is_empty();
        if empty_path || timeout.is_some_and(|t| !(1..=604800).contains(&t)) {
            p.error(
                "invalid_gate",
                "record_replay.start_gate",
                "gate requires a path and a timeout between 1 and 604800 seconds",
            );
        }
        config.start_gate = Some(StartGateConfig {
            path: path.into(),
            timeout_seconds: timeout.unwrap_or_default(),
        });
    }
}

fn parse_filters(
    p: &mut Parser,
    config: &mut RecordReplayConfig,
    map: &serde_json::Map<String, Json>,
) {
    if let Some(value) = map.get("key_filters")
        && let Some(filters) = p.table(
            value,
            "record_replay.key_filters",
            &["drop", "redact_lines", "redact_blocks", "tool_results"],
        )
    {
        config.key_filters.drop =
            p.strings(filters, "drop", "record_replay.key_filters.drop", vec![]);
        p.pointers(
            &config.key_filters.drop,
            "record_replay.key_filters.drop",
            "key_filters_invalid_pointer",
        );
        config.key_filters.redact_lines = p.strings(
            filters,
            "redact_lines",
            "record_replay.key_filters.redact_lines",
            vec![],
        );
        for (i, pattern) in config.key_filters.redact_lines.iter().enumerate() {
            if RegexBuilder::new(pattern).multi_line(true).build().is_err() {
                p.error(
                    "key_filters_invalid_regex",
                    &format!("record_replay.key_filters.redact_lines[{i}]"),
                    "regex does not compile",
                );
            }
        }
        config.key_filters.tool_results = p.string(
            filters,
            "tool_results",
            "record_replay.key_filters.tool_results",
            false,
            "keep",
        );
        if !["keep", "placeholder"].contains(&config.key_filters.tool_results.as_str()) {
            p.error(
                "unsupported_value",
                "record_replay.key_filters.tool_results",
                "expected keep or placeholder",
            );
        }
        if let Some(value) = filters.get("redact_blocks") {
            if let Some(blocks) = value.as_array() {
                for (i, value) in blocks.iter().enumerate() {
                    let field = format!("record_replay.key_filters.redact_blocks[{i}]");
                    if let Some(block) = p.table(value, &field, &["start", "end"]) {
                        let mut mistyped = false;
                        for key in ["start", "end"] {
                            if block.get(key).is_some_and(|v| !v.is_string()) {
                                mistyped = true;
                                p.error(
                                    "invalid_type",
                                    &format!("{field}.{key}"),
                                    "expected a string literal",
                                );
                            }
                        }
                        let start = block
                            .get("start")
                            .and_then(Json::as_str)
                            .unwrap_or_default();
                        let end = block.get("end").and_then(Json::as_str).unwrap_or_default();
                        if mistyped {
                        } else if start.is_empty() || end.is_empty() {
                            p.error(
                                "key_filters_invalid_block",
                                &field,
                                "block requires non-empty start and end literals",
                            );
                        } else {
                            config.key_filters.redact_blocks.push(RedactBlock {
                                start: start.into(),
                                end: end.into(),
                            });
                        }
                    }
                }
            } else {
                p.error(
                    "invalid_type",
                    "record_replay.key_filters.redact_blocks",
                    "expected a list",
                );
            }
        }
    }
}

fn parse_divergence(
    p: &mut Parser,
    config: &mut RecordReplayConfig,
    map: &serde_json::Map<String, Json>,
) {
    if let Some(value) = map.get("divergence")
        && let Some(div) = p.table(
            value,
            "record_replay.divergence",
            &[
                "tool_names",
                "tool_fields",
                "flag_fields",
                "output_size_tolerance",
            ],
        )
    {
        config.divergence.tool_names = p.strings(
            div,
            "tool_names",
            "record_replay.divergence.tool_names",
            vec![],
        );
        let mut seen = HashSet::new();
        for (i, name) in config.divergence.tool_names.iter().enumerate() {
            if name.is_empty() || !seen.insert(name) {
                p.error(
                    "invalid_divergence",
                    &format!("record_replay.divergence.tool_names[{i}]"),
                    "tool names must be non-empty and unique",
                );
            }
        }
        config.divergence.tool_fields = p.strings(
            div,
            "tool_fields",
            "record_replay.divergence.tool_fields",
            config.divergence.tool_fields.clone(),
        );
        config.divergence.flag_fields = p.strings(
            div,
            "flag_fields",
            "record_replay.divergence.flag_fields",
            config.divergence.flag_fields.clone(),
        );
        p.pointers(
            &config.divergence.tool_fields.clone(),
            "record_replay.divergence.tool_fields",
            "invalid_divergence",
        );
        p.pointers(
            &config.divergence.flag_fields.clone(),
            "record_replay.divergence.flag_fields",
            "invalid_divergence",
        );
        if config
            .divergence
            .flag_fields
            .iter()
            .any(|p| !config.divergence.tool_fields.contains(p))
        {
            p.error(
                "invalid_divergence",
                "record_replay.divergence.flag_fields",
                "flags must be a subset of outcome fields",
            );
        }
        config.divergence.output_size_tolerance = p.number(
            div,
            "output_size_tolerance",
            "record_replay.divergence.output_size_tolerance",
            0.5,
        );
        if config.divergence.output_size_tolerance < 0.0 {
            p.error(
                "invalid_divergence",
                "record_replay.divergence.output_size_tolerance",
                "tolerance must be non-negative",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn malformed_sections_are_warnings_and_preserve_output() {
        for value in [
            Json::Null,
            json!(false),
            json!([]),
            json!({"mode": "bogus", "trace_directory": 7, "output_directory": "/tmp/out", "latency_scale": null, "key_filters": {"drop": [3, "bad~2"], "redact_lines": ["("], "unknown": true}}),
        ] {
            let parsed = parse_record_replay(&value);
            assert!(parsed.config.is_none());
            assert!(!parsed.diagnostics.is_empty());
            assert!(
                parsed
                    .diagnostics
                    .iter()
                    .all(|d| d.level == DiagnosticLevel::Warning)
            );
        }
        let parsed = parse_record_replay(&json!({"output_directory":"/tmp/out"}));
        assert_eq!(parsed.output_directory, Some(PathBuf::from("/tmp/out")));
        assert_eq!(parsed.diagnostics.len(), 2);
    }
    #[test]
    fn each_problem_reports_its_table_code_once() {
        let base = json!({"mode":"replay","trace_directory":"/t","output_directory":"/o"});
        let cases = [
            (
                json!({"trace_directory":""}),
                "missing_field",
                "record_replay.trace_directory",
            ),
            (
                json!({"start_gate":{}}),
                "missing_field",
                "record_replay.start_gate.path",
            ),
            (
                json!({"start_gate":{"path":"/g","timeout_seconds":1.5}}),
                "invalid_type",
                "record_replay.start_gate.timeout_seconds",
            ),
            (
                json!({"start_gate":{"path":""}}),
                "invalid_gate",
                "record_replay.start_gate",
            ),
            (
                json!({"key_filters":{"redact_blocks":[{"start":5,"end":"e"}]}}),
                "invalid_type",
                "record_replay.key_filters.redact_blocks[0].start",
            ),
            (
                json!({"key_filters":{"redact_blocks":[{"start":"","end":"e"}]}}),
                "key_filters_invalid_block",
                "record_replay.key_filters.redact_blocks[0]",
            ),
            (
                json!({"divergence":{"tool_names":["a","a"]}}),
                "invalid_divergence",
                "record_replay.divergence.tool_names[1]",
            ),
        ];
        for (overlay, code, field) in cases {
            let mut value = base.clone();
            for (key, item) in overlay.as_object().unwrap() {
                value[key] = item.clone();
            }
            let parsed = parse_record_replay(&value);
            let found: Vec<_> = parsed
                .diagnostics
                .iter()
                .map(|d| (d.code.clone(), d.field.clone()))
                .collect();
            assert_eq!(
                found,
                [(
                    format!("response_cache.record_replay_{code}"),
                    Some(field.to_string())
                )],
                "{overlay}"
            );
        }
    }
    #[test]
    fn defaults_and_integer_scale_parse() {
        let parsed = parse_record_replay(
            &json!({"mode":"record", "trace_directory":"/tmp/trace", "output_directory":"/tmp/out", "latency_scale":0}),
        );
        assert!(parsed.diagnostics.is_empty());
        assert_eq!(parsed.config.unwrap().key_filters.tool_results, "keep");
        assert!(valid_pointer("/a~1b/~0"));
        assert!(!valid_pointer("/a~2b"));
    }
}
