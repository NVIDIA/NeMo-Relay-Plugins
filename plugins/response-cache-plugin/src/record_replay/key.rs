// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::config::KeyFiltersConfig;
use crate::config::ResponseCacheConfig;
use crate::key::{fingerprint, record_replay_document};
use nemo_relay_types::api::llm::LlmRequest;
use regex::{Regex, RegexBuilder};
use serde_json::{Value as Json, json};
use std::collections::BTreeMap;

pub const REDACTED: &str = "<relay-redacted>";
pub const TOOL_RESULT: &str = "<relay-tool-result>";

pub struct KeyBuilder {
    filters: KeyFiltersConfig,
    patterns: Vec<Regex>,
    policy: Json,
    pub policy_hash: String,
}
pub struct KeyedRequest {
    pub hash: Option<String>,
    pub reason: Option<String>,
    pub filtered: Json,
    pub matches: BTreeMap<String, u64>,
    pub document: Option<Json>,
}

impl KeyBuilder {
    pub fn new(cache: &ResponseCacheConfig, filters: KeyFiltersConfig) -> Self {
        let patterns = filters
            .redact_lines
            .iter()
            .map(|p| {
                RegexBuilder::new(p)
                    .multi_line(true)
                    .build()
                    .expect("validated regex")
            })
            .collect();
        let policy = json!({"filter_version":1, "key_filters":filters, "redacted_placeholder":REDACTED,
            "tool_result_placeholder":TOOL_RESULT, "namespace":cache.namespace, "key_strategy":cache.key_strategy,
            "cache_schema_version":crate::store::CACHE_SCHEMA_VERSION});
        let policy_hash = fingerprint(&policy).expect("validated key policy");
        Self {
            filters,
            patterns,
            policy,
            policy_hash,
        }
    }
    pub fn policy(&self) -> &Json {
        &self.policy
    }
    pub fn build(
        &self,
        operation: &str,
        request: &LlmRequest,
        cache: &ResponseCacheConfig,
    ) -> KeyedRequest {
        let mut filtered = request.content.clone();
        let mut matches = BTreeMap::new();
        for pointer in &self.filters.drop {
            matches.insert(
                format!("drop:{pointer}"),
                u64::from(remove(&mut filtered, pointer)),
            );
        }
        if let Some(body) = filtered.as_object_mut() {
            body.remove("extra_headers");
            body.remove("headers");
        }
        for i in 0..self.patterns.len() {
            matches.insert(format!("redact_lines:{i}"), 0);
        }
        for i in 0..self.filters.redact_blocks.len() {
            matches.insert(format!("redact_blocks:{i}"), 0);
        }
        let mut reason = None;
        self.filter_messages(&mut filtered, &mut matches, &mut reason);
        if let Some(instructions) = filtered.get_mut("instructions") {
            self.redact_content(instructions, &mut matches, &mut reason);
        }
        let document = if reason.is_none() {
            record_replay_document(
                operation,
                &LlmRequest {
                    content: filtered.clone(),
                    ..request.clone()
                },
                cache,
                &json!(self.filters),
            )
        } else {
            None
        };
        let hash = document.as_ref().and_then(fingerprint);
        if hash.is_none() && reason.is_none() {
            reason = Some(
                if filtered.is_null() {
                    "unparseable_body"
                } else if document.is_none() {
                    "unrepresentable_number"
                } else {
                    "canonicalization_failed"
                }
                .into(),
            );
        }
        KeyedRequest {
            hash,
            reason,
            filtered,
            matches,
            document,
        }
    }
    fn filter_messages(
        &self,
        filtered: &mut Json,
        matches: &mut BTreeMap<String, u64>,
        reason: &mut Option<String>,
    ) {
        if let Some(input) = filtered.get_mut("input").filter(|input| input.is_string()) {
            self.redact_content(input, matches, reason);
        }
        for key in ["messages", "input"] {
            if let Some(messages) = filtered.get_mut(key).and_then(Json::as_array_mut) {
                for message in messages {
                    if let Some(content) = message.get_mut("content") {
                        self.redact_content(content, matches, reason);
                    }
                    if self.filters.tool_results == "placeholder" {
                        if message.get("role").and_then(Json::as_str) == Some("tool") {
                            message["content"] = json!(TOOL_RESULT);
                        }
                        if message.get("type").and_then(Json::as_str)
                            == Some("function_call_output")
                        {
                            message["output"] = json!(TOOL_RESULT);
                        }
                        if let Some(parts) = message.get_mut("content").and_then(Json::as_array_mut)
                        {
                            for part in parts {
                                if part.get("type").and_then(Json::as_str) == Some("tool_result") {
                                    part["content"] = json!(TOOL_RESULT);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    fn redact_content(
        &self,
        value: &mut Json,
        counts: &mut BTreeMap<String, u64>,
        reason: &mut Option<String>,
    ) {
        match value {
            Json::String(text) => {
                for (i, block) in self.filters.redact_blocks.iter().enumerate() {
                    let mut result = String::new();
                    let mut rest = text.as_str();
                    let mut count = 0;
                    while let Some(start) = rest.find(&block.start) {
                        result.push_str(&rest[..start]);
                        let tail = &rest[start + block.start.len()..];
                        let Some(end) = tail.find(&block.end) else {
                            // Redact through the end of the text, so an unclosed
                            // block never reaches the trace; the request stays
                            // keyless.
                            *reason = Some("unterminated_block".into());
                            result.push_str(REDACTED);
                            count += 1;
                            rest = "";
                            break;
                        };
                        result.push_str(REDACTED);
                        count += 1;
                        rest = &tail[end + block.end.len()..];
                    }
                    result.push_str(rest);
                    *text = result;
                    *counts.entry(format!("redact_blocks:{i}")).or_default() += count;
                }
                for (i, regex) in self.patterns.iter().enumerate() {
                    let count = regex.find_iter(text).count() as u64;
                    *text = regex.replace_all(text, REDACTED).into_owned();
                    *counts.entry(format!("redact_lines:{i}")).or_default() += count;
                }
            }
            Json::Array(parts) => {
                for part in parts {
                    if let Some(text) = part.get_mut("text") {
                        self.redact_content(text, counts, reason);
                    }
                }
            }
            _ => {}
        }
    }
}

fn remove(value: &mut Json, pointer: &str) -> bool {
    if pointer.is_empty() {
        let present = !value.is_null();
        *value = Json::Null;
        return present;
    }
    let Some((parent, leaf)) = pointer.rsplit_once('/') else {
        return false;
    };
    let key = leaf.replace("~1", "/").replace("~0", "~");
    match value.pointer_mut(parent) {
        Some(Json::Object(map)) => map.remove(&key).is_some(),
        Some(Json::Array(items)) => match key.parse::<usize>() {
            Ok(i) if i < items.len() && key == i.to_string() => {
                items.remove(i);
                true
            }
            _ => false,
        },
        _ => false,
    }
}

pub fn first_difference(a: &Json, b: &Json) -> Option<Json> {
    fn walk(a: &Json, b: &Json, path: String) -> Option<Json> {
        if a == b {
            return None;
        }
        if let (Some(a), Some(b)) = (a.as_object(), b.as_object()) {
            let mut keys: Vec<_> = a.keys().chain(b.keys()).collect();
            keys.sort_by_cached_key(|key| key.encode_utf16().collect::<Vec<_>>());
            keys.dedup();
            for key in keys {
                let escaped = key.replace('~', "~0").replace('/', "~1");
                match (a.get(key), b.get(key)) {
                    (Some(a), Some(b)) => {
                        if let Some(diff) = walk(a, b, format!("{path}/{escaped}")) {
                            return Some(diff);
                        }
                    }
                    _ => {
                        return Some(
                            json!({"pointer":format!("{path}/{escaped}"),"recorded":a.get(key),"live":b.get(key)}),
                        );
                    }
                }
            }
        } else if let (Some(a), Some(b)) = (a.as_array(), b.as_array()) {
            for i in 0..a.len().max(b.len()) {
                match (a.get(i), b.get(i)) {
                    (Some(a), Some(b)) => {
                        if let Some(diff) = walk(a, b, format!("{path}/{i}")) {
                            return Some(diff);
                        }
                    }
                    _ => {
                        return Some(
                            json!({"pointer":format!("{path}/{i}"),"recorded":a.get(i),"live":b.get(i)}),
                        );
                    }
                }
            }
        } else if let (Some(a), Some(b)) = (a.as_str(), b.as_str()) {
            let offset = a
                .chars()
                .zip(b.chars())
                .position(|(a, b)| a != b)
                .unwrap_or(a.chars().count().min(b.chars().count()));
            return Some(
                json!({"pointer":path,"character_offset":offset,"recorded_length":a.chars().count(),"live_length":b.chars().count(),
                "recorded":a.chars().skip(offset.saturating_sub(80)).take(200).collect::<String>(),
                "live":b.chars().skip(offset.saturating_sub(80)).take(200).collect::<String>()}),
            );
        }
        fn bounded(v: &Json) -> Json {
            let text = v.to_string();
            if text.len() > 4096 {
                json!({"truncated":text.chars().take(200).collect::<String>(),"sha256":fingerprint(v)})
            } else {
                v.clone()
            }
        }
        Some(json!({"pointer":path,"recorded":bounded(a),"live":bounded(b)}))
    }
    walk(a, b, String::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn drift_matches_but_tool_arguments_and_prompts_do_not() {
        let cache = ResponseCacheConfig {
            namespace: "test".into(),
            ..Default::default()
        };
        let key = KeyBuilder::new(
            &cache,
            KeyFiltersConfig {
                tool_results: "placeholder".into(),
                redact_lines: vec!["^Time: .*$".into()],
                ..Default::default()
            },
        );
        let a = LlmRequest {
            content: json!({"model":"m","temperature":1,"messages":[{"role":"system","content":"Time: a\nStable"},{"role":"tool","tool_call_id":"a","content":"output A"}]}),
            headers: Default::default(),
        };
        let mut b = a.clone();
        b.content["messages"][0]["content"] = json!("Time: b\nStable");
        b.content["messages"][1]["content"] = json!("output B");
        assert_eq!(
            key.build("op", &a, &cache).hash,
            key.build("op", &b, &cache).hash
        );
        b.content["messages"][0]["content"] = json!("Different");
        assert_ne!(
            key.build("op", &a, &cache).hash,
            key.build("op", &b, &cache).hash
        );
        assert_eq!(
            first_difference(&json!({"x":"abc"}), &json!({"x":"abd"})).unwrap()["character_offset"],
            2
        );
        let input = |time: &str| LlmRequest {
            content: json!({"model":"m","input":format!("Time: {time}\nStable")}),
            headers: Default::default(),
        };
        assert_eq!(
            key.build("op", &input("a"), &cache).hash,
            key.build("op", &input("b"), &cache).hash
        );
    }
}
