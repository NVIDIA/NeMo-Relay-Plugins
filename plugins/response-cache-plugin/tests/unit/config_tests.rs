// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for response-cache configuration parsing and defaults.

use serde_json::json;

use super::*;

#[test]
fn defaults_match_the_former_adaptive_section() {
    let config = ResponseCacheConfig::default();
    assert_eq!(config.ttl_seconds, 3600);
    assert_eq!(config.priority, 150);
    assert!(!config.cache_nondeterministic);
    assert_eq!(config.key_strategy, ResponseCacheKeyStrategy::ExactRequest);
    assert_eq!(config.backend.kind, "in_memory");
    assert!(config.tools.is_none());

    let tools = ToolCacheConfig::default();
    assert!(!tools.enabled);
    assert!(!tools.cache_errors);
    assert_eq!(tools.priority, 150);
}

#[test]
fn tool_cache_deserializes_explicit_error_caching_opt_in() {
    let tools: ToolCacheConfig = serde_json::from_value(json!({
        "enabled": true,
        "cache_errors": true,
    }))
    .unwrap();
    assert!(tools.enabled);
    assert!(tools.cache_errors);
}

#[test]
fn key_strategy_roundtrips_and_preserves_unknown_values() {
    let logical: ResponseCacheConfig = serde_json::from_value(json!({
        "key_strategy": "logical"
    }))
    .unwrap();
    assert_eq!(logical.key_strategy, ResponseCacheKeyStrategy::Logical);
    assert_eq!(
        serde_json::to_value(logical).unwrap()["key_strategy"],
        json!("logical")
    );

    let unsupported: ResponseCacheConfig = serde_json::from_value(json!({
        "key_strategy": "future"
    }))
    .unwrap();
    assert_eq!(
        unsupported.key_strategy,
        ResponseCacheKeyStrategy::Unknown("future".to_string())
    );
}

#[test]
fn component_config_ignores_the_sdk_executor_section() {
    let config = crate::parse_config(
        json!({"namespace": "team", "executor": {"worker_threads": 4}})
            .as_object()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(config.namespace, "team");
    assert!(crate::validate_config(json!({"namespace": "team"}).as_object().unwrap()).is_empty());
}
