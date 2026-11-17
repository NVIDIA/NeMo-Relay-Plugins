// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Helpers shared by the response-cache integration test binaries. Included
//! via `#[path]` from each `[[test]]` target, so this file is not a test
//! target of its own.

use nemo_relay_response_cache_plugin::ResponseCacheConfig;
use nemo_relay_runtime::api::llm::{LlmCallExecuteParams, LlmRequest, llm_call_execute};
use nemo_relay_runtime::api::runtime::LlmExecutionNextFn;
use serde_json::{Value as Json, json};

use crate::plugin_host_test_support::activate_cache_report;

pub fn chat_request(prompt: &str) -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }),
    }
}

/// Loads the built plugin library and activates it with `config`.
pub async fn activate_cache(config: ResponseCacheConfig) {
    let report = activate_cache_report(config).await.unwrap();
    assert!(
        report.diagnostics.is_empty(),
        "unexpected diagnostics: {:?}",
        report.diagnostics
    );
}

#[allow(dead_code)]
pub async fn call(provider: &LlmExecutionNextFn, request: LlmRequest) -> Json {
    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(request)
            .func(provider.clone())
            .model_name("gpt-4o")
            .build(),
    )
    .await
    .unwrap()
}
