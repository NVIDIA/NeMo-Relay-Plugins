// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Record and replay through the built plugin library in a Relay host.

#![cfg(unix)]

#[path = "../support/plugin_host.rs"]
mod plugin_host_test_support;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use nemo_relay_runtime::api::llm::{
    LlmCallExecuteParams, LlmRequest, LlmStreamCallExecuteParams, llm_call_execute,
    llm_stream_call_execute,
};
use nemo_relay_runtime::api::registry::{
    deregister_llm_execution_intercept, register_llm_execution_intercept,
};
use nemo_relay_runtime::api::runtime::{
    LlmExecutionFn, LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn,
};
use serde_json::{Map, Value as Json, json};
use tempfile::TempDir;
use tokio_stream::StreamExt;

use plugin_host_test_support::{activate_plugin, test_close_plugin_host};

fn config(mode: &str, trace: &Path, output: &Path) -> Map<String, Json> {
    json!({
        "namespace": "record-replay-integration",
        "record_replay": {
            "mode": mode,
            "trace_directory": trace,
            "output_directory": output,
            "latency_scale": 0
        }
    })
    .as_object()
    .unwrap()
    .clone()
}

fn request(text: &str) -> LlmRequest {
    LlmRequest {
        headers: Default::default(),
        content: json!({"model": "m", "messages": [{"role": "user", "content": text}]}),
    }
}

fn frames() -> Vec<Json> {
    vec![
        json!({"id": "r", "object": "chat.completion.chunk", "model": "m",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "a"}, "finish_reason": null}]}),
        json!({"id": "r", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "b"}, "finish_reason": "stop"}]}),
    ]
}

fn stream_provider(calls: Arc<AtomicUsize>) -> LlmStreamExecutionNextFn {
    Arc::new(move |_| {
        calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Ok(LlmJsonStream::new(tokio_stream::iter(
                frames().into_iter().map(Ok),
            )))
        })
    })
}

fn provider(calls: Arc<AtomicUsize>) -> LlmExecutionNextFn {
    Arc::new(move |_| {
        calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Ok(json!({"id": "b", "choices": [{"message": {"content": "buffered"}}]}))
        })
    })
}

/// Reads a stream call to its end.
async fn stream_call(provider: &LlmStreamExecutionNextFn, text: &str) -> Vec<Json> {
    let params = LlmStreamCallExecuteParams::builder()
        .name("openai")
        .request(request(text))
        .func(provider.clone())
        .collector(Box::new(|_chunk| Ok(())))
        .finalizer(Box::new(|| json!({"done": true})))
        .model_name("m")
        .build();
    let mut stream = llm_stream_call_execute(params).await.unwrap();
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.unwrap());
    }
    chunks
}

async fn buffered_call(provider: &LlmExecutionNextFn, text: &str) -> Result<Json, String> {
    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(request(text))
            .func(provider.clone())
            .model_name("m")
            .build(),
    )
    .await
    .map_err(|error| error.to_string())
}

fn json_lines(path: &Path) -> Vec<Json> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

/// Copies the finished trace without the recorder's lock file, as a launcher does.
fn finalized_copy(trace: &Path, root: &Path) -> PathBuf {
    let copy = root.join("replay-trace");
    fs::create_dir_all(&copy).unwrap();
    for file in ["trace.json", "calls.jsonl", "requests.jsonl", "tools.jsonl"] {
        fs::copy(trace.join(file), copy.join(file)).unwrap();
    }
    copy
}

#[tokio::test]
async fn plugin_records_and_replays_without_the_provider() {
    let root = TempDir::new().unwrap();
    let trace = root.path().join("trace");
    fs::create_dir_all(&trace).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let streaming = stream_provider(Arc::clone(&calls));
    let buffered = provider(Arc::clone(&calls));

    let report = activate_plugin(
        config("record", &trace, &root.path().join("record")),
        Vec::new(),
    )
    .await
    .unwrap();
    assert!(!report.has_errors(), "{:?}", report.diagnostics);
    let recorded = stream_call(&streaming, "stream").await;
    assert_eq!(recorded, frames());
    let answer = buffered_call(&buffered, "buffered").await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    // Closing the component closes the session and finishes the trace.
    test_close_plugin_host().unwrap();
    let lines = json_lines(&trace.join("calls.jsonl"));
    let stream_line = lines
        .iter()
        .find(|line| line["stream"] == json!(true))
        .expect("the stream call is recorded");
    assert_eq!(stream_line["outcome"], json!("ok"));

    // Replay serves both answers from the trace and never calls the provider.
    let replay_trace = finalized_copy(&trace, root.path());
    let report = activate_plugin(
        config("replay", &replay_trace, &root.path().join("replay")),
        Vec::new(),
    )
    .await
    .unwrap();
    assert!(!report.has_errors(), "{:?}", report.diagnostics);
    assert_eq!(stream_call(&streaming, "stream").await, frames());
    assert_eq!(buffered_call(&buffered, "buffered").await.unwrap(), answer);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    // An unrecorded request fails closed instead of reaching the provider.
    let miss = buffered_call(&buffered, "unrecorded").await.unwrap_err();
    assert!(miss.contains("RR-MISS"), "{miss}");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    test_close_plugin_host().unwrap();

    // An explicit priority is forbidden, so recording fails closed.
    let mut forbidden = config("record", &trace, &root.path().join("forbidden"));
    forbidden.insert("priority".into(), json!(7));
    activate_plugin(forbidden, Vec::new()).await.unwrap();
    let error = buffered_call(&buffered, "buffered").await.unwrap_err();
    assert!(error.contains("RR-UNAVAILABLE"), "{error}");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    test_close_plugin_host().unwrap();

    // Record/replay is the outermost intercept: an intercept that answers on
    // its own cannot get around a fail-closed replay.
    let cached: LlmExecutionFn =
        Arc::new(|_, _, _, _| Box::pin(async { Ok(json!({"cached": true})) }));
    register_llm_execution_intercept("earlier_answer", 150, cached).unwrap();
    activate_plugin(
        config(
            "replay",
            &root.path().join("missing"),
            &root.path().join("missing-out"),
        ),
        Vec::new(),
    )
    .await
    .unwrap();
    let error = buffered_call(&buffered, "buffered").await.unwrap_err();
    deregister_llm_execution_intercept("earlier_answer").unwrap();
    assert!(error.contains("RR-UNAVAILABLE"), "{error}");
    test_close_plugin_host().unwrap();
}
