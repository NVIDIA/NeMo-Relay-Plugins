// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::session::{Session, lock};
use super::trace::{Chunk, json_lines};
use super::*;
use crate::config::ResponseCacheConfig;
use nemo_relay_types::api::llm::LlmRequest;
use nemo_relay_types::api::runtime::{
    LlmExecutionContext, LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn,
    ToolExecutionContext, ToolExecutionNextFn,
};
use nemo_relay_types::api::tool::ToolExecutionResult;
use serde_json::{Value as Json, json};
use std::fs;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;
use tokio_stream::StreamExt;
use uuid::Uuid;

struct Dirs {
    root: PathBuf,
    trace: PathBuf,
    out: PathBuf,
}
impl Dirs {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("relay-rr-{}", Uuid::new_v4()));
        let trace = root.join("trace");
        let out = root.join("out");
        fs::create_dir_all(&trace).unwrap();
        Self { root, trace, out }
    }
    fn cache(&self, mode: &str) -> ResponseCacheConfig {
        ResponseCacheConfig {
            namespace: "rr-test".into(),
            record_replay: Some(
                json!({"mode":mode,"trace_directory":self.trace,"output_directory":self.out,"latency_scale":0}),
            ),
            ..Default::default()
        }
    }
    fn copy_finalized(&self) -> Self {
        let copy = Self::new();
        for file in ["trace.json", "calls.jsonl", "requests.jsonl", "tools.jsonl"] {
            fs::copy(self.trace.join(file), copy.trace.join(file)).unwrap();
        }
        copy
    }
    fn output(&self, prefix: &str) -> Vec<Json> {
        let mut result = vec![];
        if let Ok(files) = fs::read_dir(&self.out) {
            for file in files.flatten() {
                if file.file_name().to_string_lossy().starts_with(prefix) {
                    if file.path().extension().is_some_and(|e| e == "jsonl") {
                        result.extend(json_lines(&file.path()).unwrap());
                    } else {
                        result
                            .push(serde_json::from_slice(&fs::read(file.path()).unwrap()).unwrap());
                    }
                }
            }
        }
        result
    }
}
impl Drop for Dirs {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
fn request(text: &str) -> LlmRequest {
    LlmRequest {
        headers: Default::default(),
        content: json!({"model":"m","temperature":0.5,"messages":[{"role":"user","content":text}]}),
    }
}
fn provider(count: Arc<AtomicUsize>) -> LlmExecutionNextFn {
    Arc::new(move |_| {
        let n = count.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(json!({"n":n})) })
    })
}
fn frames() -> Vec<Json> {
    vec![
        json!({"id":"r","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"a","reasoning_content":"why","vendor":"x"},"finish_reason":null}]}),
        json!({"choices":[{"index":0,"delta":{"content":"b","reasoning_content":"?","vendor":"y"},"finish_reason":"stop"}]}),
        json!({"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3,"reasoning_tokens":1}}),
    ]
}
fn stream_provider(count: Arc<AtomicUsize>) -> LlmStreamExecutionNextFn {
    Arc::new(move |_| {
        count.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Ok(LlmJsonStream::new(tokio_stream::iter(
                frames().into_iter().map(Ok),
            )))
        })
    })
}
async fn unary(
    session: Arc<Session>,
    req: LlmRequest,
    next: LlmExecutionNextFn,
) -> nemo_relay_types::error::Result<Json> {
    stream::make_buffered(session)("openai", req, LlmExecutionContext::default(), next).await
}
async fn streaming(session: Arc<Session>, next: LlmStreamExecutionNextFn) -> LlmJsonStream {
    stream::make_stream(session)(
        "openai",
        request("stable"),
        LlmExecutionContext::default(),
        next,
    )
    .await
    .unwrap()
}
async fn recorded_buffered() -> Dirs {
    let dirs = Dirs::new();
    let session = Session::activate(dirs.cache("record"), false);
    unary(
        session.clone(),
        request("stable"),
        provider(Arc::new(AtomicUsize::new(0))),
    )
    .await
    .unwrap();
    session.close();
    dirs.copy_finalized()
}

#[tokio::test]
async fn repeated_answers_and_failures_never_use_replay_provider() {
    let dirs = Dirs::new();
    let session = Session::activate(dirs.cache("record"), false);
    let count = Arc::new(AtomicUsize::new(0));
    let mut req = request("stable");
    req.headers
        .insert("Authorization".into(), json!("sentinel-auth-secret"));
    req.content["extra_headers"] = json!({"api-key":"sentinel-auth-secret"});
    for n in 0..2 {
        assert_eq!(
            unary(session.clone(), req.clone(), provider(count.clone()))
                .await
                .unwrap()["n"],
            n
        );
    }
    session.close();
    assert_eq!(count.load(Ordering::SeqCst), 2);
    let status = dirs.output("record_status").pop().unwrap();
    assert_eq!(status["lines"]["calls"], 2);
    assert_eq!(status["model_calls"], 2);
    assert_eq!(status["io_errors"], 0);
    for file in ["trace.json", "calls.jsonl", "requests.jsonl", "tools.jsonl"] {
        assert!(
            !fs::read_to_string(dirs.trace.join(file))
                .unwrap()
                .contains("sentinel-auth-secret")
        );
    }
    let replay = dirs.copy_finalized();
    let session = Session::activate(replay.cache("replay"), false);
    count.store(0, Ordering::SeqCst);
    for n in 0..2 {
        assert_eq!(
            unary(session.clone(), req.clone(), provider(count.clone()))
                .await
                .unwrap()["n"],
            n
        );
    }
    assert!(
        unary(session.clone(), req, provider(count.clone()))
            .await
            .unwrap_err()
            .to_string()
            .contains("RR-EXHAUSTED")
    );
    assert!(
        unary(session, request("different"), provider(count.clone()))
            .await
            .unwrap_err()
            .to_string()
            .contains("RR-MISS")
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert!(
        replay
            .output("divergence")
            .iter()
            .any(|d| d["type"] == "RR-MISS" && d["detail"]["recorded"].is_null())
    );
    let first_miss = dirs.copy_finalized();
    let session = Session::activate(first_miss.cache("replay"), false);
    assert!(
        unary(session, request("different"), provider(count))
            .await
            .is_err()
    );
    assert!(first_miss.output("divergence").iter().any(|d| {
        d["detail"]["pointer"]
            .as_str()
            .is_some_and(|p| p.starts_with("/body"))
    }));
}

#[tokio::test]
async fn malformed_section_and_missing_trace_are_fail_closed() {
    for raw in [
        Json::Null,
        json!(true),
        json!({"mode":"replay","trace_directory":"missing","output_directory":7}),
        json!({"mode":"unknown","trace_directory":"missing","output_directory":"","key_filters":{"redact_lines":["("]}}),
    ] {
        let dirs = Dirs::new();
        let mut cfg = dirs.cache("replay");
        cfg.record_replay = Some(raw);
        let session = Session::activate(cfg, false);
        let count = Arc::new(AtomicUsize::new(0));
        assert!(
            unary(session.clone(), request("stable"), provider(count.clone()))
                .await
                .unwrap_err()
                .to_string()
                .contains("RR-UNAVAILABLE")
        );
        assert_eq!(count.load(Ordering::SeqCst), 0);
        session.close();
    }
    let dirs = Dirs::new();
    let session = Session::activate(dirs.cache("replay"), false);
    let count = Arc::new(AtomicUsize::new(0));
    assert!(
        unary(session, request("stable"), provider(count.clone()))
            .await
            .is_err()
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert!(
        dirs.output("activation")
            .iter()
            .all(|a| a["fail_closed_reason"] == "trace_unreadable")
    );
}

#[tokio::test]
async fn raw_chunks_and_aggregate_preserve_reasoning() {
    let dirs = Dirs::new();
    let session = Session::activate(dirs.cache("record"), false);
    let mut stream = streaming(
        session.clone(),
        stream_provider(Arc::new(AtomicUsize::new(0))),
    )
    .await;
    let mut output = vec![];
    while let Some(item) = stream.next().await {
        output.push(item.unwrap());
    }
    assert_eq!(output, frames());
    session.close();
    let line = json_lines(&dirs.trace.join("calls.jsonl"))
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(line["outcome"], "ok");
    assert_eq!(
        line["response"]["choices"][0]["message"]["reasoning_content"],
        "why?"
    );
    assert_eq!(line["response"]["choices"][0]["message"]["vendor"], "xy");
    assert_eq!(line["usage_chunk"], true);
    let replay = dirs.copy_finalized();
    let session = Session::activate(replay.cache("verify"), false);
    let count = Arc::new(AtomicUsize::new(0));
    let mut stream = streaming(session, stream_provider(count.clone())).await;
    let mut replayed = vec![];
    while let Some(item) = stream.next().await {
        replayed.push(item.unwrap());
    }
    assert_eq!(replayed, frames());
    assert_eq!(count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn abandoned_answer_is_not_complete() {
    let dirs = Dirs::new();
    let session = Session::activate(dirs.cache("record"), false);
    let mut stream = streaming(
        session.clone(),
        stream_provider(Arc::new(AtomicUsize::new(0))),
    )
    .await;
    stream.next().await.unwrap().unwrap();
    drop(stream);
    session.close();
    assert_eq!(
        json_lines(&dirs.trace.join("calls.jsonl")).unwrap()[0]["outcome"],
        "consumer_dropped"
    );
    assert_eq!(dirs.output("record_status")[0]["in_flight"], false);
}

#[tokio::test(start_paused = true)]
async fn replay_deadlines_keep_end_gap_and_do_not_drift() {
    let dirs = Dirs::new();
    let session = Session::activate(dirs.cache("record"), false);
    let mut stream = streaming(
        session.clone(),
        stream_provider(Arc::new(AtomicUsize::new(0))),
    )
    .await;
    while stream.next().await.is_some() {}
    session.close();
    for scale in [0.0, 0.5, 1.0] {
        let replay = dirs.copy_finalized();
        let mut call = json_lines(&replay.trace.join("calls.jsonl"))
            .unwrap()
            .pop()
            .unwrap();
        for (i, chunk) in call["chunks"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            chunk["offset_us"] = json!((i + 1) * 10000);
        }
        call["ttft_us"] = json!(10000);
        call["total_us"] = json!(50000);
        fs::write(replay.trace.join("calls.jsonl"), format!("{call}\n")).unwrap();
        let mut cache = replay.cache("replay");
        cache.record_replay.as_mut().unwrap()["latency_scale"] = json!(scale);
        let session = Session::activate(cache, false);
        let count = Arc::new(AtomicUsize::new(0));
        let mut stream = streaming(session, stream_provider(count.clone())).await;
        let start = tokio::time::Instant::now();
        assert!(stream.next().await.unwrap().is_ok());
        assert_eq!(
            start.elapsed(),
            Duration::from_micros((10000.0 * scale) as u64)
        );
        tokio::time::sleep(Duration::from_micros((35000.0 * scale) as u64)).await;
        assert!(stream.next().await.unwrap().is_ok());
        assert!(stream.next().await.unwrap().is_ok());
        assert!(stream.next().await.is_none());
        assert_eq!(
            start.elapsed(),
            Duration::from_micros((50000.0 * scale) as u64)
        );
        assert_eq!(count.load(Ordering::SeqCst), 0);
        assert_eq!(replay.output("timing")[0]["outcome"], "complete");
    }
}

#[tokio::test(start_paused = true)]
async fn gate_timeout_latches_and_first_miss_never_reports_ready() {
    let dirs = recorded_buffered().await;
    let mut cfg = dirs.cache("replay");
    cfg.record_replay.as_mut().unwrap()["start_gate"] =
        json!({"path":dirs.root.join("go"),"timeout_seconds":1});
    let session = Session::activate(cfg, false);
    let count = Arc::new(AtomicUsize::new(0));
    for _ in 0..2 {
        assert!(
            unary(session.clone(), request("stable"), provider(count.clone()))
                .await
                .unwrap_err()
                .to_string()
                .contains("RR-GATE-TIMEOUT")
        );
    }
    let armed: Json =
        serde_json::from_slice(&fs::read(dirs.out.join("armed.json")).unwrap()).unwrap();
    assert_eq!(armed["gate_outcome"], "timed_out");
    assert_eq!(count.load(Ordering::SeqCst), 0);
    let miss = recorded_buffered().await;
    let mut cfg = miss.cache("replay");
    cfg.record_replay.as_mut().unwrap()["start_gate"] =
        json!({"path":miss.root.join("go"),"timeout_seconds":1});
    let session = Session::activate(cfg, false);
    assert!(
        unary(session, request("different"), provider(count))
            .await
            .unwrap_err()
            .to_string()
            .contains("RR-MISS")
    );
    let armed: Json =
        serde_json::from_slice(&fs::read(miss.out.join("armed.json")).unwrap()).unwrap();
    assert_eq!(armed["found"], false);
    assert_eq!(armed["gate_outcome"], "skipped");
}

#[tokio::test]
async fn gate_release_waits_for_go_and_starts_response_after_release() {
    let dirs = recorded_buffered().await;
    let mut cfg = dirs.cache("replay");
    let go = dirs.root.join("go");
    cfg.record_replay.as_mut().unwrap()["start_gate"] = json!({"path":go,"timeout_seconds":2});
    let session = Session::activate(cfg, false);
    let count = Arc::new(AtomicUsize::new(0));
    let call = tokio::spawn(unary(
        session.clone(),
        request("stable"),
        provider(count.clone()),
    ));
    tokio::time::timeout(Duration::from_secs(1), async {
        while !dirs.out.join("armed.json").exists() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
    assert!(!call.is_finished());
    let armed: Json =
        serde_json::from_slice(&fs::read(dirs.out.join("armed.json")).unwrap()).unwrap();
    assert_eq!(armed["gate_outcome"], "waiting");
    fs::write(go, "go").unwrap();
    assert_eq!(call.await.unwrap().unwrap()["n"], 0);
    let armed: Json =
        serde_json::from_slice(&fs::read(dirs.out.join("armed.json")).unwrap()).unwrap();
    assert_eq!(armed["gate_outcome"], "released");
    assert!(
        armed["gate_release_mono_ns"].as_u64().unwrap()
            >= armed["gate_entry_mono_ns"].as_u64().unwrap()
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
    session.close();
}

#[tokio::test]
async fn record_reactivation_preserves_fifo_and_gate() {
    let dirs = Dirs::new();
    let count = Arc::new(AtomicUsize::new(0));
    let first = Session::activate(dirs.cache("record"), false);
    unary(first.clone(), request("stable"), provider(count.clone()))
        .await
        .unwrap();
    first.close();
    let armed = fs::read(dirs.out.join("armed.json")).unwrap();
    let second = Session::activate(dirs.cache("record"), false);
    unary(second.clone(), request("stable"), provider(count.clone()))
        .await
        .unwrap();
    second.close();
    assert_eq!(fs::read(dirs.out.join("armed.json")).unwrap(), armed);
    let lines = json_lines(&dirs.trace.join("calls.jsonl")).unwrap();
    assert_eq!(lines[0]["repeat_index"], 0);
    assert_eq!(lines[1]["repeat_index"], 1);
    let mut cfg = dirs.cache("record");
    cfg.namespace = "changed".into();
    let changed = Session::activate(cfg, false);
    assert!(
        unary(changed, request("stable"), provider(count.clone()))
            .await
            .unwrap_err()
            .to_string()
            .contains("RR-UNAVAILABLE")
    );
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn tools_are_live_and_verify_latches_material_divergence() {
    let dirs = Dirs::new();
    let mut cfg = dirs.cache("record");
    cfg.record_replay.as_mut().unwrap()["divergence"] =
        json!({"tool_names":["terminal"],"tool_fields":["/exit_code"],"flag_fields":[]});
    let record = Session::activate(cfg, false);
    unary(
        record.clone(),
        request("stable"),
        provider(Arc::new(AtomicUsize::new(0))),
    )
    .await
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let tool_next = |calls: Arc<AtomicUsize>, exit: u64| -> ToolExecutionNextFn {
        Arc::new(move |_| {
            calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                Ok(ToolExecutionResult {
                    result: json!({"exit_code":exit,"output":"live"}),
                    annotation: Some(json!({"tag":"preserved"})),
                })
            })
        })
    };
    let context = ToolExecutionContext::new("terminal", json!({"command":"echo"}))
        .with_tool_call_id(Some("call-1".into()));
    let result = tool::make_tool(record.clone())(context.clone(), tool_next(calls.clone(), 0))
        .await
        .unwrap();
    assert_eq!(result.annotation, Some(json!({"tag":"preserved"})));
    record.close();
    let replay = dirs.copy_finalized();
    let mut cfg = replay.cache("verify");
    cfg.record_replay.as_mut().unwrap()["divergence"] =
        json!({"tool_names":["terminal"],"tool_fields":["/exit_code"],"flag_fields":[]});
    let verify = Session::activate(cfg, false);
    let models = Arc::new(AtomicUsize::new(0));
    unary(verify.clone(), request("stable"), provider(models.clone()))
        .await
        .unwrap();
    let result = tool::make_tool(verify.clone())(context, tool_next(calls.clone(), 1))
        .await
        .unwrap();
    assert_eq!(result.result["exit_code"], 1);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(
        unary(verify, request("next"), provider(models.clone()))
            .await
            .unwrap_err()
            .to_string()
            .contains("RR-TOOL")
    );
    assert_eq!(models.load(Ordering::SeqCst), 0);
    assert_eq!(replay.output("tools")[0]["status"], "mismatch");
}

#[tokio::test]
async fn json_object_tool_text_is_inspected_without_changing_live_result() {
    let dirs = Dirs::new();
    let mut cfg = dirs.cache("record");
    cfg.record_replay.as_mut().unwrap()["divergence"] =
        json!({"tool_names":["terminal"],"tool_fields":["/exit_code"],"flag_fields":[]});
    let record = Session::activate(cfg, false);
    unary(
        record.clone(),
        request("stable"),
        provider(Arc::new(AtomicUsize::new(0))),
    )
    .await
    .unwrap();
    let raw = "{ \"exit_code\": 0, \"output\": \"live\" }";
    let next: ToolExecutionNextFn = Arc::new(move |_| {
        Box::pin(async move {
            Ok(ToolExecutionResult {
                result: json!(raw),
                annotation: None,
            })
        })
    });
    let result =
        tool::make_tool(record.clone())(ToolExecutionContext::new("terminal", json!({})), next)
            .await
            .unwrap();
    assert_eq!(result.result, json!(raw));
    record.close();
    let line = &json_lines(&dirs.trace.join("tools.jsonl")).unwrap()[0];
    assert_eq!(line["result_kind"], "json_object");
    assert_eq!(line["shape_ok"], true);
    assert_eq!(line["fields"]["/exit_code"], 0);
    assert_eq!(line["output_bytes"], raw.len());
}

#[tokio::test]
async fn tool_signature_failures_preserve_live_results_and_latch_verify() {
    fn cache(dirs: &Dirs, mode: &str) -> ResponseCacheConfig {
        let mut cfg = dirs.cache(mode);
        cfg.record_replay.as_mut().unwrap()["divergence"] = json!({"tool_names":["terminal","execute_code"],"tool_fields":["/exit_code","/tool_calls_made","/kernel/reused","/tool_errors"],"flag_fields":["/tool_errors"]});
        cfg
    }
    fn next(value: Json) -> ToolExecutionNextFn {
        Arc::new(move |_| {
            let value = value.clone();
            Box::pin(async move {
                Ok(ToolExecutionResult {
                    result: value,
                    annotation: None,
                })
            })
        })
    }
    let dirs = Dirs::new();
    let record = Session::activate(cache(&dirs, "record"), false);
    unary(
        record.clone(),
        request("stable"),
        provider(Arc::new(AtomicUsize::new(0))),
    )
    .await
    .unwrap();
    let original =
        json!({"exit_code":0,"output":"old","tool_calls_made":1,"kernel":{"reused":true}});
    let context = ToolExecutionContext::new("terminal", json!({"command":"work"}))
        .with_tool_call_id(Some("call-1".into()));
    tool::make_tool(record.clone())(context.clone(), next(original.clone()))
        .await
        .unwrap();
    record.close();
    for case in [
        "name",
        "args",
        "shape",
        "unknown",
        "duplicate",
        "missing",
        "nested",
        "restart",
        "size",
    ] {
        let replay = dirs.copy_finalized();
        let verify = Session::activate(cache(&replay, "verify"), false);
        let models = Arc::new(AtomicUsize::new(0));
        unary(verify.clone(), request("stable"), provider(models.clone()))
            .await
            .unwrap();
        let mut value = original.clone();
        let mut name = "terminal";
        let mut args = json!({"command":"work"});
        match case {
            "name" => name = "execute_code",
            "args" => args = json!({"command":"changed"}),
            "shape" => value = json!("not an object"),
            "unknown" => name = "unknown",
            "nested" => value["tool_errors"] = json!(["failure"]),
            "restart" => value["kernel"]["reused"] = json!(false),
            "size" => value["output"] = json!("x".repeat(1000)),
            _ => {}
        }
        if case != "missing" {
            let context =
                ToolExecutionContext::new(name, args).with_tool_call_id(Some("call-1".into()));
            let result = tool::make_tool(verify.clone())(context.clone(), next(value.clone()))
                .await
                .unwrap();
            assert_eq!(result.result, value);
            if case == "duplicate" {
                tool::make_tool(verify.clone())(context, next(value))
                    .await
                    .unwrap();
            }
        }
        let failure = unary(verify.clone(), request("stable"), provider(models.clone()))
            .await
            .unwrap_err()
            .to_string();
        // A recorded tool that never ran is a completeness leftover, not
        // a divergence, so it neither latches verify nor writes a tool line.
        assert!(
            failure.contains(if matches!(case, "size" | "missing") {
                "RR-EXHAUSTED"
            } else {
                "RR-TOOL"
            }),
            "{case}: {failure}"
        );
        assert_eq!(models.load(Ordering::SeqCst), 0);
        if case == "missing" {
            assert!(replay.output("tools").is_empty());
            assert!(
                replay
                    .output("divergence")
                    .iter()
                    .all(|d| !d["type"].as_str().unwrap().starts_with("tool_"))
            );
        }
        if case == "size" {
            assert_eq!(replay.output("tools")[0]["status"], "size_warning");
        }
        if case == "name" || case == "unknown" {
            let differences: Vec<_> = replay
                .output("divergence")
                .into_iter()
                .filter(|d| d["type"].as_str().is_some_and(|t| t.starts_with("tool_")))
                .collect();
            assert_eq!(differences.len(), 1);
            assert_eq!(differences[0]["type"], "tool_name_mismatch");
        }
        verify.close();
    }
}

#[tokio::test]
async fn valid_non_object_json_tool_text_is_json_other() {
    for raw in ["[]", "1", "\"text\""] {
        let dirs = Dirs::new();
        let record = Session::activate(dirs.cache("record"), false);
        let next: ToolExecutionNextFn = Arc::new(move |_| {
            Box::pin(async move {
                Ok(ToolExecutionResult {
                    result: json!(raw),
                    annotation: None,
                })
            })
        });
        let result =
            tool::make_tool(record.clone())(ToolExecutionContext::new("tool", json!({})), next)
                .await
                .unwrap();
        assert_eq!(result.result, json!(raw));
        record.close();
        assert_eq!(
            json_lines(&dirs.trace.join("tools.jsonl")).unwrap()[0]["result_kind"],
            "json_other"
        );
    }
}

#[test]
fn ordinary_cache_key_matches_the_installed_baseline_wheel() {
    use crate::key::{KeyOutcome, build_cache_key};
    let cache = ResponseCacheConfig {
        namespace: "rr-golden".into(),
        ..Default::default()
    };
    let req = LlmRequest {
        headers: Default::default(),
        content: json!({"model":"m","messages":[{"role":"user","content":"stable"}],"temperature":0.0}),
    };
    assert_eq!(
        build_cache_key("openai", &req, &cache),
        KeyOutcome::Key(
            "sha256:bb320836c4e359572e70265b33c24abaf423cd450b65eec3ed96fed1f10283e6".into()
        )
    );
    let encoded = serde_json::to_value(cache).unwrap();
    assert!(encoded.get("record_replay").is_none());
    let config: ResponseCacheConfig =
        serde_json::from_value(json!({"record_replay":null})).unwrap();
    assert_eq!(config.record_replay, Some(Json::Null));
}

#[test]
fn responses_terminal_envelope_is_preserved_verbatim() {
    let response = json!({"id":"r","object":"response","status":"completed","output":[],"reasoning":{"effort":"medium"},"usage":{"output_tokens_details":{"reasoning_tokens":4}},"vendor_extra":{"keep":true}});
    let chunks = vec![Chunk {
        offset_us: 1,
        data: json!({"type":"response.completed","response":response}),
    }];
    let (actual, complete) = stream::aggregate(&chunks);
    assert!(complete);
    assert_eq!(actual, response);
}

#[tokio::test]
async fn append_failure_preserves_live_result_and_latches_future_calls() {
    let dirs = Dirs::new();
    let session = Session::activate(dirs.cache("record"), false);
    lock(&session.state.data)
        .writer
        .as_mut()
        .unwrap()
        .fail_calls();
    let count = Arc::new(AtomicUsize::new(0));
    assert_eq!(
        unary(session.clone(), request("stable"), provider(count.clone()))
            .await
            .unwrap()["n"],
        0
    );
    assert!(
        unary(session.clone(), request("stable"), provider(count.clone()))
            .await
            .unwrap_err()
            .to_string()
            .contains("RR-UNAVAILABLE")
    );
    session.close();
    let status = dirs.output("record_status").pop().unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert!(status["io_errors"].as_u64().unwrap() >= 1);
    assert_eq!(status["first_io_error"]["file"], "calls.jsonl");
    assert_eq!(status["lines"]["calls"], 0);
}

#[tokio::test]
async fn concurrent_call_does_not_consume_a_sequence_or_repeat() {
    let dirs = Dirs::new();
    let session = Session::activate(dirs.cache("record"), false);
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let count = Arc::new(AtomicUsize::new(0));
    let next: LlmExecutionNextFn = {
        let started = started.clone();
        let release = release.clone();
        let count = count.clone();
        Arc::new(move |_| {
            let started = started.clone();
            let release = release.clone();
            let count = count.clone();
            Box::pin(async move {
                count.fetch_add(1, Ordering::SeqCst);
                started.notify_one();
                release.notified().await;
                Ok(json!({"ok":true}))
            })
        })
    };
    let first = tokio::spawn(unary(session.clone(), request("stable"), next.clone()));
    started.notified().await;
    assert!(
        unary(session.clone(), request("stable"), next)
            .await
            .unwrap_err()
            .to_string()
            .contains("RR-CONCURRENT")
    );
    release.notify_one();
    first.await.unwrap().unwrap();
    session.close();
    assert_eq!(count.load(Ordering::SeqCst), 1);
    let status = dirs.output("record_status").pop().unwrap();
    assert_eq!(status["model_calls"], 1);
    let line = json_lines(&dirs.trace.join("calls.jsonl"))
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(line["seq"], 0);
    assert_eq!(line["repeat_index"], 0);
}

#[tokio::test]
async fn locked_corrupt_and_policy_mismatched_traces_never_call_provider() {
    let dirs = recorded_buffered().await;
    for kind in ["locked", "partial", "policy", "duplicate", "repeat_gap"] {
        let trace = dirs.copy_finalized();
        let mut cfg = trace.cache("replay");
        match kind {
            "locked" => fs::write(trace.trace.join("record.lock"), "{}\n").unwrap(),
            "partial" => fs::write(trace.trace.join("calls.jsonl"), "{\"v\":1}").unwrap(),
            "policy" => cfg.namespace = "different".into(),
            "duplicate" => {
                let bytes = fs::read(trace.trace.join("calls.jsonl")).unwrap();
                fs::write(
                    trace.trace.join("calls.jsonl"),
                    [bytes.clone(), bytes].concat(),
                )
                .unwrap();
            }
            "repeat_gap" => {
                let mut line = json_lines(&trace.trace.join("calls.jsonl"))
                    .unwrap()
                    .pop()
                    .unwrap();
                line["repeat_index"] = json!(7);
                fs::write(trace.trace.join("calls.jsonl"), format!("{line}\n")).unwrap();
            }
            _ => unreachable!(),
        }
        let session = Session::activate(cfg, false);
        let count = Arc::new(AtomicUsize::new(0));
        assert!(
            unary(session, request("stable"), provider(count.clone()))
                .await
                .unwrap_err()
                .to_string()
                .contains("RR-UNAVAILABLE")
        );
        assert_eq!(count.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn forbidden_settings_activate_error_returning_middleware() {
    for kind in ["bypass", "headers", "tools", "backend", "priority"] {
        let dirs = Dirs::new();
        let mut cache = dirs.cache("replay");
        match kind {
            "bypass" => cache.bypass_rate = 1.0,
            "headers" => cache.header_allowlist = vec!["authorization".into()],
            "tools" => cache.tools = Some(Default::default()),
            "backend" => cache.backend.kind = "redis".into(),
            "priority" => cache.priority = 7,
            _ => unreachable!(),
        }
        let raw = match serde_json::to_value(&cache).unwrap() {
            Json::Object(raw) => raw,
            _ => unreachable!(),
        };
        // Forbidden settings warn but never fail activation.
        let diagnostics = crate::validate_config(&raw);
        assert!(
            diagnostics
                .iter()
                .all(|d| d.level == nemo_relay_plugin::DiagnosticLevel::Warning),
            "{kind}: {diagnostics:?}"
        );
        let codes: Vec<String> = diagnostics
            .into_iter()
            .map(|d| d.code)
            .filter(|code| code.starts_with("response_cache.record_replay_"))
            .collect();
        assert!(
            codes
                .iter()
                .any(|code| code == "response_cache.record_replay_forbidden_cache_setting"),
            "{kind}: {codes:?}"
        );
        let activation = activate(cache, codes);
        let count = Arc::new(AtomicUsize::new(0));
        let result = (activation.llm)(
            "openai",
            request("stable"),
            LlmExecutionContext::default(),
            provider(count.clone()),
        )
        .await;
        assert!(result.unwrap_err().to_string().contains("RR-UNAVAILABLE"));
        assert_eq!(count.load(Ordering::SeqCst), 0);
        assert!(
            !dirs.output("activation")[0]["diagnostic_codes"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
}

fn chunk(data: Json) -> Chunk {
    Chunk { offset_us: 1, data }
}

#[test]
fn responses_aggregate_rebuilds_output_and_accepts_incomplete() {
    let created =
        json!({"type":"response.created","response":{"id":"r","status":"in_progress","output":[]}});
    let call_added = json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc","call_id":"c","name":"terminal","arguments":""}});
    let delta = json!({"type":"response.function_call_arguments.delta","item_id":"fc","output_index":1,"delta":"{}"});
    let call_done = json!({"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc","call_id":"c","name":"terminal","arguments":"{}"}});
    let message_done = json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"m","content":[]}});
    let reasoning_done = json!({"type":"response.output_item.done","output_index":2,"item":{"type":"reasoning","id":"rs","encrypted_content":"blob"}});
    let terminal = |kind: &str| json!({"type":kind,"response":{"id":"r","status":"x","output":null,"usage":{"output_tokens":3}}});
    let stream = |last: Json| {
        [
            created.clone(),
            call_added.clone(),
            delta.clone(),
            call_done.clone(),
            message_done.clone(),
            reasoning_done.clone(),
            last,
        ]
        .into_iter()
        .map(chunk)
        .collect::<Vec<_>>()
    };
    for (kind, ok) in [
        ("response.completed", true),
        ("response.incomplete", true),
        ("response.failed", false),
    ] {
        let (response, terminal_ok) = stream::aggregate(&stream(terminal(kind)));
        assert_eq!(terminal_ok, ok, "{kind}");
        assert_eq!(response["usage"]["output_tokens"], 3);
        let types: Vec<_> = response["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["type"].clone())
            .collect();
        assert_eq!(types, ["message", "function_call", "reasoning"]);
        assert_eq!(response["output"][2]["encrypted_content"], "blob");
    }
    // Without a terminal event the base is the last created snapshot, and the call is not ok.
    let (response, ok) = stream::aggregate(&[chunk(created.clone()), chunk(message_done.clone())]);
    assert!(!ok);
    assert_eq!(response["status"], "in_progress");
    // A completed response settles an announced call that never got a done event.
    let (response, ok) =
        stream::aggregate(&[created, call_added, delta, terminal("response.completed")].map(chunk));
    assert!(ok);
    assert_eq!(response["output"][0]["arguments"], "{}");
}

#[tokio::test]
async fn responses_incomplete_stream_is_recorded_ok() {
    let dirs = Dirs::new();
    let session = Session::activate(dirs.cache("record"), false);
    let next: LlmStreamExecutionNextFn = Arc::new(|_| {
        Box::pin(async {
            Ok(LlmJsonStream::new(tokio_stream::iter(
                [
                    json!({"type":"response.created","response":{"id":"r","output":[]}}),
                    json!({"type":"response.incomplete","response":{"id":"r","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[],"usage":{"output_tokens":1}}}),
                ]
                .map(Ok),
            )))
        })
    });
    let mut stream = streaming(session.clone(), next).await;
    while stream.next().await.is_some() {}
    session.close();
    let line = json_lines(&dirs.trace.join("calls.jsonl"))
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(line["outcome"], "ok");
    assert_eq!(line["usage_chunk"], true);
    assert_eq!(
        line["response"]["incomplete_details"]["reason"],
        "max_output_tokens"
    );
}

#[tokio::test]
async fn held_stream_does_not_block_the_next_call() {
    let dirs = Dirs::new();
    let session = Session::activate(dirs.cache("record"), false);
    let mut stream = streaming(
        session.clone(),
        stream_provider(Arc::new(AtomicUsize::new(0))),
    )
    .await;
    while stream.next().await.is_some() {}
    // The consumer still holds the stream, but the upstream already ended.
    unary(
        session.clone(),
        request("other"),
        provider(Arc::new(AtomicUsize::new(0))),
    )
    .await
    .unwrap();
    drop(stream);
    session.close();
    assert_eq!(dirs.output("record_status")[0]["in_flight"], false);
}

#[tokio::test]
async fn fail_closed_tools_still_run_live() {
    let dirs = Dirs::new();
    let mut cfg = dirs.cache("replay");
    cfg.record_replay = Some(Json::Null);
    let session = Session::activate(cfg, false);
    let next: ToolExecutionNextFn = Arc::new(|_| {
        Box::pin(async {
            Ok(ToolExecutionResult {
                result: json!({"exit_code":0}),
                annotation: None,
            })
        })
    });
    let result = tool::make_tool(session)(ToolExecutionContext::new("terminal", json!({})), next)
        .await
        .unwrap();
    assert_eq!(result.result, json!({"exit_code":0}));
}

#[tokio::test]
async fn cancelled_buffered_replay_writes_a_dropped_timing_line() {
    let dirs = recorded_buffered().await;
    let mut call = json_lines(&dirs.trace.join("calls.jsonl"))
        .unwrap()
        .pop()
        .unwrap();
    call["total_us"] = json!(60_000_000);
    fs::write(dirs.trace.join("calls.jsonl"), format!("{call}\n")).unwrap();
    let mut cfg = dirs.cache("replay");
    cfg.record_replay.as_mut().unwrap()["latency_scale"] = json!(1);
    let session = Session::activate(cfg, false);
    let task = tokio::spawn(unary(
        session.clone(),
        request("stable"),
        provider(Arc::new(AtomicUsize::new(0))),
    ));
    tokio::time::sleep(Duration::from_millis(50)).await;
    task.abort();
    let _ = task.await;
    let timing = dirs.output("timing");
    assert_eq!(timing.len(), 1);
    assert_eq!(timing[0]["outcome"], "consumer_dropped");
    assert!(!lock(&session.state.data).busy);
}

#[test]
fn an_existing_record_lock_reports_trace_locked() {
    let dirs = Dirs::new();
    fs::write(dirs.trace.join("record.lock"), "{}\n").unwrap();
    fs::write(dirs.trace.join("calls.jsonl"), "").unwrap();
    let session = Session::activate(dirs.cache("record"), false);
    assert_eq!(
        lock(&session.state.data).fail.as_deref(),
        Some("trace_locked")
    );
}

#[tokio::test]
async fn unterminated_redaction_block_never_reaches_the_trace() {
    let dirs = Dirs::new();
    let mut cfg = dirs.cache("record");
    cfg.record_replay.as_mut().unwrap()["key_filters"] =
        json!({"redact_blocks":[{"start":"<secret>","end":"</secret>"}]});
    let session = Session::activate(cfg, false);
    unary(
        session.clone(),
        request("before <secret>token-123 and the rest"),
        provider(Arc::new(AtomicUsize::new(0))),
    )
    .await
    .unwrap();
    session.close();
    let line = &json_lines(&dirs.trace.join("requests.jsonl")).unwrap()[0];
    let content = line["filtered_request"]["messages"][0]["content"]
        .as_str()
        .unwrap();
    assert!(content.starts_with("before "), "{content}");
    assert!(!content.contains("token-123"), "{content}");
    assert!(line["key_hash"].is_null());
}

#[tokio::test(start_paused = true)]
async fn recorded_chunk_times_exclude_caller_pauses() {
    let dirs = Dirs::new();
    let session = Session::activate(dirs.cache("record"), false);
    let mut stream = streaming(
        session.clone(),
        stream_provider(Arc::new(AtomicUsize::new(0))),
    )
    .await;
    stream.next().await.unwrap().unwrap();
    tokio::time::sleep(Duration::from_millis(350)).await;
    while stream.next().await.is_some() {}
    session.close();
    let line = json_lines(&dirs.trace.join("calls.jsonl"))
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(line["outcome"], "ok");
    let offsets: Vec<_> = line["chunks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["offset_us"].as_u64().unwrap())
        .collect();
    // The provider produced every chunk at once; the caller's pause is not
    // part of the recorded schedule.
    assert!(offsets.iter().all(|&offset| offset < 1_000), "{offsets:?}");
}

#[tokio::test]
async fn cancelled_recordings_keep_the_trace_readable() {
    let dirs = Dirs::new();
    let session = Session::activate(dirs.cache("record"), false);
    let hang: LlmExecutionNextFn = Arc::new(|_| Box::pin(std::future::pending()));
    let cancelled = tokio::spawn(unary(session.clone(), request("stable"), hang));
    tokio::time::sleep(Duration::from_millis(20)).await;
    cancelled.abort();
    let _ = cancelled.await;
    let hang_stream: LlmStreamExecutionNextFn = Arc::new(|_| Box::pin(std::future::pending()));
    let cancelled = tokio::spawn(async move {
        let stream = stream::make_stream(session.clone())(
            "openai",
            request("stream"),
            LlmExecutionContext::default(),
            hang_stream,
        );
        drop(stream.await);
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    cancelled.abort();
    let _ = cancelled.await;
    let session = Session::activate(dirs.cache("record"), false);
    unary(
        session.clone(),
        request("stable"),
        provider(Arc::new(AtomicUsize::new(0))),
    )
    .await
    .unwrap();
    session.close();
    let calls = json_lines(&dirs.trace.join("calls.jsonl")).unwrap();
    let outcomes: Vec<_> = calls
        .iter()
        .map(|c| {
            (
                c["stream"].clone(),
                c["repeat_index"].clone(),
                c["outcome"].clone(),
            )
        })
        .collect();
    assert_eq!(
        outcomes,
        vec![
            (json!(false), json!(0), json!("consumer_dropped")),
            (json!(true), json!(0), json!("consumer_dropped")),
            (json!(false), json!(1), json!("ok")),
        ]
    );
    let replay = dirs.copy_finalized();
    let session = Session::activate(replay.cache("replay"), false);
    assert_eq!(lock(&session.state.data).fail, None);
}

#[tokio::test]
async fn cancelled_gate_wait_keeps_the_gate_closed_for_the_retry() {
    let dirs = Dirs::new();
    let mut cfg = dirs.cache("record");
    let go = dirs.root.join("go");
    cfg.record_replay.as_mut().unwrap()["start_gate"] = json!({"path":go,"timeout_seconds":5});
    let session = Session::activate(cfg, false);
    let count = Arc::new(AtomicUsize::new(0));
    let first = tokio::spawn(unary(
        session.clone(),
        request("stable"),
        provider(count.clone()),
    ));
    tokio::time::timeout(Duration::from_secs(1), async {
        while !dirs.out.join("armed.json").exists() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
    first.abort();
    let _ = first.await;
    let retry = tokio::spawn(unary(
        session.clone(),
        request("stable"),
        provider(count.clone()),
    ));
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!retry.is_finished());
    assert_eq!(count.load(Ordering::SeqCst), 0);
    fs::write(go, "go").unwrap();
    retry.await.unwrap().unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);
    session.close();
    // The cancelled wait gave its repeat index back.
    let calls = json_lines(&dirs.trace.join("calls.jsonl")).unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["repeat_index"], 0);
}

#[tokio::test]
async fn verify_accepts_identical_results_from_unchecked_tools() {
    fn cache(dirs: &Dirs, mode: &str) -> ResponseCacheConfig {
        let mut cfg = dirs.cache(mode);
        cfg.record_replay.as_mut().unwrap()["divergence"] =
            json!({"tool_names":["terminal"],"tool_fields":[],"flag_fields":[]});
        cfg
    }
    async fn run(
        session: Arc<Session>,
        models: Arc<AtomicUsize>,
    ) -> nemo_relay_types::error::Result<Json> {
        unary(session.clone(), request("first"), provider(models.clone())).await?;
        let next: ToolExecutionNextFn = Arc::new(|_| {
            Box::pin(async {
                Ok(ToolExecutionResult {
                    result: json!("file contents"),
                    annotation: None,
                })
            })
        });
        tool::make_tool(session.clone())(
            ToolExecutionContext::new("read_file", json!({"path":"a"}))
                .with_tool_call_id(Some("call-1".into())),
            next,
        )
        .await?;
        unary(session, request("second"), provider(models)).await
    }
    let dirs = Dirs::new();
    let record = Session::activate(cache(&dirs, "record"), false);
    run(record.clone(), Arc::new(AtomicUsize::new(0)))
        .await
        .unwrap();
    record.close();
    let replay = dirs.copy_finalized();
    let verify = Session::activate(cache(&replay, "verify"), false);
    let models = Arc::new(AtomicUsize::new(0));
    run(verify.clone(), models.clone()).await.unwrap();
    assert_eq!(models.load(Ordering::SeqCst), 0);
    verify.close();
}

#[tokio::test]
async fn record_and_replay_admit_the_next_call_at_the_same_point() {
    async fn sequence(session: Arc<Session>, models: Arc<AtomicUsize>) {
        let mut stream = streaming(session.clone(), stream_provider(models.clone())).await;
        // Let the recorder read the whole provider stream ahead of the caller.
        tokio::time::sleep(Duration::from_millis(20)).await;
        let early = unary(session.clone(), request("other"), provider(models.clone())).await;
        assert!(
            early.unwrap_err().to_string().contains("RR-CONCURRENT"),
            "the unread stream is still in flight"
        );
        while stream.next().await.is_some() {}
        unary(session, request("other"), provider(models))
            .await
            .unwrap();
    }
    let dirs = Dirs::new();
    let record = Session::activate(dirs.cache("record"), false);
    sequence(record.clone(), Arc::new(AtomicUsize::new(0))).await;
    record.close();
    let replay = dirs.copy_finalized();
    let session = Session::activate(replay.cache("replay"), false);
    let models = Arc::new(AtomicUsize::new(0));
    sequence(session.clone(), models.clone()).await;
    assert_eq!(models.load(Ordering::SeqCst), 0);
    session.close();
}

#[tokio::test]
async fn first_miss_leaves_the_gate_armed_for_a_later_match() {
    let dirs = recorded_buffered().await;
    let mut cfg = dirs.cache("replay");
    let go = dirs.root.join("go");
    cfg.record_replay.as_mut().unwrap()["start_gate"] = json!({"path":go,"timeout_seconds":5});
    let session = Session::activate(cfg, false);
    let count = Arc::new(AtomicUsize::new(0));
    let miss = unary(
        session.clone(),
        request("different"),
        provider(count.clone()),
    )
    .await;
    assert!(miss.unwrap_err().to_string().contains("RR-MISS"));
    let hit = tokio::spawn(unary(
        session.clone(),
        request("stable"),
        provider(count.clone()),
    ));
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!hit.is_finished(), "a match waits for the release");
    let armed: Json =
        serde_json::from_slice(&fs::read(dirs.out.join("armed.json")).unwrap()).unwrap();
    assert_eq!(armed["gate_outcome"], "waiting");
    fs::write(go, "go").unwrap();
    assert_eq!(hit.await.unwrap().unwrap()["n"], 0);
    assert_eq!(count.load(Ordering::SeqCst), 0);
    session.close();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_tools_track_every_call() {
    const CALLS: usize = 256;
    let dirs = Dirs::new();
    let session = Session::activate(dirs.cache("record"), false);
    unary(
        session.clone(),
        request("stable"),
        provider(Arc::new(AtomicUsize::new(0))),
    )
    .await
    .unwrap();
    // Every call waits until all have started, so each one overlaps another.
    let barrier = Arc::new(tokio::sync::Barrier::new(CALLS));
    let calls: Vec<_> = (0..CALLS)
        .map(|i| {
            let barrier = barrier.clone();
            let next: ToolExecutionNextFn = Arc::new(move |_| {
                let barrier = barrier.clone();
                Box::pin(async move {
                    barrier.wait().await;
                    Ok(ToolExecutionResult {
                        result: json!({"exit_code":0}),
                        annotation: None,
                    })
                })
            });
            let context = ToolExecutionContext::new("terminal", json!({"i":i}))
                .with_tool_call_id(Some(format!("call-{i}")));
            tokio::spawn(tool::make_tool(session.clone())(context, next))
        })
        .collect();
    for call in calls {
        call.await.unwrap().unwrap();
    }
    assert!(lock(&session.state.data).tools_active.is_empty());
    session.close();
    let lines = json_lines(&dirs.trace.join("tools.jsonl")).unwrap();
    assert_eq!(lines.len(), CALLS);
    assert!(lines.iter().all(|line| line["overlapped"] == true));
}
