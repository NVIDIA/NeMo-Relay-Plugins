// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    config::{RecordReplayConfig, parse_record_replay},
    key::{KeyBuilder, KeyedRequest, first_difference},
    trace::{self, Call, Reader, Writer},
};
use crate::config::ResponseCacheConfig;
use nemo_relay_types::api::llm::LlmRequest;
use nemo_relay_types::error::{FlowError, Result};
use serde_json::{Value as Json, json};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;
use tokio::time::Instant;
use uuid::Uuid;

static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Arc<State>>>> = OnceLock::new();
static NONCE: OnceLock<String> = OnceLock::new();

pub fn error(code: &str) -> FlowError {
    FlowError::Internal(format!("relay-record-replay: {code}"))
}
pub fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Clone)]
pub struct Parent {
    pub seq: u64,
    pub hash: Option<String>,
    pub repeat: Option<u64>,
}
pub struct Data {
    pub fail: Option<String>,
    pub seq: u64,
    pub repeats: HashMap<String, u64>,
    pub busy: bool,
    pub last: Option<Parent>,
    pub tools_active: HashMap<u64, bool>,
    pub tool_calls: u64,
    pub executed: HashSet<String>,
    pub ordinals: HashMap<String, u64>,
    pub verify_failed: bool,
    pub writer: Option<Writer>,
    pub io_errors: u64,
    pub first_io_error: Option<Json>,
    pub gate_entered: bool,
    pub gate_failed: bool,
    pub miss_reported: bool,
}
impl Data {
    fn new(fail: Option<String>, writer: Option<Writer>) -> Self {
        Self {
            fail,
            writer,
            seq: 0,
            repeats: HashMap::new(),
            busy: false,
            last: None,
            tools_active: HashMap::new(),
            tool_calls: 0,
            executed: HashSet::new(),
            ordinals: HashMap::new(),
            verify_failed: false,
            io_errors: 0,
            first_io_error: None,
            gate_entered: false,
            gate_failed: false,
            miss_reported: false,
        }
    }
}
pub struct State {
    pub mode: String,
    pub builder: KeyBuilder,
    pub reader: Option<Arc<Reader>>,
    pub data: Mutex<Data>,
}
pub struct Session {
    pub id: String,
    pub config: RecordReplayConfig,
    pub cache: ResponseCacheConfig,
    pub state: Arc<State>,
}
pub struct ModelGuard {
    state: Arc<State>,
}
impl Drop for ModelGuard {
    fn drop(&mut self) {
        lock(&self.state.data).busy = false;
    }
}
pub struct Run {
    pub seq: u64,
    pub key: KeyedRequest,
    pub repeat: Option<u64>,
    pub entry: Option<Call>,
    /// Released when the model call stops being in flight.
    pub guard: Option<ModelGuard>,
    pub t0: Instant,
    pub gate_wait_ms: u64,
}

impl Session {
    #[cfg(all(test, unix))]
    pub fn activate(cache: ResponseCacheConfig, invalid: bool) -> Arc<Self> {
        Self::activate_with_diagnostics(
            cache,
            if invalid {
                vec!["response_cache.record_replay_config_invalid".into()]
            } else {
                vec![]
            },
        )
    }
    pub fn activate_with_diagnostics(
        cache: ResponseCacheConfig,
        mut diagnostics: Vec<String>,
    ) -> Arc<Self> {
        if !cfg!(unix) {
            diagnostics.push("response_cache.record_replay_platform_unsupported".into());
        }
        let invalid = !diagnostics.is_empty();
        let id = Uuid::new_v4().to_string();
        let parsed = parse_record_replay(
            cache
                .record_replay
                .as_ref()
                .expect("selected record/replay"),
        );
        diagnostics.extend(parsed.diagnostics.iter().map(|d| d.code.clone()));
        diagnostics.sort();
        diagnostics.dedup();
        let mut config = parsed.config.clone().unwrap_or_default();
        if parsed.config.is_none() {
            if let Some(out) = parsed.output_directory {
                config.output_directory = out;
            }
            if cache
                .record_replay
                .as_ref()
                .and_then(|v| v.get("mode"))
                .and_then(Json::as_str)
                == Some("record")
            {
                config.mode = "record".into();
            }
        }
        let builder = KeyBuilder::new(&cache, config.key_filters.clone());
        let state = if invalid || parsed.config.is_none() {
            Arc::new(State {
                mode: config.mode.clone(),
                builder,
                reader: None,
                data: Mutex::new(Data::new(Some("config_invalid".into()), None)),
            })
        } else {
            let directory = fs::canonicalize(&config.trace_directory);
            let mut registry = lock(REGISTRY.get_or_init(|| Mutex::new(HashMap::new())));
            if let Ok(directory) = directory {
                if let Some(state) = registry.get(&directory) {
                    if state.mode != config.mode || state.builder.policy_hash != builder.policy_hash
                    {
                        lock(&state.data).fail = Some("reactivation_config_changed".into());
                    }
                    state.clone()
                } else {
                    let mut fail = None;
                    let mut writer = None;
                    let mut reader = None;
                    if config.mode == "record" {
                        let metadata = json!({"format":{"major":1,"minor":0},"relay_version":env!("CARGO_PKG_VERSION"),
                            "cache_schema_version":crate::store::CACHE_SCHEMA_VERSION,
                            "key_policy":builder.policy(),"key_policy_sha256":builder.policy_hash,"activation_id":id,"created_unix_ns":trace::now_ns()});
                        match Writer::create(
                            &directory,
                            &metadata,
                            NONCE.get_or_init(|| Uuid::new_v4().to_string()),
                            &id,
                        ) {
                            Ok(value) => writer = Some(value),
                            Err(reason) => fail = Some(reason.into()),
                        }
                    } else {
                        match Reader::load(&directory, &builder.policy_hash) {
                            Ok(value) => reader = Some(Arc::new(value)),
                            Err(reason) => fail = Some(reason.into()),
                        }
                    }
                    let mut data = Data::new(fail, writer);
                    if data.fail.as_deref() == Some("trace_unwritable") {
                        data.io_errors = 1;
                        data.first_io_error =
                            Some(json!({"file":"trace_startup","errno":null,"seq":null}));
                    }
                    let state = Arc::new(State {
                        mode: config.mode.clone(),
                        builder,
                        reader,
                        data: Mutex::new(data),
                    });
                    registry.insert(directory, state.clone());
                    state
                }
            } else {
                Arc::new(State {
                    mode: config.mode.clone(),
                    builder,
                    reader: None,
                    data: Mutex::new(Data::new(Some("trace_missing".into()), None)),
                })
            }
        };
        let session = Arc::new(Self {
            id,
            config,
            cache,
            state,
        });
        let reason = lock(&session.state.data).fail.clone();
        let activation = json!({"activation_id":session.id,"mode":session.config.mode,"pid":std::process::id(),
            "relay_version":env!("CARGO_PKG_VERSION"),"key_policy_sha256":session.state.builder.policy_hash,
            "fail_closed_reason":reason,"diagnostic_codes":diagnostics});
        if (session.config.output_directory.as_os_str().is_empty()
            || fs::create_dir_all(&session.config.output_directory).is_err()
            || trace::atomic_json(
                &session
                    .config
                    .output_directory
                    .join(format!("activation.{}.json", session.id)),
                &activation,
            )
            .is_err())
            && reason.as_deref() != Some("config_invalid")
        {
            lock(&session.state.data)
                .fail
                .get_or_insert_with(|| "output_directory_unwritable".into());
        }
        session
    }
    pub fn output(&self, kind: &str, value: Json) {
        if self.config.output_directory.as_os_str().is_empty() {
            return;
        }
        if trace::append_output(
            &self
                .config
                .output_directory
                .join(format!("{kind}.{}.jsonl", self.id)),
            &trace::envelope(&self.id, value),
        )
        .is_err()
        {
            lock(&self.state.data)
                .fail
                .get_or_insert_with(|| "output_directory_unwritable".into());
        }
    }
    pub fn append(&self, kind: &'static str, value: Json, seq: Option<u64>) {
        let mut data = lock(&self.state.data);
        if data.io_errors > 0 {
            return;
        }
        let Some(writer) = data.writer.as_mut() else {
            return;
        };
        if let Err(error) = writer.append(kind, &trace::envelope(&self.id, value)) {
            data.io_errors += 1;
            data.first_io_error.get_or_insert(
                json!({"file":format!("{kind}.jsonl"),"errno":error.raw_os_error(),"seq":seq}),
            );
            data.fail = Some("record_io_failed".into());
        }
    }
    pub fn divergence(
        &self,
        seq: u64,
        kind: &str,
        hash: Option<&str>,
        repeat: Option<u64>,
        detail: Json,
    ) {
        self.output(
            "divergence",
            json!({"seq":seq,"type":kind,"key_hash":hash,"repeat_index":repeat,"detail":detail}),
        );
    }
    pub fn close(&self) {
        if self.state.mode != "record" || self.config.output_directory.as_os_str().is_empty() {
            return;
        }
        let mut data = lock(&self.state.data);
        if let Some(writer) = &data.writer
            && let Err(error) = writer.sync()
        {
            data.io_errors += 1;
            data.fail = Some("record_io_failed".into());
            data.first_io_error.get_or_insert(
                json!({"file":"trace_sync","errno":error.raw_os_error(),"seq":null}),
            );
        }
        let lines = data
            .writer
            .as_ref()
            .map(|w| json!(w.lines))
            .unwrap_or(json!({"calls":0,"requests":0,"tools":0}));
        let value = json!({"activation_id":self.id,"closed_unix_ns":trace::now_ns(),"fail_closed_reason":data.fail,
            "in_flight":data.busy||!data.tools_active.is_empty(),"io_errors":data.io_errors,"first_io_error":data.first_io_error,
            "model_calls":data.seq,"tool_calls":data.tool_calls,"lines":lines});
        let _ = trace::atomic_json(
            &self
                .config
                .output_directory
                .join(format!("record_status.{}.json", self.id)),
            &value,
        );
    }
    pub async fn enter(
        self: &Arc<Self>,
        operation: &str,
        request: &LlmRequest,
        stream: bool,
    ) -> Result<Run> {
        let (seq, guard) = {
            let mut data = lock(&self.state.data);
            if data.busy {
                drop(data);
                // The rejected call consumes no seq, so its line carries none.
                self.output(
                    "divergence",
                    json!({"seq":null,"type":"RR-CONCURRENT","key_hash":null,"repeat_index":null,"detail":null}),
                );
                return Err(error("RR-CONCURRENT"));
            }
            data.busy = true;
            let seq = data.seq;
            data.seq += 1;
            (
                seq,
                ModelGuard {
                    state: self.state.clone(),
                },
            )
        };
        let key = self.state.builder.build(operation, request, &self.cache);
        let (repeat, fail, verify_failed, gate_failed) = {
            let mut data = lock(&self.state.data);
            let repeat = key.hash.as_ref().map(|hash| {
                let count = data.repeats.entry(hash.clone()).or_default();
                let i = *count;
                *count += 1;
                i
            });
            (
                repeat,
                data.fail.clone(),
                data.verify_failed,
                data.gate_failed,
            )
        };
        let entry = key.hash.as_ref().zip(repeat).and_then(|(hash, repeat)| {
            self.state
                .reader
                .as_ref()?
                .calls
                .get(&(hash.clone(), repeat))
                .cloned()
        });
        let lookup_time = Instant::now();
        let mut code = None;
        if gate_failed {
            code = Some("RR-GATE-TIMEOUT");
        } else if fail.is_some() {
            code = Some("RR-UNAVAILABLE");
        } else if self.config.mode == "verify" && verify_failed {
            code = Some("RR-TOOL");
        } else if self.config.mode != "record" {
            if key.hash.is_none() {
                code = Some("RR-KEYLESS");
            } else if entry.is_none() {
                let exists = self.state.reader.as_ref().is_some_and(|reader| {
                    reader
                        .calls
                        .keys()
                        .any(|(hash, _)| Some(hash) == key.hash.as_ref())
                });
                code = Some(if exists { "RR-EXHAUSTED" } else { "RR-MISS" });
            } else if entry.as_ref().is_some_and(|e| e.outcome != "ok") {
                code = Some("RR-UNSERVABLE");
            } else if stream && entry.as_ref().is_some_and(|e| !e.stream) {
                code = Some("RR-STREAM-MISMATCH");
            }
        }
        if self.config.mode == "record" && code.is_none() {
            self.append(
                "requests",
                json!({"seq":seq,"key_hash":key.hash,"repeat_index":repeat,"operation":operation,
                "filtered_request":key.filtered,"filter_matches":key.matches}),
                Some(seq),
            );
        }
        let found = if self.config.mode == "record" {
            None
        } else {
            Some(code.is_none())
        };
        let (gate_wait_ms, release) = self.gate(seq, &key, found, code.is_none()).await?;
        if let Some(code) = code {
            let detail = if code == "RR-MISS" {
                self.miss(seq, &key)
            } else {
                json!({"fail_closed_reason":fail,"keyless_reason":key.reason})
            };
            self.divergence(seq, code, key.hash.as_deref(), repeat, detail);
            return Err(error(code));
        }
        let t0 = release.unwrap_or_else(|| {
            if self.config.mode == "record" {
                Instant::now()
            } else {
                lookup_time
            }
        });
        Ok(Run {
            seq,
            key,
            repeat,
            entry,
            guard: Some(guard),
            t0,
            gate_wait_ms,
        })
    }
    pub fn completed_model(&self, run: &Run) {
        lock(&self.state.data).last = Some(Parent {
            seq: run.seq,
            hash: run.key.hash.clone(),
            repeat: run.repeat,
        });
    }

    async fn gate(
        &self,
        seq: u64,
        key: &KeyedRequest,
        found: Option<bool>,
        usable: bool,
    ) -> Result<(u64, Option<Instant>)> {
        {
            let mut data = lock(&self.state.data);
            if data.gate_entered {
                return Ok((0, None));
            }
            data.gate_entered = true;
        }
        let start = Instant::now();
        let unix = trace::now_ns();
        let mono = mono_ns();
        let mut armed = json!({"activation_id":self.id,"mode":self.config.mode,"pid":std::process::id(),
            "key_policy_sha256":self.state.builder.policy_hash,"first_key_hash":key.hash,"found":found,
            "gate_entry_unix_ns":unix,"gate_entry_mono_ns":mono,"gate_outcome":"skipped",
            "gate_release_unix_ns":null,"gate_release_mono_ns":null,"gate_wait_ms":0});
        if usable {
            armed["gate_outcome"] = json!(if self.config.start_gate.is_some() {
                "waiting"
            } else {
                "none"
            });
        }
        let path = self.config.output_directory.join("armed.json");
        if !self.config.output_directory.as_os_str().is_empty()
            && trace::atomic_json(&path, &armed).is_err()
        {
            lock(&self.state.data)
                .fail
                .get_or_insert_with(|| "output_directory_unwritable".into());
            self.divergence(
                seq,
                "RR-UNAVAILABLE",
                key.hash.as_deref(),
                Some(0),
                Json::Null,
            );
            return Err(error("RR-UNAVAILABLE"));
        }
        if !usable {
            return Ok((0, None));
        }
        let mut release = start;
        if let Some(gate) = &self.config.start_gate {
            let deadline = start + Duration::from_secs(gate.timeout_seconds);
            loop {
                match tokio::fs::try_exists(&gate.path).await {
                    Ok(true) => {
                        release = Instant::now();
                        break;
                    }
                    Err(_) => {
                        lock(&self.state.data)
                            .fail
                            .get_or_insert_with(|| "config_invalid".into());
                        armed["gate_outcome"] = json!("unreadable");
                        armed["gate_wait_ms"] = json!(start.elapsed().as_millis() as u64);
                        let _ = trace::atomic_json(&path, &armed);
                        self.divergence(
                            seq,
                            "RR-UNAVAILABLE",
                            key.hash.as_deref(),
                            Some(0),
                            json!({"gate":"unreadable"}),
                        );
                        return Err(error("RR-UNAVAILABLE"));
                    }
                    Ok(false) => {}
                }
                if Instant::now() >= deadline {
                    lock(&self.state.data).gate_failed = true;
                    armed["gate_outcome"] = json!("timed_out");
                    armed["gate_wait_ms"] = json!(start.elapsed().as_millis() as u64);
                    let _ = trace::atomic_json(&path, &armed);
                    self.divergence(
                        seq,
                        "RR-GATE-TIMEOUT",
                        key.hash.as_deref(),
                        Some(0),
                        Json::Null,
                    );
                    return Err(error("RR-GATE-TIMEOUT"));
                }
                tokio::time::sleep_until(
                    (Instant::now() + Duration::from_millis(20)).min(deadline),
                )
                .await;
            }
            armed["gate_outcome"] = json!("released");
        }
        armed["gate_release_unix_ns"] = json!(if self.config.start_gate.is_none() {
            unix
        } else {
            trace::now_ns()
        });
        armed["gate_release_mono_ns"] = json!(if self.config.start_gate.is_none() {
            mono
        } else {
            mono_ns()
        });
        let wait = if self.config.start_gate.is_none() {
            0
        } else {
            release.duration_since(start).as_millis() as u64
        };
        armed["gate_wait_ms"] = json!(wait);
        if trace::atomic_json(&path, &armed).is_err() {
            lock(&self.state.data)
                .fail
                .get_or_insert_with(|| "output_directory_unwritable".into());
            return Err(error("RR-UNAVAILABLE"));
        }
        Ok((wait, self.config.start_gate.as_ref().map(|_| release)))
    }
    fn miss(&self, seq: u64, key: &KeyedRequest) -> Json {
        {
            let mut data = lock(&self.state.data);
            if data.miss_reported {
                return Json::Null;
            }
            data.miss_reported = true;
        }
        let recorded = self
            .state
            .reader
            .as_ref()
            .and_then(|reader| trace::json_lines(&reader.directory.join("requests.jsonl")).ok())
            .and_then(|lines| lines.into_iter().find(|line| line["seq"] == seq));
        let Some(recorded) = recorded else {
            return json!({"recorded":null});
        };
        let operation = recorded["operation"].as_str().unwrap_or_default();
        let prior = self.state.builder.build(
            operation,
            &LlmRequest {
                headers: Default::default(),
                content: recorded["filtered_request"].clone(),
            },
            &self.cache,
        );
        if let (Some(mut a), Some(mut b)) = (prior.document, key.document.clone()) {
            let a_body = a
                .as_object_mut()
                .unwrap()
                .remove("body")
                .unwrap_or_default();
            let b_body = b
                .as_object_mut()
                .unwrap()
                .remove("body")
                .unwrap_or_default();
            first_difference(&a, &b)
                .or_else(|| {
                    first_difference(&a_body, &b_body).map(|mut d| {
                        d["pointer"] = json!(format!(
                            "/body{}",
                            d["pointer"].as_str().unwrap_or_default()
                        ));
                        d
                    })
                })
                .unwrap_or(Json::Null)
        } else {
            json!({"recorded":null,"keyless_reason":key.reason})
        }
    }
}

#[cfg(not(unix))]
pub fn mono_ns() -> u64 {
    static ORIGIN: OnceLock<std::time::Instant> = OnceLock::new();
    ORIGIN
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_nanos() as u64
}

#[cfg(unix)]
pub fn mono_ns() -> u64 {
    let mut value = std::mem::MaybeUninit::<libc::timespec>::uninit();
    // SAFETY: CLOCK_MONOTONIC initializes the timespec on success.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, value.as_mut_ptr()) } != 0 {
        return 0;
    }
    let value = unsafe { value.assume_init() };
    (value.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(value.tv_nsec as u64)
}
