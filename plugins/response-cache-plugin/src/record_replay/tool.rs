// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    session::{Session, State, lock},
    trace::tool_key,
};
use crate::key::fingerprint;
use nemo_relay_types::api::runtime::ToolExecutionFn;
use nemo_relay_types::api::tool::ToolExecutionInterceptOutcome;
use serde_json::{Value as Json, json};
use std::sync::Arc;
use std::time::Instant;

struct ToolGuard {
    state: Arc<State>,
    token: u64,
}
impl Drop for ToolGuard {
    fn drop(&mut self) {
        lock(&self.state.data).tools_active.remove(&self.token);
    }
}

pub fn make_tool(session: Arc<Session>) -> ToolExecutionFn {
    Arc::new(move |context, next| {
        let session = session.clone();
        Box::pin(async move {
            let name = context.tool_name().to_string();
            let args = context.args().clone();
            let id = context.tool_call_id().map(str::to_string);
            let failed = {
                let mut data = lock(&session.state.data);
                data.tool_calls += 1;
                data.fail.is_some()
            };
            if failed {
                // Tools always run live; fail-closed only refuses model
                // calls, so the tool runs unrecorded and uncompared.
                return next(args)
                    .await
                    .map(|result| ToolExecutionInterceptOutcome {
                        result: result.result,
                        annotation: result.annotation,
                        pending_marks: vec![],
                    });
            }
            let (token, parent, ordinal) = {
                let mut data = lock(&session.state.data);
                let token = data.tool_calls - 1;
                let overlap = !data.tools_active.is_empty();
                if overlap {
                    for value in data.tools_active.values_mut() {
                        *value = true;
                    }
                }
                data.tools_active.insert(token, overlap);
                let parent = data.last.clone();
                let key = json!([
                    parent.as_ref().and_then(|p| p.hash.clone()),
                    parent.as_ref().and_then(|p| p.repeat),
                    name
                ])
                .to_string();
                let ordinal = data.ordinals.entry(key).or_default();
                let index = *ordinal;
                *ordinal += 1;
                (token, parent, index)
            };
            let _guard = ToolGuard {
                state: session.state.clone(),
                token,
            };
            let t0 = Instant::now();
            let result = next(args.clone()).await;
            let wall_us = t0.elapsed().as_micros() as u64;
            let overlap = lock(&session.state.data)
                .tools_active
                .remove(&token)
                .unwrap_or(false);
            let (kind, value) = match &result {
                Ok(result) => {
                    // Inspect JSON text for signatures while returning its original
                    // value untouched, including non-object JSON outcomes.
                    let parsed = result
                        .result
                        .as_str()
                        .and_then(|text| serde_json::from_str::<Json>(text).ok());
                    let unparsed_text = result.result.is_string() && parsed.is_none();
                    let value = parsed.unwrap_or_else(|| result.result.clone());
                    (
                        if value.is_object() {
                            "json_object"
                        } else if unparsed_text {
                            "text"
                        } else {
                            "json_other"
                        },
                        value,
                    )
                }
                Err(_) => ("raised", Json::Null),
            };
            // Only tools named in `divergence.tool_names` have a checked shape.
            let shape = session
                .config
                .divergence
                .tool_names
                .contains(&name)
                .then(|| {
                    kind == "json_object"
                        && !value
                            .get("error")
                            .and_then(Json::as_str)
                            .is_some_and(|s| s.starts_with("[TOOL_ERROR]"))
                });
            let fields: serde_json::Map<String, Json> = session
                .config
                .divergence
                .tool_fields
                .iter()
                .map(|pointer| {
                    let field = value.pointer(pointer);
                    let field = if session.config.divergence.flag_fields.contains(pointer) {
                        json!(field.is_some_and(|v| !v.is_null()
                            && v != &Json::Bool(false)
                            && v != &Json::String(String::new())))
                    } else {
                        field.cloned().unwrap_or_else(|| json!("<absent>"))
                    };
                    (pointer.clone(), field)
                })
                .collect();
            let bytes = if kind == "raised" {
                0
            } else if let Some(text) = result.as_ref().ok().and_then(|r| r.result.as_str()) {
                text.len()
            } else {
                serde_json_canonicalizer::to_vec(&value)
                    .map(|v| v.len())
                    .unwrap_or(0)
            };
            let mut line = json!({"parent_seq":parent.as_ref().map(|p|p.seq),"parent_key_hash":parent.as_ref().and_then(|p|p.hash.clone()),
                "parent_repeat_index":parent.as_ref().and_then(|p|p.repeat),"tool_call_id":id,"tool_name":name,"ordinal":ordinal,
                "match":if id.is_some(){"id"}else{"fallback"},"args_sha256":fingerprint(&args),
                "result_kind":kind,"shape_ok":shape,"fields":fields,"output_bytes":bytes,"wall_us":wall_us,"overlapped":overlap});
            if session.config.mode == "record" {
                session.append("tools", line, parent.as_ref().map(|p| p.seq));
            } else {
                let key = tool_key(&line).expect("canonical tool signature");
                let duplicate = !lock(&session.state.data).executed.insert(key.clone());
                let expected = session
                    .state
                    .reader
                    .as_ref()
                    .and_then(|reader| reader.tools.get(&key));
                let mut problems = vec![];
                let mut status = "ok";
                if duplicate {
                    status = "duplicate";
                    problems.push("tool_duplicate");
                } else if expected.is_some_and(|e| e["tool_name"] != line["tool_name"]) {
                    status = "mismatch";
                    problems.push("tool_name_mismatch");
                } else if shape == Some(false) {
                    status = "shape";
                    problems.push("tool_shape");
                } else if let Some(expected) = expected {
                    if expected["args_sha256"] != line["args_sha256"] {
                        problems.push("tool_args_mismatch");
                    }
                    if expected["result_kind"] != line["result_kind"]
                        || expected["fields"] != line["fields"]
                    {
                        problems.push("tool_outcome_mismatch");
                    }
                    if !problems.is_empty() {
                        status = "mismatch";
                    } else {
                        let a = expected["output_bytes"].as_u64().unwrap_or(0) as f64;
                        let b = bytes as f64;
                        if (a - b).abs() / a.max(b).max(1.0)
                            > session.config.divergence.output_size_tolerance
                        {
                            status = "size_warning";
                            problems.push("tool_size");
                        }
                    }
                } else {
                    status = "unmatched";
                    problems.push("tool_unmatched");
                }
                for problem in &problems {
                    session.divergence(parent.as_ref().map(|p|p.seq).unwrap_or(0),problem,parent.as_ref().and_then(|p|p.hash.as_deref()),parent.as_ref().and_then(|p|p.repeat),json!({"tool_name":name,"tool_call_id":id,"expected":expected,"actual":line}));
                }
                if session.config.mode == "verify" && problems.iter().any(|p| *p != "tool_size") {
                    lock(&session.state.data).verify_failed = true;
                }
                line["status"] = json!(status);
                session.output("tools", line);
            }
            result.map(|result| ToolExecutionInterceptOutcome {
                result: result.result,
                annotation: result.annotation,
                pending_marks: vec![],
            })
        })
    })
}
