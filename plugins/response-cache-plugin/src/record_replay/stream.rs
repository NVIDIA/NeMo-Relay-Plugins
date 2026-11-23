// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    session::{Run, Session, error, lock, mono_ns},
    trace::{Call, Chunk},
};
use nemo_relay_types::api::llm::LlmRequest;
use nemo_relay_types::api::runtime::{
    LlmExecutionFn, LlmJsonStream, LlmStreamExecutionFn, LlmStreamInner,
};
use nemo_relay_types::error::Result;
use serde_json::{Value as Json, json};
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Instant, Sleep};
use tokio_stream::{Stream, StreamExt};

pub fn make_buffered(session: Arc<Session>) -> LlmExecutionFn {
    Arc::new(move |operation, request, _context, next| {
        let session = session.clone();
        let operation = operation.to_string();
        Box::pin(async move {
            let run = session.enter(&operation, &request, false).await?;
            if session.config.mode == "record" {
                // A cancelled call still writes its line, so its repeat index
                // is never left out of the trace.
                let mut pending = PendingRecord {
                    session: session.clone(),
                    operation: operation.clone(),
                    request: request.clone(),
                    stream: false,
                    run: Some(run),
                };
                let result = next(request.clone()).await;
                let run = pending.run.take().expect("pending run");
                if result.is_ok() {
                    session.completed_model(&run);
                }
                let total = run.t0.elapsed().as_micros() as u64;
                session.append(
                    "calls",
                    call_line(
                        &operation,
                        &request,
                        &run,
                        RecordedAnswer {
                            stream: false,
                            outcome: if result.is_ok() {
                                "ok"
                            } else {
                                "provider_error"
                            },
                            total,
                            first: None,
                            chunks: None,
                            response: result.as_ref().ok().cloned().unwrap_or(Json::Null),
                        },
                    ),
                    Some(run.seq),
                );
                result
            } else {
                let entry = run.entry.as_ref().expect("matched entry");
                let deadline = checked_deadline(&session, &run, entry.total_us)?;
                let response = entry.response.clone();
                // Writes the timing line even when the caller cancels the wait.
                let mut timing = BufferedTiming {
                    session: session.clone(),
                    run: Some(run),
                    complete: false,
                };
                tokio::time::sleep_until(deadline).await;
                timing.complete = true;
                drop(timing);
                Ok(response)
            }
        })
    })
}

pub fn make_stream(session: Arc<Session>) -> LlmStreamExecutionFn {
    Arc::new(move |operation, request, _context, next| {
        let session = session.clone();
        let operation = operation.to_string();
        Box::pin(async move {
            let run = session.enter(&operation, &request, true).await?;
            if session.config.mode == "record" {
                let mut pending = PendingRecord {
                    session: session.clone(),
                    operation: operation.clone(),
                    request: request.clone(),
                    stream: true,
                    run: Some(run),
                };
                let opened = next(request.clone()).await;
                let run = pending.run.take().expect("pending run");
                let upstream = match opened {
                    Ok(stream) => stream,
                    Err(error) => {
                        session.append(
                            "calls",
                            call_line(
                                &operation,
                                &request,
                                &run,
                                RecordedAnswer {
                                    stream: true,
                                    outcome: "provider_error",
                                    total: run.t0.elapsed().as_micros() as u64,
                                    first: None,
                                    chunks: Some(vec![]),
                                    response: Json::Null,
                                },
                            ),
                            Some(run.seq),
                        );
                        return Err(error);
                    }
                };
                let state = Arc::new(Mutex::new(Recording {
                    session,
                    operation,
                    request,
                    run: Some(run),
                    chunks: vec![],
                    error: false,
                    producer_ended: false,
                    consumer_ended: false,
                    total_us: None,
                }));
                let (chunks, reader) = spawn_reader(upstream, state.clone());
                Ok(LlmJsonStream::from_closeable(RecordStream {
                    chunks,
                    reader: Some(reader),
                    upstream: None,
                    state,
                    finished: false,
                }))
            } else {
                let entry = run.entry.as_ref().expect("matched entry").clone();
                let deadlines: Vec<_> = entry
                    .chunks
                    .as_ref()
                    .expect("stream entry")
                    .iter()
                    .map(|c| c.offset_us)
                    .chain(std::iter::once(entry.total_us))
                    .map(|us| checked_deadline(&session, &run, us))
                    .collect::<Result<_>>()?;
                let t0_mono_ns = mono_ns().saturating_sub(run.t0.elapsed().as_nanos() as u64);
                let state = Arc::new(Mutex::new(Replaying {
                    session,
                    run: Some(run),
                    entry,
                    actual: vec![],
                    producer_ended: false,
                    t0_mono_ns,
                }));
                Ok(LlmJsonStream::from_closeable(ReplayStream {
                    state,
                    deadlines,
                    index: 0,
                    timer: None,
                    finished: false,
                }))
            }
        })
    })
}

/// Writes a `consumer_dropped` line for a recording cancelled before the
/// provider answered or opened its stream.
struct PendingRecord {
    session: Arc<Session>,
    operation: String,
    request: LlmRequest,
    stream: bool,
    run: Option<Run>,
}
impl Drop for PendingRecord {
    fn drop(&mut self) {
        let Some(run) = self.run.take() else {
            return;
        };
        let answer = RecordedAnswer {
            stream: self.stream,
            outcome: "consumer_dropped",
            total: run.t0.elapsed().as_micros() as u64,
            first: None,
            chunks: self.stream.then(Vec::new),
            response: Json::Null,
        };
        self.session.append(
            "calls",
            call_line(&self.operation, &self.request, &run, answer),
            Some(run.seq),
        );
    }
}

struct RecordedAnswer<'a> {
    stream: bool,
    outcome: &'a str,
    total: u64,
    first: Option<u64>,
    chunks: Option<Vec<Chunk>>,
    response: Json,
}
fn call_line(operation: &str, request: &LlmRequest, run: &Run, answer: RecordedAnswer<'_>) -> Json {
    let RecordedAnswer {
        stream,
        outcome,
        total,
        first,
        chunks,
        response,
    } = answer;
    let usage = response.get("usage").cloned().filter(Json::is_object);
    json!({"seq":run.seq,"key_hash":run.key.hash,"repeat_index":run.repeat,"keyless_reason":run.key.reason,
        "operation":operation,"stream":stream,"model":request.content.get("model"),"outcome":outcome,
        "error":if matches!(outcome,"provider_error"|"stream_error"){json!({"class":outcome,"message":"upstream call did not complete"})}else{Json::Null},
        "gate_wait_ms":run.gate_wait_ms,"ttft_us":first,"total_us":total,"chunks":chunks,
        "response":response,"usage_chunk":usage.is_some(),"usage":usage})
}

struct Recording {
    session: Arc<Session>,
    operation: String,
    request: LlmRequest,
    run: Option<Run>,
    chunks: Vec<Chunk>,
    error: bool,
    producer_ended: bool,
    consumer_ended: bool,
    total_us: Option<u64>,
}

/// Asks the reader to stop and hand back the upstream stream.
type ReaderStop = oneshot::Sender<oneshot::Sender<LlmJsonStream>>;

/// Reads the upstream as fast as the provider produces it, so each chunk's
/// time is the provider's and does not include pauses by the caller.
fn spawn_reader(
    mut upstream: LlmJsonStream,
    state: Arc<Mutex<Recording>>,
) -> (mpsc::UnboundedReceiver<Result<Json>>, ReaderStop) {
    let (tx, rx) = mpsc::unbounded_channel();
    let (stop, mut stopped) = oneshot::channel::<oneshot::Sender<LlmJsonStream>>();
    tokio::spawn(async move {
        let mut tx = Some(tx);
        loop {
            let Some(sender) = tx.clone() else {
                break;
            };
            tokio::select! {
                biased;
                request = &mut stopped => {
                    if let Ok(reply) = request {
                        let _ = reply.send(upstream);
                    }
                    return;
                }
                item = upstream.next() => match item {
                    Some(item) => {
                        record_item(&state, &item);
                        let _ = sender.send(item);
                    }
                    None => {
                        // The call stays in flight until the caller reads the
                        // end, as in replay, so both modes admit the next call
                        // at the same point.
                        let mut state = lock(&state);
                        if let Some(run) = state.run.as_ref() {
                            let total = run.t0.elapsed().as_micros() as u64;
                            state.total_us = Some(total);
                            state.producer_ended = true;
                        }
                        tx = None;
                    }
                },
            }
        }
        // Keep the upstream until the caller closes or drops its stream.
        if let Ok(reply) = stopped.await {
            let _ = reply.send(upstream);
        }
    });
    (rx, stop)
}

fn record_item(state: &Arc<Mutex<Recording>>, item: &Result<Json>) {
    let mut state = lock(state);
    match item {
        Ok(data) => {
            if let Some(run) = &state.run {
                let offset = run.t0.elapsed().as_micros() as u64;
                // `response.incomplete` is a normal terminal.
                state.error |= data.get("error").is_some_and(|e| !e.is_null())
                    || matches!(
                        data.get("type").and_then(Json::as_str),
                        Some("response.failed" | "error")
                    );
                state.chunks.push(Chunk {
                    offset_us: offset,
                    data: data.clone(),
                });
            }
        }
        Err(_) => state.error = true,
    }
}

struct RecordStream {
    chunks: mpsc::UnboundedReceiver<Result<Json>>,
    reader: Option<ReaderStop>,
    upstream: Option<LlmJsonStream>,
    state: Arc<Mutex<Recording>>,
    finished: bool,
}
impl Stream for RecordStream {
    type Item = Result<Json>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.finished {
            return Poll::Ready(None);
        }
        match this.chunks.poll_recv(cx) {
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            Poll::Ready(None) => {
                this.finished = true;
                lock(&this.state).consumer_ended = true;
                finalize_record(&this.state);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
impl LlmStreamInner for RecordStream {
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let this = self.get_mut();
        Box::pin(async move {
            if let Some(reader) = this.reader.take() {
                let (reply, upstream) = oneshot::channel();
                if reader.send(reply).is_ok() {
                    this.upstream = upstream.await.ok();
                }
            }
            let result = match this.upstream.as_mut() {
                Some(upstream) => upstream.close().await,
                None => Ok(()),
            };
            if !this.finished {
                this.finished = true;
                finalize_record(&this.state);
            }
            result
        })
    }
}
impl Drop for RecordStream {
    fn drop(&mut self) {
        if !self.finished {
            finalize_record(&self.state);
        }
    }
}

fn finalize_record(state: &Arc<Mutex<Recording>>) {
    let mut state = lock(state);
    let Some(run) = state.run.take() else {
        return;
    };
    let (response, terminal) = aggregate(&state.chunks);
    let outcome = if state.error {
        "stream_error"
    } else if !state.producer_ended || !state.consumer_ended {
        "consumer_dropped"
    } else if terminal {
        "ok"
    } else {
        "stream_error"
    };
    let total = state
        .total_us
        .unwrap_or_else(|| run.t0.elapsed().as_micros() as u64);
    if outcome == "ok" {
        state.session.completed_model(&run);
    }
    let first = state.chunks.first().map(|c| c.offset_us);
    let chunks = std::mem::take(&mut state.chunks);
    state.session.append(
        "calls",
        call_line(
            &state.operation,
            &state.request,
            &run,
            RecordedAnswer {
                stream: true,
                outcome,
                total,
                first,
                chunks: Some(chunks),
                response,
            },
        ),
        Some(run.seq),
    );
}

struct Replaying {
    session: Arc<Session>,
    run: Option<Run>,
    entry: Call,
    actual: Vec<u64>,
    producer_ended: bool,
    t0_mono_ns: u64,
}
struct ReplayStream {
    state: Arc<Mutex<Replaying>>,
    deadlines: Vec<Instant>,
    index: usize,
    timer: Option<Pin<Box<Sleep>>>,
    finished: bool,
}
impl Stream for ReplayStream {
    type Item = Result<Json>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.finished {
            return Poll::Ready(None);
        }
        let timer = this
            .timer
            .get_or_insert_with(|| Box::pin(tokio::time::sleep_until(this.deadlines[this.index])));
        if timer.as_mut().poll(cx).is_pending() {
            return Poll::Pending;
        }
        this.timer = None;
        let result = {
            let mut state = lock(&this.state);
            let actual = state
                .run
                .as_ref()
                .map(|r| r.t0.elapsed().as_micros() as u64)
                .unwrap_or_default();
            state.actual.push(actual);
            state
                .entry
                .chunks
                .as_ref()
                .and_then(|chunks| chunks.get(this.index))
                .map(|c| c.data.clone())
        };
        this.index += 1;
        if let Some(data) = result {
            Poll::Ready(Some(Ok(data)))
        } else {
            this.finished = true;
            {
                let mut state = lock(&this.state);
                state.producer_ended = true;
                if let Some(run) = state.run.as_mut() {
                    run.guard.take();
                }
            }
            finalize_replay(&this.state);
            Poll::Ready(None)
        }
    }
}
impl LlmStreamInner for ReplayStream {
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let this = self.get_mut();
        if !this.finished {
            this.finished = true;
            finalize_replay(&this.state);
        }
        Box::pin(async { Ok(()) })
    }
}
impl Drop for ReplayStream {
    fn drop(&mut self) {
        if !self.finished {
            finalize_replay(&self.state);
        }
    }
}

fn finalize_replay(state: &Arc<Mutex<Replaying>>) {
    let mut state = lock(state);
    let Some(run) = state.run.take() else {
        return;
    };
    let chunks = state.entry.chunks.as_ref().expect("stream entry");
    let complete = state.producer_ended;
    if complete {
        state.session.completed_model(&run);
    }
    // Timing describes the paced stream's own releases.
    let actual = &state.actual;
    let scale = state.session.config.latency_scale;
    let released = actual
        .len()
        .saturating_sub(usize::from(state.producer_ended))
        .min(chunks.len());
    let mut lateness: Vec<_> = actual
        .iter()
        .take(released)
        .zip(chunks)
        .map(|(actual, chunk)| actual.saturating_sub(scaled_us(chunk.offset_us, scale)))
        .collect();
    let end = if complete {
        actual.last().copied()
    } else {
        Some(run.t0.elapsed().as_micros() as u64)
    };
    if let Some(end) = end {
        lateness.push(end.saturating_sub(scaled_us(state.entry.total_us, scale)));
    }
    lateness.sort_unstable();
    let quantile = |q: f64| {
        lateness
            .get(((lateness.len().saturating_sub(1)) as f64 * q) as usize)
            .copied()
            .unwrap_or(0)
    };
    state.session.output("timing",json!({"seq":run.seq,"key_hash":run.key.hash,"repeat_index":run.repeat,"stream":true,
        "outcome":if complete{"complete"}else{"consumer_dropped"},"t0_mono_ns":state.t0_mono_ns,
        "scheduled_first_us":state.entry.ttft_us.map(|v|scaled_us(v,scale)),"actual_first_us":actual.first().filter(|_|released>0),
        "scheduled_end_us":scaled_us(state.entry.total_us,scale),"actual_end_us":end,
        "lateness_us":{"p50":quantile(0.5),"p99":quantile(0.99),"max":lateness.last().copied().unwrap_or(0)},
        "chunks":released,"gate_wait_ms":run.gate_wait_ms}));
}

/// Computes a replay deadline, reporting an overflow as a divergence line.
fn checked_deadline(session: &Session, run: &Run, us: u64) -> Result<Instant> {
    deadline(run.t0, us, session.config.latency_scale).inspect_err(|_| {
        session.divergence(
            run.seq,
            "RR-UNAVAILABLE",
            run.key.hash.as_deref(),
            run.repeat,
            json!({"reason":"deadline_overflow","recorded_us":us}),
        );
    })
}

struct BufferedTiming {
    session: Arc<Session>,
    run: Option<Run>,
    complete: bool,
}
impl Drop for BufferedTiming {
    fn drop(&mut self) {
        let Some(run) = self.run.take() else {
            return;
        };
        let session = &self.session;
        let entry = run.entry.as_ref().expect("matched entry");
        let actual = run.t0.elapsed().as_micros() as u64;
        let scheduled = scaled_us(entry.total_us, session.config.latency_scale);
        if self.complete {
            session.completed_model(&run);
        }
        let late = actual.saturating_sub(scheduled);
        session.output("timing",json!({"seq":run.seq,"key_hash":run.key.hash,"repeat_index":run.repeat,"stream":false,
            "outcome":if self.complete{"complete"}else{"consumer_dropped"},"t0_mono_ns":mono_ns().saturating_sub(run.t0.elapsed().as_nanos() as u64),
            "scheduled_first_us":null,"actual_first_us":null,"scheduled_end_us":scheduled,"actual_end_us":actual,
            "lateness_us":{"p50":late,"p99":late,"max":late},
            "chunks":0,"gate_wait_ms":run.gate_wait_ms}));
    }
}

fn deadline(t0: Instant, us: u64, scale: f64) -> Result<Instant> {
    let duration = Duration::try_from_secs_f64(us as f64 * scale / 1_000_000.0)
        .map_err(|_| error("RR-UNAVAILABLE"))?;
    t0.checked_add(duration)
        .ok_or_else(|| error("RR-UNAVAILABLE"))
}
fn scaled_us(us: u64, scale: f64) -> u64 {
    (us as f64 * scale).min(u64::MAX as f64) as u64
}

pub fn aggregate(chunks: &[Chunk]) -> (Json, bool) {
    if chunks.iter().any(|chunk| {
        chunk
            .data
            .get("type")
            .and_then(Json::as_str)
            .is_some_and(|kind| kind.starts_with("response."))
    }) {
        return aggregate_responses(chunks);
    }
    let mut response = json!({"object":"chat.completion"});
    let mut choices: BTreeMap<u64, Json> = BTreeMap::new();
    let mut tools: BTreeMap<(u64, u64), Json> = BTreeMap::new();
    let mut terminal = false;
    for chunk in chunks {
        let data = &chunk.data;
        for key in ["id", "model", "created", "system_fingerprint"] {
            if response.get(key).is_none()
                && let Some(value) = data.get(key)
            {
                response[key] = value.clone();
            }
        }
        if let Some(usage) = data.get("usage").filter(|u| u.is_object()) {
            response["usage"] = usage.clone();
        }
        for choice in data
            .get("choices")
            .and_then(Json::as_array)
            .into_iter()
            .flatten()
        {
            let index = choice.get("index").and_then(Json::as_u64).unwrap_or(0);
            let output = choices
                .entry(index)
                .or_insert_with(|| json!({"index":index,"message":{}}));
            if let Some(delta) = choice.get("delta").and_then(Json::as_object) {
                merge_chat_delta(index, delta, &mut output["message"], &mut tools);
            }
            for key in ["finish_reason", "logprobs"] {
                if let Some(value) = choice.get(key) {
                    output[key] = value.clone();
                    if key == "finish_reason" && !value.is_null() {
                        terminal = true;
                    }
                }
            }
        }
    }
    for (index, choice) in choices.iter_mut() {
        let calls: Vec<_> = tools
            .iter()
            .filter(|((choice, _), _)| choice == index)
            .map(|(_, value)| value.clone())
            .collect();
        if !calls.is_empty() {
            choice["message"]["tool_calls"] = json!(calls);
        }
    }
    response["choices"] = json!(choices.into_values().collect::<Vec<_>>());
    (response, terminal)
}

/// Builds a Responses aggregate the way a stream consumer assembles it:
/// the terminal `response` object, with `output` rebuilt from the
/// `response.output_item.done` items in `output_index` order.
fn aggregate_responses(chunks: &[Chunk]) -> (Json, bool) {
    let mut base = Json::Null;
    let mut terminal = None;
    let mut done: Vec<(Option<u64>, usize, Json)> = vec![];
    let mut pending: Vec<(Option<u64>, usize, Json)> = vec![];
    let mut arrival = 0;
    let index_of = |data: &Json| data.get("output_index").and_then(Json::as_u64);
    let matches = |entry: &(Option<u64>, usize, Json), data: &Json| {
        let id = data
            .get("item_id")
            .or_else(|| data.pointer("/item/id"))
            .and_then(Json::as_str);
        match (id, entry.2.get("id").and_then(Json::as_str)) {
            (Some(a), Some(b)) => a == b,
            _ => index_of(data).is_some() && entry.0 == index_of(data),
        }
    };
    for chunk in chunks {
        let data = &chunk.data;
        let response = data.get("response").filter(|r| r.is_object());
        match data.get("type").and_then(Json::as_str) {
            Some("response.created" | "response.in_progress") if terminal.is_none() => {
                if let Some(response) = response {
                    base = response.clone();
                }
            }
            Some(kind @ ("response.completed" | "response.incomplete" | "response.failed")) => {
                if let Some(response) = response {
                    base = response.clone();
                }
                terminal = Some(kind);
            }
            Some("response.output_item.added")
                if data.pointer("/item/type").and_then(Json::as_str) == Some("function_call") =>
            {
                pending.push((index_of(data), arrival, data["item"].clone()));
                arrival += 1;
            }
            Some("response.function_call_arguments.delta") => {
                if let (Some(entry), Some(delta)) = (
                    pending.iter_mut().find(|entry| matches(entry, data)),
                    data.get("delta").and_then(Json::as_str),
                ) {
                    let mut text = entry.2["arguments"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string();
                    text.push_str(delta);
                    entry.2["arguments"] = json!(text);
                }
            }
            Some("response.output_item.done") => {
                if let Some(item) = data.get("item") {
                    let announced = pending
                        .iter()
                        .position(|entry| matches(entry, data))
                        .map(|i| pending.remove(i).1);
                    done.push((index_of(data), announced.unwrap_or(arrival), item.clone()));
                    arrival += 1;
                }
            }
            _ => {}
        }
    }
    // Consumers settle announced but unfinished calls only after a completed response.
    if terminal == Some("response.completed") {
        done.extend(pending);
    }
    if !done.is_empty() && base.is_object() {
        if done.iter().all(|entry| entry.0.is_some()) {
            done.sort_by_key(|entry| entry.0);
        } else {
            done.sort_by_key(|entry| entry.1);
        }
        base["output"] = json!(done.into_iter().map(|entry| entry.2).collect::<Vec<_>>());
    }
    let ok =
        matches!(terminal, Some("response.completed" | "response.incomplete")) && base.is_object();
    (base, ok)
}

fn merge_chat_delta(
    index: u64,
    delta: &serde_json::Map<String, Json>,
    message: &mut Json,
    tools: &mut BTreeMap<(u64, u64), Json>,
) {
    for (key, value) in delta {
        if key == "tool_calls" {
            merge_tool_deltas(index, value, tools);
        } else if key == "role" {
            if message.get(key).is_none() {
                message[key] = value.clone();
            }
        } else if value.is_null() && message.get(key).is_some() {
            // A later null delta never erases accumulated text.
        } else if let Some(text) = value.as_str() {
            let mut accumulated = message[key].as_str().unwrap_or_default().to_string();
            accumulated.push_str(text);
            message[key] = json!(accumulated);
        } else {
            message[key] = value.clone();
        }
    }
}

fn merge_tool_deltas(index: u64, value: &Json, tools: &mut BTreeMap<(u64, u64), Json>) {
    for call in value.as_array().into_iter().flatten() {
        let tool_index = call["index"].as_u64().unwrap_or(0);
        let tool = tools
            .entry((index, tool_index))
            .or_insert_with(|| json!({"function":{}}));
        for key in ["id", "type"] {
            if tool.get(key).is_none()
                && let Some(value) = call.get(key)
            {
                tool[key] = value.clone();
            }
        }
        if tool["function"].get("name").is_none()
            && let Some(name) = call.pointer("/function/name")
        {
            tool["function"]["name"] = name.clone();
        }
        if let Some(args) = call.pointer("/function/arguments").and_then(Json::as_str) {
            let mut text = tool["function"]["arguments"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            text.push_str(args);
            tool["function"]["arguments"] = json!(text);
        }
    }
}
