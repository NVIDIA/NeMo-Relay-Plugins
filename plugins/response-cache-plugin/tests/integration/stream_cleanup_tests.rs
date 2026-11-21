// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A failed upstream stream close through the response-cache plugin.
//!
//! When an upstream close fails, Relay keeps the plugin loaded because it
//! cannot prove a safe unload. That leaves this process's test host unusable
//! for other tests, so these checks run in their own binary.

#[path = "../support/plugin_host.rs"]
mod plugin_host_test_support;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use nemo_relay_response_cache_plugin::ResponseCacheConfig;
use nemo_relay_runtime::api::llm::{
    LlmRequest, LlmStreamCallExecuteParams, llm_stream_call_execute,
};
use nemo_relay_runtime::api::runtime::{LlmJsonStream, LlmStreamExecutionNextFn, LlmStreamInner};
use nemo_relay_types::error::FlowError;
use serde_json::{Value as Json, json};
use tokio_stream::{Stream, StreamExt};

#[path = "response_cache_common.rs"]
mod response_cache_common;
use response_cache_common::{activate_cache, chat_request};

fn scoped_cache_config() -> ResponseCacheConfig {
    ResponseCacheConfig {
        namespace: "response-cache-stream-cleanup-test".to_string(),
        ..ResponseCacheConfig::default()
    }
}

struct CloseTrackingStream {
    stream: Pin<Box<dyn Stream<Item = Result<Json, FlowError>> + Send>>,
    close_calls: Arc<AtomicUsize>,
    close_error: Option<FlowError>,
    closed: bool,
}

impl Stream for CloseTrackingStream {
    type Item = Result<Json, FlowError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.closed {
            Poll::Ready(None)
        } else {
            self.stream.as_mut().poll_next(cx)
        }
    }
}

impl LlmStreamInner for CloseTrackingStream {
    fn close(
        mut self: Pin<&mut Self>,
    ) -> Pin<Box<dyn Future<Output = Result<(), FlowError>> + Send + '_>> {
        self.closed = true;
        self.close_calls.fetch_add(1, Ordering::SeqCst);
        let close_error = self.close_error.clone();
        Box::pin(async move { close_error.map_or(Ok(()), Err) })
    }
}

fn close_tracking_stream_provider(
    calls: Arc<AtomicUsize>,
    close_calls: Arc<AtomicUsize>,
    chunks: Vec<Json>,
    close_error: Option<FlowError>,
) -> LlmStreamExecutionNextFn {
    Arc::new(move |_req: LlmRequest| {
        let calls = Arc::clone(&calls);
        let close_calls = Arc::clone(&close_calls);
        let chunks = chunks.clone();
        let close_error = close_error.clone();
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(LlmJsonStream::from_closeable(CloseTrackingStream {
                stream: Box::pin(tokio_stream::iter(
                    chunks.into_iter().map(Ok::<Json, FlowError>),
                )),
                close_calls,
                close_error,
                closed: false,
            }))
        })
    })
}

async fn start_stream(provider: &LlmStreamExecutionNextFn, request: LlmRequest) -> LlmJsonStream {
    llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("openai")
            .request(request)
            .func(provider.clone())
            .collector(Box::new(|_chunk| Ok(())))
            .finalizer(Box::new(|| json!({"done": true})))
            .model_name("gpt-4o")
            .build(),
    )
    .await
    .unwrap()
}

fn openai_chat_stream_chunks() -> Vec<Json> {
    vec![
        json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1_700_000_000, "model": "gpt-4o", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}),
        json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "The answer "}, "finish_reason": null}]}),
        json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "is 42."}, "finish_reason": null}]}),
        json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}),
    ]
}

/// A failed upstream close keeps the answer out of the cache and reaches the caller.
///
/// Both cases share one activation: after a failed close Relay keeps the plugin
/// loaded, so this binary cannot start a second one.
#[tokio::test]
async fn failed_upstream_closes_never_populate_the_cache() {
    activate_cache(scoped_cache_config()).await;

    // A fully drained stream whose upstream close fails is not stored.
    let calls = Arc::new(AtomicUsize::new(0));
    let close_calls = Arc::new(AtomicUsize::new(0));
    let provider = close_tracking_stream_provider(
        Arc::clone(&calls),
        Arc::clone(&close_calls),
        openai_chat_stream_chunks(),
        Some(FlowError::NotFound("drained cleanup failed".into())),
    );
    let mut drained = start_stream(&provider, chat_request("drained cleanup failure")).await;
    let mut chunks = 0;
    let mut failure = None;
    while let Some(item) = drained.next().await {
        match item {
            Ok(_) => chunks += 1,
            Err(error) => failure = Some(error),
        }
    }
    assert_eq!(chunks, openai_chat_stream_chunks().len());
    let failure = failure.expect("the caller must see the upstream close failure");
    assert!(
        matches!(&failure, FlowError::NotFound(message) if message == "drained cleanup failed"),
        "{failure:?}"
    );
    let _ = drained.close().await;
    // Native plugin streams call the provider on their first read.
    let mut repeat = start_stream(&provider, chat_request("drained cleanup failure")).await;
    while repeat.next().await.is_some() {}
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "the repeat must reach the provider because nothing was stored"
    );
    let _ = repeat.close().await;

    // Closing early reports the failure, typed, on every close attempt.
    let calls = Arc::new(AtomicUsize::new(0));
    let close_calls = Arc::new(AtomicUsize::new(0));
    let provider = close_tracking_stream_provider(
        Arc::clone(&calls),
        Arc::clone(&close_calls),
        openai_chat_stream_chunks(),
        Some(FlowError::NotFound("cache tee cleanup failed".into())),
    );
    let mut first = start_stream(&provider, chat_request("cleanup failure")).await;
    assert!(first.next().await.unwrap().is_ok());
    for _ in 0..2 {
        let error = first.close().await.expect_err("close should fail");
        assert!(
            error.to_string().contains("cache tee cleanup failed"),
            "{error}"
        );
        assert!(matches!(error, FlowError::NotFound(_)), "{error:?}");
    }
    assert!(first.next().await.is_none());
    assert_eq!(close_calls.load(Ordering::SeqCst), 1);

    let mut second = start_stream(&provider, chat_request("cleanup failure")).await;
    assert!(second.next().await.unwrap().is_ok());
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "an upstream cleanup failure must prevent the cache write"
    );
    assert!(second.close().await.is_err());
    assert_eq!(close_calls.load(Ordering::SeqCst), 2);
}
