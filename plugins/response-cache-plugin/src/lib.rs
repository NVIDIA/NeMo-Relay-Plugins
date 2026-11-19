// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact-match LLM response and opt-in tool-result cache for NeMo Relay.
//!
//! The plugin installs buffered and streaming LLM execution intercepts and,
//! when `tools.enabled` is set, a tool execution intercept. LLM responses and
//! tool results share one store but use disjoint keys.
//!
//! [`intercept`] holds the LLM cache decisions and streaming tee, [`key`] the
//! key derivation, [`store`] the backends, [`replay`] the streaming replay,
//! [`tool`] the tool-result cache, and [`mark`] the observability surface.

#[cfg(test)]
#[path = "../tests/unit/support.rs"]
pub(crate) mod test_support;

mod anthropic;
pub mod config;
mod error;
mod host;
mod intercept;
mod key;
mod mark;
mod replay;
/// Storage backends. Public for integration tests; not a stable API.
#[doc(hidden)]
pub mod store;
mod tool;
mod validation;

use std::sync::Arc;

use futures::StreamExt;
use nemo_relay_plugin::{
    ConfigDiagnostic, DiagnosticLevel, Json, LlmInputStream, LlmJsonAsyncStream,
    NativeExecutorConfig, Plugin, PluginContext, PluginError,
};
use nemo_relay_types::api::runtime::{
    LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn, LlmStreamInner,
    ToolExecutionNextFn,
};
use nemo_relay_types::error::FlowError;
use serde_json::Map;

pub use config::{
    BackendConfig, ResponseCacheConfig, ResponseCacheKeyStrategy, ToolCacheConfig, ToolClass,
    ToolOverride,
};
pub use error::CacheError;
pub use mark::RESPONSE_CACHE_MARK;

/// Relay plugin ID and component kind.
pub const PLUGIN_KIND: &str = "nvidia.response_cache";

/// The response-cache native plugin.
#[derive(Default)]
pub struct ResponseCachePlugin;

impl ResponseCachePlugin {
    fn diagnostic(code: &str, field: Option<&str>, message: impl Into<String>) -> ConfigDiagnostic {
        ConfigDiagnostic {
            level: DiagnosticLevel::Error,
            code: code.into(),
            component: Some(PLUGIN_KIND.into()),
            field: field.map(str::to_owned),
            message: message.into(),
        }
    }
}

/// Parses a component configuration object.
///
/// # Errors
/// Returns a message when the object does not match the configuration types.
pub fn parse_config(plugin_config: &Map<String, Json>) -> Result<ResponseCacheConfig, String> {
    serde_json::from_value(Json::Object(plugin_config.clone()))
        .map_err(|error| format!("invalid response cache configuration: {error}"))
}

/// Validates a component configuration without loading a Relay host.
#[must_use]
pub fn validate_config(plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
    ResponseCachePlugin.validate(plugin_config)
}

impl Plugin for ResponseCachePlugin {
    fn plugin_kind(&self) -> &str {
        PLUGIN_KIND
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn executor_config(&self) -> Option<NativeExecutorConfig> {
        // Redis connections and the streaming tee need Tokio.
        Some(NativeExecutorConfig { worker_threads: 2 })
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        let mut diagnostics = Vec::new();
        match parse_config(plugin_config) {
            Ok(config) => validation::validate_response_cache(&mut diagnostics, &config),
            Err(message) => diagnostics.push(Self::diagnostic(
                "response_cache.invalid_config",
                None,
                message,
            )),
        }
        if let Err(message) = self.executor_config_for_component(plugin_config) {
            diagnostics.push(Self::diagnostic(
                "response_cache.invalid_executor",
                Some("executor.worker_threads"),
                message,
            ));
        }
        diagnostics
    }

    fn register<'a>(
        &'a self,
        plugin_config: &'a Map<String, Json>,
        context: &'a mut PluginContext<'_>,
    ) -> nemo_relay_plugin::PluginFuture<'a, ()> {
        Box::pin(async move {
            let config = parse_config(plugin_config).map_err(PluginError::InvalidConfig)?;
            host::install(Arc::new(host::RuntimeHost(context.runtime())));
            let store = match store::build_store(&config).await {
                Ok(store) => store,
                Err(CacheError::Storage(error)) => {
                    // The cache is optional: an unreachable backend leaves calls live.
                    if let Some(host) = host::current() {
                        host.warn(
                            "response_cache_store_init_failed",
                            &format!(
                                "Response cache could not initialize its store; managed LLM and \
                                 tool calls will run live: {error}"
                            ),
                        );
                    }
                    return Ok(());
                }
                Err(error) => return Err(PluginError::InvalidConfig(error.to_string())),
            };
            register_intercepts(context, store, config).map_err(PluginError::from)
        })
    }
}

fn register_intercepts(
    context: &mut PluginContext<'_>,
    store: Arc<dyn store::CacheStore>,
    config: ResponseCacheConfig,
) -> nemo_relay_plugin::Result<()> {
    let config = Arc::new(config);
    context.register_llm_execution_intercept("response_cache_llm_execution", config.priority, {
        let store = Arc::clone(&store);
        let config = Arc::clone(&config);
        move |provider, request, _context, next| {
            let next: LlmExecutionNextFn = Arc::new(move |request| {
                let next = next.clone();
                Box::pin(async move { next.call(request).await.map_err(FlowError::Internal) })
            });
            let run = intercept::run_cache(
                provider,
                request,
                next,
                Arc::clone(&store),
                Arc::clone(&config),
            );
            async move { run.await.map_err(sdk_error) }
        }
    })?;
    context.register_llm_stream_execution_intercept(
        "response_cache_llm_stream_execution",
        config.priority,
        {
            let store = Arc::clone(&store);
            let config = Arc::clone(&config);
            move |provider, request, _context, next| {
                let next: LlmStreamExecutionNextFn = Arc::new(move |request| {
                    let next = next.clone();
                    Box::pin(async move {
                        let stream = next.open(request).await.map_err(FlowError::Internal)?;
                        Ok(LlmJsonStream::from_closeable(UpstreamStream(stream)))
                    })
                });
                let run = intercept::run_cache_stream(
                    provider,
                    request,
                    next,
                    Arc::clone(&store),
                    Arc::clone(&config),
                );
                async move {
                    let stream = run.await.map_err(sdk_error)?;
                    let stream: LlmJsonAsyncStream =
                        Box::pin(stream.map(|item| item.map_err(sdk_error)));
                    Ok(stream)
                }
            }
        },
    )?;
    if let Some(tools) = config.tools.clone().filter(|tools| tools.enabled) {
        let tools = Arc::new(tools);
        context.register_tool_execution_intercept(
            "response_cache_tool_execution",
            tools.priority,
            move |context, next| {
                let next: ToolExecutionNextFn = Arc::new(move |args| {
                    let next = next.clone();
                    Box::pin(async move { next.call(args).await.map_err(FlowError::Internal) })
                });
                let run = tool::run_tool_cache(
                    context.tool_name,
                    context.args,
                    next,
                    Arc::clone(&store),
                    Arc::clone(&config),
                    Arc::clone(&tools),
                );
                async move { run.await.map_err(sdk_error) }
            },
        )?;
    }
    Ok(())
}

/// A downstream stream whose close reports the upstream cleanup result.
///
/// The cache stores a streamed answer only after the upstream closes cleanly.
struct UpstreamStream(LlmInputStream);

impl futures::Stream for UpstreamStream {
    type Item = nemo_relay_types::error::Result<Json>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.0)
            .poll_next(cx)
            .map(|item| item.map(|item| item.map_err(FlowError::Internal)))
    }
}

impl LlmStreamInner for UpstreamStream {
    fn close(
        self: std::pin::Pin<&mut Self>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = nemo_relay_types::error::Result<()>> + Send + '_>,
    > {
        let this = self.get_mut();
        Box::pin(async move { this.0.close().await.map_err(FlowError::Internal) })
    }
}

/// Converts a cache error to the SDK's string error without double-wrapping.
///
/// Continuation errors arrive as SDK strings and are carried as
/// `FlowError::Internal`; returning the bare message keeps the caller's error
/// unchanged when the cache passes it through.
fn sdk_error(error: FlowError) -> String {
    match error {
        FlowError::Internal(message) => message,
        other => other.to_string(),
    }
}

nemo_relay_plugin::nemo_relay_plugin!(kind = "nvidia.response_cache", || ResponseCachePlugin);
