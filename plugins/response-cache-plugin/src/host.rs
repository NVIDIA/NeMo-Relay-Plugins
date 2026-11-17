// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host services the cache uses: marks, model pricing, and logging.
//!
//! The plugin reaches Relay only through [`PluginRuntime`]. Unit tests install
//! a recording host instead, so cache logic runs without a Relay process.

use std::sync::{Arc, RwLock};

use nemo_relay_plugin::{
    AnnotatedLlmResponse, CostEstimate, Json, PluginLogLevel, PluginRuntime, Usage,
};

/// Relay services used by the cache.
pub(crate) trait Host: Send + Sync {
    /// Emits a mark event in the caller's current scope.
    fn emit_mark(&self, name: &str, data: &Json, metadata: &Json);
    /// Prices token usage with the host's active pricing catalog.
    fn estimate_cost(
        &self,
        provider: Option<&str>,
        model: &str,
        usage: &Usage,
    ) -> Option<CostEstimate>;
    /// Prices a normalized response with the host's full pricing rules.
    fn estimate_response_cost(
        &self,
        response: &AnnotatedLlmResponse,
        provider: Option<&str>,
    ) -> Option<CostEstimate>;
    /// Writes a warning to the host's operational log.
    fn warn(&self, event: &str, message: &str);
}

static HOST: RwLock<Option<Arc<dyn Host>>> = RwLock::new(None);

/// Installs the host used by every cache component in this library.
pub(crate) fn install(host: Arc<dyn Host>) {
    *HOST.write().unwrap_or_else(|error| error.into_inner()) = Some(host);
}

/// Returns the installed host, or `None` before registration.
pub(crate) fn current() -> Option<Arc<dyn Host>> {
    let host = HOST
        .read()
        .unwrap_or_else(|error| error.into_inner())
        .clone();
    #[cfg(test)]
    if host.is_none() {
        return Some(crate::test_support::host());
    }
    host
}

/// [`Host`] backed by the Relay plugin runtime.
pub(crate) struct RuntimeHost(pub(crate) PluginRuntime);

impl Host for RuntimeHost {
    fn emit_mark(&self, name: &str, data: &Json, metadata: &Json) {
        // Marks are observability only; a failed emit must not fail the call.
        let _ = self.0.emit_mark(name, Some(data), Some(metadata));
    }

    fn estimate_cost(
        &self,
        provider: Option<&str>,
        model: &str,
        usage: &Usage,
    ) -> Option<CostEstimate> {
        // A host without the pricing operation reports an unknown cost.
        self.0.estimate_cost(provider, model, usage).ok().flatten()
    }

    fn estimate_response_cost(
        &self,
        response: &AnnotatedLlmResponse,
        provider: Option<&str>,
    ) -> Option<CostEstimate> {
        self.0
            .estimate_response_cost(response, provider)
            .ok()
            .flatten()
    }

    fn warn(&self, event: &str, message: &str) {
        let fields = serde_json::json!({ "event": event });
        let _ = self.0.log(
            PluginLogLevel::Warn,
            "response_cache",
            message,
            Some(&fields),
        );
    }
}
