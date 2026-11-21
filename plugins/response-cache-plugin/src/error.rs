// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error type for cache configuration and storage operations.

use thiserror::Error;

/// An error from cache configuration or a storage backend.
#[derive(Debug, Error)]
pub enum CacheError {
    /// The configuration is invalid.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    /// A storage operation failed.
    #[error("storage error: {0}")]
    Storage(String),
    /// A stored value could not be serialized or deserialized.
    #[error("serialization error: {0}")]
    Serialization(serde_json::Error),
    /// An internal error, such as lock poisoning.
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<serde_json::Error> for CacheError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value)
    }
}

/// Result type for cache operations.
pub type Result<T> = std::result::Result<T, CacheError>;
