# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate Relay's invocation-scoped LLM codec capabilities."""

from __future__ import annotations

from dataclasses import dataclass

from nemo_relay_plugin import (
    LlmCodecIdentity,
    LlmExecutionContext,
    WorkerRequestCodec,
    WorkerResponseCodec,
)

SUPPORTED_BUILTIN_CODECS = frozenset(
    {
        "anthropic_messages",
        "gemini_generate_content",
        "oci_genai",
        "openai_chat",
        "openai_responses",
    }
)


class ExecutionCodecContextError(ValueError):
    """Relay did not provide a usable execution-codec capability."""


@dataclass(frozen=True, slots=True)
class DirectionCodec:
    """One request or response codec identity supplied by Relay."""

    identity: LlmCodecIdentity

    def builtin_name(self, direction: str) -> str | None:
        """Return a reviewed built-in name, or ``None`` for runtime/opaque codecs."""

        if self.identity.kind in {"runtime", "opaque"}:
            return None
        if self.identity.kind == "builtin" and self.identity.id in SUPPORTED_BUILTIN_CODECS:
            return self.identity.id
        if self.identity.kind == "builtin":
            raise ExecutionCodecContextError(f"the Relay {direction} codec is not supported")
        if self.identity.kind == "none":
            raise ExecutionCodecContextError(f"the Relay {direction} codec is unavailable")
        raise ExecutionCodecContextError(f"the Relay {direction} codec identity is invalid")


@dataclass(frozen=True, slots=True)
class ExecutionCodecContext:
    """Authoritative identities and invocation-scoped codec proxies for one call."""

    request: DirectionCodec
    response: DirectionCodec
    request_codec: WorkerRequestCodec
    response_codec: WorkerResponseCodec | None


def execution_codec_context(value: LlmExecutionContext) -> ExecutionCodecContext:
    """Validate the context-aware worker SDK value for policy execution."""

    request_codec = value.request_codec.resolve_codec()
    if request_codec is None:
        raise ExecutionCodecContextError("the Relay request codec capability is unavailable")
    response_context = value.response_codec
    return ExecutionCodecContext(
        DirectionCodec(value.request_codec.codec),
        DirectionCodec(response_context.codec if response_context is not None else LlmCodecIdentity("none")),
        request_codec,
        response_context.resolve_codec() if response_context is not None else None,
    )
