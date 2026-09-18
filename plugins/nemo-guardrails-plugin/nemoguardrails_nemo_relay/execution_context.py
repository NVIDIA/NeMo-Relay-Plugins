# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interpret Relay's invocation-scoped LLM codec context.

New Relay hosts tell execution middleware which request and response codecs
actually own the call. Older hosts omit that additive context, so the worker
retains its conservative shape-based compatibility path. A present but opaque
or runtime codec is different from an absent context: silently guessing in
that case would turn an unknown payload into a false policy pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nemo_relay_plugin import AnnotatedLlmRequest

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
    """Relay supplied codec context that this worker cannot verify."""


@dataclass(frozen=True, slots=True)
class DirectionCodec:
    """One request or response codec identity supplied by Relay."""

    kind: str
    codec_name: str | None = None

    def supported_name(self, direction: str) -> str | None:
        """Return a built-in codec name or reject a known unsupported codec."""

        if self.kind == "none":
            return None
        if self.kind == "builtin" and self.codec_name in SUPPORTED_BUILTIN_CODECS:
            return self.codec_name
        if self.kind == "builtin":
            raise ExecutionCodecContextError(f"the Relay {direction} codec is not supported")
        if self.kind in {"opaque", "runtime"}:
            raise ExecutionCodecContextError(f"the Relay {direction} codec cannot prove complete coverage")
        raise ExecutionCodecContextError(f"the Relay {direction} codec identity is invalid")


@dataclass(frozen=True, slots=True)
class ExecutionCodecContext:
    """Authoritative codec identities and normalized request for one call."""

    request: DirectionCodec
    response: DirectionCodec
    annotated_request: AnnotatedLlmRequest | None = None
    allow_shape_inference: bool = False

    @classmethod
    def legacy(cls) -> "ExecutionCodecContext":
        """Return the compatibility context used by pre-contract Relay hosts."""

        return cls(
            DirectionCodec("none"),
            DirectionCodec("none"),
            allow_shape_inference=True,
        )


def _direction_codec(value: object, direction: str) -> DirectionCodec:
    codec = getattr(value, "codec", None)
    kind = getattr(codec, "kind", None)
    codec_name = getattr(codec, "id", None)
    if not isinstance(kind, str):
        raise ExecutionCodecContextError(f"the Relay {direction} codec identity is invalid")
    if codec_name is not None and not isinstance(codec_name, str):
        raise ExecutionCodecContextError(f"the Relay {direction} codec identity is invalid")
    if kind == "none" and codec_name is not None:
        raise ExecutionCodecContextError(f"the Relay {direction} codec identity is invalid")
    if kind in {"builtin", "runtime"} and not codec_name:
        raise ExecutionCodecContextError(f"the Relay {direction} codec identity is invalid")
    if kind == "opaque" and codec_name is not None:
        raise ExecutionCodecContextError(f"the Relay {direction} codec identity is invalid")
    return DirectionCodec(kind, codec_name)


def execution_codec_context(value: Any) -> ExecutionCodecContext:
    """Validate a duck-typed SDK context without requiring a new SDK to import.

    The worker package remains importable with the previous SDK so an older
    Relay host can produce a clear compatibility result. Once the context-aware
    registration method is present, malformed context is always fail-closed.
    """

    request = _direction_codec(getattr(value, "request", None), "request")
    response = _direction_codec(getattr(value, "response", None), "response")
    annotated = getattr(value, "annotated_request", None)
    if annotated is not None and not isinstance(annotated, dict):
        raise ExecutionCodecContextError("the Relay annotated request is invalid")
    return ExecutionCodecContext(request, response, annotated)
