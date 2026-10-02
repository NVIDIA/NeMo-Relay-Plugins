# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-pass inspection of supported provider response shapes."""

from __future__ import annotations

from collections.abc import Callable

from ..payload_policy import PayloadPolicy
from . import anthropic, gemini, oci, openai_chat, openai_responses
from .common import ResponseInspection, _UnsupportedRequest

_Inspector = Callable[..., ResponseInspection]
_INSPECTORS: dict[str, _Inspector] = {
    "openai_chat": openai_chat.inspect_response,
    "openai_responses": openai_responses.inspect_response,
    "anthropic_messages": anthropic.inspect_response,
    "gemini_generate_content": gemini.inspect_response,
    "oci_genai": oci.inspect_response,
}


def inspect_response(
    codec_name: str,
    codec_variant: str | None,
    response: object,
    payload_policy: PayloadPolicy,
) -> ResponseInspection:
    if not isinstance(response, dict):
        raise _UnsupportedRequest("provider response is not an object")
    inspector = _INSPECTORS.get(codec_name)
    if inspector is None:
        raise _UnsupportedRequest("response codec is not supported")
    return inspector(response, payload_policy=payload_policy, codec_variant=codec_variant)


__all__ = ["ResponseInspection", "inspect_response"]
