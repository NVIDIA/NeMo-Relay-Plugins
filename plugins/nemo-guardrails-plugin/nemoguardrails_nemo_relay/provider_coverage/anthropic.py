# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anthropic Messages response inspection."""

from __future__ import annotations

from typing import Any

from ..payload_policy import MultimodalPolicy, PayloadPolicy, ReasoningPolicy
from .common import (
    ResponseCandidate,
    ResponseInspection,
    _reject_structural_extensions,
    _reject_unmodeled_fields,
    _UnsupportedRequest,
    text_fragment,
    tool_call,
)

TEXT_PART_METADATA = frozenset({"cache_control", "citations"})
_ROOT_FIELDS = frozenset(
    {"content", "context_management", "id", "model", "role", "stop_reason", "stop_sequence", "type", "usage"}
)


def inspect_response(
    response: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
    codec_variant: str | None = None,
) -> ResponseInspection:
    del codec_variant
    root_extensions = _reject_structural_extensions(response, _ROOT_FIELDS, "Anthropic response")
    if response.get("role") not in (None, "assistant") or response.get("type") not in (None, "message"):
        raise _UnsupportedRequest("Anthropic response role or type is invalid")
    blocks = response.get("content")
    if not isinstance(blocks, list):
        raise _UnsupportedRequest("Anthropic response content is missing")

    fragments = []
    calls = []
    reasoning: list[str] = []
    mutation_safe = not root_extensions
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise _UnsupportedRequest("Anthropic response block is invalid")
        block_type = block.get("type")
        if block_type == "text":
            _reject_unmodeled_fields(
                block,
                {"cache_control", "citations", "text", "type"},
                "Anthropic text block",
            )
            mutation_safe = mutation_safe and block.get("citations") in (None, [], {})
            fragments.append(text_fragment(block.get("text"), ("content", index, "text")))
        elif block_type == "tool_use":
            if set(block) - {"caller", "id", "input", "name", "toolset_name", "type"}:
                raise _UnsupportedRequest("Anthropic tool call contains unsupported fields")
            caller = block.get("caller")
            if caller is not None and caller != {"type": "direct"}:
                raise _UnsupportedRequest("Anthropic hosted tool calls are not covered")
            if block.get("toolset_name") is not None:
                raise _UnsupportedRequest("Anthropic toolset calls are not covered")
            calls.append(tool_call(block.get("id"), block.get("name"), block.get("input")))
        elif block_type == "thinking":
            _reject_unmodeled_fields(
                block,
                {"signature", "thinking", "type"},
                "Anthropic thinking block",
            )
            thought = block.get("thinking")
            if not isinstance(thought, str) or not isinstance(block.get("signature"), str):
                raise _UnsupportedRequest("Anthropic thinking output is invalid")
            if payload_policy.reasoning is ReasoningPolicy.REJECT:
                raise _UnsupportedRequest("Anthropic thinking output is not covered")
            if payload_policy.reasoning is ReasoningPolicy.CHECK_OUTPUT:
                reasoning.append(thought)
        elif block_type == "redacted_thinking":
            _reject_unmodeled_fields(
                block,
                {"data", "type"},
                "Anthropic redacted-thinking block",
            )
            if not isinstance(block.get("data"), str):
                raise _UnsupportedRequest("Anthropic redacted thinking output is invalid")
            if payload_policy.reasoning is ReasoningPolicy.REJECT:
                raise _UnsupportedRequest("Anthropic thinking output is not covered")
            if payload_policy.reasoning is ReasoningPolicy.CHECK_OUTPUT:
                raise _UnsupportedRequest("Anthropic redacted thinking cannot be checked")
        elif block_type in ("image", "document"):
            _reject_unmodeled_fields(
                block,
                {"source", "type"},
                "Anthropic multimodal block",
            )
            if payload_policy.multimodal is not MultimodalPolicy.TEXT_ONLY or not isinstance(block.get("source"), dict):
                raise _UnsupportedRequest("Anthropic multimodal output is not covered")
        else:
            raise _UnsupportedRequest("Anthropic response block is not covered")

    if response.get("stop_reason") == "tool_use" and not calls:
        raise _UnsupportedRequest("Anthropic reported tool use without a tool call")
    candidate = ResponseCandidate(tuple(fragments), tuple(calls), mutation_safe=mutation_safe)
    return ResponseInspection("anthropic_messages", None, (candidate,), tuple(reasoning))
