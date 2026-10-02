# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Chat response inspection."""

from __future__ import annotations

from typing import Any

from ..payload_policy import MultimodalPolicy, PayloadPolicy, ReasoningPolicy
from ..structural_tools import ToolCall
from .common import (
    MAX_RESPONSE_SEGMENTS,
    ResponseCandidate,
    ResponseInspection,
    _reject_structural_extensions,
    _reject_unmodeled_fields,
    _UnsupportedRequest,
    text_fragment,
    tool_call,
)

_ROOT_FIELDS = frozenset({"choices", "created", "id", "model", "object", "service_tier", "system_fingerprint", "usage"})
_CHOICE_FIELDS = frozenset({"finish_reason", "index", "logprobs", "message"})


def _calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        return ()
    if not isinstance(raw_calls, list):
        raise _UnsupportedRequest("OpenAI Chat tool_calls must be an array")
    calls = []
    for raw in raw_calls:
        if (
            not isinstance(raw, dict)
            or set(raw) - {"function", "id", "type"}
            or raw.get("type", "function") != "function"
        ):
            raise _UnsupportedRequest("OpenAI Chat output contains an unsupported tool call")
        function = raw.get("function")
        if not isinstance(function, dict) or set(function) - {"arguments", "name"} or "arguments" not in function:
            raise _UnsupportedRequest("OpenAI Chat function call is incomplete")
        calls.append(tool_call(raw.get("id"), function.get("name"), function["arguments"]))
    return tuple(calls)


def inspect_response(
    response: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
    codec_variant: str | None = None,
) -> ResponseInspection:
    del codec_variant
    root_extensions = _reject_structural_extensions(response, _ROOT_FIELDS, "OpenAI Chat response")
    choices = response.get("choices")
    if not isinstance(choices, list):
        raise _UnsupportedRequest("OpenAI Chat output choices are missing")
    if len(choices) > MAX_RESPONSE_SEGMENTS:
        raise _UnsupportedRequest("OpenAI Chat output has too many candidates")

    candidates: list[ResponseCandidate] = []
    reasoning: list[str] = []
    for choice_index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            raise _UnsupportedRequest("OpenAI Chat response choice is invalid")
        _reject_unmodeled_fields(choice, _CHOICE_FIELDS, "OpenAI Chat response choice")
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") not in (None, "assistant"):
            raise _UnsupportedRequest("OpenAI Chat response message is invalid")
        if message.get("function_call") is not None:
            raise _UnsupportedRequest("legacy OpenAI function calls are not covered")
        known = {
            "annotations",
            "audio",
            "content",
            "function_call",
            "reasoning_content",
            "refusal",
            "role",
            "tool_calls",
        }
        _reject_unmodeled_fields(message, known, "OpenAI Chat response message")

        calls = _calls(message)
        mutation_safe = not root_extensions
        mutation_safe = mutation_safe and choice.get("logprobs") in (None, [], {})
        mutation_safe = mutation_safe and message.get("annotations") in (None, [], {})
        if choice.get("finish_reason") in ("function_call", "tool_calls") and not calls:
            raise _UnsupportedRequest("OpenAI Chat reported tool use without a tool call")

        audio = message.get("audio")
        if audio is not None and (
            payload_policy.multimodal is not MultimodalPolicy.TEXT_ONLY or not isinstance(audio, dict)
        ):
            raise _UnsupportedRequest("OpenAI Chat audio output is not covered")
        mutation_safe = mutation_safe and audio is None

        thought = message.get("reasoning_content")
        if thought is not None and not isinstance(thought, str):
            raise _UnsupportedRequest("OpenAI Chat reasoning output is invalid")
        checked_reasoning = False
        if thought:
            if payload_policy.reasoning is ReasoningPolicy.REJECT:
                raise _UnsupportedRequest("OpenAI Chat reasoning output is not covered")
            if payload_policy.reasoning is ReasoningPolicy.CHECK_OUTPUT:
                reasoning.append(thought)
                checked_reasoning = True

        content = message.get("content")
        refusal = message.get("refusal")
        if content is not None and not isinstance(content, str):
            raise _UnsupportedRequest("OpenAI Chat multipart output is not covered")
        if refusal is not None and not isinstance(refusal, str):
            raise _UnsupportedRequest("OpenAI Chat refusal output is invalid")
        if isinstance(content, str) and isinstance(refusal, str):
            raise _UnsupportedRequest("OpenAI Chat content and refusal cannot be ordered losslessly")

        fragments = ()
        if isinstance(content, str):
            fragments = (text_fragment(content, ("choices", choice_index, "message", "content")),)
        elif isinstance(refusal, str):
            fragments = (text_fragment(refusal, ("choices", choice_index, "message", "refusal")),)
        if not fragments and not calls and not checked_reasoning:
            raise _UnsupportedRequest("OpenAI Chat candidate has no covered output")
        candidates.append(ResponseCandidate(fragments, calls, mutation_safe=mutation_safe))

    return ResponseInspection("openai_chat", None, tuple(candidates), tuple(reasoning))
