# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemini GenerateContent response inspection."""

from __future__ import annotations

from typing import Any

from ..payload_policy import MultimodalPolicy, PayloadPolicy, ReasoningPolicy
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

TEXT_PART_METADATA = frozenset({"partMetadata", "thought", "thoughtSignature"})

_PART_VALUES = frozenset(
    {
        "codeExecutionResult",
        "executableCode",
        "fileData",
        "functionCall",
        "functionResponse",
        "inlineData",
        "text",
    }
)
_INVALID_TOOL_FINISH_REASONS = frozenset(
    {
        "MALFORMED_FUNCTION_CALL",
        "MALFORMED_RESPONSE",
        "MISSING_THOUGHT_SIGNATURE",
        "TOO_MANY_TOOL_CALLS",
        "UNEXPECTED_TOOL_CALL",
    }
)
_ROOT_FIELDS = frozenset(
    {
        "candidates",
        "createTime",
        "modelStatus",
        "modelVersion",
        "promptFeedback",
        "responseId",
        "usageMetadata",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "avgLogprobs",
        "citationMetadata",
        "content",
        "finishMessage",
        "finishReason",
        "groundingAttributions",
        "groundingMetadata",
        "index",
        "logprobsResult",
        "safetyRatings",
        "tokenCount",
        "urlContextMetadata",
    }
)


def inspect_response(
    response: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
    codec_variant: str | None = None,
) -> ResponseInspection:
    del codec_variant
    root_extensions = _reject_structural_extensions(response, _ROOT_FIELDS, "Gemini response")
    raw_candidates = response.get("candidates")
    if not isinstance(raw_candidates, list):
        raise _UnsupportedRequest("Gemini output candidates are missing")
    if len(raw_candidates) > MAX_RESPONSE_SEGMENTS:
        raise _UnsupportedRequest("Gemini output has too many candidates")

    root_mutation_safe = not root_extensions and all(
        response.get(field) in (None, "", [], {}) for field in {"modelStatus", "promptFeedback"}
    )

    candidates: list[ResponseCandidate] = []
    reasoning: list[str] = []
    for candidate_index, raw_candidate in enumerate(raw_candidates):
        reasoning_before = len(reasoning)
        if not isinstance(raw_candidate, dict):
            raise _UnsupportedRequest("Gemini response candidate is invalid")
        _reject_unmodeled_fields(
            raw_candidate,
            _CANDIDATE_FIELDS,
            "Gemini response candidate",
        )
        finish_reason = str(raw_candidate.get("finishReason", "")).upper()
        if finish_reason in _INVALID_TOOL_FINISH_REASONS:
            raise _UnsupportedRequest("Gemini reported invalid function-tool output")
        content = raw_candidate.get("content")
        if not isinstance(content, dict) or content.get("role") not in (None, "model"):
            raise _UnsupportedRequest("Gemini response content is invalid")
        _reject_unmodeled_fields(content, {"parts", "role"}, "Gemini response content")
        parts = content.get("parts")
        if not isinstance(parts, list):
            raise _UnsupportedRequest("Gemini response parts are invalid")

        fragments = []
        calls = []
        mutation_safe = root_mutation_safe and all(
            raw_candidate.get(field) in (None, "", [], {})
            for field in {
                "avgLogprobs",
                "citationMetadata",
                "finishMessage",
                "groundingAttributions",
                "groundingMetadata",
                "logprobsResult",
                "safetyRatings",
                "urlContextMetadata",
            }
        )
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict):
                raise _UnsupportedRequest("Gemini response part is invalid")
            _reject_unmodeled_fields(
                part,
                _PART_VALUES | {"partMetadata", "thought", "thoughtSignature", "videoMetadata"},
                "Gemini response part",
            )
            values = _PART_VALUES.intersection(part)
            if len(values) != 1:
                raise _UnsupportedRequest("Gemini response part has an unsupported data variant")
            value_type = next(iter(values))
            if value_type == "text":
                text = part.get("text")
                if not isinstance(text, str) or not isinstance(part.get("thought", False), bool):
                    raise _UnsupportedRequest("Gemini response text is invalid")
                if part.get("thought") is True:
                    if payload_policy.reasoning is ReasoningPolicy.REJECT:
                        raise _UnsupportedRequest("Gemini thought output is not covered")
                    if payload_policy.reasoning is ReasoningPolicy.CHECK_OUTPUT:
                        reasoning.append(text)
                else:
                    mutation_safe = mutation_safe and part.get("partMetadata") in (None, [], {})
                    mutation_safe = mutation_safe and part.get("thoughtSignature") in (None, "")
                    mutation_safe = mutation_safe and part.get("videoMetadata") in (None, [], {})
                    fragments.append(
                        text_fragment(text, ("candidates", candidate_index, "content", "parts", part_index, "text"))
                    )
            elif value_type == "functionCall":
                call = part.get("functionCall")
                if not isinstance(call, dict) or set(call) - {"args", "id", "name"}:
                    raise _UnsupportedRequest("Gemini function call is invalid")
                name = call.get("name")
                calls.append(tool_call(call.get("id", name), name, call.get("args", {})))
            elif value_type in {"inlineData", "fileData"}:
                if payload_policy.multimodal is not MultimodalPolicy.TEXT_ONLY or not isinstance(
                    part.get(value_type), dict
                ):
                    raise _UnsupportedRequest("Gemini multimodal output is not covered")
            else:
                raise _UnsupportedRequest("Gemini provider tool output is not covered")

        if finish_reason == "TOOL_CODE" and not calls:
            raise _UnsupportedRequest("Gemini reported tool use without a function call")
        if not fragments and not calls and len(reasoning) == reasoning_before:
            raise _UnsupportedRequest("Gemini candidate has no covered output")
        candidates.append(ResponseCandidate(tuple(fragments), tuple(calls), mutation_safe=mutation_safe))

    return ResponseInspection("gemini_generate_content", None, tuple(candidates), tuple(reasoning))
