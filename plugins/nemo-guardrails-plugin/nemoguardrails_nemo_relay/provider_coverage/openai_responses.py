# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Responses response inspection."""

from __future__ import annotations

from typing import Any

from ..payload_policy import PayloadPolicy, ReasoningPolicy
from .common import (
    ResponseCandidate,
    ResponseInspection,
    _reject_structural_extensions,
    _reject_unmodeled_fields,
    _UnsupportedRequest,
    text_fragment,
    tool_call,
)

MESSAGE_PHASES = frozenset({"commentary", "final_answer"})
TEXT_PART_METADATA = frozenset({"annotations", "logprobs"})
_ROOT_FIELDS = frozenset(
    {
        "background",
        "billing",
        "completed_at",
        "conversation",
        "created_at",
        "error",
        "id",
        "incomplete_details",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "model",
        "object",
        "output",
        "output_text",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "status",
        "store",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
        "usage",
        "user",
    }
)


def _reasoning_text(item: dict[str, Any], policy: ReasoningPolicy) -> tuple[str, ...]:
    _reject_unmodeled_fields(
        item,
        {"content", "encrypted_content", "id", "status", "summary", "type"},
        "OpenAI Responses reasoning item",
    )
    encrypted = item.get("encrypted_content")
    if encrypted is not None and not isinstance(encrypted, str):
        raise _UnsupportedRequest("OpenAI Responses encrypted reasoning is invalid")
    texts: list[str] = []
    for field, block_type in (("summary", "summary_text"), ("content", "reasoning_text")):
        blocks = item.get(field, [])
        if not isinstance(blocks, list):
            raise _UnsupportedRequest("OpenAI Responses reasoning output is invalid")
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != block_type or not isinstance(block.get("text"), str):
                raise _UnsupportedRequest("OpenAI Responses reasoning output is invalid")
            _reject_unmodeled_fields(
                block,
                {"text", "type"},
                "OpenAI Responses reasoning block",
            )
            texts.append(block["text"])
    if policy is ReasoningPolicy.REJECT:
        raise _UnsupportedRequest("OpenAI Responses reasoning output is not covered")
    if encrypted or texts:
        if policy is ReasoningPolicy.CHECK_OUTPUT:
            if encrypted:
                raise _UnsupportedRequest("OpenAI Responses encrypted reasoning cannot be checked")
            return tuple(texts)
    return ()


def inspect_response(
    response: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
    codec_variant: str | None = None,
) -> ResponseInspection:
    del codec_variant
    root_extensions = _reject_structural_extensions(response, _ROOT_FIELDS, "OpenAI Responses response")
    output = response.get("output")
    aggregate = response.get("output_text")
    if aggregate is not None and not isinstance(aggregate, str):
        raise _UnsupportedRequest("OpenAI Responses output_text is invalid")
    mutation_safe = not root_extensions
    mutation_safe = mutation_safe and response.get("error") in (None, {}, [])
    mutation_safe = mutation_safe and response.get("instructions") in (None, "")
    mutation_safe = mutation_safe and response.get("prompt") in (None, {}, [])
    if output is None:
        if isinstance(aggregate, str):
            candidate = ResponseCandidate(
                (text_fragment(aggregate, ("output_text",)),),
                mutation_safe=mutation_safe,
            )
            return ResponseInspection("openai_responses", None, (candidate,))
        raise _UnsupportedRequest("OpenAI Responses output is missing")
    if not isinstance(output, list):
        raise _UnsupportedRequest("OpenAI Responses output is invalid")

    fragments = []
    aggregate_parts: list[str] = []
    calls = []
    reasoning: list[str] = []
    for item_index, item in enumerate(output):
        if not isinstance(item, dict):
            raise _UnsupportedRequest("OpenAI Responses output item is invalid")
        item_type = item.get("type")
        if item_type == "message":
            _reject_unmodeled_fields(
                item,
                {"content", "id", "phase", "role", "status", "type"},
                "OpenAI Responses output message",
            )
            if item.get("role") not in (None, "assistant") or item.get("phase") not in (None, *MESSAGE_PHASES):
                raise _UnsupportedRequest("OpenAI Responses output message is invalid")
            blocks = item.get("content", [])
            if not isinstance(blocks, list):
                raise _UnsupportedRequest("OpenAI Responses message content is invalid")
            for block_index, block in enumerate(blocks):
                if not isinstance(block, dict):
                    raise _UnsupportedRequest("OpenAI Responses content block is invalid")
                _reject_unmodeled_fields(
                    block,
                    {"annotations", "logprobs", "refusal", "text", "type"},
                    "OpenAI Responses content block",
                )
                block_type = block.get("type")
                if block_type == "output_text":
                    mutation_safe = mutation_safe and block.get("annotations") in (None, [], {})
                    mutation_safe = mutation_safe and block.get("logprobs") in (None, [], {})
                    fragment = text_fragment(
                        block.get("text"),
                        ("output", item_index, "content", block_index, "text"),
                    )
                    fragments.append(fragment)
                    aggregate_parts.append(fragment.text)
                elif block_type == "refusal":
                    mutation_safe = mutation_safe and block.get("annotations") in (None, [], {})
                    mutation_safe = mutation_safe and block.get("logprobs") in (None, [], {})
                    fragments.append(
                        text_fragment(
                            block.get("refusal"),
                            ("output", item_index, "content", block_index, "refusal"),
                        )
                    )
                else:
                    raise _UnsupportedRequest("OpenAI Responses content block is not covered")
        elif item_type == "output_text":
            _reject_unmodeled_fields(
                item,
                {"annotations", "id", "logprobs", "status", "text", "type"},
                "OpenAI Responses output-text item",
            )
            mutation_safe = mutation_safe and item.get("annotations") in (None, [], {})
            mutation_safe = mutation_safe and item.get("logprobs") in (None, [], {})
            fragment = text_fragment(item.get("text"), ("output", item_index, "text"))
            fragments.append(fragment)
            aggregate_parts.append(fragment.text)
        elif item_type == "function_call":
            if set(item) - {"arguments", "call_id", "id", "name", "status", "type"} or "arguments" not in item:
                raise _UnsupportedRequest("OpenAI Responses function call is incomplete")
            calls.append(tool_call(item.get("call_id"), item.get("name"), item["arguments"]))
        elif item_type == "reasoning":
            reasoning.extend(_reasoning_text(item, payload_policy.reasoning))
        else:
            # Every output entry is a tagged content/tool union. Unknown tags
            # could carry unchecked output or invoke a hosted tool.
            raise _UnsupportedRequest("OpenAI Responses output item is not covered")

    expected_aggregate = "\n".join(aggregate_parts)
    if isinstance(aggregate, str) and aggregate not in {"", expected_aggregate}:
        raise _UnsupportedRequest("OpenAI Responses aggregate text disagrees with detailed output")
    if len(fragments) == 1 and aggregate == fragments[0].text and aggregate_parts:
        fragment = fragments[0]
        fragments[0] = type(fragment)(fragment.text, fragment.path, (("output_text",),))
    candidate = ResponseCandidate(tuple(fragments), tuple(calls), mutation_safe=mutation_safe)
    return ResponseInspection("openai_responses", None, (candidate,), tuple(reasoning))
