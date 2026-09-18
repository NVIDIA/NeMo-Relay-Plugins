# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Responses wire-shape rules."""

from __future__ import annotations

from typing import Any, cast

from ..payload_policy import PayloadPolicy, ReasoningPolicy, validate_openai_responses_reasoning
from .common import (
    RawResponseText,
    _reject_nonempty_metadata,
    _reject_structural_extras,
    _reject_unmodeled_fields,
    _reject_unmodeled_structural_data,
    _UnsupportedRequest,
)
from .openai_common import validate_function_call

MESSAGE_PHASES = frozenset({"commentary", "final_answer"})
TEXT_PART_METADATA = frozenset({"annotations", "logprobs"})
EXTRA_CONTROL_KEYS = frozenset({"client_metadata"})

RESPONSE_KEYS = frozenset(
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
        "moderation",
        "object",
        "output",
        "output_text",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt",
        "prompt_cache_diagnostics",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
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


def response_candidate_payloads(response: dict[str, Any], _max_segments: int) -> tuple[dict[str, Any], ...]:
    """Responses exposes one aggregate response envelope to Relay's codec."""

    return (response,)


def expected_native_response_text(response: dict[str, Any], _projected_text: str | None) -> str | None:
    """Mirror the aggregate text exposed by Relay's Responses codec."""

    output_texts: list[str] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "output_text" and isinstance(item.get("text"), str):
                output_texts.append(cast(str, item["text"]))
            elif item.get("type") == "message" and isinstance(item.get("content"), list):
                output_texts.extend(
                    cast(str, block["text"])
                    for block in item["content"]
                    if isinstance(block, dict)
                    and block.get("type") == "output_text"
                    and isinstance(block.get("text"), str)
                )
    if output_texts:
        return "\n".join(output_texts)
    aggregate = response.get("output_text")
    return aggregate if isinstance(aggregate, str) and aggregate else None


def validate_raw_coverage(content: dict[str, Any]) -> None:
    if "client_metadata" not in content:
        return
    metadata = content["client_metadata"]
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"x-codex-installation-id"}
        or not isinstance(metadata["x-codex-installation-id"], str)
        or not metadata["x-codex-installation-id"]
    ):
        raise _UnsupportedRequest("OpenAI Responses client metadata is not recognized")


def validate_request_tool_shapes(content: dict[str, Any]) -> None:
    items = content.get("input")
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            validate_function_call(item, responses=True)
        elif item_type == "function_call_output":
            if not isinstance(item.get("call_id"), str) or not item["call_id"]:
                raise _UnsupportedRequest("OpenAI Responses tool result identity is incomplete")
            if "output" not in item:
                raise _UnsupportedRequest("OpenAI Responses tool result output is missing")
            _reject_unmodeled_fields(
                item,
                {"call_id", "id", "output", "status", "type"},
                "OpenAI Responses function-call output",
            )
        elif item_type == "item_reference":
            raise _UnsupportedRequest("OpenAI Responses referenced history is not structurally covered")
        elif item_type in {None, "message"} and isinstance(item.get("role"), str):
            phase = item.get("phase")
            if phase is not None and (item.get("role") != "assistant" or phase not in MESSAGE_PHASES):
                raise _UnsupportedRequest("OpenAI Responses message phase is invalid")
            message_content = item.get("content")
            if not isinstance(message_content, (str, list, type(None))):
                raise _UnsupportedRequest("OpenAI Responses message content is invalid")
            if isinstance(message_content, list):
                for block in message_content:
                    if not isinstance(block, dict) or block.get("type") not in {
                        "input_file",
                        "input_image",
                        "input_text",
                        "output_text",
                        "refusal",
                    }:
                        raise _UnsupportedRequest("OpenAI Responses message content is not structurally covered")
                    _reject_structural_extras(
                        block,
                        {
                            "annotations",
                            "detail",
                            "file_data",
                            "file_id",
                            "file_url",
                            "filename",
                            "image_url",
                            "logprobs",
                            "refusal",
                            "text",
                            "type",
                        },
                    )
            _reject_structural_extras(
                item,
                {"content", "id", "phase", "role", "status", "type"},
            )
        elif isinstance(item_type, str) and ("call" in item_type or "tool" in item_type or "mcp" in item_type):
            raise _UnsupportedRequest("OpenAI Responses hosted tool traffic is not covered")
        else:
            _reject_unmodeled_structural_data(item)


def raw_response_call_count(response: dict[str, Any]) -> int:
    covered_keys: set[tuple[int, str]] = {
        (id(response), key)
        for key in {"max_tool_calls", "parallel_tool_calls", "tool_choice", "tools", "usage"}
        if key in response
    }
    output = response.get("output")
    if output is None and isinstance(response.get("output_text"), str):
        _reject_unmodeled_structural_data(response, covered_keys=covered_keys)
        return 0
    if not isinstance(output, list):
        raise _UnsupportedRequest("OpenAI Responses output is missing")
    count = 0
    for item in output:
        if not isinstance(item, dict):
            raise _UnsupportedRequest("OpenAI Responses output item is invalid")
        item_type = item.get("type")
        if item_type == "function_call":
            validate_function_call(item, responses=True)
            covered_keys.update(
                {
                    (id(item), "arguments"),
                    (id(item), "call_id"),
                    (id(item), "type"),
                }
            )
            count += 1
        elif item_type == "message":
            if item.get("role") not in {None, "assistant"}:
                raise _UnsupportedRequest("OpenAI Responses output role is invalid")
            content = item.get("content")
            if content is not None:
                if not isinstance(content, list):
                    raise _UnsupportedRequest("OpenAI Responses message content is invalid")
                for block in content:
                    if not isinstance(block, dict) or block.get("type") not in {"output_text", "refusal"}:
                        raise _UnsupportedRequest("OpenAI Responses message content is not structurally covered")
        elif item_type == "output_text":
            if not isinstance(item.get("text"), str):
                raise _UnsupportedRequest("OpenAI Responses output text is invalid")
        elif item_type != "reasoning":
            # Hosted tools, computer use, custom tools, and future
            # output item types are not silently treated as ordinary
            # text. The exact Relay codec does not normalize them into
            # provider-neutral function calls.
            raise _UnsupportedRequest("OpenAI Responses output item is not structurally covered")
    _reject_unmodeled_structural_data(response, covered_keys=covered_keys)
    return count


def raw_response_texts(
    response: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
) -> RawResponseText:
    texts: list[str] = []
    reasoning_texts: list[str] = []
    _reject_unmodeled_fields(
        response,
        RESPONSE_KEYS,
        "OpenAI Responses response",
    )
    if response.get("metadata") is not None and not isinstance(response["metadata"], dict):
        raise _UnsupportedRequest("OpenAI Responses metadata must be an object")
    if response.get("error") is not None:
        raise _UnsupportedRequest("OpenAI Responses error output is not covered")
    _reject_nonempty_metadata(response, {"moderation"}, "OpenAI Responses response")
    native_texts: list[str] = []
    output = response.get("output")
    if output is not None and not isinstance(output, list):
        raise _UnsupportedRequest("OpenAI Responses output is invalid")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                raise _UnsupportedRequest("OpenAI Responses output item is invalid")
            item_type = item.get("type")
            if item_type == "message":
                _reject_unmodeled_fields(
                    item,
                    {"content", "id", "phase", "role", "status", "type"},
                    "OpenAI Responses output message",
                )
                if item.get("role") not in {None, "assistant"}:
                    raise _UnsupportedRequest("OpenAI Responses output role is invalid")
                if item.get("phase") not in (None, *MESSAGE_PHASES):
                    raise _UnsupportedRequest("OpenAI Responses output phase is invalid")
                blocks = item.get("content")
                if blocks is None:
                    continue
                if not isinstance(blocks, list):
                    raise _UnsupportedRequest("OpenAI Responses message content is invalid")
                for block in blocks:
                    if not isinstance(block, dict):
                        raise _UnsupportedRequest("OpenAI Responses content block is invalid")
                    if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                        _reject_unmodeled_fields(
                            block,
                            {"annotations", "logprobs", "text", "type"},
                            "OpenAI Responses output-text block",
                        )
                        _reject_nonempty_metadata(
                            block,
                            {"annotations", "logprobs"},
                            "OpenAI Responses output-text block",
                        )
                        texts.append(cast(str, block["text"]))
                        native_texts.append(cast(str, block["text"]))
                    elif block.get("type") == "refusal" and isinstance(block.get("refusal"), str):
                        _reject_unmodeled_fields(
                            block,
                            {"refusal", "type"},
                            "OpenAI Responses refusal block",
                        )
                        texts.append(cast(str, block["refusal"]))
                    else:
                        raise _UnsupportedRequest("OpenAI Responses content block is not covered")
            elif item_type == "output_text" and isinstance(item.get("text"), str):
                _reject_unmodeled_fields(
                    item,
                    {"annotations", "id", "logprobs", "status", "text", "type"},
                    "OpenAI Responses output-text item",
                )
                _reject_nonempty_metadata(
                    item,
                    {"annotations", "logprobs"},
                    "OpenAI Responses output-text item",
                )
                texts.append(cast(str, item["text"]))
                native_texts.append(cast(str, item["text"]))
            elif item_type == "function_call":
                continue
            elif item_type == "reasoning":
                if not validate_openai_responses_reasoning(item):
                    raise _UnsupportedRequest("OpenAI Responses reasoning output is not covered")
                if payload_policy.reasoning is ReasoningPolicy.REJECT:
                    raise _UnsupportedRequest("OpenAI Responses reasoning output is not covered")
                if payload_policy.reasoning is ReasoningPolicy.CHECK_OUTPUT:
                    if item.get("encrypted_content") not in (None, ""):
                        raise _UnsupportedRequest("OpenAI Responses encrypted reasoning cannot be checked")
                    for field in ("summary", "content"):
                        reasoning_texts.extend(
                            cast(str, block["text"]) for block in cast(list[dict[str, Any]], item.get(field, []))
                        )
            else:
                # The structural pass rejects hosted/computer/custom
                # tool items; this also rejects unknown visible output.
                raise _UnsupportedRequest("OpenAI Responses output item is not covered")
    aggregate = response.get("output_text")
    if aggregate is not None and not isinstance(aggregate, str):
        raise _UnsupportedRequest("OpenAI Responses output_text is invalid")
    if isinstance(aggregate, str):
        if native_texts and aggregate != "\n".join(native_texts):
            raise _UnsupportedRequest("OpenAI Responses aggregate text disagrees with detailed output")
        if texts and not native_texts and aggregate:
            raise _UnsupportedRequest("OpenAI Responses aggregate text is not represented by detailed output")
        if not texts and aggregate:
            texts.append(aggregate)
            native_texts.append(aggregate)
    return RawResponseText(
        fragments=tuple(texts),
        reasoning=tuple(reasoning_texts),
    )
