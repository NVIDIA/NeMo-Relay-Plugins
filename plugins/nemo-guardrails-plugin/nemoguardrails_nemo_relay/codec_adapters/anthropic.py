# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anthropic Messages wire-shape rules."""

from __future__ import annotations

from typing import Any, cast

from ..payload_policy import (
    MultimodalPolicy,
    PayloadPolicy,
    ReasoningPolicy,
    validate_anthropic_reasoning,
)
from ..structural_tools import ToolCall
from .common import (
    RawResponseText,
    _reject_nonempty_metadata,
    _reject_structural_extras,
    _reject_unmodeled_fields,
    _reject_unmodeled_structural_data,
    _structural_key,
    _UnsupportedRequest,
    normalized_annotation,
    tool_call,
)

ONLY_KEYS = frozenset(
    {
        "anthropic-user-profile-id",
        "cache_control",
        "container",
        "inference_geo",
        "output_config",
        "stop_sequences",
        "system",
        "thinking",
        "top_k",
    }
)
TEXT_PART_METADATA = frozenset({"cache_control", "citations"})

RESPONSE_KEYS = frozenset(
    {
        "container",
        "content",
        "context_management",
        "id",
        "model",
        "role",
        "service_tier",
        "stop_details",
        "stop_reason",
        "stop_sequence",
        "type",
        "usage",
    }
)

REQUEST_KEYS = frozenset(
    {
        "system",
        "messages",
        "model",
        "max_tokens",
        "temperature",
        "top_p",
        "stop_sequences",
        "tools",
        "tool_choice",
        "metadata",
        "service_tier",
        "stream",
        *ONLY_KEYS,
    }
)


def _content(value: object) -> object:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise _UnsupportedRequest("Anthropic content must be text or an array")
    parts: list[dict[str, Any]] = []
    for value in value:
        if not isinstance(value, dict):
            raise _UnsupportedRequest("Anthropic content block must be an object")
        kind = value.get("type")
        if kind in {"text", "tool_result", "tool_use"}:
            parts.append(dict(value))
        elif kind == "image":
            parts.append({"type": "image", "image": {k: v for k, v in value.items() if k != "type"}})
        elif kind == "document":
            parts.append({"type": "file", "file": {k: v for k, v in value.items() if k != "type"}})
        else:
            parts.append(
                {"type": "provider_native", "provider": "anthropic_messages", "kind": kind or "unknown", "value": value}
            )
    return parts


def request_annotation(content: dict[str, Any]) -> dict[str, Any]:
    raw_messages = content.get("messages")
    if not isinstance(raw_messages, list):
        raise _UnsupportedRequest("Anthropic messages must be an array")
    messages: list[dict[str, Any]] = []
    for value in raw_messages:
        if not isinstance(value, dict) or not isinstance(value.get("role"), str):
            raise _UnsupportedRequest("Anthropic message is invalid")
        role = value["role"]
        if set(value) - {"content", "role"} or role not in {"assistant", "system", "user"}:
            messages.append({"role": "provider_native", "provider": "anthropic_messages", "kind": role, "value": value})
        else:
            messages.append({"role": role, "content": _content(value.get("content"))})
    instructions = content.get("system")
    if instructions is not None:
        instructions = _content(instructions)
    raw_tools = content.get("tools")
    tools = None
    if raw_tools is not None:
        if not isinstance(raw_tools, list):
            raise _UnsupportedRequest("Anthropic tools must be an array")
        tools = []
        for value in raw_tools:
            if not isinstance(value, dict):
                raise _UnsupportedRequest("Anthropic tool must be an object")
            if value.get("type") in (None, "custom"):
                if value.get("type") == "custom":
                    tools.append(
                        {"type": "provider_native", "provider": "anthropic_messages", "kind": "custom", "value": value}
                    )
                else:
                    function = {"name": value.get("name")}
                    if "input_schema" in value:
                        function["parameters"] = value["input_schema"]
                    if "description" in value:
                        function["description"] = value["description"]
                    if "strict" in value:
                        function["strict"] = value["strict"]
                    tools.append(
                        {
                            "type": "function",
                            "function": function,
                            **{
                                key: item
                                for key, item in value.items()
                                if key not in {"description", "input_schema", "name", "strict", "type"}
                            },
                        }
                    )
            else:
                tools.append(
                    {
                        "type": "provider_native",
                        "provider": "anthropic_messages",
                        "kind": value.get("type", "unknown"),
                        "value": value,
                    }
                )
    return normalized_annotation(
        content,
        messages=messages,
        modeled_keys=REQUEST_KEYS,
        api_specific={"api": "anthropic_messages", "container": content.get("container")},
        instructions=instructions,
        tools=tools,
    )


def response_calls(response: dict[str, Any]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for value in response.get("content") or []:
        if isinstance(value, dict) and value.get("type") == "tool_use":
            calls.append(tool_call(value.get("id"), value.get("name"), value.get("input", {})))
    return tuple(calls)


def response_candidate_payloads(response: dict[str, Any], _max_segments: int) -> tuple[dict[str, Any], ...]:
    """Anthropic exposes one response candidate per Messages envelope."""

    return (response,)


def has_cache_control(messages: list[Any]) -> bool:
    return any(
        isinstance(message, dict)
        and isinstance(message.get("content"), list)
        and any(isinstance(part, dict) and "cache_control" in part for part in message["content"])
        for message in messages
    )


def has_tool_blocks(messages: list[Any]) -> bool:
    """Whether history carries an Anthropic-only tool block discriminator."""

    return any(
        isinstance(message, dict)
        and isinstance(message.get("content"), list)
        and any(
            isinstance(part, dict) and part.get("type") in {"tool_use", "tool_result"} for part in message["content"]
        )
        for message in messages
    )


def validate_direct_caller(value: object, context: str) -> None:
    """Admit only Anthropic's ordinary client-tool caller identity."""

    if value is None:
        return
    if value != {"type": "direct"}:
        raise _UnsupportedRequest(f"{context} is not an ordinary direct tool call")


def validate_request_tool_shapes(content: dict[str, Any]) -> None:
    for message in content.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        _reject_structural_extras(message, {"content", "role"})
        message_content = message.get("content")
        if not isinstance(message_content, list):
            continue
        for block in message_content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_use":
                if role != "assistant":
                    raise _UnsupportedRequest("Anthropic tool-use blocks must be carried by assistant messages")
                if (
                    not isinstance(block.get("id"), str)
                    or not block["id"]
                    or not isinstance(block.get("name"), str)
                    or not block["name"]
                    or not isinstance(block.get("input"), dict)
                ):
                    raise _UnsupportedRequest("Anthropic tool-use block is incomplete")
                _reject_unmodeled_fields(
                    block,
                    {"cache_control", "caller", "id", "input", "name", "toolset_name", "type"},
                    "Anthropic tool-use block",
                )
                validate_direct_caller(block.get("caller"), "Anthropic tool-use caller")
                if block.get("toolset_name") is not None:
                    raise _UnsupportedRequest("Anthropic toolset-member calls are not covered")
            elif block_type == "tool_result":
                if role != "user":
                    raise _UnsupportedRequest("Anthropic tool-result blocks must be carried by user messages")
                if not isinstance(block.get("tool_use_id"), str) or not block["tool_use_id"]:
                    raise _UnsupportedRequest("Anthropic tool-result identity is incomplete")
                _reject_unmodeled_fields(
                    block,
                    {"cache_control", "content", "is_error", "tool_use_id", "type"},
                    "Anthropic tool-result block",
                )
            elif isinstance(block_type, str) and _structural_key(block_type):
                raise _UnsupportedRequest("Anthropic hosted tool traffic is not covered")
            else:
                _reject_unmodeled_structural_data(block)


def codec_content(content: dict[str, Any]) -> dict[str, Any]:
    """Represent Anthropic's valid omitted empty result for Relay 0.9's codec."""

    messages = content.get("messages")
    if not isinstance(messages, list):
        return content
    changed = False
    normalized_messages: list[object] = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            normalized_messages.append(message)
            continue
        normalized_blocks: list[object] = []
        message_changed = False
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result" and "content" not in block:
                normalized_blocks.append({**block, "content": None})
                changed = True
                message_changed = True
            else:
                normalized_blocks.append(block)
        normalized_messages.append({**message, "content": normalized_blocks} if message_changed else message)
    return {**content, "messages": normalized_messages} if changed else content


def raw_response_call_count(response: dict[str, Any]) -> int:
    covered_keys: set[tuple[int, str]] = {(id(response), "usage")} if "usage" in response else set()
    if response.get("role") not in {None, "assistant"} or response.get("type") not in {None, "message"}:
        raise _UnsupportedRequest("Anthropic response role or type is invalid")
    if "context_management" in response:
        if not isinstance(response["context_management"], dict):
            raise _UnsupportedRequest("Anthropic context-management metadata is invalid")
        covered_keys.add((id(response), "context_management"))
    content = response.get("content")
    if not isinstance(content, list):
        raise _UnsupportedRequest("Anthropic response content is missing")
    count = 0
    for block in content:
        if not isinstance(block, dict):
            raise _UnsupportedRequest("Anthropic response block is invalid")
        block_type = block.get("type")
        if block_type == "tool_use":
            if (
                not isinstance(block.get("id"), str)
                or not block["id"]
                or not isinstance(block.get("name"), str)
                or not block["name"]
                or not isinstance(block.get("input"), dict)
            ):
                raise _UnsupportedRequest("Anthropic tool-use block is incomplete")
            _reject_unmodeled_fields(
                block,
                {"caller", "id", "input", "name", "toolset_name", "type"},
                "Anthropic response tool-use block",
            )
            validate_direct_caller(
                block.get("caller"),
                "Anthropic response tool-use caller",
            )
            if block.get("toolset_name") is not None:
                raise _UnsupportedRequest("Anthropic response toolset-member calls are not covered")
            covered_keys.update(
                {
                    (id(block), "input"),
                    (id(block), "type"),
                }
            )
            covered_keys.update((id(block), field) for field in {"caller", "toolset_name"}.intersection(block))
            count += 1
        elif block_type not in {"text", "thinking", "redacted_thinking"}:
            raise _UnsupportedRequest("Anthropic response block is not structurally covered")
    if response.get("stop_reason") == "tool_use" and count == 0:
        raise _UnsupportedRequest("Anthropic reported tool use without a tool call")
    _reject_unmodeled_structural_data(response, covered_keys=covered_keys)
    return count


def raw_response_texts(
    response: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
) -> RawResponseText:
    texts: list[str] = []
    reasoning_texts: list[str] = []
    _reject_unmodeled_fields(response, RESPONSE_KEYS, "Anthropic response")
    if response.get("stop_details") is not None:
        raise _UnsupportedRequest("Anthropic stop-details output is not covered")
    blocks = response.get("content")
    if not isinstance(blocks, list):
        raise _UnsupportedRequest("Anthropic response content is missing")
    for block in blocks:
        if not isinstance(block, dict):
            raise _UnsupportedRequest("Anthropic response block is invalid")
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            _reject_unmodeled_fields(
                block,
                {"cache_control", "citations", "text", "type"},
                "Anthropic response text block",
            )
            _reject_nonempty_metadata(block, {"citations"}, "Anthropic response text block")
            texts.append(cast(str, block["text"]))
        elif block_type == "tool_use":
            continue
        elif block_type in {"thinking", "redacted_thinking"}:
            if not validate_anthropic_reasoning(block):
                raise _UnsupportedRequest("Anthropic thinking output is not covered")
            if payload_policy.reasoning is ReasoningPolicy.REJECT:
                raise _UnsupportedRequest("Anthropic thinking output is not covered")
            if payload_policy.reasoning is ReasoningPolicy.CHECK_OUTPUT:
                if block_type == "redacted_thinking":
                    raise _UnsupportedRequest("Anthropic redacted thinking cannot be checked")
                reasoning_texts.append(cast(str, block["thinking"]))
        elif block_type in {"image", "document"}:
            if (
                payload_policy.multimodal is not MultimodalPolicy.TEXT_ONLY
                or set(block) != {"source", "type"}
                or not isinstance(block.get("source"), dict)
            ):
                raise _UnsupportedRequest("Anthropic multimodal output is not covered")
        else:
            raise _UnsupportedRequest("Anthropic response block is not covered")
    return RawResponseText(
        fragments=tuple(texts),
        reasoning=tuple(reasoning_texts),
    )
