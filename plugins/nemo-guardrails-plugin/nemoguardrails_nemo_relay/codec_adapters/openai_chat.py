# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Chat Completions wire-shape rules."""

from __future__ import annotations

from typing import Any

from ..payload_policy import MultimodalPolicy, PayloadPolicy, ReasoningPolicy
from .common import (
    RawResponseText,
    _reject_nonempty_metadata,
    _reject_structural_extras,
    _reject_unmodeled_fields,
    _reject_unmodeled_structural_data,
    _UnsupportedRequest,
)
from .openai_common import validate_function_call

ONLY_KEYS = frozenset(
    {
        "audio",
        "chat_template_kwargs",
        "frequency_penalty",
        "function_call",
        "functions",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "modalities",
        "moderation",
        "n",
        "parallel_tool_calls",
        "prediction",
        "presence_penalty",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "reasoning_effort",
        "response_format",
        "safety_identifier",
        "seed",
        "stop",
        "store",
        "stream_options",
        "top_logprobs",
        "user",
        "verbosity",
        "web_search_options",
    }
)
EXTRA_CONTROL_KEYS = frozenset({"chat_template_kwargs"})
STRUCTURAL_CONTROL_KEYS = EXTRA_CONTROL_KEYS

RESPONSE_KEYS = frozenset(
    {
        "choices",
        "created",
        "id",
        "metadata",
        "model",
        "moderation",
        "object",
        "service_tier",
        "system_fingerprint",
        "usage",
    }
)

CHOICE_KEYS = frozenset({"finish_reason", "index", "logprobs", "message"})


def has_message_marker(messages: list[Any]) -> bool:
    return any(
        isinstance(message, dict)
        and (message.get("role") in {"developer", "function", "tool"} or "name" in message or "tool_calls" in message)
        for message in messages
    )


def validate_raw_coverage(content: dict[str, Any]) -> None:
    if content.get("prediction") is not None:
        # ``prediction.content`` is literal per-request provider content,
        # unlike a tool or output-schema declaration. Input rails do not
        # inspect it, so never silently report complete text coverage.
        raise _UnsupportedRequest("OpenAI prediction content is not covered by input rails")

    if "chat_template_kwargs" not in content:
        return
    # NVIDIA NIM exposes this narrow switch for reasoning-capable chat
    # templates. Arbitrary template variables can become hidden prompt
    # content, so admit only the reviewed boolean control.
    template_kwargs = content["chat_template_kwargs"]
    if (
        not isinstance(template_kwargs, dict)
        or set(template_kwargs) != {"enable_thinking"}
        or not isinstance(template_kwargs.get("enable_thinking"), bool)
    ):
        raise _UnsupportedRequest("OpenAI Chat template controls are not supported")


def validate_request_tool_shapes(content: dict[str, Any]) -> None:
    for message in content.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if message.get("function_call") is not None:
            raise _UnsupportedRequest("legacy OpenAI function calls are not structurally covered")
        calls = message.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list):
                raise _UnsupportedRequest("OpenAI Chat tool_calls must be an array")
            if calls and role != "assistant":
                raise _UnsupportedRequest("OpenAI Chat tool calls must be carried by an assistant message")
            for call in calls:
                validate_function_call(call, responses=False)
        call_id = message.get("tool_call_id")
        if call_id is not None and (role != "tool" or not isinstance(call_id, str) or not call_id):
            raise _UnsupportedRequest("OpenAI Chat tool result identity is invalid")
        if role == "tool" and (not isinstance(call_id, str) or not call_id):
            raise _UnsupportedRequest("OpenAI Chat tool message is missing a tool_call_id")
        _reject_structural_extras(
            message,
            {"content", "function_call", "name", "role", "tool_call_id", "tool_calls"},
        )
        message_content = message.get("content")
        if isinstance(message_content, list):
            for block in message_content:
                if isinstance(block, dict):
                    _reject_unmodeled_structural_data(block)


def response_candidate_payloads(response: dict[str, Any], max_segments: int) -> tuple[dict[str, Any], ...]:
    choices = response.get("choices")
    if isinstance(choices, list) and len(choices) > 1:
        if len(choices) > max_segments:
            raise _UnsupportedRequest("OpenAI Chat response has too many choices")
        return tuple({**response, "choices": [choice]} for choice in choices)
    return (response,)


def expected_native_response_text(response: dict[str, Any], projected_text: str | None) -> str | None:
    """Mirror the text exposed by Relay's OpenAI Chat response codec."""

    choices = response.get("choices")
    if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("refusal"), str):
            return None
    return projected_text


def raw_response_call_count(response: dict[str, Any]) -> int:
    covered_keys: set[tuple[int, str]] = {(id(response), "usage")} if "usage" in response else set()
    choices = response.get("choices")
    if not isinstance(choices, list):
        raise _UnsupportedRequest("OpenAI Chat output choices are invalid")
    if not choices:
        _reject_unmodeled_structural_data(response)
        return 0
    count = 0
    for choice in choices:
        if not isinstance(choice, dict):
            raise _UnsupportedRequest("OpenAI Chat choice is invalid")
        message = choice.get("message")
        if message is None:
            continue
        if (
            not isinstance(message, dict)
            or message.get("role") not in {None, "assistant"}
            or message.get("function_call") is not None
        ):
            raise _UnsupportedRequest("OpenAI Chat response message is not structurally covered")
        if "function_call" in message:
            covered_keys.add((id(message), "function_call"))
        calls = message.get("tool_calls")
        if calls is None:
            calls = []
        if not isinstance(calls, list):
            raise _UnsupportedRequest("OpenAI Chat response tool_calls are invalid")
        for call in calls:
            validate_function_call(call, responses=False)
        if choice.get("finish_reason") in {"function_call", "tool_calls"} and not calls:
            raise _UnsupportedRequest("OpenAI Chat reported tool use without a tool call")
        covered_keys.add((id(message), "tool_calls"))
        count += len(calls)
    _reject_unmodeled_structural_data(response, covered_keys=covered_keys)
    return count


def raw_response_texts(
    response: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
) -> RawResponseText:
    candidates_text: list[str] = []
    reasoning_texts: list[str] = []
    _reject_unmodeled_fields(response, RESPONSE_KEYS, "OpenAI Chat response")
    if response.get("metadata") is not None and not isinstance(response["metadata"], dict):
        raise _UnsupportedRequest("OpenAI Chat response metadata must be an object")
    _reject_nonempty_metadata(response, {"moderation"}, "OpenAI Chat response")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _UnsupportedRequest("OpenAI Chat output choices are missing")
    for choice in choices:
        if not isinstance(choice, dict):
            raise _UnsupportedRequest("OpenAI Chat response choice is invalid")
        _reject_unmodeled_fields(
            choice,
            CHOICE_KEYS,
            "OpenAI Chat response choice",
        )
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") not in {None, "assistant"}:
            raise _UnsupportedRequest("OpenAI Chat response message is invalid")
        _reject_nonempty_metadata(choice, {"logprobs"}, "OpenAI Chat response choice")
        _reject_unmodeled_fields(
            message,
            {
                "annotations",
                "audio",
                "content",
                "function_call",
                "reasoning_content",
                "refusal",
                "role",
                "tool_calls",
            },
            "OpenAI Chat response message",
        )
        _reject_nonempty_metadata(message, {"annotations"}, "OpenAI Chat response")
        audio = message.get("audio")
        if audio is not None and (
            payload_policy.multimodal is not MultimodalPolicy.TEXT_ONLY or not isinstance(audio, dict)
        ):
            raise _UnsupportedRequest("OpenAI Chat audio output is not covered")
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise _UnsupportedRequest("OpenAI Chat reasoning output is invalid")
        if reasoning not in (None, ""):
            if payload_policy.reasoning is ReasoningPolicy.REJECT:
                raise _UnsupportedRequest("OpenAI Chat reasoning output is not covered")
            if payload_policy.reasoning is ReasoningPolicy.CHECK_OUTPUT:
                reasoning_texts.append(reasoning)
        content = message.get("content")
        refusal = message.get("refusal")
        if content is not None and not isinstance(content, str):
            # Relay's OpenAI Chat response codec does not decode
            # multipart assistant output, so text-only mode cannot
            # prove which text it would expose.
            raise _UnsupportedRequest("OpenAI Chat multipart output is not covered")
        if refusal is not None and not isinstance(refusal, str):
            raise _UnsupportedRequest("OpenAI Chat refusal output is invalid")
        if isinstance(content, str) and isinstance(refusal, str):
            raise _UnsupportedRequest("OpenAI Chat content and refusal cannot be ordered losslessly")
        if isinstance(content, str):
            candidates_text.append(content)
        elif isinstance(refusal, str):
            candidates_text.append(refusal)
        elif not message.get("tool_calls"):
            raise _UnsupportedRequest("OpenAI Chat candidate has no covered visible text")
    return RawResponseText(
        candidates=tuple(candidates_text),
        reasoning=tuple(reasoning_texts),
        fragment_separator="",
    )
