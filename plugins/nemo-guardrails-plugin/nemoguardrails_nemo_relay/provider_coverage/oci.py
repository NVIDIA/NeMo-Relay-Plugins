# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OCI Generative AI response inspection."""

from __future__ import annotations

from typing import Any, cast

from ..payload_policy import PayloadPolicy, ReasoningPolicy
from ..structural_tools import ToolCall
from .common import (
    MAX_RESPONSE_SEGMENTS,
    JsonPath,
    ResponseCandidate,
    ResponseInspection,
    _reject_structural_extensions,
    _reject_unmodeled_fields,
    _UnsupportedRequest,
    text_fragment,
    tool_call,
)

API_FORMATS = frozenset({"GENERIC", "COHERE", "COHEREV2"})
_ROOT_FIELDS = frozenset({"chatResponse", "modelId", "modelVersion", "opcRequestId"})
_CHAT_FIELDS = {
    "GENERIC": frozenset({"apiFormat", "choices", "serviceTier", "timeCreated", "usage"}),
    "COHERE": frozenset(
        {
            "apiFormat",
            "chatHistory",
            "citations",
            "documents",
            "errorMessage",
            "finishReason",
            "isSearchRequired",
            "prompt",
            "searchQueries",
            "searchResults",
            "text",
            "toolCalls",
            "usage",
        }
    ),
    "COHEREV2": frozenset({"apiFormat", "errorMessage", "finishReason", "logProbabilities", "message", "usage"}),
}
_GENERIC_CHOICE_FIELDS = frozenset(
    {"finishReason", "groundingMetadata", "index", "logprobs", "message", "serviceTier", "usage"}
)


def response_variant(response: dict[str, Any]) -> str:
    chat = response.get("chatResponse", response)
    if not isinstance(chat, dict):
        raise _UnsupportedRequest("OCI chat response is invalid")
    variant = chat.get("apiFormat", "GENERIC")
    if not isinstance(variant, str) or variant.upper() not in API_FORMATS:
        raise _UnsupportedRequest("OCI response apiFormat is not supported")
    return variant.upper()


def parse_tool_calls(raw_calls: object, variant: str) -> tuple[ToolCall, ...]:
    if raw_calls is None:
        return ()
    if not isinstance(raw_calls, list):
        raise _UnsupportedRequest("OCI response toolCalls are invalid")
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            raise _UnsupportedRequest("OCI response tool call is invalid")
        if variant == "COHERE":
            if set(raw) - {"name", "parameters"}:
                raise _UnsupportedRequest("OCI COHERE function call has unsupported fields")
            calls.append(tool_call(f"call_{index}", raw.get("name"), raw.get("parameters", {})))
            continue
        if variant == "COHEREV2":
            function = raw.get("function")
            if (
                set(raw) - {"function", "id", "type"}
                or not isinstance(function, dict)
                or set(function) - {"arguments", "name"}
                or str(raw.get("type", "")).upper() != "FUNCTION"
            ):
                raise _UnsupportedRequest("OCI COHEREV2 function call is invalid")
            calls.append(tool_call(raw.get("id"), function.get("name"), function.get("arguments", {})))
            continue
        if set(raw) - {"arguments", "id", "name", "parameters", "type"}:
            raise _UnsupportedRequest("OCI GENERIC function call has unsupported fields")
        if "arguments" in raw and "parameters" in raw:
            raise _UnsupportedRequest("OCI GENERIC function call has two argument payloads")
        if str(raw.get("type", "FUNCTION")).upper() != "FUNCTION":
            raise _UnsupportedRequest("OCI GENERIC tool call is not a function call")
        calls.append(
            tool_call(
                raw.get("id"),
                raw.get("name"),
                raw.get("arguments", raw.get("parameters", {})),
            )
        )
    return tuple(calls)


def _reasoning(value: object, policy: ReasoningPolicy, output: list[str]) -> None:
    if value is None or value == "":
        return
    if not isinstance(value, str):
        raise _UnsupportedRequest("OCI reasoning output is invalid")
    if policy is ReasoningPolicy.REJECT:
        raise _UnsupportedRequest("OCI reasoning output is not covered")
    if policy is ReasoningPolicy.CHECK_OUTPUT:
        output.append(value)


def _message_candidate(
    message: object,
    *,
    variant: str,
    path: JsonPath,
    payload_policy: PayloadPolicy,
    reasoning: list[str],
    mutation_safe: bool = True,
) -> ResponseCandidate:
    if not isinstance(message, dict) or str(message.get("role", "ASSISTANT")).upper() != "ASSISTANT":
        raise _UnsupportedRequest("OCI response message is invalid")
    if message.get("toolCallId") is not None:
        raise _UnsupportedRequest("OCI assistant response carries a tool-result identity")
    _reject_unmodeled_fields(
        message,
        {
            "annotations",
            "citations",
            "content",
            "reasoningContent",
            "refusal",
            "role",
            "toolCallId",
            "toolCalls",
            "toolPlan",
        },
        "OCI response message",
    )
    mutation_safe = mutation_safe and message.get("annotations") in (None, [], {})
    mutation_safe = mutation_safe and message.get("citations") in (None, [], {})

    reasoning_before = len(reasoning)
    _reasoning(message.get("reasoningContent"), payload_policy.reasoning, reasoning)
    _reasoning(message.get("toolPlan"), payload_policy.reasoning, reasoning)
    calls = parse_tool_calls(message.get("toolCalls"), variant)

    content = message.get("content")
    refusal = message.get("refusal")
    if refusal is not None and not isinstance(refusal, str):
        raise _UnsupportedRequest("OCI refusal output is invalid")
    if isinstance(refusal, str) and content not in (None, "", []):
        raise _UnsupportedRequest("OCI content and refusal cannot be ordered losslessly")
    if isinstance(refusal, str):
        fragments = [text_fragment(refusal, (*path, "refusal"))]
    elif isinstance(content, str):
        fragments = [text_fragment(content, (*path, "content"))]
    elif content is None:
        fragments = []
    elif isinstance(content, list):
        fragments = []
        for index, part in enumerate(content):
            if not isinstance(part, dict) or str(part.get("type", "")).upper() != "TEXT":
                raise _UnsupportedRequest("OCI response content part is not covered")
            _reject_unmodeled_fields(
                part,
                {"text", "type"},
                "OCI response text part",
            )
            fragments.append(text_fragment(part.get("text", ""), (*path, "content", index, "text")))
    else:
        raise _UnsupportedRequest("OCI response content is invalid")

    if not fragments and not calls and len(reasoning) == reasoning_before:
        raise _UnsupportedRequest("OCI response candidate has no covered output")
    return ResponseCandidate(tuple(fragments), calls, separator="", mutation_safe=mutation_safe)


def inspect_response(
    response: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
    codec_variant: str | None = None,
) -> ResponseInspection:
    nested = isinstance(response.get("chatResponse"), dict)
    root_extensions = False
    if nested:
        root_extensions = _reject_structural_extensions(response, _ROOT_FIELDS, "OCI response")
    prefix: JsonPath = ("chatResponse",) if nested else ()
    chat = cast(dict[str, Any], response["chatResponse"] if nested else response)
    variant = response_variant(response)
    if codec_variant is not None and codec_variant != variant:
        raise _UnsupportedRequest("OCI response apiFormat does not match the request")

    reasoning: list[str] = []
    chat_extensions = _reject_structural_extensions(chat, _CHAT_FIELDS[variant], "OCI chat response")
    expected_payload = {"GENERIC": "choices", "COHERE": "text", "COHEREV2": "message"}[variant]
    for field in {"choices", "message", "text"} - {expected_payload}:
        if chat.get(field) not in (None, "", [], {}):
            raise _UnsupportedRequest(f"OCI {variant} response contains an unexpected {field}")
    mutation_safe = not (root_extensions or chat_extensions)
    if variant == "COHERE":
        calls = parse_tool_calls(chat.get("toolCalls"), variant)
        text = chat.get("text")
        fragments = () if text is None else (text_fragment(text, (*prefix, "text")),)
        mutation_safe = mutation_safe and all(
            chat.get(field) in (None, "", [], {})
            for field in {
                "chatHistory",
                "citations",
                "documents",
                "errorMessage",
                "prompt",
                "searchQueries",
                "searchResults",
            }
        )
        candidates = (ResponseCandidate(fragments, calls, separator="", mutation_safe=mutation_safe),)
        if not fragments and not calls:
            raise _UnsupportedRequest("OCI response candidate has no covered output")
        finish_reasons = (chat.get("finishReason"),)
    elif variant == "COHEREV2":
        candidates = (
            _message_candidate(
                chat.get("message"),
                variant=variant,
                path=(*prefix, "message"),
                payload_policy=payload_policy,
                reasoning=reasoning,
                mutation_safe=mutation_safe
                and chat.get("errorMessage") in (None, "")
                and chat.get("logProbabilities") in (None, [], {}),
            ),
        )
        finish_reasons = (chat.get("finishReason"),)
    else:
        choices = chat.get("choices")
        if not isinstance(choices, list):
            raise _UnsupportedRequest("OCI output choices are missing")
        if len(choices) > MAX_RESPONSE_SEGMENTS:
            raise _UnsupportedRequest("OCI output has too many candidates")
        built = []
        finish_reasons = []
        for index, choice in enumerate(choices):
            if not isinstance(choice, dict):
                raise _UnsupportedRequest("OCI response choice is invalid")
            _reject_unmodeled_fields(
                choice,
                _GENERIC_CHOICE_FIELDS,
                "OCI response choice",
            )
            built.append(
                _message_candidate(
                    choice.get("message"),
                    variant=variant,
                    path=(*prefix, "choices", index, "message"),
                    payload_policy=payload_policy,
                    reasoning=reasoning,
                    mutation_safe=mutation_safe
                    and all(choice.get(field) in (None, [], {}) for field in {"groundingMetadata", "logprobs"}),
                )
            )
            finish_reasons.append(choice.get("finishReason"))
        candidates = tuple(built)

    for reason, candidate in zip(finish_reasons, candidates, strict=True):
        if str(reason).upper() in {"TOOL_CALL", "TOOL_CALLS"} and not candidate.calls:
            raise _UnsupportedRequest("OCI reported tool use without a tool call")
    return ResponseInspection("oci_genai", variant, candidates, tuple(reasoning))
