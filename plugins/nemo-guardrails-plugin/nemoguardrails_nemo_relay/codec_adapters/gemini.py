# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemini GenerateContent wire-shape rules."""

from __future__ import annotations

from typing import Any, cast

from ..payload_policy import MultimodalPolicy, PayloadPolicy, ReasoningPolicy
from .common import (
    RawResponseText,
    _reject_nonempty_metadata,
    _reject_unmodeled_fields,
    _reject_unmodeled_structural_data,
    _UnsupportedRequest,
)

RESPONSE_KEYS = frozenset(
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
TEXT_PART_METADATA = frozenset({"partMetadata", "thought", "thoughtSignature"})
EXTRA_CONTROL_KEYS = frozenset({"toolConfig"})
STRUCTURAL_CONTROL_KEYS = EXTRA_CONTROL_KEYS

CANDIDATE_KEYS = frozenset(
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


def _part_lists(content: dict[str, Any]) -> list[Any]:
    part_lists: list[Any] = []
    system_instruction = content.get("systemInstruction")
    if system_instruction is not None and (
        not isinstance(system_instruction, dict) or set(system_instruction) - {"role", "parts"}
    ):
        raise _UnsupportedRequest("Gemini system instruction contains unmodeled fields")
    if isinstance(system_instruction, dict):
        part_lists.append(system_instruction.get("parts"))

    contents = content.get("contents")
    if not isinstance(contents, list):
        return part_lists
    for item in contents:
        if not isinstance(item, dict):
            continue
        if set(item) - {"role", "parts"}:
            raise _UnsupportedRequest("Gemini content contains unmodeled fields")
        part_lists.append(item.get("parts"))
    return part_lists


def validate_raw_coverage(
    content: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
    allow_structural_tools: bool,
    require_text_coverage: bool,
) -> None:
    allowed_data = {"partMetadata", "text"}
    if allow_structural_tools:
        allowed_data.update({"functionCall", "functionResponse"})
    if not require_text_coverage or payload_policy.multimodal is MultimodalPolicy.TEXT_ONLY:
        # Multimodal data is unrelated to structural function-call validation.
        # The native codec still validates each part's shape. Code execution is
        # deliberately excluded because Guardrails has no matching validator.
        allowed_data.update({"inlineData", "fileData"})
    for parts in _part_lists(content):
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            if set(part) - allowed_data - {"thought", "thoughtSignature"}:
                raise _UnsupportedRequest("Gemini part contains unmodeled fields")
            if "thought" in part and not isinstance(part["thought"], bool):
                raise _UnsupportedRequest("Gemini thought marker is invalid")
            if "thoughtSignature" in part and not isinstance(part["thoughtSignature"], str):
                raise _UnsupportedRequest("Gemini thought signature is invalid")
            if "partMetadata" in part and not isinstance(part["partMetadata"], dict):
                raise _UnsupportedRequest("Gemini part metadata must be an object")
            if (
                require_text_coverage
                and part.get("thought") is True
                and "functionCall" not in part
                and payload_policy.reasoning not in {ReasoningPolicy.FINAL_ANSWER_ONLY, ReasoningPolicy.CHECK_OUTPUT}
            ):
                raise _UnsupportedRequest("Gemini hidden thought text is not covered by input rails")
            if "functionCall" in part and isinstance(part["functionCall"], dict):
                _reject_unmodeled_fields(
                    part["functionCall"],
                    {"args", "id", "name"},
                    "Gemini function call",
                )
            if "functionResponse" in part and isinstance(part["functionResponse"], dict):
                _reject_unmodeled_fields(
                    part["functionResponse"],
                    {"id", "name", "parts", "response"},
                    "Gemini function response",
                )


def codec_content(content: dict[str, Any]) -> dict[str, Any]:
    """Restore result IDs that Relay's Gemini annotation would otherwise lose.

    Gemini permits an omitted functionResponse ID. When a unique preceding
    call has the same required name, use its ID in the worker-local decode
    copy. Also reject a response that supplies a matching ID with a different
    name before the codec drops that name from its normalized Tool message.
    """

    contents = content.get("contents")
    if not isinstance(contents, list):
        return content
    open_calls: dict[str, str] | None = None
    rewritten_contents: list[object] = []
    changed = False
    for item in contents:
        if not isinstance(item, dict) or not isinstance(item.get("parts"), list):
            rewritten_contents.append(item)
            open_calls = None
            continue
        calls: dict[str, str] = {}
        responses: list[tuple[int, dict[str, Any]]] = []
        for index, part in enumerate(item["parts"]):
            if not isinstance(part, dict):
                continue
            call = part.get("functionCall")
            if isinstance(call, dict) and isinstance(call.get("name"), str) and call["name"]:
                call_id = call.get("id", call["name"])
                if isinstance(call_id, str) and call_id:
                    calls[call_id] = call["name"]
            response = part.get("functionResponse")
            if isinstance(response, dict):
                responses.append((index, response))

        if calls:
            open_calls = calls
            rewritten_contents.append(item)
            continue
        if responses and open_calls is not None:
            rewritten_parts = list(item["parts"])
            item_changed = False
            for index, response in responses:
                name = response.get("name")
                response_id = response.get("id")
                if isinstance(response_id, str) and response_id in open_calls:
                    if name != open_calls[response_id]:
                        raise _UnsupportedRequest("Gemini function response name does not match its call")
                    continue
                if response_id is not None or not isinstance(name, str) or not name:
                    continue
                matching_ids = [call_id for call_id, call_name in open_calls.items() if call_name == name]
                if len(matching_ids) != 1:
                    continue
                rewritten_response = {**response, "id": matching_ids[0]}
                rewritten_parts[index] = {
                    **rewritten_parts[index],  # type: ignore[misc]
                    "functionResponse": rewritten_response,
                }
                item_changed = True
            if item_changed:
                rewritten_contents.append({**item, "parts": rewritten_parts})
                changed = True
            else:
                rewritten_contents.append(item)
            continue
        rewritten_contents.append(item)
        open_calls = None
    return {**content, "contents": rewritten_contents} if changed else content


def response_candidate_payloads(response: dict[str, Any], max_segments: int) -> tuple[dict[str, Any], ...]:
    candidates = response.get("candidates")
    if isinstance(candidates, list) and len(candidates) > 1:
        if len(candidates) > max_segments:
            raise _UnsupportedRequest("Gemini response has too many candidates")
        return tuple({**response, "candidates": [candidate]} for candidate in candidates)
    return (response,)


def expected_native_response_text(response: dict[str, Any], projected_text: str | None) -> str | None:
    """Mirror the text exposed by Relay's Gemini response codec."""

    candidates = response.get("candidates")
    if isinstance(candidates, list) and len(candidates) == 1 and isinstance(candidates[0], dict):
        content = candidates[0].get("content")
        if isinstance(content, dict) and isinstance(content.get("parts"), list):
            parts = content["parts"]
            metadata_bearing_text = any(
                isinstance(part, dict) and isinstance(part.get("text"), str) and set(part) != {"text"} for part in parts
            )
            if metadata_bearing_text:
                return next(
                    (
                        cast(str, part["text"])
                        for part in parts
                        if isinstance(part, dict)
                        and isinstance(part.get("text"), str)
                        and part.get("thought") is not True
                    ),
                    None,
                )
    return projected_text


def raw_response_call_count(response: dict[str, Any]) -> int:
    covered_keys: set[tuple[int, str]] = {(id(response), "usageMetadata")} if "usageMetadata" in response else set()
    candidates = response.get("candidates")
    if candidates is None and isinstance(response.get("promptFeedback"), dict):
        _reject_unmodeled_structural_data(response, covered_keys=covered_keys)
        return 0
    if not isinstance(candidates, list):
        raise _UnsupportedRequest("Gemini output candidates are invalid")
    if not candidates:
        _reject_unmodeled_structural_data(response, covered_keys=covered_keys)
        return 0
    count = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise _UnsupportedRequest("Gemini candidate is invalid")
        finish_reason = str(candidate.get("finishReason", "")).upper()
        if finish_reason in {
            "MALFORMED_FUNCTION_CALL",
            "MALFORMED_RESPONSE",
            "MISSING_THOUGHT_SIGNATURE",
            "TOO_MANY_TOOL_CALLS",
            "UNEXPECTED_TOOL_CALL",
        }:
            raise _UnsupportedRequest("Gemini reported invalid function-tool output")
        content = candidate.get("content")
        if content is None:
            if finish_reason == "TOOL_CODE":
                raise _UnsupportedRequest("Gemini reported tool use without a function call")
            continue
        if not isinstance(content, dict):
            raise _UnsupportedRequest("Gemini response content is invalid")
        if content.get("role") not in {None, "model"}:
            raise _UnsupportedRequest("Gemini response role is invalid")
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            raise _UnsupportedRequest("Gemini response parts are invalid")
        candidate_count = 0
        for part in parts:
            if not isinstance(part, dict):
                raise _UnsupportedRequest("Gemini response part is invalid")
            allowed = {
                "fileData",
                "functionCall",
                "inlineData",
                "partMetadata",
                "text",
                "thought",
                "thoughtSignature",
                "videoMetadata",
            }
            if set(part) - allowed:
                raise _UnsupportedRequest("Gemini response part is not structurally covered")
            data_fields = set(part) & {"fileData", "functionCall", "inlineData", "text"}
            if len(data_fields) > 1:
                raise _UnsupportedRequest("Gemini response part mixes multiple data fields")
            if "thought" in part and not isinstance(part["thought"], bool):
                raise _UnsupportedRequest("Gemini response thought marker is invalid")
            if "thoughtSignature" in part and not isinstance(part["thoughtSignature"], str):
                raise _UnsupportedRequest("Gemini response thought signature is invalid")
            if "partMetadata" in part:
                if not isinstance(part["partMetadata"], dict):
                    raise _UnsupportedRequest("Gemini response part metadata must be an object")
                covered_keys.add((id(part), "partMetadata"))
            if "functionCall" in part:
                function_call = part["functionCall"]
                if (
                    not isinstance(function_call, dict)
                    or not isinstance(function_call.get("name"), str)
                    or not function_call["name"]
                    or ("args" in function_call and not isinstance(function_call["args"], dict))
                ):
                    raise _UnsupportedRequest("Gemini function call is incomplete")
                _reject_unmodeled_fields(
                    function_call,
                    {"args", "id", "name"},
                    "Gemini response function call",
                )
                covered_keys.add((id(part), "functionCall"))
                candidate_count += 1
            if any(key in part for key in {"functionResponse", "executableCode", "codeExecutionResult"}):
                raise _UnsupportedRequest("Gemini provider tool output is not covered")
        if finish_reason == "TOOL_CODE" and candidate_count == 0:
            raise _UnsupportedRequest("Gemini reported tool use without a function call")
        count += candidate_count
    _reject_unmodeled_structural_data(response, covered_keys=covered_keys)
    return count


def raw_response_texts(
    response: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
) -> RawResponseText:
    candidates_text: list[str] = []
    reasoning_texts: list[str] = []
    _reject_unmodeled_fields(response, RESPONSE_KEYS, "Gemini response")
    _reject_nonempty_metadata(response, {"promptFeedback"}, "Gemini response")
    model_status = response.get("modelStatus")
    if model_status is not None:
        if not isinstance(model_status, dict):
            raise _UnsupportedRequest("Gemini model status is invalid")
        _reject_unmodeled_fields(
            model_status,
            {"message", "modelStage", "retirementTime"},
            "Gemini model status",
        )
        for field in {"message", "modelStage", "retirementTime"}.intersection(model_status):
            if model_status[field] is not None and not isinstance(model_status[field], str):
                raise _UnsupportedRequest(f"Gemini model status {field} is invalid")
        if model_status.get("message") not in (None, ""):
            raise _UnsupportedRequest("Gemini model status message is not covered by output rails")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise _UnsupportedRequest("Gemini output candidates are missing")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise _UnsupportedRequest("Gemini response candidate is invalid")
        _reject_unmodeled_fields(
            candidate,
            CANDIDATE_KEYS,
            "Gemini response candidate",
        )
        content = candidate.get("content")
        if not isinstance(content, dict) or content.get("role") not in {None, "model"}:
            raise _UnsupportedRequest("Gemini response content is invalid")
        _reject_unmodeled_fields(content, {"parts", "role"}, "Gemini response content")
        _reject_nonempty_metadata(
            candidate,
            {
                "citationMetadata",
                "groundingAttributions",
                "groundingMetadata",
                "logprobsResult",
                "urlContextMetadata",
            },
            "Gemini candidate",
        )
        if candidate.get("finishMessage") not in (None, ""):
            raise _UnsupportedRequest("Gemini candidate finishMessage is not covered by output rails")
        parts = content.get("parts")
        if not isinstance(parts, list):
            raise _UnsupportedRequest("Gemini response parts are invalid")
        candidate_parts: list[str] = []
        has_call = False
        for part in parts:
            if not isinstance(part, dict):
                raise _UnsupportedRequest("Gemini response part is invalid")
            _reject_nonempty_metadata(part, {"partMetadata"}, "Gemini response part")
            signature = part.get("thoughtSignature")
            if signature is not None and not isinstance(signature, str):
                raise _UnsupportedRequest("Gemini thought signature is invalid")
            if "text" in part:
                _reject_unmodeled_fields(
                    part,
                    {"partMetadata", "text", "thought", "thoughtSignature"},
                    "Gemini response text part",
                )
                if not isinstance(part["text"], str):
                    raise _UnsupportedRequest("Gemini response text is invalid")
                if part.get("thought") is True:
                    if payload_policy.reasoning is ReasoningPolicy.REJECT:
                        raise _UnsupportedRequest("Gemini thought output is not covered")
                    if payload_policy.reasoning is ReasoningPolicy.CHECK_OUTPUT:
                        reasoning_texts.append(cast(str, part["text"]))
                    continue
                candidate_parts.append(cast(str, part["text"]))
            elif "functionCall" in part:
                has_call = True
            elif any(key in part for key in {"fileData", "inlineData", "videoMetadata"}):
                if payload_policy.multimodal is not MultimodalPolicy.TEXT_ONLY:
                    raise _UnsupportedRequest("Gemini multimodal output is not covered")
            else:
                raise _UnsupportedRequest("Gemini response part is not covered")
        if candidate_parts:
            candidates_text.append("\n".join(candidate_parts))
        elif not has_call:
            raise _UnsupportedRequest("Gemini candidate has no covered visible text")
    return RawResponseText(
        candidates=tuple(candidates_text),
        reasoning=tuple(reasoning_texts),
    )
