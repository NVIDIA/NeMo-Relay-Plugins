# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemini GenerateContent wire-shape rules."""

from __future__ import annotations

import json
from typing import Any, cast

from ..payload_policy import MultimodalPolicy, PayloadPolicy, ReasoningPolicy
from ..structural_tools import ToolCall
from .common import (
    RawResponseText,
    _reject_nonempty_metadata,
    _reject_unmodeled_fields,
    _reject_unmodeled_structural_data,
    _UnsupportedRequest,
    normalized_annotation,
    tool_call,
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
REQUEST_KEYS = frozenset({"contents", "generationConfig", "model", "systemInstruction", "tools"})
PART_DATA_KEYS = frozenset(
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


def _text_parts(values: list[Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            raise _UnsupportedRequest("Gemini part must be an object")
        if value.get("thought") is True:
            continue
        if "text" in value:
            if not isinstance(value["text"], str):
                raise _UnsupportedRequest("Gemini text part is invalid")
            parts.append({"type": "text", "text": value["text"], **{k: v for k, v in value.items() if k != "text"}})
        elif "inlineData" in value or "fileData" in value:
            kind = "inlineData" if "inlineData" in value else "fileData"
            parts.append({"type": "provider_native", "provider": "gemini", "kind": kind, "value": value})
        elif not PART_DATA_KEYS.intersection(value):
            parts.append(
                {
                    "type": "provider_native",
                    "provider": "gemini",
                    "kind": "unknown",
                    "value": value,
                }
            )
    return parts


def _part_data_key(part: dict[str, Any], context: str) -> str | None:
    keys = PART_DATA_KEYS.intersection(part)
    if len(keys) > 1:
        raise _UnsupportedRequest(f"{context} mixes multiple data fields")
    return next(iter(keys), None)


def _optional_id(value: object, context: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise _UnsupportedRequest(f"{context} id must be a non-empty string")


def _validate_function_call(value: object) -> str:
    if not isinstance(value, dict):
        raise _UnsupportedRequest("Gemini function call must be an object")
    _reject_unmodeled_fields(value, {"args", "id", "name"}, "Gemini function call")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise _UnsupportedRequest("Gemini function call name is missing")
    if "args" in value and not isinstance(value["args"], dict):
        raise _UnsupportedRequest("Gemini function call arguments must be an object")
    _optional_id(value.get("id"), "Gemini function call")
    return value.get("id", name)


def _validate_function_response(value: object) -> str:
    if not isinstance(value, dict):
        raise _UnsupportedRequest("Gemini function response must be an object")
    _reject_unmodeled_fields(value, {"id", "name", "parts", "response"}, "Gemini function response")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise _UnsupportedRequest("Gemini function response name is missing")
    if not isinstance(value.get("response"), dict):
        raise _UnsupportedRequest("Gemini function response payload must be an object")
    _optional_id(value.get("id"), "Gemini function response")
    nested = value.get("parts")
    if nested is not None:
        if not isinstance(nested, list):
            raise _UnsupportedRequest("Gemini function response parts must be an array")
        for part in nested:
            if not isinstance(part, dict):
                raise _UnsupportedRequest("Gemini function response part must be an object")
            kind = _part_data_key(part, "Gemini function response part")
            if kind in {"functionCall", "functionResponse"}:
                raise _UnsupportedRequest("Gemini function response cannot contain nested tool traffic")
            if kind == "text" and not isinstance(part.get("text"), str):
                raise _UnsupportedRequest("Gemini function response text part is invalid")
    return value.get("id", name)


def _function_response_content(value: dict[str, Any]) -> object:
    text = json.dumps(value["response"], separators=(",", ":"), sort_keys=True)
    nested = value.get("parts")
    if nested is None:
        return text
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for part in cast(list[dict[str, Any]], nested):
        content.append(
            {
                "type": "provider_native",
                "provider": "gemini",
                "kind": _part_data_key(part, "Gemini function response part") or "unknown",
                "value": part,
            }
        )
    return content


def _validate_unique_ids(ids: list[str], context: str) -> None:
    if len(ids) != len(set(ids)):
        raise _UnsupportedRequest(f"{context} contains duplicate effective ids")


def _system_text(value: object) -> str | None:
    if not isinstance(value, dict) or set(value) - {"parts", "role"}:
        raise _UnsupportedRequest("Gemini system instruction is invalid")
    if value.get("role") is not None and not isinstance(value["role"], str):
        raise _UnsupportedRequest("Gemini system instruction role must be a string")
    parts = value.get("parts")
    if not isinstance(parts, list):
        raise _UnsupportedRequest("Gemini system instruction parts must be an array")
    text: list[str] = []
    for part in parts:
        if not isinstance(part, dict) or _part_data_key(part, "Gemini system instruction part") != "text":
            raise _UnsupportedRequest("Gemini system instruction must contain only text parts")
        if set(part) - {"partMetadata", "text", "thought", "thoughtSignature"}:
            raise _UnsupportedRequest("Gemini system instruction part contains unmodeled fields")
        if not isinstance(part.get("text"), str):
            raise _UnsupportedRequest("Gemini system instruction text part is invalid")
        if part.get("partMetadata") not in (None, {}):
            raise _UnsupportedRequest("Gemini system instruction metadata is not covered")
        if part.get("thought") not in (None, False) or part.get("thoughtSignature") not in (None, ""):
            raise _UnsupportedRequest("Gemini system instruction thought content is not covered")
        text.append(part["text"])
    joined = "\n".join(text)
    return joined or None


def request_annotation(content: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system = content.get("systemInstruction")
    if system is not None:
        text = _system_text(system)
        if text:
            messages.append({"role": "system", "content": text})
    raw_contents = content.get("contents")
    if not isinstance(raw_contents, list):
        raise _UnsupportedRequest("Gemini contents must be an array")
    for value in raw_contents:
        if not isinstance(value, dict) or not isinstance(value.get("parts"), list):
            raise _UnsupportedRequest("Gemini content is invalid")
        role = value.get("role", "user")
        if role not in {"model", "user"}:
            raise _UnsupportedRequest("Gemini content role is unsupported")
        for part in value["parts"]:
            if not isinstance(part, dict):
                raise _UnsupportedRequest("Gemini part must be an object")
            _part_data_key(part, "Gemini content part")
        calls = [part["functionCall"] for part in value["parts"] if isinstance(part, dict) and "functionCall" in part]
        results = [
            part["functionResponse"] for part in value["parts"] if isinstance(part, dict) and "functionResponse" in part
        ]
        if calls and results:
            raise _UnsupportedRequest("Gemini content mixes calls and results")
        if results:
            if role != "user" or any(
                "functionResponse" not in part and part.get("thought") is not True for part in value["parts"]
            ):
                raise _UnsupportedRequest("Gemini function responses must be isolated in a user content item")
            response_ids = [_validate_function_response(raw) for raw in results]
            _validate_unique_ids(response_ids, "Gemini function responses")
            for raw in results:
                if not isinstance(raw, dict):
                    raise _UnsupportedRequest("Gemini function response must be an object")
                call_id = raw.get("id", raw.get("name"))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _function_response_content(raw),
                    }
                )
            continue
        message: dict[str, Any] = {"role": "assistant" if role == "model" else "user"}
        parts = _text_parts(value["parts"])
        if parts:
            if all(isinstance(part, dict) and set(part) == {"text", "type"} for part in parts):
                message["content"] = "\n".join(cast(str, part["text"]) for part in parts)
            else:
                message["content"] = parts
        if calls:
            if role != "model":
                raise _UnsupportedRequest("Gemini function calls must be in a model content item")
            call_ids = [_validate_function_call(raw) for raw in calls]
            _validate_unique_ids(call_ids, "Gemini function calls")
            message["tool_calls"] = [
                {
                    "id": raw.get("id", raw.get("name")),
                    "type": "function",
                    "function": {
                        "name": raw.get("name"),
                        "arguments": json.dumps(raw.get("args", {}), separators=(",", ":"), sort_keys=True),
                    },
                }
                for raw in calls
            ]
        messages.append(message)
    raw_tools = content.get("tools")
    tools = None
    if raw_tools is not None:
        if not isinstance(raw_tools, list):
            raise _UnsupportedRequest("Gemini tools must be an array")
        tools = []
        for group in raw_tools:
            if not isinstance(group, dict):
                raise _UnsupportedRequest("Gemini tool group must be an object")
            declarations = group.get("functionDeclarations", [])
            if not isinstance(declarations, list):
                raise _UnsupportedRequest("Gemini function declarations must be an array")
            for value in declarations:
                if not isinstance(value, dict):
                    raise _UnsupportedRequest("Gemini function declaration must be an object")
                function = {
                    "name": value.get("name"),
                    **{k: v for k, v in value.items() if k not in {"description", "name", "parameters"}},
                }
                if "parameters" in value:
                    function["parameters"] = value["parameters"]
                if "description" in value:
                    function["description"] = value["description"]
                tools.append(
                    {
                        "type": "function",
                        "function": function,
                    }
                )
            native = {key: item for key, item in group.items() if key != "functionDeclarations"}
            if native:
                tools.append(
                    {"type": "provider_native", "provider": "gemini", "kind": next(iter(native)), "value": native}
                )
    return normalized_annotation(content, messages=messages, modeled_keys=REQUEST_KEYS, api_specific=None, tools=tools)


def response_calls(response: dict[str, Any]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for candidate in response.get("candidates") or []:
        content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
        for part in content.get("parts") or []:
            raw = part.get("functionCall") if isinstance(part, dict) else None
            if isinstance(raw, dict):
                calls.append(tool_call(raw.get("id", raw.get("name")), raw.get("name"), raw.get("args", {})))
    return tuple(calls)


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
        # Code execution is deliberately excluded because Guardrails has no
        # matching validator.
        allowed_data.update({"inlineData", "fileData"})
    for parts in _part_lists(content):
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                raise _UnsupportedRequest("Gemini part must be an object")
            if set(part) - allowed_data - {"thought", "thoughtSignature"}:
                raise _UnsupportedRequest("Gemini part contains unmodeled fields")
            _part_data_key(part, "Gemini part")
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
            if "functionCall" in part:
                _validate_function_call(part["functionCall"])
            if "functionResponse" in part:
                _validate_function_response(part["functionResponse"])


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
