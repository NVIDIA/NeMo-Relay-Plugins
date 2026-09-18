# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OCI Generative AI chat wire-shape rules."""

from __future__ import annotations

from typing import Any, cast

from ..payload_policy import PayloadPolicy, ReasoningPolicy
from .common import (
    RawResponseText,
    _reject_nonempty_metadata,
    _reject_unmodeled_fields,
    _reject_unmodeled_structural_data,
    _required_json_object_text,
    _structural_key,
    _UnsupportedRequest,
)

API_FORMATS = frozenset({"GENERIC", "COHERE", "COHEREV2"})
UPPERCASE_ROLES = frozenset({"SYSTEM", "USER", "ASSISTANT", "TOOL", "CHATBOT"})
ENVELOPE_KEYS = frozenset({"chatRequest", "compartmentId", "servingMode"})
SERVING_MODE_KEYS = frozenset({"endpointId", "modelId", "servingType"})
GENERIC_MESSAGE_KEYS = frozenset({"content", "role", "toolCallId", "toolCalls"})
GENERIC_NAMED_ROLES = frozenset({"ASSISTANT", "DEVELOPER", "SYSTEM", "USER"})
GENERIC_ASSISTANT_METADATA = frozenset({"annotations", "reasoningContent", "refusal"})
COHERE_V2_ASSISTANT_METADATA = frozenset({"citations", "toolPlan"})
COHERE_HISTORY_KEYS = frozenset({"message", "role"})
RESPONSE_ENVELOPE_KEYS = frozenset({"chatResponse", "modelId", "modelVersion"})
RESPONSE_KEYS_BY_FORMAT = {
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
    "COHEREV2": frozenset(
        {
            "apiFormat",
            "errorMessage",
            "finishReason",
            "id",
            "logProbabilities",
            "message",
            "usage",
        }
    ),
}
GENERIC_CHOICE_KEYS = frozenset(
    {"finishReason", "groundingMetadata", "index", "logprobs", "message", "serviceTier", "usage"}
)


def has_text_parts(messages: list[Any]) -> bool:
    return any(
        isinstance(message, dict)
        and isinstance(message.get("content"), list)
        and any(isinstance(part, dict) and part.get("type") == "TEXT" for part in message["content"])
        for message in messages
    )


def request_variant(content: dict[str, Any]) -> str:
    chat_request = content.get("chatRequest", content)
    if not isinstance(chat_request, dict):
        raise _UnsupportedRequest("OCI chat request is invalid")
    api_format = chat_request.get("apiFormat", "GENERIC")
    if not isinstance(api_format, str) or api_format.upper() not in API_FORMATS:
        raise _UnsupportedRequest("OCI request apiFormat is not supported")
    return api_format.upper()


def validate_response_variant(codec_variant: str | None, response: dict[str, Any]) -> None:
    chat_response = response.get("chatResponse", response)
    if not isinstance(chat_response, dict):
        raise _UnsupportedRequest("OCI chat response is invalid")
    api_format = chat_response.get("apiFormat", "GENERIC")
    if not isinstance(api_format, str) or api_format.upper() != codec_variant:
        raise _UnsupportedRequest("OCI response apiFormat does not match the request")


def response_variant(response: dict[str, Any]) -> str:
    """Return the validated OCI response dialect."""

    chat_response = response.get("chatResponse", response)
    if not isinstance(chat_response, dict):
        raise _UnsupportedRequest("OCI chat response is invalid")
    api_format = chat_response.get("apiFormat", "GENERIC")
    if not isinstance(api_format, str) or api_format.upper() not in API_FORMATS:
        raise _UnsupportedRequest("OCI response apiFormat is not supported")
    return api_format.upper()


def response_candidate_payloads(response: dict[str, Any], max_segments: int) -> tuple[dict[str, Any], ...]:
    chat_response = response.get("chatResponse", response)
    if isinstance(chat_response, dict) and str(chat_response.get("apiFormat", "GENERIC")).upper() == "GENERIC":
        choices = chat_response.get("choices")
        if isinstance(choices, list) and len(choices) > 1:
            if len(choices) > max_segments:
                raise _UnsupportedRequest("OCI response has too many choices")
            if chat_response is response:
                return tuple({**response, "choices": [choice]} for choice in choices)
            return tuple({**response, "chatResponse": {**chat_response, "choices": [choice]}} for choice in choices)
    return (response,)


def expected_native_response_text(response: dict[str, Any], projected_text: str | None) -> str | None:
    """Mirror the text exposed by Relay's OCI response codec."""

    chat_response = response.get("chatResponse", response)
    if isinstance(chat_response, dict):
        api_format = str(chat_response.get("apiFormat", "GENERIC")).upper()
        message: object = None
        if api_format == "GENERIC" and isinstance(chat_response.get("choices"), list):
            choices = chat_response["choices"]
            if len(choices) == 1 and isinstance(choices[0], dict):
                message = choices[0].get("message")
        elif api_format == "COHEREV2":
            message = chat_response.get("message")
        if isinstance(message, dict) and isinstance(message.get("refusal"), str):
            return None
    return projected_text


def _validate_oci_response_calls(calls: object, api_format: str) -> int:
    if not isinstance(calls, list):
        raise _UnsupportedRequest("OCI response toolCalls are invalid")
    for call in calls:
        if not isinstance(call, dict):
            raise _UnsupportedRequest("OCI response tool call is invalid")
        if api_format == "COHEREV2":
            body = call.get("function")
            if not isinstance(body, dict):
                raise _UnsupportedRequest("OCI COHEREV2 function call body is missing")
            if not isinstance(call.get("id"), str) or not call["id"]:
                raise _UnsupportedRequest("OCI COHEREV2 function call id is missing")
            if str(call.get("type", "")).upper() != "FUNCTION":
                raise _UnsupportedRequest("OCI COHEREV2 tool call is not a function call")
            arguments = body.get("arguments")
            _reject_unmodeled_fields(call, {"function", "id", "type"}, "OCI COHEREV2 tool call")
            _reject_unmodeled_fields(body, {"arguments", "name"}, "OCI COHEREV2 function")
        else:
            body = call
            if api_format == "GENERIC" and "arguments" in call and "parameters" in call:
                raise _UnsupportedRequest("OCI GENERIC function call has two argument payloads")
            arguments = call.get("arguments", call.get("parameters"))
            if api_format == "GENERIC" and (
                not isinstance(call.get("id"), str)
                or not call["id"]
                or str(call.get("type", "FUNCTION")).upper() != "FUNCTION"
            ):
                raise _UnsupportedRequest("OCI GENERIC function call identity is incomplete")
            allowed = (
                {"name", "parameters"} if api_format == "COHERE" else {"arguments", "id", "name", "parameters", "type"}
            )
            _reject_unmodeled_fields(call, allowed, "OCI function call")
        if not isinstance(body.get("name"), str) or not body["name"]:
            raise _UnsupportedRequest("OCI function call name is missing")
        if isinstance(arguments, str):
            _required_json_object_text(arguments, "OCI function-call arguments")
        elif not isinstance(arguments, dict):
            raise _UnsupportedRequest("OCI function-call arguments must be an object")
    return len(calls)


def _validate_oci_envelope(content: dict[str, Any]) -> None:
    if "chatRequest" not in content:
        return
    if set(content) - ENVELOPE_KEYS:
        raise _UnsupportedRequest("OCI envelope contains unmodeled fields")
    serving_mode = content.get("servingMode")
    if serving_mode is not None and (not isinstance(serving_mode, dict) or set(serving_mode) - SERVING_MODE_KEYS):
        raise _UnsupportedRequest("OCI servingMode contains unmodeled fields")


def _validate_oci_cohere_history(chat_request: dict[str, Any]) -> None:
    history = chat_request.get("chatHistory", [])
    if not isinstance(history, list):
        return
    if any(isinstance(turn, dict) and set(turn) - COHERE_HISTORY_KEYS for turn in history):
        raise _UnsupportedRequest("OCI COHERE history contains unmodeled fields")


def _validate_oci_cohere_v2_citations(value: object) -> None:
    """Validate citation provenance without treating it as executable tool traffic."""

    if value is None:
        return
    if not isinstance(value, list):
        raise _UnsupportedRequest("OCI COHEREV2 citations must be an array")
    for citation in value:
        if not isinstance(citation, dict):
            raise _UnsupportedRequest("OCI COHEREV2 citation must be an object")
        _reject_unmodeled_fields(
            citation,
            {"contentIndex", "end", "sources", "start", "text", "type"},
            "OCI COHEREV2 citation",
        )

        for field in {"contentIndex", "end", "start"}.intersection(citation):
            if citation[field] is not None and (
                isinstance(citation[field], bool) or not isinstance(citation[field], int)
            ):
                raise _UnsupportedRequest(f"OCI COHEREV2 citation {field} must be an integer")
        if citation.get("text") is not None and not isinstance(citation["text"], str):
            raise _UnsupportedRequest("OCI COHEREV2 citation text must be a string")
        if citation.get("type") is not None and citation["type"] not in {
            "PLAN",
            "TEXT_CONTENT",
            "THINKING_CONTENT",
        }:
            raise _UnsupportedRequest("OCI COHEREV2 citation type is invalid")
        sources = citation.get("sources", [])
        if sources is None:
            sources = []
        if not isinstance(sources, list):
            raise _UnsupportedRequest("OCI COHEREV2 citation sources must be an array")
        for source in sources:
            if not isinstance(source, dict):
                raise _UnsupportedRequest("OCI COHEREV2 citation source must be an object")
            _reject_unmodeled_fields(
                source,
                {"document", "tool", "type"},
                "OCI COHEREV2 citation source",
            )
            source_type = source.get("type")
            if source_type not in {"DOCUMENT", "TOOL"}:
                raise _UnsupportedRequest("OCI COHEREV2 citation source type is invalid")
            active_field = "tool" if source_type == "TOOL" else "document"
            inactive_field = "document" if source_type == "TOOL" else "tool"
            if source.get(inactive_field) is not None:
                raise _UnsupportedRequest("OCI COHEREV2 citation source mixes source variants")
            payload = source.get(active_field)
            if payload is None:
                continue
            if not isinstance(payload, dict):
                raise _UnsupportedRequest("OCI COHEREV2 citation source payload must be an object")
            allowed_payload_fields = {"id", "toolOutput"} if source_type == "TOOL" else {"document", "id"}
            _reject_unmodeled_fields(
                payload,
                allowed_payload_fields,
                "OCI COHEREV2 citation source payload",
            )
            if payload.get("id") is not None and not isinstance(payload["id"], str):
                raise _UnsupportedRequest("OCI COHEREV2 citation source id must be a string")


def raw_response_call_count(response: dict[str, Any]) -> int:
    chat_response = response.get("chatResponse", response)
    if not isinstance(chat_response, dict):
        raise _UnsupportedRequest("OCI chat response is invalid")
    api_format = str(chat_response.get("apiFormat", "GENERIC")).upper()
    covered_keys: set[tuple[int, str]] = set()
    if "usage" in chat_response:
        covered_keys.add((id(chat_response), "usage"))
    count = 0
    if api_format == "COHERE":
        calls = chat_response.get("toolCalls")
        if calls is None:
            calls = []
        covered_keys.add((id(chat_response), "toolCalls"))
        count = _validate_oci_response_calls(calls, api_format)
        finish_reasons = [chat_response.get("finishReason")]
    elif api_format == "COHEREV2":
        message = chat_response.get("message")
        if message is None:
            _reject_unmodeled_structural_data(response, covered_keys=covered_keys)
            return 0
        if not isinstance(message, dict):
            raise _UnsupportedRequest("OCI COHEREV2 response message is invalid")
        if message.get("role") not in {None, "ASSISTANT"}:
            raise _UnsupportedRequest("OCI COHEREV2 response role is invalid")
        calls = message.get("toolCalls")
        if calls is None:
            calls = []
        covered_keys.add((id(message), "toolCalls"))
        if "toolPlan" in message:
            if not isinstance(message["toolPlan"], str):
                raise _UnsupportedRequest("OCI COHEREV2 tool plan is invalid")
            covered_keys.add((id(message), "toolPlan"))
        if "citations" in message:
            _validate_oci_cohere_v2_citations(message["citations"])
            covered_keys.add((id(message), "citations"))
        count = _validate_oci_response_calls(calls, api_format)
        finish_reasons = [chat_response.get("finishReason")]
    elif api_format == "GENERIC":
        choices = chat_response.get("choices")
        if not isinstance(choices, list):
            raise _UnsupportedRequest("OCI output choices are invalid")
        if not choices:
            _reject_unmodeled_structural_data(response, covered_keys=covered_keys)
            return 0
        finish_reasons = []
        for choice in choices:
            if not isinstance(choice, dict):
                raise _UnsupportedRequest("OCI response choice is invalid")
            finish_reasons.append(choice.get("finishReason"))
            message = choice.get("message")
            if message is None:
                continue
            if not isinstance(message, dict):
                raise _UnsupportedRequest("OCI response message is invalid")
            if message.get("role") not in {None, "ASSISTANT"}:
                raise _UnsupportedRequest("OCI GENERIC response role is invalid")
            calls = message.get("toolCalls")
            if calls is None:
                calls = []
            covered_keys.add((id(message), "toolCalls"))
            choice_count = _validate_oci_response_calls(calls, api_format)
            if str(choice.get("finishReason")).upper() in {"TOOL_CALL", "TOOL_CALLS"} and choice_count == 0:
                raise _UnsupportedRequest("OCI reported tool use without a tool call")
            count += choice_count
    else:
        raise _UnsupportedRequest("OCI response apiFormat is not supported")
    if (
        api_format != "GENERIC"
        and any(str(reason).upper() in {"TOOL_CALL", "TOOL_CALLS"} for reason in finish_reasons)
        and count == 0
    ):
        raise _UnsupportedRequest("OCI reported tool use without a tool call")
    _reject_unmodeled_structural_data(response, covered_keys=covered_keys)
    return count


def _validate_oci_generic_messages(
    chat_request: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
    api_format: str,
    allow_structural_tools: bool,
    require_text_coverage: bool,
) -> None:
    messages = chat_request.get("messages", [])
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).upper()
        allowed_keys = set(GENERIC_MESSAGE_KEYS)
        if api_format == "GENERIC":
            if role in GENERIC_NAMED_ROLES:
                allowed_keys.add("name")
            if role == "ASSISTANT":
                allowed_keys.update(GENERIC_ASSISTANT_METADATA)
        elif api_format == "COHEREV2" and role == "ASSISTANT":
            allowed_keys.update(COHERE_V2_ASSISTANT_METADATA)
        if set(message) - allowed_keys:
            raise _UnsupportedRequest("OCI message contains unmodeled fields")
        for key in {"name", "reasoningContent", "refusal", "toolPlan"} & message.keys():
            if message[key] is not None and not isinstance(message[key], str):
                raise _UnsupportedRequest("OCI message metadata has an unsupported shape")
        for key in {"annotations", "citations"} & message.keys():
            value = message[key]
            if value is not None and (not isinstance(value, list) or any(not isinstance(item, dict) for item in value)):
                raise _UnsupportedRequest("OCI message metadata has an unsupported shape")
            if key == "citations" and api_format == "COHEREV2":
                _validate_oci_cohere_v2_citations(value)
                continue
            if isinstance(value, list):
                for item in value:
                    _reject_unmodeled_structural_data(item)
        if require_text_coverage:
            for key in {"name", "reasoningContent", "refusal", "toolPlan"} & message.keys():
                reasoning_metadata = key in {"reasoningContent", "toolPlan"}
                if message[key] not in (None, "") and not (
                    reasoning_metadata
                    and payload_policy.reasoning in {ReasoningPolicy.FINAL_ANSWER_ONLY, ReasoningPolicy.CHECK_OUTPUT}
                ):
                    raise _UnsupportedRequest("OCI message contains unchecked text metadata")
            for key in {"annotations", "citations"} & message.keys():
                if message[key] not in (None, []):
                    raise _UnsupportedRequest("OCI message contains unchecked text metadata")
        # Relay intentionally normalizes these structural fields only on their
        # expected roles. Reject them on every role before normalization can
        # hide provider-visible tool traffic.
        calls = message.get("toolCalls")
        call_id = message.get("toolCallId")
        if not allow_structural_tools and calls not in (None, []):
            raise _UnsupportedRequest("OCI tool calls are not covered by input rails")
        if not allow_structural_tools and call_id is not None:
            raise _UnsupportedRequest("OCI tool results are not covered by input rails")
        if allow_structural_tools and calls not in (None, []):
            if role != "ASSISTANT":
                raise _UnsupportedRequest("OCI tool calls must be carried by an assistant message")
            _validate_oci_response_calls(calls, api_format)
        if (
            allow_structural_tools
            and call_id is not None
            and (role != "TOOL" or not isinstance(call_id, str) or not call_id)
        ):
            raise _UnsupportedRequest("OCI tool result identity is invalid")
        if allow_structural_tools and role == "TOOL" and (not isinstance(call_id, str) or not call_id):
            raise _UnsupportedRequest("OCI tool message is missing a toolCallId")
        parts = message.get("content")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if isinstance(part_type, str) and part_type.upper() != "TEXT" and _structural_key(part_type):
                raise _UnsupportedRequest("OCI content contains unmodeled tool traffic")
            if part_type == "TEXT" and set(part) - {"type", "text"}:
                raise _UnsupportedRequest("OCI text part contains unmodeled fields")
            if part_type != "TEXT":
                _reject_unmodeled_structural_data(part)


def validate_raw_coverage(
    content: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
    allow_structural_tools: bool,
    require_text_coverage: bool,
) -> None:
    chat_request = content.get("chatRequest", content)
    if not isinstance(chat_request, dict):
        raise _UnsupportedRequest("OCI chatRequest must be an object")
    api_format = chat_request.get("apiFormat", "GENERIC")
    if not isinstance(api_format, str) or api_format.upper() not in API_FORMATS:
        raise _UnsupportedRequest("OCI apiFormat is not supported")
    _validate_oci_envelope(content)
    if api_format.upper() == "COHERE":
        _validate_oci_cohere_history(chat_request)
    else:
        _validate_oci_generic_messages(
            chat_request,
            payload_policy=payload_policy,
            api_format=api_format.upper(),
            allow_structural_tools=allow_structural_tools,
            require_text_coverage=require_text_coverage,
        )


def _oci_message_text(
    message: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
    api_format: str,
    reasoning_texts: list[str] | None = None,
) -> str | None:
    """Return one OCI candidate's complete visible text, if it has any."""

    allowed_message_fields = (
        set(GENERIC_MESSAGE_KEYS) | set(GENERIC_ASSISTANT_METADATA)
        if api_format == "GENERIC"
        else {"citations", "content", "role", "toolCalls", "toolPlan"}
    )
    _reject_unmodeled_fields(message, allowed_message_fields, "OCI response message")
    _reject_nonempty_metadata(message, {"annotations", "citations"}, "OCI response message")
    if message.get("toolCallId") is not None:
        raise _UnsupportedRequest("OCI assistant response carries a tool-result identity")
    reasoning_values = (message.get("reasoningContent"), message.get("toolPlan"))
    if any(value is not None and not isinstance(value, str) for value in reasoning_values):
        raise _UnsupportedRequest("OCI reasoning output is invalid")
    present_reasoning = [cast(str, value) for value in reasoning_values if value not in (None, "")]
    if present_reasoning:
        if payload_policy.reasoning is ReasoningPolicy.REJECT:
            raise _UnsupportedRequest("OCI response carries unchecked text metadata")
        if payload_policy.reasoning is ReasoningPolicy.CHECK_OUTPUT:
            if reasoning_texts is None:
                raise _UnsupportedRequest("OCI reasoning output cannot be projected")
            reasoning_texts.extend(present_reasoning)

    refusal = message.get("refusal")
    content = message.get("content")
    if refusal is not None and not isinstance(refusal, str):
        raise _UnsupportedRequest("OCI refusal output is invalid")
    if isinstance(refusal, str) and content not in (None, [], ""):
        raise _UnsupportedRequest("OCI content and refusal cannot be ordered losslessly")
    if isinstance(refusal, str):
        return refusal
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        segments: list[str] = []
        for part in content:
            if (
                not isinstance(part, dict)
                or part.get("type") != "TEXT"
                or part.get("text") is not None
                and not isinstance(part.get("text"), str)
                or set(part) - {"type", "text"}
            ):
                raise _UnsupportedRequest("OCI response content part is not covered")
            segments.append(cast(str, part.get("text") or ""))
        return "".join(segments) if segments else None
    if content is not None:
        raise _UnsupportedRequest("OCI response content is invalid")
    return None


def raw_response_texts(
    response: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
) -> RawResponseText:
    candidates_text: list[str] = []
    texts: list[str] = []
    reasoning_texts: list[str] = []
    chat_response = response.get("chatResponse", response)
    if not isinstance(chat_response, dict):
        raise _UnsupportedRequest("OCI chat response is invalid")
    if chat_response is not response:
        _reject_unmodeled_fields(response, RESPONSE_ENVELOPE_KEYS, "OCI response envelope")
    api_format = str(chat_response.get("apiFormat", "GENERIC")).upper()
    allowed_response_keys = RESPONSE_KEYS_BY_FORMAT.get(api_format)
    if allowed_response_keys is None:
        raise _UnsupportedRequest("OCI response apiFormat is not supported")
    _reject_unmodeled_fields(chat_response, allowed_response_keys, f"OCI {api_format} response")
    if api_format == "COHERE":
        _reject_nonempty_metadata(
            chat_response,
            {
                "chatHistory",
                "citations",
                "documents",
                "searchQueries",
                "searchResults",
            },
            "OCI COHERE response",
        )
        for field in {"errorMessage", "prompt"}.intersection(chat_response):
            if chat_response[field] is not None and not isinstance(chat_response[field], str):
                raise _UnsupportedRequest(f"OCI COHERE response {field} is invalid")
            if chat_response[field] not in (None, ""):
                raise _UnsupportedRequest(f"OCI COHERE response {field} is not covered by output rails")
        text = chat_response.get("text")
        if text is not None and not isinstance(text, str):
            raise _UnsupportedRequest("OCI COHERE response text is invalid")
        if isinstance(text, str):
            texts.append(text)
    elif api_format == "GENERIC":
        choices = chat_response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise _UnsupportedRequest("OCI output choices are missing")
        for choice in choices:
            if not isinstance(choice, dict):
                raise _UnsupportedRequest("OCI GENERIC response choice is invalid")
            _reject_unmodeled_fields(
                choice,
                GENERIC_CHOICE_KEYS,
                "OCI GENERIC response choice",
            )
            _reject_nonempty_metadata(
                choice,
                {"groundingMetadata", "logprobs"},
                "OCI GENERIC response choice",
            )
            message = choice.get("message")
            if not isinstance(message, dict):
                raise _UnsupportedRequest("OCI response message is invalid")
            candidate_text = _oci_message_text(
                message,
                payload_policy=payload_policy,
                api_format="GENERIC",
                reasoning_texts=reasoning_texts,
            )
            if candidate_text is not None:
                candidates_text.append(candidate_text)
            elif not message.get("toolCalls"):
                raise _UnsupportedRequest("OCI candidate has no covered visible text")
    elif api_format == "COHEREV2":
        _reject_nonempty_metadata(
            chat_response,
            {"logProbabilities"},
            "OCI COHEREV2 response",
        )
        if "errorMessage" in chat_response:
            error_message = chat_response["errorMessage"]
            if error_message is not None and not isinstance(error_message, str):
                raise _UnsupportedRequest("OCI COHEREV2 response errorMessage is invalid")
            if error_message not in (None, ""):
                raise _UnsupportedRequest("OCI COHEREV2 response errorMessage is not covered by output rails")
        message = chat_response.get("message")
        if not isinstance(message, dict):
            raise _UnsupportedRequest("OCI response message is invalid")
        candidate_text = _oci_message_text(
            message,
            payload_policy=payload_policy,
            api_format="COHEREV2",
            reasoning_texts=reasoning_texts,
        )
        if candidate_text is not None:
            texts.append(candidate_text)
        elif not message.get("toolCalls"):
            raise _UnsupportedRequest("OCI response has no covered visible text")
    else:
        raise _UnsupportedRequest("OCI response apiFormat is not supported")
    return RawResponseText(
        candidates=tuple(candidates_text),
        fragments=tuple(texts),
        reasoning=tuple(reasoning_texts),
        fragment_separator="",
    )
