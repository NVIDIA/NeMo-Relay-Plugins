# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized request projection and provider response inspection.

Relay's active codec owns request decoding and encoding. This module projects
that normalized request into Guardrails and inspects policy-bearing response
data that Relay's normalized types do not expose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from nemo_relay_plugin import AnnotatedLlmRequest, Json, LlmRequest

from .payload_policy import (
    MultimodalPolicy,
    PayloadPolicy,
    ProjectedOutputText,
    ReasoningPolicy,
    is_supported_normalized_modality,
    validate_anthropic_reasoning,
    validate_openai_responses_reasoning,
)
from .provider_coverage import anthropic as anthropic_coverage
from .provider_coverage import gemini as gemini_coverage
from .provider_coverage import inspect_response as inspect_provider_response
from .provider_coverage import oci as oci_coverage
from .provider_coverage import openai_responses as openai_responses_coverage
from .provider_coverage.common import (
    MAX_RESPONSE_SEGMENTS,
    ResponseInspection,
    _reject_structural_extensions,
    _reject_unmodeled_fields,
    _structural_key,
    _UnsupportedRequest,
    tool_call,
)
from .structural_tools import ToolCall
from .tool_projection import (
    ToolProjectionError,
    ToolRequestProjection,
    project_request_tools,
)

_SUPPORTED_CODECS = frozenset(
    {
        "openai_chat",
        "openai_responses",
        "anthropic_messages",
        "gemini_generate_content",
        "oci_genai",
    }
)

# Provider request discriminators and known wire hazards that Relay's
# normalized request does not expose. Unrelated provider metadata is preserved;
# hidden conversation state and structural variants still reject.
_TEXT_ROLES = frozenset({"system", "user", "assistant", "developer"})

_TEXT_PART_METADATA_BY_CODEC = {
    "anthropic_messages": anthropic_coverage.TEXT_PART_METADATA,
    "gemini_generate_content": gemini_coverage.TEXT_PART_METADATA,
    "openai_responses": openai_responses_coverage.TEXT_PART_METADATA,
}


def _project_text_content(
    content: Json,
    field: str,
    allowed_metadata: frozenset[str],
    *,
    codec_name: str = "",
    payload_policy: PayloadPolicy | None = None,
    allow_structural_tools: bool = False,
    allow_tool_results: bool = False,
    allow_responses_refusal: bool = False,
) -> str:
    """Return all ordered text represented by one normalized content value."""

    payload_policy = payload_policy or PayloadPolicy()
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _UnsupportedRequest(f"{field} is not normalized text")

    segments: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if payload_policy.multimodal is MultimodalPolicy.TEXT_ONLY and is_supported_normalized_modality(
                codec_name, part
            ):
                continue
            if (
                payload_policy.reasoning in {ReasoningPolicy.FINAL_ANSWER_ONLY, ReasoningPolicy.CHECK_OUTPUT}
                and codec_name == "anthropic_messages"
                and part.get("type") == "provider_native"
                and part.get("provider") == "anthropic_messages"
                and part.get("kind") in ("thinking", "redacted_thinking")
                and validate_anthropic_reasoning(part.get("value"))
            ):
                continue
            if allow_structural_tools and part.get("type") == "tool_use":
                continue
            if allow_tool_results and part.get("type") == "tool_result":
                continue
            if allow_responses_refusal and part.get("type") == "refusal":
                if set(part) != {"refusal", "type"} or not isinstance(part.get("refusal"), str):
                    raise _UnsupportedRequest(f"{field} contains an invalid refusal block")
                segments.append(cast(str, part["refusal"]))
                continue
        if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
            raise _UnsupportedRequest(f"{field} contains non-text content")
        if set(part) - {"type", "text"} - allowed_metadata:
            raise _UnsupportedRequest(f"{field} contains unmodeled text-part metadata")
        if part.get("thought") not in (None, False):
            raise _UnsupportedRequest(f"{field} contains unchecked thought content")
        if part.get("thoughtSignature") not in (None, ""):
            raise _UnsupportedRequest(f"{field} contains unchecked thought metadata")
        for metadata_field in {"annotations", "citations", "logprobs", "partMetadata"}.intersection(part):
            if part[metadata_field] not in (None, [], {}):
                raise _UnsupportedRequest(f"{field} contains unchecked text-part metadata")
        segments.append(cast(str, part["text"]))
    return "".join(segments)


def _validate_annotation_state(
    annotated: AnnotatedLlmRequest,
    codec_name: str,
    *,
    allow_structural_tools: bool,
    require_text_coverage: bool = True,
) -> None:
    extra = annotated.get("extra", {})
    if not isinstance(extra, dict):
        raise _UnsupportedRequest("normalized provider extras are invalid")
    if not codec_name and extra:
        raise _UnsupportedRequest("runtime codec fields cannot prove complete coverage")

    if codec_name == "openai_chat":
        if extra.get("prediction") is not None:
            raise _UnsupportedRequest("OpenAI prediction content is not covered by input rails")
        template = extra.get("chat_template_kwargs")
        if template is not None and (
            not isinstance(template, dict)
            or set(template) != {"enable_thinking"}
            or not isinstance(template.get("enable_thinking"), bool)
        ):
            raise _UnsupportedRequest("OpenAI Chat template controls are not supported")
    if codec_name == "gemini_generate_content" and extra.get("cachedContent") is not None:
        raise _UnsupportedRequest("the request refers to provider-side cached content")
    if codec_name == "oci_genai" and require_text_coverage and extra.get("documents") not in (None, []):
        raise _UnsupportedRequest("OCI document context is not covered by input rails")
    if codec_name == "anthropic_messages" and extra.get("context_management") not in (None, {}):
        raise _UnsupportedRequest("the request refers to provider-side context state")
    if allow_structural_tools:
        controls = {"chat_template_kwargs", "client_metadata", "toolConfig"}
        if any(_structural_key(key) for key in set(extra) - controls):
            raise _UnsupportedRequest("the provider request contains unmodeled tool state")
    if annotated.get("previous_response_id") is not None:
        raise _UnsupportedRequest("the request refers to provider-side conversation state")

    api_specific = annotated.get("api_specific")
    if api_specific is None:
        return
    if not isinstance(api_specific, dict):
        raise _UnsupportedRequest("normalized provider fields are invalid")
    if not codec_name:
        raise _UnsupportedRequest("runtime codec provider fields cannot prove complete coverage")
    if api_specific.get("api") == "openai_responses" and (
        any(api_specific.get(key) is not None for key in {"conversation", "prompt"})
        or api_specific.get("context_management") not in (None, [])
    ):
        raise _UnsupportedRequest("the request refers to provider-side prompt or conversation state")
    if api_specific.get("api") == "anthropic_messages" and api_specific.get("container") is not None:
        raise _UnsupportedRequest("the request refers to provider-side container state")
    if (
        allow_structural_tools
        and api_specific.get("api") == "openai_chat"
        and any(api_specific.get(key) is not None for key in {"function_call", "functions", "web_search_options"})
    ):
        raise _UnsupportedRequest("the request contains an unsupported legacy or hosted tool surface")


def _project_responses_native_message(
    message: dict[str, Any],
    payload_policy: PayloadPolicy,
) -> tuple[str, str]:
    """Project a Responses message that Relay kept exact because it has an item id."""

    _reject_unmodeled_fields(
        message,
        {"kind", "provider", "role", "value"},
        "OpenAI Responses provider-native message",
    )
    value = message.get("value")
    if message.get("provider") != "openai_responses" or message.get("kind") != "message" or not isinstance(value, dict):
        raise _UnsupportedRequest("OpenAI Responses provider-native message is invalid")
    role = value.get("role")
    phase = value.get("phase")
    if (
        role not in _TEXT_ROLES
        or value.get("type") not in (None, "message")
        or (phase is not None and (role != "assistant" or phase not in openai_responses_coverage.MESSAGE_PHASES))
        or ("id" in value and (not isinstance(value["id"], str) or not value["id"]))
        or value.get("status") not in (None, "completed", "in_progress", "incomplete")
    ):
        raise _UnsupportedRequest("OpenAI Responses provider-native message is invalid")
    _reject_unmodeled_fields(
        value,
        {"content", "id", "phase", "role", "status", "type"},
        "OpenAI Responses provider-native message",
    )

    content = value.get("content")
    if isinstance(content, str):
        return cast(str, role), content
    if not isinstance(content, list):
        raise _UnsupportedRequest("OpenAI Responses message content is invalid")

    normalized: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            raise _UnsupportedRequest("OpenAI Responses message content is invalid")
        block_type = block.get("type")
        if block_type in ("input_text", "output_text") and isinstance(block.get("text"), str):
            _reject_unmodeled_fields(
                block,
                {"annotations", "logprobs", "text", "type"},
                "OpenAI Responses text block",
            )
            normalized.append({"type": "text", **{key: child for key, child in block.items() if key != "type"}})
        elif block_type == "refusal" and isinstance(block.get("refusal"), str):
            _reject_unmodeled_fields(block, {"refusal", "type"}, "OpenAI Responses refusal block")
            normalized.append(block)
        elif block_type == "input_image":
            _reject_unmodeled_fields(
                block,
                {"detail", "file_id", "image_url", "type"},
                "OpenAI Responses image block",
            )
            normalized.append({"type": "image", "image": {key: child for key, child in block.items() if key != "type"}})
        elif block_type == "input_file":
            _reject_unmodeled_fields(
                block,
                {"file_data", "file_id", "file_url", "filename", "type"},
                "OpenAI Responses file block",
            )
            normalized.append({"type": "file", "file": {key: child for key, child in block.items() if key != "type"}})
        else:
            raise _UnsupportedRequest("OpenAI Responses message content is not covered")

    return cast(str, role), _project_text_content(
        normalized,
        "message content",
        openai_responses_coverage.TEXT_PART_METADATA,
        codec_name="openai_responses",
        payload_policy=payload_policy,
        allow_responses_refusal=role == "assistant",
    )


def _project_normalized_messages(
    messages: list[Any],
    codec_name: str,
    allowed_metadata: frozenset[str],
    *,
    payload_policy: PayloadPolicy | None = None,
    allow_structural_tools: bool = False,
    allow_tool_results: bool = False,
    require_user: bool = True,
) -> tuple[list[dict[str, str]], int | None, str | None]:
    payload_policy = payload_policy or PayloadPolicy()
    projected: list[dict[str, str]] = []
    latest_user_index: int | None = None
    latest_user_text: str | None = None
    for message in messages:
        if not isinstance(message, dict):
            raise _UnsupportedRequest("normalized message is not an object")

        role = message.get("role")
        if role == "provider_native" and codec_name == "openai_chat":
            _reject_unmodeled_fields(
                message,
                {"kind", "provider", "role", "value"},
                "OpenAI Chat provider-native message",
            )
            value = message.get("value")
            if (
                message.get("provider") != "openai_chat"
                or message.get("kind") != "assistant"
                or not isinstance(value, dict)
                or value.get("role") != "assistant"
            ):
                raise _UnsupportedRequest("OpenAI Chat provider-native message is invalid")
            _reject_unmodeled_fields(
                value,
                {
                    "annotations",
                    "audio",
                    "content",
                    "function_call",
                    "name",
                    "reasoning_content",
                    "refusal",
                    "role",
                    "tool_calls",
                },
                "OpenAI Chat provider-native assistant message",
            )
            if value.get("annotations") not in (None, []):
                raise _UnsupportedRequest("OpenAI Chat assistant annotations are not covered")
            if value.get("audio") is not None:
                raise _UnsupportedRequest("OpenAI Chat assistant audio is not covered")
            if value.get("function_call") is not None or value.get("name") is not None:
                raise _UnsupportedRequest("OpenAI Chat legacy or named assistant output is not covered")
            reasoning_content = value.get("reasoning_content")
            if reasoning_content is not None and not isinstance(reasoning_content, str):
                raise _UnsupportedRequest("OpenAI Chat assistant reasoning is invalid")
            if reasoning_content not in (None, "") and payload_policy.reasoning not in {
                ReasoningPolicy.FINAL_ANSWER_ONLY,
                ReasoningPolicy.CHECK_OUTPUT,
            }:
                raise _UnsupportedRequest("OpenAI Chat assistant reasoning is not covered")
            calls = value.get("tool_calls")
            if calls is not None and not isinstance(calls, list):
                raise _UnsupportedRequest("OpenAI Chat assistant tool calls are invalid")
            if calls and not allow_structural_tools:
                raise _UnsupportedRequest("assistant tool calls are not covered by input rails")
            content = value.get("content")
            refusal = value.get("refusal")
            if refusal is not None and not isinstance(refusal, str):
                raise _UnsupportedRequest("OpenAI Chat assistant refusal is invalid")
            if isinstance(refusal, str) and content not in (None, "", []):
                raise _UnsupportedRequest("OpenAI Chat assistant content and refusal cannot be ordered losslessly")
            if isinstance(refusal, str):
                projected.append({"role": "assistant", "content": refusal})
                continue
            if content is None:
                if calls and allow_structural_tools:
                    continue
                raise _UnsupportedRequest("OpenAI Chat assistant content is missing")
            projected.append(
                {
                    "role": "assistant",
                    "content": _project_text_content(
                        content,
                        "message content",
                        allowed_metadata,
                        codec_name=codec_name,
                        payload_policy=payload_policy,
                    ),
                }
            )
            continue
        if role == "provider_native" and codec_name == "openai_responses":
            value = message.get("value")
            if (
                payload_policy.reasoning in {ReasoningPolicy.FINAL_ANSWER_ONLY, ReasoningPolicy.CHECK_OUTPUT}
                and message.get("provider") == "openai_responses"
                and message.get("kind") == "reasoning"
                and set(message) == {"kind", "provider", "role", "value"}
                and validate_openai_responses_reasoning(value)
            ):
                continue
        if role == "provider_native" and codec_name == "openai_responses":
            native_role, text = _project_responses_native_message(message, payload_policy)
            projected.append({"role": "system" if native_role == "developer" else native_role, "content": text})
            if native_role == "user":
                latest_user_index = len(projected) - 1
                latest_user_text = text
            continue
        if allow_structural_tools and role == "tool_call":
            continue
        if allow_tool_results and role in ("tool", "tool_result"):
            continue
        if allow_tool_results and role == "provider_native":
            value = message.get("value")
            if isinstance(value, dict) and str(value.get("role", "")).lower() == "tool":
                continue
        if not isinstance(role, str) or role not in _TEXT_ROLES:
            raise _UnsupportedRequest("normalized request contains structural tool or provider-native traffic")
        if message.get("name") is not None:
            raise _UnsupportedRequest("named messages are not covered by the text projection")
        if not allow_structural_tools and role == "assistant" and message.get("tool_calls") not in (None, []):
            raise _UnsupportedRequest("assistant tool calls are not covered by input rails")
        if "content" not in message or message.get("content") is None:
            if allow_structural_tools and role == "assistant" and message.get("tool_calls") not in (None, []):
                continue
            raise _UnsupportedRequest("normalized message content is missing")

        text = _project_text_content(
            message["content"],
            "message content",
            allowed_metadata,
            codec_name=codec_name,
            payload_policy=payload_policy,
            allow_structural_tools=allow_structural_tools,
            allow_tool_results=allow_tool_results,
            allow_responses_refusal=codec_name == "openai_responses" and role == "assistant",
        )
        if allow_structural_tools and not text and isinstance(message["content"], list) and message["content"]:
            # A structural-only Anthropic content block is represented by the
            # tool adapter, not as an empty human or assistant utterance.
            continue
        projected.append({"role": "system" if role == "developer" else cast(str, role), "content": text})
        if role == "user":
            latest_user_index = len(projected) - 1
            latest_user_text = text

    if require_user and (latest_user_index is None or latest_user_text is None):
        raise _UnsupportedRequest("at least one user message is required")
    if require_user and latest_user_text is not None and not latest_user_text.strip():
        raise _UnsupportedRequest("the latest user message must contain non-whitespace text")
    return projected, latest_user_index, latest_user_text


def _project_annotated_request(
    annotated: AnnotatedLlmRequest,
    codec_name: str,
    *,
    payload_policy: PayloadPolicy | None = None,
    allow_structural_tools: bool = False,
    allow_tool_results: bool = False,
    require_user: bool = True,
) -> list[dict[str, str]]:
    """Project Relay's provider-neutral text annotation to Guardrails messages."""

    messages = annotated.get("messages")
    if not isinstance(messages, list):
        raise _UnsupportedRequest("normalized messages are missing")
    allowed_metadata = _TEXT_PART_METADATA_BY_CODEC.get(codec_name, frozenset())

    instructions = annotated.get("instructions")
    projected, latest_user_index, _ = _project_normalized_messages(
        messages,
        codec_name,
        allowed_metadata,
        payload_policy=payload_policy,
        allow_structural_tools=allow_structural_tools,
        allow_tool_results=allow_tool_results,
        require_user=require_user,
    )
    if instructions is not None:
        projected.insert(
            0,
            {
                "role": "system",
                "content": _project_text_content(
                    instructions,
                    "top-level instructions",
                    allowed_metadata,
                    codec_name=codec_name,
                    payload_policy=payload_policy,
                ),
            },
        )
        if latest_user_index is not None:
            latest_user_index += 1

    # Guardrails' explicit input check can consume assistant context after
    # the latest user message, but a later system message produces an invalid
    # MODIFIED result instead of checking that user turn. Reject rather than
    # misreporting that upstream behavior as a policy decision.
    if (
        require_user
        and latest_user_index is not None
        and any(message["role"] == "system" for message in projected[latest_user_index + 1 :])
    ):
        raise _UnsupportedRequest("system messages after the latest user message are not supported")
    return projected


_ANNOTATED_REQUEST_OPTIONAL_FIELDS = (
    "api_specific",
    "include",
    "instructions",
    "max_output_tokens",
    "max_tool_calls",
    "metadata",
    "model",
    "parallel_tool_calls",
    "params",
    "previous_response_id",
    "reasoning",
    "service_tier",
    "store",
    "stream",
    "tool_choice",
    "tools",
    "top_logprobs",
    "truncation",
    "user",
)
_ANNOTATED_REQUEST_MODELED_FIELDS = frozenset({"messages", *_ANNOTATED_REQUEST_OPTIONAL_FIELDS})


def _host_annotation_payload(annotated: AnnotatedLlmRequest) -> AnnotatedLlmRequest:
    """Undo serde's flattened ``extra`` field for an SDK JSON annotation."""

    messages = annotated.get("messages", [])
    payload: dict[str, Any] = {"messages": messages}
    payload.update({field: annotated.get(field) for field in _ANNOTATED_REQUEST_OPTIONAL_FIELDS})
    payload["extra"] = {key: value for key, value in annotated.items() if key not in _ANNOTATED_REQUEST_MODELED_FIELDS}
    return cast(AnnotatedLlmRequest, payload)


def _gemini_wire_hazards(
    content: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
    allow_structural_tools: bool,
    require_text_coverage: bool,
) -> None:
    """Check request fields that Relay's Gemini annotation intentionally drops."""

    if allow_structural_tools:
        _reject_structural_extensions(
            content,
            {"contents", "systemInstruction", "toolConfig", "tools"},
            "Gemini request",
        )
    containers = [content.get("systemInstruction")]
    raw_contents = content.get("contents")
    if isinstance(raw_contents, list):
        containers.extend(raw_contents)
    for container in containers:
        if container is None:
            continue
        if not isinstance(container, dict) or not isinstance(container.get("parts"), list):
            raise _UnsupportedRequest("Gemini request content is invalid")
        if allow_structural_tools:
            _reject_structural_extensions(container, {"parts"}, "Gemini request content")
        for part in container["parts"]:
            if not isinstance(part, dict):
                raise _UnsupportedRequest("Gemini request part is invalid")
            if "thought" in part and not isinstance(part["thought"], bool):
                raise _UnsupportedRequest("Gemini thought marker is invalid")
            if "thoughtSignature" in part and not isinstance(part["thoughtSignature"], str):
                raise _UnsupportedRequest("Gemini thought signature is invalid")
            if part.get("thought") is True and isinstance(part.get("text"), str) and require_text_coverage:
                if payload_policy.reasoning is ReasoningPolicy.REJECT:
                    raise _UnsupportedRequest("Gemini hidden thought text is not covered by input rails")
            data_fields = {
                "codeExecutionResult",
                "executableCode",
                "fileData",
                "functionCall",
                "functionResponse",
                "inlineData",
                "text",
            }.intersection(part)
            if len(data_fields) != 1:
                raise _UnsupportedRequest("Gemini request part must contain one data variant")
            if allow_structural_tools:
                _reject_structural_extensions(
                    part,
                    data_fields | {"partMetadata", "thought", "thoughtSignature", "videoMetadata"},
                    "Gemini request part",
                )
            if not allow_structural_tools:
                continue
            call = part.get("functionCall")
            if call is not None and (
                not isinstance(call, dict)
                or set(call) - {"args", "id", "name"}
                or not isinstance(call.get("name"), str)
                or not call["name"]
                or "args" in call
                and not isinstance(call["args"], dict)
            ):
                raise _UnsupportedRequest("Gemini function call is not completely covered")
            result = part.get("functionResponse")
            if result is not None and (
                not isinstance(result, dict)
                or set(result) - {"id", "name", "parts", "response"}
                or not isinstance(result.get("name"), str)
                or not result["name"]
                or not isinstance(result.get("response"), dict)
            ):
                raise _UnsupportedRequest("Gemini function result is not completely covered")
            if isinstance(result, dict) and result.get("parts") is not None:
                nested = result["parts"]
                if not isinstance(nested, list):
                    raise _UnsupportedRequest("Gemini function result parts are invalid")
                for nested_part in nested:
                    if not isinstance(nested_part, dict):
                        raise _UnsupportedRequest("Gemini function result part is invalid")
                    nested_fields = {
                        "codeExecutionResult",
                        "executableCode",
                        "fileData",
                        "functionCall",
                        "functionResponse",
                        "inlineData",
                        "text",
                    }.intersection(nested_part)
                    if len(nested_fields) != 1 or nested_fields.intersection({"functionCall", "functionResponse"}):
                        raise _UnsupportedRequest("Gemini function result contains nested tool traffic")
                    _reject_structural_extensions(
                        nested_part,
                        nested_fields | {"partMetadata", "thought", "thoughtSignature", "videoMetadata"},
                        "Gemini function result part",
                    )
                    if "text" in nested_fields and not isinstance(nested_part.get("text"), str):
                        raise _UnsupportedRequest("Gemini function result text is invalid")


def _openai_chat_wire_hazards(content: dict[str, Any], *, allow_structural_tools: bool) -> None:
    if not allow_structural_tools:
        return
    _reject_structural_extensions(
        content,
        {"messages", "parallel_tool_calls", "tool_choice", "tools"},
        "OpenAI Chat request",
    )
    messages = content.get("messages", [])
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if message.get("function_call") is not None:
            raise _UnsupportedRequest("legacy OpenAI function calls are not structurally covered")
        calls = message.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list) or calls and role != "assistant":
                raise _UnsupportedRequest("OpenAI Chat tool calls are invalid")
            for raw in calls:
                if (
                    not isinstance(raw, dict)
                    or set(raw) - {"function", "id", "type"}
                    or raw.get("type", "function") != "function"
                ):
                    raise _UnsupportedRequest("OpenAI Chat tool call is not a function call")
                function = raw.get("function")
                if (
                    not isinstance(function, dict)
                    or set(function) - {"arguments", "name"}
                    or "arguments" not in function
                ):
                    raise _UnsupportedRequest("OpenAI Chat function call is incomplete")
                tool_call(raw.get("id"), function.get("name"), function["arguments"])
        call_id = message.get("tool_call_id")
        if call_id is not None and (role != "tool" or not isinstance(call_id, str) or not call_id):
            raise _UnsupportedRequest("OpenAI Chat tool result identity is invalid")
        if role == "tool" and (not isinstance(call_id, str) or not call_id):
            raise _UnsupportedRequest("OpenAI Chat tool message is missing a tool_call_id")
        known = {"audio", "content", "function_call", "name", "refusal", "role", "tool_call_id", "tool_calls"}
        _reject_structural_extensions(message, known, "OpenAI Chat message")
        blocks = message.get("content")
        if isinstance(blocks, list):
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if isinstance(block_type, str) and _structural_key(block_type):
                    raise _UnsupportedRequest("OpenAI Chat content contains unsupported tool traffic")
                _reject_structural_extensions(block, {"type"}, "OpenAI Chat content block")


def _openai_responses_wire_hazards(content: dict[str, Any], *, allow_structural_tools: bool) -> None:
    if not allow_structural_tools:
        return
    _reject_structural_extensions(
        content,
        {"client_metadata", "input", "max_tool_calls", "parallel_tool_calls", "tool_choice", "tools"},
        "OpenAI Responses request",
    )
    items = content.get("input")
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            if set(item) - {"arguments", "call_id", "id", "name", "status", "type"} or "arguments" not in item:
                raise _UnsupportedRequest("OpenAI Responses function call is incomplete")
            tool_call(item.get("call_id"), item.get("name"), item["arguments"])
        elif item_type == "function_call_output":
            if (
                set(item) - {"call_id", "id", "output", "status", "type"}
                or not isinstance(item.get("call_id"), str)
                or not item["call_id"]
                or "output" not in item
            ):
                raise _UnsupportedRequest("OpenAI Responses function result is incomplete")
        elif item_type == "item_reference":
            raise _UnsupportedRequest("OpenAI Responses referenced history is not structurally covered")
        elif item_type in (None, "message") and isinstance(item.get("role"), str):
            _reject_structural_extensions(
                item,
                {"content", "id", "phase", "role", "status", "type"},
                "OpenAI Responses message",
            )
            blocks = item.get("content")
            if isinstance(blocks, list):
                for block in blocks:
                    if not isinstance(block, dict) or block.get("type") not in (
                        "input_file",
                        "input_image",
                        "input_text",
                        "output_text",
                        "refusal",
                    ):
                        raise _UnsupportedRequest("OpenAI Responses message content is not structurally covered")
                    allowed = {
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
                    }
                    _reject_structural_extensions(block, allowed, "OpenAI Responses content")
        elif isinstance(item_type, str) and _structural_key(item_type):
            raise _UnsupportedRequest("OpenAI Responses hosted tool traffic is not covered")


def _anthropic_wire_hazards(content: dict[str, Any], *, allow_structural_tools: bool) -> None:
    if not allow_structural_tools:
        return
    _reject_structural_extensions(content, {"messages", "tool_choice", "tools"}, "Anthropic request")
    messages = content.get("messages", [])
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if any(_structural_key(key) for key in set(message) - {"content", "role"}):
            raise _UnsupportedRequest("Anthropic message contains unsupported tool traffic")
        blocks = message.get("content")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_use":
                if role != "assistant" or set(block) - {
                    "cache_control",
                    "caller",
                    "id",
                    "input",
                    "name",
                    "toolset_name",
                    "type",
                }:
                    raise _UnsupportedRequest("Anthropic tool-use block is not completely covered")
                caller = block.get("caller")
                if caller is not None and caller != {"type": "direct"}:
                    raise _UnsupportedRequest("Anthropic hosted tool calls are not covered")
                if block.get("toolset_name") is not None:
                    raise _UnsupportedRequest("Anthropic toolset calls are not covered")
                tool_call(block.get("id"), block.get("name"), block.get("input"))
            elif block_type == "tool_result":
                if (
                    role != "user"
                    or set(block) - {"cache_control", "content", "is_error", "tool_use_id", "type"}
                    or not isinstance(block.get("tool_use_id"), str)
                    or not block["tool_use_id"]
                ):
                    raise _UnsupportedRequest("Anthropic tool-result block is not completely covered")
            elif isinstance(block_type, str) and _structural_key(block_type):
                raise _UnsupportedRequest("Anthropic hosted tool traffic is not covered")
            else:
                _reject_structural_extensions(block, {"type"}, "Anthropic content block")


def _oci_wire_hazards(
    content: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
    allow_structural_tools: bool,
    require_text_coverage: bool,
) -> None:
    """Check request fields omitted by Relay's OCI annotation."""

    chat = content.get("chatRequest", content)
    if not isinstance(chat, dict):
        raise _UnsupportedRequest("OCI chatRequest must be an object")
    variant = str(chat.get("apiFormat", "GENERIC")).upper()
    if variant not in oci_coverage.API_FORMATS:
        raise _UnsupportedRequest("OCI apiFormat is not supported")
    if require_text_coverage and chat.get("documents") not in (None, []):
        raise _UnsupportedRequest("OCI document context is not covered by input rails")
    if allow_structural_tools:
        _reject_structural_extensions(
            chat,
            {"chatHistory", "messages", "toolChoice", "tools"},
            "OCI request",
        )

    messages = chat.get("chatHistory" if variant == "COHERE" else "messages", [])
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        if variant == "COHERE" and any(_structural_key(key) for key in set(message) - {"message", "role"}):
            raise _UnsupportedRequest("OCI COHERE history contains unsupported tool state")
        _reject_structural_extensions(
            message,
            {"toolCallId", "toolCalls", "toolPlan"},
            "OCI request message",
        )
        if require_text_coverage:
            for field in {"annotations", "citations", "name", "reasoningContent", "refusal", "toolPlan"}:
                value = message.get(field)
                if value in (None, "", []):
                    continue
                if field in {"reasoningContent", "toolPlan"} and payload_policy.reasoning in {
                    ReasoningPolicy.FINAL_ANSWER_ONLY,
                    ReasoningPolicy.CHECK_OUTPUT,
                }:
                    continue
                raise _UnsupportedRequest("OCI message contains unchecked text metadata")
        if allow_structural_tools:
            role = str(message.get("role", "")).upper()
            calls = message.get("toolCalls")
            call_id = message.get("toolCallId")
            if calls not in (None, []) and role != "ASSISTANT":
                raise _UnsupportedRequest("OCI tool calls must be carried by an assistant message")
            if calls is not None:
                oci_coverage.parse_tool_calls(calls, variant)
            if call_id is not None and (role != "TOOL" or not isinstance(call_id, str) or not call_id):
                raise _UnsupportedRequest("OCI tool result identity is invalid")
        parts = message.get("content")
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    raise _UnsupportedRequest("OCI content part is invalid")
                if str(part.get("type", "TEXT")).upper() == "TEXT":
                    _reject_structural_extensions(part, {"text", "type"}, "OCI text part")
                elif _structural_key(str(part.get("type", ""))) or any(_structural_key(key) for key in part):
                    raise _UnsupportedRequest("OCI content contains unsupported tool traffic")


def _validate_wire_hazards(
    codec_name: str,
    content: dict[str, Any],
    *,
    payload_policy: PayloadPolicy,
    allow_structural_tools: bool,
    require_text_coverage: bool,
) -> None:
    if codec_name == "openai_chat":
        if content.get("prediction") is not None:
            raise _UnsupportedRequest("OpenAI prediction content is not covered by input rails")
        template = content.get("chat_template_kwargs")
        if template is not None and (
            not isinstance(template, dict)
            or set(template) != {"enable_thinking"}
            or not isinstance(template.get("enable_thinking"), bool)
        ):
            raise _UnsupportedRequest("OpenAI Chat template controls are not supported")
        _openai_chat_wire_hazards(content, allow_structural_tools=allow_structural_tools)
    elif codec_name == "openai_responses":
        _openai_responses_wire_hazards(content, allow_structural_tools=allow_structural_tools)
    elif codec_name == "anthropic_messages":
        _anthropic_wire_hazards(content, allow_structural_tools=allow_structural_tools)
    elif codec_name == "gemini_generate_content":
        _gemini_wire_hazards(
            content,
            payload_policy=payload_policy,
            allow_structural_tools=allow_structural_tools,
            require_text_coverage=require_text_coverage,
        )
    elif codec_name == "oci_genai":
        _oci_wire_hazards(
            content,
            payload_policy=payload_policy,
            allow_structural_tools=allow_structural_tools,
            require_text_coverage=require_text_coverage,
        )


@dataclass(frozen=True, slots=True)
class _DecodedToolRequest:
    """Tool state projected from Relay's authoritative request decode."""

    codec_name: str | None
    projection: ToolRequestProjection
    codec_variant: str | None = None


@dataclass(frozen=True, slots=True)
class _DecodedTextRequest:
    """Complete text context projected from Relay's authoritative decode."""

    codec_name: str | None
    messages: tuple[dict[str, str], ...]
    codec_variant: str | None = None


class _ProviderProjector:
    """Project Relay annotations and inspect policy-bearing provider output."""

    def __init__(self, payload_policy: PayloadPolicy | None = None) -> None:
        self._payload_policy = payload_policy or PayloadPolicy()

    @staticmethod
    def _request_codec_variant(codec_name: str, annotated: AnnotatedLlmRequest) -> str | None:
        """Read a provider dialect already decoded by Relay."""

        if codec_name != "oci_genai":
            return None
        api_specific = annotated.get("api_specific")
        variant = api_specific.get("api_format") if isinstance(api_specific, dict) else None
        if variant not in oci_coverage.API_FORMATS:
            raise _UnsupportedRequest("Relay did not identify the OCI request apiFormat")
        return cast(str, variant)

    def _host_request(
        self,
        request: LlmRequest,
        annotated_request: AnnotatedLlmRequest,
        *,
        codec_name: str | None,
        allow_structural_tools: bool,
        require_text_coverage: bool,
    ) -> tuple[str | None, AnnotatedLlmRequest, str | None]:
        """Validate the host decode and known provider wire hazards."""

        projection_name = codec_name or ""
        if codec_name is not None:
            if codec_name not in _SUPPORTED_CODECS:
                raise _UnsupportedRequest("Relay selected an unsupported request codec")
            if not isinstance(request, dict) or not isinstance(request.get("content"), dict):
                raise _UnsupportedRequest("provider request content is invalid")
            content = cast(dict[str, Any], request["content"])
            _validate_wire_hazards(
                codec_name,
                content,
                payload_policy=self._payload_policy,
                allow_structural_tools=allow_structural_tools,
                require_text_coverage=require_text_coverage,
            )
        annotated = _host_annotation_payload(annotated_request)
        _validate_annotation_state(
            annotated,
            projection_name,
            allow_structural_tools=allow_structural_tools,
            require_text_coverage=require_text_coverage,
        )
        codec_variant = self._request_codec_variant(codec_name, annotated) if codec_name is not None else None
        return codec_name, annotated, codec_variant

    def project_host_text(
        self,
        request: LlmRequest,
        annotated_request: AnnotatedLlmRequest,
        *,
        allow_structural_tools: bool = False,
        allow_tool_results: bool = False,
        require_user: bool,
        codec_name: str | None,
    ) -> _DecodedTextRequest:
        """Project the active host codec's normalized request into Guardrails text."""

        codec_name, annotated, codec_variant = self._host_request(
            request,
            annotated_request,
            codec_name=codec_name,
            allow_structural_tools=allow_structural_tools,
            require_text_coverage=True,
        )
        messages = _project_annotated_request(
            annotated,
            codec_name or "",
            payload_policy=self._payload_policy,
            allow_structural_tools=allow_structural_tools,
            allow_tool_results=allow_tool_results,
            require_user=require_user,
        )
        return _DecodedTextRequest(codec_name, tuple(messages), codec_variant)

    def project_host_tools(
        self,
        request: LlmRequest,
        annotated_request: AnnotatedLlmRequest,
        *,
        require_definitions: bool = True,
        codec_name: str | None,
    ) -> _DecodedToolRequest:
        """Project structural tool state from the active host codec decode."""

        codec_name, annotated, codec_variant = self._host_request(
            request,
            annotated_request,
            codec_name=codec_name,
            allow_structural_tools=True,
            require_text_coverage=False,
        )
        try:
            projection = project_request_tools(
                annotated,
                codec_name=codec_name or "",
                codec_variant=codec_variant,
                include_definitions=require_definitions,
            )
        except ToolProjectionError as exc:
            raise _UnsupportedRequest("provider tool traffic is not completely covered") from exc
        return _DecodedToolRequest(codec_name, projection, codec_variant)

    def inspect_response(
        self,
        request: _DecodedTextRequest | _DecodedToolRequest,
        response: Json,
        *,
        response_codec_name: str,
        require_text_coverage: bool = True,
    ) -> ResponseInspection:
        """Inspect all policy-bearing output with the selected provider shape."""

        if not isinstance(response, dict):
            raise _UnsupportedRequest("provider response is not an object")
        codec_variant = self._response_codec_variant(request, response, response_codec_name)
        payload_policy = (
            self._payload_policy
            if require_text_coverage
            else PayloadPolicy(
                multimodal=MultimodalPolicy.TEXT_ONLY,
                reasoning=ReasoningPolicy.FINAL_ANSWER_ONLY,
            )
        )
        inspection = inspect_provider_response(
            response_codec_name,
            codec_variant,
            response,
            payload_policy,
        )
        if len(inspection.candidates) + len(inspection.reasoning) > MAX_RESPONSE_SEGMENTS:
            raise _UnsupportedRequest("provider response has too many independently checked segments")
        return inspection

    def project_response_texts(
        self,
        request: _DecodedTextRequest,
        response: Json,
        *,
        allow_tool_calls: bool,
        response_codec_name: str,
    ) -> ProjectedOutputText:
        """Resolve one response codec and project every selectable candidate."""

        inspection = self.inspect_response(request, response, response_codec_name=response_codec_name)
        if inspection.has_calls and not allow_tool_calls:
            raise _UnsupportedRequest("provider response contains unchecked function calls")
        if not inspection.texts and not inspection.reasoning and not inspection.has_calls:
            raise _UnsupportedRequest("provider response has no completely covered visible text")
        return ProjectedOutputText(
            inspection.texts,
            response_codec_name,
            inspection.codec_variant,
            inspection.reasoning,
        )

    def project_response_tool_candidates(
        self,
        request: _DecodedToolRequest,
        response: Json,
        *,
        response_codec_name: str,
    ) -> tuple[tuple[ToolCall, ...], ...]:
        """Decode every selectable candidate and prove function-call coverage."""

        return self.inspect_response(
            request,
            response,
            response_codec_name=response_codec_name,
            require_text_coverage=False,
        ).tool_candidates

    @staticmethod
    def _response_codec_variant(
        request: _DecodedTextRequest | _DecodedToolRequest,
        response: dict[str, Any],
        response_codec_name: str,
    ) -> str | None:
        """Resolve the response dialect for the host-selected codec."""

        if response_codec_name not in _SUPPORTED_CODECS:
            raise _UnsupportedRequest("Relay selected an unsupported response codec")
        if request.codec_name == response_codec_name:
            return request.codec_variant
        if response_codec_name == "oci_genai":
            return oci_coverage.response_variant(response)
        return None
