# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless provider-aware request and response projection for Relay policies.

This module owns codec discrimination, raw provider-shape validation, and the
normalized text/tool projections used by the Guardrails execution policy.
Keeping that boundary separate lets the worker focus on lifecycle, policy
ordering, and failure handling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from nemo_relay import LLMRequest as NativeLlmRequest
from nemo_relay.codecs import (
    AnthropicMessagesCodec,
    GeminiGenerateContentCodec,
    OCIGenAIChatCodec,
    OpenAIChatCodec,
    OpenAIResponsesCodec,
)
from nemo_relay_plugin import AnnotatedLlmRequest, Json, LlmRequest

from .codec_adapters import anthropic as anthropic_adapter
from .codec_adapters import gemini as gemini_adapter
from .codec_adapters import oci as oci_adapter
from .codec_adapters import openai_chat as openai_chat_adapter
from .codec_adapters import openai_responses as openai_responses_adapter
from .codec_adapters.common import (
    _reject_nonempty_metadata,
    _reject_unmodeled_fields,
    _reject_unmodeled_structural_data,
    _structural_key,
    _UnsupportedRequest,
)
from .payload_policy import (
    MultimodalPolicy,
    PayloadPolicy,
    ProjectedOutputText,
    ReasoningPolicy,
    is_supported_normalized_modality,
    validate_anthropic_reasoning,
    validate_openai_responses_reasoning,
)
from .structural_tools import ToolCall
from .tool_projection import (
    ToolProjectionError,
    ToolRequestProjection,
    project_request_tools,
    project_response_calls,
)

logger = logging.getLogger(__name__)

_PROVIDER_ADAPTERS = {
    "openai_chat": openai_chat_adapter,
    "openai_responses": openai_responses_adapter,
    "anthropic_messages": anthropic_adapter,
    "gemini_generate_content": gemini_adapter,
    "oci_genai": oci_adapter,
}

# Provider request discriminators and exact response shapes whose complete
# text/tool coverage is verified below. Unknown provider-visible fields fail
# closed rather than being silently omitted from a Guardrails decision.
_TEXT_ROLES = frozenset({"system", "user", "assistant", "developer"})

# Each projected segment may require a separate Guardrails evaluation. Keep
# this provider-independent bound close to the projection code so both text
# and structural paths reject excessive candidate fan-out before policy work.
MAX_RESPONSE_SEGMENTS = 32

_TEXT_PART_METADATA_BY_CODEC = {
    "anthropic_messages": anthropic_adapter.TEXT_PART_METADATA,
    "gemini_generate_content": gemini_adapter.TEXT_PART_METADATA,
    "openai_responses": openai_responses_adapter.TEXT_PART_METADATA,
}

_EXTRA_CONTROL_KEYS_BY_CODEC = {
    "gemini_generate_content": gemini_adapter.EXTRA_CONTROL_KEYS,
    "openai_chat": openai_chat_adapter.EXTRA_CONTROL_KEYS,
    "openai_responses": openai_responses_adapter.EXTRA_CONTROL_KEYS,
}

_STRUCTURAL_CONTROL_KEYS_BY_CODEC = {
    "gemini_generate_content": gemini_adapter.STRUCTURAL_CONTROL_KEYS,
    "openai_chat": openai_chat_adapter.STRUCTURAL_CONTROL_KEYS,
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
                and part.get("kind") in {"thinking", "redacted_thinking"}
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
    require_text_coverage: bool = True,
) -> None:
    extra = annotated.get("extra", {})
    if not isinstance(extra, dict):
        raise _UnsupportedRequest("normalized provider extras are invalid")
    if codec_name == "gemini_generate_content" and extra.get("cachedContent") is not None:
        raise _UnsupportedRequest("the request refers to provider-side cached content")
    # ``extra`` is the native codec's lossless bucket for fields it does not
    # understand. Even if a key looks harmless today, a custom or future
    # provider codec could treat it as prompt content. Strict input coverage
    # therefore requires the bucket to be empty.
    if require_text_coverage:
        allowed_extra = _EXTRA_CONTROL_KEYS_BY_CODEC.get(codec_name, frozenset())
        if set(extra) - allowed_extra:
            raise _UnsupportedRequest("the provider request contains an unmodeled field")
    else:
        allowed_controls = _STRUCTURAL_CONTROL_KEYS_BY_CODEC.get(codec_name, frozenset())
        for key, value in extra.items():
            if key in allowed_controls:
                continue
            if _structural_key(key):
                raise _UnsupportedRequest("the provider request contains unmodeled tool state")
            _reject_unmodeled_structural_data(value)
    if annotated.get("previous_response_id") is not None:
        raise _UnsupportedRequest("the request refers to provider-side conversation state")

    api_specific = annotated.get("api_specific")
    if api_specific is None:
        return
    if not isinstance(api_specific, dict):
        raise _UnsupportedRequest("normalized provider fields are invalid")
    if api_specific.get("api") == "openai_responses" and (
        any(api_specific.get(key) is not None for key in {"conversation", "prompt"})
        or api_specific.get("context_management") not in (None, [])
    ):
        raise _UnsupportedRequest("the request refers to provider-side prompt or conversation state")
    if api_specific.get("api") == "anthropic_messages" and api_specific.get("container") is not None:
        raise _UnsupportedRequest("the request refers to provider-side container state")


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
            _reject_unmodeled_fields(
                message,
                {"kind", "provider", "role", "value"},
                "OpenAI Responses provider-native message",
            )
            value = message.get("value")
            if (
                message.get("provider") != "openai_responses"
                or message.get("kind") != "message"
                or not isinstance(value, dict)
                or value.get("role") != "assistant"
                or value.get("type") not in (None, "message")
                or value.get("phase") not in (None, *openai_responses_adapter.MESSAGE_PHASES)
                or ("id" in value and not isinstance(value["id"], str))
                or value.get("status") not in (None, "completed", "in_progress", "incomplete")
            ):
                raise _UnsupportedRequest("OpenAI Responses provider-native message is invalid")
            _reject_unmodeled_fields(
                value,
                {"content", "id", "phase", "role", "status", "type"},
                "OpenAI Responses provider-native assistant message",
            )
            content = value.get("content")
            if not isinstance(content, list):
                raise _UnsupportedRequest("OpenAI Responses assistant content is invalid")
            segments: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    raise _UnsupportedRequest("OpenAI Responses assistant content is invalid")
                if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                    _reject_unmodeled_fields(
                        block,
                        {"annotations", "logprobs", "text", "type"},
                        "OpenAI Responses assistant output-text block",
                    )
                    _reject_nonempty_metadata(
                        block,
                        {"annotations", "logprobs"},
                        "OpenAI Responses assistant output-text block",
                    )
                    segments.append(cast(str, block["text"]))
                elif block.get("type") == "refusal" and isinstance(block.get("refusal"), str):
                    _reject_unmodeled_fields(
                        block,
                        {"refusal", "type"},
                        "OpenAI Responses assistant refusal block",
                    )
                    segments.append(cast(str, block["refusal"]))
                else:
                    raise _UnsupportedRequest("OpenAI Responses assistant content is not covered")
            projected.append({"role": "assistant", "content": "".join(segments)})
            continue
        if allow_structural_tools and role == "tool_call":
            continue
        if allow_tool_results and role in {"tool", "tool_result"}:
            continue
        if allow_tool_results and role == "provider_native":
            value = message.get("value")
            if isinstance(value, dict) and str(value.get("role", "")).lower() == "tool":
                continue
        if role not in _TEXT_ROLES:
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

    if not isinstance(annotated, dict):
        raise _UnsupportedRequest("Relay did not provide a normalized request")
    _validate_annotation_state(annotated, codec_name)
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


def _native_annotation_payload(annotated: Any) -> AnnotatedLlmRequest:
    """Convert a native codec annotation to one stable JSON-shaped view."""

    payload: dict[str, Any] = {"messages": annotated.messages}
    payload.update({field: getattr(annotated, field) for field in _ANNOTATED_REQUEST_OPTIONAL_FIELDS})
    payload["extra"] = annotated.extra
    return cast(AnnotatedLlmRequest, payload)


def _host_annotation_payload(annotated: AnnotatedLlmRequest) -> AnnotatedLlmRequest:
    """Undo serde's flattened ``extra`` field for an SDK JSON annotation."""

    messages = annotated.get("messages", [])
    payload: dict[str, Any] = {"messages": messages}
    payload.update({field: annotated.get(field) for field in _ANNOTATED_REQUEST_OPTIONAL_FIELDS})
    payload["extra"] = {key: value for key, value in annotated.items() if key not in _ANNOTATED_REQUEST_MODELED_FIELDS}
    return cast(AnnotatedLlmRequest, payload)


def _validate_raw_request_tool_shapes(codec_name: str, content: dict[str, Any]) -> None:
    """Reject required structural fields that a native codec can default."""

    if codec_name == "openai_chat":
        openai_chat_adapter.validate_request_tool_shapes(content)
    elif codec_name == "openai_responses":
        openai_responses_adapter.validate_request_tool_shapes(content)
    elif codec_name == "anthropic_messages":
        anthropic_adapter.validate_request_tool_shapes(content)


def _structural_codec_candidates(content: dict[str, Any]) -> tuple[str, ...]:
    families = tuple(
        name
        for name, present in (
            ("openai_responses", "input" in content or "instructions" in content),
            ("gemini_generate_content", "contents" in content or "systemInstruction" in content),
            ("oci_genai", "chatRequest" in content or "apiFormat" in content),
        )
        if present
    )
    if len(families) > 1:
        raise _UnsupportedRequest("request mixes multiple provider protocol shapes")
    return families


def _tool_definition_protocol_markers(content: dict[str, Any]) -> tuple[bool, bool]:
    tools = content.get("tools")
    if not isinstance(tools, list):
        return False, False
    openai = any(isinstance(tool, dict) and isinstance(tool.get("function"), dict) for tool in tools)
    anthropic = any(
        isinstance(tool, dict)
        and isinstance(tool.get("name"), str)
        and ("input_schema" in tool or tool.get("type") == "custom")
        for tool in tools
    )
    return openai, anthropic


def _message_codec_candidates(
    content: dict[str, Any],
    headers: dict[Any, Any],
    messages: list[Any],
) -> tuple[str, ...]:
    roles = {message.get("role") for message in messages if isinstance(message, dict)}
    if roles.intersection(oci_adapter.UPPERCASE_ROLES):
        if any(isinstance(role, str) and role.lower() == role for role in roles):
            raise _UnsupportedRequest("request mixes provider role conventions")
        return ("oci_genai",)

    chat_markers = set(content).intersection(openai_chat_adapter.ONLY_KEYS)
    anthropic_markers = set(content).intersection(anthropic_adapter.ONLY_KEYS)
    if "anthropic-version" in {str(name).lower() for name in headers}:
        anthropic_markers.add("anthropic-version")
    if anthropic_adapter.has_cache_control(messages):
        anthropic_markers.add("message.content.cache_control")
    if anthropic_adapter.has_tool_blocks(messages):
        anthropic_markers.add("message.content.tool_block")
    openai_tools, anthropic_tools = _tool_definition_protocol_markers(content)
    if openai_tools:
        chat_markers.add("tools.function")
    if anthropic_tools:
        anthropic_markers.add("tools.input_schema")
    if chat_markers and anthropic_markers:
        raise _UnsupportedRequest("request mixes OpenAI Chat and Anthropic Messages fields")
    if oci_adapter.has_text_parts(messages):
        if chat_markers or anthropic_markers:
            raise _UnsupportedRequest("request mixes OCI and another provider protocol shape")
        return ("oci_genai",)
    if chat_markers or openai_chat_adapter.has_message_marker(messages):
        return ("openai_chat",)
    if anthropic_markers:
        return ("anthropic_messages",)
    return ("openai_chat", "anthropic_messages")


@dataclass(frozen=True, slots=True)
class _DecodedToolRequest:
    """A request decoded by every plausible Relay codec."""

    codec_names: tuple[str, ...]
    projection: ToolRequestProjection
    codec_variants: tuple[str | None, ...] = ()


@dataclass(frozen=True, slots=True)
class _DecodedTextRequest:
    """Complete text context decoded by every plausible Relay codec."""

    codec_names: tuple[str, ...]
    messages: tuple[dict[str, str], ...]
    codec_variants: tuple[str | None, ...] = ()


class _NativeCodecProjector:
    """Reuse Relay's native codecs without trusting best-effort auto-detection."""

    def __init__(self, payload_policy: PayloadPolicy | None = None) -> None:
        self._payload_policy = payload_policy or PayloadPolicy()
        self._codecs = {
            "openai_chat": OpenAIChatCodec(),
            "openai_responses": OpenAIResponsesCodec(),
            "anthropic_messages": AnthropicMessagesCodec(),
            "oci_genai": OCIGenAIChatCodec(),
            "gemini_generate_content": GeminiGenerateContentCodec(),
        }

    @staticmethod
    def _request_codec_variant(codec_name: str, content: dict[str, Any]) -> str | None:
        """Retain request discriminators that one Relay codec handles as variants."""

        if codec_name != "oci_genai":
            return None
        return oci_adapter.request_variant(content)

    @staticmethod
    def _validate_response_variant(
        codec_name: str,
        codec_variant: str | None,
        response: dict[str, Any],
    ) -> None:
        """Require provider responses to match the request's exact codec dialect."""

        if codec_name != "oci_genai":
            return
        oci_adapter.validate_response_variant(codec_variant, response)

    @staticmethod
    def _candidate_codecs(request: LlmRequest) -> tuple[str, ...]:
        if not isinstance(request, dict):
            raise _UnsupportedRequest("request is not an object")
        content = request.get("content")
        headers = request.get("headers", {})
        if not isinstance(content, dict) or not isinstance(headers, dict):
            raise _UnsupportedRequest("request content or headers are invalid")

        structural_families = _structural_codec_candidates(content)
        if structural_families:
            return structural_families

        messages = content.get("messages")
        if not isinstance(messages, list):
            raise _UnsupportedRequest("request has no recognized provider conversation field")
        return _message_codec_candidates(content, headers, messages)

    @staticmethod
    def _validate_raw_coverage(
        codec_name: str,
        content: dict[str, Any],
        *,
        payload_policy: PayloadPolicy,
        allow_structural_tools: bool,
        require_text_coverage: bool,
    ) -> None:
        """Reject provider state that a native annotation intentionally omits."""

        if allow_structural_tools:
            _validate_raw_request_tool_shapes(codec_name, content)

        if codec_name == "openai_chat":
            openai_chat_adapter.validate_raw_coverage(content)
        elif codec_name == "openai_responses":
            openai_responses_adapter.validate_raw_coverage(content)

        if codec_name == "oci_genai":
            oci_adapter.validate_raw_coverage(
                content,
                payload_policy=payload_policy,
                allow_structural_tools=allow_structural_tools,
                require_text_coverage=require_text_coverage,
            )

        if codec_name == "gemini_generate_content":
            gemini_adapter.validate_raw_coverage(
                content,
                payload_policy=payload_policy,
                allow_structural_tools=allow_structural_tools,
                require_text_coverage=require_text_coverage,
            )

    def _decode_request(
        self,
        request: LlmRequest,
        *,
        allow_structural_tools: bool,
        require_text_coverage: bool,
        codec_name: str | None = None,
        annotated_request: AnnotatedLlmRequest | None = None,
    ) -> list[tuple[str, AnnotatedLlmRequest]]:
        if codec_name is not None and codec_name not in self._codecs:
            raise _UnsupportedRequest("Relay selected an unsupported request codec")
        candidates = (codec_name,) if codec_name is not None else self._candidate_codecs(request)
        decoded: list[tuple[str, AnnotatedLlmRequest]] = []
        for codec_name in candidates:
            try:
                self._validate_raw_coverage(
                    codec_name,
                    request["content"],
                    payload_policy=self._payload_policy,
                    allow_structural_tools=allow_structural_tools,
                    require_text_coverage=require_text_coverage,
                )
                native_content = request["content"]
                if codec_name == "anthropic_messages" and allow_structural_tools:
                    native_content = anthropic_adapter.codec_content(native_content)
                elif codec_name == "gemini_generate_content" and allow_structural_tools:
                    native_content = gemini_adapter.codec_content(native_content)
                native_request = NativeLlmRequest(request.get("headers", {}), native_content)
                try:
                    annotated = self._codecs[codec_name].decode(native_request)
                    payload = _native_annotation_payload(annotated)
                except Exception as exc:
                    # Relay's native codec objects are an external boundary.
                    # Normalize their failures, but let bugs in our own
                    # projection and coverage code remain visible.
                    raise _UnsupportedRequest("the provider codec could not decode the request") from exc
                _validate_annotation_state(
                    payload,
                    codec_name,
                    require_text_coverage=require_text_coverage,
                )
                if annotated_request is not None:
                    host_payload = _host_annotation_payload(annotated_request)
                    _validate_annotation_state(
                        host_payload,
                        codec_name,
                        require_text_coverage=require_text_coverage,
                    )
                    if host_payload != payload:
                        raise _UnsupportedRequest("Relay's annotated request disagrees with its active codec")
                    payload = host_payload
                decoded.append((codec_name, payload))
            except _UnsupportedRequest as exc:
                # Every selected codec is plausible at this point. Do not let
                # one incomplete decode disappear merely because another
                # ambiguous decoder happened to succeed.
                raise _UnsupportedRequest("a plausible provider codec could not verify the request") from exc
        if not decoded:
            raise _UnsupportedRequest("request has no supported provider codec")
        return decoded

    def project(
        self,
        request: LlmRequest,
        *,
        allow_structural_tools: bool = False,
        allow_tool_results: bool = False,
        codec_name: str | None = None,
        annotated_request: AnnotatedLlmRequest | None = None,
    ) -> list[dict[str, str]]:
        """Decode with bounded candidates and require ambiguous decoders to agree."""

        return list(
            self.project_text(
                request,
                allow_structural_tools=allow_structural_tools,
                allow_tool_results=allow_tool_results,
                require_user=True,
                codec_name=codec_name,
                annotated_request=annotated_request,
            ).messages
        )

    def project_text(
        self,
        request: LlmRequest,
        *,
        allow_structural_tools: bool = False,
        allow_tool_results: bool = False,
        require_user: bool,
        codec_name: str | None = None,
        annotated_request: AnnotatedLlmRequest | None = None,
    ) -> _DecodedTextRequest:
        """Return a complete text context and every agreeing codec candidate."""

        projections: list[list[dict[str, str]]] = []
        decoded = self._decode_request(
            request,
            allow_structural_tools=allow_structural_tools,
            require_text_coverage=True,
            codec_name=codec_name,
            annotated_request=annotated_request,
        )
        for codec_name, annotated in decoded:
            projections.append(
                _project_annotated_request(
                    annotated,
                    codec_name,
                    payload_policy=self._payload_policy,
                    allow_structural_tools=allow_structural_tools,
                    allow_tool_results=allow_tool_results,
                    require_user=require_user,
                )
            )

        if not projections or any(projection != projections[0] for projection in projections[1:]):
            raise _UnsupportedRequest("ambiguous provider codecs produced different text coverage")
        return _DecodedTextRequest(
            codec_names=tuple(codec_name for codec_name, _ in decoded),
            messages=tuple(projections[0]),
            codec_variants=tuple(
                self._request_codec_variant(codec_name, request["content"]) for codec_name, _ in decoded
            ),
        )

    def project_tools(
        self,
        request: LlmRequest,
        *,
        require_definitions: bool = True,
        codec_name: str | None = None,
        annotated_request: AnnotatedLlmRequest | None = None,
    ) -> _DecodedToolRequest:
        """Normalize tool state and require ambiguous codecs to agree."""

        decoded = self._decode_request(
            request,
            allow_structural_tools=True,
            require_text_coverage=False,
            codec_name=codec_name,
            annotated_request=annotated_request,
        )
        codec_variants = tuple(self._request_codec_variant(codec_name, request["content"]) for codec_name, _ in decoded)
        try:
            projections = [
                project_request_tools(
                    annotated,
                    codec_name=codec_name,
                    codec_variant=codec_variant,
                    include_definitions=require_definitions,
                )
                for (codec_name, annotated), codec_variant in zip(decoded, codec_variants, strict=True)
            ]
        except ToolProjectionError as exc:
            raise _UnsupportedRequest("provider tool traffic is not completely covered") from exc
        if not projections or any(projection != projections[0] for projection in projections[1:]):
            raise _UnsupportedRequest("ambiguous provider codecs produced different tool coverage")
        return _DecodedToolRequest(
            codec_names=tuple(codec_name for codec_name, _ in decoded),
            projection=projections[0],
            codec_variants=codec_variants,
        )

    @staticmethod
    def _validate_response_annotation(codec_name: str, annotated: Any) -> dict[str, Any]:
        payload = {
            "message": annotated.message,
            "tool_calls": annotated.tool_calls,
            "api_specific": annotated.api_specific,
            "extra": annotated.extra,
        }
        # Response codecs deliberately retain normal wire metadata such as
        # OpenAI's ``object`` and ``created`` in ``extra``. Structural tool
        # coverage comes from the provider-specific raw checks below, not from
        # requiring unrelated response metadata to disappear.
        if payload["extra"] is not None and not isinstance(payload["extra"], dict):
            raise _UnsupportedRequest("provider response metadata is invalid")
        api_specific = payload["api_specific"]
        actual_api = api_specific.get("api") if isinstance(api_specific, dict) else None
        if codec_name == "gemini_generate_content":
            if actual_api not in {None, codec_name}:
                raise _UnsupportedRequest("provider response does not match the request codec")
        elif actual_api != codec_name:
            raise _UnsupportedRequest("provider response does not match the request codec")
        return payload

    def _decode_response_annotation(
        self,
        codec_name: str,
        response: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """Decode one response at the native-codec boundary."""

        try:
            annotated = self._codecs[codec_name].decode_response(response)
            payload = self._validate_response_annotation(codec_name, annotated)
        except _UnsupportedRequest:
            raise
        except Exception as exc:
            raise _UnsupportedRequest("the provider codec could not decode the response") from exc
        return annotated, payload

    @staticmethod
    def _normalized_response_text(annotated: Any) -> str | None:
        """Read Relay's normalized response text without hiding local bugs."""

        try:
            return cast(str | None, annotated.response_text())
        except Exception as exc:
            raise _UnsupportedRequest("the provider codec could not expose response text") from exc

    @staticmethod
    def _raw_response_call_count(codec_name: str, response: dict[str, Any]) -> int:
        adapter = _PROVIDER_ADAPTERS.get(codec_name)
        if adapter is None:
            raise _UnsupportedRequest("response codec is not supported")
        return adapter.raw_response_call_count(response)

    def _raw_response_texts(
        self,
        codec_name: str,
        response: dict[str, Any],
        *,
        allow_tool_calls: bool,
    ) -> ProjectedOutputText:
        """Project every independently selectable visible-text candidate."""

        call_count = self._raw_response_call_count(codec_name, response)
        if call_count and not allow_tool_calls:
            raise _UnsupportedRequest("provider response contains unchecked function calls")

        adapter = _PROVIDER_ADAPTERS.get(codec_name)
        if adapter is None:
            raise _UnsupportedRequest("response codec is not supported")
        raw_text = adapter.raw_response_texts(
            response,
            payload_policy=self._payload_policy,
        )

        if raw_text.candidates:
            return ProjectedOutputText(raw_text.candidates, reasoning=raw_text.reasoning)
        if raw_text.fragments:
            return ProjectedOutputText(
                (raw_text.fragment_separator.join(raw_text.fragments),),
                reasoning=raw_text.reasoning,
            )
        if raw_text.reasoning:
            return ProjectedOutputText((), reasoning=raw_text.reasoning)
        if call_count and allow_tool_calls:
            return ProjectedOutputText(())
        raise _UnsupportedRequest("provider response has no completely covered visible text")

    @staticmethod
    def _expected_native_response_text(
        codec_name: str,
        response: dict[str, Any],
        projected_text: str | None,
    ) -> str | None:
        """Mirror the text subset exposed by Relay's native response codec."""

        adapter = _PROVIDER_ADAPTERS.get(codec_name)
        if adapter is None:
            raise _UnsupportedRequest("response codec is not supported")
        return adapter.expected_native_response_text(response, projected_text)

    def project_response_texts(
        self,
        request: _DecodedTextRequest,
        response: Json,
        *,
        allow_tool_calls: bool,
        response_codec_name: str | None = None,
    ) -> ProjectedOutputText:
        """Resolve one response codec and project every selectable candidate."""

        if not isinstance(response, dict):
            raise _UnsupportedRequest("provider response is not an object")
        projections: list[ProjectedOutputText] = []
        candidates = self._response_codec_candidates(request, response, response_codec_name)
        for codec_name, codec_variant in candidates:
            try:
                self._validate_response_variant(codec_name, codec_variant, response)
                annotated, _ = self._decode_response_annotation(codec_name, response)
                projected = self._raw_response_texts(
                    codec_name,
                    response,
                    allow_tool_calls=allow_tool_calls,
                )
                if len(projected.candidates) + len(projected.reasoning) > MAX_RESPONSE_SEGMENTS:
                    raise _UnsupportedRequest("provider response has too many independently checked segments")
                normalized_text = self._normalized_response_text(annotated)
                first_text = projected.candidates[0] if projected.candidates else None
                expected_native_text = self._expected_native_response_text(codec_name, response, first_text)
                # Relay intentionally omits known refusal fields from some
                # normalized responses. Compare against that exact subset
                # while still sending every ordered visible segment to rails.
                if normalized_text != expected_native_text:
                    raise _UnsupportedRequest("provider response text coverage disagrees with its codec")
                projections.append(
                    ProjectedOutputText(
                        projected.candidates,
                        codec_name,
                        codec_variant,
                        projected.reasoning,
                    )
                )
            except _UnsupportedRequest as error:
                # Ambiguous plain-message requests are resolved by the actual
                # provider response envelope. Every surviving decode is strict.
                # Keep the diagnostic content-free: provider errors may embed
                # request or response values in their messages.
                logger.debug(
                    "response text projection candidate rejected: codec=%s error_type=%s",
                    codec_name,
                    type(error).__name__,
                )
                continue
        if len(projections) != 1:
            if projections:
                raise _UnsupportedRequest("provider response matches more than one supported codec")
            raise _UnsupportedRequest("provider response could not be completely decoded")
        return projections[0]

    def project_response_text(
        self,
        request: _DecodedTextRequest,
        response: Json,
        *,
        allow_tool_calls: bool,
        response_codec_name: str | None = None,
    ) -> str | None:
        """Compatibility helper for callers that require one output candidate."""

        projection = self.project_response_texts(
            request,
            response,
            allow_tool_calls=allow_tool_calls,
            response_codec_name=response_codec_name,
        )
        try:
            return projection.one()
        except ValueError:
            raise _UnsupportedRequest("provider response contains multiple text candidates") from None

    @staticmethod
    def _response_candidate_payloads(codec_name: str, response: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        """Split selectable provider candidates without changing their payloads."""

        adapter = _PROVIDER_ADAPTERS.get(codec_name)
        if adapter is None:
            raise _UnsupportedRequest("response codec is not supported")
        return adapter.response_candidate_payloads(response, MAX_RESPONSE_SEGMENTS)

    def project_response_tool_candidates(
        self,
        request: _DecodedToolRequest,
        response: Json,
        *,
        response_codec_name: str | None = None,
    ) -> tuple[tuple[ToolCall, ...], ...]:
        """Decode every selectable candidate and prove function-call coverage."""

        if not isinstance(response, dict):
            raise _UnsupportedRequest("provider response is not an object")
        projections: list[tuple[tuple[ToolCall, ...], ...]] = []
        candidates = self._response_codec_candidates(request, response, response_codec_name)
        for codec_name, codec_variant in candidates:
            try:
                self._validate_response_variant(codec_name, codec_variant, response)
                raw_count = self._raw_response_call_count(codec_name, response)
                candidate_projections: list[tuple[ToolCall, ...]] = []
                for candidate_response in self._response_candidate_payloads(codec_name, response):
                    _, payload = self._decode_response_annotation(codec_name, candidate_response)
                    try:
                        projected = project_response_calls(payload)
                    except ToolProjectionError as exc:
                        raise _UnsupportedRequest("provider response tool-call coverage is incomplete") from exc
                    if self._raw_response_call_count(codec_name, candidate_response) != len(projected):
                        raise _UnsupportedRequest("provider response tool-call coverage is incomplete")
                    candidate_projections.append(projected)
                if raw_count != sum(map(len, candidate_projections)):
                    raise _UnsupportedRequest("provider response tool-call coverage is incomplete")
                projections.append(tuple(candidate_projections))
            except _UnsupportedRequest as error:
                # A plain messages request can be valid for both OpenAI Chat
                # and Anthropic. Their response envelopes disambiguate the
                # actual provider, so keep only complete decodes here.
                logger.debug(
                    "response tool projection candidate rejected: codec=%s error_type=%s",
                    codec_name,
                    type(error).__name__,
                )
                continue
        if not projections:
            raise _UnsupportedRequest("provider response could not be completely decoded")
        if len(projections) != 1:
            raise _UnsupportedRequest("provider response matches more than one supported codec")
        return projections[0]

    def project_response_tools(
        self,
        request: _DecodedToolRequest,
        response: Json,
        *,
        response_codec_name: str | None = None,
    ) -> tuple[ToolCall, ...]:
        """Compatibility helper for a response with at most one candidate."""

        projections = self.project_response_tool_candidates(
            request,
            response,
            response_codec_name=response_codec_name,
        )
        if len(projections) > 1:
            raise _UnsupportedRequest("provider response contains multiple tool-call candidates")
        return projections[0] if projections else ()

    def _response_codec_candidates(
        self,
        request: _DecodedTextRequest | _DecodedToolRequest,
        response: dict[str, Any],
        response_codec_name: str | None,
    ) -> tuple[tuple[str, str | None], ...]:
        """Select response codecs independently from the request codec."""

        variants = request.codec_variants or (None,) * len(request.codec_names)
        if len(variants) != len(request.codec_names):
            raise _UnsupportedRequest("provider request codec identity is invalid")
        if response_codec_name is None:
            return tuple(zip(request.codec_names, variants, strict=True))
        if response_codec_name not in self._codecs:
            raise _UnsupportedRequest("Relay selected an unsupported response codec")
        matching_variants = [
            variant
            for codec_name, variant in zip(request.codec_names, variants, strict=True)
            if codec_name == response_codec_name
        ]
        if len(matching_variants) > 1:
            raise _UnsupportedRequest("provider response codec identity is ambiguous")
        if matching_variants:
            codec_variant = matching_variants[0]
        elif response_codec_name == "oci_genai":
            codec_variant = oci_adapter.response_variant(response)
        else:
            codec_variant = None
        return ((response_codec_name, codec_variant),)
