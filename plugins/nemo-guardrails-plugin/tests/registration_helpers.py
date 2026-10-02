# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test adapters for Relay's additive execution-context registrations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from unittest.mock import MagicMock

from nemo_relay_plugin import (
    LlmCodecIdentity,
    LlmExecutionContext,
    LlmRequestContext,
    LlmResponseContext,
)

LegacyExecutionCallback = Callable[[str, object, object], Awaitable[object]]


@dataclass(frozen=True)
class RegisteredLlmExecution:
    """One registered execution callback presented through the legacy test shape."""

    callback: LegacyExecutionCallback
    call: Any
    uses_context: bool


def _chat_annotation(request: object) -> dict[str, Any]:
    """Return the explicit normalized form used by Chat-oriented worker tests.

    This is intentionally not a provider detector or general codec. Most
    worker tests use OpenAI Chat fixtures, whose normalized scalar messages and
    standard function tools have the same shape. Provider-specific projection
    tests pass an annotation explicitly instead.
    """

    if not isinstance(request, dict) or not isinstance(request.get("content"), dict):
        raise ValueError("fixture request is invalid")
    content = request["content"]
    messages = content.get("messages")
    if not isinstance(messages, list):
        return {"messages": []}
    annotated: dict[str, Any] = {
        "messages": deepcopy(messages),
        "model": content.get("model"),
    }
    if "system" in content:
        annotated["instructions"] = deepcopy(content["system"])
    if isinstance(content.get("tools"), list):
        tools: list[object] = []
        for tool in content["tools"]:
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
                tools.append(deepcopy(tool))
            elif isinstance(tool, dict) and isinstance(tool.get("name"), str):
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description"),
                            "parameters": deepcopy(tool.get("input_schema", {})),
                        },
                    }
                )
            else:
                tools.append(deepcopy(tool))
        annotated["tools"] = tools
    return annotated


class _FixtureCodecRuntime:
    """Back real SDK codec proxies with one authoritative test annotation."""

    def __init__(
        self,
        annotation: dict[str, Any],
        *,
        decode_chat_request: bool,
    ) -> None:
        self._annotation = deepcopy(annotation)
        self._decode_chat_request = decode_chat_request

    async def _decode_llm_codec_request(
        self,
        _capability_id: str,
        _invocation_id: str,
        request: object,
    ) -> dict[str, Any]:
        if self._decode_chat_request:
            return _chat_annotation(request)
        return deepcopy(self._annotation)

    async def _encode_llm_codec_request(
        self,
        _capability_id: str,
        _invocation_id: str,
        annotated: object,
        original: object,
    ) -> dict[str, Any]:
        if not isinstance(annotated, dict) or not isinstance(original, dict):
            raise ValueError("fixture request is invalid")
        rewritten = deepcopy(original)
        normalized_messages = annotated.get("messages")
        raw_content = rewritten.get("content")
        raw_messages = raw_content.get("messages") if isinstance(raw_content, dict) else None
        if not isinstance(normalized_messages, list) or not isinstance(raw_messages, list):
            raise ValueError("fixture encoder only supports Chat message mutation")
        normalized_user = next(
            message
            for message in reversed(normalized_messages)
            if isinstance(message, dict) and message.get("role") == "user"
        )
        raw_user = next(
            message for message in reversed(raw_messages) if isinstance(message, dict) and message.get("role") == "user"
        )
        raw_user["content"] = deepcopy(normalized_user.get("content"))
        return rewritten

    async def _decode_llm_codec_response(
        self,
        _capability_id: str,
        _invocation_id: str,
        _response: object,
    ) -> dict[str, Any]:
        return {}


_UNSET = object()


def fixture_execution_context(
    _operation: str,
    request: object,
    *,
    annotated_request: dict[str, Any] | None = None,
    request_codec_name: str | None | object = _UNSET,
    response_codec_name: str | None | object = _UNSET,
) -> LlmExecutionContext:
    """Create a real SDK context backed by an explicit normalized fixture."""

    selected_request = "openai_chat" if request_codec_name is _UNSET else request_codec_name
    selected_response = "openai_chat" if response_codec_name is _UNSET else response_codec_name
    annotation = _chat_annotation(request) if annotated_request is None else deepcopy(annotated_request)
    runtime = _FixtureCodecRuntime(
        annotation,
        decode_chat_request=annotated_request is None and selected_request == "openai_chat",
    )
    request_identity = (
        LlmCodecIdentity("builtin", selected_request) if isinstance(selected_request, str) else LlmCodecIdentity("none")
    )
    response_identity = (
        LlmCodecIdentity("builtin", selected_response)
        if isinstance(selected_response, str)
        else LlmCodecIdentity("none")
    )
    return LlmExecutionContext(
        request_codec=LlmRequestContext(
            request_identity,
            runtime,  # type: ignore[arg-type]
            "request",
            "fixture",
        ),
        response_codec=LlmResponseContext(
            response_identity,
            runtime,  # type: ignore[arg-type]
            "response",
            "fixture",
        ),
    )


def llm_registration_mock(context: MagicMock, *, stream: bool = False) -> MagicMock:
    """Return the registration method production code will use for this SDK."""

    suffix = "stream_execution" if stream else "execution"
    return getattr(context, f"register_llm_{suffix}_intercept")


def registered_llm_execution(
    context: MagicMock,
    *,
    stream: bool = False,
) -> RegisteredLlmExecution:
    """Find the active registration and adapt its callback to three arguments."""

    suffix = "stream_execution" if stream else "execution"
    registration = getattr(context, f"register_llm_{suffix}_intercept")
    if registration.called:
        call = registration.call_args
        registered_callback = call.args[1]

        async def callback(operation: str, request: object, next_call: object) -> Any:
            return await registered_callback(
                operation,
                request,
                fixture_execution_context(operation, request),
                next_call,
            )

        return RegisteredLlmExecution(callback, call, True)
    raise AssertionError("worker did not register the expected LLM execution callback")
