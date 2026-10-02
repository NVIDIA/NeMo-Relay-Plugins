# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small authoritative Relay annotations for projection unit tests.

These fixtures deliberately do not decode provider payloads. Relay owns that
operation. Tests that exercise plugin-owned response coverage start from the
normalized request state Relay would have supplied.
"""

from __future__ import annotations

from nemoguardrails_nemo_relay.codec_projection import (
    _DecodedTextRequest,
    _DecodedToolRequest,
    _UnsupportedRequest,
)
from nemoguardrails_nemo_relay.structural_tools import ToolCall, ToolDefinition, ToolExchange, ToolResult
from nemoguardrails_nemo_relay.tool_projection import ToolRequestProjection


def weather_annotation(codec_name: str) -> dict[str, object]:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    if codec_name == "gemini_generate_content":
        schema = {
            "type": "OBJECT",
            "properties": {"city": {"type": "STRING"}},
            "required": ["city"],
        }
    definition = {
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": schema,
        },
    }
    if codec_name == "openai_responses":
        messages: list[dict[str, object]] = [
            {"role": "user", "content": "weather"},
            {
                "role": "tool_call",
                "call_id": "call-1",
                "name": "weather",
                "arguments": {"city": "Paris"},
            },
            {"role": "tool_result", "call_id": "call-1", "output": "sunny"},
        ]
    elif codec_name in {"anthropic_messages", "gemini_generate_content"}:
        messages = [
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "weather",
                        "input": {"city": "Paris"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": "sunny",
                    }
                ],
            },
        ]
    else:
        messages = [
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
        ]
    annotation: dict[str, object] = {"messages": messages, "tools": [definition]}
    if codec_name == "oci_genai":
        annotation["api_specific"] = {"api": "oci_genai", "api_format": "GENERIC"}
    return annotation


def decoded_text(
    codec_name: str,
    *,
    codec_variant: str | None = None,
    messages: tuple[dict[str, str], ...] = ({"role": "user", "content": "hello"},),
) -> _DecodedTextRequest:
    return _DecodedTextRequest(codec_name, messages, codec_variant)


def decoded_tools(
    codec_name: str,
    *,
    codec_variant: str | None = None,
    definitions: tuple[ToolDefinition, ...] | None = None,
    exchanges: tuple[ToolExchange, ...] = (),
) -> _DecodedToolRequest:
    if definitions is None:
        definitions = (
            ToolDefinition(
                "weather",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            ),
        )
    return _DecodedToolRequest(
        codec_name,
        ToolRequestProjection(definitions, exchanges),
        codec_variant,
    )


def response_text(projector, request, response, *, allow_tool_calls):
    projection = response_projection(projector, request, response, allow_tool_calls=allow_tool_calls)
    try:
        return projection.one()
    except ValueError:
        raise _UnsupportedRequest("provider response contains multiple text candidates") from None


def response_projection(projector, request, response, *, allow_tool_calls):
    return projector.project_response_texts(
        request, response, allow_tool_calls=allow_tool_calls, response_codec_name=request.codec_name or ""
    )


def response_tools(projector, request, response):
    projections = projector.project_response_tool_candidates(
        request, response, response_codec_name=request.codec_name or ""
    )
    if len(projections) > 1:
        raise _UnsupportedRequest("provider response contains multiple tool-call candidates")
    return projections[0] if projections else ()


def weather_exchange(*, result: bool = True) -> ToolExchange:
    return ToolExchange(
        (ToolCall("call-1", "weather", {"city": "Paris"}),),
        (ToolResult("call-1", None, "sunny"),) if result else (),
    )
