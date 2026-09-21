# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

from nemoguardrails_nemo_relay.structural_tools import (
    Guardrails024StructuralToolAdapter,
    ToolCall,
    ToolDefinition,
    ToolExchange,
    ToolVerdict,
)


async def check_calls(
    definitions: tuple[ToolDefinition, ...],
    calls: tuple[ToolCall, ...],
) -> ToolVerdict:
    adapter = Guardrails024StructuralToolAdapter()
    try:
        return await adapter.check_calls(definitions, calls)
    finally:
        await adapter.close()


async def check_results(exchanges: tuple[ToolExchange, ...]) -> ToolVerdict:
    adapter = Guardrails024StructuralToolAdapter()
    try:
        return await adapter.check_results(exchanges)
    finally:
        await adapter.close()


def json_object_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }


def gemini_object_schema() -> dict[str, object]:
    return {
        "type": "OBJECT",
        "properties": {"city": {"type": "STRING"}},
        "required": ["city"],
    }


def _build_protocol_cases() -> dict[str, tuple[dict[str, object], dict[str, object]]]:
    return {
        "openai_chat": (
            {
                "headers": {},
                "content": {
                    "model": "fixture",
                    "messages": [
                        {"role": "user", "content": "weather"},
                        {
                            "role": "assistant",
                            "content": None,
                            # The OpenAI SDK emits this deprecated field as
                            # null on ordinary assistant messages.
                            "function_call": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "weather", "parameters": json_object_schema()},
                        }
                    ],
                },
            },
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-2",
                                    "type": "function",
                                    "function": {"name": "weather", "arguments": '{"city":"Rome"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ),
        "openai_responses": (
            {
                "headers": {},
                "content": {
                    "model": "fixture",
                    "input": [
                        {"role": "user", "content": "weather"},
                        {
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "weather",
                            "arguments": '{"city":"Paris"}',
                        },
                        {"type": "function_call_output", "call_id": "call-1", "output": "sunny"},
                    ],
                    "tools": [{"type": "function", "name": "weather", "parameters": json_object_schema()}],
                },
            },
            {
                "id": "resp-1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "fixture",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc-2",
                        "call_id": "call-2",
                        "name": "weather",
                        "arguments": '{"city":"Rome"}',
                        "status": "completed",
                    }
                ],
            },
        ),
        "anthropic_messages": (
            {
                "headers": {"anthropic-version": "2023-06-01"},
                "content": {
                    "model": "fixture",
                    "max_tokens": 64,
                    "messages": [
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
                            "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "sunny"}],
                        },
                    ],
                    "tools": [{"name": "weather", "input_schema": json_object_schema()}],
                },
            },
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "model": "fixture",
                "content": [{"type": "tool_use", "id": "call-2", "name": "weather", "input": {"city": "Rome"}}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ),
        "gemini_generate_content": (
            {
                "headers": {},
                "content": {
                    "contents": [
                        {"role": "user", "parts": [{"text": "weather"}]},
                        {
                            "role": "model",
                            "parts": [{"functionCall": {"id": "call-1", "name": "weather", "args": {"city": "Paris"}}}],
                        },
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "functionResponse": {
                                        "id": "call-1",
                                        "name": "weather",
                                        "response": {"result": "sunny"},
                                    }
                                }
                            ],
                        },
                    ],
                    "tools": [{"functionDeclarations": [{"name": "weather", "parameters": gemini_object_schema()}]}],
                },
            },
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"functionCall": {"id": "call-2", "name": "weather", "args": {"city": "Rome"}}}],
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
        ),
        "oci_genai": (
            {
                "headers": {},
                "content": {
                    "chatRequest": {
                        "apiFormat": "GENERIC",
                        "messages": [
                            {"role": "USER", "content": [{"type": "TEXT", "text": "weather"}]},
                            {
                                "role": "ASSISTANT",
                                "content": [],
                                "toolCalls": [
                                    {
                                        "id": "call-1",
                                        "type": "FUNCTION",
                                        "name": "weather",
                                        "arguments": {"city": "Paris"},
                                    }
                                ],
                            },
                            {
                                "role": "TOOL",
                                "content": [{"type": "TEXT", "text": "sunny"}],
                                "toolCallId": "call-1",
                            },
                        ],
                        "tools": [{"type": "FUNCTION", "name": "weather", "parameters": json_object_schema()}],
                    }
                },
            },
            {
                "chatResponse": {
                    "apiFormat": "GENERIC",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "ASSISTANT",
                                "content": [],
                                "toolCalls": [
                                    {
                                        "id": "call-2",
                                        "type": "FUNCTION",
                                        "name": "weather",
                                        "arguments": {"city": "Rome"},
                                    }
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                }
            },
        ),
    }


PROTOCOLS = (
    "openai_chat",
    "openai_responses",
    "anthropic_messages",
    "gemini_generate_content",
    "oci_genai",
)


def protocol_case(protocol: str) -> tuple[dict[str, object], dict[str, object]]:
    try:
        return _build_protocol_cases()[protocol]
    except KeyError as exc:  # pragma: no cover - callers use the catalog above
        raise AssertionError(f"unknown protocol fixture: {protocol}") from exc


def openai_chat_case() -> tuple[dict[str, object], dict[str, object]]:
    return protocol_case("openai_chat")


def openai_responses_case() -> tuple[dict[str, object], dict[str, object]]:
    return protocol_case("openai_responses")


def anthropic_messages_case() -> tuple[dict[str, object], dict[str, object]]:
    return protocol_case("anthropic_messages")


def gemini_generate_content_case() -> tuple[dict[str, object], dict[str, object]]:
    return protocol_case("gemini_generate_content")


def oci_generic_case() -> tuple[dict[str, object], dict[str, object]]:
    return protocol_case("oci_genai")


def response_with_text(protocol: str, response: dict[str, object], text: str) -> dict[str, object]:
    response = deepcopy(response)
    if protocol == "openai_chat":
        response["choices"][0]["message"]["content"] = text  # type: ignore[index]
    elif protocol == "openai_responses":
        response["output"].insert(  # type: ignore[union-attr]
            0,
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
        )
    elif protocol == "anthropic_messages":
        response["content"].insert(0, {"type": "text", "text": text})  # type: ignore[union-attr]
    elif protocol == "gemini_generate_content":
        response["candidates"][0]["content"]["parts"].insert(0, {"text": text})  # type: ignore[index]
    elif protocol == "oci_genai":
        response["chatResponse"]["choices"][0]["message"]["content"] = [  # type: ignore[index]
            {"type": "TEXT", "text": text}
        ]
    else:  # pragma: no cover - the parameter list is the protocol catalog
        raise AssertionError(protocol)
    return response


def oci_cohere_v2_case() -> tuple[dict[str, object], dict[str, object]]:
    request = {
        "headers": {},
        "content": {
            "chatRequest": {
                "apiFormat": "COHEREV2",
                "messages": [
                    {"role": "USER", "content": [{"type": "TEXT", "text": "weather"}]},
                    {
                        "role": "ASSISTANT",
                        "content": [],
                        "toolCalls": [
                            {
                                "id": "call-1",
                                "type": "FUNCTION",
                                "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                            }
                        ],
                    },
                    {
                        "role": "TOOL",
                        "content": [{"type": "TEXT", "text": "sunny"}],
                        "toolCallId": "call-1",
                    },
                ],
                "tools": [
                    {
                        "type": "FUNCTION",
                        "function": {"name": "weather", "parameters": json_object_schema()},
                    }
                ],
            }
        },
    }
    response = {
        "chatResponse": {
            "apiFormat": "COHEREV2",
            "message": {
                "role": "ASSISTANT",
                "content": [],
                "toolCalls": [
                    {
                        "id": "call-2",
                        "type": "FUNCTION",
                        "function": {"name": "weather", "arguments": '{"city":"Rome"}'},
                    }
                ],
            },
            "finishReason": "TOOL_CALL",
        }
    }
    return request, response
