# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh provider request and response builders for projection tests."""

from __future__ import annotations


def chat_request(
    messages: list[dict[str, object]] | None = None,
    *,
    headers: dict[str, str] | None = None,
    model: str = "fixture",
    include_response_format: bool = True,
    **content: object,
) -> dict[str, object]:
    request_content: dict[str, object] = {
        "model": model,
        "messages": ([{"role": "user", "content": "hello"}] if messages is None else messages),
    }
    if include_response_format:
        request_content["response_format"] = {"type": "text"}
    request_content.update(content)
    return {
        "headers": {} if headers is None else headers,
        "content": request_content,
    }


def guardrails_chat_request(
    messages: list[dict[str, object]],
    **content: object,
) -> dict[str, object]:
    """Build the common provider request used by Guardrails runtime tests."""

    return chat_request(
        messages,
        headers={"authorization": "not-forwarded-to-guardrails"},
        model="fixture-model",
        include_response_format=False,
        **content,
    )


def responses_request(input_value: object = "hello", **content: object) -> dict[str, object]:
    return {
        "headers": {},
        "content": {"model": "fixture", "input": input_value, **content},
    }


def anthropic_request(
    messages: list[dict[str, object]] | None = None,
    *,
    max_tokens: int = 32,
    **content: object,
) -> dict[str, object]:
    return {
        "headers": {"anthropic-version": "2023-06-01"},
        "content": {
            "model": "fixture",
            "max_tokens": max_tokens,
            "messages": ([{"role": "user", "content": "hello"}] if messages is None else messages),
            **content,
        },
    }


def oci_request(
    api_format: str = "GENERIC",
    *,
    messages: list[dict[str, object]] | None = None,
    include_routing: bool = False,
    **chat_request_fields: object,
) -> dict[str, object]:
    if api_format == "COHERE":
        request_body: dict[str, object] = {
            "apiFormat": "COHERE",
            "message": "hello",
            "chatHistory": [],
        }
    else:
        request_body = {
            "apiFormat": api_format,
            "messages": (
                [{"role": "USER", "content": [{"type": "TEXT", "text": "hello"}]}] if messages is None else messages
            ),
        }
    request_body.update(chat_request_fields)
    content: dict[str, object] = {"chatRequest": request_body}
    if include_routing:
        content.update(
            {
                "compartmentId": "ocid1.compartment.fixture",
                "servingMode": {
                    "servingType": "DEDICATED",
                    "endpointId": "ocid1.endpoint.fixture",
                },
            }
        )
    return {"headers": {}, "content": content}


def gemini_request(
    contents: list[dict[str, object]] | None = None,
    **content: object,
) -> dict[str, object]:
    return {
        "headers": {},
        "content": {
            "contents": ([{"role": "user", "parts": [{"text": "hello"}]}] if contents is None else contents),
            **content,
        },
    }


def chat_response(text: object = "safe") -> dict[str, object]:
    return {
        "id": "chatcmpl-fixture",
        "choices": [
            {
                "index": 0,
                # The OpenAI SDK serializes the deprecated field as null on
                # otherwise ordinary chat responses. Treat that wire shape as
                # equivalent to an omitted legacy function call.
                "message": {
                    "role": "assistant",
                    "content": text,
                    "function_call": None,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
    }


def tool_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }


def chat_tool_request(
    *,
    tool_name: str = "weather",
    user_text: str = "weather",
    with_result: bool = False,
    trailing_user_text: str | None = None,
) -> dict[str, object]:
    messages: list[dict[str, object]] = [{"role": "user", "content": user_text}]
    if with_result:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": '{"city":"Paris"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
            ]
        )
    if trailing_user_text is not None:
        messages.append({"role": "user", "content": trailing_user_text})
    return chat_request(
        messages,
        include_response_format=False,
        tools=[
            {
                "type": "function",
                "function": {"name": tool_name, "parameters": tool_schema()},
            }
        ],
    )


def chat_tool_response(
    name: str = "weather",
    arguments: str = '{"city":"Rome"}',
) -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
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
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def combined_protocol_case(
    protocol: str,
    *,
    text: str = "safe",
    call_name: str = "weather",
) -> tuple[dict[str, object], dict[str, object]]:
    if protocol == "openai_chat":
        request = chat_request()
        request["content"]["tools"] = [  # type: ignore[index]
            {
                "type": "function",
                "function": {"name": "weather", "parameters": tool_schema()},
            }
        ]
        response = chat_response(text)
        response["choices"][0]["message"]["tool_calls"] = [  # type: ignore[index]
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": call_name, "arguments": '{"city":"Paris"}'},
            }
        ]
        response["choices"][0]["finish_reason"] = "tool_calls"  # type: ignore[index]
        return request, response

    if protocol == "openai_responses":
        request = responses_request()
        request["content"]["tools"] = [  # type: ignore[index]
            {"type": "function", "name": "weather", "parameters": tool_schema()}
        ]
        return request, {
            "id": "resp-fixture",
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                },
                {
                    "type": "function_call",
                    "id": "fc-1",
                    "call_id": "call-1",
                    "name": call_name,
                    "arguments": '{"city":"Paris"}',
                    "status": "completed",
                },
            ],
        }

    if protocol == "anthropic_messages":
        request = anthropic_request()
        request["content"]["tools"] = [  # type: ignore[index]
            {"name": "weather", "input_schema": tool_schema()}
        ]
        return request, {
            "id": "msg-fixture",
            "type": "message",
            "role": "assistant",
            "model": "fixture",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": call_name,
                    "input": {"city": "Paris"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    if protocol == "gemini_generate_content":
        request = gemini_request()
        request["content"]["tools"] = [  # type: ignore[index]
            {
                "functionDeclarations": [
                    {
                        "name": "weather",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {"city": {"type": "STRING"}},
                            "required": ["city"],
                        },
                    }
                ]
            }
        ]
        return request, {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": text},
                            {
                                "functionCall": {
                                    "id": "call-1",
                                    "name": call_name,
                                    "args": {"city": "Paris"},
                                }
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        }

    if protocol == "oci_genai":
        request = oci_request()
        request["content"]["chatRequest"]["tools"] = [  # type: ignore[index]
            {"type": "FUNCTION", "name": "weather", "parameters": tool_schema()}
        ]
        return request, {
            "chatResponse": {
                "apiFormat": "GENERIC",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "ASSISTANT",
                            "content": [{"type": "TEXT", "text": text}],
                            "toolCalls": [
                                {
                                    "id": "call-1",
                                    "type": "FUNCTION",
                                    "name": call_name,
                                    "arguments": {"city": "Paris"},
                                }
                            ],
                        },
                        "finishReason": "TOOL_CALL",
                    }
                ],
            }
        }

    if protocol == "oci_cohere":
        request = oci_request("COHERE")
        request["content"]["chatRequest"]["tools"] = [  # type: ignore[index]
            {
                "name": "weather",
                "description": "Look up weather",
                "parameterDefinitions": {"city": {"type": "str", "description": "City name", "isRequired": True}},
            }
        ]
        return request, {
            "chatResponse": {
                "apiFormat": "COHERE",
                "text": text,
                "toolCalls": [{"name": call_name, "parameters": {"city": "Paris"}}],
                "finishReason": "COMPLETE",
            }
        }

    raise AssertionError(f"unknown protocol fixture: {protocol}")
