# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest
from projection_fixtures import decoded_text, decoded_tools, response_projection, response_text, response_tools
from provider_cases import chat_response
from tool_projection_cases import protocol_case, response_with_text

from nemoguardrails_nemo_relay import (
    codec_projection,
)


@pytest.mark.parametrize(
    ("codec_name", "response", "expected"),
    [
        ("openai_chat", chat_response("safe"), "safe"),
        (
            "openai_responses",
            {
                "id": "resp-fixture",
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "safe", "annotations": []},
                            {"type": "output_text", "text": "second", "annotations": []},
                        ],
                    }
                ],
                "output_text": "safe\nsecond",
            },
            "safe\nsecond",
        ),
        (
            "openai_responses",
            {
                "id": "resp-fixture",
                "object": "response",
                "output": [
                    {
                        "id": "msg-fixture",
                        "status": "completed",
                        "type": "output_text",
                        "text": "safe",
                    }
                ],
                "output_text": "safe",
            },
            "safe",
        ),
        (
            "anthropic_messages",
            {
                "id": "msg-fixture",
                "type": "message",
                "role": "assistant",
                "model": "fixture",
                "content": [
                    {"type": "text", "text": "safe"},
                    {"type": "text", "text": "second"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
            "safe\nsecond",
        ),
        (
            "oci_genai",
            {
                "chatResponse": {
                    "apiFormat": "GENERIC",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "ASSISTANT",
                                "content": [
                                    {"type": "TEXT", "text": "safe"},
                                    {"type": "TEXT", "text": "second"},
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                }
            },
            "safesecond",
        ),
        (
            "gemini_generate_content",
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "safe"},
                                {"text": "second"},
                            ],
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ]
            },
            "safe\nsecond",
        ),
    ],
    ids=["chat", "responses-message", "responses-output-text", "anthropic", "oci", "gemini"],
)
def test_output_projection_covers_all_visible_text(
    codec_name: str,
    response: dict[str, object],
    expected: str,
) -> None:
    projector = codec_projection._ProviderProjector()
    decoded = decoded_text(
        codec_name,
        codec_variant="GENERIC" if codec_name == "oci_genai" else None,
    )

    assert response_text(projector, decoded, response, allow_tool_calls=False) == expected


@pytest.mark.parametrize(
    ("codec_name", "response"),
    [
        (
            "openai_chat",
            chat_response([{"type": "text", "text": "safe"}, {"type": "image_url", "image_url": {}}]),
        ),
        (
            "openai_responses",
            {"id": "r", "object": "response", "output": [{"type": "web_search_call", "id": "search"}]},
        ),
        (
            "openai_responses",
            {
                "id": "r",
                "object": "response",
                "output": [
                    {"type": "reasoning", "id": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "safe", "annotations": []}],
                    },
                ],
            },
        ),
        (
            "anthropic_messages",
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "fixture",
                "content": [{"type": "image", "source": {"type": "base64", "data": "private"}}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ),
        (
            "anthropic_messages",
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "fixture",
                "content": [
                    {"type": "thinking", "thinking": "private", "signature": "signature"},
                    {"type": "text", "text": "safe"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ),
        (
            "gemini_generate_content",
            {"candidates": [{"content": {"role": "model", "parts": [{"inlineData": {"data": "private"}}]}}]},
        ),
        (
            "gemini_generate_content",
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "private", "thought": True}, {"text": "safe"}],
                        }
                    }
                ]
            },
        ),
    ],
    ids=[
        "chat-multipart",
        "hosted-tool",
        "responses-reasoning",
        "anthropic-image",
        "anthropic-thinking",
        "gemini-image",
        "gemini-thought",
    ],
)
def test_output_projection_rejects_incomplete_coverage(
    codec_name: str,
    response: dict[str, object],
) -> None:
    projector = codec_projection._ProviderProjector()
    decoded = decoded_text(codec_name)

    with pytest.raises(codec_projection._UnsupportedRequest):
        response_text(projector, decoded, response, allow_tool_calls=False)


def test_current_operational_response_metadata_remains_compatible() -> None:
    projector = codec_projection._ProviderProjector()

    chat_result = chat_response()
    chat_result.update(
        {
            "futureDiagnostic": "preserved",
            "metadata": {"request": "fixture"},
            "moderation": {"flagged": False},
            "object": "chat.completion",
        }
    )
    assert (
        response_text(
            projector,
            decoded_text("openai_chat"),
            chat_result,
            allow_tool_calls=False,
        )
        == "safe"
    )

    responses_response = {
        "id": "resp-fixture",
        "object": "response",
        "output": [{"type": "output_text", "text": "safe"}],
        "output_text": "safe",
        "instructions": "checked request context",
        "error": {"message": "diagnostic"},
        "metadata": {"request": "fixture"},
        "moderation": {"flagged": False},
        "prompt": None,
        "prompt_cache_diagnostics": None,
        "service_tier": "default",
    }
    assert (
        response_text(
            projector,
            decoded_text("openai_responses"),
            responses_response,
            allow_tool_calls=False,
        )
        == "safe"
    )

    anthropic_response = {
        "id": "msg-fixture",
        "type": "message",
        "role": "assistant",
        "model": "fixture",
        "content": [{"type": "text", "text": "safe"}],
        "stop_reason": "end_turn",
        "stop_details": {"type": "diagnostic", "message": "preserved"},
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    assert (
        response_text(
            projector,
            decoded_text("anthropic_messages"),
            anthropic_response,
            allow_tool_calls=False,
        )
        == "safe"
    )

    gemini_response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "safe"}]},
                "finishReason": "STOP",
                "index": 0,
                "avgLogprobs": None,
                "groundingMetadata": {"searchEntryPoint": {"renderedContent": "preserved"}},
                "safetyRatings": [],
                "tokenCount": 1,
            }
        ],
        "createTime": "2026-09-14T00:00:00Z",
        "modelVersion": "fixture",
        "modelStatus": {"message": "preserved"},
        "promptFeedback": {},
        "responseId": "response-fixture",
        "usageMetadata": {"totalTokenCount": 1},
    }
    assert (
        response_text(
            projector,
            decoded_text("gemini_generate_content"),
            gemini_response,
            allow_tool_calls=False,
        )
        == "safe"
    )

    oci_response = {
        "modelId": "fixture",
        "modelVersion": "1",
        "chatResponse": {
            "apiFormat": "GENERIC",
            "timeCreated": "2026-09-14T00:00:00Z",
            "serviceTier": "default",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "safe"}]},
                    "finishReason": "STOP",
                    "groundingMetadata": {"sources": [{"title": "preserved"}]},
                    "logprobs": [],
                    "serviceTier": "default",
                    "usage": {},
                }
            ],
            "usage": {"totalTokens": 1},
        },
    }
    assert (
        response_text(
            projector,
            decoded_text("oci_genai", codec_variant="GENERIC"),
            oci_response,
            allow_tool_calls=False,
        )
        == "safe"
    )


@pytest.mark.parametrize("text", ["", " "])
def test_output_projection_preserves_empty_and_whitespace_text(text: str) -> None:
    projector = codec_projection._ProviderProjector()
    decoded = decoded_text("openai_chat")

    assert response_text(projector, decoded, chat_response(text), allow_tool_calls=False) == text


@pytest.mark.parametrize(
    ("codec_name", "codec_variant", "response", "expected"),
    [
        (
            "openai_chat",
            None,
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": None, "refusal": "declined"},
                        "finish_reason": "stop",
                    }
                ]
            },
            "declined",
        ),
        (
            "openai_responses",
            None,
            {
                "id": "r",
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "refusal", "refusal": "declined"}],
                    }
                ],
            },
            "declined",
        ),
        (
            "oci_genai",
            "COHERE",
            {"chatResponse": {"apiFormat": "COHERE", "text": "safe", "finishReason": "COMPLETE"}},
            "safe",
        ),
        (
            "oci_genai",
            "COHEREV2",
            {
                "chatResponse": {
                    "apiFormat": "COHEREV2",
                    "message": {
                        "role": "ASSISTANT",
                        "content": [
                            {"type": "TEXT", "text": "safe"},
                            {"type": "TEXT", "text": "second"},
                        ],
                    },
                    "finishReason": "COMPLETE",
                }
            },
            "safesecond",
        ),
    ],
    ids=["chat-refusal", "responses-refusal", "oci-cohere", "oci-cohere-v2"],
)
def test_output_projection_includes_known_scalar_variants(
    codec_name: str,
    codec_variant: str | None,
    response: dict[str, object],
    expected: str,
) -> None:
    projector = codec_projection._ProviderProjector()
    decoded = decoded_text(codec_name, codec_variant=codec_variant)

    assert response_text(projector, decoded, response, allow_tool_calls=False) == expected


@pytest.mark.parametrize("api_format", ["GENERIC", "COHERE", "COHEREV2"])
def test_oci_response_must_match_request_api_format(api_format: str) -> None:
    projector = codec_projection._ProviderProjector()
    text_request = decoded_text("oci_genai", codec_variant=api_format)
    tool_request = decoded_tools("oci_genai", codec_variant=api_format, definitions=())
    mismatched_response = {
        "chatResponse": {
            "apiFormat": "COHERE" if api_format != "COHERE" else "GENERIC",
            "text": "safe",
            "choices": [],
        }
    }

    with pytest.raises(codec_projection._UnsupportedRequest, match="does not match"):
        response_text(projector, text_request, mismatched_response, allow_tool_calls=False)
    with pytest.raises(codec_projection._UnsupportedRequest, match="does not match"):
        response_tools(projector, tool_request, mismatched_response)


@pytest.mark.parametrize(
    ("api_format", "extra_field", "extra_value"),
    [
        ("GENERIC", "text", "unchecked"),
        ("GENERIC", "message", {"role": "ASSISTANT", "content": "unchecked"}),
        ("COHEREV2", "text", "unchecked"),
        ("COHERE", "choices", [{"message": {"role": "ASSISTANT", "content": "unchecked"}}]),
    ],
)
def test_oci_response_rejects_payloads_from_another_dialect(
    api_format: str,
    extra_field: str,
    extra_value: object,
) -> None:
    _, response = protocol_case("oci_genai")
    chat = response["chatResponse"]  # type: ignore[assignment]
    if api_format == "COHERE":
        chat.clear()  # type: ignore[union-attr]
        chat.update({"apiFormat": api_format, "text": "safe", extra_field: extra_value})  # type: ignore[union-attr]
    elif api_format == "COHEREV2":
        chat.clear()  # type: ignore[union-attr]
        chat.update(  # type: ignore[union-attr]
            {
                "apiFormat": api_format,
                "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "safe"}]},
                extra_field: extra_value,
            }
        )
    else:
        chat[extra_field] = extra_value  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest, match="unexpected"):
        response_text(
            codec_projection._ProviderProjector(),
            decoded_text("oci_genai", codec_variant=api_format),
            response,
            allow_tool_calls=True,
        )


def test_output_only_rejects_an_unchecked_tool_only_response() -> None:
    projector = codec_projection._ProviderProjector()
    decoded = decoded_text("openai_chat")
    response = chat_response(None)
    response["choices"][0]["message"]["tool_calls"] = [  # type: ignore[index]
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "weather", "arguments": "{}"},
        }
    ]
    response["choices"][0]["finish_reason"] = "tool_calls"  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        response_text(projector, decoded, response, allow_tool_calls=False)


def test_openai_chat_rejects_a_falsey_nonarray_tool_calls_value() -> None:
    response = chat_response(None)
    response["choices"][0]["message"]["tool_calls"] = {}  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest, match="must be an array"):
        response_tools(codec_projection._ProviderProjector(), decoded_tools("openai_chat"), response)


@pytest.mark.parametrize("protocol", ["openai_chat", "gemini_generate_content"])
def test_every_text_candidate_is_projected_independently(protocol: str) -> None:
    if protocol == "openai_chat":
        response = chat_response("first")
        response["choices"].append(chat_response("second")["choices"][0])  # type: ignore[union-attr,index]
    else:
        response = {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "first"}]}},
                {"content": {"role": "model", "parts": [{"text": "second"}]}},
            ]
        }
    projector = codec_projection._ProviderProjector()
    decoded = decoded_text(protocol)

    assert response_projection(projector, decoded, response, allow_tool_calls=False).candidates == ("first", "second")


@pytest.mark.parametrize("protocol", ["openai_chat", "gemini_generate_content", "oci_genai"])
def test_one_empty_candidate_cannot_hide_behind_a_covered_candidate(protocol: str) -> None:
    _, response = protocol_case(protocol)
    response = response_with_text(protocol, response, "safe")
    if protocol == "openai_chat":
        empty = deepcopy(response["choices"][0])  # type: ignore[index]
        empty["message"]["content"] = None  # type: ignore[index]
        empty["message"].pop("tool_calls")  # type: ignore[union-attr]
        empty["finish_reason"] = "stop"
        response["choices"].append(empty)  # type: ignore[union-attr]
    elif protocol == "gemini_generate_content":
        empty = deepcopy(response["candidates"][0])  # type: ignore[index]
        empty["content"]["parts"] = []  # type: ignore[index]
        response["candidates"].append(empty)  # type: ignore[union-attr]
    else:
        empty = deepcopy(response["chatResponse"]["choices"][0])  # type: ignore[index]
        empty["message"]["content"] = []  # type: ignore[index]
        empty["message"].pop("toolCalls")  # type: ignore[union-attr]
        response["chatResponse"]["choices"].append(empty)  # type: ignore[index,union-attr]

    with pytest.raises(codec_projection._UnsupportedRequest, match="no covered output"):
        response_text(
            codec_projection._ProviderProjector(),
            decoded_text(protocol, codec_variant="GENERIC" if protocol == "oci_genai" else None),
            response,
            allow_tool_calls=True,
        )


@pytest.mark.parametrize(
    "protocol",
    ["openai_chat", "openai_responses", "anthropic_messages", "gemini_generate_content", "oci_genai"],
)
def test_unknown_text_field_in_response_content_fails_closed(protocol: str) -> None:
    _, response = protocol_case(protocol)
    response = response_with_text(protocol, response, "safe")
    if protocol == "openai_chat":
        target = response["choices"][0]["message"]  # type: ignore[index]
    elif protocol == "openai_responses":
        target = response["output"][0]["content"][0]  # type: ignore[index]
    elif protocol == "anthropic_messages":
        target = response["content"][0]  # type: ignore[index]
    elif protocol == "gemini_generate_content":
        target = response["candidates"][0]["content"]["parts"][0]  # type: ignore[index]
    else:
        target = response["chatResponse"]["choices"][0]["message"]["content"][0]  # type: ignore[index]
    target["future_text"] = "unchecked"  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest, match="unmodeled fields"):
        response_text(
            codec_projection._ProviderProjector(),
            decoded_text(protocol, codec_variant="GENERIC" if protocol == "oci_genai" else None),
            response,
            allow_tool_calls=True,
        )


@pytest.mark.parametrize(
    ("protocol", "response"),
    [
        (
            "openai_chat",
            {"choices": [{"message": {"role": [], "content": "safe"}, "finish_reason": "stop"}]},
        ),
        (
            "openai_responses",
            {
                "output": [
                    {
                        "type": "message",
                        "role": [],
                        "content": [{"type": "output_text", "text": "safe"}],
                    }
                ]
            },
        ),
        (
            "anthropic_messages",
            {"type": "message", "role": [], "content": [{"type": "text", "text": "safe"}]},
        ),
        (
            "gemini_generate_content",
            {"candidates": [{"content": {"role": [], "parts": [{"text": "safe"}]}}]},
        ),
    ],
)
def test_unhashable_response_discriminators_fail_with_coverage_error(
    protocol: str,
    response: dict[str, object],
) -> None:
    with pytest.raises(codec_projection._UnsupportedRequest):
        response_text(
            codec_projection._ProviderProjector(),
            decoded_text(protocol),
            response,
            allow_tool_calls=True,
        )


@pytest.mark.parametrize("protocol", ["openai_chat", "gemini_generate_content", "oci_genai"])
def test_candidate_fanout_is_bounded_before_inspection(protocol: str) -> None:
    _, response = protocol_case(protocol)
    if protocol == "openai_chat":
        response["choices"] = [None] * 33
    elif protocol == "gemini_generate_content":
        response["candidates"] = [None] * 33
    else:
        response["chatResponse"]["choices"] = [None] * 33  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest, match="too many candidates"):
        response_text(
            codec_projection._ProviderProjector(),
            decoded_text(protocol, codec_variant="GENERIC" if protocol == "oci_genai" else None),
            response,
            allow_tool_calls=True,
        )
