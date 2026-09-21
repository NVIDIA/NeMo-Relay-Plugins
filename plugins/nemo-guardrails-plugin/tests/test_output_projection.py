# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from provider_cases import (
    anthropic_request,
    chat_request,
    chat_response,
    combined_protocol_case,
    gemini_request,
    oci_request,
    responses_request,
)

from nemoguardrails_nemo_relay import (
    codec_projection,
)


@pytest.mark.parametrize(
    ("payload", "response", "expected"),
    [
        (chat_request(), chat_response("safe"), "safe"),
        (
            responses_request(),
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
            responses_request(),
            {
                "id": "resp-fixture",
                "object": "response",
                "output": [{"type": "output_text", "text": "safe"}],
                "output_text": "safe",
            },
            "safe",
        ),
        (
            anthropic_request(),
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
            oci_request(),
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
            gemini_request(),
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
    payload: dict[str, object],
    response: dict[str, object],
    expected: str,
) -> None:
    projector = codec_projection._ProviderProjector()
    decoded = projector.project_text(payload, require_user=True)

    assert projector.project_response_text(decoded, response, allow_tool_calls=False) == expected


@pytest.mark.parametrize(
    ("payload", "response"),
    [
        (
            chat_request(),
            {"choices": [chat_response()["choices"][0], chat_response("second")["choices"][0]]},
        ),
        (
            chat_request(),
            chat_response([{"type": "text", "text": "safe"}, {"type": "image_url", "image_url": {}}]),
        ),
        (
            responses_request(),
            {"id": "r", "object": "response", "output": [{"type": "web_search_call", "id": "search"}]},
        ),
        (
            responses_request(),
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
            anthropic_request(),
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
            anthropic_request(),
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
            gemini_request(),
            {"candidates": [{"content": {"role": "model", "parts": [{"inlineData": {"data": "private"}}]}}]},
        ),
        (
            gemini_request(),
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
        "multiple-chat-choices",
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
    payload: dict[str, object],
    response: dict[str, object],
) -> None:
    projector = codec_projection._ProviderProjector()
    decoded = projector.project_text(payload, require_user=True)

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_text(decoded, response, allow_tool_calls=False)


@pytest.mark.parametrize(
    "protocol",
    [
        "openai_chat",
        "openai_responses",
        "anthropic_messages",
        "gemini_generate_content",
        "oci_genai",
        "oci_cohere",
        "oci_cohere_v2",
    ],
)
def test_output_projection_rejects_unknown_response_siblings(protocol: str) -> None:
    if protocol == "oci_cohere_v2":
        request = oci_request("COHEREV2")
        response: dict[str, object] = {
            "chatResponse": {
                "apiFormat": "COHEREV2",
                "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "safe"}]},
                "finishReason": "COMPLETE",
            }
        }
        allow_tool_calls = False
    else:
        request, response = combined_protocol_case(protocol)
        allow_tool_calls = True
    response["futureExplanation"] = "UNCHECKED_CANARY"
    projector = codec_projection._ProviderProjector()
    decoded = projector.project_text(request, require_user=True)

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_text(decoded, response, allow_tool_calls=allow_tool_calls)


@pytest.mark.parametrize("protocol", ["openai_chat", "gemini_generate_content", "oci_genai"])
def test_output_projection_rejects_unknown_choice_or_candidate_siblings(protocol: str) -> None:
    request, response = combined_protocol_case(protocol)
    if protocol == "openai_chat":
        response["choices"][0]["futureExplanation"] = "UNCHECKED_CANARY"  # type: ignore[index]
    elif protocol == "gemini_generate_content":
        response["candidates"][0]["futureExplanation"] = "UNCHECKED_CANARY"  # type: ignore[index]
    else:
        response["chatResponse"]["choices"][0]["futureExplanation"] = "UNCHECKED_CANARY"  # type: ignore[index]
    projector = codec_projection._ProviderProjector()
    decoded = projector.project_text(request, require_user=True)

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_text(decoded, response, allow_tool_calls=True)


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses", "gemini_generate_content"])
def test_output_projection_rejects_known_unchecked_response_diagnostics(protocol: str) -> None:
    if protocol == "openai_chat":
        request = chat_request()
        response = chat_response()
        response["moderation"] = {"message": "UNCHECKED_CANARY"}
    elif protocol == "openai_responses":
        request = responses_request()
        response = {
            "id": "resp-fixture",
            "object": "response",
            "output": [{"type": "output_text", "text": "safe"}],
            "error": {"message": "UNCHECKED_CANARY"},
        }
    else:
        request = gemini_request()
        response = {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "safe"}]}}],
            "promptFeedback": {"blockReasonMessage": "UNCHECKED_CANARY"},
        }
    projector = codec_projection._ProviderProjector()
    decoded = projector.project_text(request, require_user=True)

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_text(decoded, response, allow_tool_calls=False)


def test_current_operational_response_metadata_remains_compatible() -> None:
    projector = codec_projection._ProviderProjector()

    chat_result = chat_response()
    chat_result.update({"metadata": {}, "moderation": None, "object": "chat.completion"})
    assert (
        projector.project_response_text(
            projector.project_text(chat_request(), require_user=True),
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
        "metadata": {},
        "moderation": {},
        "prompt": None,
        "prompt_cache_diagnostics": None,
        "service_tier": "default",
    }
    assert (
        projector.project_response_text(
            projector.project_text(responses_request(), require_user=True),
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
        "stop_details": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    assert (
        projector.project_response_text(
            projector.project_text(anthropic_request(), require_user=True),
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
                "safetyRatings": [],
                "tokenCount": 1,
            }
        ],
        "createTime": "2026-09-14T00:00:00Z",
        "modelVersion": "fixture",
        "promptFeedback": {},
        "responseId": "response-fixture",
        "usageMetadata": {"totalTokenCount": 1},
    }
    assert (
        projector.project_response_text(
            projector.project_text(gemini_request(), require_user=True),
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
                    "groundingMetadata": {},
                    "logprobs": [],
                    "serviceTier": "default",
                    "usage": {},
                }
            ],
            "usage": {"totalTokens": 1},
        },
    }
    assert (
        projector.project_response_text(
            projector.project_text(oci_request(), require_user=True),
            oci_response,
            allow_tool_calls=False,
        )
        == "safe"
    )


@pytest.mark.parametrize("text", ["", " "])
def test_output_projection_preserves_empty_and_whitespace_text(text: str) -> None:
    projector = codec_projection._ProviderProjector()
    decoded = projector.project_text(chat_request(), require_user=True)

    assert projector.project_response_text(decoded, chat_response(text), allow_tool_calls=False) == text


@pytest.mark.parametrize(
    ("payload", "response", "expected"),
    [
        (
            chat_request(),
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
            responses_request(),
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
            oci_request("COHERE"),
            {"chatResponse": {"apiFormat": "COHERE", "text": "safe", "finishReason": "COMPLETE"}},
            "safe",
        ),
        (
            oci_request("COHEREV2"),
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
    payload: dict[str, object],
    response: dict[str, object],
    expected: str,
) -> None:
    projector = codec_projection._ProviderProjector()
    decoded = projector.project_text(payload, require_user=True)

    assert projector.project_response_text(decoded, response, allow_tool_calls=False) == expected


@pytest.mark.parametrize(
    ("payload", "response"),
    [
        (
            chat_request(),
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "safe",
                            "reasoning_content": "unchecked",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        ),
        (
            chat_request(),
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "safe"},
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [
                                {
                                    "token": "123-45-6789",
                                    "top_logprobs": [{"token": "123-45-6789", "logprob": -0.1}],
                                }
                            ]
                        },
                    }
                ]
            },
        ),
        (
            responses_request(),
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "safe",
                                "annotations": [{"type": "url_citation", "url": "https://private.test"}],
                            }
                        ],
                    }
                ]
            },
        ),
        (
            anthropic_request(),
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "safe",
                        "citations": [{"type": "web_search_result_location", "title": "unchecked"}],
                    }
                ],
            },
        ),
        (
            gemini_request(),
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "safe"}]},
                        "groundingMetadata": {"searchEntryPoint": {"renderedContent": "unchecked"}},
                    }
                ]
            },
        ),
        (
            oci_request("GENERIC"),
            {
                "chatResponse": {
                    "apiFormat": "GENERIC",
                    "choices": [
                        {
                            "message": {
                                "role": "ASSISTANT",
                                "content": [{"type": "TEXT", "text": "safe"}],
                                "annotations": [{"title": "unchecked"}],
                            }
                        }
                    ],
                }
            },
        ),
        (
            oci_request("GENERIC"),
            {
                "chatResponse": {
                    "apiFormat": "GENERIC",
                    "choices": [
                        {
                            "message": {
                                "role": "ASSISTANT",
                                "content": [{"type": "TEXT", "text": "safe"}],
                            },
                            "groundingMetadata": {"sources": [{"title": "unchecked"}]},
                        }
                    ],
                }
            },
        ),
    ],
    ids=[
        "chat-reasoning",
        "chat-logprobs",
        "responses-annotations",
        "anthropic-citations",
        "gemini-grounding",
        "oci-annotations",
        "oci-choice-grounding",
    ],
)
def test_output_projection_rejects_unchecked_nested_text_metadata(
    payload: dict[str, object],
    response: dict[str, object],
) -> None:
    projector = codec_projection._ProviderProjector()
    decoded = projector.project_text(payload, require_user=True)

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_text(decoded, response, allow_tool_calls=False)


@pytest.mark.parametrize("api_format", ["GENERIC", "COHERE", "COHEREV2"])
def test_oci_response_must_match_request_api_format(api_format: str) -> None:
    projector = codec_projection._ProviderProjector()
    text_request = projector.project_text(oci_request(api_format), require_user=True)
    tool_request = projector.project_tools(oci_request(api_format), require_definitions=False)
    mismatched_response = {
        "chatResponse": {
            "apiFormat": "COHERE" if api_format != "COHERE" else "GENERIC",
            "text": "safe",
            "choices": [],
        }
    }

    with pytest.raises(codec_projection._UnsupportedRequest, match="could not be completely decoded"):
        projector.project_response_text(text_request, mismatched_response, allow_tool_calls=False)
    with pytest.raises(codec_projection._UnsupportedRequest, match="could not be completely decoded"):
        projector.project_response_tools(tool_request, mismatched_response)


def test_provider_response_projection_failures_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    projector = codec_projection._ProviderProjector()
    decoded = projector.project_text(chat_request(), require_user=True, codec_name="openai_chat")

    def unsupported_projection(*_args: object, **_kwargs: object) -> object:
        raise codec_projection._UnsupportedRequest("unsupported")

    monkeypatch.setattr(codec_projection.openai_chat_adapter, "raw_response_texts", unsupported_projection)
    with pytest.raises(codec_projection._UnsupportedRequest, match="could not be completely decoded"):
        projector.project_response_text(
            decoded,
            chat_response("safe"),
            allow_tool_calls=False,
            response_codec_name="openai_chat",
        )


def test_projection_programming_errors_are_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    projector = codec_projection._ProviderProjector()
    decoded = projector.project_text(chat_request(), require_user=True, codec_name="openai_chat")

    def broken_projection(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("projection bug")

    monkeypatch.setattr(codec_projection.openai_chat_adapter, "raw_response_texts", broken_projection)
    with pytest.raises(RuntimeError, match="projection bug"):
        projector.project_response_text(
            decoded,
            chat_response("safe"),
            allow_tool_calls=False,
            response_codec_name="openai_chat",
        )


def test_output_only_rejects_an_unchecked_tool_only_response() -> None:
    projector = codec_projection._ProviderProjector()
    decoded = projector.project_text(chat_request(), require_user=True)
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
        projector.project_response_text(decoded, response, allow_tool_calls=False)


@pytest.mark.parametrize("protocol", ["openai_chat", "gemini_generate_content"])
def test_every_text_candidate_is_projected_independently(protocol: str) -> None:
    if protocol == "openai_chat":
        request = chat_request()
        response = chat_response("first")
        response["choices"].append(chat_response("second")["choices"][0])  # type: ignore[union-attr,index]
    else:
        request = gemini_request()
        response = {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "first"}]}},
                {"content": {"role": "model", "parts": [{"text": "second"}]}},
            ]
        }
    projector = codec_projection._ProviderProjector()
    decoded = projector.project_text(request, require_user=True)

    assert projector.project_response_texts(
        decoded,
        response,
        allow_tool_calls=False,
    ).candidates == ("first", "second")
