# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from projection_fixtures import decoded_text, response_projection
from provider_cases import (
    anthropic_request,
    chat_request,
    chat_response,
    gemini_request,
    responses_request,
)

from nemoguardrails_nemo_relay import (
    codec_projection,
    payload_policy,
)


@pytest.mark.parametrize(
    ("codec_name", "codec_variant", "response", "expected_reasoning"),
    [
        (
            "openai_chat",
            None,
            {
                **chat_response("safe"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "safe",
                            "reasoning_content": "private reasoning",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            ("private reasoning",),
        ),
        (
            "openai_responses",
            None,
            {
                "id": "resp-1",
                "object": "response",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs-1",
                        "summary": [{"type": "summary_text", "text": "summary"}],
                        "content": [{"type": "reasoning_text", "text": "private reasoning"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "safe", "annotations": []}],
                    },
                ],
            },
            ("summary", "private reasoning"),
        ),
        (
            "anthropic_messages",
            None,
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private reasoning", "signature": "sig"},
                    {"type": "text", "text": "safe"},
                ],
            },
            ("private reasoning",),
        ),
        (
            "gemini_generate_content",
            None,
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "private reasoning", "thought": True, "thoughtSignature": "sig"},
                                {"text": "safe"},
                            ],
                        }
                    }
                ]
            },
            ("private reasoning",),
        ),
        (
            "oci_genai",
            "GENERIC",
            {
                "chatResponse": {
                    "apiFormat": "GENERIC",
                    "choices": [
                        {
                            "message": {
                                "role": "ASSISTANT",
                                "content": [{"type": "TEXT", "text": "safe"}],
                                "reasoningContent": "private reasoning",
                            }
                        }
                    ],
                }
            },
            ("private reasoning",),
        ),
    ],
    ids=["chat", "responses", "anthropic", "gemini", "oci"],
)
def test_reasoning_policies_separate_plaintext_reasoning_from_the_final_answer(
    codec_name: str,
    codec_variant: str | None,
    response: dict[str, object],
    expected_reasoning: tuple[str, ...],
) -> None:
    check_output = codec_projection._ProviderProjector(
        payload_policy.PayloadPolicy(reasoning=payload_policy.ReasoningPolicy.CHECK_OUTPUT)
    )
    decoded = decoded_text(codec_name, codec_variant=codec_variant)

    projected = response_projection(check_output, decoded, response, allow_tool_calls=False)
    assert projected.candidates == ("safe",)
    assert projected.reasoning == expected_reasoning

    final_only = codec_projection._ProviderProjector(
        payload_policy.PayloadPolicy(reasoning=payload_policy.ReasoningPolicy.FINAL_ANSWER_ONLY)
    )
    final = response_projection(final_only, decoded, response, allow_tool_calls=False)
    assert final.candidates == ("safe",)
    assert final.reasoning == ()


@pytest.mark.parametrize(
    ("codec_name", "response"),
    [
        (
            "openai_responses",
            {
                "id": "resp-1",
                "object": "response",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs-1",
                        "summary": [],
                        "encrypted_content": "ciphertext",
                    },
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
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "ciphertext"},
                    {"type": "text", "text": "safe"},
                ],
            },
        ),
    ],
    ids=["responses-encrypted", "anthropic-redacted"],
)
def test_check_output_policy_rejects_reasoning_that_cannot_be_inspected(
    codec_name: str,
    response: dict[str, object],
) -> None:
    projector = codec_projection._ProviderProjector(
        payload_policy.PayloadPolicy(reasoning=payload_policy.ReasoningPolicy.CHECK_OUTPUT)
    )
    decoded = decoded_text(codec_name)

    with pytest.raises(codec_projection._UnsupportedRequest):
        response_projection(projector, decoded, response, allow_tool_calls=False)


@pytest.mark.parametrize(
    ("codec_name", "payload", "annotation"),
    [
        (
            "openai_chat",
            chat_request(include_response_format=False),
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                        ],
                    }
                ]
            },
        ),
        (
            "openai_responses",
            responses_request(),
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image", "image": {"image_url": "data:image/png;base64,AA=="}},
                        ],
                    }
                ]
            },
        ),
        (
            "anthropic_messages",
            anthropic_request(),
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {
                                "type": "image",
                                "image": {"source": {"type": "base64", "data": "AA=="}},
                            },
                        ],
                    }
                ]
            },
        ),
        (
            "gemini_generate_content",
            gemini_request(),
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {
                                "type": "provider_native",
                                "provider": "gemini",
                                "kind": "inlineData",
                                "value": {"inlineData": {"mimeType": "image/png", "data": "AA=="}},
                            },
                        ],
                    }
                ]
            },
        ),
    ],
    ids=["chat", "responses", "anthropic", "gemini"],
)
def test_text_only_policy_checks_text_while_preserving_supported_modalities(
    codec_name: str,
    payload: dict[str, object],
    annotation: dict[str, object],
) -> None:
    projector = codec_projection._ProviderProjector(
        payload_policy.PayloadPolicy(multimodal=payload_policy.MultimodalPolicy.TEXT_ONLY)
    )

    assert projector.project_host_text(
        payload,
        annotation,
        require_user=True,
        codec_name=codec_name,
    ).messages == ({"role": "user", "content": "describe"},)


@pytest.mark.parametrize(
    ("codec_name", "payload", "annotation"),
    [
        (
            "openai_chat",
            chat_request(include_response_format=False),
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AA=="},
                                "future_prompt": "unchecked",
                            },
                        ],
                    }
                ]
            },
        ),
        (
            "openai_responses",
            responses_request(),
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {
                                "type": "image",
                                "image": {"image_url": "data:image/png;base64,AA=="},
                                "future_prompt": "unchecked",
                            },
                        ],
                    }
                ]
            },
        ),
    ],
    ids=["chat", "responses"],
)
def test_text_only_policy_rejects_unknown_fields_on_supported_modalities(
    codec_name: str,
    payload: dict[str, object],
    annotation: dict[str, object],
) -> None:
    projector = codec_projection._ProviderProjector(
        payload_policy.PayloadPolicy(multimodal=payload_policy.MultimodalPolicy.TEXT_ONLY)
    )

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_host_text(
            payload,
            annotation,
            require_user=True,
            codec_name=codec_name,
        )
