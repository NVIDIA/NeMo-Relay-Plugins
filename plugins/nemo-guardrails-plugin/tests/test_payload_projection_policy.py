# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from provider_cases import (
    anthropic_request,
    chat_request,
    chat_response,
    gemini_request,
    oci_request,
    responses_request,
)

from nemoguardrails_nemo_relay import (
    codec_projection,
    payload_policy,
)


@pytest.mark.parametrize(
    ("payload", "response", "expected_reasoning"),
    [
        (
            chat_request(),
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
            responses_request(),
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
            anthropic_request(),
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
            gemini_request(),
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
            oci_request(),
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
    payload: dict[str, object],
    response: dict[str, object],
    expected_reasoning: tuple[str, ...],
) -> None:
    check_output = codec_projection._ProviderProjector(
        payload_policy.PayloadPolicy(reasoning=payload_policy.ReasoningPolicy.CHECK_OUTPUT)
    )
    decoded = check_output.project_text(payload, require_user=True)

    projected = check_output.project_response_texts(decoded, response, allow_tool_calls=False)
    assert projected.candidates == ("safe",)
    assert projected.reasoning == expected_reasoning

    final_only = codec_projection._ProviderProjector(
        payload_policy.PayloadPolicy(reasoning=payload_policy.ReasoningPolicy.FINAL_ANSWER_ONLY)
    )
    final = final_only.project_response_texts(decoded, response, allow_tool_calls=False)
    assert final.candidates == ("safe",)
    assert final.reasoning == ()


@pytest.mark.parametrize(
    "response",
    [
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
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": "ciphertext"},
                {"type": "text", "text": "safe"},
            ],
        },
    ],
    ids=["responses-encrypted", "anthropic-redacted"],
)
def test_check_output_policy_rejects_reasoning_that_cannot_be_inspected(
    response: dict[str, object],
) -> None:
    request = responses_request() if response.get("object") == "response" else anthropic_request()
    projector = codec_projection._ProviderProjector(
        payload_policy.PayloadPolicy(reasoning=payload_policy.ReasoningPolicy.CHECK_OUTPUT)
    )
    decoded = projector.project_text(request, require_user=True)

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_texts(decoded, response, allow_tool_calls=False)


@pytest.mark.parametrize(
    "payload",
    [
        chat_request(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    ],
                }
            ]
        ),
        {
            "headers": {},
            "content": {
                "model": "fixture",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "describe"},
                            {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                        ],
                    }
                ],
            },
        },
        {
            "headers": {"anthropic-version": "2023-06-01"},
            "content": {
                "model": "fixture",
                "max_tokens": 32,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
                            },
                        ],
                    }
                ],
            },
        },
        {
            "headers": {},
            "content": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": "describe"},
                            {"inlineData": {"mimeType": "image/png", "data": "AA=="}},
                        ],
                    }
                ]
            },
        },
    ],
    ids=["chat", "responses", "anthropic", "gemini"],
)
def test_text_only_policy_checks_text_while_preserving_supported_modalities(
    payload: dict[str, object],
) -> None:
    projector = codec_projection._ProviderProjector(
        payload_policy.PayloadPolicy(multimodal=payload_policy.MultimodalPolicy.TEXT_ONLY)
    )

    assert projector.project(payload) == [{"role": "user", "content": "describe"}]


@pytest.mark.parametrize(
    "payload",
    [
        chat_request(
            [
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
        ),
        responses_request(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AA==",
                            "future_prompt": "unchecked",
                        },
                    ],
                }
            ]
        ),
    ],
    ids=["chat", "responses"],
)
def test_text_only_policy_rejects_unknown_fields_on_supported_modalities(
    payload: dict[str, object],
) -> None:
    projector = codec_projection._ProviderProjector(
        payload_policy.PayloadPolicy(multimodal=payload_policy.MultimodalPolicy.TEXT_ONLY)
    )

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project(payload)
