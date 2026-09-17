# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemoguardrails_nemo_relay.payload_policy import (
    MultimodalPolicy,
    PayloadPolicy,
    ProjectedOutputText,
    ReasoningPolicy,
    is_supported_normalized_modality,
    payload_policy_from_config,
    validate_anthropic_reasoning,
    validate_openai_responses_reasoning,
)


def test_payload_policy_is_strict_by_default() -> None:
    assert payload_policy_from_config(None) == PayloadPolicy()


def test_payload_policy_parses_each_explicit_boundary() -> None:
    assert payload_policy_from_config({"multimodal": "text_only", "reasoning": "final_answer_only"}) == PayloadPolicy(
        multimodal=MultimodalPolicy.TEXT_ONLY,
        reasoning=ReasoningPolicy.FINAL_ANSWER_ONLY,
    )
    assert payload_policy_from_config({"reasoning": "check_output"}) == PayloadPolicy(
        reasoning=ReasoningPolicy.CHECK_OUTPUT
    )


@pytest.mark.parametrize(
    "value",
    [[], {"future": "value"}, {"multimodal": "pass"}, {"reasoning": "all"}],
)
def test_payload_policy_rejects_unknown_configuration(value: object) -> None:
    with pytest.raises(ValueError):
        payload_policy_from_config(value)


@pytest.mark.parametrize(
    ("codec", "part"),
    [
        ("openai_chat", {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}),
        ("openai_chat", {"type": "audio", "audio": {"data": "AA==", "format": "wav"}}),
        ("openai_responses", {"type": "file", "file": {"file_id": "file-1"}}),
        (
            "anthropic_messages",
            {"type": "image", "image": {"source": {"type": "base64", "data": "AA=="}}},
        ),
        (
            "gemini_generate_content",
            {
                "type": "provider_native",
                "provider": "gemini",
                "kind": "inlineData",
                "value": {"inlineData": {"mimeType": "image/png", "data": "AA=="}},
            },
        ),
    ],
)
def test_recognizes_only_reviewed_normalized_modalities(codec: str, part: dict[str, object]) -> None:
    assert is_supported_normalized_modality(codec, part)


def test_unknown_provider_native_block_is_not_treated_as_a_modality() -> None:
    assert not is_supported_normalized_modality(
        "anthropic_messages",
        {
            "type": "provider_native",
            "provider": "anthropic_messages",
            "kind": "search_result",
            "value": {"content": "unchecked text"},
        },
    )


def test_reasoning_validators_accept_exact_known_shapes() -> None:
    assert validate_openai_responses_reasoning(
        {
            "type": "reasoning",
            "id": "rs-1",
            "summary": [{"type": "summary_text", "text": "summary"}],
            "content": [{"type": "reasoning_text", "text": "private"}],
            "encrypted_content": "ciphertext",
            "status": "completed",
        }
    )
    assert validate_anthropic_reasoning({"type": "thinking", "thinking": "private", "signature": "signature"})
    assert validate_anthropic_reasoning({"type": "redacted_thinking", "data": "ciphertext"})


@pytest.mark.parametrize(
    "value",
    [
        {"type": "reasoning", "id": "rs-1", "summary": [], "tool_call": {}},
        {"type": "reasoning", "id": "", "summary": []},
        {"type": "thinking", "thinking": "private", "signature": "sig", "future": "text"},
        {"type": "redacted_thinking", "data": 1},
    ],
)
def test_reasoning_validators_reject_unmodeled_or_invalid_shapes(value: object) -> None:
    validator = (
        validate_openai_responses_reasoning
        if isinstance(value, dict) and value.get("type") == "reasoning"
        else validate_anthropic_reasoning
    )
    assert not validator(value)


def test_projected_output_requires_one_candidate_for_legacy_access() -> None:
    assert ProjectedOutputText(()).one() is None
    assert ProjectedOutputText(("one",)).one() == "one"
    with pytest.raises(ValueError):
        ProjectedOutputText(("one", "two")).one()
