# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral input projection and raw-envelope coverage tests."""

from __future__ import annotations

from copy import deepcopy

import pytest
from provider_cases import chat_request, gemini_request, oci_request

from nemoguardrails_nemo_relay import codec_projection, payload_policy


def _runtime_request() -> dict[str, object]:
    return {"headers": {"x-provider": "preserved"}, "content": {"opaque": True}}


def test_host_annotation_preserves_roles_maps_developer_and_inserts_instructions() -> None:
    annotation = {
        "instructions": "top-level policy",
        "messages": [
            {"role": "system", "content": "system policy"},
            {"role": "developer", "content": "developer policy"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "prefill"},
        ],
    }

    projected = codec_projection._ProviderProjector().project_host_text(
        _runtime_request(),
        annotation,
        require_user=True,
        codec_name=None,
    )

    assert projected.messages == (
        {"role": "system", "content": "top-level policy"},
        {"role": "system", "content": "system policy"},
        {"role": "system", "content": "developer policy"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "prefill"},
    )


def test_host_annotation_joins_ordered_text_parts_without_mutating_input() -> None:
    annotation = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hel"},
                    {"type": "text", "text": "lo"},
                ],
            }
        ]
    }
    original = deepcopy(annotation)

    projected = codec_projection._ProviderProjector().project_host_text(
        _runtime_request(),
        annotation,
        require_user=True,
        codec_name=None,
    )

    assert projected.messages == ({"role": "user", "content": "hello"},)
    assert annotation == original


def test_strict_projection_rejects_normalized_multimodal_content() -> None:
    annotation = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
            }
        ]
    }

    with pytest.raises(codec_projection._UnsupportedRequest, match="non-text content"):
        codec_projection._ProviderProjector().project_host_text(
            chat_request(include_response_format=False),
            annotation,
            require_user=True,
            codec_name="openai_chat",
        )


def test_text_only_projection_checks_text_and_preserves_native_modality_outside_policy() -> None:
    annotation = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ],
            }
        ]
    }
    projector = codec_projection._ProviderProjector(
        payload_policy.PayloadPolicy(multimodal=payload_policy.MultimodalPolicy.TEXT_ONLY)
    )

    projected = projector.project_host_text(
        chat_request(include_response_format=False),
        annotation,
        require_user=True,
        codec_name="openai_chat",
    )

    assert projected.messages == ({"role": "user", "content": "describe"},)


def test_reasoning_policy_is_explicit_for_normalized_anthropic_thinking() -> None:
    annotation = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "provider_native",
                        "provider": "anthropic_messages",
                        "kind": "thinking",
                        "value": {"type": "thinking", "thinking": "private", "signature": "sig"},
                    },
                    {"type": "text", "text": "visible"},
                ],
            }
        ]
    }
    raw = {
        "headers": {"anthropic-version": "2023-06-01"},
        "content": {
            "model": "fixture",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "visible"}],
        },
    }

    with pytest.raises(codec_projection._UnsupportedRequest, match="non-text content"):
        codec_projection._ProviderProjector().project_host_text(
            raw,
            annotation,
            require_user=True,
            codec_name="anthropic_messages",
        )

    projector = codec_projection._ProviderProjector(
        payload_policy.PayloadPolicy(reasoning=payload_policy.ReasoningPolicy.FINAL_ANSWER_ONLY)
    )
    projected = projector.project_host_text(
        raw,
        annotation,
        require_user=True,
        codec_name="anthropic_messages",
    )
    assert projected.messages == ({"role": "user", "content": "visible"},)


def test_runtime_annotation_must_be_portable_and_cannot_hide_provider_state() -> None:
    projector = codec_projection._ProviderProjector()
    portable = {"messages": [{"role": "user", "content": "hello"}], "model": "fixture"}

    assert projector.project_host_text(
        _runtime_request(),
        portable,
        require_user=True,
        codec_name=None,
    ).messages == ({"role": "user", "content": "hello"},)

    for nonportable in (
        {**portable, "api_specific": {"hidden_prompt": "unchecked"}},
        {**portable, "runtime_extension": {"hidden_prompt": "unchecked"}},
    ):
        with pytest.raises(codec_projection._UnsupportedRequest):
            projector.project_host_text(
                _runtime_request(),
                nonportable,
                require_user=True,
                codec_name=None,
            )


def test_reviewed_builtin_control_and_operational_metadata_are_preserved() -> None:
    raw = chat_request(
        include_response_format=False,
        chat_template_kwargs={"enable_thinking": True},
    )
    annotation = {
        "messages": [{"role": "user", "content": "hello"}],
        "chat_template_kwargs": {"enable_thinking": True},
    }
    projector = codec_projection._ProviderProjector()

    assert projector.project_host_text(
        raw,
        {**annotation, "deployment_metadata": {"region": "west"}},
        require_user=True,
        codec_name="openai_chat",
    ).messages == ({"role": "user", "content": "hello"},)


def test_unsafe_chat_template_control_is_rejected() -> None:
    with pytest.raises(codec_projection._UnsupportedRequest, match="template controls"):
        codec_projection._ProviderProjector().project_host_text(
            chat_request(include_response_format=False),
            {
                "messages": [{"role": "user", "content": "hello"}],
                "chat_template_kwargs": {"custom_template": "unchecked"},
            },
            require_user=True,
            codec_name="openai_chat",
        )


@pytest.mark.parametrize(
    ("codec_name", "raw"),
    [
        (
            "openai_chat",
            chat_request(
                include_response_format=False,
                prediction={"type": "content", "content": "unchecked"},
            ),
        ),
    ],
)
def test_known_hidden_provider_state_cannot_hide_content_from_host_annotation(
    codec_name: str,
    raw: dict[str, object],
) -> None:
    annotation = {"messages": [{"role": "user", "content": "hello"}]}

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project_host_text(
            raw,
            annotation,
            require_user=True,
            codec_name=codec_name,
        )


@pytest.mark.parametrize(
    ("codec_name", "raw"),
    [
        (
            "gemini_generate_content",
            gemini_request([{"role": "user", "parts": [{"text": "hello", "providerMetadata": "kept"}]}]),
        ),
        (
            "oci_genai",
            oci_request(
                messages=[
                    {
                        "role": "USER",
                        "content": [{"type": "TEXT", "text": "hello", "providerMetadata": "kept"}],
                    }
                ]
            ),
        ),
    ],
)
def test_unknown_text_part_metadata_fails_closed_after_host_decode(
    codec_name: str,
    raw: dict[str, object],
) -> None:
    annotation: dict[str, object] = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello", "providerMetadata": "kept"}],
            }
        ]
    }
    if codec_name == "oci_genai":
        annotation["api_specific"] = {"api": "oci_genai", "api_format": "GENERIC"}

    with pytest.raises(codec_projection._UnsupportedRequest, match="unmodeled text-part metadata"):
        codec_projection._ProviderProjector().project_host_text(
            raw,
            annotation,
            require_user=True,
            codec_name=codec_name,
        )


def test_system_after_latest_user_still_fails_closed() -> None:
    annotation = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "late policy"},
        ]
    }

    with pytest.raises(codec_projection._UnsupportedRequest, match="system messages after"):
        codec_projection._ProviderProjector().project_host_text(
            _runtime_request(),
            annotation,
            require_user=True,
            codec_name=None,
        )
