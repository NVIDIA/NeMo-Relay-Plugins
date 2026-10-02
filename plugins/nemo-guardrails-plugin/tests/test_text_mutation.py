# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest

from nemoguardrails_nemo_relay.payload_policy import PayloadPolicy, ReasoningPolicy
from nemoguardrails_nemo_relay.provider_coverage import inspect_response
from nemoguardrails_nemo_relay.text_mutation import (
    MutationMode,
    MutationPolicy,
    TextMutationError,
    mutation_policy_from_config,
    rewrite_annotated_request_text,
    rewrite_response_texts,
)


def _inspection(codec: str, variant: str | None, response: object):
    return inspect_response(
        codec,
        variant,
        response,
        PayloadPolicy(reasoning=ReasoningPolicy.FINAL_ANSWER_ONLY),
    )


def test_mutation_policy_rejects_by_default_and_parses_each_phase() -> None:
    assert mutation_policy_from_config(None) == MutationPolicy()
    assert mutation_policy_from_config({"input": "apply", "output": "apply"}) == MutationPolicy(
        input=MutationMode.APPLY,
        output=MutationMode.APPLY,
    )


@pytest.mark.parametrize(
    "value",
    [[], {"future": "apply"}, {"input": "pass_original"}, {"output": True}],
)
def test_mutation_policy_rejects_unknown_shapes(value: object) -> None:
    with pytest.raises(ValueError, match="mutation_policy"):
        mutation_policy_from_config(value)


def test_request_rewrite_changes_only_the_normalized_latest_user_leaf() -> None:
    annotated = {
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "prior"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "original"},
                    {"type": "image", "image": {"url": "fixture"}},
                ],
            },
        ],
        "model": "fixture",
        "provider_extension": {"preserve": True},
    }
    original = deepcopy(annotated)

    rewritten = rewrite_annotated_request_text(
        annotated,
        expected="original",
        replacement="replacement",
        max_text_characters=1_000,
    )

    assert annotated == original
    expected = deepcopy(original)
    expected["messages"][2]["content"][0]["text"] = "replacement"  # type: ignore[index]
    assert rewritten == expected


@pytest.mark.parametrize(
    "payload",
    [
        {
            "headers": {},
            "content": {
                "model": "fixture",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "orig"},
                            {"type": "text", "text": "inal"},
                        ],
                    }
                ],
            },
        },
        {
            "headers": {},
            "content": {"model": "fixture", "messages": [{"role": "assistant", "content": "original"}]},
        },
    ],
)
def test_request_rewrite_rejects_ambiguous_or_missing_user_leaf(payload: dict[str, object]) -> None:
    with pytest.raises(TextMutationError):
        rewrite_annotated_request_text(
            payload,
            expected="original",
            replacement="replacement",
            max_text_characters=1_000,
        )


def test_request_rewrite_rejects_size_violation() -> None:
    request = {"messages": [{"role": "user", "content": "original"}]}
    with pytest.raises(TextMutationError, match="size"):
        rewrite_annotated_request_text(
            request,
            expected="original",
            replacement="too long",
            max_text_characters=2,
        )


@pytest.mark.parametrize(
    ("codec", "variant", "response", "expected", "replacements", "changed_paths"),
    [
        (
            "openai_chat",
            None,
            {
                "id": "chatcmpl-1",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "first"}, "finish_reason": "stop"},
                    {"index": 1, "message": {"role": "assistant", "refusal": "second"}, "finish_reason": "stop"},
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            },
            ("first", "second"),
            ("one", "two"),
            (("choices", 0, "message", "content"), ("choices", 1, "message", "refusal")),
        ),
        (
            "openai_responses",
            None,
            {
                "id": "resp-1",
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "first", "annotations": []}],
                    }
                ],
                "output_text": "first",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            ("first",),
            ("one",),
            (("output", 0, "content", 0, "text"), ("output_text",)),
        ),
        (
            "anthropic_messages",
            None,
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private", "signature": "sig"},
                    {"type": "text", "text": "first", "citations": []},
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            ("first",),
            ("one",),
            (("content", 1, "text"),),
        ),
        (
            "gemini_generate_content",
            None,
            {
                "candidates": [
                    {"index": 0, "content": {"role": "model", "parts": [{"text": "first"}]}},
                    {
                        "index": 1,
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "private", "thought": True, "thoughtSignature": "sig"},
                                {"text": "second"},
                            ],
                        },
                    },
                ],
                "usageMetadata": {"totalTokenCount": 3},
            },
            ("first", "second"),
            ("one", "two"),
            (("candidates", 0, "content", "parts", 0, "text"), ("candidates", 1, "content", "parts", 1, "text")),
        ),
        (
            "oci_genai",
            "GENERIC",
            {
                "chatResponse": {
                    "apiFormat": "GENERIC",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "first"}]},
                        },
                        {"index": 1, "message": {"role": "ASSISTANT", "content": "second"}},
                    ],
                },
                "opcRequestId": "keep",
            },
            ("first", "second"),
            ("one", "two"),
            (
                ("chatResponse", "choices", 0, "message", "content", 0, "text"),
                ("chatResponse", "choices", 1, "message", "content"),
            ),
        ),
        (
            "oci_genai",
            "COHERE",
            {"chatResponse": {"apiFormat": "COHERE", "text": "first", "finishReason": "COMPLETE"}},
            ("first",),
            ("one",),
            (("chatResponse", "text"),),
        ),
        (
            "oci_genai",
            "COHEREV2",
            {
                "chatResponse": {
                    "apiFormat": "COHEREV2",
                    "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "first"}]},
                }
            },
            ("first",),
            ("one",),
            (("chatResponse", "message", "content", 0, "text"),),
        ),
    ],
)
def test_response_rewrite_is_atomic_and_preserves_every_other_field(
    codec: str,
    variant: str | None,
    response: dict[str, object],
    expected: tuple[str, ...],
    replacements: tuple[str, ...],
    changed_paths: tuple[tuple[str | int, ...], ...],
) -> None:
    original = deepcopy(response)

    rewritten = rewrite_response_texts(
        response,
        inspection=_inspection(codec, variant, response),
        expected=expected,
        replacements=replacements,
        max_text_characters=1_000,
    )

    assert response == original
    wanted = deepcopy(original)
    changed_values = (
        replacements if len(changed_paths) == len(replacements) else (replacements[0],) * len(changed_paths)
    )
    for path, value in zip(changed_paths, changed_values, strict=True):
        cursor: object = wanted
        for part in path[:-1]:
            cursor = cursor[part]  # type: ignore[index]
        cursor[path[-1]] = value  # type: ignore[index]
    assert rewritten == wanted


def test_response_rewrite_preserves_tool_and_reasoning_payloads() -> None:
    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "private reasoning", "thought": True, "thoughtSignature": "sig"},
                        {"text": "safe"},
                        {"functionCall": {"name": "weather", "args": {"city": "Paris"}}},
                    ],
                }
            }
        ]
    }
    rewritten = rewrite_response_texts(
        response,
        inspection=_inspection("gemini_generate_content", None, response),
        expected=("safe",),
        replacements=("redacted",),
        max_text_characters=1_000,
    )
    assert rewritten["candidates"][0]["content"]["parts"] == [  # type: ignore[index]
        {"text": "private reasoning", "thought": True, "thoughtSignature": "sig"},
        {"text": "redacted"},
        {"functionCall": {"name": "weather", "args": {"city": "Paris"}}},
    ]


@pytest.mark.parametrize(
    ("response", "expected", "replacements", "match"),
    [
        (
            {
                "content": [
                    {"type": "text", "text": "one"},
                    {"type": "text", "text": "two"},
                ]
            },
            ("one\ntwo",),
            ("safe",),
            "one exact native text leaf",
        ),
        (
            {"choices": [{"message": {"role": "assistant", "content": "one"}}]},
            ("different",),
            ("safe",),
            "does not match",
        ),
        (
            {"choices": [{"message": {"role": "assistant", "content": "one"}}]},
            ("one",),
            ("first", "second"),
            "counts",
        ),
    ],
)
def test_response_rewrite_rejects_non_lossless_shapes(
    response: dict[str, object],
    expected: tuple[str, ...],
    replacements: tuple[str, ...],
    match: str,
) -> None:
    codec = "anthropic_messages" if "content" in response else "openai_chat"
    with pytest.raises(TextMutationError, match=match):
        rewrite_response_texts(
            response,
            inspection=_inspection(codec, None, response),
            expected=expected,
            replacements=replacements,
            max_text_characters=1_000,
        )


def test_response_rewrite_rejects_excessive_total_size() -> None:
    response = {"choices": [{"message": {"role": "assistant", "content": "one"}}]}
    with pytest.raises(TextMutationError, match="size"):
        rewrite_response_texts(
            response,
            inspection=_inspection("openai_chat", None, response),
            expected=("one",),
            replacements=("too long",),
            max_text_characters=2,
        )


@pytest.mark.parametrize(
    ("codec", "variant", "response"),
    [
        (
            "openai_chat",
            None,
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "one",
                            "annotations": [{"type": "url_citation", "text": "one"}],
                        }
                    }
                ]
            },
        ),
        (
            "openai_responses",
            None,
            {"output_text": "one", "providerMetadata": {"transcript": "one"}},
        ),
        (
            "openai_responses",
            None,
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "refusal",
                                "refusal": "one",
                                "annotations": [{"text": "one"}],
                            }
                        ],
                    }
                ]
            },
        ),
        (
            "anthropic_messages",
            None,
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "one", "citations": [{"cited_text": "one"}]}],
            },
        ),
        (
            "gemini_generate_content",
            None,
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "one"}]},
                        "logprobsResult": {"chosenCandidates": [{"token": "one"}]},
                    }
                ]
            },
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
                                "citations": [{"text": "one"}],
                                "content": [
                                    {
                                        "type": "TEXT",
                                        "text": "one",
                                    }
                                ],
                            }
                        }
                    ],
                }
            },
        ),
    ],
    ids=[
        "chat-annotations",
        "responses-aggregate",
        "responses-refusal",
        "anthropic-citations",
        "gemini-logprobs",
        "oci-citations",
    ],
)
def test_response_rewrite_rejects_text_bearing_metadata(
    codec: str,
    variant: str | None,
    response: dict[str, object],
) -> None:
    with pytest.raises(TextMutationError, match="text-bearing metadata"):
        rewrite_response_texts(
            response,
            inspection=_inspection(codec, variant, response),
            expected=("one",),
            replacements=("redacted",),
            max_text_characters=1_000,
        )
