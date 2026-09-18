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


@pytest.mark.parametrize("protocol", ["oci_genai", "oci_cohere", "oci_cohere_v2"])
def test_output_projection_rejects_unknown_oci_chat_response_siblings(protocol: str) -> None:
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
    response["chatResponse"]["futureExplanation"] = "UNCHECKED_CANARY"  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(request, require_user=True)

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_text(decoded, response, allow_tool_calls=allow_tool_calls)


def test_anthropic_output_rejects_stop_details_explanation() -> None:
    request, response = combined_protocol_case("anthropic_messages")
    response["stop_details"] = {
        "type": "refusal",
        "category": "safety",
        "explanation": "UNCHECKED_CANARY",
    }
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(request, require_user=True)

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_text(decoded, response, allow_tool_calls=True)


@pytest.mark.parametrize(
    "metadata",
    [
        {"thought": False},
        {"thoughtSignature": ""},
        {"partMetadata": {}},
    ],
)
def test_gemini_output_checks_all_text_when_empty_metadata_forces_normalized_parts(
    metadata: dict[str, object],
) -> None:
    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "one", **metadata}, {"text": "two"}],
                }
            }
        ]
    }
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(gemini_request(), require_user=True)

    assert projector.project_response_text(decoded, response, allow_tool_calls=False) == "one\ntwo"


def test_openai_responses_output_preserves_ordered_refusal_and_text_for_guardrails() -> None:
    response = {
        "id": "resp-fixture",
        "object": "response",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "refusal", "refusal": "declined"},
                    {
                        "type": "output_text",
                        "text": "safe alternative",
                        "annotations": [],
                        "logprobs": [],
                    },
                ],
            }
        ],
        # The Responses aggregate contains output_text blocks, not refusals.
        "output_text": "safe alternative",
    }
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(responses_request(), require_user=True)

    assert projector.project_response_text(decoded, response, allow_tool_calls=False) == "declined\nsafe alternative"


@pytest.mark.parametrize("phase", ["commentary", "final_answer"])
def test_openai_responses_output_accepts_message_phase(phase: str) -> None:
    response = {
        "id": "resp-fixture",
        "object": "response",
        "output": [
            {
                "id": "msg-fixture",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": phase,
                "content": [{"type": "output_text", "text": "safe", "annotations": []}],
            }
        ],
        "output_text": "safe",
    }
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(responses_request(), require_user=True)

    assert projector.project_response_text(decoded, response, allow_tool_calls=False) == "safe"


@pytest.mark.parametrize("aggregate", ["", None])
def test_openai_responses_refusal_accepts_only_an_empty_aggregate(
    aggregate: str | None,
) -> None:
    response: dict[str, object] = {
        "id": "resp-fixture",
        "object": "response",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "declined"}],
            }
        ],
    }
    if aggregate is not None:
        response["output_text"] = aggregate
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(responses_request(), require_user=True)

    assert projector.project_response_text(decoded, response, allow_tool_calls=False) == "declined"


def test_openai_responses_refusal_rejects_unrepresented_aggregate_text() -> None:
    response = {
        "id": "resp-fixture",
        "object": "response",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "declined"}],
            }
        ],
        "output_text": "unchecked application text",
    }
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(responses_request(), require_user=True)

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_text(decoded, response, allow_tool_calls=False)


@pytest.mark.parametrize("api_format", ["GENERIC", "COHERE", "COHEREV2"])
def test_oci_output_accepts_nullable_tool_call_arrays(api_format: str) -> None:
    if api_format == "COHERE":
        response = {
            "chatResponse": {
                "apiFormat": api_format,
                "text": "safe",
                "toolCalls": None,
            }
        }
    elif api_format == "GENERIC":
        response = {
            "chatResponse": {
                "apiFormat": api_format,
                "choices": [
                    {
                        "message": {
                            "role": "ASSISTANT",
                            "content": [{"type": "TEXT", "text": "safe"}],
                            "toolCalls": None,
                        }
                    }
                ],
            }
        }
    else:
        response = {
            "chatResponse": {
                "apiFormat": api_format,
                "message": {
                    "role": "ASSISTANT",
                    "content": [{"type": "TEXT", "text": "safe"}],
                    "toolCalls": None,
                },
            }
        }
    projector = codec_projection._NativeCodecProjector()
    text_request = projector.project_text(oci_request(api_format), require_user=True)
    tool_request = projector.project_tools(oci_request(api_format), require_definitions=False)

    assert projector.project_response_text(text_request, response, allow_tool_calls=True) == "safe"
    assert projector.project_response_tools(tool_request, response) == ()


def test_openai_responses_replayed_assistant_text_accepts_empty_provider_metadata() -> None:
    request = responses_request()
    request["content"]["input"] = [  # type: ignore[index]
        {"role": "user", "content": [{"type": "input_text", "text": "first"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "prior answer",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        },
        {"role": "user", "content": [{"type": "input_text", "text": "follow-up"}]},
    ]

    assert codec_projection._NativeCodecProjector().project(request) == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "prior answer"},
        {"role": "user", "content": "follow-up"},
    ]


def test_openai_chat_replayed_assistant_accepts_empty_provider_metadata() -> None:
    request = chat_request(
        [
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": "prior answer",
                "refusal": None,
                "audio": None,
                "annotations": [],
            },
            {"role": "user", "content": "follow-up"},
        ]
    )

    assert codec_projection._NativeCodecProjector().project(request) == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "prior answer"},
        {"role": "user", "content": "follow-up"},
    ]


@pytest.mark.parametrize("citations", [None, []])
def test_anthropic_replayed_text_accepts_empty_citations(citations: object) -> None:
    request = anthropic_request()
    request["content"]["messages"] = [  # type: ignore[index]
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "prior answer", "citations": citations}],
        },
        {"role": "user", "content": "follow-up"},
    ]

    assert codec_projection._NativeCodecProjector().project(request) == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "prior answer"},
        {"role": "user", "content": "follow-up"},
    ]


def test_anthropic_replayed_text_rejects_nonempty_citations() -> None:
    request = anthropic_request()
    request["content"]["messages"] = [  # type: ignore[index]
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "prior answer",
                    "citations": [{"type": "web_search_result_location", "title": "unchecked"}],
                }
            ],
        },
        {"role": "user", "content": "follow-up"},
    ]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project(request)


def test_openai_chat_replayed_assistant_rejects_audio_output() -> None:
    request = chat_request(
        [
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": "prior answer",
                "refusal": None,
                "audio": {"id": "audio-private"},
                "annotations": [],
            },
            {"role": "user", "content": "follow-up"},
        ]
    )

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project(request)


@pytest.mark.parametrize("logprobs", [None, [], {}])
def test_openai_chat_output_accepts_empty_logprobs(logprobs: object) -> None:
    response = chat_response()
    response["choices"][0]["logprobs"] = logprobs  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(chat_request(), require_user=True)

    assert projector.project_response_text(decoded, response, allow_tool_calls=False) == "safe"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "logprobsResult",
            {
                "chosenCandidates": [{"token": "123-45-6789", "tokenId": 1, "logProbability": -0.1}],
                "topCandidates": [],
            },
        ),
        ("finishMessage", "unchecked provider diagnostic"),
        (
            "groundingAttributions",
            [{"content": {"role": "model", "parts": [{"text": "unchecked source text"}]}}],
        ),
    ],
)
def test_gemini_output_rejects_unchecked_candidate_metadata(field: str, value: object) -> None:
    response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "safe"}]},
                "finishReason": "STOP",
                field: value,
            }
        ]
    }
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(gemini_request(), require_user=True)

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_text(decoded, response, allow_tool_calls=False)


def test_gemini_output_accepts_empty_candidate_metadata() -> None:
    response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "safe"}]},
                "finishReason": "STOP",
                "finishMessage": "",
                "groundingAttributions": [],
                "logprobsResult": {},
            }
        ]
    }
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(gemini_request(), require_user=True)

    assert projector.project_response_text(decoded, response, allow_tool_calls=False) == "safe"


def test_gemini_output_rejects_unchecked_model_status_message() -> None:
    response = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "safe"}]}}],
        "modelStatus": {"modelStage": "STABLE", "message": "unchecked provider notice"},
    }
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(gemini_request(), require_user=True)

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_text(decoded, response, allow_tool_calls=False)


def test_gemini_output_accepts_model_status_without_visible_message() -> None:
    response = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "safe"}]}}],
        "modelStatus": {
            "modelStage": "STABLE",
            "retirementTime": "2030-01-01T00:00:00Z",
            "message": "",
        },
    }
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(gemini_request(), require_user=True)

    assert projector.project_response_text(decoded, response, allow_tool_calls=False) == "safe"


def test_gemini_input_accepts_empty_thought_metadata() -> None:
    request = gemini_request()
    request["content"]["contents"][0]["parts"][0].update(  # type: ignore[index,union-attr]
        {"partMetadata": {}, "thought": False, "thoughtSignature": ""}
    )

    assert codec_projection._NativeCodecProjector().project(request) == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("thought", True),
        ("thoughtSignature", "opaque-signature"),
        ("partMetadata", {"private": "unchecked"}),
    ],
)
def test_gemini_input_rejects_nonempty_thought_metadata(field: str, value: object) -> None:
    request = gemini_request()
    request["content"]["contents"][0]["parts"][0][field] = value  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project(request)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chatHistory", [{"role": "CHATBOT", "message": "unchecked history"}]),
        ("prompt", "unchecked full prompt"),
        ("errorMessage", "unchecked provider error"),
    ],
)
def test_oci_cohere_output_rejects_unchecked_response_metadata(field: str, value: object) -> None:
    response = {
        "chatResponse": {
            "apiFormat": "COHERE",
            "text": "safe",
            "finishReason": "COMPLETE",
            field: value,
        }
    }
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(oci_request("COHERE"), require_user=True)

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_text(decoded, response, allow_tool_calls=False)


def test_oci_cohere_output_accepts_empty_response_metadata() -> None:
    response = {
        "chatResponse": {
            "apiFormat": "COHERE",
            "text": "safe",
            "finishReason": "COMPLETE",
            "chatHistory": [],
            "prompt": "",
            "errorMessage": "",
        }
    }
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(oci_request("COHERE"), require_user=True)

    assert projector.project_response_text(decoded, response, allow_tool_calls=False) == "safe"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("logProbabilities", [{"token": "123-45-6789", "logProbability": -0.1}]),
        ("errorMessage", "unchecked provider error"),
    ],
)
def test_oci_cohere_v2_output_rejects_unchecked_response_metadata(field: str, value: object) -> None:
    response = {
        "chatResponse": {
            "apiFormat": "COHEREV2",
            "id": "chat-fixture",
            "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "safe"}]},
            "finishReason": "COMPLETE",
            field: value,
        }
    }
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(oci_request("COHEREV2"), require_user=True)

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_text(decoded, response, allow_tool_calls=False)


def test_oci_cohere_v2_output_accepts_empty_response_metadata() -> None:
    response = {
        "chatResponse": {
            "apiFormat": "COHEREV2",
            "id": "chat-fixture",
            "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "safe"}]},
            "finishReason": "COMPLETE",
            "logProbabilities": [],
            "errorMessage": "",
        }
    }
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_text(oci_request("COHEREV2"), require_user=True)

    assert projector.project_response_text(decoded, response, allow_tool_calls=False) == "safe"
