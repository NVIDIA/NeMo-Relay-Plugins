# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from provider_cases import (
    anthropic_request,
    gemini_request,
    guardrails_chat_request,
    oci_request,
    responses_request,
)

from nemoguardrails_nemo_relay import (
    codec_projection,
)


def _request(messages: list[dict[str, object]], **content: object) -> dict[str, object]:
    return guardrails_chat_request(messages, **content)


def _anthropic_request(messages: list[dict[str, object]], **content: object) -> dict[str, object]:
    return anthropic_request(messages, max_tokens=128, **content)


def _oci_generic_request(messages: list[dict[str, object]], **chat_request: object) -> dict[str, object]:
    return oci_request(
        messages=messages,
        include_routing=True,
        **chat_request,
    )


@pytest.mark.parametrize(
    "payload",
    [
        _request([{"role": "system", "content": "policy"}, {"role": "user", "content": "hello"}]),
        responses_request([{"role": "user", "content": "hello"}], instructions="policy"),
        _anthropic_request([{"role": "user", "content": "hello"}], system="policy"),
        _oci_generic_request(
            [
                {"role": "SYSTEM", "content": [{"type": "TEXT", "text": "policy"}]},
                {"role": "USER", "content": [{"type": "TEXT", "text": "hello"}]},
            ]
        ),
        {
            "headers": {},
            "content": {
                "compartmentId": "ocid1.compartment.fixture",
                "servingMode": {"servingType": "DEDICATED", "endpointId": "ocid1.endpoint.fixture"},
                "chatRequest": {
                    "apiFormat": "COHERE",
                    "preambleOverride": "policy",
                    "chatHistory": [],
                    "message": "hello",
                },
            },
        },
        {
            "headers": {},
            "content": {
                "apiFormat": "COHEREV2",
                "messages": [
                    {"role": "SYSTEM", "content": [{"type": "TEXT", "text": "policy"}]},
                    {"role": "USER", "content": [{"type": "TEXT", "text": "hello"}]},
                ],
            },
        },
        gemini_request(
            [{"role": "user", "parts": [{"text": "hello"}]}],
            systemInstruction={"parts": [{"text": "policy"}]},
        ),
    ],
    ids=[
        "openai-chat",
        "openai-responses",
        "anthropic",
        "oci-generic",
        "oci-cohere",
        "oci-cohere-v2",
        "gemini",
    ],
)
def test_projection_normalizes_every_builtin_provider_text_shape(payload: dict[str, object]) -> None:
    original = json.loads(json.dumps(payload))

    assert codec_projection._ProviderProjector().project(payload) == [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "hello"},
    ]
    assert payload == original


def test_projection_preserves_order_tools_and_assistant_context_without_mutating_request() -> None:
    request = _request(
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "prior answer"},
            {"role": "user", "content": "follow-up"},
        ],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )
    original = json.loads(json.dumps(request))

    assert codec_projection._ProviderProjector().project(request) == request["content"]["messages"]
    assert request == original


def test_projection_maps_developer_to_system_and_joins_ordered_text_parts() -> None:
    request = _request(
        [
            {"role": "developer", "content": "policy"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hel"},
                    {"type": "text", "text": "lo"},
                ],
            },
        ]
    )

    assert codec_projection._ProviderProjector().project(request) == [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "hello"},
    ]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            responses_request(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "hel"},
                            {"type": "input_text", "text": "lo"},
                        ],
                    }
                ]
            ),
            "hello",
        ),
        (
            _anthropic_request(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hel"},
                            {"type": "text", "text": "lo"},
                        ],
                    }
                ]
            ),
            "hello",
        ),
        (
            _oci_generic_request(
                [
                    {
                        "role": "USER",
                        "content": [
                            {"type": "TEXT", "text": "hel"},
                            {"type": "TEXT", "text": "lo"},
                        ],
                    }
                ]
            ),
            "hello",
        ),
        (
            gemini_request(
                [
                    {
                        "role": "user",
                        "parts": [
                            {"text": "hel"},
                            {"text": "lo"},
                        ],
                    }
                ]
            ),
            "hel\nlo",
        ),
    ],
    ids=["openai-responses", "anthropic", "oci", "gemini"],
)
def test_projection_preserves_each_builtin_codec_multipart_text_order(
    payload: dict[str, object], expected: str
) -> None:
    assert codec_projection._ProviderProjector().project(payload) == [{"role": "user", "content": expected}]


def test_projection_includes_replayed_openai_responses_refusal_text() -> None:
    request = responses_request(
        [
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": [
                    {"type": "refusal", "refusal": "declined"},
                    {"type": "output_text", "text": "safe alternative", "annotations": []},
                ],
            },
            {"role": "user", "content": "follow-up"},
        ]
    )

    assert codec_projection._ProviderProjector().project(request) == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "declinedsafe alternative"},
        {"role": "user", "content": "follow-up"},
    ]


def test_projection_accepts_replayed_openai_responses_assistant_phase() -> None:
    request = responses_request(
        [
            {"role": "user", "content": "first"},
            {
                "id": "msg-fixture",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "working", "annotations": []}],
            },
            {"role": "user", "content": "follow-up"},
        ],
        tools=[{"type": "function", "name": "lookup", "parameters": {}}],
    )
    projector = codec_projection._ProviderProjector()

    assert projector.project(request) == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": "follow-up"},
    ]
    assert projector.project_tools(request).projection.exchanges == ()


@pytest.mark.parametrize("phase", ["analysis", "final", 1])
def test_projection_rejects_unknown_openai_responses_assistant_phase(phase: object) -> None:
    request = responses_request(
        [
            {"role": "user", "content": "first"},
            {
                "type": "message",
                "role": "assistant",
                "phase": phase,
                "content": [{"type": "output_text", "text": "working"}],
            },
            {"role": "user", "content": "follow-up"},
        ]
    )

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project(request)


@pytest.mark.parametrize("api_format", ["GENERIC", "COHEREV2"])
def test_projection_accepts_empty_oci_assistant_response_metadata(api_format: str) -> None:
    metadata: dict[str, object]
    if api_format == "GENERIC":
        metadata = {"annotations": [], "reasoningContent": None, "refusal": None}
    else:
        metadata = {"citations": [], "toolPlan": None}
    request = {
        "headers": {},
        "content": {
            "apiFormat": api_format,
            "messages": [
                {"role": "USER", "content": [{"type": "TEXT", "text": "first"}]},
                {
                    "role": "ASSISTANT",
                    "content": [{"type": "TEXT", "text": "answer"}],
                    **metadata,
                },
                {"role": "USER", "content": [{"type": "TEXT", "text": "follow-up"}]},
            ],
        },
    }

    assert codec_projection._ProviderProjector().project(request) == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow-up"},
    ]


def test_projection_accepts_systemless_anthropic_without_relying_on_headers() -> None:
    request = {
        "headers": {},
        "content": {
            "model": "claude-sonnet",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "hello"}],
        },
    }

    assert codec_projection._ProviderProjector().project(request) == [{"role": "user", "content": "hello"}]


def test_projection_accepts_anthropic_prompt_cache_metadata_without_treating_it_as_text() -> None:
    request = _anthropic_request(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
        system=[
            {
                "type": "text",
                "text": "policy",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    )

    assert codec_projection._ProviderProjector().project(request) == [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "hello"},
    ]


def test_projection_accepts_systemless_anthropic_prompt_cache_without_headers() -> None:
    request = {
        "headers": {},
        "content": {
            "model": "claude-sonnet",
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hello",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        },
    }

    assert codec_projection._ProviderProjector().project(request) == [{"role": "user", "content": "hello"}]


def test_projection_accepts_bare_oci_uppercase_roles_without_provider_hint() -> None:
    request = {
        "headers": {},
        "content": {
            "messages": [{"role": "USER", "content": [{"type": "TEXT", "text": "hello"}]}],
        },
    }

    assert codec_projection._ProviderProjector().project(request) == [{"role": "user", "content": "hello"}]


def test_projection_accepts_relay_decodable_oci_text_parts_with_lowercase_roles() -> None:
    request = {
        "headers": {},
        "content": {
            "messages": [{"role": "user", "content": [{"type": "TEXT", "text": "hello"}]}],
        },
    }

    assert codec_projection._ProviderProjector().project(request) == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_projection_accepts_openai_chat_null_message_name(role: str) -> None:
    request = _request(
        [
            {"role": "user", "content": "hello"},
            {"role": role, "content": "context", "name": None},
        ]
    )

    expected = [
        {"role": "user", "content": "hello"},
        {"role": role, "content": "context"},
    ]
    if role == "system":
        # A system message after the latest user is rejected by the separate
        # Guardrails conversation-order boundary. Put it first for this case.
        request["content"]["messages"].reverse()
        expected.reverse()
    assert codec_projection._ProviderProjector().project(request) == expected


@pytest.mark.parametrize("role", ["developer", "system", "user"])
def test_openai_chat_message_extensions_cannot_hide_prompt_text(role: str) -> None:
    request = _request([{"role": role, "content": "hello", "future_prompt": "unchecked"}])

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project(request)


def test_openai_chat_tool_calls_are_rejected_on_non_assistant_messages() -> None:
    request = _request(
        [
            {
                "role": "user",
                "content": "hello",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            }
        ]
    )

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project(request)


def test_projection_accepts_validated_codex_responses_client_metadata() -> None:
    request = responses_request(
        "hello",
        client_metadata={"x-codex-installation-id": "installation-fixture"},
    )

    assert codec_projection._ProviderProjector().project(request) == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"x-codex-installation-id": ""},
        {"x-codex-installation-id": 7},
        {"x-codex-installation-id": "fixture", "future_prompt": "unchecked"},
    ],
)
def test_projection_rejects_unrecognized_responses_client_metadata(metadata: object) -> None:
    request = responses_request("hello", client_metadata=metadata)

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project(request)


def test_codex_responses_tool_result_remains_unsupported() -> None:
    request = responses_request(
        [
            {
                "type": "function_call_output",
                "name": "private_tool",
                "namespace": "fixture",
                "output": "private result",
            },
            {"role": "user", "content": "hello"},
        ],
        client_metadata={"x-codex-installation-id": "installation-fixture"},
    )

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project(request)


@pytest.mark.parametrize("role", ["SYSTEM", "USER", "ASSISTANT"])
@pytest.mark.parametrize(
    "structural_field",
    [
        {"toolCalls": [{"id": "call-1", "type": "FUNCTION", "name": "private_tool", "arguments": {}}]},
        {"toolCallId": "private-correlation"},
    ],
)
def test_oci_structural_fields_never_disappear_behind_text_projection(
    role: str,
    structural_field: dict[str, object],
) -> None:
    request = _oci_generic_request(
        [
            {
                "role": role,
                "content": [{"type": "TEXT", "text": "hello"}],
                **structural_field,
            }
        ]
    )

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project(request)


@pytest.mark.parametrize(
    "payload",
    [
        _request([{"role": "user", "content": "hello"}], response_format={"type": "json_object"}),
        {
            "headers": {},
            "content": {
                "model": "claude-sonnet",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hello"}],
                "thinking": {"type": "enabled", "budget_tokens": 1024},
            },
        },
    ],
    ids=["openai-chat-specific-control", "anthropic-specific-control"],
)
def test_provider_specific_markers_select_one_complete_builtin_codec(payload: dict[str, object]) -> None:
    assert codec_projection._ProviderProjector().project(payload) == [{"role": "user", "content": "hello"}]


def test_projection_accepts_relay_openai_chat_only_controls() -> None:
    controls: dict[str, object] = {
        "audio": {},
        "frequency_penalty": 0.0,
        "function_call": "none",
        "functions": [],
        "logit_bias": {},
        "logprobs": False,
        "max_completion_tokens": 32,
        "modalities": ["text"],
        "moderation": {},
        "n": 1,
        "parallel_tool_calls": False,
        "presence_penalty": 0.0,
        "prompt_cache_key": "cache-key",
        "prompt_cache_options": {},
        "prompt_cache_retention": "24h",
        "reasoning_effort": "low",
        "response_format": {"type": "text"},
        "safety_identifier": "safety-id",
        "seed": 7,
        "stop": ["END"],
        "store": False,
        "stream_options": {"include_usage": True},
        "top_logprobs": 1,
        "user": "user-id",
        "verbosity": "low",
        "web_search_options": {},
    }
    request = _request([{"role": "user", "content": "hello"}], **controls)

    assert codec_projection._ProviderProjector().project(request) == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize("enabled", [False, True])
def test_projection_accepts_nvidia_thinking_control_without_treating_it_as_prompt(enabled: bool) -> None:
    request = _request(
        [{"role": "user", "content": "hello"}],
        chat_template_kwargs={"enable_thinking": enabled},
    )

    assert codec_projection._ProviderProjector().project(request) == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize(
    "template_kwargs",
    [
        None,
        {"enable_thinking": "false"},
        {"enable_thinking": False, "system_prompt": "unchecked"},
        {"future_prompt": "unchecked"},
    ],
)
def test_projection_rejects_unreviewed_chat_template_controls(template_kwargs: object) -> None:
    request = _request(
        [{"role": "user", "content": "hello"}],
        chat_template_kwargs=template_kwargs,
    )

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project(request)


def test_projection_accepts_relay_anthropic_only_controls() -> None:
    request = _anthropic_request(
        [{"role": "user", "content": "hello"}],
        **{
            "anthropic-user-profile-id": "profile-id",
            "cache_control": {"type": "ephemeral"},
            "container": None,
            "inference_geo": "us",
            "output_config": {},
            "stop_sequences": ["END"],
            "system": "policy",
            "thinking": {"type": "disabled"},
            "top_k": 4,
        },
    )

    assert codec_projection._ProviderProjector().project(request) == [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "hello"},
    ]


def test_projection_rejects_mixed_openai_chat_and_anthropic_controls() -> None:
    request = _request(
        [{"role": "user", "content": "hello"}],
        stop=["OPENAI"],
        stop_sequences=["ANTHROPIC"],
    )

    with pytest.raises(codec_projection._UnsupportedRequest, match="mixes OpenAI Chat and Anthropic"):
        codec_projection._ProviderProjector().project(request)


def test_ambiguous_wire_projection_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    projector = codec_projection._ProviderProjector()
    monkeypatch.setattr(
        codec_projection.anthropic_adapter,
        "request_annotation",
        lambda _content: (_ for _ in ()).throw(codec_projection._UnsupportedRequest("unsupported")),
    )

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project(_request([{"role": "user", "content": "hello"}]))


@pytest.mark.parametrize(
    "payload",
    [
        {"headers": {}, "content": []},
        _request([]),
        _request([{"role": "system", "content": "only system"}]),
        _request([{"role": "tool", "content": "result"}, {"role": "user", "content": "hi"}]),
        _request([{"role": "function", "content": "result"}, {"role": "user", "content": "hi"}]),
        _request(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    ],
                }
            ]
        ),
        _request([{"role": "user", "content": None}]),
        _request([{"role": "user", "content": "hi", "name": "alex"}]),
        _request([{"role": "user", "content": "hi"}], input="responses-shape"),
        _request([{"role": "user", "content": "hi"}], contents=[]),
        _request([{"role": "user", "content": "hi"}], provider_prompt="unchecked custom codec prompt"),
        _request(
            [{"role": "user", "content": "hello"}],
            prediction={"type": "content", "content": "unchecked predicted output"},
        ),
        _request(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello", "futurePrompt": "unchecked nested text"}],
                }
            ]
        ),
        responses_request("hello", previous_response_id="response_with_hidden_history"),
        responses_request("hello", conversation="conversation_with_hidden_history"),
        responses_request("hello", prompt={"id": "provider_prompt_template"}),
        responses_request("hello", context_management=[{"type": "compaction"}]),
        _anthropic_request([{"role": "user", "content": "hello"}], container="container_with_external_state"),
        {
            "headers": {},
            "content": {
                "apiFormat": "FUTURE",
                "messages": [{"role": "USER", "content": [{"type": "TEXT", "text": "hello"}]}],
            },
        },
        gemini_request(
            [
                {
                    "role": "user",
                    "parts": [
                        {"text": "hidden", "thought": True},
                        {"text": "hello"},
                    ],
                }
            ]
        ),
        gemini_request([{"role": "user", "parts": [{"text": "hello", "thoughtSignature": "opaque-state"}]}]),
        gemini_request([{"role": "user", "parts": [{"text": "hello", "futurePrompt": "unchecked"}]}]),
        gemini_request(
            [{"role": "user", "parts": [{"text": "hello"}]}],
            systemInstruction={"parts": [{"text": "policy"}], "futurePrompt": "unchecked"},
        ),
        gemini_request(
            [{"role": "user", "parts": [{"text": "hello"}]}],
            systemInstruction={"parts": [{"text": "policy", "futurePrompt": "unchecked"}]},
        ),
        gemini_request(
            [{"role": "user", "parts": [{"text": "hello"}], "futurePrompt": "unchecked"}],
        ),
        gemini_request([{"role": "user", "parts": [{"text": "hello"}]}], cachedContent="cachedContents/fixture"),
        {
            "headers": {},
            "content": {
                "compartmentId": "ocid1.compartment.fixture",
                "servingMode": {"servingType": "DEDICATED", "endpointId": "ocid1.endpoint.fixture"},
                "futureEnvelopePrompt": "unchecked",
                "chatRequest": {
                    "apiFormat": "GENERIC",
                    "messages": [{"role": "USER", "content": [{"type": "TEXT", "text": "hello"}]}],
                },
            },
        },
        {
            "headers": {},
            "content": {
                "apiFormat": "GENERIC",
                "messages": [{"role": "USER", "content": "hello", "futurePrompt": "unchecked"}],
            },
        },
        {
            "headers": {},
            "content": {
                "apiFormat": "GENERIC",
                "messages": [
                    {
                        "role": "USER",
                        "content": [{"type": "TEXT", "text": "hello", "futurePrompt": "unchecked"}],
                    }
                ],
            },
        },
        {
            "headers": {},
            "content": {
                "apiFormat": "COHERE",
                "chatHistory": [{"role": "USER", "message": "prior", "futurePrompt": "unchecked"}],
                "message": "hello",
            },
        },
        _request([{"role": "user", "content": ""}]),
        _request([{"role": "user", "content": " \n\t"}]),
        _request([{"role": "user", "content": "first"}, {"role": "user", "content": " "}]),
        _request([{"role": "system", "content": "policy"}]),
        _request([{"role": "assistant", "content": "answer"}]),
        _request([{"role": "user", "content": "hello"}, {"role": "system", "content": "late policy"}]),
    ],
)
def test_projection_rejects_unsupported_shapes(payload: dict[str, object]) -> None:
    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project(payload)


def test_projection_accepts_assistant_context_after_the_latest_user() -> None:
    request = _request(
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "prefill"},
        ]
    )

    assert codec_projection._ProviderProjector().project(request) == request["content"]["messages"]


def test_projection_accepts_empty_responses_context_management() -> None:
    request = responses_request(
        [{"role": "user", "content": "hello"}],
        context_management=[],
    )

    assert codec_projection._ProviderProjector().project(request) == [{"role": "user", "content": "hello"}]


_ECHO_LATEST_USER = object()
