# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts between Relay-owned request decoding and plugin-owned tool coverage."""

from __future__ import annotations

from copy import deepcopy

import pytest
from projection_fixtures import (
    decoded_text,
    decoded_tools,
    response_text,
    response_tools,
    weather_annotation,
    weather_exchange,
)
from tool_projection_cases import (
    PROTOCOLS,
    check_calls,
    check_results,
    protocol_case,
    response_with_text,
)

from nemoguardrails_nemo_relay import codec_projection
from nemoguardrails_nemo_relay.structural_tools import ToolVerdictKind


def _project_tools(protocol: str, request: dict[str, object]) -> codec_projection._DecodedToolRequest:
    return codec_projection._ProviderProjector().project_host_tools(
        request,
        weather_annotation(protocol),
        codec_name=protocol,
    )


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_host_annotations_project_one_structural_contract_for_every_builtin(protocol: str) -> None:
    request, response = protocol_case(protocol)
    projector = codec_projection._ProviderProjector()

    decoded = _project_tools(protocol, request)
    calls = response_tools(projector, decoded, response)

    assert decoded.codec_name == protocol
    assert decoded.projection.definitions[0].name == "weather"
    exchange = decoded.projection.exchanges[0]
    assert exchange.calls[0].call_id == "call-1"
    assert exchange.calls[0].arguments == {"city": "Paris"}
    assert exchange.results[0].call_id == "call-1"
    assert calls[0].call_id == "call-2"


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_projected_contract_passes_public_guardrails_checks(protocol: str) -> None:
    request, response = protocol_case(protocol)
    projector = codec_projection._ProviderProjector()
    decoded = _project_tools(protocol, request)
    calls = response_tools(projector, decoded, response)

    assert (await check_results(decoded.projection.exchanges)).kind is ToolVerdictKind.PASSED
    assert (await check_calls(decoded.projection.definitions, calls)).kind is ToolVerdictKind.PASSED


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_raw_response_covers_text_and_function_calls_together(protocol: str) -> None:
    _, response = protocol_case(protocol)
    response = response_with_text(protocol, response, "safe output")
    projector = codec_projection._ProviderProjector()

    assert (
        response_text(
            projector,
            decoded_text(protocol, codec_variant="GENERIC" if protocol == "oci_genai" else None),
            response,
            allow_tool_calls=True,
        )
        == "safe output"
    )
    assert (
        response_tools(
            projector,
            decoded_tools(protocol, codec_variant="GENERIC" if protocol == "oci_genai" else None),
            response,
        )[0].name
        == "weather"
    )


@pytest.mark.parametrize(
    ("protocol", "invalid_role"),
    [
        ("openai_chat", "user"),
        ("anthropic_messages", "user"),
        ("gemini_generate_content", "user"),
        ("oci_genai", "USER"),
    ],
)
def test_response_function_calls_require_provider_assistant_role(
    protocol: str,
    invalid_role: str,
) -> None:
    _, response = protocol_case(protocol)
    broken = deepcopy(response)
    if protocol == "openai_chat":
        broken["choices"][0]["message"]["role"] = invalid_role  # type: ignore[index]
    elif protocol == "anthropic_messages":
        broken["role"] = invalid_role
    elif protocol == "gemini_generate_content":
        broken["candidates"][0]["content"]["role"] = invalid_role  # type: ignore[index]
    else:
        broken["chatResponse"]["choices"][0]["message"]["role"] = invalid_role  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools(protocol, codec_variant="GENERIC" if protocol == "oci_genai" else None),
            broken,
        )


@pytest.mark.parametrize(
    ("protocol", "response"),
    [
        (
            "openai_chat",
            {"id": "chatcmpl-empty", "object": "chat.completion", "choices": []},
        ),
        (
            "gemini_generate_content",
            {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}},
        ),
        (
            "oci_genai",
            {"chatResponse": {"apiFormat": "GENERIC", "choices": []}},
        ),
    ],
)
def test_empty_candidates_do_not_create_false_tool_calls(
    protocol: str,
    response: dict[str, object],
) -> None:
    assert (
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools(protocol, codec_variant="GENERIC" if protocol == "oci_genai" else None),
            response,
        )
        == ()
    )


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_unknown_operational_annotation_metadata_does_not_change_tool_projection(protocol: str) -> None:
    request, _ = protocol_case(protocol)
    annotation = weather_annotation(protocol)
    annotation["deployment_metadata"] = {"region": "test"}

    projection = codec_projection._ProviderProjector().project_host_tools(
        request,
        annotation,
        codec_name=protocol,
    )

    assert projection.projection.definitions[0].name == "weather"


@pytest.mark.parametrize("protocol", ["openai_chat", "gemini_generate_content", "oci_genai"])
def test_multiple_tool_candidates_are_projected_independently(protocol: str) -> None:
    _, response = protocol_case(protocol)
    multiple = deepcopy(response)
    if protocol == "openai_chat":
        multiple["choices"].append(deepcopy(multiple["choices"][0]))  # type: ignore[index,union-attr]
    elif protocol == "gemini_generate_content":
        multiple["candidates"].append(deepcopy(multiple["candidates"][0]))  # type: ignore[index,union-attr]
    else:
        chat_response = multiple["chatResponse"]
        chat_response["choices"].append(deepcopy(chat_response["choices"][0]))  # type: ignore[index,union-attr]

    projected = codec_projection._ProviderProjector().project_response_tool_candidates(
        decoded_tools(protocol, codec_variant="GENERIC" if protocol == "oci_genai" else None),
        multiple,
        response_codec_name=protocol,
    )
    assert len(projected) == 2


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses"])
def test_duplicate_json_argument_keys_are_rejected_from_raw_envelopes(protocol: str) -> None:
    request, response = protocol_case(protocol)
    if protocol == "openai_chat":
        request["content"]["messages"][1]["tool_calls"][0]["function"]["arguments"] = (  # type: ignore[index]
            '{"city":"Paris","city":"Rome"}'
        )
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (  # type: ignore[index]
            '{"city":"Paris","city":"Rome"}'
        )
    else:
        request["content"]["input"][1]["arguments"] = '{"city":"Paris","city":"Rome"}'  # type: ignore[index]
        response["output"][0]["arguments"] = '{"city":"Paris","city":"Rome"}'  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        _project_tools(protocol, request)
    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(codec_projection._ProviderProjector(), decoded_tools(protocol), response)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_normalized_empty_descriptions_and_history_remain_valid(protocol: str) -> None:
    annotation = weather_annotation(protocol)
    annotation["tools"][0]["function"]["description"] = ""  # type: ignore[index]
    request, _ = protocol_case(protocol)

    projected = codec_projection._ProviderProjector().project_host_tools(
        request,
        annotation,
        codec_name=protocol,
    )
    assert projected.projection.definitions[0].description == ""
    assert projected.projection.exchanges == (weather_exchange(),)


@pytest.mark.parametrize("protocol", ["openai_chat", "anthropic_messages", "oci_genai"])
def test_tool_finish_reason_without_a_projected_call_is_rejected(protocol: str) -> None:
    _, response = protocol_case(protocol)
    if protocol == "openai_chat":
        response["choices"][0]["message"]["tool_calls"] = []  # type: ignore[index]
    elif protocol == "anthropic_messages":
        response["content"] = []
    else:
        response["chatResponse"]["choices"][0]["message"]["toolCalls"] = []  # type: ignore[index]
        response["chatResponse"]["choices"][0]["finishReason"] = "TOOL_CALL"  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools(protocol, codec_variant="GENERIC" if protocol == "oci_genai" else None),
            response,
        )


@pytest.mark.parametrize(
    ("protocol", "discriminator"),
    [(protocol, "kind") for protocol in PROTOCOLS] + [("openai_chat", "event"), ("openai_chat", "object")],
)
def test_nested_future_tool_variants_fail_closed(protocol: str, discriminator: str) -> None:
    request, response = protocol_case(protocol)
    extension = {"futureMetadata": {discriminator: "tool_call"}}
    body = request["content"]
    if protocol == "oci_genai":
        body = body["chatRequest"]  # type: ignore[index]
    body.update(extension)  # type: ignore[union-attr]
    response.update(extension)

    with pytest.raises(codec_projection._UnsupportedRequest):
        _project_tools(protocol, request)
    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools(protocol, codec_variant="GENERIC" if protocol == "oci_genai" else None),
            response,
        )


def test_deep_json_tool_arguments_fail_closed() -> None:
    _, response = protocol_case("openai_chat")
    response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (  # type: ignore[index]
        "[" * 2_000 + "0" + "]" * 2_000
    )

    with pytest.raises(codec_projection._UnsupportedRequest, match="arguments are invalid"):
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools("openai_chat"),
            response,
        )
