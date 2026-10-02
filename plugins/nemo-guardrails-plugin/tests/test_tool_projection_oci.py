# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OCI Generative AI structural coverage above Relay's host codec."""

from __future__ import annotations

from copy import deepcopy

import pytest
from projection_fixtures import decoded_tools, response_tools, weather_annotation
from tool_projection_cases import check_calls, oci_cohere_v2_case, oci_generic_case

from nemoguardrails_nemo_relay import codec_projection
from nemoguardrails_nemo_relay.structural_tools import ToolVerdictKind


def _project_host(
    request: dict[str, object],
    annotation: dict[str, object] | None = None,
):
    return codec_projection._ProviderProjector().project_host_tools(
        request,
        annotation or weather_annotation("oci_genai"),
        codec_name="oci_genai",
    )


def _cohere_v1_request(parameter_type: object = "str") -> dict[str, object]:
    return {
        "headers": {},
        "content": {
            "chatRequest": {
                "apiFormat": "COHERE",
                "message": "weather",
                "tools": [
                    {
                        "name": "weather",
                        "description": "Look up weather",
                        "parameterDefinitions": {
                            "value": {
                                "type": parameter_type,
                                "description": "Input value",
                                "isRequired": True,
                            }
                        },
                    }
                ],
            }
        },
    }


def _cohere_v1_annotation(request: dict[str, object]) -> dict[str, object]:
    native = request["content"]["chatRequest"]["tools"][0]  # type: ignore[index]
    return {
        "messages": [{"role": "user", "content": "weather"}],
        "api_specific": {"api": "oci_genai", "api_format": "COHERE"},
        "tools": [
            {
                "type": "provider_native",
                "provider": "oci_genai",
                "kind": "FUNCTION",
                "value": deepcopy(native),
            }
        ],
    }


def test_generic_response_call_cannot_mix_arguments_and_parameters() -> None:
    _, response = oci_generic_case()
    call = response["chatResponse"]["choices"][0]["message"]["toolCalls"][0]  # type: ignore[index]
    call["parameters"] = {"city": "Paris"}  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools("oci_genai", codec_variant="GENERIC"),
            response,
        )


def test_generic_request_call_cannot_mix_arguments_and_parameters() -> None:
    request, _ = oci_generic_case()
    call = request["content"]["chatRequest"]["messages"][1]["toolCalls"][0]  # type: ignore[index]
    call["parameters"] = {"city": "Paris"}  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        _project_host(request)


def test_cohere_v2_projects_response_function_calls() -> None:
    _, response = oci_cohere_v2_case()

    calls = response_tools(
        codec_projection._ProviderProjector(),
        decoded_tools("oci_genai", codec_variant="COHEREV2"),
        response,
    )

    assert calls[0].call_id == "call-2"
    assert calls[0].name == "weather"


def test_cohere_v2_tool_plan_and_citations_are_not_mistaken_for_calls() -> None:
    _, response = oci_cohere_v2_case()
    message = response["chatResponse"]["message"]  # type: ignore[index]
    message["toolPlan"] = "I will check the weather."  # type: ignore[index]
    message["citations"] = [  # type: ignore[index]
        {
            "contentIndex": 0,
            "start": 0,
            "end": 4,
            "text": "plan",
            "type": "PLAN",
            "sources": [
                {
                    "type": "TOOL",
                    "tool": {"id": "call-1", "toolOutput": {"forecast": "sunny"}},
                }
            ],
        }
    ]

    calls = response_tools(
        codec_projection._ProviderProjector(),
        decoded_tools("oci_genai", codec_variant="COHEREV2"),
        response,
    )
    assert [call.name for call in calls] == ["weather"]


@pytest.mark.parametrize("call_type", [None, "CUSTOM"])
def test_cohere_v2_response_requires_function_call_type(call_type: str | None) -> None:
    _, response = oci_cohere_v2_case()
    call = response["chatResponse"]["message"]["toolCalls"][0]  # type: ignore[index]
    if call_type is None:
        call.pop("type")  # type: ignore[union-attr]
    else:
        call["type"] = call_type  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools("oci_genai", codec_variant="COHEREV2"),
            response,
        )


@pytest.mark.parametrize("role", ["USER", "SYSTEM", "TOOL"])
def test_raw_oci_calls_on_non_assistant_roles_are_rejected(role: str) -> None:
    request, _ = oci_cohere_v2_case()
    request["content"]["chatRequest"]["messages"][1]["role"] = role  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        _project_host(request)


@pytest.mark.parametrize("role", ["USER", "SYSTEM", "ASSISTANT"])
def test_raw_oci_result_ids_on_non_tool_roles_are_rejected(role: str) -> None:
    request, _ = oci_cohere_v2_case()
    request["content"]["chatRequest"]["messages"][2]["role"] = role  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        _project_host(request)


def test_generic_assistant_metadata_is_opaque_only_to_structural_checks() -> None:
    request, _ = oci_generic_case()
    assistant = request["content"]["chatRequest"]["messages"][1]  # type: ignore[index]
    assistant.update(  # type: ignore[union-attr]
        {
            "name": "assistant-a",
            "refusal": None,
            "annotations": [{"type": "url_citation", "urlCitation": {"url": "https://example.test"}}],
            "reasoningContent": "private reasoning",
        }
    )

    assert _project_host(request).projection.exchanges[0].calls[0].name == "weather"


@pytest.mark.parametrize(
    ("parameter_type", "expected"),
    [
        ("str", {"type": "string"}),
        ("List[str]", {"items": {"type": "string"}, "type": "array"}),
        ("Dict[str, int]", {"additionalProperties": {"type": "integer"}, "type": "object"}),
        (
            "list[Dict[str, bool]]",
            {
                "items": {"additionalProperties": {"type": "boolean"}, "type": "object"},
                "type": "array",
            },
        ),
    ],
)
def test_cohere_v1_translates_bounded_python_types(
    parameter_type: str,
    expected: dict[str, object],
) -> None:
    request = _cohere_v1_request(parameter_type)

    definition = _project_host(request, _cohere_v1_annotation(request)).projection.definitions[0]

    assert definition.input_schema is not None
    assert definition.input_schema["properties"]["value"] == {  # type: ignore[index]
        **expected,
        "description": "Input value",
    }
    assert definition.input_schema["required"] == ["value"]


@pytest.mark.parametrize(
    "parameter_type",
    [None, "datetime", "List[str, int]", "Dict[int, str]", "list[str] | None", []],
)
def test_cohere_v1_rejects_unsupported_parameter_types(parameter_type: object) -> None:
    request = _cohere_v1_request(parameter_type)

    with pytest.raises(codec_projection._UnsupportedRequest, match="tool traffic is not completely covered"):
        _project_host(request, _cohere_v1_annotation(request))


async def test_cohere_v1_response_calls_have_stable_synthesized_ids() -> None:
    request = _cohere_v1_request()
    response = {
        "chatResponse": {
            "apiFormat": "COHERE",
            "text": "",
            "toolCalls": [
                {"name": "weather", "parameters": {"value": "Rome"}},
                {"name": "weather", "parameters": {"value": "Paris"}},
            ],
            "finishReason": "COMPLETE",
        }
    }
    decoded = _project_host(request, _cohere_v1_annotation(request))

    calls = response_tools(codec_projection._ProviderProjector(), decoded, response)

    assert tuple(call.call_id for call in calls) == ("call_0", "call_1")
    assert (await check_calls(decoded.projection.definitions, calls)).kind is ToolVerdictKind.PASSED


def test_cohere_v1_no_argument_definition_maps_to_closed_empty_schema() -> None:
    request = _cohere_v1_request()
    tool = request["content"]["chatRequest"]["tools"][0]  # type: ignore[index]
    tool.pop("parameterDefinitions")  # type: ignore[union-attr]

    definition = _project_host(request, _cohere_v1_annotation(request)).projection.definitions[0]

    assert definition.input_schema == {
        "additionalProperties": False,
        "properties": {},
        "type": "object",
    }


def test_cohere_v1_result_linkage_remains_unsupported() -> None:
    request = _cohere_v1_request()
    annotation = _cohere_v1_annotation(request)
    annotation["toolResults"] = [
        {
            "call": {"name": "weather", "parameters": {"value": "Rome"}},
            "outputs": [{"temperature": 24}],
        }
    ]

    with pytest.raises(codec_projection._UnsupportedRequest, match="unmodeled tool state"):
        _project_host(request, annotation)


def test_provider_native_content_cannot_hide_future_tool_traffic() -> None:
    request, _ = oci_generic_case()
    request["content"]["chatRequest"]["messages"] = [  # type: ignore[index]
        {
            "role": "USER",
            "content": [{"type": "FUTURE_TOOL_CALL", "name": "weather"}],
        }
    ]

    with pytest.raises(codec_projection._UnsupportedRequest):
        _project_host(request)
