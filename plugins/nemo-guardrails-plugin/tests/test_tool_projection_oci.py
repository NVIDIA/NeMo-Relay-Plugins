# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest
from tool_projection_cases import (
    check_calls,
    oci_cohere_v2_case,
    oci_generic_case,
)

from nemoguardrails_nemo_relay import codec_projection
from nemoguardrails_nemo_relay.structural_tools import ToolVerdictKind


def test_oci_generic_response_call_cannot_mix_arguments_and_parameters() -> None:
    request, response = oci_generic_case()
    broken = deepcopy(response)
    call = broken["chatResponse"]["choices"][0]["message"]["toolCalls"][0]  # type: ignore[index]
    call["parameters"] = {"city": "Paris"}  # type: ignore[index]
    projector = codec_projection._ProviderProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), broken)


def test_oci_cohere_v2_projects_definitions_calls_and_results() -> None:
    request, response = oci_cohere_v2_case()
    projector = codec_projection._ProviderProjector()

    decoded = projector.project_tools(request)
    calls = projector.project_response_tools(decoded, response)

    assert decoded.codec_names == ("oci_genai",)
    assert decoded.projection.definitions[0].name == "weather"
    assert decoded.projection.exchanges[0].results[0].call_id == "call-1"
    assert calls[0].call_id == "call-2"


def test_oci_cohere_v2_tool_plan_metadata_is_not_mistaken_for_a_call() -> None:
    request, response = oci_cohere_v2_case()
    response["chatResponse"]["message"]["toolPlan"] = "I will check the weather."  # type: ignore[index]
    projector = codec_projection._ProviderProjector()

    calls = projector.project_response_tools(projector.project_tools(request), response)

    assert calls[0].name == "weather"


def test_oci_generic_message_metadata_is_opaque_to_structural_checks_only() -> None:
    request, _ = oci_generic_case()
    messages = request["content"]["chatRequest"]["messages"]  # type: ignore[index]
    messages[0]["name"] = "caller"  # type: ignore[index]
    messages[1].update(  # type: ignore[union-attr]
        {
            "name": "assistant-a",
            "refusal": None,
            "annotations": [{"type": "url_citation", "urlCitation": {"url": "https://example.test"}}],
            "reasoningContent": "private reasoning",
        }
    )
    projector = codec_projection._ProviderProjector()

    decoded = projector.project_tools(request)

    assert decoded.projection.exchanges[0].calls[0].name == "weather"
    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project(request, allow_structural_tools=True, allow_tool_results=True)


def test_oci_cohere_v2_message_metadata_is_opaque_to_structural_checks_only() -> None:
    request, _ = oci_cohere_v2_case()
    assistant = request["content"]["chatRequest"]["messages"][1]  # type: ignore[index]
    assistant["toolPlan"] = "Call the declared weather function."  # type: ignore[index]
    assistant["citations"] = [  # type: ignore[index]
        {
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
    projector = codec_projection._ProviderProjector()

    decoded = projector.project_tools(request)

    assert decoded.projection.exchanges[0].calls[0].name == "weather"
    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project(request, allow_structural_tools=True, allow_tool_results=True)


def test_oci_cohere_v2_response_tool_citation_is_not_mistaken_for_a_call() -> None:
    request, response = oci_cohere_v2_case()
    response["chatResponse"]["message"]["citations"] = [  # type: ignore[index]
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
    projector = codec_projection._ProviderProjector()
    decoded = projector.project_tools(request)

    calls = projector.project_response_tools(decoded, response)

    assert calls[0].name == "weather"
    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_text(decoded, response, allow_tool_calls=True)


@pytest.mark.parametrize(
    ("api_format", "role", "field", "value"),
    [
        ("GENERIC", "USER", "refusal", "not valid on a user message"),
        ("GENERIC", "ASSISTANT", "annotations", ["not-an-object"]),
        ("GENERIC", "ASSISTANT", "annotations", [{"toolCalls": []}]),
        ("GENERIC", "ASSISTANT", "reasoningContent", {"text": "not-a-string"}),
        ("GENERIC", "ASSISTANT", "toolPlan", "Cohere-only"),
        ("COHEREV2", "USER", "name", "not valid in Cohere V2"),
        ("COHEREV2", "ASSISTANT", "refusal", "GENERIC-only"),
        ("COHEREV2", "ASSISTANT", "citations", ["not-an-object"]),
        ("COHEREV2", "ASSISTANT", "citations", [{"toolCalls": []}]),
    ],
)
def test_oci_structural_metadata_remains_format_and_role_scoped(
    api_format: str,
    role: str,
    field: str,
    value: object,
) -> None:
    if api_format == "GENERIC":
        request, _ = oci_generic_case()
    else:
        request, _ = oci_cohere_v2_case()
    messages = request["content"]["chatRequest"]["messages"]  # type: ignore[index]
    target = next(message for message in messages if message["role"] == role)  # type: ignore[index]
    target[field] = value

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project_tools(request)


@pytest.mark.parametrize("role", ["USER", "SYSTEM", "TOOL"])
def test_oci_calls_on_non_assistant_roles_are_rejected(role: str) -> None:
    request, _ = oci_cohere_v2_case()
    request["content"]["chatRequest"]["messages"][1]["role"] = role  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project_tools(request)


@pytest.mark.parametrize("role", ["USER", "SYSTEM", "ASSISTANT"])
def test_oci_result_ids_on_non_tool_roles_are_rejected(role: str) -> None:
    request, _ = oci_cohere_v2_case()
    request["content"]["chatRequest"]["messages"][2]["role"] = role  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project_tools(request)


@pytest.mark.parametrize("call_type", [None, "CUSTOM"])
def test_oci_cohere_v2_response_requires_function_call_type(call_type: str | None) -> None:
    request, response = oci_cohere_v2_case()
    call = response["chatResponse"]["message"]["toolCalls"][0]  # type: ignore[index]
    if call_type is None:
        call.pop("type")  # type: ignore[union-attr]
    else:
        call["type"] = call_type  # type: ignore[index]
    projector = codec_projection._ProviderProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), response)


async def test_oci_cohere_v1_validates_emitted_calls_with_synthesized_ids() -> None:
    request = {
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
                            "city": {"type": "str", "description": "City name", "isRequired": True},
                        },
                    }
                ],
            }
        },
    }
    response = {
        "chatResponse": {
            "apiFormat": "COHERE",
            "text": "",
            "toolCalls": [{"name": "weather", "parameters": {"city": "Rome"}}],
            "finishReason": "COMPLETE",
        }
    }
    projector = codec_projection._ProviderProjector()

    decoded = projector.project_tools(request)
    calls = projector.project_response_tools(decoded, response)

    assert decoded.projection.definitions[0].input_schema == {
        "additionalProperties": False,
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
        "type": "object",
    }
    assert calls[0].call_id == "call_0"
    assert (await check_calls(decoded.projection.definitions, calls)).kind is ToolVerdictKind.PASSED

    invalid_response = deepcopy(response)
    invalid_response["chatResponse"]["toolCalls"][0]["parameters"]["city"] = 42  # type: ignore[index]
    invalid_calls = projector.project_response_tools(decoded, invalid_response)
    assert (await check_calls(decoded.projection.definitions, invalid_calls)).kind is ToolVerdictKind.POLICY_BLOCK


@pytest.mark.parametrize(
    ("parameter_type", "expected"),
    [
        ("str", {"type": "string"}),
        ("int", {"type": "integer"}),
        ("float", {"type": "number"}),
        ("bool", {"type": "boolean"}),
        ("List", {"items": {}, "type": "array"}),
        ("Dict", {"additionalProperties": True, "type": "object"}),
        ("List[str]", {"items": {"type": "string"}, "type": "array"}),
        (
            "List[List[str]]",
            {"items": {"items": {"type": "string"}, "type": "array"}, "type": "array"},
        ),
        (
            "Dict[str, int]",
            {"additionalProperties": {"type": "integer"}, "type": "object"},
        ),
        (
            "list[Dict[str, bool]]",
            {
                "items": {"additionalProperties": {"type": "boolean"}, "type": "object"},
                "type": "array",
            },
        ),
    ],
)
def test_oci_cohere_v1_translates_json_compatible_python_types(
    parameter_type: str,
    expected: dict[str, object],
) -> None:
    request = {
        "headers": {},
        "content": {
            "chatRequest": {
                "apiFormat": "COHERE",
                "message": "weather",
                "tools": [
                    {
                        "name": "weather",
                        "description": "Look up weather",
                        "parameterDefinitions": {"value": {"type": parameter_type}},
                    }
                ],
            }
        },
    }

    definition = codec_projection._ProviderProjector().project_tools(request).projection.definitions[0]

    assert definition.input_schema["properties"]["value"] == expected  # type: ignore[index]


@pytest.mark.parametrize(
    "parameter_type",
    [
        None,
        "datetime",
        "CustomRecord",
        "string",
        "integer",
        "array",
        "object",
        "List[str, int]",
        "Dict[str]",
        "Dict[int, str]",
        "tuple[str]",
        "typing.List[str]",
        "list[str] | None",
        "List[",
        f"List[{' ' * 300}str]",
        "List[List[List[List[List[List[List[List[List[str]]]]]]]]]",
        [],
        {},
    ],
)
def test_oci_cohere_v1_rejects_unsupported_parameter_types(parameter_type: object) -> None:
    request = {
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
                            "city": {"type": parameter_type, "isRequired": True},
                        },
                    }
                ],
            }
        },
    }

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project_tools(request)


async def test_oci_cohere_v1_assigns_distinct_ids_to_parallel_calls() -> None:
    request = {
        "headers": {},
        "content": {
            "chatRequest": {
                "apiFormat": "COHERE",
                "message": "weather",
                "tools": [
                    {
                        "name": "weather",
                        "description": "Look up weather",
                        "parameterDefinitions": {"city": {"type": "str", "isRequired": True}},
                    }
                ],
            }
        },
    }
    response = {
        "chatResponse": {
            "apiFormat": "COHERE",
            "text": "",
            "toolCalls": [
                {"name": "weather", "parameters": {"city": "Rome"}},
                {"name": "weather", "parameters": {"city": "Paris"}},
            ],
            "finishReason": "COMPLETE",
        }
    }
    projector = codec_projection._ProviderProjector()
    decoded = projector.project_tools(request)

    calls = projector.project_response_tools(decoded, response)

    assert tuple(call.call_id for call in calls) == ("call_0", "call_1")
    assert (await check_calls(decoded.projection.definitions, calls)).kind is ToolVerdictKind.PASSED


def test_oci_cohere_v1_accepts_a_no_argument_definition() -> None:
    request = {
        "headers": {},
        "content": {
            "chatRequest": {
                "apiFormat": "COHERE",
                "message": "ping",
                "tools": [{"name": "ping", "description": "Return service health"}],
            }
        },
    }

    definition = codec_projection._ProviderProjector().project_tools(request).projection.definitions[0]

    assert definition.input_schema == {
        "additionalProperties": False,
        "properties": {},
        "type": "object",
    }


def test_oci_generic_does_not_accept_a_cohere_v1_definition_shape() -> None:
    request, _ = oci_generic_case()
    request["content"]["chatRequest"]["tools"] = [  # type: ignore[index]
        {"name": "weather", "description": "Look up weather"}
    ]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project_tools(request)


def test_oci_cohere_v1_result_linkage_remains_unsupported() -> None:
    request = {
        "headers": {},
        "content": {
            "chatRequest": {
                "apiFormat": "COHERE",
                "message": "continue",
                "tools": [
                    {
                        "name": "weather",
                        "description": "Look up weather",
                        "parameterDefinitions": {"city": {"type": "str", "isRequired": True}},
                    }
                ],
                "toolResults": [
                    {
                        "call": {"name": "weather", "parameters": {"city": "Rome"}},
                        "outputs": [{"temperature": 24}],
                    }
                ],
            }
        },
    }

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project_tools(request)


def test_oci_provider_native_content_cannot_hide_tool_traffic() -> None:
    request, _ = oci_generic_case()
    broken = deepcopy(request)
    broken["content"]["chatRequest"]["messages"] = [  # type: ignore[index]
        {
            "role": "USER",
            "content": [{"type": "FUTURE_TOOL_CALL", "name": "weather"}],
        }
    ]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project_tools(broken)
