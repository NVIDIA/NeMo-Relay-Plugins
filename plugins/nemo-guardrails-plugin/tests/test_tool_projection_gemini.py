# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemini-specific structural coverage above Relay's host codec."""

from __future__ import annotations

import pytest
from projection_fixtures import decoded_tools, response_tools, weather_annotation
from tool_projection_cases import check_calls, check_results, gemini_generate_content_case, json_object_schema

from nemoguardrails_nemo_relay import codec_projection
from nemoguardrails_nemo_relay.structural_tools import ToolVerdictKind


def _project_host(request: dict[str, object], annotation: dict[str, object] | None = None):
    return codec_projection._ProviderProjector().project_host_tools(
        request,
        annotation or weather_annotation("gemini_generate_content"),
        codec_name="gemini_generate_content",
    )


def _normalized_function(annotation: dict[str, object]) -> dict[str, object]:
    return annotation["tools"][0]["function"]  # type: ignore[index,return-value]


def test_normalized_execution_and_output_schema_controls_remain_outside_structural_scope() -> None:
    request, _ = gemini_generate_content_case()
    annotation = weather_annotation("gemini_generate_content")
    function = _normalized_function(annotation)
    function.update(
        {
            "behavior": "BLOCKING",
            "response": {"type": "object", "properties": {"forecast": {"type": "string"}}},
            "responseJsonSchema": {
                "type": "object",
                "properties": {"forecast": {"type": "string"}},
            },
        }
    )
    annotation["toolConfig"] = {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["weather"]}}

    definition = _project_host(request, annotation).projection.definitions[0]

    assert definition.name == "weather"
    assert definition.input_schema == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }


@pytest.mark.parametrize("object_type,string_type", [("OBJECT", "STRING"), ("object", "string"), ("Object", "String")])
async def test_normalized_gemini_schema_types_are_translated(
    object_type: str,
    string_type: str,
) -> None:
    request, response = gemini_generate_content_case()
    annotation = weather_annotation("gemini_generate_content")
    schema = _normalized_function(annotation)["parameters"]  # type: ignore[index]
    schema["type"] = object_type  # type: ignore[index]
    schema["properties"]["city"]["type"] = string_type  # type: ignore[index]

    decoded = _project_host(request, annotation)
    calls = response_tools(codec_projection._ProviderProjector(), decoded, response)

    assert decoded.projection.definitions[0].input_schema == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    assert (await check_calls(decoded.projection.definitions, calls)).kind is ToolVerdictKind.PASSED


def test_parameters_json_schema_is_used_without_gemini_dialect_translation() -> None:
    request, _ = gemini_generate_content_case()
    annotation = weather_annotation("gemini_generate_content")
    function = _normalized_function(annotation)
    function.pop("parameters")
    function["parametersJsonSchema"] = json_object_schema()

    definition = _project_host(request, annotation).projection.definitions[0]

    assert definition.input_schema == json_object_schema()


async def test_gemini_int64_schema_bounds_are_translated() -> None:
    request, _ = gemini_generate_content_case()
    annotation = weather_annotation("gemini_generate_content")
    _normalized_function(annotation)["parameters"] = {
        "type": "OBJECT",
        "properties": {
            "cities": {
                "type": "ARRAY",
                "items": {"type": "STRING", "minLength": "1", "maxLength": 64},
                "minItems": "1",
                "maxItems": 10,
            }
        },
    }

    schema = _project_host(request, annotation).projection.definitions[0].input_schema

    assert schema is not None
    cities = schema["properties"]["cities"]  # type: ignore[index]
    assert cities["minItems"] == 1
    assert cities["items"]["maxLength"] == 64


@pytest.mark.parametrize("invalid", [True, -1, "01", "1.0", str(1 << 63)])
def test_invalid_gemini_int64_schema_bounds_fail_closed(invalid: object) -> None:
    request, _ = gemini_generate_content_case()
    annotation = weather_annotation("gemini_generate_content")
    _normalized_function(annotation)["parameters"] = {"type": "STRING", "maxLength": invalid}

    with pytest.raises(codec_projection._UnsupportedRequest, match="tool traffic is not completely covered"):
        _project_host(request, annotation)


def test_deep_gemini_schema_fails_with_a_bounded_coverage_error() -> None:
    request, _ = gemini_generate_content_case()
    annotation = weather_annotation("gemini_generate_content")
    schema: dict[str, object] = {"type": "STRING"}
    for _ in range(1_200):
        schema = {"type": "ARRAY", "items": schema}
    _normalized_function(annotation)["parameters"] = schema

    with pytest.raises(codec_projection._UnsupportedRequest, match="tool traffic is not completely covered"):
        _project_host(request, annotation)


def test_cached_content_in_host_annotation_cannot_hide_tool_history() -> None:
    request, _ = gemini_generate_content_case()
    annotation = weather_annotation("gemini_generate_content")
    annotation["cachedContent"] = "cachedContents/private-history"

    with pytest.raises(codec_projection._UnsupportedRequest, match="provider-side cached content"):
        _project_host(request, annotation)


async def test_normalized_parallel_results_retain_relay_call_ids() -> None:
    request, _ = gemini_generate_content_case()
    annotation = weather_annotation("gemini_generate_content")
    assistant = annotation["messages"][1]["content"]  # type: ignore[index]
    assistant.append(  # type: ignore[union-attr]
        {"type": "tool_use", "id": "call-2", "name": "search", "input": {"query": "Rome"}}
    )
    results = annotation["messages"][2]["content"]  # type: ignore[index]
    results.append(  # type: ignore[union-attr]
        {"type": "tool_result", "tool_use_id": "call-2", "content": "found"}
    )

    projection = _project_host(request, annotation).projection

    assert [result.call_id for result in projection.exchanges[0].results] == ["call-1", "call-2"]
    assert (await check_results(projection.exchanges)).kind is ToolVerdictKind.PASSED


@pytest.mark.parametrize(
    "finish_reason",
    [
        "MALFORMED_FUNCTION_CALL",
        "MALFORMED_RESPONSE",
        "MISSING_THOUGHT_SIGNATURE",
        "TOO_MANY_TOOL_CALLS",
        "UNEXPECTED_TOOL_CALL",
    ],
)
def test_invalid_tool_finish_reason_rejects_even_with_a_function_call(finish_reason: str) -> None:
    _, response = gemini_generate_content_case()
    response["candidates"][0]["finishReason"] = finish_reason  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools("gemini_generate_content"),
            response,
        )


def test_tool_finish_reason_without_a_function_call_is_rejected() -> None:
    response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "safe"}]},
                "finishReason": "TOOL_CODE",
            }
        ]
    }

    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools("gemini_generate_content"),
            response,
        )


def test_candidate_and_usage_metadata_do_not_hide_function_calls() -> None:
    _, response = gemini_generate_content_case()
    response["candidates"][0]["safetyRatings"] = [  # type: ignore[index]
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "NEGLIGIBLE"}
    ]
    response["usageMetadata"] = {
        "promptTokenCount": 10,
        "toolUsePromptTokenCount": 3,
        "totalTokenCount": 20,
    }

    calls = response_tools(
        codec_projection._ProviderProjector(),
        decoded_tools("gemini_generate_content"),
        response,
    )
    assert [call.name for call in calls] == ["weather"]


def test_unknown_response_part_cannot_hide_a_future_tool_call() -> None:
    _, response = gemini_generate_content_case()
    response["candidates"][0]["content"]["parts"] = [  # type: ignore[index]
        {"futureToolCall": {"name": "weather", "args": {"city": "Rome"}}}
    ]

    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools("gemini_generate_content"),
            response,
        )


@pytest.mark.parametrize("field", ["functionCall", "functionResponse"])
def test_raw_request_function_extensions_cannot_hide_tool_traffic(field: str) -> None:
    request, _ = gemini_generate_content_case()
    content_index = 1 if field == "functionCall" else 2
    value = request["content"]["contents"][content_index]["parts"][0][field]  # type: ignore[index]
    value["mcp_call"] = {"name": "hidden"}  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        _project_host(request)


def test_raw_request_part_cannot_mix_text_and_a_function_call() -> None:
    request, _ = gemini_generate_content_case()
    call_part = request["content"]["contents"][1]["parts"][0]  # type: ignore[index]
    call_part["text"] = "unchecked"  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        _project_host(request)


def test_raw_request_part_requires_one_data_variant() -> None:
    request, _ = gemini_generate_content_case()
    request["content"]["contents"][0]["parts"] = [{"partMetadata": {"source": "fixture"}}]  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest, match="one data variant"):
        _project_host(request)


def test_function_result_parts_cannot_nest_tool_traffic() -> None:
    request, _ = gemini_generate_content_case()
    result = request["content"]["contents"][2]["parts"][0]["functionResponse"]  # type: ignore[index]
    result["parts"] = [{"functionCall": {"name": "hidden", "args": {}}}]  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest, match="nested tool traffic"):
        _project_host(request)


def test_response_part_metadata_is_opaque_to_structural_checks() -> None:
    _, response = gemini_generate_content_case()
    response["candidates"][0]["content"]["parts"][0]["partMetadata"] = {  # type: ignore[index]
        "audit": "function-call"
    }

    calls = response_tools(
        codec_projection._ProviderProjector(),
        decoded_tools("gemini_generate_content"),
        response,
    )
    assert calls[0].name == "weather"
