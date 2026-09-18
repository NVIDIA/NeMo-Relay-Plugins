# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest
from tool_projection_cases import (
    check_calls,
    check_results,
    gemini_generate_content_case,
    json_object_schema,
)

from nemoguardrails_nemo_relay import codec_projection
from nemoguardrails_nemo_relay.structural_tools import ToolVerdictKind


def test_gemini_execution_and_result_schema_controls_remain_outside_structural_scope() -> None:
    request, response = gemini_generate_content_case()
    controlled = deepcopy(request)
    declaration = controlled["content"]["tools"][0]["functionDeclarations"][0]  # type: ignore[index]
    declaration.update(  # type: ignore[union-attr]
        {
            "behavior": "BLOCKING",
            "response": {"type": "object", "properties": {"function": {"type": "string"}}},
            "responseJsonSchema": {
                "type": "object",
                "properties": {"tool": {"type": "string"}},
            },
        }
    )
    controlled["content"]["toolConfig"] = {  # type: ignore[index]
        "functionCallingConfig": {
            "mode": "ANY",
            "allowedFunctionNames": ["weather"],
        }
    }
    projector = codec_projection._NativeCodecProjector()

    decoded = projector.project_tools(controlled)
    calls = projector.project_response_tools(decoded, response)

    assert decoded.projection.definitions[0].name == "weather"
    assert calls[0].name == "weather"
    assert calls[0].arguments == {"city": "Rome"}
    assert projector.project(
        request,
        allow_structural_tools=True,
        allow_tool_results=True,
    ) == [{"role": "user", "content": "weather"}]


@pytest.mark.parametrize(
    "finish_reason",
    [
        "MALFORMED_FUNCTION_CALL",
        "MALFORMED_RESPONSE",
        "MISSING_THOUGHT_SIGNATURE",
        "TOO_MANY_TOOL_CALLS",
        "UNEXPECTED_TOOL_CALL",
        "TOOL_CODE",
    ],
)
def test_gemini_tool_finish_reason_without_a_function_call_rejects(
    finish_reason: str,
) -> None:
    request, _ = gemini_generate_content_case()
    response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "safe"}]},
                "finishReason": finish_reason,
            }
        ]
    }
    projector = codec_projection._NativeCodecProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), response)


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
def test_gemini_invalid_tool_finish_reason_rejects_even_with_a_function_call(
    finish_reason: str,
) -> None:
    request, response = gemini_generate_content_case()
    response["candidates"][0]["finishReason"] = finish_reason  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), response)


def test_gemini_object_function_response_uses_the_codec_normalization() -> None:
    request, _ = gemini_generate_content_case()

    projection = codec_projection._NativeCodecProjector().project_tools(request).projection

    assert projection.exchanges[0].results[0].content == '{"result":"sunny"}'


def test_gemini_candidate_metadata_does_not_hide_function_calls() -> None:
    request, response = gemini_generate_content_case()
    enriched = deepcopy(response)
    enriched["candidates"][0]["safetyRatings"] = [  # type: ignore[index]
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "NEGLIGIBLE"}
    ]
    projector = codec_projection._NativeCodecProjector()

    calls = projector.project_response_tools(projector.project_tools(request), enriched)

    assert [call.name for call in calls] == ["weather"]


def test_unknown_gemini_response_part_cannot_hide_a_future_tool_call() -> None:
    request, response = gemini_generate_content_case()
    broken = deepcopy(response)
    broken["candidates"][0]["content"]["parts"] = [  # type: ignore[index]
        {"futureToolCall": {"name": "weather", "args": {"city": "Rome"}}}
    ]
    projector = codec_projection._NativeCodecProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), broken)


def test_gemini_signed_thought_history_is_allowed_for_structural_checks_only() -> None:
    request, _ = gemini_generate_content_case()
    signed = deepcopy(request)
    parts = signed["content"]["contents"][1]["parts"]  # type: ignore[index]
    parts.insert(0, {"text": "private reasoning", "thought": True, "thoughtSignature": "sig-1"})
    parts[1]["thoughtSignature"] = "sig-2"
    projector = codec_projection._NativeCodecProjector()

    decoded = projector.project_tools(signed)

    assert decoded.projection.exchanges[0].calls[0].name == "weather"
    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project(signed, allow_structural_tools=True, allow_tool_results=True)


def test_gemini_tool_usage_metadata_is_not_mistaken_for_a_call() -> None:
    request, response = gemini_generate_content_case()
    response["usageMetadata"] = {
        "promptTokenCount": 10,
        "toolUsePromptTokenCount": 3,
        "toolUsePromptTokensDetails": [{"modality": "TEXT", "tokenCount": 3}],
        "totalTokenCount": 20,
    }
    projector = codec_projection._NativeCodecProjector()

    calls = projector.project_response_tools(projector.project_tools(request), response)

    assert calls[0].name == "weather"


def test_gemini_parameters_json_schema_is_used_for_argument_validation() -> None:
    request, _ = gemini_generate_content_case()
    declaration = request["content"]["tools"][0]["functionDeclarations"][0]  # type: ignore[index]
    declaration.pop("parameters")  # type: ignore[union-attr]
    declaration["parametersJsonSchema"] = json_object_schema()  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()

    projection = projector.project_tools(request).projection

    assert projection.definitions[0].input_schema == json_object_schema()


def test_null_gemini_parameters_falls_back_to_parameters_json_schema() -> None:
    request, _ = gemini_generate_content_case()
    declaration = request["content"]["tools"][0]["functionDeclarations"][0]  # type: ignore[index]
    declaration["parameters"] = None  # type: ignore[index]
    declaration["parametersJsonSchema"] = json_object_schema()  # type: ignore[index]

    projection = codec_projection._NativeCodecProjector().project_tools(request).projection

    assert projection.definitions[0].input_schema == json_object_schema()


async def test_native_gemini_schema_types_are_translated_before_guardrails_validation() -> None:
    request, response = gemini_generate_content_case()
    projector = codec_projection._NativeCodecProjector()

    decoded = projector.project_tools(request)
    calls = projector.project_response_tools(decoded, response)

    assert decoded.projection.definitions[0].input_schema == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    assert (await check_calls(decoded.projection.definitions, calls)).kind is ToolVerdictKind.PASSED


@pytest.mark.parametrize("object_type,string_type", [("OBJECT", "STRING"), ("object", "string"), ("Object", "String")])
async def test_native_gemini_schema_type_casing_is_supported(object_type: str, string_type: str) -> None:
    request, response = gemini_generate_content_case()
    schema = request["content"]["tools"][0]["functionDeclarations"][0]["parameters"]  # type: ignore[index]
    schema["type"] = object_type  # type: ignore[index]
    schema["properties"]["city"]["type"] = string_type  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()

    decoded = projector.project_tools(request)
    calls = projector.project_response_tools(decoded, response)

    assert (await check_calls(decoded.projection.definitions, calls)).kind is ToolVerdictKind.PASSED


async def test_native_gemini_int64_schema_bounds_are_translated() -> None:
    request, response = gemini_generate_content_case()
    declaration = request["content"]["tools"][0]["functionDeclarations"][0]  # type: ignore[index]
    declaration["parameters"] = {  # type: ignore[index]
        "type": "OBJECT",
        "properties": {
            "city": {"type": "STRING", "minLength": "1", "maxLength": "64"},
            "aliases": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "minItems": "1",
                "maxItems": "4",
            },
        },
        "required": ["city", "aliases"],
        "minProperties": "2",
        "maxProperties": "2",
    }
    response["candidates"][0]["content"]["parts"][0]["functionCall"]["args"] = {  # type: ignore[index]
        "city": "Rome",
        "aliases": ["Roma"],
    }
    projector = codec_projection._NativeCodecProjector()

    decoded = projector.project_tools(request)
    calls = projector.project_response_tools(decoded, response)

    schema = decoded.projection.definitions[0].input_schema
    assert schema is not None
    assert schema["minProperties"] == 2
    assert schema["properties"]["aliases"]["minItems"] == 1  # type: ignore[index]
    assert (await check_calls(decoded.projection.definitions, calls)).kind is ToolVerdictKind.PASSED


@pytest.mark.parametrize("bound", [True, -1, "-1", "01", "1.0", str(1 << 63)])
def test_invalid_gemini_int64_schema_bounds_fail_coverage_projection(bound: object) -> None:
    request, _ = gemini_generate_content_case()
    declaration = request["content"]["tools"][0]["functionDeclarations"][0]  # type: ignore[index]
    declaration["parameters"]["minProperties"] = bound  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest, match="tool traffic is not completely covered"):
        codec_projection._NativeCodecProjector().project_tools(request)


def test_unsupported_gemini_schema_dialect_fails_coverage_projection() -> None:
    request, _ = gemini_generate_content_case()
    declaration = request["content"]["tools"][0]["functionDeclarations"][0]  # type: ignore[index]
    declaration["parameters"]["futureSchemaKeyword"] = {"tool": "hidden"}  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest, match="tool traffic is not completely covered"):
        codec_projection._NativeCodecProjector().project_tools(request)


def test_deep_gemini_schema_fails_with_bounded_coverage_error() -> None:
    request, _ = gemini_generate_content_case()
    schema: dict[str, object] = {"type": "STRING"}
    for _ in range(1_200):
        schema = {"type": "ARRAY", "items": schema}
    declaration = request["content"]["tools"][0]["functionDeclarations"][0]  # type: ignore[index]
    declaration["parameters"] = schema  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest, match="tool traffic is not completely covered"):
        codec_projection._NativeCodecProjector().project_tools(request)


def test_gemini_tool_selection_control_is_compatible_with_text_input_rails() -> None:
    gemini, _ = gemini_generate_content_case()
    gemini["content"]["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()

    assert projector.project_tools(gemini).projection.definitions[0].name == "weather"
    assert projector.project(
        gemini,
        allow_structural_tools=True,
        allow_tool_results=True,
    ) == [{"role": "user", "content": "weather"}]


def test_gemini_cached_content_cannot_hide_tool_history() -> None:
    request, _ = gemini_generate_content_case()
    request["content"]["cachedContent"] = "cachedContents/private-history"  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()

    with pytest.raises(codec_projection._UnsupportedRequest, match="provider codec could not verify"):
        projector.project_tools(request)


async def test_gemini_function_response_name_must_match_call_id() -> None:
    request, _ = gemini_generate_content_case()
    response = request["content"]["contents"][2]["parts"][0]["functionResponse"]  # type: ignore[index]
    response["name"] = "different_function"  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(request)


async def test_gemini_function_response_with_wrong_id_is_policy_blocked() -> None:
    request, _ = gemini_generate_content_case()
    response = request["content"]["contents"][2]["parts"][0]["functionResponse"]  # type: ignore[index]
    response["id"] = "missing-call"  # type: ignore[index]

    projection = codec_projection._NativeCodecProjector().project_tools(request).projection

    assert (await check_results(projection.exchanges)).kind is ToolVerdictKind.POLICY_BLOCK


@pytest.mark.parametrize("omit_call_id", [False, True])
async def test_gemini_missing_response_id_uses_unique_prior_call(
    omit_call_id: bool,
) -> None:
    request, _ = gemini_generate_content_case()
    call = request["content"]["contents"][1]["parts"][0]["functionCall"]  # type: ignore[index]
    response = request["content"]["contents"][2]["parts"][0]["functionResponse"]  # type: ignore[index]
    response.pop("id")  # type: ignore[union-attr]
    if omit_call_id:
        call.pop("id")  # type: ignore[union-attr]
    projection = codec_projection._NativeCodecProjector().project_tools(request).projection

    exchange = projection.exchanges[0]
    expected_id = "weather" if omit_call_id else "call-1"
    assert exchange.calls[0].call_id == expected_id
    assert exchange.results[0].call_id == expected_id
    assert (await check_results(projection.exchanges)).kind is ToolVerdictKind.PASSED


async def test_parallel_gemini_results_without_ids_link_by_unique_name() -> None:
    request, _ = gemini_generate_content_case()
    request["content"]["contents"][1]["parts"].append(  # type: ignore[index,union-attr]
        {"functionCall": {"id": "call-2", "name": "search", "args": {"query": "Rome"}}}
    )
    request["content"]["contents"][2]["parts"] = [  # type: ignore[index]
        {"functionResponse": {"name": "weather", "response": {"result": "sunny"}}},
        {"functionResponse": {"name": "search", "response": {"result": "found"}}},
    ]

    projection = codec_projection._NativeCodecProjector().project_tools(request, require_definitions=False).projection

    exchange = projection.exchanges[0]
    assert [result.call_id for result in exchange.results] == ["call-1", "call-2"]
    assert (await check_results(projection.exchanges)).kind is ToolVerdictKind.PASSED


def test_gemini_part_metadata_is_opaque_only_to_structural_checks() -> None:
    request, _ = gemini_generate_content_case()
    request["content"]["contents"][0]["parts"][0]["partMetadata"] = {  # type: ignore[index]
        "tool": "audit-label"
    }
    text_response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": "done",
                            "partMetadata": {"function": "audit-label"},
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ]
    }
    projector = codec_projection._NativeCodecProjector()

    decoded = projector.project_tools(request)

    assert projector.project_response_tools(decoded, text_response) == ()
    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project(request, allow_structural_tools=True, allow_tool_results=True)

    clean_request, _ = gemini_generate_content_case()
    text_decoded = projector.project_text(
        clean_request,
        allow_structural_tools=True,
        allow_tool_results=True,
        require_user=True,
    )
    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_text(text_decoded, text_response, allow_tool_calls=True)


@pytest.mark.parametrize("field", ["willContinue", "scheduling"])
def test_gemini_generator_function_results_are_outside_v1_contract(field: str) -> None:
    request, _ = gemini_generate_content_case()
    response = request["content"]["contents"][2]["parts"][0]["functionResponse"]  # type: ignore[index]
    response[field] = True if field == "willContinue" else "SILENT"  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(request)


@pytest.mark.parametrize("field", ["functionCall", "functionResponse"])
def test_gemini_function_extensions_cannot_hide_tool_traffic(field: str) -> None:
    request, _ = gemini_generate_content_case()
    parts = request["content"]["contents"]  # type: ignore[index]
    if field == "functionCall":
        value = parts[1]["parts"][0]["functionCall"]  # type: ignore[index]
    else:
        value = parts[2]["parts"][0]["functionResponse"]  # type: ignore[index]
    value["mcp_call"] = {"name": "hidden"}  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(request)
