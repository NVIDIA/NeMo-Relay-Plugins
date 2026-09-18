# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest
from tool_projection_cases import (
    PROTOCOLS,
    check_calls,
    check_results,
    gemini_generate_content_case,
    oci_cohere_v2_case,
    protocol_case,
    response_with_text,
)

from nemoguardrails_nemo_relay import codec_projection
from nemoguardrails_nemo_relay.structural_tools import ToolVerdictKind


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_all_relay_codecs_project_the_same_structural_contract(protocol: str) -> None:
    request, response = protocol_case(protocol)
    projector = codec_projection._NativeCodecProjector()

    decoded = projector.project_tools(request)
    calls = projector.project_response_tools(decoded, response)

    assert decoded.codec_names == (protocol,)
    assert decoded.projection.definitions[0].name == "weather"
    exchange = decoded.projection.exchanges[0]
    assert exchange.calls[0].call_id == "call-1"
    assert exchange.calls[0].arguments == {"city": "Paris"}
    assert exchange.results[0].call_id == "call-1"
    assert exchange.results[0].name is None
    assert calls[0].call_id == "call-2"


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_all_relay_codec_projections_pass_public_guardrails_checks(protocol: str) -> None:
    request, response = protocol_case(protocol)
    projector = codec_projection._NativeCodecProjector()

    decoded = projector.project_tools(request)
    calls = projector.project_response_tools(decoded, response)

    assert (await check_results(decoded.projection.exchanges)).kind is ToolVerdictKind.PASSED
    assert (await check_calls(decoded.projection.definitions, calls)).kind is ToolVerdictKind.PASSED


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_all_relay_codecs_cover_mixed_text_and_function_call_outputs(protocol: str) -> None:
    request, response = protocol_case(protocol)
    response = response_with_text(protocol, response, "safe output")
    projector = codec_projection._NativeCodecProjector()

    text_request = projector.project_text(
        request,
        allow_structural_tools=True,
        allow_tool_results=True,
        require_user=True,
    )
    tool_request = projector.project_tools(request)

    assert projector.project_response_text(text_request, response, allow_tool_calls=True) == "safe output"
    assert projector.project_response_tools(tool_request, response)[0].name == "weather"


@pytest.mark.parametrize(
    ("protocol", "invalid_role"),
    [
        ("openai_chat", "user"),
        ("anthropic_messages", "user"),
        ("gemini_generate_content", "user"),
        ("oci_genai", "USER"),
    ],
)
def test_response_function_calls_require_the_provider_assistant_role(
    protocol: str,
    invalid_role: str,
) -> None:
    request, response = protocol_case(protocol)
    broken = deepcopy(response)
    if protocol == "openai_chat":
        broken["choices"][0]["message"]["role"] = invalid_role  # type: ignore[index]
    elif protocol == "anthropic_messages":
        broken["role"] = invalid_role
    elif protocol == "gemini_generate_content":
        broken["candidates"][0]["content"]["role"] = invalid_role  # type: ignore[index]
    elif protocol == "oci_genai":
        broken["chatResponse"]["choices"][0]["message"]["role"] = invalid_role  # type: ignore[index]
    else:  # pragma: no cover - the parameter list is the protocol catalog
        raise AssertionError(protocol)
    projector = codec_projection._NativeCodecProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), broken)


def test_openai_and_anthropic_ambiguous_text_request_is_resolved_by_response() -> None:
    request = {"headers": {}, "content": {"model": "fixture", "messages": [{"role": "user", "content": "hi"}]}}
    response = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "function_call": None,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ]
    }
    projector = codec_projection._NativeCodecProjector()

    decoded = projector.project_tools(request)

    assert decoded.codec_names == ("openai_chat", "anthropic_messages")
    assert projector.project_response_tools(decoded, response) == ()


@pytest.mark.parametrize(
    ("protocol", "response"),
    [
        (
            "openai_chat",
            {
                "id": "chatcmpl-empty",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [],
            },
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
def test_empty_provider_candidates_do_not_create_false_tool_calls(
    protocol: str,
    response: dict[str, object],
) -> None:
    request, _ = protocol_case(protocol)
    projector = codec_projection._NativeCodecProjector()

    assert projector.project_response_tools(projector.project_tools(request), response) == ()


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_nested_unknown_request_extension_cannot_hide_tool_traffic(protocol: str) -> None:
    request, _ = protocol_case(protocol)
    content = request["content"]
    target = content["chatRequest"] if protocol == "oci_genai" else content  # type: ignore[index]
    target["extensions"] = {"mcp_call": {"name": "hidden"}}  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(request)


@pytest.mark.parametrize("protocol", ["openai_responses", "anthropic_messages", "oci_genai"])
def test_unknown_provider_item_cannot_hide_nested_tool_traffic(protocol: str) -> None:
    request, _ = protocol_case(protocol)
    broken = deepcopy(request)
    hidden = {"type": "future_state", "payload": {"tool_call": {"id": "hidden"}}}
    if protocol == "openai_responses":
        broken["content"]["input"].append(hidden)  # type: ignore[index,union-attr]
    elif protocol == "anthropic_messages":
        broken["content"]["messages"][0]["content"] = [hidden]  # type: ignore[index]
    else:
        broken["content"]["chatRequest"]["messages"][0]["content"].append(hidden)  # type: ignore[index,union-attr]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(broken)


@pytest.mark.parametrize("protocol", ["openai_chat", "gemini_generate_content", "oci_genai"])
def test_multiple_output_candidates_are_rejected(protocol: str) -> None:
    request, response = protocol_case(protocol)
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_tools(request)
    broken = deepcopy(response)
    if protocol == "openai_chat":
        broken["choices"].append(deepcopy(broken["choices"][0]))  # type: ignore[index,union-attr]
    elif protocol == "gemini_generate_content":
        broken["candidates"].append(deepcopy(broken["candidates"][0]))  # type: ignore[index,union-attr]
    else:
        chat_response = broken["chatResponse"]
        chat_response["choices"].append(deepcopy(chat_response["choices"][0]))  # type: ignore[index,union-attr]

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(decoded, broken)


@pytest.mark.parametrize("protocol", ["openai_chat", "gemini_generate_content", "oci_genai"])
def test_multiple_tool_candidates_are_projected_independently(protocol: str) -> None:
    request, response = protocol_case(protocol)
    projector = codec_projection._NativeCodecProjector()
    decoded = projector.project_tools(request)
    multiple = deepcopy(response)
    if protocol == "openai_chat":
        multiple["choices"].append(deepcopy(multiple["choices"][0]))  # type: ignore[index,union-attr]
    elif protocol == "gemini_generate_content":
        multiple["candidates"].append(deepcopy(multiple["candidates"][0]))  # type: ignore[index,union-attr]
    else:
        chat_response = multiple["chatResponse"]
        chat_response["choices"].append(deepcopy(chat_response["choices"][0]))  # type: ignore[index,union-attr]

    candidates = projector.project_response_tool_candidates(decoded, multiple)

    assert len(candidates) == 2
    assert [[call.name for call in candidate] for candidate in candidates] == [["weather"], ["weather"]]


@pytest.mark.parametrize("content", [{"result": "sunny"}, True, 7, 1.5])
@pytest.mark.parametrize("protocol", ["openai_responses", "anthropic_messages"])
def test_provider_invalid_scalar_or_object_tool_result_is_not_stringified(
    protocol: str,
    content: object,
) -> None:
    request, _ = protocol_case(protocol)
    if protocol == "openai_responses":
        request["content"]["input"][2]["output"] = content  # type: ignore[index]
    else:
        request["content"]["messages"][2]["content"][0]["content"] = content  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(request)


@pytest.mark.parametrize("protocol", ["openai_chat", "openai_responses"])
def test_duplicate_json_argument_keys_are_rejected(protocol: str) -> None:
    request, response = protocol_case(protocol)
    duplicate = '{"city":"unsafe","city":"safe"}'
    if protocol == "openai_chat":
        request["content"]["messages"][1]["tool_calls"][0]["function"]["arguments"] = duplicate  # type: ignore[index]
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = duplicate  # type: ignore[index]
    else:
        request["content"]["input"][1]["arguments"] = duplicate  # type: ignore[index]
        response["output"][0]["arguments"] = duplicate  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_tools(request)

    clean_request, _ = protocol_case(protocol)
    decoded = projector.project_tools(clean_request)
    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(decoded, response)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_empty_tool_descriptions_remain_valid(protocol: str) -> None:
    request, _ = protocol_case(protocol)
    content = request["content"]
    if protocol == "oci_genai":
        content["chatRequest"]["tools"][0]["description"] = ""  # type: ignore[index]
    elif protocol == "gemini_generate_content":
        content["tools"][0]["functionDeclarations"][0]["description"] = ""  # type: ignore[index]
    elif protocol == "openai_chat":
        content["tools"][0]["function"]["description"] = ""  # type: ignore[index]
    else:
        content["tools"][0]["description"] = ""  # type: ignore[index]

    projection = codec_projection._NativeCodecProjector().project_tools(request).projection

    assert projection.definitions[0].description == ""


@pytest.mark.parametrize(
    ("protocol", "field"),
    [
        ("openai_chat", "response"),
        ("anthropic_messages", "output_schema"),
        ("oci_genai", "response"),
    ],
)
def test_provider_specific_definition_controls_are_not_accepted_cross_codec(protocol: str, field: str) -> None:
    request, _ = protocol_case(protocol)
    content = request["content"]  # type: ignore[index]
    if protocol == "openai_chat":
        definition = content["tools"][0]["function"]  # type: ignore[index]
    elif protocol == "anthropic_messages":
        definition = content["tools"][0]  # type: ignore[index]
    else:
        definition = content["chatRequest"]["tools"][0]  # type: ignore[index]
    definition[field] = {"properties": {"tool": {"type": "string"}}}  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest, match="tool traffic is not completely covered"):
        codec_projection._NativeCodecProjector().project_tools(request)


def test_unmodeled_provider_controls_still_require_complete_text_coverage() -> None:
    gemini, _ = gemini_generate_content_case()
    gemini["content"]["safetySettings"] = [  # type: ignore[index]
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_LOW_AND_ABOVE"}
    ]
    oci, _ = oci_cohere_v2_case()
    oci["content"]["chatRequest"]["citationOptions"] = {"mode": "FAST"}  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()

    assert projector.project_tools(oci).projection.definitions[0].name == "weather"
    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project(gemini, allow_structural_tools=True, allow_tool_results=True)
    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project(oci, allow_structural_tools=True, allow_tool_results=True)


@pytest.mark.parametrize("protocol", ["gemini_generate_content", "oci_genai"])
def test_empty_provider_result_cannot_hide_a_mixed_protocol_call(protocol: str) -> None:
    request, _ = protocol_case(protocol)
    response: dict[str, object]
    if protocol == "gemini_generate_content":
        response = {"candidates": [], "output": [{"type": "function_call", "call_id": "hidden"}]}
    else:
        response = {
            "chatResponse": {"apiFormat": "GENERIC", "choices": []},
            "output": [{"type": "function_call", "call_id": "hidden"}],
        }
    projector = codec_projection._NativeCodecProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), response)


@pytest.mark.parametrize("protocol", ["openai_chat", "anthropic_messages", "oci_genai"])
def test_tool_finish_reason_without_a_projected_call_is_rejected(protocol: str) -> None:
    request, response = protocol_case(protocol)
    broken = deepcopy(response)
    if protocol == "openai_chat":
        broken["choices"][0]["message"].pop("tool_calls")  # type: ignore[index,union-attr]
        broken["choices"][0]["finish_reason"] = "tool_calls"  # type: ignore[index]
    elif protocol == "anthropic_messages":
        broken["content"] = [{"type": "text", "text": "hello"}]
        broken["stop_reason"] = "tool_use"
    else:
        broken["chatResponse"]["choices"][0]["message"].pop("toolCalls")  # type: ignore[index,union-attr]
        broken["chatResponse"]["choices"][0]["finishReason"] = "TOOL_CALL"  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), broken)
