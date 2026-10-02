# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest
from projection_fixtures import decoded_tools, response_tools, weather_annotation
from tool_projection_cases import (
    check_calls,
    check_results,
    json_object_schema,
    openai_responses_case,
)

from nemoguardrails_nemo_relay import codec_projection


def _project(
    request: dict[str, object],
    annotation: dict[str, object] | None = None,
) -> codec_projection._DecodedToolRequest:
    return codec_projection._ProviderProjector().project_host_tools(
        request,
        weather_annotation("openai_responses") if annotation is None else annotation,
        codec_name="openai_responses",
    )


def test_responses_invocation_and_output_controls_remain_outside_structural_scope() -> None:
    request, response = openai_responses_case()
    controlled = deepcopy(request)
    controlled["content"]["tools"][0].update(  # type: ignore[index,union-attr]
        {
            "allowed_callers": ["programmatic"],
            "async": True,
            "defer_loading": True,
            "output_schema": {
                "type": "object",
                "properties": {"tool": {"type": "string"}},
            },
        }
    )
    projector = codec_projection._ProviderProjector()
    annotation = weather_annotation("openai_responses")
    annotation["tools"][0]["function"].update(  # type: ignore[index,union-attr]
        {
            "allowed_callers": ["programmatic"],
            "async": True,
            "defer_loading": True,
            "output_schema": {
                "type": "object",
                "properties": {"tool": {"type": "string"}},
            },
        }
    )

    decoded = _project(controlled, annotation)
    calls = response_tools(projector, decoded, response)

    assert decoded.projection.definitions[0].name == "weather"
    assert calls[0].name == "weather"


def test_unknown_definition_extension_cannot_hide_tool_structure() -> None:
    request, _ = openai_responses_case()
    request["content"]["tools"][0]["future_metadata"] = {  # type: ignore[index]
        "tool": {"type": "server_side"}
    }
    annotation = weather_annotation("openai_responses")
    annotation["tools"][0]["function"]["future_metadata"] = {  # type: ignore[index]
        "tool": {"type": "server_side"}
    }

    with pytest.raises(codec_projection._UnsupportedRequest):
        _project(request, annotation)


def test_openai_responses_output_text_fallback_has_no_tool_calls() -> None:
    request, _ = openai_responses_case()
    response = {
        "id": "resp-output-text",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "output_text": "safe",
    }
    projector = codec_projection._ProviderProjector()

    assert response_tools(projector, decoded_tools("openai_responses"), response) == ()


async def test_parallel_openai_responses_calls_and_results_share_one_exchange() -> None:
    request, _ = openai_responses_case()
    annotation = weather_annotation("openai_responses")
    annotation["messages"] = [
        {"role": "user", "content": "weather"},
        {
            "role": "tool_call",
            "call_id": "call-1",
            "name": "weather",
            "arguments": {"city": "Paris"},
        },
        {
            "role": "tool_call",
            "call_id": "call-2",
            "name": "weather",
            "arguments": {"city": "Rome"},
        },
        {"role": "tool_result", "call_id": "call-1", "output": "sunny"},
        {"role": "tool_result", "call_id": "call-2", "output": "rainy"},
    ]
    decoded = _project(request, annotation)

    assert len(decoded.projection.exchanges) == 1
    assert [call.call_id for call in decoded.projection.exchanges[0].calls] == ["call-1", "call-2"]
    assert [result.call_id for result in decoded.projection.exchanges[0].results] == ["call-1", "call-2"]

    verdict = await check_results(decoded.projection.exchanges)
    assert verdict.passed


async def test_responses_reasoning_item_does_not_break_call_result_linkage() -> None:
    request, _ = openai_responses_case()
    annotation = weather_annotation("openai_responses")
    annotation["messages"].insert(  # type: ignore[union-attr]
        2,
        {
            "role": "provider_native",
            "provider": "openai_responses",
            "kind": "reasoning",
            "value": {"type": "reasoning", "id": "rs-1", "summary": []},
        },
    )

    projection = _project(request, annotation).projection
    verdict = await check_results(projection.exchanges)

    assert len(projection.exchanges) == 1
    assert verdict.passed


def test_unused_hosted_tool_definition_is_left_outside_the_function_allowlist() -> None:
    request = {
        "headers": {},
        "content": {
            "model": "fixture",
            "input": "search",
            "tools": [{"type": "web_search_preview"}],
        },
    }

    projection = (
        codec_projection._ProviderProjector()
        .project_host_tools(
            request,
            {
                "messages": [{"role": "user", "content": "search"}],
                "tools": [
                    {
                        "type": "provider_native",
                        "provider": "openai_responses",
                        "kind": "web_search_preview",
                        "value": {"type": "web_search_preview"},
                    }
                ],
            },
            codec_name="openai_responses",
        )
        .projection
    )

    assert projection.definitions == ()


async def test_function_call_can_be_checked_when_a_hosted_tool_is_also_declared() -> None:
    request, response = openai_responses_case()
    request["content"]["tools"].append({"type": "web_search_preview"})  # type: ignore[index]
    projector = codec_projection._ProviderProjector()
    annotation = weather_annotation("openai_responses")
    annotation["tools"].append(  # type: ignore[union-attr]
        {
            "type": "provider_native",
            "provider": "openai_responses",
            "kind": "web_search_preview",
            "value": {"type": "web_search_preview"},
        }
    )

    decoded = _project(request, annotation)
    calls = response_tools(projector, decoded, response)
    verdict = await check_calls(decoded.projection.definitions, calls)

    assert [definition.name for definition in decoded.projection.definitions] == ["weather"]
    assert verdict.passed


def test_hosted_response_tool_output_is_not_hidden_by_normalized_codec() -> None:
    response = {
        "id": "resp-1",
        "status": "completed",
        "output": [{"type": "web_search_call", "id": "search-1", "status": "completed"}],
    }
    projector = codec_projection._ProviderProjector()
    decoded = decoded_tools("openai_responses", definitions=())

    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(projector, decoded, response)


def test_openai_responses_item_reference_cannot_hide_prior_tool_state() -> None:
    request = {
        "headers": {},
        "content": {
            "model": "fixture",
            "input": [{"type": "item_reference", "id": "fc_from_hidden_history"}],
        },
    }

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project_host_tools(
            request,
            {
                "messages": [
                    {
                        "role": "provider_native",
                        "provider": "openai_responses",
                        "kind": "item_reference",
                        "value": {"type": "item_reference", "id": "fc_from_hidden_history"},
                    }
                ]
            },
            codec_name="openai_responses",
        )


@pytest.mark.parametrize("block_type", ["tool_call", "custom_tool_call", "computer_call", "mcp_call"])
def test_openai_responses_message_item_cannot_hide_tool_traffic(block_type: str) -> None:
    request = {
        "headers": {},
        "content": {
            "model": "fixture",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": block_type, "name": "weather", "arguments": "{}"}],
                }
            ],
        },
    }

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project_host_tools(
            request,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": block_type, "name": "weather", "arguments": {}}],
                    }
                ]
            },
            codec_name="openai_responses",
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda item: item.update({"call_id": "hidden-call"}),
        lambda item: item["content"][0].update({"tool_call_id": "hidden-call"}),
    ],
)
def test_openai_responses_message_extensions_cannot_hide_tool_links(mutate: object) -> None:
    item: dict[str, object] = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hello"}],
    }
    mutate(item)  # type: ignore[operator]
    request = {"headers": {}, "content": {"model": "fixture", "input": [item]}}

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._ProviderProjector().project_host_tools(
            request,
            {"messages": [item]},
            codec_name="openai_responses",
        )


def test_openai_responses_result_extensions_cannot_hide_tool_calls() -> None:
    request, _ = openai_responses_case()
    result = request["content"]["input"][2]  # type: ignore[index]
    result["mcp_call"] = {"name": "hidden"}  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        _project(request)


@pytest.mark.parametrize(
    "response",
    [
        {
            "output": [
                {
                    "type": "output_text",
                    "text": "safe",
                    "futureMetadata": {"futureToolCall": {"name": "hidden"}},
                }
            ]
        },
        {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": "safe",
                            "futureToolCall": {"name": "hidden"},
                        }
                    ],
                }
            ]
        },
    ],
    ids=["output-text", "reasoning-block"],
)
def test_openai_responses_output_extensions_cannot_hide_tool_calls(response: dict[str, object]) -> None:
    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools("openai_responses"),
            response,
        )


@pytest.mark.parametrize("field", ["namespace", "caller", "async"])
@pytest.mark.parametrize("location", ["request", "response"])
def test_openai_responses_unmodeled_function_identity_is_rejected(
    location: str,
    field: str,
) -> None:
    request, response = openai_responses_case()
    if location == "request":
        request["content"]["input"][1][field] = "crm"  # type: ignore[index]
        with pytest.raises(codec_projection._UnsupportedRequest):
            _project(request)
        return

    response["output"][0][field] = "crm"  # type: ignore[index]
    projector = codec_projection._ProviderProjector()
    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(projector, decoded_tools("openai_responses"), response)


@pytest.mark.parametrize("block_type", ["tool_call", "custom_tool_call"])
def test_openai_responses_output_message_cannot_hide_tool_traffic(block_type: str) -> None:
    response = {
        "id": "resp-1",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": block_type, "name": "weather", "arguments": "{}"}],
            }
        ],
    }
    projector = codec_projection._ProviderProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(projector, decoded_tools("openai_responses", definitions=()), response)


def test_openai_responses_echoed_tool_configuration_is_not_mistaken_for_a_call() -> None:
    response = {
        "id": "resp-1",
        "status": "completed",
        "parallel_tool_calls": True,
        "max_tool_calls": 4,
        "tool_choice": "auto",
        "tools": [{"type": "function", "name": "weather", "parameters": json_object_schema()}],
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}],
            }
        ],
    }
    projector = codec_projection._ProviderProjector()

    assert response_tools(projector, decoded_tools("openai_responses", definitions=()), response) == ()
