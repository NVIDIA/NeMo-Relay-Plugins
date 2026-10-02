# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anthropic-specific structural coverage above Relay's host codec."""

from __future__ import annotations

from copy import deepcopy

import pytest
from projection_fixtures import decoded_tools, response_tools, weather_annotation
from tool_projection_cases import anthropic_messages_case, check_results, json_object_schema

from nemoguardrails_nemo_relay import codec_projection
from nemoguardrails_nemo_relay.structural_tools import ToolVerdictKind


def _project_host(request: dict[str, object], annotation: dict[str, object] | None = None):
    return codec_projection._ProviderProjector().project_host_tools(
        request,
        annotation or weather_annotation("anthropic_messages"),
        codec_name="anthropic_messages",
    )


def test_normalized_custom_tool_maps_to_a_function_definition() -> None:
    request, _ = anthropic_messages_case()
    annotation = weather_annotation("anthropic_messages")
    annotation["tools"] = [
        {
            "type": "provider_native",
            "provider": "anthropic_messages",
            "kind": "custom",
            "value": {
                "type": "custom",
                "name": "weather",
                "description": "Get weather",
                "input_schema": json_object_schema(),
                "allowed_callers": ["direct"],
                "defer_loading": True,
                "eager_input_streaming": True,
                "input_examples": [{"city": "Paris"}],
            },
        }
    ]

    definition = _project_host(request, annotation).projection.definitions[0]

    assert definition.name == "weather"
    assert definition.input_schema == json_object_schema()


def test_normalized_anthropic_definition_controls_are_not_treated_as_tool_traffic() -> None:
    request, _ = anthropic_messages_case()
    annotation = weather_annotation("anthropic_messages")
    annotation["tools"][0].update(  # type: ignore[index,union-attr]
        {
            "allowed_callers": ["direct"],
            "cache_control": {"type": "ephemeral"},
            "defer_loading": True,
            "eager_input_streaming": True,
            "input_examples": [{"city": "Paris"}],
        }
    )

    assert _project_host(request, annotation).projection.definitions[0].name == "weather"


def test_normalized_empty_tool_result_remains_linked_without_mutation() -> None:
    request, _ = anthropic_messages_case()
    annotation = weather_annotation("anthropic_messages")
    result = annotation["messages"][2]["content"][0]  # type: ignore[index]
    result.pop("content")  # type: ignore[union-attr]
    original = deepcopy(annotation)

    projection = _project_host(request, annotation).projection

    assert projection.exchanges[0].results[0].call_id == "call-1"
    assert projection.exchanges[0].results[0].content is None
    assert annotation == original


@pytest.mark.parametrize("location", ["request", "response"])
@pytest.mark.parametrize("caller", [None, {"type": "direct"}])
def test_direct_or_nullable_caller_is_covered(location: str, caller: object) -> None:
    request, response = anthropic_messages_case()
    if location == "request":
        request["content"]["messages"][1]["content"][0]["caller"] = caller  # type: ignore[index]
        assert _project_host(request).projection.exchanges[0].calls[0].name == "weather"
        return

    response["content"][0]["caller"] = caller  # type: ignore[index]
    calls = response_tools(
        codec_projection._ProviderProjector(),
        decoded_tools("anthropic_messages"),
        response,
    )
    assert calls[0].name == "weather"


@pytest.mark.parametrize("location", ["request", "response"])
def test_server_caller_is_not_misreported_as_a_checked_function(location: str) -> None:
    request, response = anthropic_messages_case()
    caller = {"type": "code_execution_20260120", "tool_id": "srvtoolu_1"}
    if location == "request":
        request["content"]["messages"][1]["content"][0]["caller"] = caller  # type: ignore[index]
        with pytest.raises(codec_projection._UnsupportedRequest):
            _project_host(request)
        return

    response["content"][0]["caller"] = caller  # type: ignore[index]
    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools("anthropic_messages"),
            response,
        )


@pytest.mark.parametrize("location", ["request", "response"])
def test_toolset_member_is_outside_the_function_tool_contract(location: str) -> None:
    request, response = anthropic_messages_case()
    if location == "request":
        request["content"]["messages"][1]["content"][0]["toolset_name"] = "computer"  # type: ignore[index]
        with pytest.raises(codec_projection._UnsupportedRequest):
            _project_host(request)
        return

    response["content"][0]["toolset_name"] = "computer"  # type: ignore[index]
    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools("anthropic_messages"),
            response,
        )


def test_hosted_tool_response_mixed_with_function_call_is_rejected() -> None:
    _, response = anthropic_messages_case()
    response["content"].append(  # type: ignore[union-attr]
        {
            "type": "server_tool_use",
            "id": "hosted-1",
            "name": "web_search",
            "input": {"query": "weather"},
        }
    )

    with pytest.raises(codec_projection._UnsupportedRequest):
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools("anthropic_messages"),
            response,
        )


def test_hosted_tool_history_mixed_with_function_call_is_rejected() -> None:
    request, _ = anthropic_messages_case()
    request["content"]["messages"][1]["content"].append(  # type: ignore[index,union-attr]
        {
            "type": "server_tool_use",
            "id": "hosted-1",
            "name": "web_search",
            "input": {"query": "weather"},
        }
    )

    with pytest.raises(codec_projection._UnsupportedRequest):
        _project_host(request)


@pytest.mark.parametrize(
    ("target", "extra"),
    [
        ("message", {"tool_use_id": "hidden-call"}),
        ("tool_use", {"mcp_call": {"name": "hidden"}}),
        ("tool_result", {"function_call": {"name": "hidden"}}),
    ],
)
def test_raw_request_extensions_cannot_hide_tool_traffic(target: str, extra: dict[str, object]) -> None:
    request, _ = anthropic_messages_case()
    messages = request["content"]["messages"]  # type: ignore[index]
    if target == "message":
        messages[0].update(extra)  # type: ignore[index,union-attr]
    elif target == "tool_use":
        messages[1]["content"][0].update(extra)  # type: ignore[index,union-attr]
    else:
        messages[2]["content"][0].update(extra)  # type: ignore[index,union-attr]

    with pytest.raises(codec_projection._UnsupportedRequest):
        _project_host(request)


def test_context_editing_metadata_is_not_mistaken_for_a_call() -> None:
    _, response = anthropic_messages_case()
    response["content"] = [{"type": "text", "text": "hello"}]
    response["stop_reason"] = "end_turn"
    response["context_management"] = {"applied_edits": [{"type": "clear_tool_uses_20250919", "cleared_tool_uses": 2}]}

    assert (
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools("anthropic_messages"),
            response,
        )
        == ()
    )


def test_response_tool_use_requires_an_input_object() -> None:
    _, response = anthropic_messages_case()
    response["content"][0].pop("input")  # type: ignore[index,union-attr]

    with pytest.raises(codec_projection._UnsupportedRequest, match="arguments must be an object"):
        response_tools(
            codec_projection._ProviderProjector(),
            decoded_tools("anthropic_messages"),
            response,
        )


async def test_normalized_result_linkage_reaches_guardrails() -> None:
    request, _ = anthropic_messages_case()
    projection = _project_host(request).projection

    assert (await check_results(projection.exchanges)).kind is ToolVerdictKind.PASSED
