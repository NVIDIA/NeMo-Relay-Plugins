# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest
from tool_projection_cases import (
    anthropic_messages_case,
    check_results,
    json_object_schema,
)

from nemoguardrails_nemo_relay import codec_projection
from nemoguardrails_nemo_relay.structural_tools import ToolVerdictKind


def test_anthropic_explicit_custom_tool_is_a_function_definition() -> None:
    request, response = anthropic_messages_case()
    custom = deepcopy(request)
    custom["content"]["tools"] = [  # type: ignore[index]
        {
            "type": "custom",
            "name": "weather",
            "description": "Get weather",
            "input_schema": json_object_schema(),
            "allowed_callers": ["direct"],
            "defer_loading": True,
            "eager_input_streaming": True,
            "input_examples": [{"city": "Paris"}],
        }
    ]
    projector = codec_projection._NativeCodecProjector()

    decoded = projector.project_tools(custom)
    calls = projector.project_response_tools(decoded, response)

    assert decoded.projection.definitions[0].name == "weather"
    assert decoded.projection.definitions[0].input_schema == json_object_schema()
    assert calls[0].name == "weather"


def test_standard_anthropic_controls_are_opaque_at_the_normalized_wrapper() -> None:
    request, response = anthropic_messages_case()
    request["content"]["tools"][0].update(  # type: ignore[index,union-attr]
        {
            "allowed_callers": ["direct"],
            "cache_control": {"type": "ephemeral"},
            "defer_loading": True,
            "eager_input_streaming": True,
            "input_examples": [{"tool": "hammer", "city": "Paris"}],
        }
    )
    projector = codec_projection._NativeCodecProjector()

    decoded = projector.project_tools(request)
    calls = projector.project_response_tools(decoded, response)

    assert decoded.projection.definitions[0].name == "weather"
    assert calls[0].name == "weather"


def test_mixed_anthropic_function_and_hosted_calls_are_rejected() -> None:
    request, response = anthropic_messages_case()
    broken = deepcopy(response)
    broken["content"].append(  # type: ignore[union-attr]
        {
            "type": "server_tool_use",
            "id": "hosted-1",
            "name": "web_search",
            "input": {"query": "weather"},
        }
    )
    projector = codec_projection._NativeCodecProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), broken)


@pytest.mark.parametrize("location", ["request", "response"])
def test_anthropic_tool_use_accepts_nullable_caller(location: str) -> None:
    request, response = anthropic_messages_case()
    if location == "request":
        request["content"]["messages"][1]["content"][0]["caller"] = None  # type: ignore[index]
    else:
        response["content"][0]["caller"] = None  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()

    decoded = projector.project_tools(request)
    calls = projector.project_response_tools(decoded, response)

    assert decoded.projection.exchanges[0].calls[0].name == "weather"
    assert calls[0].name == "weather"


@pytest.mark.parametrize("location", ["request", "response"])
def test_anthropic_tool_use_accepts_direct_caller(location: str) -> None:
    request, response = anthropic_messages_case()
    caller = {"type": "direct"}
    if location == "request":
        request["content"]["messages"][1]["content"][0]["caller"] = caller  # type: ignore[index]
    else:
        response["content"][0]["caller"] = caller  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()

    decoded = projector.project_tools(request)
    calls = projector.project_response_tools(decoded, response)

    assert decoded.projection.exchanges[0].calls[0].name == "weather"
    assert calls[0].name == "weather"


@pytest.mark.parametrize("location", ["request", "response"])
def test_anthropic_tool_use_rejects_server_caller(location: str) -> None:
    request, response = anthropic_messages_case()
    caller = {"type": "code_execution_20260120", "tool_id": "srvtoolu_1"}
    projector = codec_projection._NativeCodecProjector()
    if location == "request":
        request["content"]["messages"][1]["content"][0]["caller"] = caller  # type: ignore[index]
        with pytest.raises(codec_projection._UnsupportedRequest):
            projector.project_tools(request)
        return

    response["content"][0]["caller"] = caller  # type: ignore[index]
    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), response)


@pytest.mark.parametrize("location", ["request", "response"])
def test_anthropic_tool_use_accepts_nullable_toolset_name(location: str) -> None:
    request, response = anthropic_messages_case()
    if location == "request":
        request["content"]["messages"][1]["content"][0]["toolset_name"] = None  # type: ignore[index]
    else:
        response["content"][0]["toolset_name"] = None  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()

    decoded = projector.project_tools(request)
    calls = projector.project_response_tools(decoded, response)

    assert decoded.projection.exchanges[0].calls[0].name == "weather"
    assert calls[0].name == "weather"


@pytest.mark.parametrize(
    ("role", "block"),
    [
        ("user", {"type": "tool_use", "id": "call-1", "name": "weather", "input": {}}),
        ("assistant", {"type": "tool_result", "tool_use_id": "call-1", "content": "sunny"}),
    ],
)
def test_anthropic_tool_blocks_on_the_wrong_role_are_rejected(role: str, block: dict[str, object]) -> None:
    request, _ = anthropic_messages_case()
    broken = deepcopy(request)
    broken["content"]["messages"] = [{"role": role, "content": [block]}]  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(broken)


async def test_headerless_anthropic_tool_history_without_definitions_is_unambiguous() -> None:
    request, _ = anthropic_messages_case()
    request["headers"] = {}
    request["content"].pop("tools")  # type: ignore[union-attr]

    decoded = codec_projection._NativeCodecProjector().project_tools(request, require_definitions=False)

    assert decoded.codec_names == ("anthropic_messages",)
    assert decoded.projection.definitions == ()
    assert (await check_results(decoded.projection.exchanges)).kind is ToolVerdictKind.PASSED


def test_anthropic_tool_blocks_mixed_with_openai_tool_calls_are_rejected() -> None:
    request, _ = anthropic_messages_case()
    request["headers"] = {}
    request["content"].pop("tools")  # type: ignore[union-attr]
    request["content"]["messages"].append(  # type: ignore[index]
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-openai",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"Rome"}'},
                }
            ],
        }
    )

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(request, require_definitions=False)


def test_anthropic_function_call_alias_cannot_hide_tool_traffic() -> None:
    request, _ = anthropic_messages_case()
    broken = deepcopy(request)
    broken["content"]["messages"] = [  # type: ignore[index]
        {
            "role": "assistant",
            "content": [{"type": "function_call", "name": "weather", "arguments": {}}],
        }
    ]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(broken)


def test_anthropic_empty_tool_result_can_omit_content_without_request_mutation() -> None:
    request, _ = anthropic_messages_case()
    result = request["content"]["messages"][2]["content"][0]  # type: ignore[index]
    result.pop("content")  # type: ignore[union-attr]

    projection = codec_projection._NativeCodecProjector().project_tools(request).projection

    assert projection.exchanges[0].results[0].content is None
    assert "content" not in result


@pytest.mark.parametrize("location", ["request", "response"])
@pytest.mark.parametrize("toolset_name", ["", "computer"])
def test_anthropic_toolset_member_is_not_validated_as_an_unrelated_function(
    location: str,
    toolset_name: str,
) -> None:
    request, response = anthropic_messages_case()
    if location == "request":
        request["content"]["messages"][1]["content"][0]["toolset_name"] = toolset_name  # type: ignore[index]
        with pytest.raises(codec_projection._UnsupportedRequest):
            codec_projection._NativeCodecProjector().project_tools(request)
        return

    response["content"][0]["toolset_name"] = toolset_name  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()
    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), response)


def test_anthropic_context_editing_metadata_is_not_mistaken_for_a_call() -> None:
    request, response = anthropic_messages_case()
    response["content"] = [{"type": "text", "text": "hello"}]
    response["stop_reason"] = "end_turn"
    response["context_management"] = {"applied_edits": [{"type": "clear_tool_uses_20250919", "cleared_tool_uses": 2}]}
    projector = codec_projection._NativeCodecProjector()

    assert projector.project_response_tools(projector.project_tools(request), response) == ()


@pytest.mark.parametrize(
    ("target", "extra"),
    [
        ("message", {"tool_use_id": "hidden-call"}),
        ("tool_use", {"mcp_call": {"name": "hidden"}}),
        ("tool_result", {"function_call": {"name": "hidden"}}),
    ],
)
def test_anthropic_extensions_cannot_hide_tool_traffic(target: str, extra: dict[str, object]) -> None:
    request, _ = anthropic_messages_case()
    messages = request["content"]["messages"]  # type: ignore[index]
    if target == "message":
        messages[0].update(extra)  # type: ignore[index,union-attr]
    elif target == "tool_use":
        messages[1]["content"][0].update(extra)  # type: ignore[index,union-attr]
    else:
        messages[2]["content"][0].update(extra)  # type: ignore[index,union-attr]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(request)


def test_mixed_known_and_hosted_request_tool_blocks_are_rejected() -> None:
    request, _ = anthropic_messages_case()
    broken = deepcopy(request)
    broken["content"]["messages"][1]["content"].append(  # type: ignore[index]
        {
            "type": "server_tool_use",
            "id": "hosted-1",
            "name": "web_search",
            "input": {"query": "weather"},
        }
    )

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(broken)
