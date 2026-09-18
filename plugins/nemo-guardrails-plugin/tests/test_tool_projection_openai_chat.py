# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest
from tool_projection_cases import (
    check_results,
    openai_chat_case,
)

from nemoguardrails_nemo_relay import codec_projection


async def test_adjacent_chat_assistant_turns_are_not_merged_into_one_exchange() -> None:
    request, _ = openai_chat_case()
    messages = request["content"]["messages"]  # type: ignore[index]
    second_call = deepcopy(messages[1])
    second_call["tool_calls"][0]["id"] = "call-2"  # type: ignore[index]
    messages.insert(2, second_call)  # type: ignore[union-attr]
    messages.append({"role": "tool", "tool_call_id": "call-2", "content": "rainy"})  # type: ignore[union-attr]

    projection = codec_projection._NativeCodecProjector().project_tools(request).projection
    verdict = await check_results(projection.exchanges)

    assert len(projection.exchanges) == 2
    assert verdict.kind.name == "POLICY_BLOCK"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("refusal", None),
        ("refusal", "I cannot answer directly."),
        ("audio", {"id": "audio-1"}),
        ("refusal_and_audio", ("I cannot answer directly.", {"id": "audio-1"})),
    ],
)
def test_openai_chat_assistant_metadata_does_not_hide_valid_tool_history(
    field: str,
    value: object,
) -> None:
    request, _ = openai_chat_case()
    assistant = request["content"]["messages"][1]  # type: ignore[index]
    if field == "refusal_and_audio":
        refusal, audio = value  # type: ignore[misc]
        assistant["refusal"] = refusal  # type: ignore[index]
        assistant["audio"] = audio  # type: ignore[index]
    else:
        assistant[field] = value  # type: ignore[index]
    original = deepcopy(request)

    projection = codec_projection._NativeCodecProjector().project_tools(request).projection

    assert projection.exchanges[0].calls[0].call_id == "call-1"
    assert projection.exchanges[0].calls[0].name == "weather"
    assert request == original


def test_legacy_openai_function_call_is_not_misreported_as_checked() -> None:
    request, _ = openai_chat_case()
    response = {
        "id": "chatcmpl-legacy",
        "object": "chat.completion",
        "created": 1,
        "model": "fixture",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": "weather", "arguments": '{"city":"Rome"}'},
                },
                "finish_reason": "function_call",
            }
        ],
    }
    projector = codec_projection._NativeCodecProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), response)


def test_non_function_openai_chat_call_is_not_retyped_as_a_function() -> None:
    request, response = openai_chat_case()
    broken = deepcopy(response)
    broken["choices"][0]["message"]["tool_calls"][0]["type"] = "hosted_search"  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), broken)


@pytest.mark.parametrize("location", ["request", "response"])
def test_missing_openai_chat_arguments_are_not_defaulted_to_an_empty_object(location: str) -> None:
    request, response = openai_chat_case()
    if location == "request":
        function = request["content"]["messages"][1]["tool_calls"][0]["function"]  # type: ignore[index]
        del function["arguments"]  # type: ignore[index]
        with pytest.raises(codec_projection._UnsupportedRequest):
            codec_projection._NativeCodecProjector().project_tools(request)
        return

    function = response["choices"][0]["message"]["tool_calls"][0]["function"]  # type: ignore[index]
    del function["arguments"]  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()
    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), response)


def test_non_function_openai_chat_history_call_is_not_silently_dropped() -> None:
    request, _ = openai_chat_case()
    broken = deepcopy(request)
    broken["content"]["messages"][1]["tool_calls"][0]["type"] = "custom"  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(broken)


def test_openai_chat_content_part_cannot_hide_tool_traffic() -> None:
    request = {
        "headers": {},
        "content": {
            "model": "fixture",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "tool_call", "name": "weather", "arguments": "{}"}],
                }
            ],
        },
    }

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(request)


@pytest.mark.parametrize("role", ["user", "assistant", "system"])
def test_openai_chat_tool_result_id_on_wrong_role_is_rejected(role: str) -> None:
    request = {
        "headers": {},
        "content": {
            "model": "fixture",
            "messages": [{"role": role, "content": "hello", "tool_call_id": "hidden-call"}],
        },
    }

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(request)


@pytest.mark.parametrize("field", ["custom_tool_calls", "tool_call", "server_tool_use"])
def test_unmodeled_openai_chat_response_tool_fields_are_rejected(field: str) -> None:
    request, response = openai_chat_case()
    broken = deepcopy(response)
    message = broken["choices"][0]["message"]  # type: ignore[index]
    message.pop("tool_calls")  # type: ignore[union-attr]
    message[field] = {"name": "weather", "arguments": "{}"}  # type: ignore[index]
    projector = codec_projection._NativeCodecProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), broken)


def test_mixed_openai_response_envelopes_are_rejected() -> None:
    request, response = openai_chat_case()
    broken = deepcopy(response)
    broken["choices"][0]["message"].pop("tool_calls")  # type: ignore[index,union-attr]
    broken["output"] = [
        {
            "type": "function_call",
            "call_id": "hidden-call",
            "name": "weather",
            "arguments": "{}",
        }
    ]
    projector = codec_projection._NativeCodecProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), broken)


@pytest.mark.parametrize(
    ("location", "payload"),
    [
        ("call", {"mcp_call": {"name": "hidden"}}),
        ("function", {"tool": {"name": "hidden"}}),
    ],
)
def test_structural_extensions_inside_openai_chat_calls_are_rejected(
    location: str,
    payload: dict[str, object],
) -> None:
    request, response = openai_chat_case()
    call = response["choices"][0]["message"]["tool_calls"][0]  # type: ignore[index]
    target = call if location == "call" else call["function"]  # type: ignore[index]
    target.update(payload)  # type: ignore[union-attr]
    projector = codec_projection._NativeCodecProjector()

    with pytest.raises(codec_projection._UnsupportedRequest):
        projector.project_response_tools(projector.project_tools(request), response)


def test_non_json_openai_number_is_rejected_during_projection() -> None:
    request, _ = openai_chat_case()
    broken = deepcopy(request)
    broken["content"]["messages"][1]["tool_calls"][0]["function"]["arguments"] = '{"city":NaN}'  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(broken)


def test_malformed_tool_arguments_are_rejected_during_projection() -> None:
    request, _ = openai_chat_case()
    broken = deepcopy(request)
    broken["content"]["messages"][1]["tool_calls"][0]["function"]["arguments"] = "not-json"  # type: ignore[index]

    with pytest.raises(codec_projection._UnsupportedRequest):
        codec_projection._NativeCodecProjector().project_tools(broken)
