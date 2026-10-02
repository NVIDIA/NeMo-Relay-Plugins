# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from projection_fixtures import decoded_text, decoded_tools, response_text, response_tools
from provider_cases import chat_response, responses_request

from nemoguardrails_nemo_relay import (
    codec_projection,
)

_PROJECTOR = codec_projection._ProviderProjector()


def _response_text(codec_name: str, response: object, *, variant: str | None = None, tools: bool = False):
    return response_text(_PROJECTOR, decoded_text(codec_name, codec_variant=variant), response, allow_tool_calls=tools)


def _codex_metadata(**extra: str) -> dict[str, str]:
    return {
        "x-codex-installation-id": "installation-fixture",
        "session_id": "session-fixture",
        "thread_id": "thread-fixture",
        "x-codex-turn-metadata": json.dumps(
            {"request_kind": "turn", "session_id": "session-fixture"},
            separators=(",", ":"),
        ),
        **extra,
    }


def _native_responses_message(item: dict[str, object]) -> dict[str, object]:
    return {
        "role": "provider_native",
        "provider": "openai_responses",
        "kind": "message",
        "value": item,
    }


def _codex_message(item_id: str, role: str, text: str) -> dict[str, object]:
    return {
        "id": item_id,
        "type": "message",
        "role": role,
        "content": [{"type": "output_text" if role == "assistant" else "input_text", "text": text}],
    }


def test_codex_first_turn_projects_id_bearing_messages_and_bounded_metadata() -> None:
    metadata = _codex_metadata(future_codex_field="bounded extension")
    items = [
        _codex_message("msg-developer", "developer", "Follow repository instructions."),
        _codex_message("msg-user", "user", "Read README.md."),
    ]
    decoded = _PROJECTOR.project_host_text(
        responses_request(items, client_metadata=metadata),
        {
            "messages": [_native_responses_message(item) for item in items],
            "client_metadata": metadata,
        },
        require_user=True,
        codec_name="openai_responses",
    )

    assert decoded.messages == (
        {"role": "system", "content": "Follow repository instructions."},
        {"role": "user", "content": "Read README.md."},
    )


def test_codex_function_result_turn_preserves_text_and_structural_history() -> None:
    metadata = _codex_metadata(mcp_attribution='{"status":"complete"}')
    messages = [
        _native_responses_message(_codex_message("msg-developer", "developer", "Use read_file.")),
        _native_responses_message(_codex_message("msg-user-1", "user", "Read README.md.")),
        {
            "role": "tool_call",
            "id": "fc-1",
            "call_id": "call-1",
            "name": "read_file",
            "arguments": {"path": "README.md"},
        },
        {"role": "tool_result", "call_id": "call-1", "output": "Relay documentation"},
        _native_responses_message(_codex_message("msg-user-2", "user", "Summarize it.")),
    ]
    items = [
        _codex_message("msg-developer", "developer", "Use read_file."),
        _codex_message("msg-user-1", "user", "Read README.md."),
        {
            "id": "fc-1",
            "type": "function_call",
            "call_id": "call-1",
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "Relay documentation",
        },
        _codex_message("msg-user-2", "user", "Summarize it."),
    ]
    tool = {
        "type": "function",
        "name": "read_file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    }
    annotation = {
        "messages": messages,
        "tools": [{"type": "function", "function": {key: value for key, value in tool.items() if key != "type"}}],
        "client_metadata": metadata,
    }
    request = responses_request(items, tools=[tool], client_metadata=metadata)

    text = _PROJECTOR.project_host_text(
        request,
        annotation,
        allow_structural_tools=True,
        allow_tool_results=True,
        require_user=True,
        codec_name="openai_responses",
    )
    tools = _PROJECTOR.project_host_tools(request, annotation, codec_name="openai_responses")

    assert text.messages[-1] == {"role": "user", "content": "Summarize it."}
    assert tools.projection.definitions[0].name == "read_file"
    assert tools.projection.exchanges[0].calls[0].call_id == "call-1"
    assert tools.projection.exchanges[0].results[0].content == "Relay documentation"


@pytest.mark.parametrize("role", ["system", "developer", "user", "assistant"])
def test_responses_id_bearing_message_supports_text_roles(role: str) -> None:
    item = _codex_message("msg-1", role, "covered")
    decoded = _PROJECTOR.project_host_text(
        responses_request([item]),
        {"messages": [_native_responses_message(item)]},
        require_user=False,
        codec_name="openai_responses",
    )

    assert decoded.messages == ({"role": "system" if role == "developer" else role, "content": "covered"},)


@pytest.mark.parametrize("block_type", ["custom_tool_call", "computer_call", "mcp_call", "namespace_call"])
def test_responses_id_bearing_message_rejects_structural_blocks(block_type: str) -> None:
    item = {
        "id": "msg-1",
        "type": "message",
        "role": "user",
        "content": [{"type": block_type, "name": "unsupported"}],
    }
    with pytest.raises(codec_projection._UnsupportedRequest):
        _PROJECTOR.project_host_text(
            responses_request([item]),
            {"messages": [_native_responses_message(item)]},
            require_user=True,
            codec_name="openai_responses",
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"thought": False},
        {"thoughtSignature": ""},
        {"partMetadata": {}},
    ],
)
def test_gemini_output_checks_all_text_when_empty_metadata_forces_normalized_parts(
    metadata: dict[str, object],
) -> None:
    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "one", **metadata}, {"text": "two"}],
                }
            }
        ]
    }
    assert _response_text("gemini_generate_content", response) == "one\ntwo"


def test_openai_responses_output_preserves_ordered_refusal_and_text_for_guardrails() -> None:
    response = {
        "id": "resp-fixture",
        "object": "response",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "refusal", "refusal": "declined"},
                    {
                        "type": "output_text",
                        "text": "safe alternative",
                        "annotations": [],
                        "logprobs": [],
                    },
                ],
            }
        ],
        # The Responses aggregate contains output_text blocks, not refusals.
        "output_text": "safe alternative",
    }
    assert _response_text("openai_responses", response) == "declined\nsafe alternative"


@pytest.mark.parametrize("phase", ["commentary", "final_answer"])
def test_openai_responses_output_accepts_message_phase(phase: str) -> None:
    response = {
        "id": "resp-fixture",
        "object": "response",
        "output": [
            {
                "id": "msg-fixture",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": phase,
                "content": [{"type": "output_text", "text": "safe", "annotations": []}],
            }
        ],
        "output_text": "safe",
    }
    assert _response_text("openai_responses", response) == "safe"


@pytest.mark.parametrize("aggregate", ["", None])
def test_openai_responses_refusal_accepts_only_an_empty_aggregate(
    aggregate: str | None,
) -> None:
    response: dict[str, object] = {
        "id": "resp-fixture",
        "object": "response",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "declined"}],
            }
        ],
    }
    if aggregate is not None:
        response["output_text"] = aggregate
    assert _response_text("openai_responses", response) == "declined"


def test_openai_responses_refusal_rejects_unrepresented_aggregate_text() -> None:
    response = {
        "id": "resp-fixture",
        "object": "response",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "declined"}],
            }
        ],
        "output_text": "unchecked application text",
    }
    with pytest.raises(codec_projection._UnsupportedRequest):
        _response_text("openai_responses", response)


@pytest.mark.parametrize("api_format", ["GENERIC", "COHERE", "COHEREV2"])
def test_oci_output_accepts_nullable_tool_call_arrays(api_format: str) -> None:
    if api_format == "COHERE":
        response = {
            "chatResponse": {
                "apiFormat": api_format,
                "text": "safe",
                "toolCalls": None,
            }
        }
    elif api_format == "GENERIC":
        response = {
            "chatResponse": {
                "apiFormat": api_format,
                "choices": [
                    {
                        "message": {
                            "role": "ASSISTANT",
                            "content": [{"type": "TEXT", "text": "safe"}],
                            "toolCalls": None,
                        }
                    }
                ],
            }
        }
    else:
        response = {
            "chatResponse": {
                "apiFormat": api_format,
                "message": {
                    "role": "ASSISTANT",
                    "content": [{"type": "TEXT", "text": "safe"}],
                    "toolCalls": None,
                },
            }
        }
    text_request = decoded_text("oci_genai", codec_variant=api_format)
    tool_request = decoded_tools("oci_genai", codec_variant=api_format, definitions=())

    assert response_text(_PROJECTOR, text_request, response, allow_tool_calls=True) == "safe"
    assert response_tools(_PROJECTOR, tool_request, response) == ()


@pytest.mark.parametrize("logprobs", [None, [], {}])
def test_openai_chat_output_accepts_empty_logprobs(logprobs: object) -> None:
    response = chat_response()
    response["choices"][0]["logprobs"] = logprobs  # type: ignore[index]
    assert _response_text("openai_chat", response) == "safe"
