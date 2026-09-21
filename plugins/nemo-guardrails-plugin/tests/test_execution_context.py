# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_relay_plugin import PluginContext
from provider_cases import chat_request

from nemoguardrails_nemo_relay import execution_policy, worker
from nemoguardrails_nemo_relay.codec_projection import _ProviderProjector, _UnsupportedRequest
from nemoguardrails_nemo_relay.execution_context import (
    ExecutionCodecContextError,
    execution_codec_context,
)


def _direction(kind: str, codec_name: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(codec=SimpleNamespace(kind=kind, id=codec_name))


def _context(
    request_kind: str,
    request_codec: str | None,
    response_kind: str,
    response_codec: str | None,
    annotated_request: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        request=_direction(request_kind, request_codec),
        response=_direction(response_kind, response_codec),
        annotated_request=annotated_request,
    )


_HOST_ANNOTATIONS: dict[str, dict[str, object]] = {
    "openai_chat": {
        "api_specific": {"api": "openai_chat"},
        "messages": [{"role": "user", "content": "hello"}],
        "model": "fixture",
    },
    "openai_responses": {
        "api_specific": {"api": "openai_responses"},
        "messages": [{"role": "user", "content": "hello"}],
        "model": "fixture",
    },
    "anthropic_messages": {
        "api_specific": {"api": "anthropic_messages"},
        "messages": [{"role": "user", "content": "hello"}],
        "model": "fixture",
    },
    "gemini_generate_content": {
        "messages": [{"role": "user", "content": "hello"}],
    },
    "oci_genai": {
        "api_specific": {"api": "oci_genai", "api_format": "GENERIC"},
        "messages": [{"role": "user", "content": "hello"}],
    },
}


def test_context_accepts_all_relay_builtin_codecs_and_legacy_hosts() -> None:
    for codec_name in (
        "openai_chat",
        "openai_responses",
        "anthropic_messages",
        "oci_genai",
        "gemini_generate_content",
    ):
        parsed = execution_codec_context(_context("builtin", codec_name, "builtin", codec_name))
        assert parsed.request.supported_name("request") == codec_name
        assert parsed.response.supported_name("response") == codec_name

    legacy = execution_codec_context(_context("none", None, "none", None))
    assert legacy.request.supported_name("request") is None
    assert legacy.response.supported_name("response") is None
    assert not legacy.allow_shape_inference

    compatibility = execution_policy.ExecutionCodecContext.legacy()
    assert compatibility.allow_shape_inference


@pytest.mark.parametrize(
    ("kind", "codec_name"),
    [
        ("runtime", "com.example.codec"),
        ("opaque", None),
        ("builtin", "future_provider"),
    ],
)
def test_context_does_not_turn_known_unknown_codecs_into_shape_guesses(kind: str, codec_name: str | None) -> None:
    parsed = execution_codec_context(_context(kind, codec_name, "none", None))
    with pytest.raises(ExecutionCodecContextError, match="request codec"):
        parsed.request.supported_name("request")


@pytest.mark.parametrize(
    "context",
    [
        SimpleNamespace(request=None, response=_direction("none"), annotated_request=None),
        _context("builtin", None, "none", None),
        _context("none", "unexpected", "none", None),
        _context("future", None, "none", None),
    ],
)
def test_context_rejects_malformed_identities(context: SimpleNamespace) -> None:
    with pytest.raises(ExecutionCodecContextError):
        parsed = execution_codec_context(context)
        parsed.request.supported_name("request")


@pytest.mark.parametrize(
    ("codec_name", "native_request"),
    [
        ("openai_chat", chat_request(include_response_format=False)),
        ("openai_responses", {"headers": {}, "content": {"model": "fixture", "input": "hello"}}),
        (
            "anthropic_messages",
            {
                "headers": {"anthropic-version": "2023-06-01"},
                "content": {
                    "model": "fixture",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            },
        ),
        (
            "gemini_generate_content",
            {"headers": {}, "content": {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}},
        ),
        (
            "oci_genai",
            {
                "headers": {},
                "content": {
                    "chatRequest": {
                        "apiFormat": "GENERIC",
                        "messages": [{"role": "USER", "content": [{"type": "TEXT", "text": "hello"}]}],
                    }
                },
            },
        ),
    ],
)
def test_authoritative_codec_and_annotation_must_agree_on_policy_inputs(
    codec_name: str,
    native_request: dict[str, object],
) -> None:
    projector = _ProviderProjector()
    host_annotation = deepcopy(_HOST_ANNOTATIONS[codec_name])
    projected = projector.project_text(
        native_request,
        require_user=True,
        codec_name=codec_name,
        annotated_request=host_annotation,
    )
    assert projected.codec_names == (codec_name,)
    assert projected.messages[-1] == {"role": "user", "content": "hello"}

    stale = deepcopy(host_annotation)
    stale["messages"][-1]["content"] = "different prompt"  # type: ignore[index]
    with pytest.raises(_UnsupportedRequest, match="annotated request disagrees"):
        projector.project_text(
            native_request,
            require_user=True,
            codec_name=codec_name,
            annotated_request=stale,
        )


def test_response_codec_is_selected_independently_from_request_codec() -> None:
    projector = _ProviderProjector()
    request = projector.project_text(
        chat_request(include_response_format=False),
        require_user=True,
        codec_name="openai_chat",
    )
    response = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "safe"}],
    }

    projection = projector.project_response_texts(
        request,
        response,
        allow_tool_calls=False,
        response_codec_name="anthropic_messages",
    )
    assert projection.candidates == ("safe",)
    assert projection.codec_name == "anthropic_messages"
    assert projection.codec_variant is None


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(PluginContext, "register_llm_execution_intercept_with_context"),
    reason="requires the additive Relay execution-context SDK",
)
async def test_worker_uses_authoritative_context_when_the_sdk_exposes_it() -> None:
    context = MagicMock(spec=PluginContext)
    context.runtime = SimpleNamespace(list_runtime_registrations=AsyncMock(return_value=[]))
    plugin = worker.NeMoGuardrailsRelayWorker()
    config_path = Path(__file__).resolve().parents[1] / "examples" / "no-model-rails"
    await plugin.register(context, {"config_path": str(config_path)})
    callback = context.register_llm_execution_intercept_with_context.call_args.args[1]
    next_call = SimpleNamespace(call=AsyncMock(return_value={"choices": []}))
    codec_context = _context("builtin", "openai_chat", "opaque", None)
    try:
        native_request = chat_request(include_response_format=False)
        assert await callback("openai.chat_completions", native_request, codec_context, next_call) == {"choices": []}
        runtime_context = _context("runtime", "com.example.codec", "opaque", None)
        with pytest.raises(execution_policy._LlmPolicyError, match="cannot prove complete coverage"):
            await callback("runtime.generate", native_request, runtime_context, next_call)
    finally:
        await plugin.close()

    context.register_llm_execution_intercept.assert_not_called()
    assert next_call.call.await_count == 1
