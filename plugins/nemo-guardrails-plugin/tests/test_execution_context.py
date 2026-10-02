# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_relay_plugin import LlmCodecIdentity, LlmExecutionContext
from provider_cases import (
    anthropic_request,
    chat_request,
    chat_response,
    gemini_request,
    oci_request,
    responses_request,
)

from nemoguardrails_nemo_relay import execution_policy, text_policy
from nemoguardrails_nemo_relay.check_backend import CheckResult, CheckStatus
from nemoguardrails_nemo_relay.codec_projection import _ProviderProjector, _UnsupportedRequest
from nemoguardrails_nemo_relay.execution_context import ExecutionCodecContextError, execution_codec_context
from nemoguardrails_nemo_relay.structural_tools import (
    StructuralToolSettings,
    ToolVerdict,
    ToolVerdictKind,
)
from nemoguardrails_nemo_relay.text_mutation import MutationMode, MutationPolicy


class _RequestCodec:
    def __init__(self, annotation: dict[str, Any], *, wire_field: str | None = None) -> None:
        self.annotation = deepcopy(annotation)
        self.wire_field = wire_field
        self.decode_calls: list[dict[str, Any]] = []
        self.encode_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def decode(self, request: dict[str, Any]) -> dict[str, Any]:
        self.decode_calls.append(deepcopy(request))
        annotation = deepcopy(self.annotation)
        if self.wire_field is not None:
            annotation["messages"][-1]["content"] = request["content"][self.wire_field]
        return annotation

    async def encode(self, annotated: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
        self.encode_calls.append((deepcopy(annotated), deepcopy(original)))
        if self.wire_field is None:
            raise RuntimeError("fixture codec is decode-only")
        rewritten = deepcopy(original)
        rewritten["content"][self.wire_field] = annotated["messages"][-1]["content"]
        return rewritten


class _ResponseCodec:
    def __init__(self) -> None:
        self.decode = AsyncMock(return_value={"message": {"role": "assistant", "content": "safe"}})


@dataclass(frozen=True)
class _DirectionContext:
    codec: LlmCodecIdentity
    resolved: object | None

    def resolve_codec(self) -> object | None:
        return self.resolved


def _context(
    request_codec: _RequestCodec,
    *,
    request_kind: str = "runtime",
    request_name: str | None = "fixture.runtime",
    response_kind: str = "opaque",
    response_name: str | None = None,
    response_codec: _ResponseCodec | None = None,
) -> LlmExecutionContext:
    return LlmExecutionContext(
        request_codec=_DirectionContext(  # type: ignore[arg-type]
            LlmCodecIdentity(request_kind, request_name),
            request_codec,
        ),
        response_codec=_DirectionContext(  # type: ignore[arg-type]
            LlmCodecIdentity(response_kind, response_name),
            response_codec,
        ),
    )


def _annotation(codec_name: str) -> dict[str, Any]:
    annotation: dict[str, Any] = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "fixture",
    }
    if codec_name != "gemini_generate_content":
        annotation["api_specific"] = {
            "api": codec_name,
            **({"api_format": "GENERIC"} if codec_name == "oci_genai" else {}),
        }
    return annotation


@pytest.mark.parametrize(
    ("codec_name", "payload"),
    [
        ("openai_chat", chat_request(include_response_format=False)),
        ("openai_responses", responses_request()),
        ("anthropic_messages", anthropic_request()),
        ("gemini_generate_content", gemini_request()),
        ("oci_genai", oci_request()),
    ],
)
def test_host_annotations_project_all_five_builtin_requests(
    codec_name: str,
    payload: dict[str, Any],
) -> None:
    projected = _ProviderProjector().project_host_text(
        payload,
        _annotation(codec_name),
        require_user=True,
        codec_name=codec_name,
    )

    assert projected.codec_name == codec_name
    assert projected.messages == ({"role": "user", "content": "hello"},)


@pytest.mark.parametrize(
    ("codec_name", "payload"),
    [
        ("openai_chat", chat_request(include_response_format=False)),
        ("openai_responses", responses_request()),
        ("anthropic_messages", anthropic_request()),
        ("gemini_generate_content", gemini_request()),
        ("oci_genai", oci_request()),
    ],
)
@pytest.mark.asyncio
async def test_execution_uses_host_decode_for_each_builtin_request(
    codec_name: str,
    payload: dict[str, Any],
) -> None:
    codec = _RequestCodec(_annotation(codec_name))
    policy = _policy(_Backend(CheckResult(CheckStatus.PASSED, "hello")))
    next_call = SimpleNamespace(call=AsyncMock(return_value={"ok": True}))

    result = await policy.execute_with_context(
        "fixture.generate",
        payload,
        _context(codec, request_kind="builtin", request_name=codec_name),
        next_call,
    )

    assert result == {"ok": True}
    assert codec.decode_calls == [payload]
    next_call.call.assert_awaited_once_with(payload)


def test_context_matches_the_worker_sdk_shape_and_rejects_missing_capabilities() -> None:
    request_codec = _RequestCodec(_annotation("openai_chat"))
    parsed = execution_codec_context(
        _context(
            request_codec,
            request_kind="builtin",
            request_name="openai_chat",
            response_kind="builtin",
            response_name="openai_chat",
            response_codec=_ResponseCodec(),
        )
    )
    assert parsed.request.builtin_name("request") == "openai_chat"
    assert parsed.response.builtin_name("response") == "openai_chat"
    assert parsed.request_codec is request_codec

    with pytest.raises(ExecutionCodecContextError, match="request codec capability is unavailable"):
        execution_codec_context(
            LlmExecutionContext(
                request_codec=_DirectionContext(  # type: ignore[arg-type]
                    LlmCodecIdentity("none"),
                    None,
                ),
                response_codec=None,
            )
        )


@pytest.mark.parametrize("request_kind", ["runtime", "opaque"])
def test_generic_codec_accepts_portable_function_tools_but_rejects_provider_fields(
    request_kind: str,
) -> None:
    projector = _ProviderProjector()
    portable = {
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }
    projected = projector.project_host_tools(
        {"content": "opaque"},
        portable,
        codec_name=None,
    )
    assert projected.projection.definitions[0].name == "weather"

    # Both generic identity kinds take this same provider-neutral path. The
    # execution-context parser must not accidentally require a runtime ID for
    # opaque codecs or infer a provider from the raw payload.
    identity = LlmCodecIdentity(request_kind, "fixture.runtime" if request_kind == "runtime" else None)
    parsed = execution_codec_context(
        LlmExecutionContext(
            request_codec=_DirectionContext(  # type: ignore[arg-type]
                identity,
                _RequestCodec(portable),
            ),
            response_codec=None,
        )
    )
    assert parsed.request.builtin_name("request") is None

    with pytest.raises(_UnsupportedRequest, match="provider fields"):
        projector.project_host_text(
            {"content": "opaque"},
            {**portable, "api_specific": {"hidden_prompt": "unchecked"}},
            require_user=True,
            codec_name=None,
        )


@pytest.mark.parametrize("request_kind", ["runtime", "opaque"])
@pytest.mark.asyncio
async def test_generic_codec_executes_portable_tool_result_checks(request_kind: str) -> None:
    annotation = {
        "messages": [
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
        ]
    }
    adapter = SimpleNamespace(
        check_results=AsyncMock(return_value=ToolVerdict(ToolVerdictKind.PASSED)),
        close=AsyncMock(),
    )
    policy = execution_policy._LlmExecutionPolicy(
        _ProviderProjector(),
        None,
        input_enabled=False,
        output_enabled=False,
        adapter=adapter,
        settings=StructuralToolSettings(False, True, (), ()),
    )
    next_call = SimpleNamespace(call=AsyncMock(return_value={"ok": True}))

    result = await policy.execute_with_context(
        "custom.generate",
        {"content": "opaque"},
        _context(
            _RequestCodec(annotation),
            request_kind=request_kind,
            request_name="fixture.runtime" if request_kind == "runtime" else None,
        ),
        next_call,
    )

    assert result == {"ok": True}
    adapter.check_results.assert_awaited_once()
    next_call.call.assert_awaited_once_with({"content": "opaque"})


class _Backend:
    def __init__(self, result: CheckResult) -> None:
        self.result = result
        self.calls: list[tuple[list[dict[str, str]], str]] = []

    async def check(self, messages: object, phase: str) -> CheckResult:
        self.calls.append(([dict(message) for message in messages], phase))  # type: ignore[union-attr]
        return self.result

    async def close(self) -> None:
        return None


def _policy(
    backend: _Backend,
    *,
    output: bool = False,
    apply_input: bool = False,
) -> execution_policy._LlmExecutionPolicy:
    projector = _ProviderProjector()
    rails_policy = text_policy._TextRailsPolicy(None, 1_000, backend=backend, projector=projector)
    return execution_policy._LlmExecutionPolicy(
        projector,
        rails_policy,
        input_enabled=not output,
        output_enabled=output,
        adapter=None,
        settings=StructuralToolSettings(False, False, (), ()),
        mutation_policy=MutationPolicy(input=MutationMode.APPLY if apply_input else MutationMode.REJECT),
    )


@pytest.mark.asyncio
async def test_custom_runtime_codec_checks_and_rewrites_without_touching_other_wire_fields() -> None:
    raw = {
        "headers": {"x-provider": "preserved"},
        "content": {
            "wire_prompt": "original",
            "vendor_control": {"preserve": True},
        },
    }
    codec = _RequestCodec(
        {"messages": [{"role": "user", "content": "original"}], "model": "custom-model"},
        wire_field="wire_prompt",
    )
    policy = _policy(
        _Backend(CheckResult(CheckStatus.MODIFIED, "replacement")),
        apply_input=True,
    )
    next_call = SimpleNamespace(call=AsyncMock(return_value={"ok": True}))

    result = await policy.execute_with_context("custom.generate", raw, _context(codec), next_call)

    assert result == {"ok": True}
    forwarded = next_call.call.await_args.args[0]
    assert forwarded["content"]["wire_prompt"] == "replacement"
    assert forwarded["content"]["vendor_control"] == {"preserve": True}
    assert forwarded["headers"] == {"x-provider": "preserved"}
    assert raw["content"]["wire_prompt"] == "original"
    assert codec.decode_calls == [raw, forwarded]
    assert len(codec.encode_calls) == 1


@pytest.mark.asyncio
async def test_input_rewrite_rejects_when_codec_does_not_update_the_wire_request() -> None:
    class NoOpEncoder(_RequestCodec):
        async def encode(self, annotated: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
            self.encode_calls.append((deepcopy(annotated), deepcopy(original)))
            return deepcopy(original)

    raw = {"content": {"wire_prompt": "original"}}
    codec = NoOpEncoder(
        {"messages": [{"role": "user", "content": "original"}], "model": "custom-model"},
        wire_field="wire_prompt",
    )
    policy = _policy(
        _Backend(CheckResult(CheckStatus.MODIFIED, "replacement")),
        apply_input=True,
    )
    next_call = SimpleNamespace(call=AsyncMock())

    with pytest.raises(execution_policy._LlmPolicyError, match="cannot be rewritten safely"):
        await policy.execute_with_context("custom.generate", raw, _context(codec), next_call)

    next_call.call.assert_not_awaited()
    assert len(codec.decode_calls) == 2
    assert len(codec.encode_calls) == 1


@pytest.mark.asyncio
async def test_codec_identity_is_authoritative_and_never_inferred_from_wire_shape() -> None:
    class ChatCodec(_RequestCodec):
        async def decode(self, request: dict[str, Any]) -> dict[str, Any]:
            if not isinstance(request.get("content", {}).get("messages"), list):
                raise ValueError("not OpenAI Chat")
            return await super().decode(request)

    codec = ChatCodec(_annotation("openai_chat"))
    policy = _policy(_Backend(CheckResult(CheckStatus.PASSED, "hello")))
    next_call = SimpleNamespace(call=AsyncMock())
    context = _context(codec, request_kind="builtin", request_name="openai_chat")

    with pytest.raises(execution_policy._LlmPolicyError, match="could not decode"):
        await policy.execute_with_context("fixture", gemini_request(), context, next_call)
    next_call.call.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_response_codec_fails_closed_before_strict_output_execution() -> None:
    codec = _RequestCodec({"messages": [{"role": "user", "content": "hello"}]})
    policy = _policy(_Backend(CheckResult(CheckStatus.PASSED, "safe")), output=True)
    next_call = SimpleNamespace(call=AsyncMock())

    with pytest.raises(execution_policy._LlmPolicyError, match="cannot prove complete output coverage"):
        await policy.execute_with_context("custom.generate", {"content": "opaque"}, _context(codec), next_call)
    next_call.call.assert_not_awaited()


@pytest.mark.parametrize(
    ("codec_name", "raw_request", "response"),
    [
        ("openai_chat", chat_request(include_response_format=False), chat_response("safe")),
        (
            "openai_responses",
            responses_request(),
            {
                "id": "resp-fixture",
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "safe", "annotations": []}],
                    }
                ],
                "output_text": "safe",
            },
        ),
        (
            "anthropic_messages",
            anthropic_request(),
            {
                "id": "msg-fixture",
                "type": "message",
                "role": "assistant",
                "model": "fixture",
                "content": [{"type": "text", "text": "safe"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ),
        (
            "gemini_generate_content",
            gemini_request(),
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "safe"}]}}]},
        ),
        (
            "oci_genai",
            oci_request(),
            {
                "chatResponse": {
                    "apiFormat": "GENERIC",
                    "choices": [
                        {
                            "message": {
                                "role": "ASSISTANT",
                                "content": [{"type": "TEXT", "text": "safe"}],
                            }
                        }
                    ],
                }
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_builtin_output_uses_response_decode_and_complete_raw_coverage(
    codec_name: str,
    raw_request: dict[str, Any],
    response: dict[str, Any],
) -> None:
    request_codec = _RequestCodec(_annotation(codec_name))
    response_codec = _ResponseCodec()
    policy = _policy(_Backend(CheckResult(CheckStatus.PASSED, "safe")), output=True)
    next_call = SimpleNamespace(call=AsyncMock(return_value=response))

    result = await policy.execute_with_context(
        "fixture.generate",
        raw_request,
        _context(
            request_codec,
            request_kind="builtin",
            request_name=codec_name,
            response_kind="builtin",
            response_name=codec_name,
            response_codec=response_codec,
        ),
        next_call,
    )

    assert result is response
    response_codec.decode.assert_awaited_once_with(response)


@pytest.mark.asyncio
async def test_input_only_stream_uses_runtime_request_codec_and_forwards_stream() -> None:
    codec = _RequestCodec({"messages": [{"role": "user", "content": "hello"}]})
    policy = _policy(_Backend(CheckResult(CheckStatus.PASSED, "hello")))
    stream = object()
    next_call = SimpleNamespace(call=MagicMock(return_value=stream))

    result = await policy.execute_stream_with_context(
        "custom.stream",
        {"content": "opaque"},
        _context(codec),
        next_call,
    )

    assert result is stream
    next_call.call.assert_called_once_with({"content": "opaque"})


@pytest.mark.asyncio
async def test_output_guarded_stream_rejects_before_provider() -> None:
    codec = _RequestCodec({"messages": [{"role": "user", "content": "hello"}]})
    policy = _policy(_Backend(CheckResult(CheckStatus.PASSED, "safe")), output=True)
    next_call = SimpleNamespace(call=AsyncMock())

    with pytest.raises(execution_policy._LlmPolicyError, match="guarded streaming response"):
        await policy.execute_stream_with_context(
            "custom.stream",
            {"content": "opaque"},
            _context(codec),
            next_call,
        )
    next_call.call.assert_not_awaited()
