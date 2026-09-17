# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_relay_plugin import PluginContext
from registration_helpers import registered_llm_execution

from nemoguardrails_nemo_relay import execution_policy, worker
from nemoguardrails_nemo_relay.check_backend import CheckResult, CheckStatus


class _Backend:
    def __init__(self, results: list[CheckResult]) -> None:
        self._results = iter(results)
        self.calls: list[tuple[list[dict[str, str]], str]] = []

    async def check(self, messages: object, phase: str) -> CheckResult:
        self.calls.append(([dict(message) for message in messages], phase))  # type: ignore[union-attr]
        return next(self._results)

    async def close(self) -> None:
        return None


def _context() -> MagicMock:
    context = MagicMock(spec=PluginContext)
    context.runtime = SimpleNamespace(list_runtime_registrations=AsyncMock(return_value=[]))
    return context


def _config(phase: str) -> dict[str, object]:
    return {
        "remote_checks": {
            "endpoint": "https://guardrails.example.test",
            "config_ids": ["policy"],
            "phases": [phase],
            "model": "guardrails-evaluator",
            "allow_remote_content_logging_and_retention": True,
        },
        "mutation_policy": {phase: "apply"},
    }


def _input_cases() -> list[tuple[str, dict[str, object], tuple[str | int, ...]]]:
    return [
        (
            "openai.chat_completions",
            {
                "headers": {},
                "content": {
                    "model": "fixture",
                    "messages": [{"role": "user", "content": "original"}],
                    "response_format": {"type": "text"},
                    "temperature": 0.2,
                },
            },
            ("content", "messages", 0, "content"),
        ),
        (
            "openai.responses",
            {
                "headers": {},
                "content": {"model": "fixture", "input": "original", "store": False},
            },
            ("content", "input"),
        ),
        (
            "anthropic.messages",
            {
                "headers": {"anthropic-version": "2023-06-01"},
                "content": {
                    "model": "fixture",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "original"}],
                },
            },
            ("content", "messages", 0, "content"),
        ),
        (
            "gemini.generate_content",
            {
                "headers": {},
                "content": {
                    "contents": [{"role": "user", "parts": [{"text": "original"}]}],
                    "generationConfig": {"temperature": 0.1},
                },
            },
            ("content", "contents", 0, "parts", 0, "text"),
        ),
        (
            "oci.chat",
            {
                "headers": {},
                "content": {
                    "compartmentId": "ocid1.compartment.fixture",
                    "servingMode": {"servingType": "DEDICATED", "endpointId": "ocid1.endpoint.fixture"},
                    "chatRequest": {
                        "apiFormat": "GENERIC",
                        "messages": [{"role": "USER", "content": [{"type": "TEXT", "text": "original"}]}],
                    },
                },
            },
            ("content", "chatRequest", "messages", 0, "content", 0, "text"),
        ),
    ]


@pytest.mark.parametrize(("operation", "request_payload", "path"), _input_cases())
@pytest.mark.asyncio
async def test_input_modified_rewrites_native_request_before_provider(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    request_payload: dict[str, object],
    path: tuple[str | int, ...],
) -> None:
    backend = _Backend([CheckResult(CheckStatus.MODIFIED, "replacement")])
    monkeypatch.setattr(worker, "RemoteChecksBackend", lambda *_args, **_kwargs: backend)
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, _config("input"))
    callback = registered_llm_execution(context).callback
    original = deepcopy(request_payload)
    next_call = SimpleNamespace(call=AsyncMock(return_value={"provider": "response"}))
    try:
        result = await callback(operation, request_payload, next_call)
    finally:
        await plugin.close()

    assert result == {"provider": "response"}
    assert request_payload == original
    forwarded = next_call.call.await_args.args[0]
    cursor: object = forwarded
    for part in path:
        cursor = cursor[part]  # type: ignore[index]
    assert cursor == "replacement"
    assert backend.calls == [([{"role": "user", "content": "original"}], "input")]


def _output_cases() -> list[tuple[str, dict[str, object], dict[str, object], tuple[str, ...], tuple[str | int, ...]]]:
    return [
        (
            "openai.chat_completions",
            _input_cases()[0][1],
            {
                "id": "chatcmpl-fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "unsafe",
                            "function_call": None,
                            "tool_calls": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            ("unsafe",),
            ("choices", 0, "message", "content"),
        ),
        (
            "openai.responses",
            _input_cases()[1][1],
            {
                "id": "resp-fixture",
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "unsafe", "annotations": []}],
                    }
                ],
            },
            ("unsafe",),
            ("output", 0, "content", 0, "text"),
        ),
        (
            "anthropic.messages",
            _input_cases()[2][1],
            {
                "id": "msg-fixture",
                "type": "message",
                "role": "assistant",
                "model": "fixture",
                "content": [{"type": "text", "text": "unsafe"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            ("unsafe",),
            ("content", 0, "text"),
        ),
        (
            "gemini.generate_content",
            _input_cases()[3][1],
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "unsafe"}]}}]},
            ("unsafe",),
            ("candidates", 0, "content", "parts", 0, "text"),
        ),
        (
            "oci.chat",
            _input_cases()[4][1],
            {
                "chatResponse": {
                    "apiFormat": "GENERIC",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "unsafe"}]},
                            "finishReason": "STOP",
                        }
                    ],
                }
            },
            ("unsafe",),
            ("chatResponse", "choices", 0, "message", "content", 0, "text"),
        ),
    ]


@pytest.mark.parametrize(("operation", "request_payload", "provider_response", "texts", "path"), _output_cases())
@pytest.mark.asyncio
async def test_output_modified_rewrites_held_native_response(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    request_payload: dict[str, object],
    provider_response: dict[str, object],
    texts: tuple[str, ...],
    path: tuple[str | int, ...],
) -> None:
    backend = _Backend([CheckResult(CheckStatus.MODIFIED, "replacement") for _ in texts])
    monkeypatch.setattr(worker, "RemoteChecksBackend", lambda *_args, **_kwargs: backend)
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, _config("output"))
    callback = registered_llm_execution(context).callback
    original_response = deepcopy(provider_response)
    next_call = SimpleNamespace(call=AsyncMock(return_value=provider_response))
    try:
        result = await callback(operation, request_payload, next_call)
    finally:
        await plugin.close()

    assert provider_response == original_response
    cursor: object = result
    for part in path:
        cursor = cursor[part]  # type: ignore[index]
    assert cursor == "replacement"
    next_call.call.assert_awaited_once_with(request_payload)
    assert [phase for _messages, phase in backend.calls] == ["output"]


@pytest.mark.asyncio
async def test_multiple_output_candidates_are_checked_and_rewritten_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend(
        [
            CheckResult(CheckStatus.MODIFIED, "first replacement"),
            CheckResult(CheckStatus.PASSED, "second"),
        ]
    )
    monkeypatch.setattr(worker, "RemoteChecksBackend", lambda *_args, **_kwargs: backend)
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, _config("output"))
    callback = registered_llm_execution(context).callback
    response = deepcopy(_output_cases()[0][2])
    response["choices"].append(  # type: ignore[union-attr]
        {"index": 1, "message": {"role": "assistant", "content": "second"}, "finish_reason": "stop"}
    )
    try:
        result = await callback(
            "openai.chat_completions",
            _input_cases()[0][1],
            SimpleNamespace(call=AsyncMock(return_value=response)),
        )
    finally:
        await plugin.close()

    assert [choice["message"]["content"] for choice in result["choices"]] == [  # type: ignore[index]
        "first replacement",
        "second",
    ]
    assert [call[0][-1]["content"] for call in backend.calls] == ["unsafe", "second"]


@pytest.mark.asyncio
async def test_oci_multiple_output_candidates_are_checked_and_rewritten_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend(
        [
            CheckResult(CheckStatus.MODIFIED, "first replacement"),
            CheckResult(CheckStatus.MODIFIED, "second replacement"),
        ]
    )
    monkeypatch.setattr(worker, "RemoteChecksBackend", lambda *_args, **_kwargs: backend)
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, _config("output"))
    callback = registered_llm_execution(context).callback
    response = deepcopy(_output_cases()[4][2])
    response["chatResponse"]["choices"].append(  # type: ignore[index]
        {
            "index": 1,
            "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "second"}]},
            "finishReason": "STOP",
        }
    )
    try:
        result = await callback(
            "oci.chat",
            _input_cases()[4][1],
            SimpleNamespace(call=AsyncMock(return_value=response)),
        )
    finally:
        await plugin.close()

    assert [
        choice["message"]["content"][0]["text"]
        for choice in result["chatResponse"]["choices"]  # type: ignore[index]
    ] == ["first replacement", "second replacement"]
    assert [call[0][-1]["content"] for call in backend.calls] == ["unsafe", "second"]


@pytest.mark.asyncio
async def test_default_modified_policy_still_rejects_without_mutating_or_forwarding_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend([CheckResult(CheckStatus.MODIFIED, "replacement")])
    monkeypatch.setattr(worker, "RemoteChecksBackend", lambda *_args, **_kwargs: backend)
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    config = _config("input")
    config.pop("mutation_policy")
    await plugin.register(context, config)
    callback = registered_llm_execution(context).callback
    request_payload = _input_cases()[0][1]
    original = deepcopy(request_payload)
    next_call = SimpleNamespace(call=AsyncMock())
    try:
        with pytest.raises(execution_policy._LlmPolicyError, match="modified the input"):
            await callback("openai.chat_completions", request_payload, next_call)
    finally:
        await plugin.close()

    assert request_payload == original
    next_call.call.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_rejects_a_joined_input_projection_instead_of_guessing_a_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend([CheckResult(CheckStatus.MODIFIED, "replacement")])
    monkeypatch.setattr(worker, "RemoteChecksBackend", lambda *_args, **_kwargs: backend)
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, _config("input"))
    callback = registered_llm_execution(context).callback
    request_payload = deepcopy(_input_cases()[0][1])
    request_payload["content"]["messages"][0]["content"] = [  # type: ignore[index]
        {"type": "text", "text": "orig"},
        {"type": "text", "text": "inal"},
    ]
    next_call = SimpleNamespace(call=AsyncMock())
    try:
        with pytest.raises(execution_policy._LlmPolicyError, match="cannot be rewritten safely"):
            await callback("openai.chat_completions", request_payload, next_call)
    finally:
        await plugin.close()
    next_call.call.assert_not_awaited()
