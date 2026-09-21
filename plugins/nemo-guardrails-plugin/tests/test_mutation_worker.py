# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from provider_cases import chat_request, chat_response
from registration_helpers import registered_llm_execution
from worker_test_helpers import worker_context as _context

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


_REQUEST = chat_request([{"role": "user", "content": "original"}], temperature=0.2)
_RESPONSE = chat_response("unsafe")


@pytest.mark.asyncio
async def test_input_modified_rewrites_native_request_before_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend([CheckResult(CheckStatus.MODIFIED, "replacement")])
    monkeypatch.setattr(worker, "RemoteChecksBackend", lambda *_args, **_kwargs: backend)
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, _config("input"))
    callback = registered_llm_execution(context).callback
    request_payload = deepcopy(_REQUEST)
    original = deepcopy(_REQUEST)
    next_call = SimpleNamespace(call=AsyncMock(return_value={"provider": "response"}))
    try:
        result = await callback("openai.chat_completions", request_payload, next_call)
    finally:
        await plugin.close()

    assert result == {"provider": "response"}
    assert request_payload == original
    forwarded = next_call.call.await_args.args[0]
    assert forwarded["content"]["messages"][0]["content"] == "replacement"
    assert backend.calls == [([{"role": "user", "content": "original"}], "input")]


@pytest.mark.asyncio
async def test_output_modified_rewrites_held_native_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _Backend([CheckResult(CheckStatus.MODIFIED, "replacement")])
    monkeypatch.setattr(worker, "RemoteChecksBackend", lambda *_args, **_kwargs: backend)
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, _config("output"))
    callback = registered_llm_execution(context).callback
    request_payload = deepcopy(_REQUEST)
    provider_response = deepcopy(_RESPONSE)
    original_response = deepcopy(_RESPONSE)
    next_call = SimpleNamespace(call=AsyncMock(return_value=provider_response))
    try:
        result = await callback("openai.chat_completions", request_payload, next_call)
    finally:
        await plugin.close()

    assert provider_response == original_response
    assert result["choices"][0]["message"]["content"] == "replacement"
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
    response = deepcopy(_RESPONSE)
    response["choices"].append(  # type: ignore[union-attr]
        {"index": 1, "message": {"role": "assistant", "content": "second"}, "finish_reason": "stop"}
    )
    try:
        result = await callback(
            "openai.chat_completions",
            deepcopy(_REQUEST),
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
    request_payload = deepcopy(_REQUEST)
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
    request_payload = deepcopy(_REQUEST)
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
