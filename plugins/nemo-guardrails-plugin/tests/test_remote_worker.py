# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_relay_plugin import PluginContext
from registration_helpers import registered_llm_execution

from nemoguardrails_nemo_relay import configuration, execution_policy, worker
from nemoguardrails_nemo_relay.check_backend import CheckResult, CheckStatus


def _context() -> MagicMock:
    context = MagicMock(spec=PluginContext)
    context.runtime = SimpleNamespace(list_runtime_registrations=AsyncMock(return_value=[]))
    return context


def _config(*, phases: list[str]) -> dict[str, object]:
    return {
        "remote_checks": {
            "endpoint": "https://guardrails.example.test",
            "config_ids": ["policy-a"],
            "phases": phases,
            "model": "guardrails-evaluator",
            "allow_remote_content_logging_and_retention": True,
            "header_env": {"Authorization": "REMOTE_GUARDRAILS_TOKEN"},
        },
        "secret_env": {"REMOTE_GUARDRAILS_TOKEN": "Bearer private-token"},
    }


def _request() -> dict[str, object]:
    return {
        "headers": {},
        "content": {
            "model": "fixture",
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "text"},
        },
    }


def _response() -> dict[str, object]:
    return {
        "id": "chatcmpl-fixture",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "safe response",
                    "function_call": None,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
    }


class _RecordingBackend:
    def __init__(self, results: list[CheckResult]) -> None:
        self._results = iter(results)
        self.calls: list[tuple[list[dict[str, str]], str]] = []
        self.closed = False

    async def check(self, messages: object, phase: str) -> CheckResult:
        copied = [dict(message) for message in messages]  # type: ignore[union-attr]
        self.calls.append((copied, phase))
        return next(self._results)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_remote_worker_runs_explicit_input_and_output_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _RecordingBackend(
        [
            CheckResult(CheckStatus.PASSED, "hello"),
            CheckResult(CheckStatus.PASSED, "safe response"),
        ]
    )
    observed_headers: list[dict[str, str]] = []

    def construct_backend(_settings: object, *, headers: dict[str, str]) -> _RecordingBackend:
        observed_headers.append(headers)
        return backend

    monkeypatch.setattr(worker, "RemoteChecksBackend", construct_backend)
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, _config(phases=["input", "output"]))
    callback = registered_llm_execution(context).callback
    next_call = SimpleNamespace(call=AsyncMock(return_value=_response()))
    try:
        response = await callback("openai.chat_completions", _request(), next_call)
    finally:
        await plugin.close()

    assert response == _response()
    assert observed_headers == [{"Authorization": "Bearer private-token"}]
    assert [phase for _messages, phase in backend.calls] == ["input", "output"]
    assert backend.calls[0][0] == [{"role": "user", "content": "hello"}]
    assert backend.calls[1][0][-1] == {"role": "assistant", "content": "safe response"}
    assert backend.closed


@pytest.mark.asyncio
async def test_remote_worker_blocks_before_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _RecordingBackend([CheckResult(CheckStatus.BLOCKED, "refusal", "policy rail")])
    monkeypatch.setattr(worker, "RemoteChecksBackend", lambda *_args, **_kwargs: backend)
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, _config(phases=["input"]))
    callback = registered_llm_execution(context).callback
    next_call = SimpleNamespace(call=AsyncMock(return_value=_response()))
    try:
        with pytest.raises(execution_policy._LlmPolicyError, match="prevented execution"):
            await callback("openai.chat_completions", _request(), next_call)
    finally:
        await plugin.close()

    next_call.call.assert_not_awaited()


def test_remote_worker_requires_configured_header_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(phases=["input"])
    config["secret_env"] = {}
    monkeypatch.delenv("REMOTE_GUARDRAILS_TOKEN", raising=False)

    diagnostics = configuration._validate_config(config)

    assert len(diagnostics) == 1
    assert diagnostics[0].code.endswith(".invalid_remote_check_credentials")
    assert diagnostics[0].field == "remote_checks"
