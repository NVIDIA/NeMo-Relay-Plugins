# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import httpx
import pytest
from http_test_server import HttpRequest, loopback_http_server

from nemoguardrails_nemo_relay.check_backend import (
    CheckBackendError,
    CheckResult,
    CheckStatus,
    RemoteChecksBackend,
    RemoteChecksSettings,
)


def _settings(**updates: object) -> RemoteChecksSettings:
    value: dict[str, object] = {
        "endpoint": "https://guardrails.example.test",
        "config_ids": ["policy-a"],
        "phases": ["input", "output"],
        "model": "guardrails-evaluator",
        "allow_remote_content_logging_and_retention": True,
    }
    value.update(updates)
    return RemoteChecksSettings.from_config(value)


def test_remote_settings_normalize_checks_path_and_resolve_header_env() -> None:
    settings = _settings(header_env={"Authorization": "REMOTE_TOKEN"})

    assert settings.endpoint == "https://guardrails.example.test/v1/checks"
    assert settings.headers({"REMOTE_TOKEN": "Bearer secret"}) == {"Authorization": "Bearer secret"}


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://guardrails.example.test",
        "ftp://guardrails.example.test",
        "https://user:pass@guardrails.example.test",
        "https://guardrails.example.test?token=secret",
    ],
)
def test_remote_settings_reject_unsafe_endpoints(endpoint: str) -> None:
    with pytest.raises(ValueError):
        _settings(endpoint=endpoint)


def test_remote_settings_allow_loopback_http() -> None:
    assert _settings(endpoint="http://127.0.0.1:8000/").endpoint == "http://127.0.0.1:8000/v1/checks"


def test_remote_settings_require_an_explicit_evaluator_model() -> None:
    with pytest.raises(ValueError, match="model"):
        RemoteChecksSettings.from_config(
            {
                "endpoint": "https://guardrails.example.test",
                "config_ids": ["policy-a"],
                "phases": ["input"],
                "allow_remote_content_logging_and_retention": True,
            }
        )


def test_remote_settings_require_explicit_content_logging_and_retention_acknowledgement() -> None:
    with pytest.raises(ValueError, match="allow_remote_content_logging_and_retention"):
        _settings(allow_remote_content_logging_and_retention=None)

    with pytest.raises(ValueError, match="allow_remote_content_logging_and_retention"):
        RemoteChecksSettings.from_config(
            {
                "endpoint": "https://guardrails.example.test",
                "config_ids": ["policy-a"],
                "phases": ["input"],
                "model": "guardrails-evaluator",
            }
        )


def test_remote_settings_reject_false_content_acknowledgement() -> None:
    with pytest.raises(ValueError, match="allow_remote_content_logging_and_retention"):
        _settings(allow_remote_content_logging_and_retention=False)


@pytest.mark.parametrize("phases", [[], ["retrieval"], ["input", "input"]])
def test_remote_settings_require_distinct_supported_phases(phases: list[str]) -> None:
    with pytest.raises(ValueError):
        _settings(phases=phases)


@pytest.mark.asyncio
async def test_remote_backend_sends_explicit_phase_and_decodes_result() -> None:
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "modified", "content": "safe", "rail": None})

    backend = RemoteChecksBackend(_settings(), transport=httpx.MockTransport(handler))
    try:
        result = await backend.check([{"role": "user", "content": "unsafe"}], "input")
    finally:
        await backend.close()

    assert result.status is CheckStatus.MODIFIED
    assert result.content == "safe"
    assert result.transformed_content == "safe"
    assert seen == {
        "url": "https://guardrails.example.test/v1/checks",
        "body": {
            "model": "guardrails-evaluator",
            "messages": [{"role": "user", "content": "unsafe"}],
            "guardrails": {"config_ids": ["policy-a"], "rail_types": ["input"]},
        },
    }


@pytest.mark.asyncio
async def test_remote_backend_rejects_redirect_without_following_it() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(307, headers={"location": "https://attacker.invalid/v1/checks"})

    backend = RemoteChecksBackend(_settings(), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(CheckBackendError, match="remote Guardrails check failed"):
            await backend.check([{"role": "user", "content": "hello"}], "input")
    finally:
        await backend.close()

    assert calls == 1


@pytest.mark.asyncio
async def test_remote_backend_bounds_response_body() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 2_000)

    backend = RemoteChecksBackend(
        _settings(max_response_bytes=1_024),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(CheckBackendError, match="oversized"):
            await backend.check([{"role": "user", "content": "hello"}], "input")
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_remote_backend_rejects_unknown_response_fields() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "passed", "content": "ok", "debug": "secret"})

    backend = RemoteChecksBackend(_settings(), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(CheckBackendError, match="invalid"):
            await backend.check([{"role": "user", "content": "hello"}], "input")
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_remote_backend_uses_a_real_bounded_loopback_http_connection() -> None:
    observed: list[dict[str, object]] = []

    def respond(request: HttpRequest) -> tuple[int, bytes]:
        observed.append(json.loads(request.body))
        return 200, json.dumps({"status": "passed", "content": "hello", "rail": None}).encode()

    with loopback_http_server(respond) as endpoint:
        backend = RemoteChecksBackend(_settings(endpoint=endpoint))
        try:
            result = await backend.check([{"role": "user", "content": "hello"}], "input")
        finally:
            await backend.close()

    assert result == CheckResult(CheckStatus.PASSED, "hello")
    assert observed[0]["guardrails"] == {"config_ids": ["policy-a"], "rail_types": ["input"]}
