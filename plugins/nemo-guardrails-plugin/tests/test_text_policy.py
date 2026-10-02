# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from nemoguardrails.rails.llm.options import RailStatus, RailType
from provider_cases import guardrails_chat_request as _request

from nemoguardrails_nemo_relay import (
    codec_projection,
    configuration,
    text_policy,
    worker,
)

_ECHO_LATEST_USER = object()


async def _check_input(
    policy: text_policy._TextRailsPolicy,
    messages: list[dict[str, str]],
) -> str | None:
    projected = codec_projection._DecodedTextRequest(
        "openai_chat",
        tuple(messages),
        None,
    )
    return await policy.check_input(projected)


class _FakeRails:
    def __init__(self, result: object | None = None, *, delay: float = 0) -> None:
        self.result = result
        self.delay = delay
        self.calls: list[list[dict[str, str]]] = []
        self.rail_type_calls: list[object] = []
        self.active = 0
        self.max_active = 0
        self.started = asyncio.Event()
        self.explanation = SimpleNamespace(llm_calls=[], colang_history=None)

    async def check_async(self, messages: list[dict[str, str]], rail_types: object) -> object:
        self.calls.append(messages)
        self.rail_type_calls.append(rail_types)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        self.explanation.llm_calls.append(
            SimpleNamespace(prompt=json.dumps(messages), completion="private evaluator result")
        )
        self.explanation.colang_history = json.dumps(messages)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if isinstance(self.result, BaseException):
                raise self.result
            if (
                getattr(self.result, "status", None) == RailStatus.PASSED
                and getattr(self.result, "content", None) is _ECHO_LATEST_USER
            ):
                latest_user = next(message["content"] for message in reversed(messages) if message["role"] == "user")
                return SimpleNamespace(status=RailStatus.PASSED, rail=None, content=latest_user)
            return self.result
        finally:
            self.active -= 1

    def explain(self) -> object:
        return self.explanation


async def test_policy_maps_pass_block_modify_and_unknown_without_content_leaks() -> None:
    messages = [{"role": "user", "content": "private input"}]
    cases = [
        (SimpleNamespace(status=RailStatus.PASSED, rail=None, content="private input"), None),
        (
            SimpleNamespace(status=RailStatus.BLOCKED, rail="input rail", content="private refusal"),
            "NeMo Guardrails prevented execution in input rail 'input rail'",
        ),
        (
            SimpleNamespace(status=RailStatus.MODIFIED, rail=None, content="private rewrite"),
            "NeMo Guardrails modified the input, but this worker cannot safely rewrite it",
        ),
        (
            SimpleNamespace(status="future-status", rail=None, content="private result"),
            "NeMo Guardrails input check failed",
        ),
        (
            SimpleNamespace(status=RailStatus.PASSED, rail=None, content="different input"),
            "NeMo Guardrails returned an inconsistent input-check result",
        ),
    ]

    for result, expected in cases:
        rails = _FakeRails(result)
        reason = await _check_input(text_policy._TextRailsPolicy(rails, 1_000), messages)
        assert reason == expected
        assert reason is None or "private" not in reason
        assert rails.explanation.llm_calls == []
        assert rails.explanation.colang_history is None
        assert rails.rail_type_calls == [[RailType.INPUT]]


async def test_unsupported_projection_does_not_call_guardrails() -> None:
    rails = _FakeRails(SimpleNamespace(status=RailStatus.PASSED, rail=None, content=""))
    policy = text_policy._TextRailsPolicy(rails, 1_000)

    with pytest.raises(codec_projection._UnsupportedRequest):
        policy.project_host_request(
            _request([{"role": "tool", "content": "secret"}]),
            {"messages": [{"role": "tool", "content": "secret"}]},
            require_user=True,
            codec_name="openai_chat",
        )
    assert rails.calls == []


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "x"}] * (text_policy.MAX_PROJECTED_MESSAGES + 1),
        [{"role": "user", "content": "x" * (text_policy.MAX_PROJECTED_TEXT_CHARACTERS + 1)}],
    ],
)
async def test_oversized_projected_history_is_rejected_before_guardrails(
    messages: list[dict[str, str]],
) -> None:
    rails = _FakeRails(SimpleNamespace(status=RailStatus.PASSED, rail=None, content=_ECHO_LATEST_USER))
    policy = text_policy._TextRailsPolicy(rails, 1_000)
    reason = await _check_input(policy, messages)

    assert reason == "NeMo Guardrails cannot verify this oversized input context"
    assert rails.calls == []
    assert policy._pending == 0


async def test_overload_is_rejected_before_projection() -> None:
    rails = _FakeRails(SimpleNamespace(status=RailStatus.PASSED, rail=None, content=_ECHO_LATEST_USER))
    policy = text_policy._TextRailsPolicy(rails, 1_000)
    policy._projector = MagicMock()
    policy._pending = text_policy.MAX_PENDING_CHECKS

    with pytest.raises(codec_projection._UnsupportedRequest, match="queue is full"):
        policy.project_host_request(
            _request([{"role": "user", "content": "not projected"}]),
            {"messages": [{"role": "user", "content": "not projected"}]},
            require_user=True,
            codec_name="openai_chat",
        )
    policy._projector.project_host_text.assert_not_called()
    assert rails.calls == []


async def test_tool_definitions_are_preserved_but_absent_from_guardrails_input() -> None:
    request = _request(
        [{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "policy-relevant but deliberately unchecked",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )
    original = json.loads(json.dumps(request))
    rails = _FakeRails(SimpleNamespace(status=RailStatus.PASSED, rail=None, content=_ECHO_LATEST_USER))

    policy = text_policy._TextRailsPolicy(rails, 1_000)
    projected = policy.project_host_request(
        request,
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "policy-relevant but deliberately unchecked",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
        require_user=True,
        codec_name="openai_chat",
    )
    assert await policy.check_input(projected) is None
    assert rails.calls == [[{"role": "user", "content": "hello"}]]
    assert request == original


async def test_evaluator_errors_and_timeouts_fail_closed() -> None:
    failed_rails = _FakeRails(RuntimeError("private provider error"))
    timeout_rails = _FakeRails(delay=0.05)
    messages = [{"role": "user", "content": "hello"}]
    failed = await _check_input(text_policy._TextRailsPolicy(failed_rails, 1_000), messages)
    timed_out = await _check_input(text_policy._TextRailsPolicy(timeout_rails, 10), messages)

    assert failed == "NeMo Guardrails input check failed"
    assert timed_out == "NeMo Guardrails input check timed out"
    for rails in (failed_rails, timeout_rails):
        assert rails.explanation.llm_calls == []
        assert rails.explanation.colang_history is None


async def test_concurrent_calls_are_serialized() -> None:
    rails = _FakeRails(
        SimpleNamespace(status=RailStatus.PASSED, rail=None, content=_ECHO_LATEST_USER),
        delay=0.02,
    )
    policy = text_policy._TextRailsPolicy(rails, 1_000)

    results = await asyncio.gather(
        _check_input(policy, [{"role": "user", "content": "first"}]),
        _check_input(policy, [{"role": "user", "content": "second"}]),
        _check_input(policy, [{"role": "user", "content": "third"}]),
    )

    assert results == [None, None, None]
    assert rails.max_active == 1
    assert sorted(call[-1]["content"] for call in rails.calls) == ["first", "second", "third"]
    assert rails.explanation.llm_calls == []
    assert rails.explanation.colang_history is None


async def test_cancellation_propagates_and_releases_the_permit() -> None:
    rails = _FakeRails(
        SimpleNamespace(status=RailStatus.PASSED, rail=None, content=_ECHO_LATEST_USER),
        delay=60,
    )
    policy = text_policy._TextRailsPolicy(rails, configuration.MAX_CHECK_TIMEOUT_MS)
    task = asyncio.create_task(_check_input(policy, [{"role": "user", "content": "hello"}]))
    await rails.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert rails.explanation.llm_calls == []
    assert rails.explanation.colang_history is None
    rails.delay = 0
    rails.started.clear()
    assert await _check_input(policy, [{"role": "user", "content": "after cancellation"}]) is None


async def test_queue_wait_timeout_is_bounded_and_recovers() -> None:
    rails = _FakeRails(SimpleNamespace(status=RailStatus.PASSED, rail=None, content=_ECHO_LATEST_USER))
    policy = text_policy._TextRailsPolicy(rails, 10)
    await policy._permit.acquire()
    try:
        result = await _check_input(policy, [{"role": "user", "content": "queued"}])
    finally:
        policy._permit.release()

    assert result == "NeMo Guardrails input check timed out"
    assert rails.calls == []
    assert policy._pending == 0
    policy._timeout_seconds = 1
    assert await _check_input(policy, [{"role": "user", "content": "after timeout"}]) is None


async def test_pending_queue_is_bounded_and_cancellation_recovers() -> None:
    rails = _FakeRails(SimpleNamespace(status=RailStatus.PASSED, rail=None, content=_ECHO_LATEST_USER))
    policy = text_policy._TextRailsPolicy(rails, 5_000)
    await policy._permit.acquire()
    tasks = [
        asyncio.create_task(_check_input(policy, [{"role": "user", "content": f"queued-{index}"}]))
        for index in range(text_policy.MAX_PENDING_CHECKS)
    ]

    async def wait_until_full() -> None:
        while policy._pending < text_policy.MAX_PENDING_CHECKS:
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(wait_until_full(), timeout=1)
        overloaded = await _check_input(policy, [{"role": "user", "content": "over capacity"}])
        assert overloaded == "NeMo Guardrails input check is overloaded"
        assert rails.calls == []
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        policy._permit.release()

    assert policy._pending == 0
    assert await _check_input(policy, [{"role": "user", "content": "after overload"}]) is None


def test_no_store_history_discards_all_writes() -> None:
    cache = worker._NoStoreHistory()
    cache["one"] = ["private"]
    cache.update({"two": ["private"]})
    assert cache.setdefault("three", ["private"]) == ["private"]
    cache |= {"four": ["private"]}

    assert cache == {}


@pytest.mark.parametrize("rail", [None, 17, "line\nbreak", "x" * 129, "private:rail"])
def test_unsafe_blocking_rail_names_are_not_returned(rail: object) -> None:
    assert text_policy._blocked_reason(rail) == "NeMo Guardrails prevented execution during input checking"
