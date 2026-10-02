# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nemo_relay_plugin import ToolExecutionContext, ToolExecutionResult

from nemoguardrails_nemo_relay import execution_policy
from nemoguardrails_nemo_relay.policy_error import PolicyRejectedError
from nemoguardrails_nemo_relay.semantic_tools import (
    MAX_SEMANTIC_HISTORY_RESULTS,
    SemanticResultSource,
    SemanticToolPolicy,
    SemanticToolPolicyError,
    SemanticToolSettings,
    SemanticToolVerdict,
    SemanticToolVerdictKind,
    canonical_history_tool_result,
    canonical_tool_arguments,
    canonical_tool_result,
    semantic_tool_settings_from_config,
)
from nemoguardrails_nemo_relay.structural_tools import ToolCall, ToolExchange, ToolResult


class _Checker:
    def __init__(
        self,
        *,
        arguments: SemanticToolVerdictKind = SemanticToolVerdictKind.PASSED,
        result: SemanticToolVerdictKind = SemanticToolVerdictKind.PASSED,
        delay: float = 0,
    ) -> None:
        self.argument_verdict = arguments
        self.result_verdict = result
        self.delay = delay
        self.argument_calls: list[str] = []
        self.result_calls: list[tuple[str, str]] = []

    async def check_tool_arguments(self, canonical_arguments: str) -> SemanticToolVerdict:
        self.argument_calls.append(canonical_arguments)
        if self.delay:
            await asyncio.sleep(self.delay)
        return SemanticToolVerdict(self.argument_verdict)

    async def check_tool_result(
        self,
        canonical_arguments: str,
        canonical_result: str,
    ) -> SemanticToolVerdict:
        self.result_calls.append((canonical_arguments, canonical_result))
        if self.delay:
            await asyncio.sleep(self.delay)
        return SemanticToolVerdict(self.result_verdict)


def test_llm_and_tool_blocks_share_the_relay_policy_error_base() -> None:
    assert issubclass(execution_policy._LlmPolicyError, PolicyRejectedError)
    assert issubclass(SemanticToolPolicyError, PolicyRejectedError)


def _context(index: int = 1) -> ToolExecutionContext:
    return ToolExecutionContext(
        tool_name="lookup",
        args={"index": index, "query": "weather"},
        tool_call_id=f"call-{index}",
    )


def _next(*, result: object = None, annotation: object = None) -> SimpleNamespace:
    return SimpleNamespace(
        call=AsyncMock(
            return_value=ToolExecutionResult(
                result={"answer": "sunny"} if result is None else result,
                annotation={"source": "fixture"} if annotation is None else annotation,
            )
        )
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, SemanticToolSettings()),
        (
            {"check_arguments": True, "result_source": "both", "max_payload_bytes": 4_096},
            SemanticToolSettings(
                check_arguments=True,
                result_source=SemanticResultSource.BOTH,
                max_payload_bytes=4_096,
            ),
        ),
    ],
)
def test_settings_are_explicit_and_bounded(value: object, expected: SemanticToolSettings) -> None:
    assert semantic_tool_settings_from_config(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"unknown": True},
        {"check_arguments": 1},
        {"result_source": "automatic"},
        {"max_payload_bytes": True},
        {"max_payload_bytes": 1},
        {"max_payload_bytes": 1_048_577},
    ],
)
def test_invalid_settings_fail_without_echoing_values(value: object) -> None:
    with pytest.raises(ValueError) as captured:
        semantic_tool_settings_from_config(value)
    assert "automatic" not in str(captured.value)


def test_canonical_json_is_stable_and_preserves_all_fields() -> None:
    arguments = canonical_tool_arguments("lookup", {"z": 2, "a": [True, None]}, max_bytes=4_096)
    result = canonical_tool_result(
        "lookup",
        {"z": 2, "a": [True, None]},
        {"trace": "kept"},
        max_bytes=4_096,
    )
    history = canonical_history_tool_result(
        "lookup",
        ToolResult(call_id="call-1", name="lookup", content="sunny", is_error=True),
        max_bytes=4_096,
    )

    assert arguments == '{"arguments":{"a":[true,null],"z":2},"tool_name":"lookup"}'
    assert json.loads(result) == {
        "annotation": {"trace": "kept"},
        "result": {"a": [True, None], "z": 2},
        "tool_name": "lookup",
    }
    assert json.loads(history) == {
        "is_error": True,
        "result": "sunny",
        "tool_name": "lookup",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"value": "x" * 4_096},
        {"value": float("nan")},
        {"value": object()},
        {1: "non-string-key"},
    ],
)
def test_canonical_json_rejects_oversized_or_non_json_payload(payload: object) -> None:
    with pytest.raises(SemanticToolPolicyError) as captured:
        canonical_tool_arguments("lookup", payload, max_bytes=1_024)
    assert captured.value.verdict.kind is SemanticToolVerdictKind.COVERAGE_BLOCK
    assert isinstance(captured.value, PolicyRejectedError)
    assert captured.value.phase == "arguments"
    assert "tool arguments" in str(captured.value)
    assert "x" * 32 not in str(captured.value)


def test_result_serialization_failure_reports_the_result_phase() -> None:
    with pytest.raises(SemanticToolPolicyError) as captured:
        canonical_tool_result("lookup", object(), None, max_bytes=1_024)

    assert captured.value.verdict.kind is SemanticToolVerdictKind.COVERAGE_BLOCK
    assert captured.value.phase == "result"
    assert "tool result" in str(captured.value)


async def test_execution_checks_arguments_and_result_without_mutation() -> None:
    checker = _Checker()
    policy = SemanticToolPolicy(
        SemanticToolSettings(check_arguments=True, result_source=SemanticResultSource.EXECUTION),
        checker,
        timeout_ms=1_000,
    )
    context = _context()
    next_call = _next()

    outcome = await policy.execute(context, next_call)

    assert outcome.result == {"answer": "sunny"}
    assert outcome.annotation == {"source": "fixture"}
    next_call.call.assert_awaited_once_with(context.args)
    assert json.loads(checker.argument_calls[0])["arguments"] == context.args
    assert json.loads(checker.result_calls[0][1])["result"] == outcome.result


async def test_inner_continuation_is_explicit_tool_argument_trust_boundary() -> None:
    """Pin that Relay can replace arguments after this interceptor checks them."""

    checker = _Checker()
    policy = SemanticToolPolicy(
        SemanticToolSettings(check_arguments=True, result_source=SemanticResultSource.EXECUTION),
        checker,
        timeout_ms=1_000,
    )
    original = _context()
    replacement = {"secret": "changed-by-inner-middleware"}

    class ReplacingContinuation:
        async def call(self, args: object) -> ToolExecutionResult:
            assert args is original.args
            return ToolExecutionResult(
                result={"executed_arguments": replacement},
                annotation={"source": "inner-interceptor"},
            )

    outcome = await policy.execute(original, ReplacingContinuation())  # type: ignore[arg-type]

    assert json.loads(checker.argument_calls[0])["arguments"] == original.args
    assert json.loads(checker.result_calls[0][0])["arguments"] == original.args
    assert outcome.result == {"executed_arguments": replacement}


@pytest.mark.parametrize(
    "verdict",
    [
        SemanticToolVerdictKind.POLICY_BLOCK,
        SemanticToolVerdictKind.MODIFIED,
        SemanticToolVerdictKind.COVERAGE_BLOCK,
        SemanticToolVerdictKind.TIMEOUT,
        SemanticToolVerdictKind.OVERLOADED,
        SemanticToolVerdictKind.CHECK_FAILURE,
    ],
)
async def test_non_pass_argument_outcomes_fail_before_tool_execution(verdict: SemanticToolVerdictKind) -> None:
    checker = _Checker(arguments=verdict)
    policy = SemanticToolPolicy(SemanticToolSettings(check_arguments=True), checker, timeout_ms=1_000)
    next_call = _next()

    with pytest.raises(SemanticToolPolicyError) as captured:
        await policy.execute(_context(), next_call)

    assert captured.value.verdict.kind is verdict
    next_call.call.assert_not_awaited()


async def test_modified_result_is_rejected_without_rewriting_json() -> None:
    checker = _Checker(result=SemanticToolVerdictKind.MODIFIED)
    policy = SemanticToolPolicy(
        SemanticToolSettings(result_source=SemanticResultSource.EXECUTION),
        checker,
        timeout_ms=1_000,
    )
    next_call = _next(result={"private": "original"})

    with pytest.raises(SemanticToolPolicyError, match="cannot safely rewrite JSON"):
        await policy.execute(_context(), next_call)

    next_call.call.assert_awaited_once()


async def test_concurrent_tool_calls_keep_canonical_payloads_isolated() -> None:
    checker = _Checker(delay=0.001)
    policy = SemanticToolPolicy(
        SemanticToolSettings(check_arguments=True, result_source=SemanticResultSource.EXECUTION),
        checker,
        timeout_ms=1_000,
    )
    next_calls = [_next(result={"index": index}) for index in range(16)]

    outcomes = await asyncio.gather(*(policy.execute(_context(index), next_calls[index]) for index in range(16)))

    assert [outcome.result for outcome in outcomes] == [{"index": index} for index in range(16)]
    assert {json.loads(value)["arguments"]["index"] for value in checker.argument_calls} == set(range(16))
    assert {json.loads(value)["result"]["index"] for _, value in checker.result_calls} == set(range(16))


async def test_host_cancellation_propagates_and_does_not_start_tool() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingChecker(_Checker):
        async def check_tool_arguments(self, canonical_arguments: str) -> SemanticToolVerdict:
            del canonical_arguments
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    policy = SemanticToolPolicy(
        SemanticToolSettings(check_arguments=True),
        BlockingChecker(),
        timeout_ms=1_000,
    )
    next_call = _next()
    task = asyncio.create_task(policy.execute(_context(), next_call))
    await asyncio.wait_for(started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(cancelled.wait(), timeout=1)
    next_call.call.assert_not_awaited()


async def test_history_mode_pairs_calls_and_does_not_recheck_execution() -> None:
    checker = _Checker()
    policy = SemanticToolPolicy(
        SemanticToolSettings(result_source=SemanticResultSource.HISTORY),
        checker,
        timeout_ms=1_000,
    )
    exchange = ToolExchange(
        calls=(ToolCall(call_id="call-1", name="lookup", arguments={"q": "weather"}),),
        results=(ToolResult(call_id="call-1", name=None, content="sunny"),),
    )

    await policy.check_history((exchange,))
    next_call = _next()
    outcome = await policy.execute(_context(), next_call)

    assert outcome.result == {"answer": "sunny"}
    assert len(checker.result_calls) == 1
    next_call.call.assert_awaited_once()


async def test_execution_mode_does_not_duplicate_checks_from_history() -> None:
    checker = _Checker()
    policy = SemanticToolPolicy(
        SemanticToolSettings(result_source=SemanticResultSource.EXECUTION),
        checker,
        timeout_ms=1_000,
    )
    exchange = ToolExchange(
        calls=(ToolCall(call_id="call-1", name="lookup", arguments={}),),
        results=(ToolResult(call_id="call-1", name="lookup", content="sunny"),),
    )

    await policy.check_history((exchange,))

    assert checker.result_calls == []


async def test_history_mode_rejects_ambiguous_result_linkage() -> None:
    checker = _Checker()
    policy = SemanticToolPolicy(
        SemanticToolSettings(result_source=SemanticResultSource.HISTORY),
        checker,
        timeout_ms=1_000,
    )
    exchange = ToolExchange(
        calls=(
            ToolCall(call_id="call-1", name="lookup", arguments={}),
            ToolCall(call_id="call-2", name="lookup", arguments={}),
        ),
        results=(ToolResult(call_id=None, name="lookup", content="sunny"),),
    )

    with pytest.raises(SemanticToolPolicyError) as captured:
        await policy.check_history((exchange,))

    assert captured.value.verdict.kind is SemanticToolVerdictKind.COVERAGE_BLOCK
    assert checker.result_calls == []


async def test_history_mode_rejects_too_many_results_before_checking() -> None:
    checker = _Checker()
    policy = SemanticToolPolicy(
        SemanticToolSettings(result_source=SemanticResultSource.HISTORY),
        checker,
        timeout_ms=1_000,
    )
    calls = tuple(
        ToolCall(call_id=f"call-{index}", name="lookup", arguments={})
        for index in range(MAX_SEMANTIC_HISTORY_RESULTS + 1)
    )
    results = tuple(
        ToolResult(call_id=f"call-{index}", name="lookup", content="safe")
        for index in range(MAX_SEMANTIC_HISTORY_RESULTS + 1)
    )

    with pytest.raises(SemanticToolPolicyError) as captured:
        await policy.check_history((ToolExchange(calls=calls, results=results),))

    assert captured.value.verdict.kind is SemanticToolVerdictKind.COVERAGE_BLOCK
    assert checker.result_calls == []


async def test_history_mode_has_one_deadline_for_the_whole_batch() -> None:
    started = asyncio.Event()

    class BlockingChecker(_Checker):
        async def check_tool_result(
            self,
            canonical_arguments: str,
            canonical_result: str,
        ) -> SemanticToolVerdict:
            self.result_calls.append((canonical_arguments, canonical_result))
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    checker = BlockingChecker()
    policy = SemanticToolPolicy(
        SemanticToolSettings(result_source=SemanticResultSource.HISTORY),
        checker,
        timeout_ms=20,
    )
    exchange = ToolExchange(
        calls=(ToolCall(call_id="call-1", name="lookup", arguments={}),),
        results=(ToolResult(call_id="call-1", name="lookup", content="safe"),),
    )

    with pytest.raises(SemanticToolPolicyError) as captured:
        await policy.check_history((exchange,))

    assert started.is_set()
    assert captured.value.verdict.kind is SemanticToolVerdictKind.TIMEOUT
