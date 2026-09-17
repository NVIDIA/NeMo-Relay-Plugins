# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_relay_plugin import DataSchema, LogSeverity, PluginContext, ToolExecutionContext, ToolExecutionResult
from provider_cases import anthropic_request, chat_request, chat_response
from registration_helpers import registered_llm_execution
from tool_projection_cases import openai_chat_case

from nemoguardrails_nemo_relay import execution_policy, policy_marks, worker
from nemoguardrails_nemo_relay.policy_marks import (
    MAX_DURATION_MS,
    POLICY_DECISION_MARK,
    PolicyAction,
    PolicyBackend,
    PolicyDecisionMarks,
    PolicyOutcome,
    PolicyPhase,
)
from nemoguardrails_nemo_relay.semantic_tools import (
    SemanticResultSource,
    SemanticToolPolicy,
    SemanticToolPolicyError,
    SemanticToolSettings,
    SemanticToolVerdict,
    SemanticToolVerdictKind,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_OUTPUT_CONFIG = PROJECT_ROOT / "examples" / "input-output-rails"
STRUCTURAL_TOOL_CONFIG = PROJECT_ROOT / "examples" / "structural-tool-rails"


def _context() -> tuple[MagicMock, AsyncMock]:
    context = MagicMock(spec=PluginContext)
    emit_mark = AsyncMock()
    context.runtime = SimpleNamespace(
        list_runtime_registrations=AsyncMock(return_value=[]),
        emit_mark=emit_mark,
    )
    return context, emit_mark


def _mark_data(emit_mark: AsyncMock) -> list[dict[str, object]]:
    return [call.args[1] for call in emit_mark.await_args_list]


async def test_policy_mark_has_only_bounded_content_free_fields() -> None:
    emit_mark = AsyncMock()
    marks = PolicyDecisionMarks(SimpleNamespace(emit_mark=emit_mark), text_backend=PolicyBackend.REMOTE)

    await marks.emit(
        phase=PolicyPhase.INPUT,
        outcome=PolicyOutcome.POLICY_BLOCK,
        action=PolicyAction.REJECT,
        started_ns=-(10**30),
    )

    emit_mark.assert_awaited_once()
    assert emit_mark.await_args.args[0] == POLICY_DECISION_MARK
    assert emit_mark.await_args.args[1] == {
        "phase": "input",
        "backend": "remote",
        "outcome": "policy_block",
        "action": "reject",
        "duration_ms": MAX_DURATION_MS,
    }
    assert emit_mark.await_args.kwargs == {
        "data_schema": DataSchema(POLICY_DECISION_MARK, "1"),
        "severity": LogSeverity.INFO,
    }


async def test_policy_mark_failure_does_not_replace_the_policy_decision() -> None:
    runtime = SimpleNamespace(emit_mark=AsyncMock(side_effect=RuntimeError("PRIVATE_MARK_FAILURE")))
    marks = PolicyDecisionMarks(runtime, text_backend=PolicyBackend.LOCAL)

    await marks.emit(
        phase=PolicyPhase.OUTPUT,
        outcome=PolicyOutcome.PASSED,
        action=PolicyAction.CONTINUE,
        started_ns=PolicyDecisionMarks.start(),
    )

    runtime.emit_mark.assert_awaited_once()


async def test_policy_mark_timeout_cancels_a_wedged_runtime_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def wedged_emit_mark(*_args: object, **_kwargs: object) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(policy_marks, "MARK_EMIT_TIMEOUT_SECONDS", 0.01)
    marks = PolicyDecisionMarks(
        SimpleNamespace(emit_mark=wedged_emit_mark),
        text_backend=PolicyBackend.LOCAL,
    )

    await asyncio.wait_for(
        marks.emit(
            phase=PolicyPhase.INPUT,
            outcome=PolicyOutcome.PASSED,
            action=PolicyAction.CONTINUE,
            started_ns=PolicyDecisionMarks.start(),
        ),
        timeout=0.1,
    )

    assert started.is_set()
    assert cancelled.is_set()


async def test_policy_mark_preserves_invocation_cancellation() -> None:
    runtime = SimpleNamespace(emit_mark=AsyncMock(side_effect=asyncio.CancelledError))
    marks = PolicyDecisionMarks(runtime, text_backend=PolicyBackend.LOCAL)

    with pytest.raises(asyncio.CancelledError):
        await marks.emit(
            phase=PolicyPhase.INPUT,
            outcome=PolicyOutcome.PASSED,
            action=PolicyAction.CONTINUE,
            started_ns=PolicyDecisionMarks.start(),
        )


async def test_text_policy_emits_one_content_free_mark_per_enabled_phase() -> None:
    context, emit_mark = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(
        context,
        {"config_path": str(INPUT_OUTPUT_CONFIG), "check_timeout_ms": 1_000},
    )
    callback = registered_llm_execution(context).callback
    request = chat_request([{"role": "user", "content": "PRIVATE_INPUT_CANARY"}])
    response = chat_response("PRIVATE_OUTPUT_CANARY")

    assert await callback("fixture", request, SimpleNamespace(call=AsyncMock(return_value=response))) is response

    data = _mark_data(emit_mark)
    assert data == [
        {
            "phase": "input",
            "backend": "local",
            "outcome": "passed",
            "action": "continue",
            "duration_ms": data[0]["duration_ms"],
        },
        {
            "phase": "output",
            "backend": "local",
            "outcome": "passed",
            "action": "continue",
            "duration_ms": data[1]["duration_ms"],
        },
    ]
    assert all(call.args[0] == POLICY_DECISION_MARK for call in emit_mark.await_args_list)
    serialized = json.dumps(data)
    assert "PRIVATE_INPUT_CANARY" not in serialized
    assert "PRIVATE_OUTPUT_CANARY" not in serialized

    emit_mark.reset_mock()
    blocked_next = SimpleNamespace(call=AsyncMock())
    with pytest.raises(execution_policy._LlmPolicyError, match="input rail"):
        await callback(
            "fixture",
            chat_request([{"role": "user", "content": "block input"}]),
            blocked_next,
        )
    blocked_next.call.assert_not_awaited()
    blocked = _mark_data(emit_mark)
    assert [(item["phase"], item["outcome"], item["action"]) for item in blocked] == [
        ("input", "policy_block", "reject")
    ]
    await plugin.close()


async def test_text_mutation_mark_distinguishes_rewrite_from_rejection() -> None:
    context, emit_mark = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(
        context,
        {
            "config_path": str(INPUT_OUTPUT_CONFIG),
            "check_timeout_ms": 1_000,
            "mutation_policy": {"input": "apply"},
        },
    )
    callback = registered_llm_execution(context).callback
    response = chat_response("safe")
    next_call = SimpleNamespace(call=AsyncMock(return_value=response))

    assert (
        await callback(
            "fixture",
            chat_request([{"role": "user", "content": "modify input"}]),
            next_call,
        )
        is response
    )

    data = _mark_data(emit_mark)
    assert [(item["phase"], item["outcome"], item["action"]) for item in data] == [
        ("input", "modified", "rewrite"),
        ("output", "passed", "continue"),
    ]
    forwarded = next_call.call.await_args.args[0]
    assert forwarded["content"]["messages"][-1]["content"] == "modified input"
    await plugin.close()


async def test_mark_publication_failure_preserves_text_allow_and_block() -> None:
    context, emit_mark = _context()
    emit_mark.side_effect = RuntimeError("PRIVATE_MARK_FAILURE")
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(
        context,
        {"config_path": str(INPUT_OUTPUT_CONFIG), "check_timeout_ms": 1_000},
    )
    callback = registered_llm_execution(context).callback
    response = chat_response("safe")

    assert (
        await callback(
            "fixture",
            chat_request([{"role": "user", "content": "safe"}]),
            SimpleNamespace(call=AsyncMock(return_value=response)),
        )
        is response
    )
    blocked_next = SimpleNamespace(call=AsyncMock())
    with pytest.raises(execution_policy._LlmPolicyError, match="input rail"):
        await callback(
            "fixture",
            chat_request([{"role": "user", "content": "block input"}]),
            blocked_next,
        )
    blocked_next.call.assert_not_awaited()
    await plugin.close()


async def test_text_projection_rejection_emits_worker_coverage_mark() -> None:
    context, emit_mark = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(
        context,
        {"config_path": str(INPUT_OUTPUT_CONFIG), "check_timeout_ms": 1_000},
    )
    callback = registered_llm_execution(context).callback
    next_call = SimpleNamespace(call=AsyncMock())

    with pytest.raises(execution_policy._LlmPolicyError, match="unsupported provider request"):
        await callback("fixture", {"headers": {}, "content": {}}, next_call)

    next_call.call.assert_not_awaited()
    assert [(item["phase"], item["backend"], item["outcome"], item["action"]) for item in _mark_data(emit_mark)] == [
        ("input", "worker", "coverage_block", "reject")
    ]
    await plugin.close()


async def test_guarded_stream_rejection_emits_worker_coverage_mark() -> None:
    context, emit_mark = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(
        context,
        {"config_path": str(INPUT_OUTPUT_CONFIG), "check_timeout_ms": 1_000},
    )
    callback = registered_llm_execution(context, stream=True).callback
    next_call = SimpleNamespace(call=MagicMock())

    with pytest.raises(execution_policy._LlmPolicyError, match="guarded streaming response"):
        await callback("fixture", chat_request(), next_call)

    next_call.call.assert_not_called()
    assert [(item["phase"], item["backend"], item["outcome"], item["action"]) for item in _mark_data(emit_mark)] == [
        ("output", "worker", "coverage_block", "reject")
    ]
    await plugin.close()


async def test_invalid_execution_context_emits_worker_coverage_mark() -> None:
    context, emit_mark = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(
        context,
        {"config_path": str(INPUT_OUTPUT_CONFIG), "check_timeout_ms": 1_000},
    )
    assert plugin._execution_policy is not None
    next_call = SimpleNamespace(call=AsyncMock())

    with pytest.raises(execution_policy._LlmPolicyError):
        await plugin._execution_policy.execute_with_context(
            "fixture",
            chat_request(),
            SimpleNamespace(),
            next_call,
        )

    next_call.call.assert_not_awaited()
    assert [(item["phase"], item["backend"], item["outcome"], item["action"]) for item in _mark_data(emit_mark)] == [
        ("input", "worker", "coverage_block", "reject")
    ]
    await plugin.close()


async def test_count_tokens_response_shape_rejection_is_observable() -> None:
    context, emit_mark = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(
        context,
        {"config_path": str(INPUT_OUTPUT_CONFIG), "check_timeout_ms": 1_000},
    )
    callback = registered_llm_execution(context).callback
    request = anthropic_request()
    request["content"].pop("max_tokens")  # type: ignore[union-attr]
    next_call = SimpleNamespace(call=AsyncMock(return_value={"input_tokens": -1}))

    with pytest.raises(execution_policy._LlmPolicyError, match="count-tokens response"):
        await callback("anthropic.count_tokens", request, next_call)

    assert [(item["phase"], item["backend"], item["outcome"], item["action"]) for item in _mark_data(emit_mark)] == [
        ("input", "local", "passed", "continue"),
        ("output", "worker", "coverage_block", "reject"),
    ]
    await plugin.close()


async def test_structural_tool_policy_emits_one_mark_per_executed_phase() -> None:
    context, emit_mark = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, {"config_path": str(STRUCTURAL_TOOL_CONFIG)})
    callback = registered_llm_execution(context).callback
    request, response = openai_chat_case()
    request["content"]["messages"][-1]["content"] = "PRIVATE_TOOL_RESULT_CANARY"  # type: ignore[index]

    assert (
        await callback(
            "openai.chat_completions",
            request,
            SimpleNamespace(call=AsyncMock(return_value=response)),
        )
        is response
    )

    data = _mark_data(emit_mark)
    assert [(item["phase"], item["backend"], item["outcome"]) for item in data] == [
        ("tool_results", "structural", "passed"),
        ("model_tool_calls", "structural", "passed"),
    ]
    assert "PRIVATE_TOOL_RESULT_CANARY" not in json.dumps(data)
    await plugin.close()


async def test_structural_only_guarded_stream_marks_model_tool_calls() -> None:
    context, emit_mark = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, {"config_path": str(STRUCTURAL_TOOL_CONFIG)})
    callback = registered_llm_execution(context, stream=True).callback
    next_call = SimpleNamespace(call=MagicMock())

    with pytest.raises(execution_policy._LlmPolicyError, match="guarded streaming response"):
        await callback("openai.chat_completions", openai_chat_case()[0], next_call)

    next_call.call.assert_not_called()
    assert [(item["phase"], item["backend"], item["outcome"], item["action"]) for item in _mark_data(emit_mark)] == [
        ("model_tool_calls", "worker", "coverage_block", "reject")
    ]
    await plugin.close()


async def test_tool_only_response_does_not_report_a_vacuous_output_pass(tmp_path: Path) -> None:
    config_path = tmp_path / "text-and-tools"
    shutil.copytree(INPUT_OUTPUT_CONFIG, config_path)
    (config_path / "config.yml").write_text(
        "rails:\n"
        "  input:\n"
        "    flows: [input rail]\n"
        "  output:\n"
        "    flows: [output rail]\n"
        "  tool_output:\n"
        "    flows: [tool call validation]\n",
        encoding="utf-8",
    )
    context, emit_mark = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, {"config_path": str(config_path)})
    callback = registered_llm_execution(context).callback
    request, response = openai_chat_case()
    request["content"]["messages"] = request["content"]["messages"][:1]  # type: ignore[index]

    assert (
        await callback(
            "openai.chat_completions",
            request,
            SimpleNamespace(call=AsyncMock(return_value=response)),
        )
        is response
    )

    assert [item["phase"] for item in _mark_data(emit_mark)] == ["input", "model_tool_calls"]
    await plugin.close()


async def test_structural_adapter_failure_emits_content_free_failure_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, emit_mark = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, {"config_path": str(STRUCTURAL_TOOL_CONFIG)})
    assert plugin._execution_policy is not None
    adapter = plugin._execution_policy._adapter
    assert adapter is not None
    monkeypatch.setattr(
        adapter,
        "check_results",
        AsyncMock(side_effect=RuntimeError("PRIVATE_ADAPTER_FAILURE")),
    )
    callback = registered_llm_execution(context).callback
    next_call = SimpleNamespace(call=AsyncMock())
    request, _ = openai_chat_case()

    with pytest.raises(execution_policy._LlmPolicyError, match="tool-result check failed") as error:
        await callback("openai.chat_completions", request, next_call)

    assert "PRIVATE_ADAPTER_FAILURE" not in str(error.value)
    next_call.call.assert_not_awaited()
    data = _mark_data(emit_mark)
    assert [(item["phase"], item["backend"], item["outcome"], item["action"]) for item in data] == [
        ("tool_results", "structural", "check_failure", "reject")
    ]
    assert "PRIVATE_ADAPTER_FAILURE" not in json.dumps(data)
    await plugin.close()


async def test_model_call_adapter_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, emit_mark = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, {"config_path": str(STRUCTURAL_TOOL_CONFIG)})
    assert plugin._execution_policy is not None
    adapter = plugin._execution_policy._adapter
    assert adapter is not None
    monkeypatch.setattr(
        adapter,
        "check_call_candidates",
        AsyncMock(side_effect=RuntimeError("PRIVATE_MODEL_CALL_FAILURE")),
    )
    callback = registered_llm_execution(context).callback
    request, response = openai_chat_case()

    with pytest.raises(execution_policy._LlmPolicyError, match="model-call check failed") as error:
        await callback(
            "openai.chat_completions",
            request,
            SimpleNamespace(call=AsyncMock(return_value=response)),
        )

    assert "PRIVATE_MODEL_CALL_FAILURE" not in str(error.value)
    data = _mark_data(emit_mark)
    assert [(item["phase"], item["backend"], item["outcome"], item["action"]) for item in data] == [
        ("tool_results", "structural", "passed", "continue"),
        ("model_tool_calls", "structural", "check_failure", "reject"),
    ]
    assert "PRIVATE_MODEL_CALL_FAILURE" not in json.dumps(data)
    await plugin.close()


class _SemanticChecker:
    async def check_tool_arguments(self, canonical_arguments: str) -> SemanticToolVerdict:
        assert "PRIVATE_ARGUMENT_CANARY" in canonical_arguments
        return SemanticToolVerdict(SemanticToolVerdictKind.PASSED)

    async def check_tool_result(
        self,
        canonical_arguments: str,
        canonical_result: str,
    ) -> SemanticToolVerdict:
        assert "PRIVATE_ARGUMENT_CANARY" in canonical_arguments
        assert "PRIVATE_RESULT_CANARY" in canonical_result
        return SemanticToolVerdict(SemanticToolVerdictKind.POLICY_BLOCK)


async def test_semantic_tool_marks_report_decisions_without_tool_content() -> None:
    emit_mark = AsyncMock()
    marks = PolicyDecisionMarks(SimpleNamespace(emit_mark=emit_mark), text_backend=PolicyBackend.LOCAL)
    policy = SemanticToolPolicy(
        SemanticToolSettings(check_arguments=True, result_source=SemanticResultSource.EXECUTION),
        _SemanticChecker(),
        timeout_ms=1_000,
        decision_marks=marks,
    )
    context = ToolExecutionContext(
        tool_name="lookup",
        args={"query": "PRIVATE_ARGUMENT_CANARY"},
        tool_call_id="call-1",
    )
    next_call = SimpleNamespace(
        call=AsyncMock(
            return_value=ToolExecutionResult(
                result={"answer": "PRIVATE_RESULT_CANARY"},
                annotation=None,
            )
        )
    )

    with pytest.raises(SemanticToolPolicyError, match="rejected tool result"):
        await policy.execute(context, next_call)

    data = _mark_data(emit_mark)
    assert [(item["phase"], item["outcome"], item["action"]) for item in data] == [
        ("tool_arguments", "passed", "continue"),
        ("tool_result", "policy_block", "reject"),
    ]
    serialized = json.dumps(data)
    assert "PRIVATE_ARGUMENT_CANARY" not in serialized
    assert "PRIVATE_RESULT_CANARY" not in serialized
