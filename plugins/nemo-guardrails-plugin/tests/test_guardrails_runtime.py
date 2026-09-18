# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nemoguardrails.context import (
    explain_info_var,
    generation_options_var,
    llm_call_info_var,
    llm_stats_var,
    raw_llm_request,
)
from nemoguardrails.logging.explain import ExplainInfo, LLMCallInfo
from nemoguardrails.logging.processing_log import processing_log_var
from registration_helpers import registered_llm_execution
from worker_test_helpers import registered_worker, worker_context

from nemoguardrails_nemo_relay import (
    configuration,
    execution_policy,
    worker,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NO_MODEL_CONFIG = PROJECT_ROOT / "examples" / "no-model-rails"
MIGRATED_LOCAL_CONFIG = PROJECT_ROOT / "examples" / "migrated-local-rails"


def _request(messages: list[dict[str, object]], **content: object) -> dict[str, object]:
    return {
        "headers": {"authorization": "not-forwarded-to-guardrails"},
        "content": {
            "model": "fixture-model",
            "messages": messages,
            **content,
        },
    }


async def test_real_guardrails_024_passes_blocks_and_rejects_modified() -> None:
    guardrails_context_vars = (
        raw_llm_request,
        processing_log_var,
        generation_options_var,
        llm_stats_var,
        llm_call_info_var,
        explain_info_var,
    )
    context_before = tuple(variable.get() for variable in guardrails_context_vars)
    plugin, callback = await registered_worker(
        {
            "version": 2,
            "config_path": str(NO_MODEL_CONFIG),
            "check_timeout_ms": 1_000,
        }
    )

    allowed = await callback(_request([{"role": "user", "content": "hello"}]))
    blocked = await callback(_request([{"role": "user", "content": "block input"}]))
    modified_request = _request([{"role": "user", "content": "modify input"}])
    original_modified_request = json.loads(json.dumps(modified_request))
    modified = await callback(modified_request)
    prior_blocked_but_latest_allowed = await callback(
        _request(
            [
                {"role": "user", "content": "block input"},
                {"role": "assistant", "content": "prior response"},
                {"role": "user", "content": "hello"},
            ]
        )
    )
    output_only_phrase = await callback(
        _request(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "block output"},
            ]
        )
    )

    assert allowed is None
    assert blocked == "NeMo Guardrails prevented execution in input rail 'input rail'"
    assert modified == "NeMo Guardrails modified the input, but this worker cannot safely rewrite it"
    assert modified_request == original_modified_request
    assert prior_blocked_but_latest_allowed is None
    assert output_only_phrase is None
    assert plugin._rails is not None
    assert plugin._rails.events_history_cache == {}
    assert plugin._rails.explain().llm_calls == []
    assert plugin._rails.explain().colang_history is None
    assert os.environ["NEMO_GUARDRAILS_NO_USAGE_STATS"] == "1"
    assert tuple(variable.get() for variable in guardrails_context_vars) == context_before


async def test_real_guardrails_024_does_not_mutate_inherited_explanation_context() -> None:
    plugin, callback = await registered_worker(
        {
            "config_path": str(NO_MODEL_CONFIG),
            "check_timeout_ms": 1_000,
        }
    )
    inherited = ExplainInfo(
        llm_calls=[LLMCallInfo(prompt="INHERITED_PRIVATE_CANARY")],
        colang_history="INHERITED_PRIVATE_HISTORY",
    )
    token = explain_info_var.set(inherited)
    try:
        result = await callback(_request([{"role": "user", "content": "hello"}]))

        assert result is None
        assert explain_info_var.get() is inherited
        assert inherited.llm_calls[0].prompt == "INHERITED_PRIVATE_CANARY"
        assert inherited.colang_history == "INHERITED_PRIVATE_HISTORY"
        assert plugin._rails is not None
        assert plugin._rails.explain() is not inherited
        assert plugin._rails.explain().llm_calls == []
        assert plugin._rails.explain().colang_history is None
    finally:
        explain_info_var.reset(token)


async def test_real_guardrails_024_concurrent_results_remain_isolated() -> None:
    plugin, callback = await registered_worker(
        {
            "config_path": str(NO_MODEL_CONFIG),
            "check_timeout_ms": 1_000,
        }
    )

    results = await asyncio.gather(
        callback(_request([{"role": "user", "content": "hello-a"}])),
        callback(_request([{"role": "user", "content": "block input"}])),
        callback(_request([{"role": "user", "content": "hello-b"}])),
        callback(_request([{"role": "user", "content": "modify input"}])),
    )

    assert results == [
        None,
        "NeMo Guardrails prevented execution in input rail 'input rail'",
        None,
        "NeMo Guardrails modified the input, but this worker cannot safely rewrite it",
    ]
    assert plugin._rails is not None
    assert plugin._rails.events_history_cache == {}
    assert plugin._rails.explain().llm_calls == []
    assert plugin._rails.explain().colang_history is None


async def test_real_guardrails_024_sequential_flow_outcomes_use_the_net_result(tmp_path: Path) -> None:
    config_path = tmp_path / "sequential-input-rails"
    config_path.mkdir()
    (config_path / "config.yml").write_text(
        "rails:\n  input:\n    flows:\n      - first rail\n      - second rail\n",
        encoding="utf-8",
    )
    (config_path / "rails.co").write_text(
        """
define flow first rail
  if $user_message == "block first"
    bot refuse to respond
    stop
  else if $user_message == "modify net"
    $user_message = "modified by first"
  else if $user_message == "modify restore"
    $user_message = "temporary value"

define flow second rail
  if $user_message == "block second"
    bot refuse to respond
    stop
  else if $user_message == "modified by first"
    $user_message = "modified by second"
  else if $user_message == "temporary value"
    $user_message = "modify restore"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    plugin, callback = await registered_worker({"config_path": str(config_path), "check_timeout_ms": 1_000})

    assert await callback(_request([{"role": "user", "content": "hello"}])) is None
    assert await callback(_request([{"role": "user", "content": "block first"}])) == (
        "NeMo Guardrails prevented execution in input rail 'first rail'"
    )
    assert await callback(_request([{"role": "user", "content": "block second"}])) == (
        "NeMo Guardrails prevented execution in input rail 'second rail'"
    )
    assert await callback(_request([{"role": "user", "content": "modify net"}])) == (
        "NeMo Guardrails modified the input, but this worker cannot safely rewrite it"
    )
    # Guardrails reports PASSED when later flows restore the exact original
    # text. The public result describes the net outcome, not every intermediate
    # flow decision.
    assert await callback(_request([{"role": "user", "content": "modify restore"}])) is None
    assert plugin._rails is not None
    assert plugin._rails.explain().llm_calls == []
    assert plugin._rails.explain().colang_history is None


async def test_real_guardrails_024_runs_input_and_output_text_rails_in_parallel(tmp_path: Path) -> None:
    config_path = tmp_path / "parallel-text-rails"
    config_path.mkdir()
    (config_path / "config.yml").write_text(
        """
rails:
  input:
    parallel: true
    flows:
      - first input rail
      - second input rail
  output:
    parallel: true
    flows:
      - first output rail
      - second output rail
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (config_path / "rails.co").write_text(
        """
define bot refuse
  "blocked"

define flow first input rail
  $allowed = execute parallel_input_a(text=$user_message)
  if not $allowed
    bot refuse
    stop

define flow second input rail
  $allowed = execute parallel_input_b(text=$user_message)
  if not $allowed
    bot refuse
    stop

define flow first output rail
  $allowed = execute parallel_output_a(text=$bot_message)
  if not $allowed
    bot refuse
    stop

define flow second output rail
  $allowed = execute parallel_output_b(text=$bot_message)
  if not $allowed
    bot refuse
    stop
""".strip()
        + "\n",
        encoding="utf-8",
    )

    context = worker_context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, {"config_path": str(config_path), "check_timeout_ms": 1_000})
    assert plugin._rails is not None
    assert type(plugin._rails.rails_engine).__name__ == "LLMRails"

    active = {"input": 0, "output": 0}
    maximum = {"input": 0, "output": 0}
    calls: list[tuple[str, str, str]] = []

    async def run_action(phase: str, name: str, text: str) -> bool:
        calls.append((phase, name, text))
        active[phase] += 1
        maximum[phase] = max(maximum[phase], active[phase])
        try:
            await asyncio.sleep(0.03)
            return text != f"block {phase}"
        finally:
            active[phase] -= 1

    async def parallel_input_a(text: str) -> bool:
        return await run_action("input", "a", text)

    async def parallel_input_b(text: str) -> bool:
        return await run_action("input", "b", text)

    async def parallel_output_a(text: str) -> bool:
        return await run_action("output", "a", text)

    async def parallel_output_b(text: str) -> bool:
        return await run_action("output", "b", text)

    for name, action in (
        ("parallel_input_a", parallel_input_a),
        ("parallel_input_b", parallel_input_b),
        ("parallel_output_a", parallel_output_a),
        ("parallel_output_b", parallel_output_b),
    ):
        plugin._rails.register_action(action, name)

    callback = registered_llm_execution(context).callback
    safe_response = {
        "id": "chatcmpl-fixture",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "safe output",
                    "function_call": None,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
    }
    safe_request = _request([{"role": "user", "content": "safe input"}])
    safe_next = SimpleNamespace(call=AsyncMock(return_value=safe_response))
    assert await callback("fixture", safe_request, safe_next) is safe_response
    safe_next.call.assert_awaited_once_with(safe_request)

    blocked_input_next = SimpleNamespace(call=AsyncMock())
    with pytest.raises(execution_policy._LlmPolicyError, match="prevented execution in input rail"):
        await callback(
            "fixture",
            _request([{"role": "user", "content": "block input"}]),
            blocked_input_next,
        )
    blocked_input_next.call.assert_not_awaited()

    blocked_output_next = SimpleNamespace(
        call=AsyncMock(
            return_value={
                **safe_response,
                "choices": [
                    {
                        **safe_response["choices"][0],
                        "message": {
                            **safe_response["choices"][0]["message"],
                            "content": "block output",
                        },
                    }
                ],
            }
        )
    )
    with pytest.raises(execution_policy._LlmPolicyError, match="prevented execution in output rail"):
        await callback("fixture", safe_request, blocked_output_next)
    blocked_output_next.call.assert_awaited_once_with(safe_request)

    assert maximum == {"input": 2, "output": 2}
    assert [item[:2] for item in calls].count(("input", "a")) == 3
    assert [item[:2] for item in calls].count(("input", "b")) == 3
    assert [item[:2] for item in calls].count(("output", "a")) == 2
    assert [item[:2] for item in calls].count(("output", "b")) == 2
    await plugin.close()


async def test_real_guardrails_024_sequential_action_cancellation_timeout_and_recovery(tmp_path: Path) -> None:
    config_path = tmp_path / "cancellable-input-rail"
    config_path.mkdir()
    (config_path / "config.yml").write_text(
        "rails:\n  input:\n    flows:\n      - cancellable rail\n",
        encoding="utf-8",
    )
    (config_path / "rails.co").write_text(
        """
define bot refuse
  "blocked"

define flow cancellable rail
  $allowed = execute cancellable_action(text=$user_message)
  if not $allowed
    bot refuse
    stop
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cancel_plugin, cancel_callback = await registered_worker(
        {"config_path": str(config_path), "check_timeout_ms": 1_000}
    )
    assert cancel_plugin._rails is not None
    cancel_started = asyncio.Event()
    cancel_observed = asyncio.Event()
    cancel_calls = 0

    async def action_for_host_cancellation(text: str) -> bool:
        nonlocal cancel_calls
        del text
        cancel_calls += 1
        if cancel_calls > 1:
            return True
        cancel_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancel_observed.set()
            raise

    cancel_plugin._rails.register_action(action_for_host_cancellation, "cancellable_action")
    task = asyncio.create_task(cancel_callback(_request([{"role": "user", "content": "cancel me"}])))
    await asyncio.wait_for(cancel_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(cancel_observed.wait(), timeout=1)
    assert await cancel_callback(_request([{"role": "user", "content": "after cancellation"}])) is None

    timeout_plugin, timeout_callback = await registered_worker(
        {"config_path": str(config_path), "check_timeout_ms": configuration.MIN_CHECK_TIMEOUT_MS}
    )
    assert timeout_plugin._rails is not None
    assert timeout_plugin._text_policy is not None
    # Keep this cancellation regression fast. Configuration validation enforces
    # the operational floor; the policy still accepts milliseconds internally.
    timeout_plugin._text_policy._timeout_seconds = 0.025
    timeout_observed = asyncio.Event()
    timeout_calls = 0

    async def action_for_worker_timeout(text: str) -> bool:
        nonlocal timeout_calls
        del text
        timeout_calls += 1
        if timeout_calls > 1:
            return True
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            timeout_observed.set()
            raise

    timeout_plugin._rails.register_action(action_for_worker_timeout, "cancellable_action")
    started_at = time.monotonic()
    timed_out = await timeout_callback(_request([{"role": "user", "content": "time out"}]))
    elapsed = time.monotonic() - started_at

    assert timed_out == "NeMo Guardrails input check timed out"
    assert elapsed < 0.5
    await asyncio.wait_for(timeout_observed.wait(), timeout=1)
    assert await timeout_callback(_request([{"role": "user", "content": "after timeout"}])) is None
    for plugin in (cancel_plugin, timeout_plugin):
        assert plugin._rails is not None
        assert plugin._rails.explain().llm_calls == []
        assert plugin._rails.explain().colang_history is None


async def test_migrated_local_config_and_custom_action_load_from_config_path() -> None:
    plugin, callback = await registered_worker(
        {
            "config_path": str(MIGRATED_LOCAL_CONFIG),
            "check_timeout_ms": 1_000,
        }
    )

    assert await callback(_request([{"role": "user", "content": "allowed"}])) is None
    assert await callback(_request([{"role": "user", "content": "legacy-block"}])) == (
        "NeMo Guardrails prevented execution in input rail 'migrated input rail'"
    )
    assert plugin._rails is not None
    assert plugin._rails.events_history_cache == {}
    assert plugin._rails.explain().llm_calls == []
    assert plugin._rails.explain().colang_history is None


async def test_worker_forces_llmrails_for_iorails_compatible_config_and_runs_config_py(tmp_path: Path) -> None:
    config_path = tmp_path / "regex-input-rail"
    config_path.mkdir()
    marker = tmp_path / "config-py-loaded"
    (config_path / "config.yml").write_text(
        """
rails:
  config:
    regex_detection:
      input:
        patterns:
          - SECRET-[0-9]+
  input:
    flows:
      - regex check input
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (config_path / "config.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('loaded', encoding='utf-8')\n",
        encoding="utf-8",
    )

    plugin, callback = await registered_worker(
        {
            "config_path": str(config_path),
            "check_timeout_ms": 1_000,
        }
    )

    assert plugin._rails is not None
    assert plugin._rails.use_iorails_engine is False
    assert type(plugin._rails.rails_engine).__name__ == "LLMRails"
    assert marker.read_text(encoding="utf-8") == "loaded"
    assert await callback(_request([{"role": "user", "content": "hello"}])) is None
    assert await callback(_request([{"role": "user", "content": "SECRET-42"}])) == (
        "NeMo Guardrails prevented execution in input rail 'regex check input'"
    )
