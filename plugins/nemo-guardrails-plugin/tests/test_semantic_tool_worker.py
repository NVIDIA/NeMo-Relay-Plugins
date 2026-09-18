# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_relay_plugin import DiagnosticLevel, PluginContext, ToolExecutionContext, ToolExecutionResult
from registration_helpers import registered_llm_execution

from nemoguardrails_nemo_relay import configuration, worker
from nemoguardrails_nemo_relay.semantic_tools import SemanticToolVerdict, SemanticToolVerdictKind

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CONFIG = PROJECT_ROOT / "examples" / "no-model-rails"
INPUT_OUTPUT_CONFIG = PROJECT_ROOT / "examples" / "input-output-rails"


def _context() -> MagicMock:
    context = MagicMock(spec=PluginContext)
    context.runtime = SimpleNamespace(list_runtime_registrations=AsyncMock(return_value=[]))
    return context


def _chat_request_with_result() -> dict[str, object]:
    return {
        "headers": {},
        "content": {
            "model": "fixture",
            "messages": [
                {"role": "user", "content": "look it up"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"city":"Paris"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
                {"role": "user", "content": "summarize"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    }


def _chat_response() -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "fixture",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "It is sunny."},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
    }


def test_semantic_result_validation_requires_output_rails() -> None:
    diagnostics = configuration._validate_config(
        {
            "config_path": str(INPUT_CONFIG),
            "semantic_tool_policy": {"result_source": "execution"},
        }
    )

    assert [item.code for item in diagnostics if item.level == DiagnosticLevel.ERROR] == [
        "nemoguardrails.nemo_relay.semantic_tool_output_phase_missing"
    ]


def test_semantic_argument_validation_requires_input_rails(tmp_path: Path) -> None:
    config = tmp_path / "output-only"
    shutil.copytree(INPUT_OUTPUT_CONFIG, config)
    (config / "config.yml").write_text("rails:\n  output:\n    flows:\n      - output rail\n", encoding="utf-8")

    diagnostics = configuration._validate_config(
        {
            "config_path": str(config),
            "semantic_tool_policy": {"check_arguments": True},
        }
    )

    assert any(item.code.endswith("semantic_tool_input_phase_missing") for item in diagnostics)


def test_remote_semantic_validation_uses_declared_phases() -> None:
    diagnostics = configuration._validate_config(
        {
            "remote_checks": {
                "endpoint": "https://guardrails.example/v1/checks",
                "config_ids": ["default"],
                "phases": ["input"],
                "model": "guardrails-evaluator",
                "allow_remote_content_logging_and_retention": True,
            },
            "semantic_tool_policy": {"check_arguments": True, "result_source": "history"},
        }
    )

    errors = [item.code for item in diagnostics if item.level == DiagnosticLevel.ERROR]
    assert errors == ["nemoguardrails.nemo_relay.semantic_tool_output_phase_missing"]


@pytest.mark.parametrize(
    "url",
    [
        "http://actions.example.com",
        "https://user:password@actions.example.com",
        "https://actions.example.com/prefix",
    ],
)
def test_actions_server_url_fails_activation_before_runtime(
    tmp_path: Path,
    url: str,
) -> None:
    config = tmp_path / "remote-actions"
    shutil.copytree(INPUT_CONFIG, config)
    (config / "config.yml").write_text(
        f"actions_server_url: {url}\nrails:\n  input:\n    flows:\n      - input rail\n",
        encoding="utf-8",
    )

    diagnostics = configuration._validate_config({"config_path": str(config)})

    errors = [item for item in diagnostics if item.level == DiagnosticLevel.ERROR]
    assert any(item.code.endswith("invalid_actions_server_url") for item in errors)
    assert all("password" not in item.message for item in diagnostics)


async def test_execution_mode_registers_actual_tool_intercept_and_preserves_result() -> None:
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(
        context,
        {
            "config_path": str(INPUT_OUTPUT_CONFIG),
            "check_timeout_ms": 1_000,
            "semantic_tool_policy": {
                "check_arguments": True,
                "result_source": "execution",
            },
        },
    )
    callback = context.register_tool_execution_intercept.call_args.args[1]
    tool_context = ToolExecutionContext(
        tool_name="lookup",
        args={"city": "Paris"},
        tool_call_id="call-1",
    )
    downstream = ToolExecutionResult(
        result={"weather": "sunny"},
        annotation={"provider": "fixture"},
    )
    next_call = SimpleNamespace(call=AsyncMock(return_value=downstream))

    outcome = await callback(tool_context, next_call)

    assert outcome.result is downstream.result
    assert outcome.annotation is downstream.annotation
    next_call.call.assert_awaited_once_with(tool_context.args)
    assert context.register_tool_execution_intercept.call_args.args[0] == "guardrails-semantic-tools"
    assert context.register_tool_execution_intercept.call_args.kwargs == {
        "priority": worker.OUTERMOST_EXECUTION_PRIORITY
    }
    await plugin.close()


async def test_history_mode_checks_each_prior_result_once_without_tool_intercept() -> None:
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(
        context,
        {
            "config_path": str(INPUT_OUTPUT_CONFIG),
            "check_timeout_ms": 1_000,
            "semantic_tool_policy": {"result_source": "history"},
        },
    )
    assert plugin._text_policy is not None
    plugin._text_policy.check_tool_result = AsyncMock(  # type: ignore[method-assign]
        return_value=SemanticToolVerdict(SemanticToolVerdictKind.PASSED)
    )
    callback = registered_llm_execution(context).callback
    request = _chat_request_with_result()
    response = _chat_response()
    next_call = SimpleNamespace(call=AsyncMock(return_value=response))

    assert await callback("openai.chat_completions", request, next_call) is response

    plugin._text_policy.check_tool_result.assert_awaited_once()  # type: ignore[attr-defined]
    context.register_tool_execution_intercept.assert_not_called()
    next_call.call.assert_awaited_once_with(request)
    await plugin.close()


async def test_semantic_tool_block_is_content_free_at_worker_boundary() -> None:
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(
        context,
        {
            "config_path": str(INPUT_OUTPUT_CONFIG),
            "check_timeout_ms": 1_000,
            "semantic_tool_policy": {"check_arguments": True},
        },
    )
    assert plugin._text_policy is not None
    plugin._text_policy.check_tool_arguments = AsyncMock(  # type: ignore[method-assign]
        return_value=SemanticToolVerdict(SemanticToolVerdictKind.POLICY_BLOCK)
    )
    callback = context.register_tool_execution_intercept.call_args.args[1]
    canary = "PRIVATE_ARGUMENT_CANARY_9341"
    next_call = SimpleNamespace(call=AsyncMock())

    with pytest.raises(Exception) as captured:
        await callback(
            ToolExecutionContext(tool_name="lookup", args={"secret": canary}),
            next_call,
        )

    assert str(captured.value) == "NeMo Guardrails rejected tool arguments"
    assert canary not in str(captured.value)
    next_call.call.assert_not_awaited()
    await plugin.close()
