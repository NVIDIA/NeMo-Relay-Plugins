# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_relay_plugin import DiagnosticLevel, PluginContext
from registration_helpers import llm_registration_mock, registered_llm_execution
from tool_projection_cases import gemini_generate_content_case

from nemoguardrails_nemo_relay import configuration, execution_policy, worker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRUCTURAL_TOOL_CONFIG = PROJECT_ROOT / "examples" / "structural-tool-rails"


def _context() -> MagicMock:
    context = MagicMock(spec=PluginContext)
    context.runtime = SimpleNamespace(list_runtime_registrations=AsyncMock(return_value=[]))
    return context


def _chat_request(*, with_result: bool = False) -> dict[str, object]:
    messages: list[dict[str, object]] = [{"role": "user", "content": "weather"}]
    if with_result:
        messages.extend(
            [
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
        )
    return {
        "headers": {},
        "content": {
            "model": "fixture",
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
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


def _chat_tool_response(name: str = "weather", arguments: str = '{"city":"Rome"}') -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


async def _registered_callbacks() -> tuple[worker.NeMoGuardrailsRelayWorker, MagicMock, object, object]:
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, {"config_path": str(STRUCTURAL_TOOL_CONFIG)})
    unary = registered_llm_execution(context).callback
    stream = registered_llm_execution(context, stream=True).callback
    return plugin, context, unary, stream


def test_tool_only_configuration_is_supported_without_starting_llmrails() -> None:
    assert configuration._validate_config({"config_path": str(STRUCTURAL_TOOL_CONFIG)}) == []


async def test_registration_uses_existing_llm_surfaces_and_no_text_runtime() -> None:
    plugin, context, _, _ = await _registered_callbacks()

    assert plugin._rails is None
    assert plugin._text_policy is None
    assert plugin._execution_policy is not None
    context.register_llm_conditional_execution_guardrail.assert_not_called()
    unary = registered_llm_execution(context)
    stream = registered_llm_execution(context, stream=True)
    assert unary.call.args[0] == "guardrails"
    assert stream.call.args[0] == "guardrails"
    assert unary.call.kwargs["priority"] == worker.OUTERMOST_EXECUTION_PRIORITY
    assert stream.call.kwargs["priority"] == worker.OUTERMOST_EXECUTION_PRIORITY


async def test_registration_passes_configured_timeout_to_structural_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, int] = {}

    class FakeAdapter:
        def __init__(self, timeout_ms: int) -> None:
            captured["timeout_ms"] = timeout_ms

        async def verify_compatibility(self) -> None:
            return None

        async def close(self) -> None:
            return None

    monkeypatch.setattr(worker, "Guardrails024StructuralToolAdapter", FakeAdapter)
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()

    await plugin.register(
        context,
        {"config_path": str(STRUCTURAL_TOOL_CONFIG), "check_timeout_ms": 1_234},
    )

    assert captured == {"timeout_ms": 1_234}
    await plugin.close()


@pytest.mark.parametrize("fail_probe", [False, True])
async def test_registration_failure_closes_structural_adapter(
    monkeypatch: pytest.MonkeyPatch,
    fail_probe: bool,
) -> None:
    adapter = SimpleNamespace(
        verify_compatibility=AsyncMock(),
        close=AsyncMock(),
    )
    if fail_probe:
        adapter.verify_compatibility.side_effect = RuntimeError("PRIVATE_PROBE_FAILURE")
    monkeypatch.setattr(
        worker,
        "Guardrails024StructuralToolAdapter",
        MagicMock(return_value=adapter),
    )
    context = _context()
    if not fail_probe:
        llm_registration_mock(context, stream=True).side_effect = RuntimeError("PRIVATE_REGISTRATION_FAILURE")
    plugin = worker.NeMoGuardrailsRelayWorker()

    with pytest.raises(Exception) as captured:
        await plugin.register(context, {"config_path": str(STRUCTURAL_TOOL_CONFIG)})

    if fail_probe:
        assert str(captured.value) == "Guardrails structural-tool adapter is incompatible"
    adapter.close.assert_awaited_once_with()
    assert plugin._execution_policy is None


async def test_worker_close_releases_structural_adapter_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = SimpleNamespace(
        verify_compatibility=AsyncMock(),
        close=AsyncMock(),
    )
    monkeypatch.setattr(
        worker,
        "Guardrails024StructuralToolAdapter",
        MagicMock(return_value=adapter),
    )
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(_context(), {"config_path": str(STRUCTURAL_TOOL_CONFIG)})

    await plugin.close()
    await plugin.close()

    adapter.close.assert_awaited_once_with()


async def test_valid_tool_result_and_model_call_pass_without_payload_mutation() -> None:
    _, _, unary, _ = await _registered_callbacks()
    request = _chat_request(with_result=True)
    response = _chat_tool_response()
    next_call = SimpleNamespace(call=AsyncMock(return_value=response))

    assert await unary("openai.chat_completions", request, next_call) is response
    next_call.call.assert_awaited_once_with(request)


async def test_multiple_model_tool_candidates_are_checked_independently() -> None:
    _, _, unary, _ = await _registered_callbacks()
    request = _chat_request()
    response = _chat_tool_response()
    second = _chat_tool_response()["choices"][0]  # type: ignore[index]
    response["choices"].append(second)  # type: ignore[index,union-attr]
    next_call = SimpleNamespace(call=AsyncMock(return_value=response))

    assert await unary("openai.chat_completions", request, next_call) is response

    blocked = _chat_tool_response(name="delete_everything")
    blocked["choices"].insert(0, _chat_tool_response()["choices"][0])  # type: ignore[index,union-attr]
    blocked_next = SimpleNamespace(call=AsyncMock(return_value=blocked))
    with pytest.raises(execution_policy._LlmPolicyError, match="rejected model tool calls"):
        await unary("openai.chat_completions", request, blocked_next)


async def test_count_tokens_checks_tool_results_but_not_model_tool_calls() -> None:
    _, _, unary, _ = await _registered_callbacks()
    request = {
        "headers": {"anthropic-version": "2023-06-01"},
        "content": {
            "model": "fixture",
            "messages": [
                {"role": "user", "content": "call weather"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "weather",
                            "input": {"city": "Paris"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": "sunny",
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        },
    }
    response = {"input_tokens": 23}
    next_call = SimpleNamespace(call=AsyncMock(return_value=response))

    assert await unary("anthropic.count_tokens", request, next_call) is response
    next_call.call.assert_awaited_once_with(request)

    request["content"]["messages"] = [  # type: ignore[index]
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "missing",
                    "content": "PRIVATE_CANARY",
                }
            ],
        }
    ]
    blocked_next = SimpleNamespace(call=AsyncMock())
    with pytest.raises(execution_policy._LlmPolicyError, match="rejected tool results") as captured:
        await unary("anthropic.count_tokens", request, blocked_next)
    assert "PRIVATE_CANARY" not in str(captured.value)
    blocked_next.call.assert_not_awaited()


async def test_orphan_result_is_rejected_before_provider() -> None:
    _, _, unary, _ = await _registered_callbacks()
    request = _chat_request()
    request["content"]["messages"] = [  # type: ignore[index]
        {"role": "tool", "tool_call_id": "missing", "content": "PRIVATE_CANARY"}
    ]
    next_call = SimpleNamespace(call=AsyncMock())

    with pytest.raises(execution_policy._LlmPolicyError, match="rejected tool results") as captured:
        await unary("openai.chat_completions", request, next_call)
    assert "PRIVATE_CANARY" not in str(captured.value)
    next_call.call.assert_not_awaited()


async def test_deep_gemini_tool_schema_is_rejected_before_provider() -> None:
    _, _, unary, _ = await _registered_callbacks()
    request, _ = gemini_generate_content_case()
    schema: dict[str, object] = {"type": "STRING"}
    for _ in range(1_200):
        schema = {"type": "ARRAY", "items": schema}
    declaration = request["content"]["tools"][0]["functionDeclarations"][0]  # type: ignore[index]
    declaration["parameters"] = schema  # type: ignore[index]
    next_call = SimpleNamespace(call=AsyncMock())

    with pytest.raises(execution_policy._LlmPolicyError, match="cannot verify tool traffic") as captured:
        await unary("gemini.generate_content", request, next_call)

    assert str(captured.value) == "NeMo Guardrails cannot verify tool traffic"
    next_call.call.assert_not_awaited()


@pytest.mark.parametrize(
    "name, arguments, message",
    [
        ("delete_everything", "{}", "rejected model tool calls"),
        ("weather", "{}", "rejected model tool calls"),
        ("weather", "not-json", "cannot verify model tool calls"),
    ],
)
async def test_unsafe_model_call_is_held_and_not_returned(
    name: str,
    arguments: str,
    message: str,
) -> None:
    _, _, unary, _ = await _registered_callbacks()
    request = _chat_request()
    response = _chat_tool_response(name=name, arguments=arguments)
    next_call = SimpleNamespace(call=AsyncMock(return_value=response))

    with pytest.raises(execution_policy._LlmPolicyError, match=message) as captured:
        await unary("openai.chat_completions", request, next_call)

    assert "delete_everything" not in str(captured.value)
    assert "not-json" not in str(captured.value)
    next_call.call.assert_awaited_once_with(request)


async def test_structural_only_policy_does_not_log_tool_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    plugin, _, unary, _ = await _registered_callbacks()
    request = _chat_request()
    response = _chat_tool_response(name="PRIVATE_TOOL_CANARY", arguments='{"secret":"PRIVATE_ARG_CANARY"}')
    next_call = SimpleNamespace(call=AsyncMock(return_value=response))

    with pytest.raises(execution_policy._LlmPolicyError):
        await unary("openai.chat_completions", request, next_call)
    await plugin.close()

    assert "PRIVATE_TOOL_CANARY" not in caplog.text
    assert "PRIVATE_ARG_CANARY" not in caplog.text


async def test_tool_output_guarded_stream_is_rejected_before_opening_provider() -> None:
    _, _, _, stream = await _registered_callbacks()
    request = _chat_request()
    next_call = SimpleNamespace(call=MagicMock())

    with pytest.raises(execution_policy._LlmPolicyError, match="guarded streaming response"):
        await stream("openai.chat_completions", request, next_call)

    next_call.call.assert_not_called()


async def test_tool_result_only_stream_is_checked_once_and_forwarded(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "tool-result-only"
    config_path.mkdir()
    (config_path / "config.yml").write_text(
        "rails:\n  tool_input:\n    flows:\n      - tool result validation\n",
        encoding="utf-8",
    )
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, {"config_path": str(config_path)})
    stream_callback = registered_llm_execution(context, stream=True).callback
    stream = object()
    next_call = SimpleNamespace(call=MagicMock(return_value=stream))
    request = _chat_request(with_result=True)

    assert await stream_callback("fixture", request, next_call) is stream
    next_call.call.assert_called_once_with(request)
    context.register_llm_conditional_execution_guardrail.assert_not_called()


async def test_tool_result_only_stream_rejects_before_opening_provider(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "tool-result-only"
    config_path.mkdir()
    (config_path / "config.yml").write_text(
        "rails:\n  tool_input:\n    flows:\n      - tool result validation\n",
        encoding="utf-8",
    )
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, {"config_path": str(config_path)})
    stream_callback = registered_llm_execution(context, stream=True).callback
    next_call = SimpleNamespace(call=MagicMock())
    request = _chat_request()
    request["content"]["messages"] = [  # type: ignore[index]
        {"role": "tool", "tool_call_id": "missing", "content": "PRIVATE_CANARY"}
    ]

    with pytest.raises(execution_policy._LlmPolicyError, match="rejected tool results") as captured:
        await stream_callback("fixture", request, next_call)

    assert "PRIVATE_CANARY" not in str(captured.value)
    next_call.call.assert_not_called()


def test_unknown_custom_tool_flows_require_explicit_ignored_family_acknowledgement(tmp_path: Path) -> None:
    config_path = tmp_path / "custom-tool-flow"
    config_path.mkdir()
    (config_path / "config.yml").write_text(
        "rails:\n"
        "  input:\n"
        "    flows:\n"
        "      - input rail\n"
        "  tool_output:\n"
        "    flows:\n"
        "      - custom tool output rail\n",
        encoding="utf-8",
    )
    (config_path / "rails.co").write_text(
        "define flow input rail\n  bot refuse to respond\n  stop\n",
        encoding="utf-8",
    )

    rejected = configuration._validate_config({"config_path": str(config_path)})
    acknowledged = configuration._validate_config(
        {
            "config_path": str(config_path),
            "allow_ignored_rail_families": True,
        }
    )

    ignored = next(item for item in rejected if item.code.endswith("ignored_rail_families"))
    assert ignored.level is DiagnosticLevel.ERROR
    assert "tool_output" in ignored.message
    assert any(item.level is DiagnosticLevel.WARNING for item in acknowledged)


def test_duplicate_structural_tool_flow_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "duplicate-tool-flow"
    config_path.mkdir()
    (config_path / "config.yml").write_text(
        "rails:\n  tool_output:\n    flows:\n      - tool call validation\n      - tool call validation $model=main\n",
        encoding="utf-8",
    )

    diagnostics = configuration._validate_config({"config_path": str(config_path)})

    assert any(item.code.endswith("duplicate_structural_tool_rails") for item in diagnostics)
