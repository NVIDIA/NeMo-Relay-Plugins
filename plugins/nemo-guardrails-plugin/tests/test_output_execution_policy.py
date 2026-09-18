# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_relay_plugin import PluginContext
from nemoguardrails.rails.llm.options import RailStatus, RailType
from provider_cases import (
    anthropic_request,
    chat_request,
    chat_response,
    combined_protocol_case,
)
from registration_helpers import registered_llm_execution

from nemoguardrails_nemo_relay import (
    codec_projection,
    configuration,
    execution_policy,
    payload_policy,
    text_mutation,
    worker,
)
from nemoguardrails_nemo_relay.structural_tools import (
    StructuralToolSettings,
    ToolVerdict,
    ToolVerdictKind,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_OUTPUT_CONFIG = PROJECT_ROOT / "examples" / "input-output-rails"


def _context() -> MagicMock:
    context = MagicMock(spec=PluginContext)
    context.runtime = SimpleNamespace(list_runtime_registrations=AsyncMock(return_value=[]))
    return context


def test_streaming_output_execution_mode_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "unsupported-streaming-output-rails"
    config.mkdir()
    (config / "config.yml").write_text(
        "rails:\n  output:\n    flows:\n      - output rail\n    streaming:\n      enabled: true\n",
        encoding="utf-8",
    )
    (config / "rails.co").write_text(
        'define flow output rail\n  if $bot_message == "blocked"\n    bot refuse to respond\n    stop\n',
        encoding="utf-8",
    )

    diagnostics = configuration._validate_config({"config_path": str(config)})

    assert any(item.code.endswith("unsupported_streaming_output_rails") for item in diagnostics)


async def test_real_output_rails_hold_blocked_and_modified_native_responses() -> None:
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(
        context,
        {"config_path": str(INPUT_OUTPUT_CONFIG), "check_timeout_ms": 1_000},
    )
    callback = registered_llm_execution(context).callback
    request = chat_request()

    passed_response = chat_response("safe")
    passed_next = SimpleNamespace(call=AsyncMock(return_value=passed_response))
    assert await callback("fixture", request, passed_next) is passed_response

    for text, expected in [
        ("block output", "prevented execution in output rail 'output rail'"),
        ("modify output", "modified the output"),
    ]:
        next_call = SimpleNamespace(call=AsyncMock(return_value=chat_response(text)))
        with pytest.raises(execution_policy._LlmPolicyError, match=expected):
            await callback("fixture", request, next_call)
        next_call.call.assert_awaited_once_with(request)

    blocked_input_next = SimpleNamespace(call=AsyncMock())
    with pytest.raises(execution_policy._LlmPolicyError, match="input rail"):
        await callback(
            "fixture",
            chat_request([{"role": "user", "content": "block input"}]),
            blocked_input_next,
        )
    blocked_input_next.call.assert_not_awaited()
    context.register_llm_conditional_execution_guardrail.assert_not_called()
    await plugin.close()


async def test_count_tokens_runs_input_rails_but_skips_response_rails() -> None:
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(
        context,
        {"config_path": str(INPUT_OUTPUT_CONFIG), "check_timeout_ms": 1_000},
    )
    callback = registered_llm_execution(context).callback
    request = anthropic_request()
    request["content"].pop("max_tokens")  # type: ignore[union-attr]
    response = {"input_tokens": 4}
    next_call = SimpleNamespace(call=AsyncMock(return_value=response))

    assert await callback("anthropic.count_tokens", request, next_call) is response
    next_call.call.assert_awaited_once_with(request)

    malformed_next = SimpleNamespace(call=AsyncMock(return_value={"input_tokens": 4, "output": "PRIVATE_CANARY"}))
    with pytest.raises(execution_policy._LlmPolicyError, match="cannot verify the count-tokens response") as captured:
        await callback("anthropic.count_tokens", request, malformed_next)
    assert "PRIVATE_CANARY" not in str(captured.value)
    malformed_next.call.assert_awaited_once_with(request)

    blocked_request = deepcopy(request)
    blocked_request["content"]["messages"] = [  # type: ignore[index]
        {"role": "user", "content": "block input"}
    ]
    blocked_next = SimpleNamespace(call=AsyncMock())
    with pytest.raises(execution_policy._LlmPolicyError, match="input rail"):
        await callback("anthropic.count_tokens", blocked_request, blocked_next)
    blocked_next.call.assert_not_awaited()
    await plugin.close()


async def test_output_only_accepts_system_only_context_and_rejects_stream_before_provider(tmp_path: Path) -> None:
    config = tmp_path / "output-only"
    config.mkdir()
    (config / "config.yml").write_text(
        "rails:\n  output:\n    flows:\n      - output rail\n",
        encoding="utf-8",
    )
    (config / "rails.co").write_text(
        'define flow output rail\n  if $bot_message == "blocked"\n    bot refuse to respond\n    stop\n',
        encoding="utf-8",
    )
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, {"config_path": str(config), "check_timeout_ms": 1_000})
    unary = registered_llm_execution(context).callback
    stream = registered_llm_execution(context, stream=True).callback
    request = chat_request([{"role": "system", "content": "policy"}])
    response = chat_response("safe")

    assert await unary("fixture", request, SimpleNamespace(call=AsyncMock(return_value=response))) is response
    stream_next = SimpleNamespace(call=MagicMock())
    with pytest.raises(execution_policy._LlmPolicyError, match="guarded streaming response"):
        await stream("fixture", request, stream_next)
    stream_next.call.assert_not_called()
    await plugin.close()


async def test_output_guarded_stream_rejects_before_running_input_rails() -> None:
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(
        context,
        {"config_path": str(INPUT_OUTPUT_CONFIG), "check_timeout_ms": 1_000},
    )
    stream = registered_llm_execution(context, stream=True).callback
    next_call = SimpleNamespace(call=MagicMock())

    # This request is blocked by the input rail on unary calls. Streaming is
    # categorically unsupported when output rails are enabled, so no rail or
    # provider should run first.
    with pytest.raises(execution_policy._LlmPolicyError, match="guarded streaming response"):
        await stream(
            "fixture",
            chat_request([{"role": "user", "content": "block input"}]),
            next_call,
        )

    next_call.call.assert_not_called()
    await plugin.close()


class _EchoRails:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.explanation = SimpleNamespace(llm_calls=[], colang_history=None)

    async def check_async(self, messages: list[dict[str, str]], *, rail_types: list[RailType]) -> object:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            role = "user" if rail_types == [RailType.INPUT] else "assistant"
            content = next(message["content"] for message in reversed(messages) if message["role"] == role)
            return SimpleNamespace(status=RailStatus.PASSED, rail=None, content=content)
        finally:
            self.active -= 1

    def explain(self) -> object:
        return self.explanation


async def test_output_message_limit_reserves_space_before_calling_the_provider() -> None:
    rails = _EchoRails()
    text_policy = execution_policy._TextRailsPolicy(rails, 1_000)
    policy = execution_policy._LlmExecutionPolicy(
        codec_projection._NativeCodecProjector(),
        text_policy,
        input_enabled=False,
        output_enabled=True,
        adapter=None,
        settings=StructuralToolSettings(
            check_calls=False,
            check_results=False,
            ignored_families=(),
            duplicate_families=(),
        ),
    )
    within_limit = chat_request([{"role": "system", "content": f"context {index}"} for index in range(511)])
    at_limit = deepcopy(within_limit)
    at_limit["content"]["messages"].append({"role": "system", "content": "context 511"})  # type: ignore[index,union-attr]
    response = chat_response("safe")
    accepted_next = SimpleNamespace(call=AsyncMock(return_value=response))
    rejected_next = SimpleNamespace(call=AsyncMock(return_value=response))

    assert await policy.execute("fixture", within_limit, accepted_next) is response
    accepted_next.call.assert_awaited_once_with(within_limit)
    with pytest.raises(execution_policy._LlmPolicyError, match="oversized output context"):
        await policy.execute("fixture", at_limit, rejected_next)
    rejected_next.call.assert_not_awaited()


def _codec_execution_context(request_codec: str | None, response_codec: str | None) -> SimpleNamespace:
    def direction(codec_name: str | None) -> SimpleNamespace:
        kind = "builtin" if codec_name is not None else "none"
        return SimpleNamespace(codec=SimpleNamespace(kind=kind, id=codec_name))

    return SimpleNamespace(
        request=direction(request_codec),
        response=direction(response_codec),
        annotated_request=None,
    )


async def test_authoritative_context_does_not_guess_an_absent_request_codec() -> None:
    projector = codec_projection._NativeCodecProjector()
    execution = execution_policy._LlmExecutionPolicy(
        projector,
        execution_policy._TextRailsPolicy(_EchoRails(), 1_000, projector=projector),
        input_enabled=True,
        output_enabled=False,
        adapter=None,
        settings=StructuralToolSettings(False, False, (), ()),
    )
    next_call = SimpleNamespace(call=AsyncMock(return_value=chat_response("safe")))

    with pytest.raises(execution_policy._LlmPolicyError, match="request codec is unavailable"):
        await execution.execute_with_context(
            "gateway.generate",
            chat_request(),
            _codec_execution_context(None, None),
            next_call,
        )

    next_call.call.assert_not_awaited()


async def test_authoritative_context_requires_a_response_codec_for_output_rails() -> None:
    projector = codec_projection._NativeCodecProjector()
    execution = execution_policy._LlmExecutionPolicy(
        projector,
        execution_policy._TextRailsPolicy(_EchoRails(), 1_000, projector=projector),
        input_enabled=False,
        output_enabled=True,
        adapter=None,
        settings=StructuralToolSettings(False, False, (), ()),
    )
    next_call = SimpleNamespace(call=AsyncMock(return_value=chat_response("safe")))

    with pytest.raises(execution_policy._LlmPolicyError, match="response codec is unavailable"):
        await execution.execute_with_context(
            "gateway.generate",
            chat_request(),
            _codec_execution_context("openai_chat", None),
            next_call,
        )

    next_call.call.assert_not_awaited()


async def test_execution_context_uses_independent_request_and_response_codecs_for_output_policy() -> None:
    class DecisionRails(_EchoRails):
        async def check_async(self, messages: list[dict[str, str]], *, rail_types: list[RailType]) -> object:
            role = "user" if rail_types == [RailType.INPUT] else "assistant"
            content = next(message["content"] for message in reversed(messages) if message["role"] == role)
            if rail_types == [RailType.OUTPUT] and content == "blocked":
                return SimpleNamespace(status=RailStatus.BLOCKED, rail="output rail", content=content)
            if rail_types == [RailType.OUTPUT] and content == "modified":
                return SimpleNamespace(status=RailStatus.MODIFIED, rail="output rail", content="rewritten")
            return SimpleNamespace(status=RailStatus.PASSED, rail=None, content=content)

    projector = codec_projection._NativeCodecProjector()
    execution = execution_policy._LlmExecutionPolicy(
        projector,
        execution_policy._TextRailsPolicy(DecisionRails(), 1_000, projector=projector),
        input_enabled=True,
        output_enabled=True,
        adapter=None,
        settings=StructuralToolSettings(False, False, (), ()),
        mutation_policy=text_mutation.MutationPolicy(output=text_mutation.MutationMode.APPLY),
    )
    context = _codec_execution_context("openai_chat", "anthropic_messages")

    def anthropic_response(text: str) -> dict[str, object]:
        return {
            "id": "msg-fixture",
            "type": "message",
            "role": "assistant",
            "model": "fixture",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    passed = anthropic_response("safe")
    assert (
        await execution.execute_with_context(
            "gateway.generate",
            chat_request(),
            context,
            SimpleNamespace(call=AsyncMock(return_value=passed)),
        )
        is passed
    )

    with pytest.raises(execution_policy._LlmPolicyError, match="output rail"):
        await execution.execute_with_context(
            "gateway.generate",
            chat_request(),
            context,
            SimpleNamespace(call=AsyncMock(return_value=anthropic_response("blocked"))),
        )

    rewritten = await execution.execute_with_context(
        "gateway.generate",
        chat_request(),
        context,
        SimpleNamespace(call=AsyncMock(return_value=anthropic_response("modified"))),
    )
    assert rewritten["content"][0]["text"] == "rewritten"


async def test_execution_context_uses_independent_response_codec_for_structural_tools() -> None:
    request, _ = combined_protocol_case("openai_chat")
    anthropic_response = {
        "id": "msg-fixture",
        "type": "message",
        "role": "assistant",
        "model": "fixture",
        "content": [
            {"type": "text", "text": "safe"},
            {
                "type": "tool_use",
                "id": "call-1",
                "name": "weather",
                "input": {"city": "Paris"},
            },
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    adapter = SimpleNamespace(
        check_call_candidates=AsyncMock(return_value=ToolVerdict(ToolVerdictKind.PASSED)),
        close=AsyncMock(),
    )
    execution = execution_policy._LlmExecutionPolicy(
        codec_projection._NativeCodecProjector(),
        None,
        input_enabled=False,
        output_enabled=False,
        adapter=adapter,
        settings=StructuralToolSettings(True, False, (), ()),
    )

    assert (
        await execution.execute_with_context(
            "gateway.generate",
            request,
            _codec_execution_context("openai_chat", "anthropic_messages"),
            SimpleNamespace(call=AsyncMock(return_value=anthropic_response)),
        )
        is anthropic_response
    )
    adapter.check_call_candidates.assert_awaited_once()


async def test_execution_policy_checks_reasoning_before_final_answer() -> None:
    class RecordingRails(_EchoRails):
        def __init__(self) -> None:
            super().__init__()
            self.outputs: list[str] = []

        async def check_async(self, messages: list[dict[str, str]], *, rail_types: list[RailType]) -> object:
            assert rail_types == [RailType.OUTPUT]
            content = next(message["content"] for message in reversed(messages) if message["role"] == "assistant")
            self.outputs.append(content)
            return SimpleNamespace(status=RailStatus.PASSED, rail=None, content=content)

    rails = RecordingRails()
    projector = codec_projection._NativeCodecProjector(
        payload_policy.PayloadPolicy(reasoning=payload_policy.ReasoningPolicy.CHECK_OUTPUT)
    )
    text_policy = execution_policy._TextRailsPolicy(rails, 1_000, projector=projector)
    execution = execution_policy._LlmExecutionPolicy(
        projector,
        text_policy,
        input_enabled=False,
        output_enabled=True,
        adapter=None,
        settings=StructuralToolSettings(False, False, (), ()),
    )
    response = chat_response("safe")
    response["choices"][0]["message"]["reasoning_content"] = "private reasoning"  # type: ignore[index]

    assert (
        await execution.execute(
            "openai.chat_completions",
            chat_request(),
            SimpleNamespace(call=AsyncMock(return_value=response)),
        )
        is response
    )
    assert rails.outputs == ["private reasoning", "safe"]


async def test_execution_policy_bounds_the_number_of_output_checks_before_evaluation() -> None:
    class RecordingRails(_EchoRails):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def check_async(self, messages: list[dict[str, str]], *, rail_types: list[RailType]) -> object:
            self.calls += 1
            return await super().check_async(messages, rail_types=rail_types)

    rails = RecordingRails()
    projector = codec_projection._NativeCodecProjector()
    execution = execution_policy._LlmExecutionPolicy(
        projector,
        execution_policy._TextRailsPolicy(rails, 1_000, projector=projector),
        input_enabled=False,
        output_enabled=True,
        adapter=None,
        settings=StructuralToolSettings(False, False, (), ()),
    )
    response = chat_response("safe")
    first_choice = response["choices"][0]  # type: ignore[index]
    response["choices"] = [deepcopy(first_choice) for _ in range(execution_policy.MAX_OUTPUT_CHECK_SEGMENTS + 1)]

    with pytest.raises(execution_policy._LlmPolicyError, match="cannot verify provider output"):
        await execution.execute(
            "openai.chat_completions",
            chat_request(),
            SimpleNamespace(call=AsyncMock(return_value=response)),
        )

    assert rails.calls == 0


async def test_execution_policy_bounds_aggregate_repeated_output_context_work() -> None:
    rails = _EchoRails()
    projector = codec_projection._NativeCodecProjector()
    execution = execution_policy._LlmExecutionPolicy(
        projector,
        execution_policy._TextRailsPolicy(rails, 1_000, projector=projector),
        input_enabled=False,
        output_enabled=True,
        adapter=None,
        settings=StructuralToolSettings(False, False, (), ()),
    )
    request = chat_request([{"role": "user", "content": "x" * 600_000}])
    response = chat_response("first")
    response["choices"].append(chat_response("second")["choices"][0])  # type: ignore[index,union-attr]

    with pytest.raises(execution_policy._LlmPolicyError, match="oversized output workload"):
        await execution.execute(
            "openai.chat_completions",
            request,
            SimpleNamespace(call=AsyncMock(return_value=response)),
        )

    assert rails.max_active == 0


async def test_execution_policy_applies_one_deadline_to_all_output_segments() -> None:
    class SlowRails(_EchoRails):
        async def check_async(self, messages: list[dict[str, str]], *, rail_types: list[RailType]) -> object:
            await asyncio.sleep(0.06)
            return await super().check_async(messages, rail_types=rail_types)

    projector = codec_projection._NativeCodecProjector()
    execution = execution_policy._LlmExecutionPolicy(
        projector,
        execution_policy._TextRailsPolicy(SlowRails(), 80, projector=projector),
        input_enabled=False,
        output_enabled=True,
        adapter=None,
        settings=StructuralToolSettings(False, False, (), ()),
    )
    response = chat_response("first")
    response["choices"].append(chat_response("second")["choices"][0])  # type: ignore[index,union-attr]

    with pytest.raises(execution_policy._LlmPolicyError, match="output checking timed out"):
        await execution.execute(
            "openai.chat_completions",
            chat_request(),
            SimpleNamespace(call=AsyncMock(return_value=response)),
        )


async def test_input_and_output_checks_share_one_engine_permit() -> None:
    rails = _EchoRails()
    policy = execution_policy._TextRailsPolicy(rails, 1_000)
    projected = codec_projection._DecodedTextRequest(
        ("openai_chat",),
        ({"role": "user", "content": "hello"},),
    )

    assert await asyncio.gather(
        policy.check_input(projected),
        policy.check_output(projected, "safe"),
    ) == [None, None]
    assert rails.max_active == 1
    assert policy._pending == 0


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (SimpleNamespace(status=RailStatus.PASSED, rail=None, content="different"), "inconsistent output-check"),
        (SimpleNamespace(status=RailStatus.BLOCKED, rail="line\nbreak", content="private"), "during output checking"),
        (SimpleNamespace(status=RailStatus.MODIFIED, rail=None, content="private"), "modified the output"),
        (SimpleNamespace(status="future", rail=None, content="private"), "output check failed"),
        (RuntimeError("PRIVATE_OUTPUT_CANARY"), "output check failed"),
    ],
)
async def test_output_outcomes_fail_closed_without_content_leaks(result: object, expected: str) -> None:
    class ResultRails(_EchoRails):
        async def check_async(self, messages: list[dict[str, str]], *, rail_types: list[RailType]) -> object:
            del messages
            assert rail_types == [RailType.OUTPUT]
            if isinstance(result, BaseException):
                raise result
            return result

    policy = execution_policy._TextRailsPolicy(ResultRails(), 1_000)
    projected = codec_projection._DecodedTextRequest(("openai_chat",), ({"role": "user", "content": "hello"},))

    reason = await policy.check_output(projected, "private output")

    assert reason is not None
    assert expected in reason
    assert "PRIVATE" not in reason


@pytest.mark.parametrize(
    "protocol",
    [
        "openai_chat",
        "openai_responses",
        "anthropic_messages",
        "gemini_generate_content",
        "oci_genai",
        "oci_cohere",
    ],
)
async def test_mixed_text_and_function_call_runs_output_then_structural_checks(
    tmp_path: Path,
    protocol: str,
) -> None:
    config = tmp_path / f"combined-{protocol}"
    config.mkdir()
    (config / "config.yml").write_text(
        "rails:\n"
        "  output:\n"
        "    flows:\n"
        "      - output rail\n"
        "  tool_output:\n"
        "    flows:\n"
        "      - tool call validation\n",
        encoding="utf-8",
    )
    (config / "rails.co").write_text(
        'define flow output rail\n  if $bot_message == "block output"\n    bot refuse to respond\n    stop\n',
        encoding="utf-8",
    )
    context = _context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, {"config_path": str(config), "check_timeout_ms": 1_000})
    callback = registered_llm_execution(context).callback
    try:
        request, passed = combined_protocol_case(protocol)
        assert await callback(protocol, request, SimpleNamespace(call=AsyncMock(return_value=passed))) is passed

        request, blocked_output = combined_protocol_case(
            protocol,
            text="block output",
            call_name="delete_everything",
        )
        with pytest.raises(execution_policy._LlmPolicyError, match="output rail"):
            await callback(protocol, request, SimpleNamespace(call=AsyncMock(return_value=blocked_output)))

        request, unsafe_call = combined_protocol_case(protocol, call_name="delete_everything")
        with pytest.raises(execution_policy._LlmPolicyError, match="rejected model tool calls"):
            await callback(protocol, request, SimpleNamespace(call=AsyncMock(return_value=unsafe_call)))
    finally:
        await plugin.close()
