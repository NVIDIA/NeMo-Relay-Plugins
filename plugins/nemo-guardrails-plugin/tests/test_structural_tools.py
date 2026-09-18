# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemoguardrails_nemo_relay.structural_tools import (
    Guardrails024StructuralToolAdapter,
    ToolCall,
    ToolDefinition,
    ToolExchange,
    ToolResult,
    ToolVerdict,
    ToolVerdictKind,
)

WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


_ADAPTERS: list[Guardrails024StructuralToolAdapter] = []


def _adapter(timeout_ms: int = 25_000) -> Guardrails024StructuralToolAdapter:
    adapter = Guardrails024StructuralToolAdapter(timeout_ms=timeout_ms)
    _ADAPTERS.append(adapter)
    return adapter


def _rail_result(*, is_safe: bool, failed: bool = False, reason: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(is_safe=is_safe, failed=failed, reason=reason)


@pytest.fixture(autouse=True)
async def _close_adapters() -> None:
    first = len(_ADAPTERS)
    yield
    adapters = _ADAPTERS[first:]
    del _ADAPTERS[first:]
    for adapter in adapters:
        await adapter.close()


def _weather_call(**arguments: object) -> ToolCall:
    return ToolCall(call_id="call-1", name="weather", arguments=arguments)


async def test_compatibility_probe_passes_against_pinned_guardrails() -> None:
    await _adapter().verify_compatibility()


async def test_compatibility_probe_uses_one_activation_timeout() -> None:
    adapter = _adapter(timeout_ms=10)
    release = threading.Event()

    async def hanging_check(*_args: object) -> SimpleNamespace:
        release.wait(timeout=2)
        return _rail_result(is_safe=True)

    adapter._call_check = hanging_check

    with pytest.raises(RuntimeError, match="timed out"):
        await adapter.verify_compatibility()

    release.set()
    for _ in range(100):
        if adapter._pending == 0:
            break
        await asyncio.sleep(0.01)
    assert adapter._pending == 0


async def test_compatibility_probe_exercises_both_public_manager_checks() -> None:
    adapter = _adapter()
    adapter._call_check = AsyncMock(
        side_effect=[
            _rail_result(is_safe=True),
            _rail_result(is_safe=False),
        ]
    )
    adapter._result_check = AsyncMock(return_value=_rail_result(is_safe=True))

    await adapter.verify_compatibility()

    assert adapter._call_check.await_count == 2
    adapter._result_check.assert_awaited_once()


async def test_close_stops_the_public_manager_once() -> None:
    adapter = _adapter()
    adapter._manager.stop = AsyncMock()

    await adapter.close()
    await adapter.close()

    adapter._manager.stop.assert_awaited_once()


async def test_declared_call_with_valid_arguments_passes() -> None:
    verdict = await _adapter().check_calls(
        [ToolDefinition(name="weather", input_schema=WEATHER_SCHEMA)],
        [_weather_call(city="Paris")],
    )

    assert verdict.kind is ToolVerdictKind.PASSED


@pytest.mark.parametrize(
    "definitions, calls",
    [
        ([], [_weather_call(city="Paris")]),
        ([ToolDefinition(name="weather", input_schema=WEATHER_SCHEMA)], [_weather_call()]),
        (
            [ToolDefinition(name="weather", input_schema=WEATHER_SCHEMA)],
            [_weather_call(city=7)],
        ),
        (
            [ToolDefinition(name="ping")],
            [ToolCall(call_id="call-1", name="ping", arguments={"unexpected": True})],
        ),
        (
            [ToolDefinition(name="weather", input_schema={"type": "not-a-json-schema-type"})],
            [_weather_call(city="Paris")],
        ),
    ],
)
async def test_guardrails_policy_blocks_invalid_calls(
    definitions: list[ToolDefinition],
    calls: list[ToolCall],
) -> None:
    verdict = await _adapter().check_calls(definitions, calls)

    assert verdict.kind is ToolVerdictKind.POLICY_BLOCK


@pytest.mark.parametrize(
    "call",
    [
        ToolCall(call_id="", name="weather", arguments={}),
        ToolCall(call_id="call-1", name=None, arguments={}),
        ToolCall(call_id="call-1", name="weather", arguments={}, type="hosted_search"),
        ToolCall(call_id="call-1", name="weather", arguments=[]),  # type: ignore[arg-type]
    ],
)
async def test_incomplete_call_projection_fails_closed(call: ToolCall) -> None:
    verdict = await _adapter().check_calls([ToolDefinition(name="weather")], [call])

    assert verdict.kind is ToolVerdictKind.COVERAGE_BLOCK


async def test_duplicate_call_ids_fail_closed_before_guardrails() -> None:
    verdict = await _adapter().check_calls(
        [ToolDefinition(name="weather")],
        [_weather_call(), _weather_call()],
    )

    assert verdict.kind is ToolVerdictKind.COVERAGE_BLOCK


async def test_duplicate_or_non_function_definitions_fail_closed_before_guardrails() -> None:
    duplicate = await _adapter().check_calls(
        [ToolDefinition(name="weather"), ToolDefinition(name="weather")],
        [_weather_call()],
    )
    hosted = await _adapter().check_calls(
        [ToolDefinition(name=None, type="web_search")],
        [ToolCall(call_id="call-1", name="web_search", arguments={}, type="web_search")],
    )

    assert duplicate.kind is ToolVerdictKind.COVERAGE_BLOCK
    assert hosted.kind is ToolVerdictKind.COVERAGE_BLOCK


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "https://example.invalid/tool-schema.json"},
        {"$dynamicRef": "tool-schema.json#definition"},
        {"type": "string", "pattern": "^(a+)+$"},
        {"type": "object", "patternProperties": {"^(a+)+$": {"type": "string"}}},
    ],
)
async def test_network_and_unbounded_regex_schemas_fail_before_jsonschema(schema: dict[str, object]) -> None:
    verdict = await _adapter().check_calls(
        [ToolDefinition(name="weather", input_schema=schema)],
        [_weather_call(city="Paris")],
    )

    assert verdict.kind is ToolVerdictKind.COVERAGE_BLOCK


async def test_local_json_schema_references_fail_closed_to_avoid_cycles() -> None:
    verdict = await _adapter().check_calls(
        [
            ToolDefinition(
                name="weather",
                input_schema={
                    "$defs": {"city": {"type": "string"}},
                    "type": "object",
                    "properties": {"city": {"$ref": "#/$defs/city"}},
                    "required": ["city"],
                },
            )
        ],
        [_weather_call(city="Paris")],
    )

    assert verdict.kind is ToolVerdictKind.COVERAGE_BLOCK


@pytest.mark.parametrize(
    "legacy_child",
    [
        {"properties": {"x": {"pattern": "^(a+)+$"}}},
        {"properties": {"x": {"$ref": "https://example.invalid/schema"}}},
    ],
)
async def test_legacy_draft_subschemas_cannot_bypass_schema_safety(legacy_child: dict[str, object]) -> None:
    verdict = await _adapter().check_calls(
        [
            ToolDefinition(
                name="weather",
                input_schema={
                    "$schema": "http://json-schema.org/draft-03/schema#",
                    "extends": legacy_child,
                },
            )
        ],
        [_weather_call(x="aaaaaaaaaaaaaaaaaaaa!")],
    )

    assert verdict.kind is ToolVerdictKind.COVERAGE_BLOCK


async def test_schema_property_names_that_match_keywords_remain_supported() -> None:
    verdict = await _adapter().check_calls(
        [
            ToolDefinition(
                name="weather",
                input_schema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "patternProperties": {"type": "string"},
                    },
                },
            )
        ],
        [_weather_call(pattern="plain", patternProperties="plain")],
    )

    assert verdict.kind is ToolVerdictKind.PASSED


async def test_quadratic_unique_items_schema_fails_before_jsonschema() -> None:
    verdict = await _adapter().check_calls(
        [
            ToolDefinition(
                name="weather",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cities": {"type": "array", "uniqueItems": True},
                    },
                },
            )
        ],
        [_weather_call(cities=[{"name": "Paris"}])],
    )

    assert verdict.kind is ToolVerdictKind.COVERAGE_BLOCK


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "enum": [str(index) for index in range(129)]},
        {"const": {"nested": "value"}},
        {"allOf": [{"type": "object"}], "unevaluatedProperties": True},
        {"prefixItems": [{"type": "string"}], "unevaluatedItems": True},
    ],
)
async def test_expensive_equality_schemas_fail_before_jsonschema(schema: dict[str, object]) -> None:
    verdict = await _adapter().check_calls(
        [ToolDefinition(name="weather", input_schema={"type": "object", "properties": {"city": schema}})],
        [_weather_call(city="Paris")],
    )

    assert verdict.kind is ToolVerdictKind.COVERAGE_BLOCK


async def test_bounded_scalar_enum_schema_remains_supported() -> None:
    verdict = await _adapter().check_calls(
        [
            ToolDefinition(
                name="weather",
                input_schema={
                    "type": "object",
                    "properties": {"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                    "required": ["unit"],
                },
            )
        ],
        [_weather_call(unit="celsius")],
    )

    assert verdict.kind is ToolVerdictKind.PASSED


async def test_aggregate_schema_work_is_rejected_before_guardrails() -> None:
    adapter = _adapter()
    check = AsyncMock(return_value=_rail_result(is_safe=True))
    adapter._call_check = check
    schema = {
        "type": "object",
        "properties": {f"field_{index}": {"type": "string"} for index in range(200)},
    }
    calls = [ToolCall(call_id=f"call-{index}", name="wide", arguments={}) for index in range(64)]

    verdict = await adapter.check_calls([ToolDefinition(name="wide", input_schema=schema)], calls)

    assert verdict.kind is ToolVerdictKind.COVERAGE_BLOCK
    check.assert_not_awaited()


async def test_boolean_combinator_work_is_counted_before_jsonschema() -> None:
    adapter = _adapter()
    check = AsyncMock(return_value=_rail_result(is_safe=True))
    adapter._call_check = check
    schema = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "contains": {"anyOf": [False] * 128},
            }
        },
    }

    verdict = await adapter.check_calls(
        [ToolDefinition(name="scan", input_schema=schema)],
        [ToolCall(call_id="call-1", name="scan", arguments={"values": [0] * 1_000})],
    )

    assert verdict.kind is ToolVerdictKind.COVERAGE_BLOCK
    check.assert_not_awaited()


async def test_dependent_schema_fanout_is_counted_before_jsonschema() -> None:
    adapter = _adapter()
    check = AsyncMock(return_value=_rail_result(is_safe=True))
    adapter._call_check = check
    schema = {
        "type": "object",
        "dependentSchemas": {f"field_{index}": {"propertyNames": {}} for index in range(128)},
    }
    arguments = {f"field_{index}": index for index in range(1_000)}

    verdict = await adapter.check_calls(
        [ToolDefinition(name="scan", input_schema=schema)],
        [ToolCall(call_id="call-1", name="scan", arguments=arguments)],
    )

    assert verdict.kind is ToolVerdictKind.COVERAGE_BLOCK
    check.assert_not_awaited()


async def test_deep_sequential_schema_work_is_counted_before_jsonschema() -> None:
    adapter = _adapter()
    check = AsyncMock(return_value=_rail_result(is_safe=True))
    adapter._call_check = check
    item_schema: dict[str, object] = {"type": "integer"}
    for _ in range(29):
        item_schema = {"not": {"not": item_schema}}
    schema = {
        "type": "object",
        "properties": {"values": {"type": "array", "items": item_schema}},
    }

    verdict = await adapter.check_calls(
        [ToolDefinition(name="scan", input_schema=schema)],
        [ToolCall(call_id="call-1", name="scan", arguments={"values": [0] * 1_000})],
    )

    assert verdict.kind is ToolVerdictKind.COVERAGE_BLOCK
    check.assert_not_awaited()


async def test_tool_history_has_one_aggregate_json_budget() -> None:
    adapter = _adapter()
    check = AsyncMock(return_value=_rail_result(is_safe=True))
    adapter._result_check = check
    exchanges = [
        ToolExchange(
            calls=(
                ToolCall(
                    call_id=f"call-{index}",
                    name="scan",
                    arguments={"values": [0] * 19_000},
                ),
            ),
            results=(),
        )
        for index in range(6)
    ]

    verdict = await adapter.check_results(exchanges)

    assert verdict.kind is ToolVerdictKind.COVERAGE_BLOCK
    check.assert_not_awaited()


async def test_tool_history_budget_includes_call_ids_and_names(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    check = AsyncMock(return_value=_rail_result(is_safe=True))
    adapter._result_check = check
    monkeypatch.setattr(
        "nemoguardrails_nemo_relay.structural_tools.MAX_TOOL_HISTORY_CHARACTERS",
        64,
    )
    exchange = ToolExchange(
        calls=(ToolCall(call_id="c" * 40, name="n" * 40, arguments={}),),
        results=(),
    )

    verdict = await adapter.check_results([exchange])

    assert verdict.kind is ToolVerdictKind.COVERAGE_BLOCK
    check.assert_not_awaited()


async def test_unsafe_schema_on_an_unused_function_does_not_block_a_called_tool() -> None:
    verdict = await _adapter().check_calls(
        [
            ToolDefinition(name="weather", input_schema=WEATHER_SCHEMA),
            ToolDefinition(name="unused", input_schema={"type": "string", "pattern": "^(a+)+$"}),
        ],
        [_weather_call(city="Paris")],
    )

    assert verdict.kind is ToolVerdictKind.PASSED


async def test_unused_unsupported_schema_does_not_block_a_text_response() -> None:
    verdict = await _adapter().check_calls(
        [ToolDefinition(name="weather", input_schema={"pattern": "^(a+)+$"})],
        [],
    )

    assert verdict.kind is ToolVerdictKind.PASSED


async def test_non_finite_call_arguments_fail_closed() -> None:
    verdict = await _adapter().check_calls(
        [ToolDefinition(name="weather")],
        [_weather_call(value=float("nan"))],
    )

    assert verdict.kind is ToolVerdictKind.COVERAGE_BLOCK


async def test_result_without_provider_name_uses_unique_call_link() -> None:
    verdict = await _adapter().check_results(
        [
            ToolExchange(
                calls=(_weather_call(city="Paris"),),
                results=(ToolResult(call_id="call-1", name=None, content="sunny"),),
            )
        ]
    )

    assert verdict.kind is ToolVerdictKind.PASSED


@pytest.mark.parametrize(
    "exchange",
    [
        ToolExchange(calls=(_weather_call(),), results=(ToolResult(call_id="missing", name=None, content="x"),)),
        ToolExchange(calls=(_weather_call(),), results=(ToolResult(call_id=None, name=None, content="x"),)),
        ToolExchange(
            calls=(_weather_call(),),
            results=(ToolResult(call_id="call-1", name="search", content="x"),),
        ),
        ToolExchange(
            calls=(_weather_call(),),
            results=(
                ToolResult(call_id="call-1", name=None, content="first"),
                ToolResult(call_id="call-1", name=None, content="second"),
            ),
        ),
        ToolExchange(
            calls=(_weather_call(), _weather_call()),
            results=(ToolResult(call_id="call-1", name=None, content="x"),),
        ),
        ToolExchange(
            calls=(_weather_call(),),
            results=(ToolResult(call_id="call-1", name=None, content=[1]),),  # type: ignore[list-item]
        ),
    ],
)
async def test_guardrails_policy_blocks_invalid_result_linkage(exchange: ToolExchange) -> None:
    verdict = await _adapter().check_results([exchange])

    assert verdict.kind is ToolVerdictKind.POLICY_BLOCK


@pytest.mark.parametrize("content", [None, "", [], [{"type": "text", "text": "ok"}]])
async def test_supported_result_content_shapes_pass(content: object) -> None:
    verdict = await _adapter().check_results(
        [
            ToolExchange(
                calls=(_weather_call(),),
                results=(ToolResult(call_id="call-1", name=None, content=content),),  # type: ignore[arg-type]
            )
        ]
    )

    assert verdict.kind is ToolVerdictKind.PASSED


async def test_reused_call_id_in_separate_turns_is_valid() -> None:
    exchange = ToolExchange(
        calls=(_weather_call(),),
        results=(ToolResult(call_id="call-1", name=None, content="ok"),),
    )

    verdict = await _adapter().check_results([exchange, exchange])

    assert verdict.kind is ToolVerdictKind.PASSED


@pytest.mark.parametrize(
    "result, expected",
    [
        (_rail_result(is_safe=False, failed=True, reason="PRIVATE_CANARY"), ToolVerdictKind.VALIDATOR_FAILURE),
        (_rail_result(is_safe=False, reason="PRIVATE_CANARY"), ToolVerdictKind.POLICY_BLOCK),
        (object(), ToolVerdictKind.VALIDATOR_FAILURE),
    ],
)
async def test_public_results_map_to_content_free_verdicts(result: object, expected: ToolVerdictKind) -> None:
    adapter = _adapter()
    adapter._call_check = AsyncMock(return_value=result)

    verdict = await adapter.check_calls([ToolDefinition(name="weather")], [_weather_call()])

    assert verdict.kind is expected
    assert "PRIVATE_CANARY" not in repr(verdict)


async def test_action_exception_maps_to_validator_failure() -> None:
    adapter = _adapter()
    adapter._call_check = AsyncMock(side_effect=RuntimeError("PRIVATE_CANARY"))

    verdict = await adapter.check_calls([ToolDefinition(name="weather")], [_weather_call()])

    assert verdict.kind is ToolVerdictKind.VALIDATOR_FAILURE
    assert "PRIVATE_CANARY" not in repr(verdict)


async def test_synchronous_guardrails_validation_runs_off_the_worker_event_loop() -> None:
    adapter = _adapter()
    event_loop_thread = threading.get_ident()
    action_threads: list[int] = []

    async def allow(*_args: object) -> SimpleNamespace:
        action_threads.append(threading.get_ident())
        return _rail_result(is_safe=True)

    adapter._call_check = allow

    verdict = await adapter.check_calls([ToolDefinition(name="weather")], [_weather_call()])

    assert verdict.kind is ToolVerdictKind.PASSED
    assert action_threads and action_threads[0] != event_loop_thread


async def test_timeout_keeps_capacity_owned_until_the_validator_thread_returns() -> None:
    adapter = _adapter(timeout_ms=10)
    started = threading.Event()
    release = threading.Event()

    async def slow_allow(*_args: object) -> SimpleNamespace:
        started.set()
        release.wait(timeout=2)
        return _rail_result(is_safe=True)

    adapter._call_check = slow_allow
    task = asyncio.create_task(adapter.check_calls([ToolDefinition(name="weather")], [_weather_call()]))
    assert await asyncio.to_thread(started.wait, 1)

    verdict = await task

    assert verdict.kind is ToolVerdictKind.VALIDATOR_FAILURE
    assert adapter._pending == 1
    release.set()
    for _ in range(100):
        if adapter._pending == 0:
            break
        await asyncio.sleep(0.01)
    assert adapter._pending == 0


async def test_call_candidates_share_one_request_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(timeout_ms=50)
    calls = 0
    second_started = asyncio.Event()
    never_finishes = asyncio.Event()

    async def check_calls(*_args: object) -> ToolVerdict:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ToolVerdict(ToolVerdictKind.PASSED)
        second_started.set()
        await never_finishes.wait()
        raise AssertionError("the outer request deadline did not cancel the second candidate")

    monkeypatch.setattr(adapter, "check_calls", check_calls)
    started = time.perf_counter()

    verdict = await adapter.check_call_candidates(
        [ToolDefinition(name="weather")],
        [(_weather_call(),), (_weather_call(),)],
    )
    elapsed = time.perf_counter() - started

    assert verdict.kind is ToolVerdictKind.VALIDATOR_FAILURE
    assert calls == 2
    assert second_started.is_set()
    assert elapsed < 1


async def test_result_timeout_is_one_budget_for_the_whole_request() -> None:
    adapter = _adapter(timeout_ms=50)

    async def slow_allow(*_args: object) -> SimpleNamespace:
        # RailsManager owns the per-exchange loop. Simulate two sequential
        # pieces of work inside that one public manager call and prove the
        # adapter applies one deadline to the complete request.
        await asyncio.sleep(0.04)
        await asyncio.sleep(0.04)
        return _rail_result(is_safe=True)

    adapter._result_check = slow_allow
    exchange = ToolExchange(
        calls=(_weather_call(),),
        results=(ToolResult(call_id="call-1", name=None, content="ok"),),
    )
    started = time.perf_counter()

    verdict = await adapter.check_results([exchange, exchange])
    elapsed = time.perf_counter() - started

    assert verdict.kind is ToolVerdictKind.VALIDATOR_FAILURE
    assert elapsed < 0.09
    for _ in range(100):
        if adapter._pending == 0:
            break
        await asyncio.sleep(0.01)
    assert adapter._pending == 0


async def test_cancellation_is_not_converted_to_a_policy_decision() -> None:
    adapter = _adapter()
    adapter._call_check = AsyncMock(side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await adapter.check_calls([ToolDefinition(name="weather")], [_weather_call()])
