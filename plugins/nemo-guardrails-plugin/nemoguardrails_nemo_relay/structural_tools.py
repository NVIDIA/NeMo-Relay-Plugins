# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned adapter for NeMo Guardrails 0.24 structural tool rails.

The worker keeps provider projection and resource limits in plugin-owned
types, then calls Guardrails 0.24's documented ``RailsManager`` tool-check
methods behind an exact-version compatibility adapter using
a local OpenAI-compatible adapter. No model request is made by these checks.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any, Sequence

SUPPORTED_GUARDRAILS_VERSION = "0.24.1"
MAX_PENDING_VALIDATIONS = 32
MAX_CONCURRENT_VALIDATIONS = 4
MAX_TOOL_DEFINITIONS = 256
MAX_MODEL_TOOL_CALLS = 128
MAX_TOOL_HISTORY_ITEMS = 2_048
MAX_TOOL_EXCHANGES = 256
MAX_TOOL_HISTORY_JSON_NODES = 100_000
MAX_TOOL_HISTORY_CHARACTERS = 2_000_000
MAX_JSON_NODES = 20_000
MAX_JSON_DEPTH = 64
MAX_JSON_CHARACTERS = 1_000_000
MAX_SCHEMA_NODES = 2_048
MAX_SCHEMA_ALTERNATIVES = 128
MAX_SCHEMA_ENUM_VALUES = 128
MAX_SCHEMA_VALIDATION_WORK = 20_000
_REFERENCE_KEYWORDS = frozenset({"$ref", "$dynamicRef", "$recursiveRef"})
_SINGLE_SUBSCHEMA_KEYS = frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SUBSCHEMA_ARRAY_KEYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_LEGACY_SUBSCHEMA_KEYS = frozenset({"disallow", "extends", "type"})
_SUBSCHEMA_MAP_KEYS = frozenset(
    {
        "$defs",
        "definitions",
        "properties",
    }
)
_VALIDATOR_EXECUTOR = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_VALIDATIONS,
    thread_name_prefix="nemo-guardrails-tools",
)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """One provider-neutral tool offered to the model."""

    name: str | None
    type: str = "function"
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    strict: bool | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One provider-neutral tool call emitted by a model."""

    call_id: str
    name: str | None
    arguments: dict[str, Any]
    type: str = "function"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One provider-neutral result returned for a prior tool call."""

    call_id: str | None
    name: str | None
    content: str | list[dict[str, Any]] | None
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ToolExchange:
    """Calls from one assistant turn and the results that answer them."""

    calls: tuple[ToolCall, ...]
    results: tuple[ToolResult, ...]


@dataclass(frozen=True, slots=True)
class StructuralToolSettings:
    """Structural tool checks selected by one Guardrails configuration."""

    check_calls: bool
    check_results: bool
    ignored_families: tuple[str, ...]
    duplicate_families: tuple[str, ...]

    @property
    def enabled(self) -> bool:
        return self.check_calls or self.check_results


class ToolVerdictKind(str, Enum):
    """Stable decisions returned by the plugin-owned adapter."""

    PASSED = "passed"
    POLICY_BLOCK = "policy_block"
    COVERAGE_BLOCK = "coverage_block"
    VALIDATOR_FAILURE = "validator_failure"


@dataclass(frozen=True, slots=True)
class ToolVerdict:
    """Content-free structural tool decision."""

    kind: ToolVerdictKind

    @property
    def passed(self) -> bool:
        return self.kind is ToolVerdictKind.PASSED


@dataclass(frozen=True, slots=True)
class _Guardrails024Symbols:
    rails_config: type[Any]
    engine_registry: type[Any]
    rails_manager: type[Any]
    task_manager: type[Any]
    tool_call: type[Any]
    tool_call_function: type[Any]


def _module(name: str) -> ModuleType:
    return importlib.import_module(name)


def _load_guardrails_024_symbols() -> _Guardrails024Symbols:
    """Load the exact-version API surface pinned by this adapter."""

    nemoguardrails = _module("nemoguardrails")
    if getattr(nemoguardrails, "__version__", None) != SUPPORTED_GUARDRAILS_VERSION:
        raise RuntimeError("unsupported NeMo Guardrails structural-tool API version")

    registry_module = _module("nemoguardrails.guardrails.engine_registry")
    manager_module = _module("nemoguardrails.guardrails.rails_manager")
    task_manager_module = _module("nemoguardrails.llm.taskmanager")
    types_module = _module("nemoguardrails.types")
    try:
        return _Guardrails024Symbols(
            rails_config=nemoguardrails.RailsConfig,
            engine_registry=registry_module.EngineRegistry,
            rails_manager=manager_module.RailsManager,
            task_manager=task_manager_module.LLMTaskManager,
            tool_call=types_module.ToolCall,
            tool_call_function=types_module.ToolCallFunction,
        )
    except AttributeError as exc:
        raise RuntimeError("NeMo Guardrails structural-tool API is incompatible") from exc


def _json_metrics(
    value: object,
    *,
    max_nodes: int = MAX_JSON_NODES,
    max_characters: int = MAX_JSON_CHARACTERS,
) -> tuple[int, int] | None:
    """Count a finite JSON value without recursively walking Python frames."""

    nodes = 0
    characters = 0
    stack = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes or depth > MAX_JSON_DEPTH:
            return None
        if isinstance(current, dict):
            if any(not isinstance(key, str) for key in current):
                return None
            characters += sum(len(key) for key in current)
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, str):
            characters += len(current)
        elif isinstance(current, float) and not math.isfinite(current):
            return None
        elif not isinstance(current, (str, int, float, bool, type(None))):
            return None
        if characters > max_characters:
            return None
    return nodes, characters


def _bounded_multiplier(value: int, factor: int) -> int:
    """Cap a conservative schema-work multiplier without large integers."""

    if value > MAX_SCHEMA_VALIDATION_WORK // max(factor, 1):
        return MAX_SCHEMA_VALIDATION_WORK + 1
    return value * max(factor, 1)


def _safe_schema(value: object) -> tuple[int, int] | None:
    """Return a bounded schema size and worst-case instance-work multiplier."""

    metrics = _json_metrics(value)
    if not isinstance(value, dict) or metrics is None:
        return None

    schema_nodes = 0
    alternatives = 0
    max_instance_work = 1
    stack: list[tuple[object, int, int]] = [(value, 1, 1)]
    while stack:
        current, multiplier, depth = stack.pop()
        max_instance_work = max(max_instance_work, _bounded_multiplier(multiplier, depth))
        if isinstance(current, bool):
            schema_nodes += 1
            continue
        if not isinstance(current, dict):
            continue
        schema_nodes += 1
        if schema_nodes > MAX_SCHEMA_NODES:
            return None

        for key in _REFERENCE_KEYWORDS:
            if key not in current:
                continue
            # Remote references may perform network IO, while even tiny local
            # reference cycles can expand exponentially in jsonschema. The v1
            # adapter accepts bounded inline schemas only.
            return None

        if "pattern" in current or "patternProperties" in current:
            # Python's backtracking regex engine has no execution budget.
            return None
        if "unevaluatedItems" in current or "unevaluatedProperties" in current:
            # jsonschema recursively recomputes evaluated locations for nested
            # combinators, which can grow exponentially on tiny inputs.
            return None
        if current.get("uniqueItems") is True:
            # jsonschema compares array members pairwise for uniqueness.
            return None

        enum = current.get("enum")
        if isinstance(enum, list):
            if len(enum) > MAX_SCHEMA_ENUM_VALUES or any(
                isinstance(item, (dict, list)) or (isinstance(item, float) and not math.isfinite(item)) for item in enum
            ):
                return None
            multiplier = _bounded_multiplier(multiplier, len(enum))
        const = current.get("const")
        if isinstance(const, (dict, list)):
            # Deep equality against a repeated array/object instance is not
            # interruptible once jsonschema starts running on a worker thread.
            return None
        max_instance_work = max(max_instance_work, _bounded_multiplier(multiplier, depth))

        for key in _SINGLE_SUBSCHEMA_KEYS:
            child = current.get(key)
            if isinstance(child, (dict, bool)):
                stack.append((child, multiplier, depth + 1))
            elif key == "items" and isinstance(child, list):
                alternatives += len(child)
                child_multiplier = _bounded_multiplier(multiplier, len(child))
                stack.extend((item, child_multiplier, depth + 1) for item in child if isinstance(item, (dict, bool)))

        for key in _SUBSCHEMA_ARRAY_KEYS:
            children = current.get(key)
            if isinstance(children, list):
                alternatives += len(children)
                child_multiplier = _bounded_multiplier(multiplier, len(children))
                stack.extend((item, child_multiplier, depth + 1) for item in children if isinstance(item, (dict, bool)))

        # Draft-03 permits schema values under these keywords. Modern drafts
        # use scalar type names, but the validator still supports old schemas.
        for key in _LEGACY_SUBSCHEMA_KEYS:
            children = current.get(key)
            if isinstance(children, (dict, bool)):
                stack.append((children, multiplier, depth + 1))
            elif isinstance(children, list):
                alternatives += len(children)
                child_multiplier = _bounded_multiplier(multiplier, len(children))
                stack.extend((item, child_multiplier, depth + 1) for item in children if isinstance(item, (dict, bool)))

        for key in _SUBSCHEMA_MAP_KEYS:
            children = current.get(key)
            if isinstance(children, dict):
                stack.extend(
                    (item, multiplier, depth + 1) for item in children.values() if isinstance(item, (dict, bool))
                )

        dependent_schemas = current.get("dependentSchemas")
        if isinstance(dependent_schemas, dict):
            schema_dependencies = [item for item in dependent_schemas.values() if isinstance(item, (dict, bool))]
            dependency_multiplier = _bounded_multiplier(multiplier, len(schema_dependencies))
            stack.extend((item, dependency_multiplier, depth + 1) for item in schema_dependencies)

        # Draft-07 dependencies may hold either required-property arrays or
        # subschemas. Only the latter can introduce validator work.
        dependencies = current.get("dependencies")
        if isinstance(dependencies, dict):
            schema_dependencies = [item for item in dependencies.values() if isinstance(item, (dict, bool))]
            dependency_multiplier = _bounded_multiplier(multiplier, len(schema_dependencies))
            stack.extend((item, dependency_multiplier, depth + 1) for item in schema_dependencies)

        if alternatives > MAX_SCHEMA_ALTERNATIVES:
            return None
    return metrics[0], max_instance_work


def _run_check(check: Any, *args: object) -> object:
    """Run one public async Guardrails check on a worker thread."""

    return asyncio.run(check(*args))


class Guardrails024StructuralToolAdapter:
    """Stable plugin boundary over Guardrails 0.24's public tool checks.

    Provider projection happens outside this class. The one intentional
    normalization here is result-name recovery: when a provider omits a result
    name, a unique call ID supplies the name Guardrails 0.24 requires.
    """

    def __init__(self, timeout_ms: int = 25_000) -> None:
        self._symbols = _load_guardrails_024_symbols()
        # RailsConfig currently loads a bundled jailbreak model that still uses
        # Guardrails' deprecated ``nim_url`` alias, even though this minimal
        # structural configuration does not enable that rail. Suppress only
        # that upstream warning while constructing the local adapter.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Use 'nim_base_url' instead.*",
                category=DeprecationWarning,
                module=r"nemoguardrails\.library\.jailbreak_detection\.rail_config",
            )
            config = self._symbols.rails_config.from_content(
                config={
                    "models": [
                        {
                            "type": "main",
                            # EngineRegistry's documented fallback is the
                            # OpenAI tool adapter. A private engine name keeps
                            # this structural-only model from reading provider
                            # API keys.
                            "engine": "nemo_relay_local_openai",
                            "model": "nemo-relay-structural-only",
                            "parameters": {"base_url": "http://127.0.0.1:1/v1"},
                        }
                    ]
                }
            )
        registry = self._symbols.engine_registry(config.models)
        self._manager = self._symbols.rails_manager(
            engine_registry=registry,
            task_manager=self._symbols.task_manager(config),
            input_flows=[],
            output_flows=[],
            tool_call_flows=["tool call validation"],
            tool_result_flows=["tool result validation"],
        )
        self._call_check = self._manager.are_tool_calls_safe
        self._result_check = self._manager.are_tool_results_safe
        self._timeout_seconds = timeout_ms / 1_000
        self._pending = 0
        self._closed = False

    async def _invoke(
        self,
        check: Any,
        *args: object,
        timeout_seconds: float | None = None,
    ) -> ToolVerdict:
        if self._pending >= MAX_PENDING_VALIDATIONS:
            return ToolVerdict(ToolVerdictKind.VALIDATOR_FAILURE)
        self._pending += 1
        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(_VALIDATOR_EXECUTOR, _run_check, check, *args)
        except Exception:
            self._pending -= 1
            return ToolVerdict(ToolVerdictKind.VALIDATOR_FAILURE)

        def release_capacity(_future: object) -> None:
            self._pending -= 1

        future.add_done_callback(release_capacity)
        try:
            # Shield the actual executor future. A timed-out or cancelled RPC
            # stops waiting, but its thread keeps owning admission capacity
            # until the synchronous validator really returns.
            outcome = await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self._timeout_seconds if timeout_seconds is None else timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return ToolVerdict(ToolVerdictKind.VALIDATOR_FAILURE)
        return self._result_verdict(outcome)

    @staticmethod
    def _result_verdict(result: object) -> ToolVerdict:
        is_safe = getattr(result, "is_safe", None)
        failed = getattr(result, "failed", None)
        if not isinstance(is_safe, bool) or not isinstance(failed, bool):
            return ToolVerdict(ToolVerdictKind.VALIDATOR_FAILURE)
        if is_safe:
            return ToolVerdict(ToolVerdictKind.PASSED)
        return ToolVerdict(ToolVerdictKind.VALIDATOR_FAILURE if failed else ToolVerdictKind.POLICY_BLOCK)

    def _guardrails_call(self, call: ToolCall) -> Any:
        return self._symbols.tool_call(
            id=call.call_id,
            type=call.type,
            function=self._symbols.tool_call_function(
                name=call.name or "",
                arguments=call.arguments,
            ),
        )

    @staticmethod
    def _guardrails_tool(definition: ToolDefinition) -> dict[str, object]:
        function: dict[str, object] = {"name": definition.name or ""}
        if definition.description is not None:
            function["description"] = definition.description
        if definition.input_schema is not None:
            function["parameters"] = definition.input_schema
        if definition.strict is not None:
            function["strict"] = definition.strict
        return {"type": definition.type, "function": function}

    def _guardrails_messages(self, exchanges: Sequence[ToolExchange]) -> list[dict[str, object]]:
        """Encode provider-neutral exchanges for the local OpenAI adapter."""

        messages: list[dict[str, object]] = []
        for exchange in exchanges:
            if messages:
                # Keep nominal exchanges separate even when an orphan result
                # has no assistant call to create a natural boundary.
                messages.append({"role": "user", "content": ""})

            calls_by_id: dict[str, ToolCall] = {}
            duplicate_call_id = False
            for call in exchange.calls:
                if call.call_id in calls_by_id:
                    duplicate_call_id = True
                calls_by_id[call.call_id] = call

            if exchange.calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call.call_id,
                                "type": call.type,
                                "function": {
                                    "name": call.name or "",
                                    "arguments": json.dumps(
                                        call.arguments,
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                        allow_nan=False,
                                    ),
                                },
                            }
                            for call in exchange.calls
                        ],
                    }
                )

            for result in exchange.results:
                effective_name = result.name
                if effective_name is None and not duplicate_call_id and result.call_id in calls_by_id:
                    # Several provider protocols omit the result name. A
                    # unique call ID supplies the name RailsManager requires.
                    effective_name = calls_by_id[result.call_id].name
                message: dict[str, object] = {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.content,
                }
                if effective_name is not None:
                    message["name"] = effective_name
                messages.append(message)
        return messages

    async def check_calls(
        self,
        definitions: Sequence[ToolDefinition],
        calls: Sequence[ToolCall],
    ) -> ToolVerdict:
        """Validate model-emitted calls against the request's tool declarations."""

        try:
            if not calls:
                # Match Guardrails' own manager semantics: unused declarations
                # are irrelevant when the model did not emit a tool call.
                return ToolVerdict(ToolVerdictKind.PASSED)
            if len(definitions) > MAX_TOOL_DEFINITIONS or len(calls) > MAX_MODEL_TOOL_CALLS:
                return ToolVerdict(ToolVerdictKind.COVERAGE_BLOCK)
            if any(
                not call.call_id or not call.name or call.type != "function" or not isinstance(call.arguments, dict)
                for call in calls
            ):
                return ToolVerdict(ToolVerdictKind.COVERAGE_BLOCK)
            call_ids = [call.call_id for call in calls]
            if len(call_ids) != len(set(call_ids)):
                return ToolVerdict(ToolVerdictKind.COVERAGE_BLOCK)

            definitions_by_name: dict[str, ToolDefinition] = {}
            for definition in definitions:
                if (
                    definition.type != "function"
                    or not definition.name
                    or (definition.input_schema is not None and not isinstance(definition.input_schema, dict))
                    or definition.name in definitions_by_name
                ):
                    return ToolVerdict(ToolVerdictKind.COVERAGE_BLOCK)
                definitions_by_name[definition.name] = definition
            guardrails_calls = [self._guardrails_call(call) for call in calls]
            validation_work = 0
            for call in calls:
                argument_metrics = _json_metrics(call.arguments)
                if argument_metrics is None:
                    return ToolVerdict(ToolVerdictKind.COVERAGE_BLOCK)
                definition = definitions_by_name.get(call.name or "")
                if definition is None or definition.input_schema is None:
                    validation_work += argument_metrics[0]
                else:
                    schema_safety = _safe_schema(definition.input_schema)
                    if schema_safety is None:
                        return ToolVerdict(ToolVerdictKind.COVERAGE_BLOCK)
                    schema_size, instance_multiplier = schema_safety
                    validation_work += schema_size + argument_metrics[0] * instance_multiplier
                if validation_work > MAX_SCHEMA_VALIDATION_WORK:
                    return ToolVerdict(ToolVerdictKind.COVERAGE_BLOCK)
        except asyncio.CancelledError:
            raise
        except Exception:
            return ToolVerdict(ToolVerdictKind.VALIDATOR_FAILURE)
        llm_params = {"tools": [self._guardrails_tool(definition) for definition in definitions]}
        return await self._invoke(self._call_check, guardrails_calls, llm_params)

    async def check_call_candidates(
        self,
        definitions: Sequence[ToolDefinition],
        candidates: Sequence[Sequence[ToolCall]],
    ) -> ToolVerdict:
        """Validate all response candidates within one request deadline."""

        try:
            async with asyncio.timeout(self._timeout_seconds):
                for calls in candidates:
                    if not calls:
                        continue
                    verdict = await self.check_calls(definitions, calls)
                    if not verdict.passed:
                        return verdict
        except TimeoutError:
            return ToolVerdict(ToolVerdictKind.VALIDATOR_FAILURE)
        return ToolVerdict(ToolVerdictKind.PASSED)

    async def check_results(self, exchanges: Sequence[ToolExchange]) -> ToolVerdict:
        """Validate result linkage independently for each assistant turn."""

        deadline = asyncio.get_running_loop().time() + self._timeout_seconds
        history_nodes = 0
        try:
            if (
                len(exchanges) > MAX_TOOL_EXCHANGES
                or sum(len(item.calls) + len(item.results) for item in exchanges) > MAX_TOOL_HISTORY_ITEMS
            ):
                return ToolVerdict(ToolVerdictKind.COVERAGE_BLOCK)
            for exchange in exchanges:
                for call in exchange.calls:
                    if (
                        not call.call_id
                        or not call.name
                        or call.type != "function"
                        or not isinstance(call.arguments, dict)
                    ):
                        return ToolVerdict(ToolVerdictKind.COVERAGE_BLOCK)
                    metrics = _json_metrics(call.arguments)
                    if metrics is None:
                        return ToolVerdict(ToolVerdictKind.COVERAGE_BLOCK)
                    history_nodes += metrics[0]
                for result in exchange.results:
                    if result.content is None:
                        continue
                    if not isinstance(result.content, (str, list)):
                        return ToolVerdict(ToolVerdictKind.COVERAGE_BLOCK)
                    metrics = _json_metrics(result.content)
                    if metrics is None:
                        return ToolVerdict(ToolVerdictKind.COVERAGE_BLOCK)
                    history_nodes += metrics[0]
                if history_nodes > MAX_TOOL_HISTORY_JSON_NODES:
                    return ToolVerdict(ToolVerdictKind.COVERAGE_BLOCK)

            messages = self._guardrails_messages(exchanges)
            # Bound the exact normalized history handed to Guardrails, not
            # only argument/result payloads. This also covers IDs, names, and
            # the message/tool-call wrapper objects created above.
            if (
                _json_metrics(
                    messages,
                    max_nodes=MAX_TOOL_HISTORY_JSON_NODES,
                    max_characters=MAX_TOOL_HISTORY_CHARACTERS,
                )
                is None
            ):
                return ToolVerdict(ToolVerdictKind.COVERAGE_BLOCK)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return ToolVerdict(ToolVerdictKind.VALIDATOR_FAILURE)
            return await self._invoke(
                self._result_check,
                messages,
                timeout_seconds=remaining,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return ToolVerdict(ToolVerdictKind.VALIDATOR_FAILURE)

    async def _verify_compatibility(self) -> None:
        """Behaviorally probe the public API before accepting traffic."""

        definition = ToolDefinition(
            name="nemo_relay_probe",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        )
        allowed = await self.check_calls(
            [definition],
            [ToolCall(call_id="probe-1", name="nemo_relay_probe", arguments={"value": "ok"})],
        )
        blocked = await self.check_calls(
            [definition],
            [ToolCall(call_id="probe-2", name="nemo_relay_probe", arguments={})],
        )
        linked = await self.check_results(
            [
                ToolExchange(
                    calls=(ToolCall(call_id="probe-1", name="nemo_relay_probe", arguments={"value": "ok"}),),
                    results=(ToolResult(call_id="probe-1", name=None, content="ok"),),
                )
            ]
        )
        if not allowed.passed or blocked.kind is not ToolVerdictKind.POLICY_BLOCK or not linked.passed:
            raise RuntimeError("NeMo Guardrails structural-tool compatibility probe failed")

    async def verify_compatibility(self) -> None:
        """Run the complete compatibility probe within one activation budget."""

        try:
            await asyncio.wait_for(self._verify_compatibility(), timeout=self._timeout_seconds)
        except TimeoutError as exc:
            raise RuntimeError("NeMo Guardrails structural-tool compatibility probe timed out") from exc

    async def close(self) -> None:
        """Close the HTTP client owned by the Guardrails 0.24 RailsManager."""

        if self._closed:
            return
        await self._manager.stop()
        self._closed = True
