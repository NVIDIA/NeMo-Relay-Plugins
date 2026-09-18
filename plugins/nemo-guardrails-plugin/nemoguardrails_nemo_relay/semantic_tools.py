# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in semantic checks for Relay-managed tool execution and history.

Guardrails' structural tool rails validate function names, schemas, argument
shape, and result linkage.  They deliberately do not inspect what a tool result
means.  This module keeps that structural contract separate from an optional
plugin-owned adapter that presents bounded canonical JSON to ordinary text
input/output rails.

No result is rewritten.  A text rail that returns ``MODIFIED`` is rejected,
because changing an arbitrary JSON subtree cannot be proven lossless.

The policy checks the arguments and result visible at its Relay execution-
interceptor boundary.  Relay does not reserve that boundary exclusively: an
inner tool interceptor can replace arguments after the input check, and an
equal-priority outer interceptor can replace a result after the output check.
Deployments that require these checks as enforcement must trust the remaining
tool execution middleware not to rewrite checked content.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from nemo_relay_plugin import (
    ToolExecutionContext,
    ToolExecutionInterceptOutcome,
    ToolNext,
)

from .policy_error import PolicyRejectedError
from .policy_marks import (
    PolicyAction,
    PolicyBackend,
    PolicyDecisionMarks,
    PolicyOutcome,
    PolicyPhase,
)
from .structural_tools import ToolCall, ToolExchange, ToolResult

DEFAULT_MAX_SEMANTIC_TOOL_BYTES = 64 * 1024
MIN_MAX_SEMANTIC_TOOL_BYTES = 1_024
MAX_MAX_SEMANTIC_TOOL_BYTES = 1024 * 1024
MAX_SEMANTIC_JSON_DEPTH = 64
MAX_SEMANTIC_JSON_NODES = 65_536
MAX_SEMANTIC_HISTORY_RESULTS = 32
MAX_TOOL_NAME_CHARACTERS = 256


class SemanticResultSource(str, Enum):
    """Where tool results are semantically checked."""

    OFF = "off"
    EXECUTION = "execution"
    HISTORY = "history"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class SemanticToolSettings:
    """Bounded opt-in settings for semantic tool checks."""

    check_arguments: bool = False
    result_source: SemanticResultSource = SemanticResultSource.OFF
    max_payload_bytes: int = DEFAULT_MAX_SEMANTIC_TOOL_BYTES

    @property
    def enabled(self) -> bool:
        return self.check_arguments or self.result_source is not SemanticResultSource.OFF

    @property
    def checks_execution(self) -> bool:
        return self.check_arguments or self.result_source in {
            SemanticResultSource.EXECUTION,
            SemanticResultSource.BOTH,
        }

    @property
    def checks_execution_results(self) -> bool:
        return self.result_source in {SemanticResultSource.EXECUTION, SemanticResultSource.BOTH}

    @property
    def checks_history_results(self) -> bool:
        return self.result_source in {SemanticResultSource.HISTORY, SemanticResultSource.BOTH}


def semantic_tool_settings_from_config(value: object) -> SemanticToolSettings:
    """Parse plugin-owned semantic-tool settings without retaining content."""

    if value is None:
        return SemanticToolSettings()
    if not isinstance(value, Mapping):
        raise ValueError("semantic_tool_policy must be an object")
    unknown = set(value) - {"check_arguments", "result_source", "max_payload_bytes"}
    if unknown:
        raise ValueError("semantic_tool_policy contains unknown fields")

    check_arguments = value.get("check_arguments", False)
    if not isinstance(check_arguments, bool):
        raise ValueError("semantic_tool_policy.check_arguments must be a boolean")
    try:
        result_source = SemanticResultSource(value.get("result_source", SemanticResultSource.OFF.value))
    except (TypeError, ValueError):
        raise ValueError("semantic_tool_policy.result_source is unsupported") from None
    max_payload_bytes = value.get("max_payload_bytes", DEFAULT_MAX_SEMANTIC_TOOL_BYTES)
    if (
        isinstance(max_payload_bytes, bool)
        or not isinstance(max_payload_bytes, int)
        or not MIN_MAX_SEMANTIC_TOOL_BYTES <= max_payload_bytes <= MAX_MAX_SEMANTIC_TOOL_BYTES
    ):
        raise ValueError(
            "semantic_tool_policy.max_payload_bytes must be between "
            f"{MIN_MAX_SEMANTIC_TOOL_BYTES} and {MAX_MAX_SEMANTIC_TOOL_BYTES}"
        )
    return SemanticToolSettings(
        check_arguments=check_arguments,
        result_source=result_source,
        max_payload_bytes=max_payload_bytes,
    )


class SemanticToolVerdictKind(str, Enum):
    """Content-free semantic decisions safe to retain or transport."""

    PASSED = "passed"
    POLICY_BLOCK = "policy_block"
    MODIFIED = "modified"
    COVERAGE_BLOCK = "coverage_block"
    TIMEOUT = "timeout"
    OVERLOADED = "overloaded"
    CHECK_FAILURE = "check_failure"


@dataclass(frozen=True, slots=True)
class SemanticToolVerdict:
    """One content-free text-rail decision."""

    kind: SemanticToolVerdictKind

    @property
    def passed(self) -> bool:
        return self.kind is SemanticToolVerdictKind.PASSED


class SemanticToolTextChecker(Protocol):
    """Narrow text-check contract supplied by the worker runtime."""

    async def check_tool_arguments(self, canonical_arguments: str) -> SemanticToolVerdict:
        """Evaluate canonical tool arguments as an input rail."""

    async def check_tool_result(
        self,
        canonical_arguments: str,
        canonical_result: str,
    ) -> SemanticToolVerdict:
        """Evaluate canonical result JSON as an output rail."""


class SemanticToolPolicyError(PolicyRejectedError):
    """Stop a tool or LLM execution with a content-free decision."""

    def __init__(self, verdict: SemanticToolVerdict, phase: str) -> None:
        self.verdict = verdict
        self.phase = phase
        super().__init__(semantic_tool_error_message(verdict, phase=phase))


def semantic_tool_error_message(verdict: SemanticToolVerdict, *, phase: str) -> str:
    """Map a typed verdict to a stable application-facing reason."""

    subject = "tool arguments" if phase == "arguments" else "tool result"
    if verdict.kind is SemanticToolVerdictKind.POLICY_BLOCK:
        return f"NeMo Guardrails rejected {subject}"
    if verdict.kind is SemanticToolVerdictKind.MODIFIED:
        return f"NeMo Guardrails modified {subject}, but this worker cannot safely rewrite JSON"
    if verdict.kind is SemanticToolVerdictKind.COVERAGE_BLOCK:
        return f"NeMo Guardrails cannot verify {subject}"
    if verdict.kind is SemanticToolVerdictKind.TIMEOUT:
        return f"NeMo Guardrails {subject} check timed out"
    if verdict.kind is SemanticToolVerdictKind.OVERLOADED:
        return f"NeMo Guardrails {subject} check is overloaded"
    if verdict.kind is SemanticToolVerdictKind.CHECK_FAILURE:
        return f"NeMo Guardrails {subject} check failed"
    return f"NeMo Guardrails returned an invalid {subject} decision"


class SemanticToolPolicy:
    """Apply semantic text rails without modifying tool traffic."""

    def __init__(
        self,
        settings: SemanticToolSettings,
        checker: SemanticToolTextChecker,
        *,
        timeout_ms: int,
        decision_marks: PolicyDecisionMarks | None = None,
    ) -> None:
        self.settings = settings
        self._checker = checker
        self._timeout_seconds = timeout_ms / 1_000
        self._decision_marks = decision_marks

    async def _emit_decision(
        self,
        phase: PolicyPhase,
        verdict: SemanticToolVerdict,
        started_ns: int,
        backend: PolicyBackend | None = None,
    ) -> None:
        if self._decision_marks is None:
            return
        await self._decision_marks.emit(
            phase=phase,
            outcome=PolicyOutcome(verdict.kind.value),
            action=PolicyAction.CONTINUE if verdict.passed else PolicyAction.REJECT,
            started_ns=started_ns,
            backend=backend,
        )

    async def _check_arguments(self, arguments: str) -> None:
        started_ns = PolicyDecisionMarks.start()
        verdict = await self._checker.check_tool_arguments(arguments)
        await self._emit_decision(PolicyPhase.TOOL_ARGUMENTS, verdict, started_ns)
        _require_passed(verdict, phase="arguments")

    async def _check_result(self, arguments: str, result: str) -> None:
        started_ns = PolicyDecisionMarks.start()
        verdict = await self._checker.check_tool_result(arguments, result)
        await self._emit_decision(PolicyPhase.TOOL_RESULT, verdict, started_ns)
        _require_passed(verdict, phase="result")

    async def execute(
        self,
        context: ToolExecutionContext,
        next_call: ToolNext,
    ) -> ToolExecutionInterceptOutcome:
        """Wrap one actual Relay tool call and preserve its exact result."""

        started_ns = PolicyDecisionMarks.start()
        try:
            arguments = canonical_tool_arguments(
                context.tool_name,
                context.args,
                max_bytes=self.settings.max_payload_bytes,
            )
        except SemanticToolPolicyError as error:
            await self._emit_decision(
                PolicyPhase.TOOL_ARGUMENTS,
                error.verdict,
                started_ns,
                backend=PolicyBackend.WORKER,
            )
            raise
        if self.settings.check_arguments:
            await self._check_arguments(arguments)

        downstream = await next_call.call(context.args)
        if self.settings.checks_execution_results:
            started_ns = PolicyDecisionMarks.start()
            try:
                result = canonical_tool_result(
                    context.tool_name,
                    downstream.result,
                    downstream.annotation,
                    max_bytes=self.settings.max_payload_bytes,
                )
            except SemanticToolPolicyError as error:
                await self._emit_decision(
                    PolicyPhase.TOOL_RESULT,
                    error.verdict,
                    started_ns,
                    backend=PolicyBackend.WORKER,
                )
                raise
            await self._check_result(arguments, result)
        return ToolExecutionInterceptOutcome(
            result=downstream.result,
            annotation=downstream.annotation,
        )

    async def check_history(self, exchanges: tuple[ToolExchange, ...]) -> None:
        """Check prior transcript results when a host lacks managed tool calls."""

        if not self.settings.checks_history_results:
            return
        started_ns = PolicyDecisionMarks.start()
        try:
            if sum(len(exchange.results) for exchange in exchanges) > MAX_SEMANTIC_HISTORY_RESULTS:
                raise _coverage_error("result")
            async with asyncio.timeout(self._timeout_seconds):
                for exchange in exchanges:
                    for result in exchange.results:
                        call = _matching_call(exchange.calls, result)
                        tool_name = _tool_name(call, result)
                        arguments = canonical_tool_arguments(
                            tool_name,
                            call.arguments,
                            max_bytes=self.settings.max_payload_bytes,
                        )
                        canonical_result = canonical_history_tool_result(
                            tool_name,
                            result,
                            max_bytes=self.settings.max_payload_bytes,
                        )
                        _require_passed(
                            await self._checker.check_tool_result(arguments, canonical_result),
                            phase="result",
                        )
        except TimeoutError:
            error = SemanticToolPolicyError(SemanticToolVerdict(SemanticToolVerdictKind.TIMEOUT), "result")
            await self._emit_decision(PolicyPhase.TOOL_HISTORY_RESULTS, error.verdict, started_ns)
            raise error from None
        except SemanticToolPolicyError as error:
            await self._emit_decision(PolicyPhase.TOOL_HISTORY_RESULTS, error.verdict, started_ns)
            raise
        await self._emit_decision(
            PolicyPhase.TOOL_HISTORY_RESULTS,
            SemanticToolVerdict(SemanticToolVerdictKind.PASSED),
            started_ns,
        )


def _require_passed(verdict: SemanticToolVerdict, *, phase: str) -> None:
    if not verdict.passed:
        raise SemanticToolPolicyError(verdict, phase)


def _matching_call(calls: tuple[ToolCall, ...], result: ToolResult) -> ToolCall:
    """Find one unambiguous originating call without trusting result content."""

    if result.call_id is not None:
        matches = [call for call in calls if call.call_id == result.call_id]
    elif result.name is not None:
        matches = [call for call in calls if call.name == result.name]
    else:
        matches = []
    if len(matches) != 1:
        raise SemanticToolPolicyError(
            SemanticToolVerdict(SemanticToolVerdictKind.COVERAGE_BLOCK),
            "result",
        )
    call = matches[0]
    if result.name is not None and call.name != result.name:
        raise SemanticToolPolicyError(
            SemanticToolVerdict(SemanticToolVerdictKind.COVERAGE_BLOCK),
            "result",
        )
    return call


def _tool_name(call: ToolCall, result: ToolResult) -> str:
    name = result.name if result.name is not None else call.name
    if name is None:
        raise SemanticToolPolicyError(
            SemanticToolVerdict(SemanticToolVerdictKind.COVERAGE_BLOCK),
            "result",
        )
    return name


def canonical_tool_arguments(tool_name: object, arguments: object, *, max_bytes: int) -> str:
    """Build the stable text presented to a Guardrails input rail."""

    _validate_tool_name(tool_name, phase="arguments")
    return _canonical_json(
        {"arguments": arguments, "tool_name": tool_name},
        max_bytes=max_bytes,
        phase="arguments",
    )


def canonical_tool_result(
    tool_name: object,
    result: object,
    annotation: object,
    *,
    max_bytes: int,
) -> str:
    """Build stable result text for a managed Relay tool execution."""

    _validate_tool_name(tool_name, phase="result")
    return _canonical_json(
        {
            "annotation": annotation,
            "result": result,
            "tool_name": tool_name,
        },
        max_bytes=max_bytes,
        phase="result",
    )


def canonical_history_tool_result(tool_name: object, result: ToolResult, *, max_bytes: int) -> str:
    """Build stable result text from a completely projected LLM transcript."""

    _validate_tool_name(tool_name, phase="result")
    return _canonical_json(
        {
            "is_error": result.is_error,
            "result": result.content,
            "tool_name": tool_name,
        },
        max_bytes=max_bytes,
        phase="result",
    )


def _validate_tool_name(value: object, *, phase: str) -> None:
    if not isinstance(value, str) or not value or len(value) > MAX_TOOL_NAME_CHARACTERS or "\x00" in value:
        raise _coverage_error(phase)


def _canonical_json(value: object, *, max_bytes: int, phase: str) -> str:
    """Serialize bounded JSON without materializing an unbounded intermediate."""

    stack: list[tuple[object, int]] = [(value, 0)]
    seen_containers: set[int] = set()
    nodes = 0
    string_bytes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_SEMANTIC_JSON_NODES or depth > MAX_SEMANTIC_JSON_DEPTH:
            raise _coverage_error(phase)
        if current is None or isinstance(current, bool):
            continue
        if isinstance(current, str):
            try:
                string_bytes += len(current.encode("utf-8"))
            except UnicodeEncodeError:
                raise _coverage_error(phase) from None
            if string_bytes > max_bytes:
                raise _coverage_error(phase)
            continue
        if isinstance(current, int):
            # Avoid Python's very-large-integer conversion limit and an
            # attacker-controlled serialization cost.
            if current.bit_length() > max_bytes * 3:
                raise _coverage_error(phase)
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise _coverage_error(phase)
            continue
        if isinstance(current, dict):
            identity = id(current)
            if identity in seen_containers:
                raise _coverage_error(phase)
            seen_containers.add(identity)
            for key, child in current.items():
                if not isinstance(key, str):
                    raise _coverage_error(phase)
                stack.append((key, depth + 1))
                stack.append((child, depth + 1))
            continue
        if isinstance(current, list):
            identity = id(current)
            if identity in seen_containers:
                raise _coverage_error(phase)
            seen_containers.add(identity)
            stack.extend((child, depth + 1) for child in current)
            continue
        raise _coverage_error(phase)

    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(serialized.encode("utf-8")) > max_bytes:
            raise _coverage_error(phase)
    except (TypeError, ValueError, UnicodeEncodeError):
        raise _coverage_error(phase) from None
    return serialized


def _coverage_error(phase: str) -> SemanticToolPolicyError:
    return SemanticToolPolicyError(
        SemanticToolVerdict(SemanticToolVerdictKind.COVERAGE_BLOCK),
        phase,
    )
