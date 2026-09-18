# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text and structural execution policy for the Relay Guardrails worker."""

from __future__ import annotations

import asyncio
import contextvars
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any, NoReturn, cast

# Keep imports of this module safe even when tooling bypasses ``worker``.
os.environ["NEMO_GUARDRAILS_NO_USAGE_STATS"] = "1"

from nemo_relay_plugin import Json, LlmNext, LlmRequest, LlmStreamNext  # noqa: E402
from nemoguardrails import Guardrails  # noqa: E402
from nemoguardrails.rails.llm.options import RailType  # noqa: E402

from .check_backend import (  # noqa: E402
    CheckBackend,
    CheckStatus,
    LocalCheckBackend,
)
from .codec_projection import (  # noqa: E402
    MAX_RESPONSE_SEGMENTS,
    _DecodedTextRequest,
    _DecodedToolRequest,
    _NativeCodecProjector,
    _UnsupportedRequest,
)
from .execution_context import (  # noqa: E402
    ExecutionCodecContext,
    ExecutionCodecContextError,
    execution_codec_context,
)
from .policy_error import PolicyRejectedError  # noqa: E402
from .policy_marks import (  # noqa: E402
    PolicyAction,
    PolicyBackend,
    PolicyDecisionMarks,
    PolicyOutcome,
    PolicyPhase,
)
from .runtime_selection import runtime_uses_explain_state  # noqa: E402
from .semantic_tools import (  # noqa: E402
    SemanticToolPolicy,
    SemanticToolPolicyError,
    SemanticToolVerdict,
    SemanticToolVerdictKind,
)
from .structural_tools import (  # noqa: E402
    Guardrails024StructuralToolAdapter,
    StructuralToolSettings,
    ToolVerdict,
    ToolVerdictKind,
)
from .text_mutation import (  # noqa: E402
    MutationMode,
    MutationPolicy,
    TextMutationError,
    rewrite_request_text,
    rewrite_response_texts,
)

MAX_PENDING_CHECKS = 32
MAX_PROJECTED_MESSAGES = 512
MAX_PROJECTED_TEXT_CHARACTERS = 1_000_000
MAX_OUTPUT_CHECK_SEGMENTS = MAX_RESPONSE_SEGMENTS
_ANTHROPIC_COUNT_TOKENS_OPERATION = "anthropic.count_tokens"
_SAFE_RAIL_NAME = re.compile(r"^[A-Za-z0-9_. -]{1,128}$")


def _blocked_reason(rail: Any, *, phase: str = "input") -> str:
    if isinstance(rail, str) and _SAFE_RAIL_NAME.fullmatch(rail):
        return f"NeMo Guardrails prevented execution in {phase} rail '{rail}'"
    return f"NeMo Guardrails prevented execution during {phase} checking"


class _TextCheckKind(str, Enum):
    """Content-free outcome of one bounded text check."""

    PASSED = "passed"
    POLICY_BLOCK = "policy_block"
    MODIFIED = "modified"
    OVERSIZED = "oversized"
    TIMEOUT = "timeout"
    OVERLOADED = "overloaded"
    CHECK_FAILURE = "check_failure"
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True, slots=True)
class _TextCheckOutcome:
    kind: _TextCheckKind
    rail: str | None = None
    transformed_content: str | None = None


class _TextRailsPolicy:
    """Serialize and bound input and output checks on one Guardrails engine."""

    def __init__(
        self,
        rails: Guardrails | None,
        timeout_ms: int,
        *,
        backend: CheckBackend | None = None,
        projector: _NativeCodecProjector | None = None,
        allow_structural_tools: bool = False,
        allow_tool_results: bool = False,
    ) -> None:
        self._rails = rails
        if backend is None:
            if rails is None:
                raise ValueError("a local Guardrails engine or check backend is required")
            backend = LocalCheckBackend(rails, RailType)
        self._backend = backend
        self._projector = projector or _NativeCodecProjector()
        self._allow_structural_tools = allow_structural_tools
        self._allow_tool_results = allow_tool_results
        self._timeout_seconds = timeout_ms / 1000
        self._permit = asyncio.Semaphore(1) if rails is not None and runtime_uses_explain_state(rails) else None
        self._pending = 0

    def _discard_explanation(self) -> None:
        """Drop request and evaluator content retained by Guardrails."""

        if self._rails is None or not runtime_uses_explain_state(self._rails):
            return
        explanation = self._rails.explain()
        explanation.llm_calls.clear()
        explanation.colang_history = None

    async def close(self) -> None:
        await self._backend.close()

    async def _check_outcome(
        self,
        messages: list[dict[str, str]],
        *,
        rail_type: RailType,
        expected_content: str,
    ) -> _TextCheckOutcome:
        if self._pending >= MAX_PENDING_CHECKS:
            return _TextCheckOutcome(_TextCheckKind.OVERLOADED)
        self._pending += 1
        try:
            if len(messages) > MAX_PROJECTED_MESSAGES or (
                sum(len(message["content"]) for message in messages) > MAX_PROJECTED_TEXT_CHARACTERS
            ):
                return _TextCheckOutcome(_TextCheckKind.OVERSIZED)

            async with asyncio.timeout(self._timeout_seconds):
                async with self._permit if self._permit is not None else nullcontext():
                    try:
                        result = await asyncio.create_task(
                            self._backend.check(messages, rail_type.value),
                            context=contextvars.Context(),
                        )
                    finally:
                        self._discard_explanation()
        except TimeoutError:
            return _TextCheckOutcome(_TextCheckKind.TIMEOUT)
        except Exception:
            return _TextCheckOutcome(_TextCheckKind.CHECK_FAILURE)
        finally:
            self._pending -= 1

        if result.status is CheckStatus.PASSED:
            if result.content == expected_content:
                return _TextCheckOutcome(_TextCheckKind.PASSED)
            return _TextCheckOutcome(_TextCheckKind.INCONSISTENT)
        if result.status is CheckStatus.BLOCKED:
            return _TextCheckOutcome(
                _TextCheckKind.POLICY_BLOCK,
                rail=result.rail if isinstance(result.rail, str) and _SAFE_RAIL_NAME.fullmatch(result.rail) else None,
            )
        if result.status is CheckStatus.MODIFIED:
            transformed_content = result.transformed_content
            if not isinstance(transformed_content, str) or transformed_content == expected_content:
                return _TextCheckOutcome(_TextCheckKind.INCONSISTENT)
            projected_size = sum(len(message["content"]) for message in messages)
            if projected_size - len(expected_content) + len(transformed_content) > MAX_PROJECTED_TEXT_CHARACTERS:
                return _TextCheckOutcome(_TextCheckKind.OVERSIZED)
            return _TextCheckOutcome(_TextCheckKind.MODIFIED, transformed_content=transformed_content)
        return _TextCheckOutcome(_TextCheckKind.CHECK_FAILURE)

    @staticmethod
    def _outcome_message(outcome: _TextCheckOutcome, *, phase: str) -> str | None:
        if outcome.kind is _TextCheckKind.PASSED:
            return None
        if outcome.kind is _TextCheckKind.POLICY_BLOCK:
            return _blocked_reason(outcome.rail, phase=phase)
        if outcome.kind is _TextCheckKind.MODIFIED:
            return f"NeMo Guardrails modified the {phase}, but this worker cannot safely rewrite it"
        if outcome.kind is _TextCheckKind.OVERSIZED:
            return f"NeMo Guardrails cannot verify this oversized {phase} context"
        if outcome.kind is _TextCheckKind.TIMEOUT:
            return f"NeMo Guardrails {phase} check timed out"
        if outcome.kind is _TextCheckKind.OVERLOADED:
            return f"NeMo Guardrails {phase} check is overloaded"
        if outcome.kind is _TextCheckKind.CHECK_FAILURE:
            return f"NeMo Guardrails {phase} check failed"
        if outcome.kind is _TextCheckKind.INCONSISTENT:
            return f"NeMo Guardrails returned an inconsistent {phase}-check result"
        return f"NeMo Guardrails {phase} check failed"

    @staticmethod
    def _semantic_verdict(outcome: _TextCheckOutcome) -> SemanticToolVerdict:
        mapping = {
            _TextCheckKind.PASSED: SemanticToolVerdictKind.PASSED,
            _TextCheckKind.POLICY_BLOCK: SemanticToolVerdictKind.POLICY_BLOCK,
            _TextCheckKind.MODIFIED: SemanticToolVerdictKind.MODIFIED,
            _TextCheckKind.OVERSIZED: SemanticToolVerdictKind.COVERAGE_BLOCK,
            _TextCheckKind.TIMEOUT: SemanticToolVerdictKind.TIMEOUT,
            _TextCheckKind.OVERLOADED: SemanticToolVerdictKind.OVERLOADED,
            _TextCheckKind.CHECK_FAILURE: SemanticToolVerdictKind.CHECK_FAILURE,
            _TextCheckKind.INCONSISTENT: SemanticToolVerdictKind.CHECK_FAILURE,
        }
        return SemanticToolVerdict(mapping[outcome.kind])

    async def check_tool_arguments(self, canonical_arguments: str) -> SemanticToolVerdict:
        outcome = await self._check_outcome(
            [{"role": "user", "content": canonical_arguments}],
            rail_type=RailType.INPUT,
            expected_content=canonical_arguments,
        )
        return self._semantic_verdict(outcome)

    async def check_tool_result(
        self,
        canonical_arguments: str,
        canonical_result: str,
    ) -> SemanticToolVerdict:
        outcome = await self._check_outcome(
            [
                {"role": "user", "content": canonical_arguments},
                {"role": "assistant", "content": canonical_result},
            ],
            rail_type=RailType.OUTPUT,
            expected_content=canonical_result,
        )
        return self._semantic_verdict(outcome)

    def project_request(
        self,
        request: LlmRequest,
        *,
        require_user: bool,
        codec_name: str | None = None,
        annotated_request: dict[str, Any] | None = None,
    ) -> _DecodedTextRequest:
        if self._pending >= MAX_PENDING_CHECKS:
            raise _UnsupportedRequest("the Guardrails text-check queue is full")
        return self._projector.project_text(
            request,
            allow_structural_tools=self._allow_structural_tools,
            allow_tool_results=self._allow_tool_results,
            require_user=require_user,
            codec_name=codec_name,
            annotated_request=annotated_request,
        )

    async def check_input(self, projected: _DecodedTextRequest) -> str | None:
        outcome = await self.check_input_outcome(projected)
        return self._outcome_message(outcome, phase=RailType.INPUT.value)

    async def check_input_outcome(self, projected: _DecodedTextRequest) -> _TextCheckOutcome:
        latest_user_text = next(
            message["content"] for message in reversed(projected.messages) if message["role"] == "user"
        )
        return await self._check_outcome(
            list(projected.messages),
            rail_type=RailType.INPUT,
            expected_content=latest_user_text,
        )

    async def check_output(self, projected: _DecodedTextRequest, output_text: str) -> str | None:
        outcome = await self.check_output_outcome(projected, output_text)
        return self._outcome_message(outcome, phase=RailType.OUTPUT.value)

    async def check_output_outcome(
        self,
        projected: _DecodedTextRequest,
        output_text: str,
    ) -> _TextCheckOutcome:
        return await self._check_outcome(
            [*projected.messages, {"role": "assistant", "content": output_text}],
            rail_type=RailType.OUTPUT,
            expected_content=output_text,
        )

    async def check_output_outcomes(
        self,
        projected: _DecodedTextRequest,
        output_texts: tuple[str, ...],
    ) -> tuple[_TextCheckOutcome, ...]:
        async with asyncio.timeout(self._timeout_seconds):
            outcomes: list[_TextCheckOutcome] = []
            for output_text in output_texts:
                outcomes.append(await self.check_output_outcome(projected, output_text))
            return tuple(outcomes)

    async def check(self, request: LlmRequest) -> str | None:
        if self._pending >= MAX_PENDING_CHECKS:
            return "NeMo Guardrails input check is overloaded"
        try:
            messages = self._projector.project(
                request,
                allow_structural_tools=self._allow_structural_tools,
                allow_tool_results=self._allow_tool_results,
            )
            projected = _DecodedTextRequest((), tuple(messages))
        except _UnsupportedRequest:
            return "NeMo Guardrails cannot verify this unsupported provider request"
        return await self.check_input(projected)


def _tool_verdict_message(verdict: ToolVerdict, *, output: bool) -> str | None:
    if verdict.kind is ToolVerdictKind.PASSED:
        return None
    phase = "model tool calls" if output else "tool results"
    if verdict.kind is ToolVerdictKind.POLICY_BLOCK:
        return f"NeMo Guardrails rejected {phase}"
    if verdict.kind is ToolVerdictKind.COVERAGE_BLOCK:
        return f"NeMo Guardrails cannot verify {phase}"
    return f"NeMo Guardrails {phase} check failed"


class _LlmPolicyError(PolicyRejectedError):
    """Stop an execution intercept without exposing provider or policy data."""


class _LlmExecutionPolicy:
    """Run configured text and structural rails around one LLM invocation."""

    def __init__(
        self,
        projector: _NativeCodecProjector,
        text_policy: _TextRailsPolicy | None,
        *,
        input_enabled: bool,
        output_enabled: bool,
        adapter: Guardrails024StructuralToolAdapter | None,
        settings: StructuralToolSettings,
        semantic_tool_policy: SemanticToolPolicy | None = None,
        mutation_policy: MutationPolicy | None = None,
        decision_marks: PolicyDecisionMarks | None = None,
    ) -> None:
        self._projector = projector
        self._text_policy = text_policy
        self._input_enabled = input_enabled
        self._output_enabled = output_enabled
        self._adapter = adapter
        self._settings = settings
        self._semantic_tool_policy = semantic_tool_policy
        self._mutation_policy = mutation_policy or MutationPolicy()
        self._decision_marks = decision_marks

    async def _emit_decision(
        self,
        *,
        phase: PolicyPhase,
        outcome: PolicyOutcome,
        action: PolicyAction,
        started_ns: int,
        backend: PolicyBackend | None = None,
    ) -> None:
        if self._decision_marks is None:
            return
        await self._decision_marks.emit(
            phase=phase,
            backend=backend,
            outcome=outcome,
            action=action,
            started_ns=started_ns,
        )

    async def _emit_text_decision(
        self,
        *,
        phase: PolicyPhase,
        outcome: _TextCheckOutcome,
        action: PolicyAction,
        started_ns: int,
    ) -> None:
        outcome_kind = (
            PolicyOutcome.COVERAGE_BLOCK
            if outcome.kind is _TextCheckKind.OVERSIZED
            else PolicyOutcome(outcome.kind.value)
        )
        await self._emit_decision(
            phase=phase,
            outcome=outcome_kind,
            action=action,
            started_ns=started_ns,
        )

    async def _emit_tool_decision(
        self,
        *,
        phase: PolicyPhase,
        verdict: ToolVerdict,
        started_ns: int,
    ) -> None:
        outcomes = {
            ToolVerdictKind.PASSED: PolicyOutcome.PASSED,
            ToolVerdictKind.POLICY_BLOCK: PolicyOutcome.POLICY_BLOCK,
            ToolVerdictKind.COVERAGE_BLOCK: PolicyOutcome.COVERAGE_BLOCK,
            ToolVerdictKind.VALIDATOR_FAILURE: PolicyOutcome.CHECK_FAILURE,
        }
        await self._emit_decision(
            phase=phase,
            backend=PolicyBackend.STRUCTURAL,
            outcome=outcomes[verdict.kind],
            action=PolicyAction.CONTINUE if verdict.passed else PolicyAction.REJECT,
            started_ns=started_ns,
        )

    async def _reject_text(
        self,
        reason: str,
        *,
        phase: PolicyPhase,
        outcome: _TextCheckOutcome,
        started_ns: int,
    ) -> NoReturn:
        await self._emit_text_decision(
            phase=phase,
            outcome=outcome,
            action=PolicyAction.REJECT,
            started_ns=started_ns,
        )
        raise _LlmPolicyError(reason) from None

    async def _reject_worker(
        self,
        reason: str,
        *,
        phase: PolicyPhase,
        outcome: PolicyOutcome = PolicyOutcome.COVERAGE_BLOCK,
        backend: PolicyBackend = PolicyBackend.WORKER,
        started_ns: int | None = None,
    ) -> NoReturn:
        await self._emit_decision(
            phase=phase,
            backend=backend,
            outcome=outcome,
            action=PolicyAction.REJECT,
            started_ns=started_ns if started_ns is not None else PolicyDecisionMarks.start(),
        )
        raise _LlmPolicyError(reason) from None

    async def close(self) -> None:
        failures: list[BaseException] = []
        if self._text_policy is not None:
            try:
                await self._text_policy.close()
            except BaseException as error:
                failures.append(error)
        if self._adapter is not None:
            try:
                await self._adapter.close()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise failures[0]

    def _request_mark_phase(self) -> PolicyPhase:
        if self._input_enabled:
            return PolicyPhase.INPUT
        if self._output_enabled:
            return PolicyPhase.OUTPUT
        return PolicyPhase.TOOL_TRAFFIC

    async def _project_and_check_results(
        self,
        request: LlmRequest,
        *,
        check_calls: bool | None = None,
        codec_name: str | None = None,
        annotated_request: dict[str, Any] | None = None,
    ) -> _DecodedToolRequest | None:
        check_calls = self._settings.check_calls if check_calls is None else check_calls
        check_semantic_history = (
            self._semantic_tool_policy is not None and self._semantic_tool_policy.settings.checks_history_results
        )
        if not self._settings.check_results and not check_calls and not check_semantic_history:
            return None
        projection_started_ns = PolicyDecisionMarks.start()
        try:
            decoded = self._projector.project_tools(
                request,
                require_definitions=check_calls,
                codec_name=codec_name,
                annotated_request=annotated_request,
            )
        except _UnsupportedRequest:
            await self._emit_decision(
                phase=PolicyPhase.TOOL_TRAFFIC,
                backend=PolicyBackend.WORKER,
                outcome=PolicyOutcome.COVERAGE_BLOCK,
                action=PolicyAction.REJECT,
                started_ns=projection_started_ns,
            )
            raise _LlmPolicyError("NeMo Guardrails cannot verify tool traffic") from None
        if self._settings.check_results and decoded.projection.has_results:
            started_ns = PolicyDecisionMarks.start()
            if self._adapter is None:
                await self._emit_decision(
                    phase=PolicyPhase.TOOL_RESULTS,
                    backend=PolicyBackend.STRUCTURAL,
                    outcome=PolicyOutcome.CHECK_FAILURE,
                    action=PolicyAction.REJECT,
                    started_ns=started_ns,
                )
                raise _LlmPolicyError("NeMo Guardrails tool-result checker is unavailable")
            try:
                verdict = await self._adapter.check_results(decoded.projection.exchanges)
            except Exception:
                await self._emit_decision(
                    phase=PolicyPhase.TOOL_RESULTS,
                    backend=PolicyBackend.STRUCTURAL,
                    outcome=PolicyOutcome.CHECK_FAILURE,
                    action=PolicyAction.REJECT,
                    started_ns=started_ns,
                )
                raise _LlmPolicyError("NeMo Guardrails tool-result check failed") from None
            await self._emit_tool_decision(
                phase=PolicyPhase.TOOL_RESULTS,
                verdict=verdict,
                started_ns=started_ns,
            )
            reason = _tool_verdict_message(verdict, output=False)
            if reason is not None:
                raise _LlmPolicyError(reason)
        if check_semantic_history and decoded.projection.has_results:
            try:
                await self._semantic_tool_policy.check_history(decoded.projection.exchanges)
            except SemanticToolPolicyError as error:
                raise _LlmPolicyError(str(error)) from None
        return decoded

    async def _project_and_check_input(
        self,
        request: LlmRequest,
        *,
        require_output_context: bool | None = None,
        codec_name: str | None = None,
        annotated_request: dict[str, Any] | None = None,
    ) -> tuple[_DecodedTextRequest | None, LlmRequest]:
        require_output_context = self._output_enabled if require_output_context is None else require_output_context
        if self._text_policy is None or (not self._input_enabled and not require_output_context):
            return None, request
        projection_started_ns = PolicyDecisionMarks.start()
        try:
            projected = self._text_policy.project_request(
                request,
                require_user=self._input_enabled,
                codec_name=codec_name,
                annotated_request=annotated_request,
            )
        except _UnsupportedRequest:
            await self._emit_decision(
                phase=PolicyPhase.INPUT if self._input_enabled else PolicyPhase.OUTPUT,
                backend=PolicyBackend.WORKER,
                outcome=PolicyOutcome.COVERAGE_BLOCK,
                action=PolicyAction.REJECT,
                started_ns=projection_started_ns,
            )
            raise _LlmPolicyError("NeMo Guardrails cannot verify this unsupported provider request") from None
        if require_output_context and (
            len(projected.messages) >= MAX_PROJECTED_MESSAGES
            or sum(len(message["content"]) for message in projected.messages) >= MAX_PROJECTED_TEXT_CHARACTERS
        ):
            await self._emit_decision(
                phase=PolicyPhase.OUTPUT,
                backend=PolicyBackend.WORKER,
                outcome=PolicyOutcome.COVERAGE_BLOCK,
                action=PolicyAction.REJECT,
                started_ns=projection_started_ns,
            )
            raise _LlmPolicyError("NeMo Guardrails cannot verify this oversized output context")
        if self._input_enabled:
            started_ns = PolicyDecisionMarks.start()
            outcome = await self._text_policy.check_input_outcome(projected)
            if outcome.kind is _TextCheckKind.MODIFIED:
                if self._mutation_policy.input is not MutationMode.APPLY:
                    await self._reject_text(
                        cast(str, self._text_policy._outcome_message(outcome, phase="input")),
                        phase=PolicyPhase.INPUT,
                        outcome=outcome,
                        started_ns=started_ns,
                    )
                transformed = outcome.transformed_content
                if transformed is None:
                    await self._reject_text(
                        "NeMo Guardrails input rewrite is unavailable",
                        phase=PolicyPhase.INPUT,
                        outcome=outcome,
                        started_ns=started_ns,
                    )
                latest_user_text = next(
                    message["content"] for message in reversed(projected.messages) if message["role"] == "user"
                )
                try:
                    rewritten = rewrite_request_text(
                        request,
                        codec_names=projected.codec_names,
                        expected=latest_user_text,
                        replacement=transformed,
                        max_text_characters=MAX_PROJECTED_TEXT_CHARACTERS,
                    )
                    rewritten_request = cast(LlmRequest, rewritten)
                    reprojected = self._text_policy.project_request(
                        rewritten_request,
                        require_user=True,
                        codec_name=codec_name,
                    )
                except (TextMutationError, _UnsupportedRequest):
                    await self._reject_text(
                        "NeMo Guardrails modified the input, but the provider request cannot be rewritten safely",
                        phase=PolicyPhase.INPUT,
                        outcome=outcome,
                        started_ns=started_ns,
                    )
                if (
                    reprojected.codec_names != projected.codec_names
                    or reprojected.codec_variants != projected.codec_variants
                    or next(
                        message["content"] for message in reversed(reprojected.messages) if message["role"] == "user"
                    )
                    != transformed
                ):
                    await self._reject_text(
                        "NeMo Guardrails modified the input, but the provider request cannot be rewritten safely",
                        phase=PolicyPhase.INPUT,
                        outcome=outcome,
                        started_ns=started_ns,
                    )
                projected = reprojected
                request = rewritten_request
                await self._emit_text_decision(
                    phase=PolicyPhase.INPUT,
                    outcome=outcome,
                    action=PolicyAction.REWRITE,
                    started_ns=started_ns,
                )
            else:
                reason = self._text_policy._outcome_message(outcome, phase="input")
                if reason is not None:
                    await self._reject_text(
                        reason,
                        phase=PolicyPhase.INPUT,
                        outcome=outcome,
                        started_ns=started_ns,
                    )
                await self._emit_text_decision(
                    phase=PolicyPhase.INPUT,
                    outcome=outcome,
                    action=PolicyAction.CONTINUE,
                    started_ns=started_ns,
                )
        return projected, request

    async def _require_projection_agreement(
        self,
        text_request: _DecodedTextRequest | None,
        tool_request: _DecodedToolRequest | None,
    ) -> None:
        if text_request is None or tool_request is None:
            return
        if (
            text_request.codec_names != tool_request.codec_names
            or text_request.codec_variants != tool_request.codec_variants
        ):
            await self._reject_worker(
                "NeMo Guardrails cannot verify the provider request",
                phase=PolicyPhase.TOOL_TRAFFIC,
            )

    async def _validate_count_tokens_response(self, response: Json) -> Json:
        input_tokens = response.get("input_tokens") if isinstance(response, dict) else None
        if (
            not isinstance(response, dict)
            or set(response) != {"input_tokens"}
            or isinstance(input_tokens, bool)
            or not isinstance(input_tokens, int)
            or input_tokens < 0
        ):
            await self._reject_worker(
                "NeMo Guardrails cannot verify the count-tokens response",
                phase=PolicyPhase.OUTPUT,
            )
        return response

    async def _check_and_rewrite_output(
        self,
        text_request: _DecodedTextRequest | None,
        response: Json,
        response_codec_name: str | None,
    ) -> Json:
        projection_started_ns = PolicyDecisionMarks.start()
        if self._text_policy is None or text_request is None:
            await self._emit_decision(
                phase=PolicyPhase.OUTPUT,
                backend=PolicyBackend.WORKER,
                outcome=PolicyOutcome.CHECK_FAILURE,
                action=PolicyAction.REJECT,
                started_ns=projection_started_ns,
            )
            raise _LlmPolicyError("NeMo Guardrails output checker is unavailable")
        try:
            output_projection = self._projector.project_response_texts(
                text_request,
                response,
                allow_tool_calls=self._settings.check_calls,
                response_codec_name=response_codec_name,
            )
        except _UnsupportedRequest:
            await self._emit_decision(
                phase=PolicyPhase.OUTPUT,
                backend=PolicyBackend.WORKER,
                outcome=PolicyOutcome.COVERAGE_BLOCK,
                action=PolicyAction.REJECT,
                started_ns=projection_started_ns,
            )
            raise _LlmPolicyError("NeMo Guardrails cannot verify provider output") from None

        checked_segments = (*output_projection.reasoning, *output_projection.candidates)
        # A function-only response has no text on which an output rail can run.
        # Structural call validation below records the decision for that turn.
        if not checked_segments:
            return response
        request_characters = sum(len(message["content"]) for message in text_request.messages)
        if (
            len(checked_segments) > MAX_OUTPUT_CHECK_SEGMENTS
            or len(text_request.messages) + 1 > MAX_PROJECTED_MESSAGES
            or request_characters * len(checked_segments) + sum(map(len, checked_segments))
            > MAX_PROJECTED_TEXT_CHARACTERS
        ):
            await self._emit_decision(
                phase=PolicyPhase.OUTPUT,
                backend=PolicyBackend.WORKER,
                outcome=PolicyOutcome.COVERAGE_BLOCK,
                action=PolicyAction.REJECT,
                started_ns=projection_started_ns,
            )
            raise _LlmPolicyError("NeMo Guardrails cannot verify this oversized output workload")
        try:
            started_ns = PolicyDecisionMarks.start()
            outcomes = await self._text_policy.check_output_outcomes(text_request, checked_segments)
        except TimeoutError:
            await self._reject_text(
                "NeMo Guardrails output checking timed out",
                phase=PolicyPhase.OUTPUT,
                outcome=_TextCheckOutcome(_TextCheckKind.TIMEOUT),
                started_ns=started_ns,
            )

        reasoning_outcomes = outcomes[: len(output_projection.reasoning)]
        candidate_outcomes = outcomes[len(output_projection.reasoning) :]
        for outcome in reasoning_outcomes:
            if outcome.kind is _TextCheckKind.MODIFIED:
                await self._reject_text(
                    "NeMo Guardrails modified reasoning output, which cannot be rewritten safely",
                    phase=PolicyPhase.OUTPUT,
                    outcome=outcome,
                    started_ns=started_ns,
                )
            reason = self._text_policy._outcome_message(outcome, phase="reasoning output")
            if reason is not None:
                await self._reject_text(
                    reason,
                    phase=PolicyPhase.OUTPUT,
                    outcome=outcome,
                    started_ns=started_ns,
                )

        replacements: list[str] = []
        modified = False
        for output_text, outcome in zip(output_projection.candidates, candidate_outcomes, strict=True):
            if outcome.kind is _TextCheckKind.MODIFIED:
                if self._mutation_policy.output is not MutationMode.APPLY:
                    await self._reject_text(
                        cast(str, self._text_policy._outcome_message(outcome, phase="output")),
                        phase=PolicyPhase.OUTPUT,
                        outcome=outcome,
                        started_ns=started_ns,
                    )
                if outcome.transformed_content is None:
                    await self._reject_text(
                        "NeMo Guardrails output rewrite is unavailable",
                        phase=PolicyPhase.OUTPUT,
                        outcome=outcome,
                        started_ns=started_ns,
                    )
                replacements.append(outcome.transformed_content)
                modified = True
                continue
            reason = self._text_policy._outcome_message(outcome, phase="output")
            if reason is not None:
                await self._reject_text(
                    reason,
                    phase=PolicyPhase.OUTPUT,
                    outcome=outcome,
                    started_ns=started_ns,
                )
            replacements.append(output_text)

        if not modified:
            await self._emit_text_decision(
                phase=PolicyPhase.OUTPUT,
                outcome=_TextCheckOutcome(_TextCheckKind.PASSED),
                action=PolicyAction.CONTINUE,
                started_ns=started_ns,
            )
            return response
        try:
            rewritten_response = rewrite_response_texts(
                response,
                codec_names=(cast(str, output_projection.codec_name),),
                codec_variants=(output_projection.codec_variant,),
                expected=output_projection.candidates,
                replacements=tuple(replacements),
                max_text_characters=MAX_PROJECTED_TEXT_CHARACTERS,
            )
            verified = self._projector.project_response_texts(
                text_request,
                rewritten_response,
                allow_tool_calls=self._settings.check_calls,
                response_codec_name=response_codec_name,
            )
        except (TextMutationError, _UnsupportedRequest):
            await self._reject_text(
                "NeMo Guardrails modified the output, but the provider response cannot be rewritten safely",
                phase=PolicyPhase.OUTPUT,
                outcome=_TextCheckOutcome(_TextCheckKind.MODIFIED),
                started_ns=started_ns,
            )
        if (
            verified.candidates != tuple(replacements)
            or verified.reasoning != output_projection.reasoning
            or verified.codec_name != output_projection.codec_name
            or verified.codec_variant != output_projection.codec_variant
        ):
            await self._reject_text(
                "NeMo Guardrails modified the output, but the provider response cannot be rewritten safely",
                phase=PolicyPhase.OUTPUT,
                outcome=_TextCheckOutcome(_TextCheckKind.MODIFIED),
                started_ns=started_ns,
            )
        await self._emit_text_decision(
            phase=PolicyPhase.OUTPUT,
            outcome=_TextCheckOutcome(_TextCheckKind.MODIFIED),
            action=PolicyAction.REWRITE,
            started_ns=started_ns,
        )
        return cast(Json, rewritten_response)

    async def _check_response_tool_calls(
        self,
        tool_request: _DecodedToolRequest | None,
        response: Json,
        response_codec_name: str | None,
    ) -> None:
        projection_started_ns = PolicyDecisionMarks.start()
        if tool_request is None or self._adapter is None:
            await self._emit_decision(
                phase=PolicyPhase.MODEL_TOOL_CALLS,
                backend=PolicyBackend.STRUCTURAL,
                outcome=PolicyOutcome.CHECK_FAILURE,
                action=PolicyAction.REJECT,
                started_ns=projection_started_ns,
            )
            raise _LlmPolicyError("NeMo Guardrails model-call checker is unavailable")
        try:
            call_candidates = self._projector.project_response_tool_candidates(
                tool_request,
                response,
                response_codec_name=response_codec_name,
            )
        except _UnsupportedRequest:
            await self._emit_decision(
                phase=PolicyPhase.MODEL_TOOL_CALLS,
                backend=PolicyBackend.WORKER,
                outcome=PolicyOutcome.COVERAGE_BLOCK,
                action=PolicyAction.REJECT,
                started_ns=projection_started_ns,
            )
            raise _LlmPolicyError("NeMo Guardrails cannot verify model tool calls") from None
        started_ns = PolicyDecisionMarks.start()
        try:
            verdict = await self._adapter.check_call_candidates(
                tool_request.projection.definitions,
                call_candidates,
            )
        except Exception:
            await self._emit_decision(
                phase=PolicyPhase.MODEL_TOOL_CALLS,
                backend=PolicyBackend.STRUCTURAL,
                outcome=PolicyOutcome.CHECK_FAILURE,
                action=PolicyAction.REJECT,
                started_ns=started_ns,
            )
            raise _LlmPolicyError("NeMo Guardrails model-call check failed") from None
        await self._emit_tool_decision(
            phase=PolicyPhase.MODEL_TOOL_CALLS,
            verdict=verdict,
            started_ns=started_ns,
        )
        reason = _tool_verdict_message(verdict, output=True)
        if reason is not None:
            raise _LlmPolicyError(reason)

    async def _execute(
        self,
        operation_name: str,
        request: LlmRequest,
        next_call: LlmNext,
        context: ExecutionCodecContext,
    ) -> Json:
        try:
            request_codec_name = context.request.supported_name("request")
        except ExecutionCodecContextError as error:
            await self._reject_worker(str(error), phase=self._request_mark_phase())
        if request_codec_name is None and not context.allow_shape_inference:
            await self._reject_worker(
                "the Relay request codec is unavailable",
                phase=self._request_mark_phase(),
            )
        checks_response = operation_name != _ANTHROPIC_COUNT_TOKENS_OPERATION
        response_codec_name: str | None = None
        if checks_response and (self._output_enabled or self._settings.check_calls):
            try:
                response_codec_name = context.response.supported_name("response")
            except ExecutionCodecContextError as error:
                await self._reject_worker(
                    str(error),
                    phase=(PolicyPhase.OUTPUT if self._output_enabled else PolicyPhase.MODEL_TOOL_CALLS),
                )
            if response_codec_name is None and not context.allow_shape_inference:
                await self._reject_worker(
                    "the Relay response codec is unavailable",
                    phase=(PolicyPhase.OUTPUT if self._output_enabled else PolicyPhase.MODEL_TOOL_CALLS),
                )
        text_request, effective_request = await self._project_and_check_input(
            request,
            require_output_context=self._output_enabled and checks_response,
            codec_name=request_codec_name,
            annotated_request=context.annotated_request,
        )
        tool_request = await self._project_and_check_results(
            effective_request,
            check_calls=self._settings.check_calls and checks_response,
            codec_name=request_codec_name,
            annotated_request=(context.annotated_request if effective_request is request else None),
        )
        await self._require_projection_agreement(text_request, tool_request)

        response = await next_call.call(effective_request)
        if not checks_response:
            return await self._validate_count_tokens_response(response)

        if self._output_enabled:
            response = await self._check_and_rewrite_output(
                text_request,
                response,
                response_codec_name,
            )
        if self._settings.check_calls:
            await self._check_response_tool_calls(
                tool_request,
                response,
                response_codec_name,
            )
        return response

    async def execute(self, operation_name: str, request: LlmRequest, next_call: LlmNext) -> Json:
        return await self._execute(operation_name, request, next_call, ExecutionCodecContext.legacy())

    async def execute_with_context(
        self,
        operation_name: str,
        request: LlmRequest,
        context: Any,
        next_call: LlmNext,
    ) -> Json:
        try:
            codec_context = execution_codec_context(context)
        except ExecutionCodecContextError as error:
            await self._reject_worker(str(error), phase=self._request_mark_phase())
        return await self._execute(operation_name, request, next_call, codec_context)

    async def _execute_stream(
        self,
        model_name: str,
        request: LlmRequest,
        next_call: LlmStreamNext,
        context: ExecutionCodecContext,
    ) -> Any:
        del model_name
        if self._output_enabled or self._settings.check_calls:
            await self._reject_worker(
                "NeMo Guardrails cannot validate a guarded streaming response",
                phase=(PolicyPhase.OUTPUT if self._output_enabled else PolicyPhase.MODEL_TOOL_CALLS),
            )
        try:
            request_codec_name = context.request.supported_name("request")
        except ExecutionCodecContextError as error:
            await self._reject_worker(str(error), phase=self._request_mark_phase())
        if request_codec_name is None and not context.allow_shape_inference:
            await self._reject_worker(
                "the Relay request codec is unavailable",
                phase=self._request_mark_phase(),
            )
        text_request, effective_request = await self._project_and_check_input(
            request,
            codec_name=request_codec_name,
            annotated_request=context.annotated_request,
        )
        tool_request = await self._project_and_check_results(
            effective_request,
            codec_name=request_codec_name,
            annotated_request=(context.annotated_request if effective_request is request else None),
        )
        await self._require_projection_agreement(text_request, tool_request)
        return next_call.call(effective_request)

    async def execute_stream(
        self,
        model_name: str,
        request: LlmRequest,
        next_call: LlmStreamNext,
    ) -> Any:
        return await self._execute_stream(model_name, request, next_call, ExecutionCodecContext.legacy())

    async def execute_stream_with_context(
        self,
        model_name: str,
        request: LlmRequest,
        context: Any,
        next_call: LlmStreamNext,
    ) -> Any:
        try:
            codec_context = execution_codec_context(context)
        except ExecutionCodecContextError as error:
            await self._reject_worker(str(error), phase=self._request_mark_phase())
        return await self._execute_stream(model_name, request, next_call, codec_context)
