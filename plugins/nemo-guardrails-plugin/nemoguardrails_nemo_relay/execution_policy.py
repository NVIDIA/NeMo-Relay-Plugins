# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM execution orchestration for the Relay Guardrails worker."""

from __future__ import annotations

from typing import Any, NoReturn, cast

from nemo_relay_plugin import (  # noqa: E402
    AnnotatedLlmRequest,
    Json,
    LlmExecutionContext,
    LlmNext,
    LlmRequest,
    LlmStreamNext,
    WorkerRequestCodec,
    WorkerResponseCodec,
)

from .codec_projection import (  # noqa: E402
    MAX_RESPONSE_SEGMENTS,
    _DecodedTextRequest,
    _DecodedToolRequest,
    _ProviderProjector,
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
from .provider_coverage.common import ResponseInspection  # noqa: E402
from .semantic_tools import (  # noqa: E402
    SemanticToolPolicy,
    SemanticToolPolicyError,
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
    rewrite_annotated_request_text,
    rewrite_response_texts,
)
from .text_policy import (  # noqa: E402
    MAX_PROJECTED_MESSAGES,
    MAX_PROJECTED_TEXT_CHARACTERS,
    _TextCheckKind,
    _TextCheckOutcome,
    _TextRailsPolicy,
)

MAX_OUTPUT_CHECK_SEGMENTS = MAX_RESPONSE_SEGMENTS
_ANTHROPIC_COUNT_TOKENS_OPERATION = "anthropic.count_tokens"


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
        projector: _ProviderProjector,
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
        annotated_request: AnnotatedLlmRequest,
        *,
        check_calls: bool | None = None,
        codec_name: str | None = None,
    ) -> _DecodedToolRequest | None:
        check_calls = self._settings.check_calls if check_calls is None else check_calls
        check_semantic_history = (
            self._semantic_tool_policy is not None and self._semantic_tool_policy.settings.checks_history_results
        )
        if not self._settings.check_results and not check_calls and not check_semantic_history:
            return None
        projection_started_ns = PolicyDecisionMarks.start()
        try:
            decoded = self._projector.project_host_tools(
                request,
                annotated_request,
                require_definitions=check_calls,
                codec_name=codec_name,
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
        annotated_request: AnnotatedLlmRequest,
        request_codec: WorkerRequestCodec,
        *,
        require_output_context: bool | None = None,
        codec_name: str | None = None,
    ) -> tuple[_DecodedTextRequest | None, LlmRequest, AnnotatedLlmRequest]:
        require_output_context = self._output_enabled if require_output_context is None else require_output_context
        if self._text_policy is None or (not self._input_enabled and not require_output_context):
            return None, request, annotated_request
        projection_started_ns = PolicyDecisionMarks.start()
        try:
            projected = self._text_policy.project_host_request(
                request,
                annotated_request,
                require_user=self._input_enabled,
                codec_name=codec_name,
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
                    rewritten_annotation = rewrite_annotated_request_text(
                        annotated_request,
                        expected=latest_user_text,
                        replacement=transformed,
                        max_text_characters=MAX_PROJECTED_TEXT_CHARACTERS,
                    )
                    rewritten_request = await request_codec.encode(rewritten_annotation, request)
                    if not isinstance(rewritten_request, dict):
                        raise TextMutationError("Relay returned an invalid rewritten request")
                    decoded_rewrite = await request_codec.decode(rewritten_request)
                    if not isinstance(decoded_rewrite, dict):
                        raise TextMutationError("Relay returned an invalid rewritten annotation")
                    reprojected = self._text_policy.project_host_request(
                        rewritten_request,
                        decoded_rewrite,
                        require_user=True,
                        codec_name=codec_name,
                    )
                # Codec RPC failures, unsafe leaf mappings, and a failed
                # re-projection all have the same content-free policy result.
                except Exception:
                    await self._reject_text(
                        "NeMo Guardrails modified the input, but the provider request cannot be rewritten safely",
                        phase=PolicyPhase.INPUT,
                        outcome=outcome,
                        started_ns=started_ns,
                    )
                if (
                    reprojected.codec_name != projected.codec_name
                    or reprojected.codec_variant != projected.codec_variant
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
                annotated_request = cast(AnnotatedLlmRequest, decoded_rewrite)
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
        return projected, request, annotated_request

    async def _require_projection_agreement(
        self,
        text_request: _DecodedTextRequest | None,
        tool_request: _DecodedToolRequest | None,
    ) -> None:
        if text_request is None or tool_request is None:
            return
        if (
            text_request.codec_name != tool_request.codec_name
            or text_request.codec_variant != tool_request.codec_variant
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
        inspection: ResponseInspection,
        response_codec_name: str,
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
        if inspection.has_calls and not self._settings.check_calls:
            await self._emit_decision(
                phase=PolicyPhase.OUTPUT,
                backend=PolicyBackend.WORKER,
                outcome=PolicyOutcome.COVERAGE_BLOCK,
                action=PolicyAction.REJECT,
                started_ns=projection_started_ns,
            )
            raise _LlmPolicyError("NeMo Guardrails cannot verify provider output") from None

        checked_segments = (*inspection.reasoning, *inspection.texts)
        # A function-only response has no text on which an output rail can run.
        # Structural call validation below records the decision for that turn.
        if not checked_segments:
            if inspection.has_calls:
                return response
            await self._emit_decision(
                phase=PolicyPhase.OUTPUT,
                backend=PolicyBackend.WORKER,
                outcome=PolicyOutcome.COVERAGE_BLOCK,
                action=PolicyAction.REJECT,
                started_ns=projection_started_ns,
            )
            raise _LlmPolicyError("NeMo Guardrails cannot verify provider output")
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

        reasoning_outcomes = outcomes[: len(inspection.reasoning)]
        candidate_outcomes = outcomes[len(inspection.reasoning) :]
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
        for output_text, outcome in zip(inspection.texts, candidate_outcomes, strict=True):
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
                inspection=inspection,
                expected=inspection.texts,
                replacements=tuple(replacements),
                max_text_characters=MAX_PROJECTED_TEXT_CHARACTERS,
            )
            verified = self._projector.inspect_response(
                text_request,
                rewritten_response,
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
            verified.texts != tuple(replacements)
            or verified.reasoning != inspection.reasoning
            or verified.codec_name != inspection.codec_name
            or verified.codec_variant != inspection.codec_variant
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
        inspection: ResponseInspection,
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
        started_ns = PolicyDecisionMarks.start()
        try:
            verdict = await self._adapter.check_call_candidates(
                tool_request.projection.definitions,
                inspection.tool_candidates,
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
            request_codec_name = context.request.builtin_name("request")
        except ExecutionCodecContextError as error:
            await self._reject_worker(str(error), phase=self._request_mark_phase())
        try:
            annotated_request = await context.request_codec.decode(request)
        except Exception:
            await self._reject_worker(
                "the Relay request codec could not decode the provider request",
                phase=self._request_mark_phase(),
                outcome=PolicyOutcome.CHECK_FAILURE,
            )
        if not isinstance(annotated_request, dict):
            await self._reject_worker(
                "the Relay request codec returned an invalid normalized request",
                phase=self._request_mark_phase(),
                outcome=PolicyOutcome.CHECK_FAILURE,
            )
        checks_response = operation_name != _ANTHROPIC_COUNT_TOKENS_OPERATION
        response_codec_name: str | None = None
        response_codec: WorkerResponseCodec | None = None
        if checks_response and (self._output_enabled or self._settings.check_calls):
            try:
                response_codec_name = context.response.builtin_name("response")
            except ExecutionCodecContextError as error:
                await self._reject_worker(
                    str(error),
                    phase=(PolicyPhase.OUTPUT if self._output_enabled else PolicyPhase.MODEL_TOOL_CALLS),
                )
            if response_codec_name is None:
                await self._reject_worker(
                    "the Relay response codec cannot prove complete output coverage",
                    phase=(PolicyPhase.OUTPUT if self._output_enabled else PolicyPhase.MODEL_TOOL_CALLS),
                )
            response_codec = context.response_codec
            if response_codec is None:
                await self._reject_worker(
                    "the Relay response codec capability is unavailable",
                    phase=(PolicyPhase.OUTPUT if self._output_enabled else PolicyPhase.MODEL_TOOL_CALLS),
                )
        text_request, effective_request, effective_annotation = await self._project_and_check_input(
            request,
            cast(AnnotatedLlmRequest, annotated_request),
            context.request_codec,
            require_output_context=self._output_enabled and checks_response,
            codec_name=request_codec_name,
        )
        tool_request = await self._project_and_check_results(
            effective_request,
            effective_annotation,
            check_calls=self._settings.check_calls and checks_response,
            codec_name=request_codec_name,
        )
        await self._require_projection_agreement(text_request, tool_request)

        response = await next_call.call(effective_request)
        if not checks_response:
            return await self._validate_count_tokens_response(response)

        if response_codec is not None:
            try:
                annotated_response = await response_codec.decode(response)
            except Exception:
                await self._reject_worker(
                    "the Relay response codec could not decode provider output",
                    phase=(PolicyPhase.OUTPUT if self._output_enabled else PolicyPhase.MODEL_TOOL_CALLS),
                    outcome=PolicyOutcome.CHECK_FAILURE,
                )
            if not isinstance(annotated_response, dict):
                await self._reject_worker(
                    "the Relay response codec returned invalid normalized output",
                    phase=(PolicyPhase.OUTPUT if self._output_enabled else PolicyPhase.MODEL_TOOL_CALLS),
                    outcome=PolicyOutcome.CHECK_FAILURE,
                )

        inspection: ResponseInspection | None = None
        if response_codec_name is not None:
            projection_request = text_request or tool_request
            if projection_request is None:
                await self._reject_worker(
                    "NeMo Guardrails output projection is unavailable",
                    phase=(PolicyPhase.OUTPUT if self._output_enabled else PolicyPhase.MODEL_TOOL_CALLS),
                    outcome=PolicyOutcome.CHECK_FAILURE,
                )
            try:
                inspection = self._projector.inspect_response(
                    projection_request,
                    response,
                    response_codec_name=response_codec_name,
                    require_text_coverage=self._output_enabled,
                )
            except _UnsupportedRequest:
                message = (
                    "NeMo Guardrails cannot verify provider output"
                    if self._output_enabled
                    else "NeMo Guardrails cannot verify model tool calls"
                )
                await self._reject_worker(
                    message,
                    phase=(PolicyPhase.OUTPUT if self._output_enabled else PolicyPhase.MODEL_TOOL_CALLS),
                )

        if self._output_enabled:
            if inspection is None or response_codec_name is None:
                await self._reject_worker(
                    "NeMo Guardrails output projection is unavailable",
                    phase=PolicyPhase.OUTPUT,
                    outcome=PolicyOutcome.CHECK_FAILURE,
                )
            response = await self._check_and_rewrite_output(
                text_request,
                response,
                inspection,
                response_codec_name,
            )
        if self._settings.check_calls:
            if inspection is None:
                await self._reject_worker(
                    "NeMo Guardrails model-call projection is unavailable",
                    phase=PolicyPhase.MODEL_TOOL_CALLS,
                    outcome=PolicyOutcome.CHECK_FAILURE,
                )
            await self._check_response_tool_calls(
                tool_request,
                inspection,
            )
        return response

    async def execute_with_context(
        self,
        operation_name: str,
        request: LlmRequest,
        context: LlmExecutionContext,
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
            request_codec_name = context.request.builtin_name("request")
        except ExecutionCodecContextError as error:
            await self._reject_worker(str(error), phase=self._request_mark_phase())
        try:
            annotated_request = await context.request_codec.decode(request)
        except Exception:
            await self._reject_worker(
                "the Relay request codec could not decode the provider request",
                phase=self._request_mark_phase(),
                outcome=PolicyOutcome.CHECK_FAILURE,
            )
        if not isinstance(annotated_request, dict):
            await self._reject_worker(
                "the Relay request codec returned an invalid normalized request",
                phase=self._request_mark_phase(),
                outcome=PolicyOutcome.CHECK_FAILURE,
            )
        text_request, effective_request, effective_annotation = await self._project_and_check_input(
            request,
            cast(AnnotatedLlmRequest, annotated_request),
            context.request_codec,
            codec_name=request_codec_name,
        )
        tool_request = await self._project_and_check_results(
            effective_request,
            effective_annotation,
            codec_name=request_codec_name,
        )
        await self._require_projection_agreement(text_request, tool_request)
        return next_call.call(effective_request)

    async def execute_stream_with_context(
        self,
        model_name: str,
        request: LlmRequest,
        context: LlmExecutionContext,
        next_call: LlmStreamNext,
    ) -> Any:
        try:
            codec_context = execution_codec_context(context)
        except ExecutionCodecContextError as error:
            await self._reject_worker(str(error), phase=self._request_mark_phase())
        return await self._execute_stream(model_name, request, next_call, codec_context)
