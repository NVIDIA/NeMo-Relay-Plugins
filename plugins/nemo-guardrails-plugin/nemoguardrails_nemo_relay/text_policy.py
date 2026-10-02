# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded text-rail checks for the Relay Guardrails worker."""

from __future__ import annotations

import asyncio
import contextvars
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Keep imports of this module safe even when tooling bypasses ``worker``.
os.environ["NEMO_GUARDRAILS_NO_USAGE_STATS"] = "1"

from nemo_relay_plugin import AnnotatedLlmRequest, LlmRequest  # noqa: E402
from nemoguardrails import Guardrails  # noqa: E402
from nemoguardrails.rails.llm.options import RailType  # noqa: E402

from .check_backend import CheckBackend, CheckStatus, LocalCheckBackend  # noqa: E402
from .codec_projection import (  # noqa: E402
    _DecodedTextRequest,
    _ProviderProjector,
    _UnsupportedRequest,
)
from .runtime_selection import runtime_uses_explain_state  # noqa: E402
from .semantic_tools import SemanticToolVerdict, SemanticToolVerdictKind  # noqa: E402

MAX_PENDING_CHECKS = 32
MAX_PROJECTED_MESSAGES = 512
MAX_PROJECTED_TEXT_CHARACTERS = 1_000_000
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
        projector: _ProviderProjector | None = None,
        allow_structural_tools: bool = False,
        allow_tool_results: bool = False,
    ) -> None:
        self._rails = rails
        if backend is None:
            if rails is None:
                raise ValueError("a local Guardrails engine or check backend is required")
            backend = LocalCheckBackend(rails, RailType)
        self._backend = backend
        self._projector = projector or _ProviderProjector()
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

    def project_host_request(
        self,
        request: LlmRequest,
        annotated_request: AnnotatedLlmRequest,
        *,
        require_user: bool,
        codec_name: str | None,
    ) -> _DecodedTextRequest:
        """Project the authoritative annotation decoded by Relay's active codec."""

        if self._pending >= MAX_PENDING_CHECKS:
            raise _UnsupportedRequest("the Guardrails text-check queue is full")
        return self._projector.project_host_text(
            request,
            annotated_request,
            allow_structural_tools=self._allow_structural_tools,
            allow_tool_results=self._allow_tool_results,
            require_user=require_user,
            codec_name=codec_name,
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
