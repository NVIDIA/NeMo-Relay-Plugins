# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-free observability for Guardrails policy decisions."""

from __future__ import annotations

import asyncio
import time
from enum import Enum

from nemo_relay_plugin import DataSchema, LogSeverity

POLICY_DECISION_MARK = "nemo_guardrails.policy.decision"
MAX_DURATION_MS = 2_147_483_647.0
MARK_EMIT_TIMEOUT_SECONDS = 0.25


class PolicyPhase(str, Enum):
    """Bounded policy phases emitted by this worker."""

    INPUT = "input"
    OUTPUT = "output"
    TOOL_ARGUMENTS = "tool_arguments"
    TOOL_RESULT = "tool_result"
    TOOL_HISTORY_RESULTS = "tool_history_results"
    TOOL_TRAFFIC = "tool_traffic"
    TOOL_RESULTS = "tool_results"
    MODEL_TOOL_CALLS = "model_tool_calls"


class PolicyBackend(str, Enum):
    """Bounded source that produced a policy decision."""

    LOCAL = "local"
    REMOTE = "remote"
    STRUCTURAL = "structural"
    WORKER = "worker"


class PolicyOutcome(str, Enum):
    """Content-free outcomes shared by text and tool policy adapters."""

    PASSED = "passed"
    POLICY_BLOCK = "policy_block"
    MODIFIED = "modified"
    COVERAGE_BLOCK = "coverage_block"
    TIMEOUT = "timeout"
    OVERLOADED = "overloaded"
    CHECK_FAILURE = "check_failure"
    INCONSISTENT = "inconsistent"


class PolicyAction(str, Enum):
    """The worker action taken after a policy outcome."""

    CONTINUE = "continue"
    REJECT = "reject"
    REWRITE = "rewrite"


class PolicyDecisionMarks:
    """Emit one bounded, content-free mark for each completed policy phase.

    Marks are observability only. Host-side publication failures must not alter
    a Guardrails allow, rewrite, or rejection decision. Task cancellation still
    propagates so worker shutdown and invocation cancellation remain prompt.
    """

    def __init__(self, runtime: object, *, text_backend: PolicyBackend) -> None:
        self._runtime = runtime
        self.text_backend = text_backend

    @staticmethod
    def start() -> int:
        """Return a monotonic timestamp suitable for one phase measurement."""

        return time.monotonic_ns()

    async def emit(
        self,
        *,
        phase: PolicyPhase,
        outcome: PolicyOutcome,
        action: PolicyAction,
        started_ns: int,
        backend: PolicyBackend | None = None,
    ) -> None:
        """Publish a policy mark without affecting the enforced decision."""

        emit_mark = getattr(self._runtime, "emit_mark", None)
        if not callable(emit_mark):
            return
        elapsed_ns = max(0, time.monotonic_ns() - started_ns)
        duration_ms = min(round(elapsed_ns / 1_000_000, 3), MAX_DURATION_MS)
        try:
            async with asyncio.timeout(MARK_EMIT_TIMEOUT_SECONDS):
                await emit_mark(
                    POLICY_DECISION_MARK,
                    {
                        "phase": phase.value,
                        "backend": (backend or self.text_backend).value,
                        "outcome": outcome.value,
                        "action": action.value,
                        "duration_ms": duration_ms,
                    },
                    data_schema=DataSchema(POLICY_DECISION_MARK, "1"),
                    severity=LogSeverity.INFO,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            # Policy enforcement must not depend on observability availability.
            return
