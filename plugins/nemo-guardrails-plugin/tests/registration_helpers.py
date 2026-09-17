# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test adapters for Relay's additive execution-context registrations."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Awaitable, Callable
from unittest.mock import MagicMock

LegacyExecutionCallback = Callable[[str, object, object], Awaitable[object]]


@dataclass(frozen=True)
class RegisteredLlmExecution:
    """One registered execution callback presented through the legacy test shape."""

    callback: LegacyExecutionCallback
    call: Any
    uses_context: bool


def _none_execution_context() -> SimpleNamespace:
    """Return the valid no-codec context used by compatibility behavior tests."""

    direction = SimpleNamespace(codec=SimpleNamespace(kind="none", id=None))
    return SimpleNamespace(
        request=direction,
        response=direction,
        annotated_request=None,
    )


def llm_registration_mock(context: MagicMock, *, stream: bool = False) -> MagicMock:
    """Return the registration method production code will use for this SDK."""

    suffix = "stream_execution" if stream else "execution"
    contextual = getattr(context, f"register_llm_{suffix}_intercept_with_context", None)
    if callable(contextual):
        return contextual
    return getattr(context, f"register_llm_{suffix}_intercept")


def registered_llm_execution(
    context: MagicMock,
    *,
    stream: bool = False,
) -> RegisteredLlmExecution:
    """Find the active registration and adapt its callback to three arguments."""

    suffix = "stream_execution" if stream else "execution"
    legacy = getattr(context, f"register_llm_{suffix}_intercept")
    contextual = getattr(context, f"register_llm_{suffix}_intercept_with_context", None)
    if callable(contextual) and contextual.called:
        if legacy.called:
            raise AssertionError("worker registered both legacy and context-aware LLM callbacks")
        call = contextual.call_args
        registered_callback = call.args[1]

        async def callback(operation: str, request: object, next_call: object) -> Any:
            return await registered_callback(
                operation,
                request,
                _none_execution_context(),
                next_call,
            )

        return RegisteredLlmExecution(callback, call, True)
    if legacy.called:
        if callable(contextual) and contextual.called:
            raise AssertionError("worker registered both legacy and context-aware LLM callbacks")
        call = legacy.call_args
        return RegisteredLlmExecution(call.args[1], call, False)
    raise AssertionError("worker did not register the expected LLM execution callback")
