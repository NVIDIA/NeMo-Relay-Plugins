# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared worker registration helpers used by focused test modules."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from nemo_relay_plugin import PluginContext
from registration_helpers import registered_llm_execution

from nemoguardrails_nemo_relay import execution_policy, worker


def worker_context(registrations: list[object] | None = None) -> MagicMock:
    """Create a Relay plugin context with a controllable runtime inventory."""

    context = MagicMock(spec=PluginContext)
    context.runtime = SimpleNamespace(
        list_runtime_registrations=AsyncMock(return_value=[] if registrations is None else registrations)
    )
    return context


async def registered_worker(config: dict[str, object]) -> tuple[worker.NeMoGuardrailsRelayWorker, object]:
    """Register a worker and return its input-oriented compatibility callback."""

    context = worker_context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, config)
    registration = registered_llm_execution(context)
    call = registration.call
    assert call.args[0] == "guardrails"
    assert call.kwargs == {"priority": worker.OUTERMOST_EXECUTION_PRIORITY}
    context.register_llm_conditional_execution_guardrail.assert_not_called()

    async def input_callback(request: object) -> str | None:
        next_call = SimpleNamespace(call=AsyncMock(return_value={"choices": []}))
        try:
            await registration.callback("fixture", request, next_call)
        except execution_policy._LlmPolicyError as exc:
            return str(exc)
        return None

    return plugin, input_callback
