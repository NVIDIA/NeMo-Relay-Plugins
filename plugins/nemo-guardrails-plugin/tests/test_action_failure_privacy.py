# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest
from worker_test_helpers import registered_worker, worker_context

from nemoguardrails_nemo_relay import (
    worker,
)


def _request(messages: list[dict[str, object]], **content: object) -> dict[str, object]:
    return {
        "headers": {"authorization": "not-forwarded-to-guardrails"},
        "content": {
            "model": "fixture-model",
            "messages": messages,
            **content,
        },
    }


async def test_guardrails_action_errors_do_not_log_request_content(tmp_path: Path) -> None:
    config_path = tmp_path / "action-error"
    config_path.mkdir()
    (config_path / "config.yml").write_text(
        "rails:\n  input:\n    flows:\n      - input rail\n",
        encoding="utf-8",
    )
    (config_path / "rails.co").write_text(
        """
define bot refuse
  "blocked"

define flow input rail
  $allowed = execute exploding_action(text=$user_message)
  if not $allowed
    bot refuse
    stop
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (config_path / "config.py").write_text(
        """
from nemoguardrails.actions import action

@action(name="exploding_action")
def exploding_action(text: str) -> bool:
    raise RuntimeError(f"failed for {text}")

def init(app) -> None:
    app.register_action(exploding_action)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    rejected = worker.NeMoGuardrailsRelayWorker()
    with pytest.raises(ValueError, match="action safety validation failed"):
        await rejected.register(
            worker_context(),
            {"config_path": str(config_path), "check_timeout_ms": 1_000},
        )

    plugin, callback = await registered_worker(
        {
            "config_path": str(config_path),
            "check_timeout_ms": 1_000,
            "action_safety": {"synchronous": "allow_unsafe"},
        }
    )
    canary = "CANARY_PRIVATE_ACTION_7843"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        result = await callback(_request([{"role": "user", "content": canary}]))
    finally:
        root_logger.removeHandler(handler)

    assert result == "NeMo Guardrails prevented execution in input rail 'input rail'"
    assert canary not in stream.getvalue()
    assert plugin._rails is not None
    assert plugin._rails.explain().llm_calls == []
    assert plugin._rails.explain().colang_history is None


async def test_missing_guardrails_action_fails_closed_without_logging_content(tmp_path: Path) -> None:
    config_path = tmp_path / "missing-action"
    config_path.mkdir()
    (config_path / "config.yml").write_text(
        "rails:\n  input:\n    flows:\n      - input rail\n",
        encoding="utf-8",
    )
    (config_path / "rails.co").write_text(
        """
define bot refuse
  "blocked"

define flow input rail
  $allowed = execute action_that_is_not_registered(text=$user_message)
  if not $allowed
    bot refuse
    stop
""".strip()
        + "\n",
        encoding="utf-8",
    )
    plugin, callback = await registered_worker({"config_path": str(config_path), "check_timeout_ms": 1_000})
    canary = "CANARY_PRIVATE_MISSING_ACTION_9321"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        result = await callback(_request([{"role": "user", "content": canary}]))
    finally:
        root_logger.removeHandler(handler)

    # LLMRails maps a missing action to a blocking rail result. Guardrails uses
    # different free-form content for this operational error, but RailsResult
    # has no typed error discriminator. The worker deliberately does not expose
    # or match content that may contain action details.
    assert result == "NeMo Guardrails prevented execution in input rail 'input rail'"
    assert canary not in stream.getvalue()
    assert plugin._rails is not None
    assert plugin._rails.explain().llm_calls == []
    assert plugin._rails.explain().colang_history is None
