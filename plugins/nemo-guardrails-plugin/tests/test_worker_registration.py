# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from registration_helpers import registered_llm_execution
from worker_test_helpers import registered_worker, worker_context

from nemoguardrails_nemo_relay import (
    worker,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NO_MODEL_CONFIG = PROJECT_ROOT / "examples" / "no-model-rails"


async def test_registration_rejects_colang_2(tmp_path: Path) -> None:
    config_path = tmp_path / "colang-2"
    config_path.mkdir()
    (config_path / "config.yml").write_text('colang_version: "2.x"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Colang 1.0"):
        await worker.NeMoGuardrailsRelayWorker().register(
            worker_context(),
            {"config_path": str(config_path)},
        )


async def test_registration_accepts_output_only_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "output-only"
    config_path.mkdir()
    (config_path / "config.yml").write_text(
        "rails:\n  output:\n    flows:\n      - output rail\n",
        encoding="utf-8",
    )
    (config_path / "rails.co").write_text(
        'define flow output rail\n  if $bot_message == "blocked"\n    bot refuse to respond\n    stop\n',
        encoding="utf-8",
    )

    context = worker_context()
    plugin = worker.NeMoGuardrailsRelayWorker()
    await plugin.register(context, {"config_path": str(config_path)})

    assert plugin._rails is not None
    assert plugin._text_policy is not None
    assert registered_llm_execution(context).call.args[0] == "guardrails"
    context.register_llm_conditional_execution_guardrail.assert_not_called()
    await plugin.close()


async def test_registration_rejects_guardrails_exception_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "exception-mode"
    shutil.copytree(NO_MODEL_CONFIG, config_path)
    config_file = config_path / "config.yml"
    config_file.write_text(
        "enable_rails_exceptions: true\n" + config_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="enable_rails_exceptions"):
        await worker.NeMoGuardrailsRelayWorker().register(
            worker_context(),
            {"config_path": str(config_path)},
        )


async def test_registration_rejects_invalid_guardrails_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid"
    config_path.mkdir()
    (config_path / "config.yml").write_text("rails: [\n", encoding="utf-8")

    with pytest.raises(Exception):
        await worker.NeMoGuardrailsRelayWorker().register(
            worker_context(),
            {"config_path": str(config_path)},
        )


async def test_registration_hides_config_py_initialization_details(tmp_path: Path) -> None:
    config_path = tmp_path / "failing-config-py"
    shutil.copytree(NO_MODEL_CONFIG, config_path)
    canary = "PRIVATE_CONFIG_INITIALIZATION_CANARY_4557"
    (config_path / "config.py").write_text(
        f"raise RuntimeError({canary!r})\n",
        encoding="utf-8",
    )

    previous_global_log_disable = logging.root.manager.disable
    try:
        with pytest.raises(ValueError, match="Guardrails runtime could not be initialized") as error:
            await worker.NeMoGuardrailsRelayWorker().register(
                worker_context(),
                {"config_path": str(config_path)},
            )
    finally:
        assert logging.root.manager.disable == previous_global_log_disable

    assert canary not in str(error.value)
    assert str(config_path) not in str(error.value)


async def test_registration_rejects_deprecated_builtin_coexistence() -> None:
    legacy_registration = SimpleNamespace(owner=SimpleNamespace(plugin_kind="nemo_guardrails"))
    context = worker_context([legacy_registration])

    with pytest.raises(ValueError, match="cannot run beside Relay's deprecated nemo_guardrails component"):
        await worker.NeMoGuardrailsRelayWorker().register(
            context,
            {"config_path": str(NO_MODEL_CONFIG)},
        )

    context.register_llm_conditional_execution_guardrail.assert_not_called()


async def test_registration_hides_runtime_inventory_failures() -> None:
    context = worker_context()
    context.runtime.list_runtime_registrations.side_effect = RuntimeError("PRIVATE_RUNTIME_CANARY")

    with pytest.raises(ValueError, match="Relay runtime registrations could not be inspected") as error:
        await worker.NeMoGuardrailsRelayWorker().register(
            context,
            {"config_path": str(NO_MODEL_CONFIG)},
        )

    assert "PRIVATE_RUNTIME_CANARY" not in str(error.value)
    context.register_llm_conditional_execution_guardrail.assert_not_called()


async def test_registration_suppresses_guardrails_construction_logs(tmp_path: Path) -> None:
    config_path = tmp_path / "construction-log"
    shutil.copytree(NO_MODEL_CONFIG, config_path)
    canary = "PRIVATE_GUARDRAILS_CONSTRUCTION_LOG_CANARY_3379"
    (config_path / "config.py").write_text(
        f"import logging\nlogging.getLogger('nemoguardrails.actions.action_dispatcher').error({canary!r})\n",
        encoding="utf-8",
    )
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        await registered_worker({"config_path": str(config_path), "check_timeout_ms": 1_000})
    finally:
        root_logger.removeHandler(handler)

    assert canary not in stream.getvalue()
