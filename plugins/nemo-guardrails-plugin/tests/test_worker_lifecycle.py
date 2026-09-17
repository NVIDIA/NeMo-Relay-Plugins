# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from worker_test_helpers import registered_worker, worker_context

from nemoguardrails_nemo_relay import (
    worker,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NO_MODEL_CONFIG = PROJECT_ROOT / "examples" / "no-model-rails"


async def test_manifest_entrypoint_serves_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    served: list[object] = []

    async def capture(plugin: object) -> None:
        served.append(plugin)

    monkeypatch.setattr(worker, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(worker, "serve_plugin", capture)
    await worker.main()

    assert len(served) == 1
    assert isinstance(served[0], worker.NeMoGuardrailsRelayWorker)


async def test_worker_close_releases_llm_framework_once(monkeypatch: pytest.MonkeyPatch) -> None:
    framework = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(worker, "get_framework", MagicMock(return_value=framework))
    plugin, _ = await registered_worker({"config_path": str(NO_MODEL_CONFIG)})

    await plugin.close()
    await plugin.close()

    framework.aclose.assert_awaited_once_with()
    assert plugin._rails is None
    assert plugin._text_policy is None


async def test_worker_close_resets_custom_framework_without_aclose(monkeypatch: pytest.MonkeyPatch) -> None:
    framework = SimpleNamespace(reset=AsyncMock())
    monkeypatch.setattr(worker, "get_framework", MagicMock(return_value=framework))
    plugin, _ = await registered_worker({"config_path": str(NO_MODEL_CONFIG)})

    await plugin.close()
    await plugin.close()

    framework.reset.assert_awaited_once_with()


async def test_secret_environment_is_available_for_worker_lifecycle_and_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "NEMO_RELAY_GUARDRAILS_TEST_SECRET"
    monkeypatch.setenv(name, "prior-value")
    observed: list[str | None] = []

    async def close_framework() -> None:
        observed.append(os.environ.get(name))

    framework = SimpleNamespace(aclose=AsyncMock(side_effect=close_framework))
    monkeypatch.setattr(worker, "get_framework", MagicMock(return_value=framework))
    plugin, _ = await registered_worker(
        {
            "config_path": str(NO_MODEL_CONFIG),
            "secret_env": {name: "configured-value"},
        }
    )

    assert os.environ[name] == "configured-value"
    await plugin.close()

    assert observed == ["configured-value"]
    assert os.environ[name] == "prior-value"


async def test_failed_registration_restores_secret_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "NEMO_RELAY_GUARDRAILS_TEST_SECRET"
    secret = "PRIVATE_CONFIGURED_VALUE"
    monkeypatch.delenv(name, raising=False)
    framework = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(worker, "get_framework", MagicMock(return_value=framework))
    monkeypatch.setattr(worker, "Guardrails", MagicMock(side_effect=RuntimeError(secret)))
    plugin = worker.NeMoGuardrailsRelayWorker()

    with pytest.raises(ValueError, match="runtime could not be initialized") as captured:
        await plugin.register(
            worker_context(),
            {
                "config_path": str(NO_MODEL_CONFIG),
                "secret_env": {name: secret},
            },
        )

    assert secret not in str(captured.value)
    assert name not in os.environ
    framework.aclose.assert_awaited_once_with()


async def test_close_failure_still_restores_secret_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "NEMO_RELAY_GUARDRAILS_TEST_SECRET"
    monkeypatch.setenv(name, "prior-value")
    framework = SimpleNamespace(aclose=AsyncMock(side_effect=RuntimeError("PRIVATE_CLOSE_FAILURE")))
    monkeypatch.setattr(worker, "get_framework", MagicMock(return_value=framework))
    plugin, _ = await registered_worker(
        {
            "config_path": str(NO_MODEL_CONFIG),
            "secret_env": {name: "configured-value"},
        }
    )

    with pytest.raises(RuntimeError, match="resources could not be closed") as captured:
        await plugin.close()

    assert "PRIVATE_CLOSE_FAILURE" not in str(captured.value)
    assert os.environ[name] == "prior-value"


async def test_cancelled_registration_restores_secret_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "NEMO_RELAY_GUARDRAILS_TEST_SECRET"
    monkeypatch.delenv(name, raising=False)
    context = worker_context()
    context.runtime.list_runtime_registrations.side_effect = asyncio.CancelledError()
    plugin = worker.NeMoGuardrailsRelayWorker()

    with pytest.raises(asyncio.CancelledError):
        await plugin.register(
            context,
            {
                "config_path": str(NO_MODEL_CONFIG),
                "secret_env": {name: "configured-value"},
            },
        )

    assert name not in os.environ


async def test_worker_closes_framework_selected_by_trusted_config(monkeypatch: pytest.MonkeyPatch) -> None:
    built_in = SimpleNamespace(aclose=AsyncMock())
    selected = SimpleNamespace(aclose=AsyncMock())
    default_name = "default"
    real_guardrails = worker.Guardrails

    def construct_guardrails(*args: object, **kwargs: object) -> object:
        nonlocal default_name
        default_name = "custom"
        return real_guardrails(*args, **kwargs)

    monkeypatch.setattr(worker, "Guardrails", construct_guardrails)
    monkeypatch.setattr(worker, "get_default_framework", lambda: default_name)
    monkeypatch.setattr(worker, "get_framework", lambda name: {"default": built_in, "custom": selected}[name])
    plugin, _ = await registered_worker({"config_path": str(NO_MODEL_CONFIG)})

    await plugin.close()

    selected.aclose.assert_awaited_once_with()
    built_in.aclose.assert_not_awaited()


async def test_failed_input_runtime_initialization_closes_framework(monkeypatch: pytest.MonkeyPatch) -> None:
    framework = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(worker, "get_framework", MagicMock(return_value=framework))
    monkeypatch.setattr(worker, "Guardrails", MagicMock(side_effect=RuntimeError("PRIVATE_FAILURE")))
    plugin = worker.NeMoGuardrailsRelayWorker()

    with pytest.raises(ValueError, match="runtime could not be initialized") as captured:
        await plugin.register(worker_context(), {"config_path": str(NO_MODEL_CONFIG)})

    assert "PRIVATE_FAILURE" not in str(captured.value)
    framework.aclose.assert_awaited_once_with()


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("fail", [False, True])
async def test_worker_main_always_closes_plugin(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    fail: bool,
) -> None:
    plugin = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(worker, "NeMoGuardrailsRelayWorker", MagicMock(return_value=plugin))
    monkeypatch.setattr(worker, "sys", SimpleNamespace(platform=platform))
    if platform == "win32":
        monkeypatch.setattr(
            worker,
            "signal",
            SimpleNamespace(SIGINT=2, SIG_IGN=1, signal=MagicMock(return_value=object())),
        )

    async def serve(_plugin: object) -> None:
        assert _plugin is plugin
        if fail:
            raise RuntimeError("worker failed")

    monkeypatch.setattr(worker, "serve_plugin", serve)

    if fail:
        with pytest.raises(RuntimeError, match="worker failed"):
            await worker.main()
    else:
        await worker.main()

    plugin.close.assert_awaited_once_with()


@pytest.mark.parametrize("fail", [False, True])
async def test_windows_worker_leaves_shutdown_to_relay(
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    previous_handler = object()
    signals = SimpleNamespace(SIGINT=2, SIG_IGN=1, signal=MagicMock(return_value=previous_handler))
    monkeypatch.setattr(worker, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(worker, "signal", signals)

    async def serve(_plugin: object) -> None:
        signals.signal.assert_called_once_with(signals.SIGINT, signals.SIG_IGN)
        if fail:
            raise RuntimeError("worker failed")

    monkeypatch.setattr(worker, "serve_plugin", serve)
    if fail:
        with pytest.raises(RuntimeError, match="worker failed"):
            await worker.main()
    else:
        await worker.main()

    assert signals.signal.call_args_list == [
        call(signals.SIGINT, signals.SIG_IGN),
        call(signals.SIGINT, previous_handler),
    ]


async def test_unix_worker_preserves_signal_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    signals = MagicMock()
    monkeypatch.setattr(worker, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(worker, "signal", signals)
    monkeypatch.setattr(worker, "serve_plugin", AsyncMock())

    await worker.main()

    signals.signal.assert_not_called()
    worker.serve_plugin.assert_awaited_once()
