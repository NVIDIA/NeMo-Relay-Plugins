# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from nemoguardrails import RailsConfig

from nemoguardrails_nemo_relay import integration_safety
from nemoguardrails_nemo_relay.integration_safety import (
    OptionalIntegrationSafetyError,
    enforce_guardrails_ai_privacy,
    guardrails_ai_privacy_issue,
    uses_guardrails_ai,
)


def _config(tmp_path: Path, flow: str) -> RailsConfig:
    path = tmp_path / "guardrails"
    path.mkdir()
    (path / "config.yml").write_text(
        "rails:\n  input:\n    flows:\n      - " + flow + "\n",
        encoding="utf-8",
    )
    return RailsConfig.from_path(str(path))


def _fake_import(monkeypatch: pytest.MonkeyPatch, *, enable_metrics: bool | None) -> SimpleNamespace:
    rc = SimpleNamespace(enable_metrics=enable_metrics, token="kept")
    settings = SimpleNamespace(rc=SimpleNamespace(enable_metrics=True), disable_tracing=None)

    def load(name: str) -> SimpleNamespace:
        if name == "guardrails.classes.rc":
            return SimpleNamespace(RC=SimpleNamespace(load=lambda: rc))
        if name == "guardrails.settings":
            return SimpleNamespace(settings=settings)
        raise AssertionError(name)

    monkeypatch.setattr(integration_safety.importlib, "import_module", load)
    return settings


def test_only_selected_guardrails_ai_surfaces_trigger_the_gate(tmp_path: Path) -> None:
    assert uses_guardrails_ai(_config(tmp_path, 'guardrailsai check input $validator="regex_match"'))

    other = tmp_path / "other"
    other.mkdir()
    (other / "config.yml").write_text(
        "rails:\n  input:\n    flows:\n      - regex check input\n",
        encoding="utf-8",
    )
    assert not uses_guardrails_ai(RailsConfig.from_path(str(other)))


def test_guardrails_ai_requires_the_persisted_metrics_opt_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rails_config = _config(tmp_path, 'guardrailsai check input $validator="regex_match"')
    _fake_import(monkeypatch, enable_metrics=True)

    assert guardrails_ai_privacy_issue(rails_config) == (
        "Guardrails AI validators require ~/.guardrailsrc with enable_metrics=false"
    )
    with pytest.raises(OptionalIntegrationSafetyError, match="enable_metrics=false"):
        enforce_guardrails_ai_privacy(rails_config)


def test_guardrails_ai_opt_out_is_mirrored_and_local_tracing_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rails_config = _config(tmp_path, 'guardrailsai check input $validator="regex_match"')
    settings = _fake_import(monkeypatch, enable_metrics=False)

    assert guardrails_ai_privacy_issue(rails_config) is None
    enforce_guardrails_ai_privacy(rails_config)

    assert settings.rc.enable_metrics is False
    assert settings.rc.token == "kept"
    assert settings.disable_tracing is True
