# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from guardrails_config import write_guardrails_config

from nemoguardrails_nemo_relay import configuration, evaluator_framework
from nemoguardrails_nemo_relay.evaluator_framework import (
    EvaluatorFramework,
    evaluator_framework_from_config,
    evaluator_framework_issues,
)


def _config(*engines: str) -> object:
    return SimpleNamespace(models=[SimpleNamespace(engine=engine) for engine in engines])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, EvaluatorFramework.DEFAULT),
        ("default", EvaluatorFramework.DEFAULT),
        ("langchain", EvaluatorFramework.LANGCHAIN),
    ],
)
def test_evaluator_framework_setting(value: object, expected: EvaluatorFramework) -> None:
    assert evaluator_framework_from_config(value) is expected


@pytest.mark.parametrize("value", [True, "anthropic", {}, 1])
def test_evaluator_framework_setting_rejects_unknown_values(value: object) -> None:
    with pytest.raises(ValueError):
        evaluator_framework_from_config(value)


def test_native_anthropic_is_not_misclassified_as_openai_compatible() -> None:
    issues = evaluator_framework_issues(_config("anthropic"), EvaluatorFramework.DEFAULT)

    assert [issue.code for issue in issues] == ["anthropic_evaluator_requires_langchain"]


def test_langchain_anthropic_preflight_requires_complete_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluator_framework,
        "_installed",
        lambda name: name not in {"langchain-anthropic", "langchain-community"},
    )

    issues = evaluator_framework_issues(_config("anthropic"), EvaluatorFramework.LANGCHAIN)

    assert len(issues) == 1
    assert issues[0].code == "missing_evaluator_framework_dependencies"
    assert issues[0].message.endswith("langchain-anthropic, langchain-community")


def test_langchain_anthropic_profile_passes_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluator_framework, "_installed", lambda _name: True)

    assert evaluator_framework_issues(_config("anthropic"), EvaluatorFramework.LANGCHAIN) == ()


@pytest.mark.parametrize("engines", [("openai",), ("nim",), ("anthropic", "openai")])
def test_langchain_rejects_unqualified_evaluator_engines(
    engines: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator_framework, "_installed", lambda _name: True)

    issues = evaluator_framework_issues(_config(*engines), EvaluatorFramework.LANGCHAIN)

    assert [issue.code for issue in issues] == ["unsupported_langchain_evaluator_engine"]
    assert "Anthropic" in issues[0].message


def test_default_openai_compatible_evaluator_needs_no_optional_profile() -> None:
    assert evaluator_framework_issues(_config("nim", "openai"), EvaluatorFramework.DEFAULT) == ()


@pytest.mark.parametrize(
    ("engine", "framework", "expected_error"),
    [
        ("anthropic", None, "anthropic_evaluator_requires_langchain"),
        ("anthropic", "langchain", None),
        ("openai", "langchain", "unsupported_langchain_evaluator_engine"),
    ],
)
def test_worker_validates_evaluator_framework_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    framework: str | None,
    expected_error: str | None,
) -> None:
    config_path = write_guardrails_config(
        tmp_path,
        f"""
models:
  - type: main
    engine: {engine}
    model: fixture
rails:
  config:
    regex_detection:
      input:
        patterns:
          - BLOCK
  input:
    flows:
      - regex check input
""".strip()
        + "\n",
    )
    monkeypatch.setattr(evaluator_framework, "_installed", lambda _name: True)
    config: dict[str, object] = {"config_path": str(config_path)}
    if framework is not None:
        config["evaluator_framework"] = framework

    diagnostics = configuration._validate_config(config)
    errors = [item.code.rsplit(".", 1)[-1] for item in diagnostics if item.level.value == "error"]

    assert errors == ([] if expected_error is None else [expected_error])
