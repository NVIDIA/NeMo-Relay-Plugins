# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_relay_plugin import PluginContext
from nemoguardrails import RailsConfig
from registration_helpers import registered_llm_execution

from nemoguardrails_nemo_relay import configuration, execution_policy, runtime_selection
from nemoguardrails_nemo_relay.runtime_selection import (
    GuardrailsEngine,
    RuntimeReason,
    preflight_runtime,
    select_runtime,
)


def _config(tmp_path: Path, yaml: str, colang: str | None = None) -> tuple[Path, RailsConfig]:
    config_path = tmp_path / "guardrails"
    config_path.mkdir()
    (config_path / "config.yml").write_text(yaml, encoding="utf-8")
    if colang is not None:
        (config_path / "rails.co").write_text(colang, encoding="utf-8")
    return config_path, RailsConfig.from_path(str(config_path))


def test_catalog_configuration_requires_iorails_without_fallback(tmp_path: Path) -> None:
    config_path, rails_config = _config(
        tmp_path,
        "rails:\n  input:\n    flows:\n      - regex check input\n",
    )

    plan = select_runtime(str(config_path), rails_config)

    assert plan.engine is GuardrailsEngine.IORAILS
    assert plan.reason is RuntimeReason.CATALOG
    assert plan.constructor_options() == {"use_iorails": True, "require_iorails": True}
    assert preflight_runtime(plan, rails_config) == []


def test_custom_colang_selects_llmrails_explicitly(tmp_path: Path) -> None:
    config_path, rails_config = _config(
        tmp_path,
        "rails:\n  input:\n    flows:\n      - custom input rail\n",
        "define flow custom input rail\n  $allowed = True\n",
    )

    plan = select_runtime(str(config_path), rails_config)

    assert plan.engine is GuardrailsEngine.LLMRAILS
    assert plan.reason is RuntimeReason.CUSTOM_COLANG
    assert plan.constructor_options() == {"use_iorails": False, "require_iorails": False}


def test_iorails_incompatible_configuration_selects_llmrails_explicitly(tmp_path: Path) -> None:
    config_path, rails_config = _config(
        tmp_path,
        "rails:\n"
        "  input:\n"
        "    flows:\n"
        "      - regex check input\n"
        "  retrieval:\n"
        "    flows:\n"
        "      - regex check retrieval\n",
    )

    plan = select_runtime(str(config_path), rails_config)

    assert plan.engine is GuardrailsEngine.LLMRAILS
    assert plan.reason is RuntimeReason.IORAILS_INCOMPATIBLE
    assert plan.constructor_options() == {"use_iorails": False, "require_iorails": False}


@pytest.mark.parametrize("engine", ["azure", "azure_openai", "nvidia_ai_endpoints", "ollama"])
def test_default_framework_provider_presets_select_llmrails(tmp_path: Path, engine: str) -> None:
    config_path, rails_config = _config(
        tmp_path,
        "models:\n"
        "  - type: main\n"
        f"    engine: {engine}\n"
        "    model: fixture\n"
        "rails:\n"
        "  config:\n"
        "    regex_detection:\n"
        "      input:\n"
        "        patterns: [BLOCK]\n"
        "  input:\n"
        "    flows: [regex check input]\n",
    )

    plan = select_runtime(str(config_path), rails_config)

    assert plan.engine is GuardrailsEngine.LLMRAILS
    assert plan.reason is RuntimeReason.IORAILS_INCOMPATIBLE


@pytest.mark.parametrize("engine", ["nim", "openai"])
def test_iorails_native_provider_presets_remain_on_iorails(tmp_path: Path, engine: str) -> None:
    config_path, rails_config = _config(
        tmp_path,
        "models:\n"
        "  - type: main\n"
        f"    engine: {engine}\n"
        "    model: fixture\n"
        "rails:\n"
        "  config:\n"
        "    regex_detection:\n"
        "      input:\n"
        "        patterns: [BLOCK]\n"
        "  input:\n"
        "    flows: [regex check input]\n",
    )

    plan = select_runtime(str(config_path), rails_config)

    assert plan.engine is GuardrailsEngine.IORAILS
    assert plan.reason is RuntimeReason.CATALOG


def test_langchain_evaluator_framework_selects_llmrails_explicitly(tmp_path: Path) -> None:
    config_path, rails_config = _config(
        tmp_path,
        "rails:\n  input:\n    flows:\n      - regex check input\n",
    )

    plan = select_runtime(str(config_path), rails_config, evaluator_framework="langchain")

    assert plan.engine is GuardrailsEngine.LLMRAILS
    assert plan.reason is RuntimeReason.EVALUATOR_FRAMEWORK
    assert plan.constructor_options() == {"use_iorails": False, "require_iorails": False}


def test_trusted_config_module_selects_llmrails_without_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    config_path, rails_config = _config(
        tmp_path,
        "rails:\n  input:\n    flows:\n      - regex check input\n",
    )
    (config_path / "config.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )

    plan = select_runtime(str(config_path), rails_config)

    assert plan.engine is GuardrailsEngine.LLMRAILS
    assert plan.reason is RuntimeReason.CUSTOM_ACTIONS
    assert not marker.exists()


def test_colang_two_has_a_distinct_llmrails_plan(tmp_path: Path) -> None:
    config_path, rails_config = _config(tmp_path, 'colang_version: "2.x"\n')

    plan = select_runtime(str(config_path), rails_config)

    assert plan.engine is GuardrailsEngine.LLMRAILS
    assert plan.reason is RuntimeReason.COLANG_2
    assert plan.colang_version == "2.x"


def test_missing_optional_packages_fail_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, rails_config = _config(
        tmp_path,
        "rails:\n  input:\n    flows:\n      - mask sensitive data on input\n",
    )
    missing = {"presidio-analyzer", "presidio-anonymizer", "spacy", "en-core-web-lg"}
    monkeypatch.setattr(runtime_selection, "_distribution_is_installed", lambda name: name not in missing)

    issues = preflight_runtime(select_runtime(str(config_path), rails_config), rails_config)

    by_code = {issue.code: issue.message for issue in issues}
    assert "missing_guardrails_dependencies" in by_code
    assert "presidio-analyzer" in by_code["missing_guardrails_dependencies"]
    assert "missing_guardrails_models" in by_code
    assert "spacy:en_core_web_lg" in by_code["missing_guardrails_models"]
    assert "iorails_preflight_failed" in by_code


def test_required_catalog_environment_fails_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACTIVEFENCE_API_KEY", raising=False)
    config_path, rails_config = _config(
        tmp_path,
        "rails:\n  input:\n    flows:\n      - activefence moderation on input\n",
    )

    issues = preflight_runtime(select_runtime(str(config_path), rails_config), rails_config)

    assert [(issue.code, issue.message) for issue in issues] == [
        (
            "missing_guardrails_environment",
            "Guardrails configuration requires unset environment variables: ACTIVEFENCE_API_KEY",
        )
    ]


def test_explicit_model_api_key_environment_fails_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = "NEMO_RELAY_TEST_MODEL_API_KEY"
    monkeypatch.setenv(variable, "test-only-value")
    config_path, rails_config = _config(
        tmp_path,
        "models:\n"
        "  - type: main\n"
        "    engine: openai\n"
        "    model: fixture\n"
        f"    api_key_env_var: {variable}\n"
        "rails:\n"
        "  config:\n"
        "    regex_detection:\n"
        "      input:\n"
        "        patterns: [BLOCK]\n"
        "  input:\n"
        "    flows: [regex check input]\n",
    )
    monkeypatch.delenv(variable)

    issues = preflight_runtime(select_runtime(str(config_path), rails_config), rails_config)

    assert [(issue.code, issue.message) for issue in issues] == [
        (
            "missing_guardrails_environment",
            f"Guardrails configuration requires unset environment variables: {variable}",
        )
    ]


def test_implicit_catalog_llm_binding_is_validated(tmp_path: Path) -> None:
    config_path, rails_config = _config(
        tmp_path,
        "rails:\n"
        "  input:\n"
        "    flows:\n"
        "      - self check input\n"
        "prompts:\n"
        "  - task: self_check_input\n"
        "    content: Is this safe? {{ user_input }}\n",
    )

    issues = preflight_runtime(select_runtime(str(config_path), rails_config), rails_config)

    assert any(issue.code == "missing_guardrails_models" and "main" in issue.message for issue in issues)


def test_worker_validation_reports_colang_two_as_an_execution_contract_gap(tmp_path: Path) -> None:
    config_path, _ = _config(tmp_path, 'colang_version: "2.x"\n')

    diagnostics = configuration._validate_config({"config_path": str(config_path)})

    assert [item.code for item in diagnostics] == ["nemoguardrails.nemo_relay.unsupported_colang_version"]
    assert "entrypoint discovery" in diagnostics[0].message


async def test_worker_runs_catalog_input_rail_on_iorails(tmp_path: Path) -> None:
    from nemoguardrails_nemo_relay import worker

    config_path, _ = _config(
        tmp_path,
        "rails:\n"
        "  config:\n"
        "    regex_detection:\n"
        "      input:\n"
        "        patterns:\n"
        "          - BLOCK_IORAILS\n"
        "  input:\n"
        "    parallel: true\n"
        "    flows:\n"
        "      - regex check input\n",
    )
    context = MagicMock(spec=PluginContext)
    context.runtime = SimpleNamespace(list_runtime_registrations=AsyncMock(return_value=[]))
    plugin = worker.NeMoGuardrailsRelayWorker()

    await plugin.register(context, {"config_path": str(config_path)})
    try:
        assert plugin._rails is not None
        assert plugin._rails.use_iorails_engine is True
        callback = registered_llm_execution(context).callback
        next_call = SimpleNamespace(call=AsyncMock(return_value={}))
        request = {
            "headers": {},
            "content": {
                "model": "fixture-model",
                "messages": [{"role": "user", "content": "BLOCK_IORAILS"}],
            },
        }

        with pytest.raises(execution_policy._LlmPolicyError, match="prevented execution"):
            await callback("openai.chat_completions", request, next_call)
        next_call.call.assert_not_awaited()
    finally:
        await plugin.close()
