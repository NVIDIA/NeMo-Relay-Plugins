# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import nemoguardrails
import pytest
from nemo_relay_plugin import ConfigDiagnostic, DiagnosticLevel
from nemoguardrails import RailsConfig

from nemoguardrails_nemo_relay import (
    configuration,
    worker,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NO_MODEL_CONFIG = PROJECT_ROOT / "examples" / "no-model-rails"


@pytest.mark.parametrize(
    "config, field",
    [
        ({}, None),
        (
            {
                "config_path": str(NO_MODEL_CONFIG),
                "remote_checks": {
                    "endpoint": "https://guardrails.example.test",
                    "config_ids": ["policy"],
                    "phases": ["input"],
                },
            },
            None,
        ),
        (
            {
                "remote_checks": {
                    "endpoint": "http://guardrails.example.test",
                    "config_ids": ["policy"],
                    "phases": ["input"],
                }
            },
            "remote_checks",
        ),
        ({"config_path": "relative/path"}, "config_path"),
        ({"config_path": "/path/that/does/not/exist"}, "config_path"),
        ({"config_path": str(NO_MODEL_CONFIG), "version": 1}, "version"),
        ({"config_path": str(NO_MODEL_CONFIG), "check_timeout_ms": 0}, "check_timeout_ms"),
        ({"config_path": str(NO_MODEL_CONFIG), "check_timeout_ms": 999}, "check_timeout_ms"),
        ({"config_path": str(NO_MODEL_CONFIG), "check_timeout_ms": 25_001}, "check_timeout_ms"),
        ({"config_path": str(NO_MODEL_CONFIG), "allow_ignored_rail_families": 1}, "allow_ignored_rail_families"),
        ({"config_path": str(NO_MODEL_CONFIG), "allow_known_fail_open_rails": "yes"}, "allow_known_fail_open_rails"),
        ({"config_path": str(NO_MODEL_CONFIG), "payload_policy": []}, "payload_policy"),
        (
            {"config_path": str(NO_MODEL_CONFIG), "payload_policy": {"multimodal": "unchecked"}},
            "payload_policy",
        ),
        ({"config_path": str(NO_MODEL_CONFIG), "mutation_policy": []}, "mutation_policy"),
        (
            {"config_path": str(NO_MODEL_CONFIG), "mutation_policy": {"input": "pass_original"}},
            "mutation_policy",
        ),
        ({"config_path": str(NO_MODEL_CONFIG), "evaluator_framework": "anthropic"}, "evaluator_framework"),
        ({"config_path": str(NO_MODEL_CONFIG), "action_safety": "allow"}, "action_safety"),
        (
            {
                "remote_checks": {
                    "endpoint": "https://guardrails.example.test",
                    "config_ids": ["policy"],
                    "phases": ["input"],
                    "model": "evaluator",
                    "allow_remote_content_logging_and_retention": True,
                },
                "evaluator_framework": "default",
            },
            "evaluator_framework",
        ),
        (
            {
                "remote_checks": {
                    "endpoint": "https://guardrails.example.test",
                    "config_ids": ["policy"],
                    "phases": ["input"],
                    "model": "evaluator",
                    "allow_remote_content_logging_and_retention": True,
                    "timeout_ms": 2_000,
                },
                "check_timeout_ms": 1_000,
            },
            "remote_checks",
        ),
        ({"config_path": str(NO_MODEL_CONFIG), "secret_env": []}, "secret_env"),
        ({"config_path": str(NO_MODEL_CONFIG), "secret_env": {"INVALID-NAME": "value"}}, "secret_env"),
        (
            {"config_path": str(NO_MODEL_CONFIG), "secret_env": {"DUPLICATE": "one", "duplicate": "two"}},
            "secret_env",
        ),
        ({"config_path": str(NO_MODEL_CONFIG), "secret_env": {"VALID_NAME": ""}}, "secret_env"),
        ({"config_path": str(NO_MODEL_CONFIG), "secret_env": {"VALID_NAME": 1}}, "secret_env"),
        ({"config_path": str(NO_MODEL_CONFIG), "secret_env": {"VALID_NAME": "value\x00suffix"}}, "secret_env"),
        (
            {
                "config_path": str(NO_MODEL_CONFIG),
                "secret_env": {"NEMO_GUARDRAILS_NO_USAGE_STATS": "0"},
            },
            "secret_env",
        ),
        (
            {
                "config_path": str(NO_MODEL_CONFIG),
                "secret_env": {f"SECRET_{index}": "value" for index in range(configuration.MAX_SECRET_ENV_ENTRIES + 1)},
            },
            "secret_env",
        ),
        ({"config_path": str(NO_MODEL_CONFIG), "unexpected": True}, "unexpected"),
    ],
)
def test_invalid_configuration_is_rejected(config: dict[str, object], field: str | None) -> None:
    diagnostics = configuration._validate_config(config)

    assert diagnostics
    assert any(item.field == field for item in diagnostics)


def test_valid_configuration_uses_defaults() -> None:
    for config in (
        {"config_path": str(NO_MODEL_CONFIG)},
        {"config_path": str(NO_MODEL_CONFIG), "check_timeout_ms": configuration.MIN_CHECK_TIMEOUT_MS},
        {"config_path": str(NO_MODEL_CONFIG), "check_timeout_ms": configuration.MAX_CHECK_TIMEOUT_MS},
        {
            "config_path": str(NO_MODEL_CONFIG),
            "payload_policy": {"multimodal": "text_only", "reasoning": "final_answer_only"},
        },
        {
            "config_path": str(NO_MODEL_CONFIG),
            "mutation_policy": {"input": "apply"},
        },
    ):
        diagnostics = configuration._validate_config(config)
        assert diagnostics == []


def test_reasoning_checks_and_mutation_require_their_selected_text_phase() -> None:
    local_reasoning = configuration._validate_config(
        {
            "config_path": str(NO_MODEL_CONFIG),
            "payload_policy": {"reasoning": "check_output"},
        }
    )
    local_mutation = configuration._validate_config(
        {
            "config_path": str(NO_MODEL_CONFIG),
            "mutation_policy": {"output": "apply"},
        }
    )
    remote_reasoning = configuration._validate_config(
        {
            "remote_checks": {
                "endpoint": "https://guardrails.example.test",
                "config_ids": ["policy"],
                "phases": ["input"],
                "model": "evaluator",
                "allow_remote_content_logging_and_retention": True,
            },
            "payload_policy": {"reasoning": "check_output"},
        }
    )

    assert [item.code for item in local_reasoning] == ["nemoguardrails.nemo_relay.reasoning_output_phase_missing"]
    assert [item.code for item in local_mutation] == ["nemoguardrails.nemo_relay.mutation_output_phase_missing"]
    assert [item.code for item in remote_reasoning] == ["nemoguardrails.nemo_relay.reasoning_output_phase_missing"]


def test_reasoning_check_is_valid_with_a_real_output_phase() -> None:
    assert (
        configuration._validate_config(
            {
                "config_path": str(PROJECT_ROOT / "examples" / "input-output-rails"),
                "payload_policy": {"reasoning": "check_output"},
            }
        )
        == []
    )


@pytest.mark.parametrize("prior", [None, "prior-value"])
def test_validation_applies_secret_environment_temporarily(
    monkeypatch: pytest.MonkeyPatch,
    prior: str | None,
) -> None:
    name = "NEMO_RELAY_GUARDRAILS_VALIDATION_SECRET"
    secret = "PRIVATE_VALIDATION_VALUE"
    if prior is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, prior)
    original = RailsConfig.from_path
    observed: list[str | None] = []

    def load(path: str) -> object:
        observed.append(os.environ.get(name))
        return original(path)

    monkeypatch.setattr(RailsConfig, "from_path", MagicMock(side_effect=load))

    diagnostics = configuration._validate_config(
        {
            "config_path": str(NO_MODEL_CONFIG),
            "secret_env": {name: secret},
        }
    )

    assert diagnostics == []
    assert observed == [secret]
    assert os.environ.get(name) == prior


def test_invalid_secret_environment_diagnostic_does_not_echo_content() -> None:
    name = "INVALID-NAME-PRIVATE"
    secret = "PRIVATE_SECRET_VALUE\x00"

    diagnostics = configuration._validate_config(
        {
            "config_path": str(NO_MODEL_CONFIG),
            "secret_env": {name: secret},
        }
    )

    assert diagnostics
    rendered = " ".join(f"{item.field} {item.message}" for item in diagnostics)
    assert name not in rendered
    assert secret not in rendered


@pytest.mark.parametrize("name", ["NEMO_GUARDRAILS_NO_USAGE_STATS", "nemo_guardrails_no_usage_stats"])
def test_secret_environment_cannot_reenable_guardrails_telemetry(name: str) -> None:
    diagnostics = configuration._validate_config(
        {
            "config_path": str(NO_MODEL_CONFIG),
            "secret_env": {name: "0"},
        }
    )

    assert [item.code for item in diagnostics] == ["nemoguardrails.nemo_relay.reserved_secret_env_name"]
    assert os.environ["NEMO_GUARDRAILS_NO_USAGE_STATS"] == "1"


def test_validation_resolves_explicit_model_api_key_from_secret_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = "NEMO_RELAY_TEST_MODEL_API_KEY"
    config_path = tmp_path / "explicit-model-key"
    config_path.mkdir()
    (config_path / "config.yml").write_text(
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
        encoding="utf-8",
    )
    monkeypatch.delenv(variable, raising=False)

    missing = configuration._validate_config({"config_path": str(config_path)})
    configured = configuration._validate_config(
        {
            "config_path": str(config_path),
            "secret_env": {variable: "test-only-value"},
        }
    )

    # Guardrails validates an explicit model key while parsing RailsConfig;
    # the wrapper-installed secret must therefore be present before parsing.
    assert [item.code for item in missing] == ["nemoguardrails.nemo_relay.invalid_guardrails_config"]
    assert configured == []
    assert variable not in os.environ


def test_configuration_resolution_failure_is_not_reported_as_a_secret_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "NEMO_RELAY_GUARDRAILS_VALIDATION_SECRET"
    monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        configuration,
        "_resolve_config",
        MagicMock(side_effect=RuntimeError("private resolution details")),
    )

    diagnostics = configuration._validate_config(
        {
            "config_path": str(NO_MODEL_CONFIG),
            "secret_env": {name: "PRIVATE_VALIDATION_VALUE"},
        }
    )

    assert [(item.code, item.field, item.message) for item in diagnostics] == [
        (
            "nemoguardrails.nemo_relay.configuration_resolution_failed",
            None,
            "the Guardrails configuration could not be resolved",
        )
    ]
    assert name not in os.environ


def test_failed_guardrails_validation_restores_secret_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "NEMO_RELAY_GUARDRAILS_VALIDATION_SECRET"
    secret = "PRIVATE_VALIDATION_VALUE"
    config_path = tmp_path / "invalid-guardrails-config"
    config_path.mkdir()
    (config_path / "config.yml").write_text("rails: [\n", encoding="utf-8")
    monkeypatch.delenv(name, raising=False)

    diagnostics = configuration._validate_config(
        {
            "config_path": str(config_path),
            "secret_env": {name: secret},
        }
    )

    assert [item.code for item in diagnostics] == ["nemoguardrails.nemo_relay.invalid_guardrails_config"]
    assert name not in os.environ
    assert all(secret not in item.message for item in diagnostics)


def test_validation_parses_guardrails_config_without_executing_config_py(tmp_path: Path) -> None:
    config_path = tmp_path / "trusted-config"
    marker = tmp_path / "config-py-executed"
    shutil.copytree(NO_MODEL_CONFIG, config_path)
    (config_path / "config.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )

    diagnostics = worker.NeMoGuardrailsRelayWorker().validate({"config_path": str(config_path)})

    assert diagnostics == []
    assert not marker.exists()


def test_validation_requires_exact_guardrails_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nemoguardrails, "__version__", "0.24.0")

    diagnostics = configuration._validate_config({"config_path": str(NO_MODEL_CONFIG)})

    assert [item.code for item in diagnostics] == [f"{configuration.PLUGIN_ID}.unsupported_guardrails_version"]
    assert "0.24.0" not in diagnostics[0].message


def test_validation_requires_acknowledgement_for_ignored_rail_families(tmp_path: Path) -> None:
    config_path = tmp_path / "input-and-output"
    shutil.copytree(NO_MODEL_CONFIG, config_path)
    (config_path / "config.yml").write_text(
        "rails:\n"
        "  input:\n"
        "    flows:\n"
        "      - input rail\n"
        "  output:\n"
        "    flows:\n"
        "      - output rail\n"
        "  dialog:\n"
        "    single_call:\n"
        "      enabled: true\n",
        encoding="utf-8",
    )
    rejected = configuration._validate_config({"config_path": str(config_path)})
    acknowledged = configuration._validate_config(
        {"config_path": str(config_path), "allow_ignored_rail_families": True}
    )

    assert len(rejected) == 1
    assert rejected[0].level == DiagnosticLevel.ERROR
    assert rejected[0].code == f"{configuration.PLUGIN_ID}.ignored_rail_families"
    assert "dialog" in rejected[0].message
    assert "output" not in rejected[0].message
    assert len(acknowledged) == 1
    assert acknowledged[0].level == DiagnosticLevel.WARNING
    assert acknowledged[0].code == rejected[0].code


def test_validation_requires_acknowledgement_for_retrieval_rails(tmp_path: Path) -> None:
    config_path = tmp_path / "input-and-retrieval"
    shutil.copytree(NO_MODEL_CONFIG, config_path)
    (config_path / "config.yml").write_text(
        "rails:\n  input:\n    flows:\n      - input rail\n  retrieval:\n    flows:\n      - retrieval rail\n",
        encoding="utf-8",
    )

    rejected = configuration._validate_config({"config_path": str(config_path)})
    acknowledged = configuration._validate_config(
        {"config_path": str(config_path), "allow_ignored_rail_families": True}
    )

    assert len(rejected) == 1
    assert rejected[0].level == DiagnosticLevel.ERROR
    assert rejected[0].code == f"{configuration.PLUGIN_ID}.ignored_rail_families"
    assert "retrieval" in rejected[0].message
    assert len(acknowledged) == 1
    assert acknowledged[0].level == DiagnosticLevel.WARNING
    assert acknowledged[0].code == rejected[0].code


def test_validation_rejects_output_rail_that_needs_retrieval_context(tmp_path: Path) -> None:
    config_path = tmp_path / "self-check-facts"
    config_path.mkdir()
    (config_path / "config.yml").write_text(
        "models:\n"
        "  - type: main\n"
        "    engine: nim\n"
        "    model: fixture-model\n"
        "rails:\n"
        "  output:\n"
        "    flows:\n"
        "      - self check facts\n"
        "prompts:\n"
        "  - task: self_check_facts\n"
        "    content: 'Evidence: {{ evidence }} Response: {{ response }}'\n",
        encoding="utf-8",
    )

    for config in (
        {"config_path": str(config_path)},
        {"config_path": str(config_path), "allow_ignored_rail_families": True},
    ):
        diagnostics = configuration._validate_config(config)

        assert len(diagnostics) == 1
        assert diagnostics[0].level == DiagnosticLevel.ERROR
        assert diagnostics[0].code == f"{configuration.PLUGIN_ID}.unsupported_catalog_context"
        assert "self check facts" in diagnostics[0].message
        assert "relevant_chunks" in diagnostics[0].message


def test_catalog_context_validation_covers_every_required_non_message_value() -> None:
    unsupported = {
        "alignscore check facts",
        "autoalign groundedness output",
        "fiddler bot faithfulness",
        "patronus api check output",
        "patronus lynx check output hallucination",
        "self check facts",
        "self check hallucination",
    }
    rails_config = SimpleNamespace(
        rails=SimpleNamespace(
            input=SimpleNamespace(flows=[]),
            output=SimpleNamespace(flows=sorted(unsupported)),
        )
    )

    diagnostics = configuration._catalog_context_diagnostics(rails_config)

    assert len(diagnostics) == 1
    assert diagnostics[0].code == f"{configuration.PLUGIN_ID}.unsupported_catalog_context"
    assert all(flow_name in diagnostics[0].message for flow_name in unsupported)


def test_validation_does_not_flag_default_dialog_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "default-dialog"
    shutil.copytree(NO_MODEL_CONFIG, config_path)
    (config_path / "config.yml").write_text(
        "rails:\n  input:\n    flows:\n      - input rail\n  dialog:\n    single_call:\n      enabled: false\n",
        encoding="utf-8",
    )

    assert configuration._validate_config({"config_path": str(config_path)}) == []


def test_validation_rejects_standard_colang_dialog_behavior(tmp_path: Path) -> None:
    config_path = tmp_path / "colang-dialog"
    shutil.copytree(NO_MODEL_CONFIG, config_path)
    with (config_path / "rails.co").open("a", encoding="utf-8") as rails_file:
        rails_file.write(
            '\ndefine user express greeting\n  "hello there"\n\n'
            "define flow greeting\n  user express greeting\n  bot refuse to respond\n"
        )

    for config in (
        {"config_path": str(config_path)},
        {"config_path": str(config_path), "allow_ignored_rail_families": True},
    ):
        diagnostics = configuration._validate_config(config)

        assert len(diagnostics) == 1
        assert diagnostics[0].level == DiagnosticLevel.ERROR
        assert diagnostics[0].code == f"{configuration.PLUGIN_ID}.unsupported_dialog_flows"


def test_validation_rejects_unselected_top_level_colang_flow_without_user_messages(tmp_path: Path) -> None:
    config_path = tmp_path / "implicit-dialog"
    shutil.copytree(NO_MODEL_CONFIG, config_path)
    with (config_path / "rails.co").open("a", encoding="utf-8") as rails_file:
        rails_file.write(
            '\ndefine flow unselected behavior\n  if $user_message == "trigger"\n    bot refuse to respond\n'
        )

    diagnostics = configuration._validate_config({"config_path": str(config_path), "allow_ignored_rail_families": True})

    assert len(diagnostics) == 1
    assert diagnostics[0].level == DiagnosticLevel.ERROR
    assert diagnostics[0].code == f"{configuration.PLUGIN_ID}.unsupported_dialog_flows"


def test_validation_allows_explicit_colang_subflow_helper(tmp_path: Path) -> None:
    config_path = tmp_path / "input-with-helper"
    shutil.copytree(NO_MODEL_CONFIG, config_path)
    with (config_path / "rails.co").open("a", encoding="utf-8") as rails_file:
        rails_file.write("\ndefine subflow helper\n  $helper_ran = True\n")

    assert configuration._validate_config({"config_path": str(config_path)}) == []


def test_validation_requires_acknowledgement_for_known_guardrails_fail_open_surfaces(
    tmp_path: Path,
) -> None:
    input_flows = sorted(configuration._KNOWN_UPSTREAM_FAIL_OPEN_INPUT_RAILS)
    output_flows = sorted(configuration._KNOWN_UPSTREAM_FAIL_OPEN_OUTPUT_RAILS)
    assert input_flows
    assert output_flows

    config_path = tmp_path / "known-fail-open-catalog"
    config_path.mkdir()
    input_yaml = "\n".join(f"      - {name}" for name in input_flows)
    output_yaml = "\n".join(f"      - {name}" for name in output_flows)
    (config_path / "config.yml").write_text(
        f"rails:\n  input:\n    flows:\n{input_yaml}\n  output:\n    flows:\n{output_yaml}\n",
        encoding="utf-8",
    )

    rejected = configuration._validate_config({"config_path": str(config_path)})
    acknowledged = configuration._validate_config(
        {"config_path": str(config_path), "allow_known_fail_open_rails": True}
    )

    rejected_diagnostic = next(
        item for item in rejected if item.code == f"{configuration.PLUGIN_ID}.known_upstream_fail_open_rails"
    )
    acknowledged_diagnostic = next(
        item for item in acknowledged if item.code == f"{configuration.PLUGIN_ID}.known_upstream_fail_open_rails"
    )
    assert rejected_diagnostic.level == DiagnosticLevel.ERROR
    assert acknowledged_diagnostic.level == DiagnosticLevel.WARNING
    for flow_name in [*input_flows, *output_flows]:
        assert flow_name in rejected_diagnostic.message
        assert flow_name in acknowledged_diagnostic.message


@pytest.mark.parametrize(
    ("family", "flow_name", "config_name"),
    [
        ("input", "ai defense inspect prompt", "ai_defense"),
        ("input", "f5 guardrails scan input", "f5"),
        ("output", "ai defense inspect response", "ai_defense"),
        ("output", "f5 guardrails scan output", "f5"),
    ],
)
def test_validation_rejects_configured_guardrails_fail_open(
    tmp_path: Path,
    family: str,
    flow_name: str,
    config_name: str,
) -> None:
    config_path = tmp_path / config_name
    config_path.mkdir()
    (config_path / "config.yml").write_text(
        f"rails:\n  config:\n    {config_name}:\n      fail_open: true\n  {family}:\n    flows:\n      - {flow_name}\n",
        encoding="utf-8",
    )

    rejected = configuration._validate_config({"config_path": str(config_path)})
    acknowledged = configuration._validate_config(
        {"config_path": str(config_path), "allow_known_fail_open_rails": True}
    )

    assert any(
        item.level == DiagnosticLevel.ERROR
        and item.code == f"{configuration.PLUGIN_ID}.configured_upstream_fail_open_rails"
        for item in rejected
    )
    assert any(
        item.level == DiagnosticLevel.WARNING
        and item.code == f"{configuration.PLUGIN_ID}.configured_upstream_fail_open_rails"
        for item in acknowledged
    )


def test_validation_warns_for_all_ambiguous_upstream_verdict_contracts() -> None:
    input_flows = sorted(configuration._UPSTREAM_AMBIGUOUS_VERDICT_INPUT_RAILS)
    output_flows = sorted(configuration._UPSTREAM_AMBIGUOUS_VERDICT_OUTPUT_RAILS)
    assert input_flows
    assert output_flows

    rails_config = SimpleNamespace(
        rails=SimpleNamespace(
            input=SimpleNamespace(flows=input_flows),
            output=SimpleNamespace(flows=output_flows),
            config=SimpleNamespace(ai_defense=None, f5=None),
        )
    )

    diagnostics = configuration._upstream_contract_diagnostics(rails_config, acknowledged=False)

    diagnostic = next(
        item for item in diagnostics if item.code == f"{configuration.PLUGIN_ID}.upstream_verdict_contract"
    )
    assert diagnostic.level == DiagnosticLevel.WARNING
    for flow_name in [*input_flows, *output_flows]:
        assert flow_name in diagnostic.message


@pytest.mark.parametrize(
    ("config_prefix", "expected_code"),
    [
        ("metrics:\n  enabled: true\n", "unsupported_metrics"),
        (
            "rails:\n  input:\n    speculative_generation: true\n    flows:\n      - input rail\n",
            "unsupported_speculative_generation",
        ),
        (
            "rails:\n"
            "  input:\n"
            "    flows:\n"
            "      - input rail\n"
            "  actions:\n"
            "    instant_actions:\n"
            "      - custom action\n",
            "unsupported_instant_actions",
        ),
    ],
)
def test_validation_rejects_llmrails_settings_that_have_no_effect(
    tmp_path: Path,
    config_prefix: str,
    expected_code: str,
) -> None:
    config_path = tmp_path / expected_code
    shutil.copytree(NO_MODEL_CONFIG, config_path)
    config_file = config_path / "config.yml"
    if expected_code == "unsupported_metrics":
        config_file.write_text(config_prefix + config_file.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        config_file.write_text(config_prefix, encoding="utf-8")

    diagnostics = configuration._validate_config({"config_path": str(config_path)})

    assert any(item.code == f"{configuration.PLUGIN_ID}.{expected_code}" for item in diagnostics)


@pytest.mark.parametrize(
    ("setting", "expected_code"),
    [
        ("enable_multi_step_generation: true\n", "unsupported_multi_step_generation"),
        ("passthrough: true\n", "unsupported_passthrough"),
        ("raw_llm_call_action: custom raw llm call\n", "unsupported_raw_llm_call_action"),
    ],
)
def test_validation_rejects_explicit_generation_only_settings(
    tmp_path: Path,
    setting: str,
    expected_code: str,
) -> None:
    config_path = tmp_path / expected_code
    shutil.copytree(NO_MODEL_CONFIG, config_path)
    config_file = config_path / "config.yml"
    config_file.write_text(setting + config_file.read_text(encoding="utf-8"), encoding="utf-8")

    diagnostics = configuration._validate_config({"config_path": str(config_path)})

    assert [item.code for item in diagnostics] == [f"{configuration.PLUGIN_ID}.{expected_code}"]


@pytest.mark.parametrize(
    "setting",
    [
        "enable_multi_step_generation: false\n",
        "passthrough: false\n",
        "raw_llm_call_action: raw llm call\n",
    ],
)
def test_validation_accepts_disabled_generation_only_settings(tmp_path: Path, setting: str) -> None:
    config_path = tmp_path / "disabled-generation-setting"
    shutil.copytree(NO_MODEL_CONFIG, config_path)
    config_file = config_path / "config.yml"
    config_file.write_text(setting + config_file.read_text(encoding="utf-8"), encoding="utf-8")

    assert configuration._validate_config({"config_path": str(config_path)}) == []


def test_validation_rejects_only_active_deprecated_top_level_streaming(tmp_path: Path) -> None:
    def validate(value: bool) -> list[ConfigDiagnostic]:
        config_path = tmp_path / str(value).lower()
        shutil.copytree(NO_MODEL_CONFIG, config_path)
        config_file = config_path / "config.yml"
        config_file.write_text(
            f"streaming: {str(value).lower()}\n" + config_file.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return configuration._validate_config({"config_path": str(config_path)})

    assert validate(False) == []
    assert [item.code for item in validate(True)] == [f"{configuration.PLUGIN_ID}.unsupported_top_level_streaming"]


@pytest.mark.parametrize("evaluator_framework", ["default", "langchain"])
def test_validation_rejects_runtime_controls_for_structural_only_configuration(
    tmp_path: Path,
    evaluator_framework: str,
) -> None:
    config_path = tmp_path / "structural-only"
    config_path.mkdir()
    (config_path / "config.yml").write_text(
        "actions_server_url: https://actions.example.test\nrails:\n  tool_output:\n    flows: [tool call validation]\n",
        encoding="utf-8",
    )

    diagnostics = configuration._validate_config(
        {
            "config_path": str(config_path),
            "action_safety": {"synchronous": "reject"},
            "evaluator_framework": evaluator_framework,
        }
    )

    assert [item.code for item in diagnostics] == [
        f"{configuration.PLUGIN_ID}.unsupported_action_safety_runtime",
        f"{configuration.PLUGIN_ID}.unsupported_evaluator_framework_without_text_rails",
        f"{configuration.PLUGIN_ID}.unsupported_actions_server_runtime",
    ]


def test_validation_rejects_llmrails_controls_for_iorails_text_runtime(tmp_path: Path) -> None:
    config_path = tmp_path / "iorails-text"
    config_path.mkdir()
    (config_path / "config.yml").write_text(
        "actions_server_url: https://actions.example.test\n"
        "rails:\n"
        "  config:\n"
        "    regex_detection:\n"
        "      input:\n"
        "        patterns: [BLOCK]\n"
        "  input:\n"
        "    flows: [regex check input]\n",
        encoding="utf-8",
    )

    diagnostics = configuration._validate_config(
        {
            "config_path": str(config_path),
            "action_safety": {"synchronous": "reject"},
            "evaluator_framework": "default",
        }
    )

    assert [item.code for item in diagnostics] == [
        f"{configuration.PLUGIN_ID}.unsupported_action_safety_runtime",
        f"{configuration.PLUGIN_ID}.unsupported_actions_server_runtime",
    ]


@pytest.mark.parametrize(
    ("custom_flow", "parameters", "expect_error"),
    [
        (False, {"max_retries": 0}, True),
        (False, {"connect_timeout": 1}, True),
        (False, {"max_attempts": 1, "timeout_connect": 1}, False),
        (True, {"max_attempts": 1}, True),
        (True, {"timeout_connect": 1}, True),
        (True, {"max_retries": 0, "connect_timeout": 1}, False),
    ],
)
def test_runtime_rejects_transport_parameter_names_owned_by_the_other_engine(
    custom_flow: bool,
    parameters: dict[str, int],
    expect_error: bool,
) -> None:
    flow = "custom input" if custom_flow else "self check input"
    config: dict[str, object] = {
        "models": [
            {
                "type": "main",
                "engine": "openai",
                "model": "fixture",
                "parameters": parameters,
            }
        ],
        "rails": {"input": {"flows": [flow]}},
    }
    if not custom_flow:
        config["prompts"] = [{"task": "self_check_input", "content": "{{ user_input }}"}]
    rails_config = RailsConfig.from_content(
        colang_content="define flow custom input\n  bot refuse to respond" if custom_flow else None,
        config=config,
    )
    runtime_plan = configuration.select_runtime("", rails_config)

    diagnostics = configuration._runtime_scope_diagnostics(
        rails_config,
        runtime_plan,
        evaluator_framework=configuration.EvaluatorFramework.DEFAULT,
        action_safety_configured=False,
        evaluator_framework_configured=False,
    )

    mismatches = [
        item for item in diagnostics if item.code == f"{configuration.PLUGIN_ID}.evaluator_transport_parameter_mismatch"
    ]
    assert bool(mismatches) is expect_error
    if expect_error:
        assert all(name in mismatches[0].message for name in parameters)


def test_langchain_runtime_does_not_apply_default_framework_transport_names() -> None:
    rails_config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "anthropic",
                    "model": "fixture",
                    "parameters": {"max_attempts": 1, "timeout_connect": 1},
                }
            ],
            "rails": {"input": {"flows": ["self check input"]}},
            "prompts": [{"task": "self_check_input", "content": "{{ user_input }}"}],
        }
    )
    runtime_plan = configuration.select_runtime("", rails_config, evaluator_framework="langchain")

    diagnostics = configuration._runtime_scope_diagnostics(
        rails_config,
        runtime_plan,
        evaluator_framework=configuration.EvaluatorFramework.LANGCHAIN,
        action_safety_configured=False,
        evaluator_framework_configured=True,
    )

    assert not any(item.code.endswith("evaluator_transport_parameter_mismatch") for item in diagnostics)


def test_validation_allows_controls_consumed_by_local_text_runtime(tmp_path: Path) -> None:
    config_path = tmp_path / "llmrails-text"
    shutil.copytree(NO_MODEL_CONFIG, config_path)
    config_file = config_path / "config.yml"
    config_file.write_text(
        "actions_server_url: https://actions.example.test\n" + config_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    assert (
        configuration._validate_config(
            {
                "config_path": str(config_path),
                "action_safety": {"synchronous": "reject"},
                "evaluator_framework": "default",
            }
        )
        == []
    )


@pytest.mark.parametrize("enable_content_capture", [False, True])
def test_validation_rejects_unqualified_guardrails_tracing(
    tmp_path: Path,
    enable_content_capture: bool,
) -> None:
    config_path = tmp_path / "content-tracing"
    shutil.copytree(NO_MODEL_CONFIG, config_path)
    config_file = config_path / "config.yml"
    config_file.write_text(
        "tracing:\n  enabled: true\n"
        f"  enable_content_capture: {str(enable_content_capture).lower()}\n" + config_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    diagnostics = configuration._validate_config({"config_path": str(config_path)})

    assert [item.code for item in diagnostics] == [f"{configuration.PLUGIN_ID}.unsupported_guardrails_tracing"]
    assert diagnostics[0].level == DiagnosticLevel.ERROR


def test_validation_rejects_multilingual_refusal_configuration() -> None:
    rails_config = RailsConfig.from_content(
        config={
            "rails": {
                "config": {
                    "content_safety": {"multilingual": {"enabled": True}},
                    "regex_detection": {"input": {"patterns": ["BLOCK"]}},
                },
                "input": {"flows": ["regex check input"]},
            }
        }
    )

    diagnostics = configuration._basic_rails_config_diagnostics(rails_config)

    assert [item.code for item in diagnostics] == [f"{configuration.PLUGIN_ID}.unsupported_multilingual_refusal"]
    assert diagnostics[0].level == DiagnosticLevel.ERROR


@pytest.mark.parametrize(
    ("name", "config_yaml", "rails_co", "expected_code"),
    [
        (
            "malformed-private-config",
            "rails: [PRIVATE_CONFIG_CANARY\n",
            "",
            "invalid_guardrails_config",
        ),
        (
            "colang-two-private-config",
            'colang_version: "2.x"\nrails:\n  input:\n    flows:\n      - PRIVATE_CONFIG_CANARY\n',
            "",
            "unsupported_colang_version",
        ),
        (
            "exception-private-config",
            "enable_rails_exceptions: true\nrails:\n  input:\n    flows:\n      - input rail\n",
            'define flow input rail\n  bot "PRIVATE_CONFIG_CANARY"\n',
            "unsupported_rails_exceptions",
        ),
    ],
)
def test_validation_returns_private_generic_guardrails_diagnostics(
    tmp_path: Path,
    name: str,
    config_yaml: str,
    rails_co: str,
    expected_code: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / name
    config_path.mkdir()
    (config_path / "config.yml").write_text(config_yaml, encoding="utf-8")
    if rails_co:
        (config_path / "rails.co").write_text(rails_co, encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        diagnostics = worker.NeMoGuardrailsRelayWorker().validate({"config_path": str(config_path)})
    captured = capsys.readouterr()

    assert len(diagnostics) == 1
    diagnostic = next(item for item in diagnostics if item.code.endswith(expected_code))
    assert diagnostic.code == f"{configuration.PLUGIN_ID}.{expected_code}"
    assert diagnostic.field == "config_path"
    assert "PRIVATE_CONFIG_CANARY" not in diagnostic.message
    assert str(config_path) not in diagnostic.message
    assert all("PRIVATE_CONFIG_CANARY" not in str(item.message) for item in caught_warnings)
    assert "PRIVATE_CONFIG_CANARY" not in captured.out
    assert "PRIVATE_CONFIG_CANARY" not in captured.err
    assert "PRIVATE_CONFIG_CANARY" not in caplog.text
