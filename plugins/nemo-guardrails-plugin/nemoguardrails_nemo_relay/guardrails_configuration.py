# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails configuration loading and compatibility diagnostics."""

from __future__ import annotations

import logging
import warnings
from typing import Any

from nemo_relay_plugin import ConfigDiagnostic, DiagnosticLevel, Json
from nemoguardrails import RailsConfig
from nemoguardrails.manifests import (
    RailDirection,
    default_rail_catalog,
    normalize_configured_surface_name,
)

from .evaluator_framework import (
    EvaluatorFramework,
    evaluator_framework_issues,
)
from .payload_policy import ReasoningPolicy, payload_policy_from_config
from .remote_actions import validated_actions_server_url
from .runtime_selection import RuntimePlan, preflight_runtime, select_runtime
from .semantic_tools import (
    SemanticResultSource,
    SemanticToolSettings,
)
from .structural_tools import StructuralToolSettings
from .text_mutation import MutationMode, mutation_policy_from_config

PLUGIN_ID = "nemoguardrails.nemo_relay"

_NO_CONTENT_LOG_LEVEL = logging.CRITICAL + 1
_STRUCTURAL_TOOL_CALL_FLOW = "tool call validation"
_STRUCTURAL_TOOL_RESULT_FLOW = "tool result validation"
_MESSAGE_CHECK_CONTEXT_KEYS = frozenset({"bot_message", "user_message"})

# These built-in Guardrails 0.24 input surfaces intentionally convert at least
# one evaluator, service, or partial-policy failure into an allow result. The
# public RailsResult returned to this worker does not preserve the metadata
# needed to distinguish that allow from an ordinary pass, so require an
# explicit deployment acknowledgement before activating them.
_KNOWN_UPSTREAM_FAIL_OPEN_INPUT_RAILS = frozenset(
    {
        "crowdstrike aidr guard input",
        "fiddler user safety",
        "jailbreak detection heuristics",
        "jailbreak detection model",
        "pangea ai guard input",
        "policyai moderation on input",
        "protect prompt",
        "trend ai guard input",
    }
)

_KNOWN_UPSTREAM_FAIL_OPEN_OUTPUT_RAILS = frozenset(
    {
        "crowdstrike aidr guard output",
        "fiddler bot faithfulness",
        "fiddler bot safety",
        "pangea ai guard output",
        "policyai moderation on output",
        "protect response",
        "trend ai guard output",
    }
)

# These exact-version surfaces can treat a syntactically successful but empty,
# partial, or ambiguous evaluator response as an allow. Activation remains
# possible because some deployments validate that response contract outside
# this worker, but the limitation must not be silent.
_UPSTREAM_AMBIGUOUS_VERDICT_INPUT_RAILS = frozenset(
    {
        "ai defense inspect prompt",
        "autoalign check input",
        "content safety check input",
        "detect pii on input",
        "gliner detect pii on input",
        "gliner mask pii on input",
        "hf classifier check input",
        "llama guard check input",
        "self check input",
        "topic safety check input",
    }
)

_UPSTREAM_AMBIGUOUS_VERDICT_OUTPUT_RAILS = frozenset(
    {
        "ai defense inspect response",
        "autoalign check output",
        "content safety check output",
        "detect pii on output",
        "gliner detect pii on output",
        "gliner mask pii on output",
        "hf classifier check output",
        "llama guard check output",
        "self check output",
    }
)


def _silence_guardrails_runtime_logs() -> None:
    """Keep Guardrails internals from logging request or evaluator content."""

    for name in ("nemoguardrails", "nemoguardrails.guardrails"):
        logger = logging.getLogger(name)
        logger.setLevel(_NO_CONTENT_LOG_LEVEL)
        for handler in logger.handlers:
            handler.setLevel(_NO_CONTENT_LOG_LEVEL)


def _error(code: str, message: str, *, field: str | None = None) -> ConfigDiagnostic:
    return ConfigDiagnostic(
        level=DiagnosticLevel.ERROR,
        code=f"nemoguardrails.nemo_relay.{code}",
        message=message,
        component=PLUGIN_ID,
        field=field,
    )


def _warning(code: str, message: str, *, field: str | None = None) -> ConfigDiagnostic:
    return ConfigDiagnostic(
        level=DiagnosticLevel.WARNING,
        code=f"nemoguardrails.nemo_relay.{code}",
        message=message,
        component=PLUGIN_ID,
        field=field,
    )


def _acknowledgement_diagnostic(
    *,
    acknowledged: bool,
    code: str,
    message: str,
    required_suffix: str,
) -> ConfigDiagnostic:
    factory = _warning if acknowledged else _error
    return factory(
        code,
        message if acknowledged else message + required_suffix,
        field="config_path",
    )


def _structural_tool_settings(rails_config: RailsConfig) -> StructuralToolSettings:
    configured = (
        ("tool_output", rails_config.rails.tool_output.flows, _STRUCTURAL_TOOL_CALL_FLOW),
        ("tool_input", rails_config.rails.tool_input.flows, _STRUCTURAL_TOOL_RESULT_FLOW),
    )
    enabled: dict[str, bool] = {}
    ignored: list[str] = []
    duplicates: list[str] = []
    for family, flows, supported in configured:
        normalized = [normalize_configured_surface_name(flow) for flow in flows]
        supported_count = normalized.count(supported)
        enabled[family] = supported_count == 1
        if supported_count > 1:
            duplicates.append(family)
        if any(flow != supported for flow in normalized):
            ignored.append(family)
    return StructuralToolSettings(
        check_calls=enabled["tool_output"],
        check_results=enabled["tool_input"],
        ignored_families=tuple(ignored),
        duplicate_families=tuple(duplicates),
    )


def _semantic_tool_diagnostics(
    settings: SemanticToolSettings,
    *,
    input_enabled: bool,
    output_enabled: bool,
) -> list[ConfigDiagnostic]:
    diagnostics: list[ConfigDiagnostic] = []
    if settings.check_arguments and not input_enabled:
        diagnostics.append(
            _error(
                "semantic_tool_input_phase_missing",
                "semantic tool argument checks require a configured Guardrails input phase",
                field="semantic_tool_policy",
            )
        )
    if settings.result_source is not SemanticResultSource.OFF and not output_enabled:
        diagnostics.append(
            _error(
                "semantic_tool_output_phase_missing",
                "semantic tool result checks require a configured Guardrails output phase",
                field="semantic_tool_policy",
            )
        )
    return diagnostics


def _phase_configuration_diagnostics(
    settings: dict[str, Json],
    *,
    input_enabled: bool,
    output_enabled: bool,
) -> list[ConfigDiagnostic]:
    diagnostics: list[ConfigDiagnostic] = []
    payload_policy = payload_policy_from_config(settings.get("payload_policy"))
    mutation_policy = mutation_policy_from_config(settings.get("mutation_policy"))
    if payload_policy.reasoning is ReasoningPolicy.CHECK_OUTPUT and not output_enabled:
        diagnostics.append(
            _error(
                "reasoning_output_phase_missing",
                "payload_policy.reasoning=check_output requires a configured Guardrails output phase",
                field="payload_policy",
            )
        )
    if mutation_policy.input is MutationMode.APPLY and not input_enabled:
        diagnostics.append(
            _error(
                "mutation_input_phase_missing",
                "mutation_policy.input=apply requires a configured Guardrails input phase",
                field="mutation_policy",
            )
        )
    if mutation_policy.output is MutationMode.APPLY and not output_enabled:
        diagnostics.append(
            _error(
                "mutation_output_phase_missing",
                "mutation_policy.output=apply requires a configured Guardrails output phase",
                field="mutation_policy",
            )
        )
    return diagnostics


def _catalog_context_diagnostics(rails_config: RailsConfig) -> list[ConfigDiagnostic]:
    """Reject catalog rails whose required context is absent from ``check_async``."""

    try:
        catalog = default_rail_catalog()
    except Exception:
        return [
            _error(
                "guardrails_catalog_unavailable",
                "the Guardrails rail catalog could not be inspected",
                field="config_path",
            )
        ]

    unsupported: list[str] = []
    for direction, configured_flows in (
        (RailDirection.INPUT, rails_config.rails.input.flows),
        (RailDirection.OUTPUT, rails_config.rails.output.flows),
    ):
        surfaces = catalog.surfaces(direction)
        for configured_flow in configured_flows:
            flow_name = normalize_configured_surface_name(configured_flow)
            surface = surfaces.get((direction, flow_name))
            if surface is None:
                continue
            missing = sorted(
                {
                    binding.key
                    for binding in surface.bindings
                    if binding.kind == "context"
                    and binding.required
                    and isinstance(binding.key, str)
                    and binding.key not in _MESSAGE_CHECK_CONTEXT_KEYS
                }
            )
            if missing:
                unsupported.append(f"{flow_name} ({', '.join(missing)})")

    if not unsupported:
        return []
    return [
        _error(
            "unsupported_catalog_context",
            "configured Guardrails catalog rails require runtime context this worker does not provide: "
            + ", ".join(sorted(unsupported)),
            field="config_path",
        )
    ]


def _is_colang_subflow(flow: dict[str, Any]) -> bool:
    if flow.get("is_subflow") is True:
        return True
    elements = flow.get("elements")
    if not isinstance(elements, list):
        return False
    return any(
        isinstance(element, dict)
        and element.get("_type") == "meta"
        and isinstance(element.get("meta"), dict)
        and element["meta"].get("subflow") is True
        for element in elements
    )


def _unselected_top_level_colang_flows(rails_config: RailsConfig) -> bool:
    configured = {
        normalize_configured_surface_name(flow)
        for flows in (
            rails_config.rails.input.flows,
            rails_config.rails.output.flows,
            rails_config.rails.retrieval.flows,
            rails_config.rails.tool_input.flows,
            rails_config.rails.tool_output.flows,
        )
        for flow in flows
    }
    for flow in rails_config.flows:
        if not isinstance(flow, dict):
            return True
        flow_id = flow.get("id")
        if isinstance(flow_id, str) and normalize_configured_surface_name(flow_id) in configured:
            continue
        if _is_colang_subflow(flow):
            continue
        return True
    return False


def _dialog_flow_diagnostics(rails_config: RailsConfig) -> list[ConfigDiagnostic]:
    if not _unselected_top_level_colang_flows(rails_config):
        return []
    return [
        _error(
            "unsupported_dialog_flows",
            "top-level Colang flows must be selected as input/output rails or declared as subflows; "
            "this worker does not run Guardrails dialog flows",
            field="config_path",
        )
    ]


def _basic_rails_config_diagnostics(rails_config: RailsConfig) -> list[ConfigDiagnostic]:
    diagnostics: list[ConfigDiagnostic] = []
    tool_settings = _structural_tool_settings(rails_config)
    content_safety = getattr(rails_config.rails.config, "content_safety", None)
    multilingual = getattr(content_safety, "multilingual", None)
    checks = (
        (
            not rails_config.rails.input.flows and not rails_config.rails.output.flows and not tool_settings.enabled,
            "missing_supported_rails",
            "the Relay worker requires at least one input rail, output rail, or supported structural tool rail",
        ),
        (
            rails_config.enable_rails_exceptions,
            "unsupported_rails_exceptions",
            "enable_rails_exceptions is not supported by this message-check integration",
        ),
        (
            bool(rails_config.rails.output.streaming.model_dump(exclude_defaults=True)),
            "unsupported_streaming_output_rails",
            "Guardrails output-streaming settings cannot be applied by this worker",
        ),
        (
            bool(rails_config.rails.input.speculative_generation),
            "unsupported_speculative_generation",
            "rails.input.speculative_generation is not used by this LLMRails message-check worker",
        ),
        (
            bool(rails_config.rails.actions.instant_actions),
            "unsupported_instant_actions",
            "rails.actions.instant_actions applies to Guardrails-owned generation, not this message-check worker",
        ),
        (rails_config.metrics.enabled, "unsupported_metrics", "Guardrails metrics are not exported by this worker"),
        (
            bool(rails_config.rails.tool_input.parallel or rails_config.rails.tool_output.parallel),
            "unsupported_parallel_tool_rails",
            "parallel structural tool rails are not supported by this worker",
        ),
        (
            bool(getattr(multilingual, "enabled", False)),
            "unsupported_multilingual_refusal",
            "content_safety.multilingual changes Guardrails refusal text, but this worker returns a typed policy "
            "rejection instead of a generated refusal message",
        ),
    )
    diagnostics.extend(_error(code, message, field="config_path") for active, code, message in checks if active)
    if tool_settings.duplicate_families:
        diagnostics.append(
            _error(
                "duplicate_structural_tool_rails",
                "structural tool rail flows must not be repeated",
                field="config_path",
            )
        )
    if rails_config.tracing.enabled:
        diagnostics.append(
            _error(
                "unsupported_guardrails_tracing",
                "Guardrails tracing is not packaged or qualified by this worker",
                field="config_path",
            )
        )
    return diagnostics


def _runtime_scope_diagnostics(
    rails_config: RailsConfig,
    runtime_plan: RuntimePlan,
    *,
    evaluator_framework: EvaluatorFramework,
    action_safety_configured: bool,
    evaluator_framework_configured: bool,
) -> list[ConfigDiagnostic]:
    """Reject settings that the selected ``check_async`` runtime cannot use."""

    diagnostics: list[ConfigDiagnostic] = []
    explicitly_configured = rails_config.model_fields_set
    enabled_generation_only = (
        ("enable_multi_step_generation", "unsupported_multi_step_generation"),
        ("passthrough", "unsupported_passthrough"),
    )
    for field, code in enabled_generation_only:
        if field in explicitly_configured and bool(getattr(rails_config, field)):
            diagnostics.append(
                _error(
                    code,
                    f"{field} applies to Guardrails-owned generation and is not used by this check_async worker",
                    field="config_path",
                )
            )
    if "raw_llm_call_action" in explicitly_configured and rails_config.raw_llm_call_action != "raw llm call":
        diagnostics.append(
            _error(
                "unsupported_raw_llm_call_action",
                "Guardrails 0.24.1 check_async does not consume a custom raw_llm_call_action selection",
                field="config_path",
            )
        )

    # Guardrails 0.24 already marks this top-level field as deprecated and
    # ignored. Reject only the active value so old configurations that spell
    # out ``streaming: false`` do not fail activation unnecessarily.
    streaming_enabled = bool(rails_config.model_dump(include={"streaming"}).get("streaming"))
    if "streaming" in explicitly_configured and streaming_enabled:
        diagnostics.append(
            _error(
                "unsupported_top_level_streaming",
                "top-level streaming is deprecated by Guardrails and cannot enable streaming checks in this worker",
                field="config_path",
            )
        )

    text_enabled = bool(rails_config.rails.input.flows or rails_config.rails.output.flows)
    llmrails_text_runtime = text_enabled and not runtime_plan.uses_iorails

    # Guardrails 0.24.1 gives its two local engines different names for the
    # same transport controls. Unknown model parameters become provider body
    # fields, so accepting the other engine's spelling turns a reasonable
    # retry/timeout setting into an opaque provider 400. Do not rewrite the
    # numeric value: ``max_attempts`` counts total attempts while
    # ``max_retries`` counts retries.
    if text_enabled:
        if runtime_plan.uses_iorails:
            unsupported_transport_parameters = {
                "max_retries": "max_attempts",
                "connect_timeout": "timeout_connect",
            }
            engine_name = "IORails"
        elif evaluator_framework is EvaluatorFramework.DEFAULT:
            unsupported_transport_parameters = {
                "max_attempts": "max_retries",
                "timeout_connect": "connect_timeout",
            }
            engine_name = "LLMRails default framework"
        else:
            unsupported_transport_parameters = {}
            engine_name = ""

        mismatches = sorted(
            {
                (name, replacement)
                for model in rails_config.models
                for name, replacement in unsupported_transport_parameters.items()
                if name in (model.parameters or {})
            }
        )
        if mismatches:
            rendered = ", ".join(f"{name} (use {replacement})" for name, replacement in mismatches)
            diagnostics.append(
                _error(
                    "evaluator_transport_parameter_mismatch",
                    f"{engine_name} does not consume these model transport parameters: {rendered}; "
                    "Guardrails would forward them to the provider request body",
                    field="config_path",
                )
            )

    if action_safety_configured and not llmrails_text_runtime:
        diagnostics.append(
            _error(
                "unsupported_action_safety_runtime",
                "action_safety applies only to a local LLMRails input/output runtime",
                field="action_safety",
            )
        )
    if evaluator_framework_configured and not text_enabled:
        diagnostics.append(
            _error(
                "unsupported_evaluator_framework_without_text_rails",
                "evaluator_framework requires at least one local Guardrails input or output rail",
                field="evaluator_framework",
            )
        )
    if rails_config.actions_server_url is not None and not llmrails_text_runtime:
        diagnostics.append(
            _error(
                "unsupported_actions_server_runtime",
                "actions_server_url applies only to a local LLMRails input/output runtime",
                field="config_path",
            )
        )
    return diagnostics


def _ignored_family_diagnostics(
    rails_config: RailsConfig,
    *,
    acknowledged: bool,
) -> list[ConfigDiagnostic]:
    tool_settings = _structural_tool_settings(rails_config)
    ignored_families = [name for name, flows in (("retrieval", rails_config.rails.retrieval.flows),) if flows]
    ignored_families.extend(tool_settings.ignored_families)
    if (
        rails_config.user_messages or rails_config.rails.dialog.model_dump(exclude_defaults=True)
    ) and not _unselected_top_level_colang_flows(rails_config):
        ignored_families.append("dialog")
    if not ignored_families:
        return []
    return [
        _acknowledgement_diagnostic(
            acknowledged=acknowledged,
            code="ignored_rail_families",
            message="this worker will not run configured rail families: " + ", ".join(ignored_families),
            required_suffix="; remove them or explicitly acknowledge this loss",
        )
    ]


def _configured_fail_open_flows(
    rails_config: RailsConfig,
    input_flow_names: set[str],
    output_flow_names: set[str],
) -> list[str]:
    configured = []
    for flow_name, config_name, configured_names in (
        ("ai defense inspect prompt", "ai_defense", input_flow_names),
        ("ai defense inspect response", "ai_defense", output_flow_names),
        ("f5 guardrails scan input", "f5", input_flow_names),
        ("f5 guardrails scan output", "f5", output_flow_names),
    ):
        rail_config = getattr(rails_config.rails.config, config_name, None)
        if flow_name in configured_names and getattr(rail_config, "fail_open", False):
            configured.append(flow_name)
    return configured


def _upstream_contract_diagnostics(
    rails_config: RailsConfig,
    *,
    acknowledged: bool,
) -> list[ConfigDiagnostic]:
    input_flow_names = {normalize_configured_surface_name(flow) for flow in rails_config.rails.input.flows}
    output_flow_names = {normalize_configured_surface_name(flow) for flow in rails_config.rails.output.flows}
    known_fail_open = sorted(
        (input_flow_names & _KNOWN_UPSTREAM_FAIL_OPEN_INPUT_RAILS)
        | (output_flow_names & _KNOWN_UPSTREAM_FAIL_OPEN_OUTPUT_RAILS)
    )
    configured_fail_open = _configured_fail_open_flows(rails_config, input_flow_names, output_flow_names)
    ambiguous_verdicts = sorted(
        (input_flow_names & _UPSTREAM_AMBIGUOUS_VERDICT_INPUT_RAILS)
        | (output_flow_names & _UPSTREAM_AMBIGUOUS_VERDICT_OUTPUT_RAILS)
    )
    diagnostics: list[ConfigDiagnostic] = []
    if known_fail_open:
        diagnostics.append(
            _acknowledgement_diagnostic(
                acknowledged=acknowledged,
                code="known_upstream_fail_open_rails",
                message="Guardrails 0.24 can allow operational failures in configured rails: "
                + ", ".join(known_fail_open),
                required_suffix="; explicit acknowledgement is required",
            )
        )
    if configured_fail_open:
        diagnostics.append(
            _acknowledgement_diagnostic(
                acknowledged=acknowledged,
                code="configured_upstream_fail_open_rails",
                message="configured rails permit evaluator failures to pass: " + ", ".join(configured_fail_open),
                required_suffix="; set fail_open=false or acknowledge it explicitly",
            )
        )
    if ambiguous_verdicts:
        diagnostics.append(
            _warning(
                "upstream_verdict_contract",
                "validate successful evaluator response parsing before production for rails: "
                + ", ".join(ambiguous_verdicts),
                field="config_path",
            )
        )
    return diagnostics


def _load_supported_rails_config(
    config_path: str,
    *,
    allow_ignored_rail_families: bool,
    allow_known_fail_open_rails: bool,
    evaluator_framework: EvaluatorFramework,
    action_safety_configured: bool,
    evaluator_framework_configured: bool,
) -> tuple[RailsConfig | None, RuntimePlan | None, list[ConfigDiagnostic]]:
    """Parse and validate Guardrails configuration without starting a runtime."""

    _silence_guardrails_runtime_logs()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rails_config = RailsConfig.from_path(config_path)
    except Exception:
        return (
            None,
            None,
            [_error("invalid_guardrails_config", "Guardrails configuration could not be loaded", field="config_path")],
        )

    action_server_diagnostics: list[ConfigDiagnostic] = []
    try:
        validated_actions_server_url(rails_config.actions_server_url)
    except ValueError:
        action_server_diagnostics.append(
            _error(
                "invalid_actions_server_url",
                "Guardrails actions_server_url must be a credential-free HTTPS origin or a loopback HTTP origin",
                field="config_path",
            )
        )

    runtime_plan = select_runtime(config_path, rails_config, evaluator_framework=evaluator_framework.value)
    if rails_config.colang_version == "2.x":
        return (
            rails_config,
            runtime_plan,
            [
                _error(
                    "unsupported_colang_version",
                    "the Relay message-check worker supports Colang 1.0 only; Colang 2 requires native input/output "
                    "entrypoint discovery that this worker does not yet expose",
                    field="config_path",
                )
            ],
        )
    if rails_config.colang_version != "1.0":
        return (
            rails_config,
            runtime_plan,
            [
                _error(
                    "unsupported_colang_version",
                    "the Guardrails configuration uses an unsupported Colang version",
                    field="config_path",
                )
            ],
        )
    diagnostics = action_server_diagnostics
    diagnostics.extend(_basic_rails_config_diagnostics(rails_config))
    diagnostics.extend(
        _runtime_scope_diagnostics(
            rails_config,
            runtime_plan,
            evaluator_framework=evaluator_framework,
            action_safety_configured=action_safety_configured,
            evaluator_framework_configured=evaluator_framework_configured,
        )
    )
    diagnostics.extend(_catalog_context_diagnostics(rails_config))
    diagnostics.extend(_dialog_flow_diagnostics(rails_config))
    diagnostics.extend(_ignored_family_diagnostics(rails_config, acknowledged=allow_ignored_rail_families))
    diagnostics.extend(_upstream_contract_diagnostics(rails_config, acknowledged=allow_known_fail_open_rails))
    if rails_config.rails.input.flows or rails_config.rails.output.flows:
        diagnostics.extend(
            _error(issue.code, issue.message, field="evaluator_framework")
            for issue in evaluator_framework_issues(rails_config, evaluator_framework)
        )
    if not any(item.level == DiagnosticLevel.ERROR for item in diagnostics):
        diagnostics.extend(
            _error(issue.code, issue.message, field="config_path")
            for issue in preflight_runtime(runtime_plan, rails_config)
        )
    return rails_config, runtime_plan, diagnostics
