# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration validation and resolution for the Relay Guardrails worker."""

from __future__ import annotations

import logging
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# This module can be imported independently in tests and tooling. Keep the
# telemetry boundary intact even when ``worker`` is not the first import.
os.environ["NEMO_GUARDRAILS_NO_USAGE_STATS"] = "1"

import nemoguardrails  # noqa: E402
from nemo_relay_plugin import ConfigDiagnostic, DiagnosticLevel, Json  # noqa: E402
from nemoguardrails import RailsConfig  # noqa: E402
from nemoguardrails.manifests import (  # noqa: E402
    RailDirection,
    default_rail_catalog,
    normalize_configured_surface_name,
)

from .action_safety import action_safety_settings_from_config  # noqa: E402
from .check_backend import RemoteChecksSettings  # noqa: E402
from .evaluator_framework import (  # noqa: E402
    EvaluatorFramework,
    evaluator_framework_from_config,
    evaluator_framework_issues,
)
from .payload_policy import ReasoningPolicy, payload_policy_from_config  # noqa: E402
from .remote_actions import validated_actions_server_url  # noqa: E402
from .runtime_selection import RuntimePlan, preflight_runtime, select_runtime  # noqa: E402
from .semantic_tools import (  # noqa: E402
    SemanticResultSource,
    SemanticToolSettings,
    semantic_tool_settings_from_config,
)
from .structural_tools import StructuralToolSettings  # noqa: E402
from .text_mutation import MutationMode, mutation_policy_from_config  # noqa: E402

PLUGIN_ID = "nemoguardrails.nemo_relay"
CONFIG_VERSION = 2
SUPPORTED_NEMOGUARDRAILS_VERSION = "0.24.1"
DEFAULT_CHECK_TIMEOUT_MS = 25_000
RELAY_WORKER_TIMEOUT_MS = 30_000
MIN_CHECK_TIMEOUT_MS = 1_000
MAX_CHECK_TIMEOUT_MS = RELAY_WORKER_TIMEOUT_MS - 5_000
MAX_SECRET_ENV_ENTRIES = 32
MAX_SECRET_ENV_VALUE_CHARACTERS = 65_536

_SAFE_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_RESERVED_SECRET_ENV_NAMES = frozenset({"nemo_guardrails_no_usage_stats"})
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

_MISSING_ENV_VALUE = object()


class _SecretEnvironment:
    """Apply validated secrets without retaining their configured values."""

    def __init__(self) -> None:
        self._previous: dict[str, str | object] | None = None

    def install(self, values: dict[str, str]) -> None:
        if self._previous is not None:
            raise RuntimeError("secret environment is already installed")

        previous: dict[str, str | object] = {}
        try:
            for name, value in values.items():
                previous[name] = os.environ.get(name, _MISSING_ENV_VALUE)
                os.environ[name] = value
        except BaseException:
            self._restore(previous)
            raise RuntimeError("secret environment could not be installed") from None
        self._previous = previous

    @staticmethod
    def _restore(previous: dict[str, str | object]) -> None:
        for name, value in previous.items():
            if value is _MISSING_ENV_VALUE:
                os.environ.pop(name, None)
            else:
                os.environ[name] = cast(str, value)

    def restore(self) -> None:
        previous = self._previous
        self._previous = None
        if previous is not None:
            self._restore(previous)


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


def _validate_config_path(value: Any) -> list[ConfigDiagnostic]:
    if not isinstance(value, str) or not value.strip():
        return [
            _error(
                "invalid_config_path",
                "config_path must be a non-empty absolute path",
                field="config_path",
            )
        ]

    path = Path(value)
    if not path.is_absolute():
        return [_error("invalid_config_path", "config_path must be absolute", field="config_path")]
    if not path.is_dir():
        return [
            _error(
                "missing_config_path",
                "config_path must identify an existing directory",
                field="config_path",
            )
        ]
    return []


def _validate_timeout(value: Any) -> list[ConfigDiagnostic]:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < MIN_CHECK_TIMEOUT_MS
        or value > MAX_CHECK_TIMEOUT_MS
    ):
        return [
            _error(
                "invalid_check_timeout",
                f"check_timeout_ms must be an integer between {MIN_CHECK_TIMEOUT_MS} and {MAX_CHECK_TIMEOUT_MS}",
                field="check_timeout_ms",
            )
        ]
    return []


def _validate_secret_env(value: Any) -> list[ConfigDiagnostic]:
    """Validate a portable environment overlay without exposing its contents."""

    if not isinstance(value, dict):
        return [
            _error(
                "invalid_secret_env",
                "secret_env must be an object of environment names and non-empty string values",
                field="secret_env",
            )
        ]
    if len(value) > MAX_SECRET_ENV_ENTRIES:
        return [
            _error(
                "invalid_secret_env",
                f"secret_env must contain at most {MAX_SECRET_ENV_ENTRIES} entries",
                field="secret_env",
            )
        ]

    diagnostics: list[ConfigDiagnostic] = []
    names: set[str] = set()
    for name, secret in value.items():
        normalized_name = name.casefold() if isinstance(name, str) else None
        if not isinstance(name, str) or _SAFE_ENV_NAME.fullmatch(name) is None or normalized_name in names:
            diagnostics.append(
                _error(
                    "invalid_secret_env_name",
                    "secret_env contains an invalid or case-insensitively duplicate environment name",
                    field="secret_env",
                )
            )
        else:
            names.add(normalized_name)
            if normalized_name in _RESERVED_SECRET_ENV_NAMES:
                diagnostics.append(
                    _error(
                        "reserved_secret_env_name",
                        "secret_env cannot override the Guardrails usage-telemetry opt-out",
                        field="secret_env",
                    )
                )
        if (
            not isinstance(secret, str)
            or not secret
            or "\x00" in secret
            or len(secret) > MAX_SECRET_ENV_VALUE_CHARACTERS
        ):
            diagnostics.append(
                _error(
                    "invalid_secret_env_value",
                    "secret_env values must be non-empty strings within the supported size limit",
                    field="secret_env",
                )
            )
    return diagnostics


def _configured_secret_env(config: Json) -> dict[str, str]:
    if not isinstance(config, dict):
        return {}
    return cast(dict[str, str], config.get("secret_env", {}))


def _validate_wrapper_config(config: Json) -> list[ConfigDiagnostic]:
    if not isinstance(config, dict):
        return [_error("invalid_config", "plugin configuration must be an object")]

    diagnostics: list[ConfigDiagnostic] = []
    allowed = {
        "version",
        "config_path",
        "remote_checks",
        "check_timeout_ms",
        "allow_ignored_rail_families",
        "allow_known_fail_open_rails",
        "payload_policy",
        "mutation_policy",
        "semantic_tool_policy",
        "action_safety",
        "evaluator_framework",
        "secret_env",
    }
    for key in config.keys() - allowed:
        diagnostics.append(_error("unknown_field", f"unknown field '{key}'", field=str(key)))

    version = config.get("version", CONFIG_VERSION)
    if isinstance(version, bool) or not isinstance(version, int) or version != CONFIG_VERSION:
        diagnostics.append(
            _error(
                "unsupported_version",
                f"version must be {CONFIG_VERSION}",
                field="version",
            )
        )

    has_local_config = "config_path" in config
    has_remote_config = "remote_checks" in config
    if has_local_config == has_remote_config:
        diagnostics.append(
            _error(
                "invalid_check_backend",
                "configure exactly one of config_path or remote_checks",
            )
        )
    elif has_local_config:
        diagnostics.extend(_validate_config_path(config.get("config_path")))
    else:
        try:
            RemoteChecksSettings.from_config(config.get("remote_checks"))
        except ValueError as error:
            diagnostics.append(_error("invalid_remote_checks", str(error), field="remote_checks"))

    diagnostics.extend(_validate_timeout(config.get("check_timeout_ms", DEFAULT_CHECK_TIMEOUT_MS)))
    diagnostics.extend(_validate_secret_env(config.get("secret_env", {})))
    try:
        payload_policy_from_config(config.get("payload_policy"))
    except ValueError as error:
        diagnostics.append(_error("invalid_payload_policy", str(error), field="payload_policy"))
    try:
        mutation_policy_from_config(config.get("mutation_policy"))
    except ValueError as error:
        diagnostics.append(_error("invalid_mutation_policy", str(error), field="mutation_policy"))
    try:
        semantic_tool_settings_from_config(config.get("semantic_tool_policy"))
    except ValueError as error:
        diagnostics.append(_error("invalid_semantic_tool_policy", str(error), field="semantic_tool_policy"))
    try:
        action_safety_settings_from_config(config.get("action_safety"))
    except ValueError as error:
        diagnostics.append(_error("invalid_action_safety", str(error), field="action_safety"))
    try:
        evaluator_framework_from_config(config.get("evaluator_framework"))
    except ValueError as error:
        diagnostics.append(_error("invalid_evaluator_framework", str(error), field="evaluator_framework"))

    if has_remote_config:
        for field in ("action_safety", "evaluator_framework"):
            if field in config:
                diagnostics.append(
                    _error(
                        "unsupported_remote_setting",
                        f"{field} is owned by the remote Guardrails service and cannot be set in this worker",
                        field=field,
                    )
                )
        remote_value = config.get("remote_checks")
        if isinstance(remote_value, dict):
            remote_timeout_ms = remote_value.get("timeout_ms", DEFAULT_CHECK_TIMEOUT_MS)
            check_timeout_ms = config.get("check_timeout_ms", DEFAULT_CHECK_TIMEOUT_MS)
            if (
                isinstance(remote_timeout_ms, int)
                and not isinstance(remote_timeout_ms, bool)
                and isinstance(check_timeout_ms, int)
                and not isinstance(check_timeout_ms, bool)
                and remote_timeout_ms > check_timeout_ms
            ):
                diagnostics.append(
                    _error(
                        "invalid_remote_timeout",
                        "remote_checks.timeout_ms cannot exceed check_timeout_ms",
                        field="remote_checks",
                    )
                )

    for field in ("allow_ignored_rail_families", "allow_known_fail_open_rails"):
        if not isinstance(config.get(field, False), bool):
            diagnostics.append(
                _error(
                    "invalid_boolean",
                    f"{field} must be a boolean",
                    field=field,
                )
            )

    if getattr(nemoguardrails, "__version__", None) != SUPPORTED_NEMOGUARDRAILS_VERSION:
        diagnostics.append(
            _error(
                "unsupported_guardrails_version",
                f"this worker requires NeMo Guardrails {SUPPORTED_NEMOGUARDRAILS_VERSION}",
            )
        )

    return diagnostics


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


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Named result of resolving wrapper and Guardrails configuration."""

    settings: dict[str, Json] | None
    rails_config: RailsConfig | None
    runtime_plan: RuntimePlan | None
    diagnostics: tuple[ConfigDiagnostic, ...]


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


def _resolve_config(
    config: Json,
) -> ResolvedConfig:
    """Validate both the worker wrapper and nested Guardrails configuration."""

    diagnostics = _validate_wrapper_config(config)
    if any(item.level == DiagnosticLevel.ERROR for item in diagnostics):
        return ResolvedConfig(None, None, None, tuple(diagnostics))

    settings = cast(dict[str, Json], config)
    semantic_settings = semantic_tool_settings_from_config(settings.get("semantic_tool_policy"))
    if "remote_checks" in settings:
        try:
            remote_settings = RemoteChecksSettings.from_config(settings["remote_checks"])
            remote_settings.headers()
        except ValueError:
            return ResolvedConfig(
                settings,
                None,
                None,
                (
                    *diagnostics,
                    _error(
                        "invalid_remote_check_credentials",
                        "remote check credentials are not available",
                        field="remote_checks",
                    ),
                ),
            )
        input_enabled = "input" in remote_settings.phases
        output_enabled = "output" in remote_settings.phases
        diagnostics.extend(
            _semantic_tool_diagnostics(semantic_settings, input_enabled=input_enabled, output_enabled=output_enabled)
        )
        diagnostics.extend(
            _phase_configuration_diagnostics(settings, input_enabled=input_enabled, output_enabled=output_enabled)
        )
        return ResolvedConfig(settings, None, None, tuple(diagnostics))

    evaluator_framework = evaluator_framework_from_config(settings.get("evaluator_framework"))
    rails_config, runtime_plan, guardrails_diagnostics = _load_supported_rails_config(
        cast(str, settings["config_path"]),
        allow_ignored_rail_families=cast(bool, settings.get("allow_ignored_rail_families", False)),
        allow_known_fail_open_rails=cast(bool, settings.get("allow_known_fail_open_rails", False)),
        evaluator_framework=evaluator_framework,
        action_safety_configured="action_safety" in settings,
        evaluator_framework_configured="evaluator_framework" in settings,
    )
    if rails_config is not None:
        input_enabled = bool(rails_config.rails.input.flows)
        output_enabled = bool(rails_config.rails.output.flows)
        guardrails_diagnostics.extend(
            _semantic_tool_diagnostics(semantic_settings, input_enabled=input_enabled, output_enabled=output_enabled)
        )
        guardrails_diagnostics.extend(
            _phase_configuration_diagnostics(settings, input_enabled=input_enabled, output_enabled=output_enabled)
        )
    diagnostics.extend(guardrails_diagnostics)
    return ResolvedConfig(settings, rails_config, runtime_plan, tuple(diagnostics))


def _validate_config(config: Json) -> list[ConfigDiagnostic]:
    """Return complete activation diagnostics without constructing Guardrails."""

    diagnostics = _validate_wrapper_config(config)
    if any(item.level == DiagnosticLevel.ERROR for item in diagnostics):
        return diagnostics

    environment = _SecretEnvironment()
    try:
        environment.install(_configured_secret_env(config))
    except Exception:
        return [
            _error(
                "secret_environment_failed",
                "secret_env could not be applied while validating the Guardrails configuration",
                field="secret_env",
            )
        ]

    try:
        return list(_resolve_config(config).diagnostics)
    except Exception:
        return [
            _error(
                "configuration_resolution_failed",
                "the Guardrails configuration could not be resolved",
            )
        ]
    finally:
        environment.restore()
