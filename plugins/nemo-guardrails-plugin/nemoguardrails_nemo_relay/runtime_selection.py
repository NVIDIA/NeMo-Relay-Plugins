# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select and preflight the Guardrails engine used by the Relay worker.

Guardrails normally falls back from IORails to LLMRails when a configuration is
not compatible with IORails.  Silent fallback is useful for applications, but
it is a poor fit for a policy worker: activation and request-time behavior can
otherwise use different engines without an operator-visible configuration
change.  This module makes the choice explicit and keeps it independent from
Relay payload projection.
"""

from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from nemoguardrails import RailsConfig
from nemoguardrails.guardrails.compiled_rail import unsupported_surface_reason
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.llm.constants import AZURE_PROVIDERS
from nemoguardrails.manifests import (
    RailDirection,
    default_rail_catalog,
    normalize_configured_surface_name,
    parse_configured_surface,
)

from .integration_safety import guardrails_ai_privacy_issue, uses_guardrails_ai

# Guardrails' default framework owns these provider presets. IORails only
# supplies native URL and credential defaults for ``openai`` and ``nim``.
_LLMRAILS_PROVIDER_PRESETS = AZURE_PROVIDERS | {"nvidia_ai_endpoints", "ollama"}


class GuardrailsEngine(str, Enum):
    """The engine this worker will construct."""

    IORAILS = "iorails"
    LLMRAILS = "llmrails"


class RuntimeReason(str, Enum):
    """Why an engine was selected, kept stable for tests and diagnostics."""

    CATALOG = "catalog"
    CUSTOM_COLANG = "custom_colang"
    CUSTOM_ACTIONS = "custom_actions"
    COLANG_2 = "colang_2"
    IORAILS_INCOMPATIBLE = "iorails_incompatible"
    EVALUATOR_FRAMEWORK = "evaluator_framework"


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    """One deterministic Guardrails runtime decision."""

    engine: GuardrailsEngine
    reason: RuntimeReason
    colang_version: str

    @property
    def uses_iorails(self) -> bool:
        return self.engine is GuardrailsEngine.IORAILS

    def constructor_options(self) -> dict[str, bool]:
        """Return flags that prohibit Guardrails' implicit engine fallback."""

        if self.uses_iorails:
            return {"use_iorails": True, "require_iorails": True}
        return {"use_iorails": False, "require_iorails": False}


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    """A content-safe activation issue returned to the worker validator."""

    code: str
    message: str


def _configuration_paths(config_path: str, rails_config: RailsConfig) -> tuple[Path, ...]:
    paths = [Path(config_path)]
    imported = rails_config.imported_paths or {}
    paths.extend(Path(value) for value in imported.values() if isinstance(value, str))
    return tuple(paths)


def _has_config_module(config_path: str, rails_config: RailsConfig) -> bool:
    for path in _configuration_paths(config_path, rails_config):
        candidate = path.parent / "config.py" if path.is_file() else path / "config.py"
        if candidate.is_file():
            return True
    return False


def _configured_text_flows(rails_config: RailsConfig) -> tuple[tuple[RailDirection, str], ...]:
    configured: list[tuple[RailDirection, str]] = []
    for direction, flows in (
        (RailDirection.INPUT, rails_config.rails.input.flows),
        (RailDirection.OUTPUT, rails_config.rails.output.flows),
    ):
        configured.extend((direction, flow) for flow in flows)
    return tuple(configured)


def _contains_custom_text_flow(rails_config: RailsConfig) -> bool:
    """Return whether a selected input/output flow is outside the catalog.

    A configuration can contain helper Colang flows while selecting only
    manifest-backed rails.  Those helpers still require LLMRails, because
    IORails deliberately does not execute arbitrary Colang.
    """

    if rails_config.flows:
        return True

    catalog = default_rail_catalog()
    surfaces = catalog.surfaces()
    return any(
        (direction, normalize_configured_surface_name(flow)) not in surfaces
        for direction, flow in _configured_text_flows(rails_config)
    )


def _has_static_iorails_incompatibility(rails_config: RailsConfig) -> bool:
    """Detect shape-level incompatibility without treating missing resources as fallback."""

    if any(model.engine in _LLMRAILS_PROVIDER_PRESETS for model in rails_config.models):
        return True
    if rails_config.rails.model_fields_set - IORails.SUPPORTED_RAILS:
        return True
    catalog = default_rail_catalog()
    surfaces = catalog.surfaces()
    for direction, flow in _configured_text_flows(rails_config):
        surface = surfaces.get((direction, normalize_configured_surface_name(flow)))
        if surface is not None and unsupported_surface_reason(surface) is not None:
            return True
    return False


def select_runtime(
    config_path: str,
    rails_config: RailsConfig,
    *,
    evaluator_framework: str = "default",
) -> RuntimePlan:
    """Select an engine from configuration shape, never runtime failure.

    Missing dependencies must not turn an intended IORails configuration into
    an LLMRails configuration.  They are reported separately by ``preflight``.
    """

    if rails_config.colang_version == "2.x":
        # IORails 0.24 is Colang 1-only.  Keep this explicit LLMRails decision
        # even while the Relay message-check contract for native Colang 2 flows
        # remains a separately diagnosed integration limitation.
        return RuntimePlan(GuardrailsEngine.LLMRAILS, RuntimeReason.COLANG_2, rails_config.colang_version)
    if _has_config_module(config_path, rails_config):
        return RuntimePlan(GuardrailsEngine.LLMRAILS, RuntimeReason.CUSTOM_ACTIONS, rails_config.colang_version)
    if _contains_custom_text_flow(rails_config):
        return RuntimePlan(GuardrailsEngine.LLMRAILS, RuntimeReason.CUSTOM_COLANG, rails_config.colang_version)
    if evaluator_framework == "langchain":
        # IORails 0.24 owns an OpenAI-compatible ModelEngine and does not use
        # Guardrails' pluggable LLM framework registry. Provider-native
        # evaluators such as Anthropic therefore require LLMRails.
        return RuntimePlan(
            GuardrailsEngine.LLMRAILS,
            RuntimeReason.EVALUATOR_FRAMEWORK,
            rails_config.colang_version,
        )
    if _has_static_iorails_incompatibility(rails_config):
        return RuntimePlan(
            GuardrailsEngine.LLMRAILS,
            RuntimeReason.IORAILS_INCOMPATIBLE,
            rails_config.colang_version,
        )
    return RuntimePlan(GuardrailsEngine.IORAILS, RuntimeReason.CATALOG, rails_config.colang_version)


def _selected_manifest_flows(rails_config: RailsConfig) -> tuple[tuple[str, str], ...]:
    catalog = default_rail_catalog()
    selected: set[tuple[str, str]] = set()
    for _, flow in _configured_text_flows(rails_config):
        owner = catalog.owner_for_flow(normalize_configured_surface_name(flow))
        if owner is not None:
            selected.add((owner, flow))
    return tuple(sorted(selected))


def _distribution_is_installed(name: str) -> bool:
    try:
        importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _uses_local_optional_dependencies(name: str, flow: str, rails_config: RailsConfig) -> bool:
    """Match Guardrails 0.24's two configurable local/remote backends."""

    rail_config = rails_config.rails.config
    if name == "hf_classifier":
        try:
            _, params = parse_configured_surface(flow)
        except ValueError:
            return True
        classifiers = rail_config.hf_classifier
        if not classifiers:
            return True
        selected = classifiers.get(params.get("classifier", ""))
        return selected is None or selected.engine == "local"
    if name == "jailbreak_detection":
        configured = rail_config.jailbreak_detection
        return configured is None or not (configured.server_endpoint or configured.nim_base_url)
    return True


def _manifest_requirement_issues(rails_config: RailsConfig) -> list[PreflightIssue]:
    """Validate requirements that Guardrails otherwise defers to a request.

    IORails' compiler checks action imports, configured model bindings and most
    optional packages.  Manifests also declare required environment variables
    and implicit ``llm`` requirements, which are not all compiler bindings, so
    check those here for both engines.
    """

    catalog = default_rail_catalog()
    configured_model_types = {model.type for model in rails_config.models}
    missing_distributions: set[str] = set()
    missing_environment = {
        model.api_key_env_var
        for model in rails_config.models
        if model.api_key_env_var and not os.environ.get(model.api_key_env_var)
    }
    missing_model_types: set[str] = set()

    for name, flow in _selected_manifest_flows(rails_config):
        manifest = catalog.manifests[name]
        requirements = manifest.spec.requirements
        if _uses_local_optional_dependencies(name, flow, rails_config):
            missing_distributions.update(
                dependency
                for dependency in requirements.optional_dependencies
                if not _distribution_is_installed(dependency)
            )
        missing_environment.update(
            variable.name
            for variable in requirements.env_vars
            if variable.required and not os.environ.get(variable.name)
        )
        for model in requirements.models:
            if not model.required:
                continue
            if model.type == "llm":
                if "main" not in configured_model_types:
                    missing_model_types.add("main")
            elif model.type.startswith("spacy:"):
                package = model.type.partition(":")[2]
                if package and not _distribution_is_installed(package.replace("_", "-")):
                    missing_model_types.add(model.type)
            elif ":" not in model.type and model.type not in configured_model_types:
                missing_model_types.add(model.type)

    issues: list[PreflightIssue] = []
    if missing_distributions:
        issues.append(
            PreflightIssue(
                "missing_guardrails_dependencies",
                "configured Guardrails rails require missing Python distributions: "
                + ", ".join(sorted(missing_distributions)),
            )
        )
    if missing_environment:
        issues.append(
            PreflightIssue(
                "missing_guardrails_environment",
                "Guardrails configuration requires unset environment variables: "
                + ", ".join(sorted(missing_environment)),
            )
        )
    if missing_model_types:
        issues.append(
            PreflightIssue(
                "missing_guardrails_models",
                "configured Guardrails rails require model types that are not configured: "
                + ", ".join(sorted(missing_model_types)),
            )
        )
    if uses_guardrails_ai(rails_config) and _distribution_is_installed("guardrails-ai"):
        privacy_issue = guardrails_ai_privacy_issue(rails_config)
        if privacy_issue is not None:
            issues.append(PreflightIssue("guardrails_ai_privacy", privacy_issue))
    return issues


def preflight_runtime(plan: RuntimePlan, rails_config: RailsConfig) -> list[PreflightIssue]:
    """Prove the selected runtime can start without executing trusted config.py."""

    issues = _manifest_requirement_issues(rails_config)
    if plan.uses_iorails:
        try:
            reason = IORails.unsupported_reason(rails_config)
        except Exception:
            reason = "catalog inspection failed"
        if reason is not None:
            # Do not copy Guardrails' reason across the control-plane boundary:
            # it can contain operator-authored flow strings and local details.
            issues.append(
                PreflightIssue(
                    "iorails_preflight_failed",
                    "configured catalog rails cannot be compiled by NeMo Guardrails IORails; "
                    "verify rail names, optional dependencies, model bindings, and configuration",
                )
            )
    return issues


def runtime_uses_explain_state(rails: Any) -> bool:
    """Return whether request diagnostics must be scrubbed after each check."""

    return not bool(getattr(rails, "use_iorails_engine", False))
