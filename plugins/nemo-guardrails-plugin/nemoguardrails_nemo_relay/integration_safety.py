# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safety checks for optional libraries loaded by Guardrails rails.

Optional integrations run in the same Python worker as this plugin.  Their
defaults therefore become part of the plugin's data-handling contract even
though their code is not bundled in the base archive.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

from nemoguardrails import RailsConfig
from nemoguardrails.manifests import default_rail_catalog, normalize_configured_surface_name


class OptionalIntegrationSafetyError(RuntimeError):
    """An optional integration is installed but not safe to start."""


def uses_guardrails_ai(rails_config: RailsConfig) -> bool:
    """Return whether a selected input/output surface belongs to Guardrails AI."""

    catalog = default_rail_catalog()
    flows = (*rails_config.rails.input.flows, *rails_config.rails.output.flows)
    return any(catalog.owner_for_flow(normalize_configured_surface_name(flow)) == "guardrails_ai" for flow in flows)


def _guardrails_ai_runtime() -> tuple[Any, Any]:
    """Load Guardrails AI's persisted settings without importing a validator."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rc_type = importlib.import_module("guardrails.classes.rc").RC
            settings = importlib.import_module("guardrails.settings").settings
            rc = rc_type.load()
    except Exception:
        raise OptionalIntegrationSafetyError("Guardrails AI privacy settings could not be loaded") from None
    return settings, rc


def guardrails_ai_privacy_issue(rails_config: RailsConfig) -> str | None:
    """Return a content-safe activation issue for Guardrails AI telemetry."""

    if not uses_guardrails_ai(rails_config):
        return None
    try:
        _, rc = _guardrails_ai_runtime()
    except OptionalIntegrationSafetyError:
        return "Guardrails AI privacy settings could not be verified"
    if getattr(rc, "enable_metrics", None) is not False:
        return "Guardrails AI validators require ~/.guardrailsrc with enable_metrics=false"
    return None


def enforce_guardrails_ai_privacy(rails_config: RailsConfig) -> None:
    """Disable local tracing and verify Guardrails AI's persisted opt-out.

    Guardrails AI 0.11 reloads ``~/.guardrailsrc`` inside every ``Guard``
    constructor, after a validator has already been instantiated.  Changing
    only the in-memory singleton is therefore not enough.  Require the
    supported persisted opt-out, then mirror it into the current singleton and
    disable the separate local tracing path before Guardrails imports actions.
    """

    if not uses_guardrails_ai(rails_config):
        return
    settings, rc = _guardrails_ai_runtime()
    if getattr(rc, "enable_metrics", None) is not False:
        raise OptionalIntegrationSafetyError(
            "Guardrails AI validators require ~/.guardrailsrc with enable_metrics=false"
        )
    settings.rc = rc
    settings.disable_tracing = True
