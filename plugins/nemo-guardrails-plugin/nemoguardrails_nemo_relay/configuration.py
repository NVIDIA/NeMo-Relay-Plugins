# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration validation and resolution for the Relay Guardrails worker."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# This module can be imported independently in tests and tooling. Keep the
# telemetry boundary intact even when ``worker`` is not the first import.
os.environ["NEMO_GUARDRAILS_NO_USAGE_STATS"] = "1"

import nemoguardrails  # noqa: E402
from nemo_relay_plugin import ConfigDiagnostic, DiagnosticLevel, Json  # noqa: E402
from nemoguardrails import RailsConfig  # noqa: E402

from .action_safety import action_safety_settings_from_config  # noqa: E402
from .check_backend import RemoteChecksSettings  # noqa: E402
from .evaluator_framework import (  # noqa: E402, F401
    EvaluatorFramework,
    evaluator_framework_from_config,
)
from .guardrails_configuration import (  # noqa: E402, F401
    _KNOWN_UPSTREAM_FAIL_OPEN_INPUT_RAILS,
    _KNOWN_UPSTREAM_FAIL_OPEN_OUTPUT_RAILS,
    _NO_CONTENT_LOG_LEVEL,
    _UPSTREAM_AMBIGUOUS_VERDICT_INPUT_RAILS,
    _UPSTREAM_AMBIGUOUS_VERDICT_OUTPUT_RAILS,
    PLUGIN_ID,
    _basic_rails_config_diagnostics,
    _catalog_context_diagnostics,
    _error,
    _load_supported_rails_config,
    _phase_configuration_diagnostics,
    _runtime_scope_diagnostics,
    _semantic_tool_diagnostics,
    _silence_guardrails_runtime_logs,
    _structural_tool_settings,
    _upstream_contract_diagnostics,
)
from .payload_policy import payload_policy_from_config  # noqa: E402
from .runtime_selection import RuntimePlan, select_runtime  # noqa: E402, F401
from .semantic_tools import semantic_tool_settings_from_config  # noqa: E402
from .text_mutation import mutation_policy_from_config  # noqa: E402

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


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Named result of resolving wrapper and Guardrails configuration."""

    settings: dict[str, Json] | None
    rails_config: RailsConfig | None
    runtime_plan: RuntimePlan | None
    diagnostics: tuple[ConfigDiagnostic, ...]


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
