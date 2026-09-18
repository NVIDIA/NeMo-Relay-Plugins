# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Guardrails worker lifecycle and Relay registration."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from dataclasses import dataclass
from typing import Any, cast

# Guardrails starts anonymous usage reporting when its runtime is constructed.
# This must precede every direct or transitive ``nemoguardrails`` import.
os.environ["NEMO_GUARDRAILS_NO_USAGE_STATS"] = "1"

from nemo_relay_plugin import (  # noqa: E402
    ConfigDiagnostic,
    DiagnosticLevel,
    Json,
    PluginContext,
    WorkerPlugin,
    serve_plugin,
)
from nemoguardrails import Guardrails, RailsConfig, set_default_framework  # noqa: E402
from nemoguardrails.llm.frameworks import get_default_framework, get_framework  # noqa: E402

from .action_safety import (  # noqa: E402
    action_safety_settings_from_config,
    enforce_action_safety,
)
from .check_backend import RemoteChecksBackend, RemoteChecksSettings  # noqa: E402
from .codec_projection import (  # noqa: E402
    _NativeCodecProjector,
)
from .configuration import (  # noqa: E402
    _NO_CONTENT_LOG_LEVEL,
    DEFAULT_CHECK_TIMEOUT_MS,
    PLUGIN_ID,
    _configured_secret_env,
    _resolve_config,
    _SecretEnvironment,
    _silence_guardrails_runtime_logs,
    _structural_tool_settings,
    _validate_config,
    _validate_wrapper_config,
)
from .evaluator_framework import evaluator_framework_from_config  # noqa: E402
from .execution_policy import (  # noqa: E402
    _LlmExecutionPolicy,
    _TextRailsPolicy,
)
from .integration_safety import (  # noqa: E402
    OptionalIntegrationSafetyError,
    enforce_guardrails_ai_privacy,
)
from .payload_policy import payload_policy_from_config  # noqa: E402
from .policy_marks import PolicyBackend, PolicyDecisionMarks  # noqa: E402
from .runtime_selection import RuntimePlan  # noqa: E402
from .semantic_tools import (  # noqa: E402
    SemanticToolPolicy,
    SemanticToolSettings,
    semantic_tool_settings_from_config,
)
from .structural_tools import Guardrails024StructuralToolAdapter, StructuralToolSettings  # noqa: E402
from .text_mutation import mutation_policy_from_config  # noqa: E402

# Lower numeric priorities run first and therefore wrap the rest of Relay's
# execution chain. Structural rails must see cache/short-circuit responses and
# response rewrites instead of disappearing behind another interceptor.
OUTERMOST_EXECUTION_PRIORITY = -2_147_483_648


@dataclass(frozen=True, slots=True)
class _RuntimeBindings:
    """Validated runtime pieces consumed by the Relay execution policy."""

    structural_tools: StructuralToolSettings
    input_enabled: bool
    output_enabled: bool
    structural_adapter: Guardrails024StructuralToolAdapter | None = None


class _NoStoreHistory(dict[str, Any]):
    """Dictionary-shaped sink for LLMRails' implicit conversation cache."""

    def __setitem__(self, key: str, value: Any) -> None:
        del key, value

    def update(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def setdefault(self, key: str, default: Any = None) -> Any:
        del key
        return default

    def __ior__(self, other: Any) -> "_NoStoreHistory":
        del other
        return self


class NeMoGuardrailsRelayWorker(WorkerPlugin):
    """Register one managed LLM execution policy for supported rails."""

    plugin_id = PLUGIN_ID
    allows_multiple_components = False

    def __init__(self) -> None:
        self._rails: Guardrails | None = None
        self._text_policy: _TextRailsPolicy | None = None
        self._semantic_tool_policy: SemanticToolPolicy | None = None
        self._execution_policy: _LlmExecutionPolicy | None = None
        self._llm_framework: Any | None = None
        self._secret_environment = _SecretEnvironment()

    def validate(self, config: Json) -> list[ConfigDiagnostic]:
        return _validate_config(config)

    @staticmethod
    def _raise_first_error(diagnostics: list[ConfigDiagnostic] | tuple[ConfigDiagnostic, ...]) -> None:
        errors = [item for item in diagnostics if item.level == DiagnosticLevel.ERROR]
        if errors:
            raise ValueError(errors[0].message)

    @staticmethod
    async def _ensure_legacy_plugin_is_inactive(ctx: PluginContext) -> None:
        try:
            registrations = await ctx.runtime.list_runtime_registrations()
        except Exception:
            raise ValueError("Relay runtime registrations could not be inspected") from None
        if any(registration.owner.plugin_kind == "nemo_guardrails" for registration in registrations):
            raise ValueError(
                "the dynamic Guardrails worker cannot run beside Relay's deprecated nemo_guardrails component"
            )

    async def _initialize_remote_runtime(
        self,
        settings: dict[str, Json],
        projector: _NativeCodecProjector,
        semantic_settings: SemanticToolSettings,
        timeout_ms: int,
    ) -> _RuntimeBindings:
        remote_settings = RemoteChecksSettings.from_config(settings["remote_checks"])
        self._text_policy = _TextRailsPolicy(
            None,
            timeout_ms,
            backend=RemoteChecksBackend(remote_settings, headers=remote_settings.headers()),
            projector=projector,
            allow_structural_tools=semantic_settings.checks_history_results,
            allow_tool_results=semantic_settings.checks_history_results,
        )
        return _RuntimeBindings(
            structural_tools=StructuralToolSettings(False, False, (), ()),
            input_enabled="input" in remote_settings.phases,
            output_enabled="output" in remote_settings.phases,
        )

    async def _initialize_local_runtime(
        self,
        settings: dict[str, Json],
        rails_config: RailsConfig,
        runtime_plan: RuntimePlan,
        projector: _NativeCodecProjector,
        semantic_settings: SemanticToolSettings,
        timeout_ms: int,
    ) -> _RuntimeBindings:
        tool_settings = _structural_tool_settings(rails_config)
        adapter: Guardrails024StructuralToolAdapter | None = None
        try:
            if tool_settings.enabled:
                adapter = Guardrails024StructuralToolAdapter(timeout_ms=timeout_ms)
                try:
                    await adapter.verify_compatibility()
                except Exception:
                    raise ValueError("Guardrails structural-tool adapter is incompatible") from None

            input_enabled = bool(rails_config.rails.input.flows)
            output_enabled = bool(rails_config.rails.output.flows)
            if input_enabled or output_enabled:
                _silence_guardrails_runtime_logs()
                try:
                    enforce_guardrails_ai_privacy(rails_config)
                except OptionalIntegrationSafetyError:
                    raise ValueError("Guardrails AI privacy settings are unsafe") from None

                evaluator_framework = evaluator_framework_from_config(settings.get("evaluator_framework"))
                if not runtime_plan.uses_iorails:
                    set_default_framework(evaluator_framework.value)
                    self._llm_framework = get_framework(evaluator_framework.value)
                previous_global_log_disable = logging.root.manager.disable
                logging.disable(_NO_CONTENT_LOG_LEVEL)
                try:
                    self._rails = Guardrails(rails_config, **runtime_plan.constructor_options())
                    if runtime_plan.uses_iorails:
                        await self._rails.startup()
                except Exception:
                    raise ValueError("Guardrails runtime could not be initialized") from None
                finally:
                    if not runtime_plan.uses_iorails:
                        self._llm_framework = get_framework(get_default_framework())
                    logging.disable(previous_global_log_disable)
                    _silence_guardrails_runtime_logs()

                if not runtime_plan.uses_iorails:
                    try:
                        enforce_action_safety(
                            self._rails,
                            action_safety_settings_from_config(settings.get("action_safety")),
                            actions_server_url=rails_config.actions_server_url,
                        )
                    except ValueError:
                        raise ValueError("Guardrails action safety validation failed") from None
                self._rails.events_history_cache = _NoStoreHistory()
                self._text_policy = _TextRailsPolicy(
                    self._rails,
                    timeout_ms,
                    projector=projector,
                    allow_structural_tools=tool_settings.enabled or semantic_settings.checks_history_results,
                    allow_tool_results=tool_settings.check_results or semantic_settings.checks_history_results,
                )
            return _RuntimeBindings(
                structural_tools=tool_settings,
                input_enabled=input_enabled,
                output_enabled=output_enabled,
                structural_adapter=adapter,
            )
        except BaseException:
            if adapter is not None:
                await adapter.close()
            raise

    def _register_interceptors(
        self,
        ctx: PluginContext,
        semantic_settings: SemanticToolSettings,
    ) -> None:
        if self._execution_policy is None:
            raise RuntimeError("Guardrails execution policy is unavailable")
        register_execution_with_context = getattr(ctx, "register_llm_execution_intercept_with_context", None)
        register_stream_with_context = getattr(
            ctx,
            "register_llm_stream_execution_intercept_with_context",
            None,
        )
        if callable(register_execution_with_context) and callable(register_stream_with_context):
            register_execution_with_context(
                "guardrails",
                self._execution_policy.execute_with_context,
                priority=OUTERMOST_EXECUTION_PRIORITY,
            )
            register_stream_with_context(
                "guardrails",
                self._execution_policy.execute_stream_with_context,
                priority=OUTERMOST_EXECUTION_PRIORITY,
            )
        else:
            ctx.register_llm_execution_intercept(
                "guardrails",
                self._execution_policy.execute,
                priority=OUTERMOST_EXECUTION_PRIORITY,
            )
            ctx.register_llm_stream_execution_intercept(
                "guardrails",
                self._execution_policy.execute_stream,
                priority=OUTERMOST_EXECUTION_PRIORITY,
            )
        if self._semantic_tool_policy is not None and semantic_settings.checks_execution:
            ctx.register_tool_execution_intercept(
                "guardrails-semantic-tools",
                self._semantic_tool_policy.execute,
                priority=OUTERMOST_EXECUTION_PRIORITY,
            )

    async def close(self) -> None:
        """Release process-scoped resources after registration or serving."""

        execution_policy = self._execution_policy
        self._execution_policy = None
        rails = self._rails
        self._rails = None
        llm_framework = self._llm_framework
        self._llm_framework = None
        text_policy = self._text_policy
        self._text_policy = None
        self._semantic_tool_policy = None
        failed = False
        cancelled: asyncio.CancelledError | None = None
        if execution_policy is not None:
            try:
                await execution_policy.close()
            except asyncio.CancelledError as error:
                cancelled = error
            except Exception:
                failed = True
        elif text_policy is not None:
            try:
                await text_policy.close()
            except asyncio.CancelledError as error:
                cancelled = error
            except Exception:
                failed = True
        if rails is not None:
            try:
                await rails.shutdown()
            except asyncio.CancelledError as error:
                cancelled = error
            except Exception:
                failed = True
        if llm_framework is not None:
            try:
                close = getattr(llm_framework, "aclose", None)
                if close is not None:
                    await close()
                else:
                    await llm_framework.reset()
            except asyncio.CancelledError as error:
                cancelled = error
            except Exception:
                failed = True
        try:
            self._secret_environment.restore()
        except Exception:
            failed = True
        if cancelled is not None:
            raise cancelled
        if failed:
            raise RuntimeError("Guardrails resources could not be closed")

    async def register(self, ctx: PluginContext, config: Json) -> None:
        wrapper_diagnostics = _validate_wrapper_config(config)
        self._raise_first_error(wrapper_diagnostics)
        try:
            self._secret_environment.install(_configured_secret_env(config))
        except Exception:
            raise ValueError("secret_env could not be applied") from None

        adapter: Guardrails024StructuralToolAdapter | None = None
        try:
            resolved = _resolve_config(config)
            self._raise_first_error(resolved.diagnostics)
            settings = resolved.settings
            if settings is None:
                raise ValueError("validated Guardrails configuration is unavailable")
            rails_config = resolved.rails_config
            runtime_plan = resolved.runtime_plan
            await self._ensure_legacy_plugin_is_inactive(ctx)

            timeout_ms = cast(int, settings.get("check_timeout_ms", DEFAULT_CHECK_TIMEOUT_MS))
            projector = _NativeCodecProjector(payload_policy_from_config(settings.get("payload_policy")))
            mutation_policy = mutation_policy_from_config(settings.get("mutation_policy"))
            semantic_settings = semantic_tool_settings_from_config(settings.get("semantic_tool_policy"))
            decision_marks = PolicyDecisionMarks(
                ctx.runtime,
                text_backend=(PolicyBackend.REMOTE if "remote_checks" in settings else PolicyBackend.LOCAL),
            )
            if "remote_checks" in settings:
                runtime = await self._initialize_remote_runtime(
                    settings,
                    projector,
                    semantic_settings,
                    timeout_ms,
                )
            else:
                if rails_config is None or runtime_plan is None:
                    raise ValueError("validated local Guardrails configuration is unavailable")
                runtime = await self._initialize_local_runtime(
                    settings,
                    rails_config,
                    runtime_plan,
                    projector,
                    semantic_settings,
                    timeout_ms,
                )
            adapter = runtime.structural_adapter

            if semantic_settings.enabled:
                if self._text_policy is None:
                    raise ValueError("Guardrails semantic tool checker is unavailable")
                self._semantic_tool_policy = SemanticToolPolicy(
                    semantic_settings,
                    self._text_policy,
                    timeout_ms=timeout_ms,
                    decision_marks=decision_marks,
                )

            self._execution_policy = _LlmExecutionPolicy(
                projector,
                self._text_policy,
                input_enabled=runtime.input_enabled,
                output_enabled=runtime.output_enabled,
                adapter=adapter,
                settings=runtime.structural_tools,
                semantic_tool_policy=self._semantic_tool_policy,
                mutation_policy=mutation_policy,
                decision_marks=decision_marks,
            )
            self._register_interceptors(ctx, semantic_settings)
        except BaseException:
            try:
                if adapter is not None and self._execution_policy is None:
                    await adapter.close()
                await self.close()
            except BaseException:
                try:
                    self._secret_environment.restore()
                except Exception:
                    pass
            raise


async def main() -> None:
    """Serve the worker until the Relay host requests shutdown."""

    plugin = NeMoGuardrailsRelayWorker()
    try:
        if sys.platform != "win32":
            await serve_plugin(plugin)
            return

        previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            await serve_plugin(plugin)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
    finally:
        await plugin.close()


if __name__ == "__main__":
    asyncio.run(main())
