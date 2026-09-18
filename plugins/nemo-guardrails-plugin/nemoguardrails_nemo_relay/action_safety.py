# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Activation checks for custom Guardrails actions.

NeMo Guardrails 0.24 executes synchronous LLMRails actions directly on the
calling event loop.  An asyncio timeout cannot interrupt that code, so one
blocking action can stop the Python worker from serving every Relay request.
This module keeps the policy and runtime introspection out of the worker entry
point and deliberately avoids changing Guardrails' dispatcher semantics.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class SynchronousActionPolicy(str, Enum):
    """How the worker handles custom synchronous LLMRails actions."""

    REJECT = "reject"
    ALLOW_UNSAFE = "allow_unsafe"


@dataclass(frozen=True, slots=True)
class ActionSafetySettings:
    """Bounded action-execution policy parsed from plugin configuration."""

    synchronous: SynchronousActionPolicy = SynchronousActionPolicy.REJECT


def action_safety_settings_from_config(value: object) -> ActionSafetySettings:
    """Parse the action-safety setting without executing Guardrails code."""

    if value is None:
        return ActionSafetySettings()
    if not isinstance(value, Mapping) or set(value) - {"synchronous"}:
        raise ValueError("action_safety must contain only synchronous")
    try:
        synchronous = SynchronousActionPolicy(value.get("synchronous", SynchronousActionPolicy.REJECT.value))
    except (TypeError, ValueError):
        raise ValueError("action_safety.synchronous contains an unsupported value") from None
    return ActionSafetySettings(synchronous=synchronous)


def _module_is_guardrails_owned(action: object) -> bool:
    """Return whether an action is shipped by the pinned Guardrails package."""

    module = inspect.getmodule(action)
    name = getattr(module, "__name__", "")
    return name == "nemoguardrails" or name.startswith("nemoguardrails.")


def _is_async_action(action: object) -> bool:
    """Mirror the execution shapes supported by Guardrails' dispatcher."""

    candidate = action
    if inspect.isclass(candidate):
        # Guardrails instantiates action classes before choosing ``ainvoke`` or
        # ``run``. A custom constructor or metaclass call is synchronous even
        # when ``ainvoke`` itself is async.
        if (
            candidate.__new__ is not object.__new__
            or candidate.__init__ is not object.__init__
            or type(candidate).__call__ is not type.__call__
        ):
            return False
        ainvoke = getattr(candidate, "ainvoke", None)
        return callable(ainvoke) and inspect.iscoroutinefunction(ainvoke)

    if inspect.isfunction(candidate) or inspect.ismethod(candidate):
        return inspect.iscoroutinefunction(candidate)

    ainvoke = getattr(candidate, "ainvoke", None)
    return callable(ainvoke) and inspect.iscoroutinefunction(ainvoke)


def unsafe_custom_synchronous_actions(
    rails: Any,
    *,
    remote_non_system_actions: bool = False,
) -> tuple[str, ...]:
    """Return custom action names that Guardrails would run synchronously.

    Manifest actions and actions implemented inside the pinned Guardrails
    distribution are governed by that distribution's compatibility contract.
    The worker owns the safety boundary for deployment-provided actions loaded
    by ``config.py`` or an ``actions`` module.  Lazy actions are resolved here
    during activation so an import failure cannot be deferred to live traffic.
    """

    runtime = getattr(rails, "runtime", None)
    dispatcher = getattr(runtime, "action_dispatcher", None)
    names = getattr(dispatcher, "get_registered_actions", None)
    get_action = getattr(dispatcher, "get_action", None)
    is_manifest_action = getattr(dispatcher, "is_manifest_action", None)
    if not callable(names) or not callable(get_action) or not callable(is_manifest_action):
        raise ValueError("Guardrails action dispatcher is unavailable")

    unsafe: list[str] = []
    try:
        action_names = names()
    except Exception:
        raise ValueError("Guardrails actions could not be inspected") from None
    if not isinstance(action_names, list) or any(not isinstance(name, str) for name in action_names):
        raise ValueError("Guardrails action registry is invalid")

    for name in sorted(action_names):
        try:
            if is_manifest_action(name):
                continue
            action = get_action(name)
        except Exception:
            raise ValueError("Guardrails actions could not be inspected") from None
        if action is None:
            raise ValueError("Guardrails action could not be resolved")
        if _module_is_guardrails_owned(action):
            continue
        action_meta = getattr(action, "action_meta", None)
        is_system_action = isinstance(action_meta, Mapping) and action_meta.get("is_system_action") is True
        if remote_non_system_actions and not is_system_action:
            # With actions_server_url configured, Guardrails sends ordinary
            # custom actions to that server. System actions remain local and
            # still need this worker-process safety check.
            continue
        if not _is_async_action(action):
            unsafe.append(name)
    return tuple(unsafe)


def enforce_action_safety(
    rails: Any,
    settings: ActionSafetySettings,
    *,
    actions_server_url: str | None,
) -> None:
    """Fail activation when custom synchronous actions can block the worker."""

    if settings.synchronous is SynchronousActionPolicy.ALLOW_UNSAFE:
        return
    if unsafe_custom_synchronous_actions(
        rails,
        remote_non_system_actions=actions_server_url is not None,
    ):
        # Do not include action names in the cross-process activation error.
        # A deployment may treat names as application implementation details.
        raise ValueError("custom synchronous Guardrails actions require an explicit unsafe acknowledgement")
