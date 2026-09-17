# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from nemoguardrails.actions import action
from nemoguardrails.actions.math import wolfram_alpha_request

from nemoguardrails_nemo_relay.action_safety import (
    ActionSafetySettings,
    SynchronousActionPolicy,
    action_safety_settings_from_config,
    enforce_action_safety,
    unsafe_custom_synchronous_actions,
)


@action(name="sync_custom")
def _sync_custom() -> bool:
    return True


@action(name="async_custom")
async def _async_custom() -> bool:
    return True


@action(name="sync_system_custom", is_system_action=True)
def _sync_system_custom() -> bool:
    return True


class _AsyncRunnable:
    async def ainvoke(self, input: object) -> object:  # noqa: A002
        return input


class _SyncRunnable:
    def run(self, value: object) -> object:
        return value


class _AsyncRunnableWithSyncConstructor:
    def __init__(self) -> None:
        self.ready = True

    async def ainvoke(self, input: object) -> object:  # noqa: A002
        return input


class _Dispatcher:
    def __init__(self, actions: dict[str, object], manifests: set[str] | None = None) -> None:
        self._actions = actions
        self._manifests = manifests or set()

    def get_registered_actions(self) -> list[str]:
        return list(self._actions)

    def get_action(self, name: str) -> object | None:
        return self._actions.get(name)

    def is_manifest_action(self, name: str) -> bool:
        return name in self._manifests


def _rails(actions: dict[str, object], manifests: set[str] | None = None) -> object:
    return SimpleNamespace(runtime=SimpleNamespace(action_dispatcher=_Dispatcher(actions, manifests)))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, SynchronousActionPolicy.REJECT),
        ({}, SynchronousActionPolicy.REJECT),
        ({"synchronous": "reject"}, SynchronousActionPolicy.REJECT),
        ({"synchronous": "allow_unsafe"}, SynchronousActionPolicy.ALLOW_UNSAFE),
    ],
)
def test_action_safety_settings(value: object, expected: SynchronousActionPolicy) -> None:
    assert action_safety_settings_from_config(value).synchronous is expected


@pytest.mark.parametrize(
    "value",
    ["reject", {"unknown": True}, {"synchronous": "allow"}, {"synchronous": True}],
)
def test_invalid_action_safety_settings_fail_closed(value: object) -> None:
    with pytest.raises(ValueError):
        action_safety_settings_from_config(value)


def test_only_custom_synchronous_dispatch_shapes_are_unsafe() -> None:
    rails = _rails(
        {
            "sync_function": _sync_custom,
            "async_function": _async_custom,
            "sync_runnable": _SyncRunnable(),
            "async_runnable": _AsyncRunnable(),
            "async_class": _AsyncRunnable,
            "async_class_sync_constructor": _AsyncRunnableWithSyncConstructor,
            "guardrails_owned": wolfram_alpha_request,
            "manifest_owned": _sync_custom,
        },
        {"manifest_owned"},
    )

    assert unsafe_custom_synchronous_actions(rails) == (
        "async_class_sync_constructor",
        "sync_function",
        "sync_runnable",
    )


def test_reject_policy_blocks_custom_sync_actions_without_exposing_names() -> None:
    with pytest.raises(ValueError, match="explicit unsafe acknowledgement") as caught:
        enforce_action_safety(
            _rails({"private_customer_action": _sync_custom}),
            ActionSafetySettings(),
            actions_server_url=None,
        )

    assert "private_customer_action" not in str(caught.value)


def test_explicit_unsafe_policy_allows_legacy_sync_action() -> None:
    enforce_action_safety(
        _rails({"sync": _sync_custom}),
        ActionSafetySettings(SynchronousActionPolicy.ALLOW_UNSAFE),
        actions_server_url=None,
    )


def test_remote_action_server_owns_non_system_action_safety() -> None:
    enforce_action_safety(
        _rails({"sync": _sync_custom}),
        ActionSafetySettings(),
        actions_server_url="https://actions.example.test",
    )


def test_remote_action_server_does_not_hide_local_system_action_risk() -> None:
    with pytest.raises(ValueError, match="explicit unsafe acknowledgement"):
        enforce_action_safety(
            _rails({"sync_system": _sync_system_custom}),
            ActionSafetySettings(),
            actions_server_url="https://actions.example.test",
        )


def test_dispatcher_introspection_failure_is_not_treated_as_safe() -> None:
    with pytest.raises(ValueError, match="dispatcher is unavailable"):
        unsafe_custom_synchronous_actions(SimpleNamespace(runtime=SimpleNamespace()))
