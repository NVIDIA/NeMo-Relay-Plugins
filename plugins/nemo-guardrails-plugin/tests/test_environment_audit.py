# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed-environment package audit tests."""

from nemoguardrails_nemo_relay.environment_audit import (
    normalized_package_name,
    unexpected_distributions,
)


def test_generated_wheel_adapter_is_allowed_only_at_its_exact_version() -> None:
    locked = {("nemoguardrails", "0.24.1")}
    generated = {("relay-wheel-environment", "1.0")}

    assert (
        unexpected_distributions(
            {
                "nemoguardrails": "0.24.1",
                "relay-wheel-environment": "1.0",
                "pip": "26.0",
            },
            locked,
            generated_distributions=generated,
        )
        == []
    )
    assert unexpected_distributions(
        {"relay-wheel-environment": "2.0"},
        locked,
        generated_distributions=generated,
    ) == [("relay-wheel-environment", "2.0")]


def test_unknown_distribution_is_not_hidden_by_adapter_exception() -> None:
    assert unexpected_distributions(
        {"relay-wheel-environment": "1.0", "undeclared-package": "3.2.1"},
        set(),
        generated_distributions={("relay-wheel-environment", "1.0")},
    ) == [("undeclared-package", "3.2.1")]


def test_distribution_names_use_lock_normalization() -> None:
    assert normalized_package_name("Relay_Wheel.Environment") == "relay-wheel-environment"
    assert (
        unexpected_distributions(
            {"NeMo_Relay.Plugin": "0.9.0"},
            {("nemo-relay-plugin", "0.9.0")},
        )
        == []
    )
