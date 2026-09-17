# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure helpers for auditing Relay's managed Python environment."""

from collections.abc import Mapping, Set

BOOTSTRAP_DISTRIBUTIONS = frozenset({"pip", "setuptools", "wheel"})


def normalized_package_name(name: str) -> str:
    """Return the PEP 503-style name used by uv lock records."""
    return name.lower().replace("_", "-").replace(".", "-")


def unexpected_distributions(
    installed: Mapping[str, str],
    locked_versions: Set[tuple[str, str]],
    *,
    generated_distributions: Set[tuple[str, str]] = frozenset(),
) -> list[tuple[str, str]]:
    """Find installed distributions not supplied by the lock or package adapter.

    Relay installs a small generated distribution to pull the checked wheelhouse
    into its managed environment. That adapter is intentionally absent from the
    plugin's uv.lock, so callers must identify its exact name and version.
    """
    permitted = locked_versions | {
        (normalized_package_name(name), version) for name, version in generated_distributions
    }
    return sorted(
        (name, version)
        for name, version in installed.items()
        if normalized_package_name(name) not in BOOTSTRAP_DISTRIBUTIONS
        and (normalized_package_name(name), version) not in permitted
    )
