# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom action fixture for local Guardrails configuration tests."""

from nemoguardrails.actions import action


@action(name="legacy_marker_check")
async def legacy_marker_check(text: str) -> bool:
    return "legacy-block" not in text


def init(app: object) -> None:
    app.register_action(legacy_marker_check)  # type: ignore[attr-defined]
