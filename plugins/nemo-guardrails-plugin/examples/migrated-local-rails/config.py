# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained custom action used by the local migration fixture."""

from nemoguardrails.actions import action


@action(name="legacy_marker_check")
async def legacy_marker_check(text: str) -> bool:
    """Reject the deterministic migration-test marker."""

    return "legacy-block" not in text


def init(app: object) -> None:
    """Register the action through Guardrails' configuration hook."""

    app.register_action(legacy_marker_check)  # type: ignore[attr-defined]
