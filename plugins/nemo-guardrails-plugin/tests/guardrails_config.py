# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path


def write_guardrails_config(
    root: Path,
    config_yml: str,
    *,
    name: str = "guardrails",
    rails_co: str | None = None,
    config_py: str | None = None,
) -> Path:
    """Create the small Guardrails config tree used by an integration test."""
    path = root / name
    path.mkdir()
    (path / "config.yml").write_text(config_yml, encoding="utf-8")
    if rails_co is not None:
        (path / "rails.co").write_text(rails_co, encoding="utf-8")
    if config_py is not None:
        (path / "config.py").write_text(config_py, encoding="utf-8")
    return path
