# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from nemoguardrails import RailsConfig

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize(
    "name",
    [
        "input-output-rails",
        "migrated-local-rails",
        "no-model-rails",
        "structural-tool-rails",
    ],
)
def test_example_configuration_loads(name: str) -> None:
    RailsConfig.from_path(str(EXAMPLES / name))
