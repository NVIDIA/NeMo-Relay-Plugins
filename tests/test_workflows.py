# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep external workflow dependencies immutable."""

import re

import yaml

from scripts.catalog import ROOT


def action_references(value):
    if isinstance(value, dict):
        if "uses" in value:
            yield value["uses"]
        for nested in value.values():
            yield from action_references(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from action_references(nested)


def test_external_actions_use_full_commit_shas():
    for path in (ROOT / ".github/workflows").glob("*"):
        if path.suffix not in {".yml", ".yaml"}:
            continue
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for action in action_references(workflow):
            if action.startswith("./"):
                continue
            assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", action), (
                f"{path}: {action} must use a commit SHA"
            )


def test_repository_checks_publish_dependency_inventory_link():
    workflow = yaml.safe_load((ROOT / ".github/workflows/plugins.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["licenses"]["steps"]
    generator = next(step for step in steps if step.get("name") == "Create dependency inventory")
    assert "scripts.licensing.dependency_inventory" in generator["run"]
    assert '"$RUNNER_TEMP/dependencies.csv"' in generator["run"]

    upload = next(step for step in steps if step.get("id") == "dependency-inventory")
    assert upload["with"]["path"] == "${{ runner.temp }}/dependencies.csv"
    assert upload["with"]["if-no-files-found"] == "warn"

    link = next(step for step in steps if step.get("name") == "Link dependency inventory")
    assert "steps.dependency-inventory.outputs.artifact-url" in link["if"]
    assert "$GITHUB_STEP_SUMMARY" in link["run"]
