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
