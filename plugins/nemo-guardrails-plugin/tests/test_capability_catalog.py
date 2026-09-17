# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the Guardrails 0.24 rail catalog to an explicit integration boundary."""

from __future__ import annotations

from collections import Counter

from nemoguardrails.guardrails.compiled_rail import unsupported_surface_reason
from nemoguardrails.manifests import default_rail_catalog

_NON_STANDALONE_TEXT_SURFACES = {
    ("input", "jailbreak detection heuristics"),
    ("output", "alignscore check facts"),
    ("output", "autoalign groundedness output"),
    ("output", "fiddler bot faithfulness"),
    ("output", "patronus api check output"),
    ("output", "patronus lynx check output hallucination"),
    ("output", "self check facts"),
    ("output", "self check hallucination"),
}


def test_pinned_catalog_has_no_unclassified_text_or_retrieval_surface() -> None:
    catalog = default_rail_catalog()
    surfaces = catalog.surfaces()
    directions = Counter(direction.value for direction, _ in surfaces)

    assert directions == {"input": 32, "output": 35, "retrieval": 11}

    unsupported = {
        (direction.value, name)
        for (direction, name), surface in surfaces.items()
        if unsupported_surface_reason(surface) is not None
    }
    retrieval = {(direction.value, name) for direction, name in surfaces if direction.value == "retrieval"}

    assert unsupported == _NON_STANDALONE_TEXT_SURFACES | retrieval


def test_every_standalone_text_surface_is_classified_as_verdict_or_transform() -> None:
    classified = Counter()
    for (direction, _), surface in default_rail_catalog().surfaces().items():
        if direction.value not in {"input", "output"} or unsupported_surface_reason(surface) is not None:
            continue
        classified[(direction.value, "transform" if surface.transform_target is not None else "verdict")] += 1

    # These counts describe the standalone text surfaces that the pinned
    # Guardrails architecture admits. They do not qualify each surface's
    # external service, model, optional dependency, or verdict parser. The
    # jailbreak heuristic is LLMRails-only and is tracked separately above.
    assert classified == {
        ("input", "verdict"): 22,
        ("input", "transform"): 9,
        ("output", "verdict"): 19,
        ("output", "transform"): 9,
    }
