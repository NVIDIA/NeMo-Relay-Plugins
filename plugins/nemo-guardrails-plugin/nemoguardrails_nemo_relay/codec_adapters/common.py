# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fail-closed primitives for provider codec adapters."""

from __future__ import annotations

import json
import re
from collections.abc import Set
from dataclasses import dataclass
from typing import Any


class _UnsupportedRequest(ValueError):
    """Indicate that a provider payload cannot be inspected completely."""


@dataclass(frozen=True, slots=True)
class RawResponseText:
    """Provider text grouped by how the execution policy must inspect it."""

    candidates: tuple[str, ...] = ()
    fragments: tuple[str, ...] = ()
    reasoning: tuple[str, ...] = ()
    fragment_separator: str = "\n"


def _required_json_object_text(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise _UnsupportedRequest(f"{field} must be a JSON string")
    try:
        parsed = json.loads(
            value,
            parse_constant=_reject_non_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise _UnsupportedRequest(f"{field} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise _UnsupportedRequest(f"{field} must encode an object")


def _reject_non_json_constant(token: str) -> None:
    raise ValueError(token)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(key)
        value[key] = child
    return value


def _structural_key(value: str) -> bool:
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value).lower().replace("-", "_").split("_")
    return bool({"call", "calls", "function", "functions", "mcp", "tool", "tools", "toolset"}.intersection(words))


def _reject_unmodeled_structural_data(
    value: object,
    *,
    covered_nodes: set[int] | None = None,
    covered_keys: set[tuple[int, str]] | None = None,
) -> None:
    """Reject tool-shaped data outside provider fields checked explicitly by an adapter."""

    covered_nodes = covered_nodes or set()
    covered_keys = covered_keys or set()
    seen: set[int] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if id(current) in covered_nodes:
            continue
        if isinstance(current, (dict, list)):
            if id(current) in seen:
                continue
            seen.add(id(current))
        if isinstance(current, dict):
            for key, child in current.items():
                if not isinstance(key, str):
                    raise _UnsupportedRequest("provider payload contains an invalid object key")
                if (id(current), key) in covered_keys:
                    continue
                if _structural_key(key) or (
                    key in {"role", "type"} and isinstance(child, str) and _structural_key(child)
                ):
                    raise _UnsupportedRequest("provider payload contains unmodeled tool traffic")
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)


def _reject_structural_extras(value: dict[str, Any], allowed_keys: set[str]) -> None:
    """Inspect extension fields while leaving validated argument payloads opaque."""

    for key, child in value.items():
        if key in allowed_keys:
            continue
        if _structural_key(key):
            raise _UnsupportedRequest("provider tool call contains an unmodeled structural field")
        _reject_unmodeled_structural_data(child)


def _reject_nonempty_metadata(value: dict[str, Any], fields: set[str], context: str) -> None:
    """Reject metadata that can carry unchecked provider-visible text."""

    for field in fields.intersection(value):
        metadata = value[field]
        if metadata is None or metadata == [] or metadata == {}:
            continue
        raise _UnsupportedRequest(f"{context} {field} is not covered by output rails")


def _reject_unmodeled_fields(value: dict[str, Any], allowed_keys: Set[str], context: str) -> None:
    """Require an exact shape for identity-bearing protocol objects."""

    if set(value) - allowed_keys:
        raise _UnsupportedRequest(f"{context} contains unmodeled fields")
